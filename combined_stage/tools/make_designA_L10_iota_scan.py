#!/usr/bin/env python3
"""designA x 10 iota-floor scan (2026-11-07, user: "learn from the x10 size-ups which way to go, use 2 nodes").

What the x10 runs so far say (designA_L10_beta2, designA_L10_beta_scan job 4904, depth_analysis/designA_L10_beta1.5):
  * iota is the binding constraint. Vacuum refit ends at |iota| 0.225 (axis) / 0.179 (edge); every beta rung then
    sits exactly ON its 0.115 floor (iota axis 0.115-0.13), the axis moves (R_axis - R0)/a = -0.13 -> +0.48, about
    0.6 a, by beta 1-2 %, and the beta 2.5 % rung gets VMEX rejects (status 2): the device is at its equilibrium limit.
  * designA keeps its deep magnetic well (V' falls 23 % to the edge, DWell 30-50 vs DGeod -25..-30): Mercier is
    stable over s 0.2-0.8 at beta 2 %, unstable near the axis (DGeod) and at the very edge. The W7-X-size scan
    devices (vessel_scan3, iota 0.45-0.75) have a magnetic HILL and DMerc < 0 everywhere, so the Star_Lite
    lineage is the one worth pushing.
  * B.n stalls at rms 0.5-0.7 % (multiplier at penalty_max 1e6 on every rung), coil length sits at the 35 m cap,
    X-line flatness at its 0.30 m limit, curvature is idle (1.2-1.5 vs 10.6 /m).
Question this scan answers: what does holding iota up through beta cost (QS, B.n, coils, flatness), and does it buy a
smaller Shafranov shift / a Mercier-stable core at beta 2 %?

Grid (64 devices, full factorial): iota floor {0.12 ... 0.33} x X-line flatness limit {0.30 m (user's x10 limit),
0.15 m} x coil length cap {35 m (user's limit), 42 m (exploratory arm: is coil length what limits iota / B.n?)} x
Mercier weight {1e-4 (as before), 1e-3}. Everything else = ladders/designA_L10_beta_scan.yaml (ns 51 everywhere,
same solver / engineering limits / vessel). Every device starts from designA_L10_beta_scan/00_refit_vac (ns 51
vacuum polish): vacuum iota-raise rung (the floor applies from here on; cheap, no virtual casing) -> beta 0.5 / 1 / 2 %.
8 cores per run (job 4904 used 4.7 of its 32 cores on average, RSS 12 GB).

usage: make_designA_L10_iota_scan.py [--submit] [--only NAME ...] [--nodes cn02,cn03]
"""
import argparse
import itertools
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = "/data/common/raffael.wendlinger/simflare/runs/combined_stage"
BASE = os.path.join(HERE, "ladders", "designA_L10_beta_scan.yaml")
SRC = os.path.join(RUNS, "designA_L10_beta_scan", "00_refit_vac")
OUT = os.path.join(RUNS, "designA_L10_iota_scan")
IOTAS = (0.12, 0.15, 0.18, 0.21, 0.24, 0.27, 0.30, 0.33)
FLATNESS = (0.30, 0.15)
LENGTHS = (35.0, 42.0)
MERCIER = (1.0e-4, 1.0e-3)
OUTERS = dict(vac=5, b05=6, b1=6, b2=8)


def replace(items, key, value):
    """Set `key=value` in a ladder --set list (replacing an existing entry for that key, else appending)."""
    out, hit = [], False
    for x in items:
        if x.split("=", 1)[0] == key:
            out.append(f"{key}={value}")
            hit = True
        else:
            out.append(x)
    return out if hit else out + [f"{key}={value}"]


def ladder(iota, flat, length, merc):
    base = yaml.safe_load(open(BASE))
    common = replace(base["common"], "double_null.flatness_threshold", flat)
    common = replace(common, "coils.length_threshold", length)
    mercier = f"plasma_objective.mercier={{enabled: true, margin: 0.0, smoothing: 1.0e-3, weight: {merc:g}, s_min: 0.1}}"

    def beta(name, b, outer):
        return {"name": name, "beta_seed": b, "set": [
            "vmex.plasma_field=virtual_casing", "consistency.field_strength=net_poloidal_current",
            f"plasma_objective.iota_floor={iota}", mercier, f"augmented_lagrangian.outer_maxiter={outer}"]}

    stages = [{"name": f"vac_iota{iota:.2f}", "set": [
        "vmex.plasma_field=vacuum", "consistency.field_strength=toroidal_flux",
        f"plasma_objective.iota_floor={iota}", f"augmented_lagrangian.outer_maxiter={OUTERS['vac']}"]},
        beta("beta0.5_G", 0.005, OUTERS["b05"]), beta("beta1_G", 0.010, OUTERS["b1"]), beta("beta2_G", 0.020, OUTERS["b2"])]
    return {"config": base["config"], "common": common, "stages": stages}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--nodes", default="cn02,cn03", help="jobs alternate over these nodes (one node each)")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--mem", default="24G")
    p.add_argument("--time", default="14-00:00:00")
    p.add_argument("--stagger-min", type=float, default=1.0)
    p.add_argument("--startup-jitter", type=int, default=1800)
    args = p.parse_args()
    nodes = args.nodes.split(",")
    ldir = os.path.join(HERE, "ladders", "designA_L10_iota_scan")
    os.makedirs(ldir, exist_ok=True)
    k = 0
    for iota, flat, length, merc in itertools.product(IOTAS, FLATNESS, LENGTHS, MERCIER):
        name = f"i{iota:.2f}_f{int(round(flat * 100))}_L{int(length)}_m{int(round(-__import__('math').log10(merc)))}"
        if args.only and name not in args.only:
            continue
        path = os.path.join(ldir, f"{name}.yaml")
        with open(path, "w") as fh:
            fh.write(f"# designA x10 iota scan device {name}: written by tools/make_designA_L10_iota_scan.py "
                     "(edit the generator, not this file)\n")
            yaml.safe_dump(ladder(iota, flat, length, merc), fh, sort_keys=False, width=200)
        if not args.submit:
            print(name, path)
            continue
        out = os.path.join(OUT, name)
        if os.path.exists(os.path.join(out, "logs")):
            print(f"EXISTS {out}, not submitted (resubmit by hand to resume)")
            continue
        os.makedirs(os.path.join(out, "logs"), exist_ok=True)
        env = dict(os.environ, LADDER=os.path.relpath(path, HERE), SEED=f"{SRC}/inputs/input.combined_stage_final",
                   COILS=f"{SRC}/coils/coils_final.json", XPOINT=f"{SRC}/xpoints/xpoint_final.json", OUTDIR=out,
                   SETS=f"vessel.seed_file={SRC}/vessel/vessel_final.json", STARTUP_JITTER=str(args.startup_jitter))
        begin = f"now+{int(round(k * args.stagger_min * 60))}"
        jid = subprocess.check_output(
            ["sbatch", "--parsable", f"--job-name=aL10_{name}", f"--nodelist={nodes[k % len(nodes)]}",
             f"--cpus-per-task={args.cpus}", f"--mem={args.mem}", f"--time={args.time}", f"--begin={begin}",
             f"--output={out}/logs/ladder_%j.out", "sbatch/ladder.sbatch"], cwd=HERE, env=env, text=True).strip()
        print(f"{name} {jid} {nodes[k % len(nodes)]} begin {begin}", flush=True)
        k += 1


if __name__ == "__main__":
    main()
