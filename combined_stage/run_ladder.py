#!/usr/bin/env python3
"""Run a continuation ladder of combined-stage optimizations, each seeded from the previous.

    ./run_ladder.py ladders/hsx_variant_vacuum.yaml --vmec-input <seed namelist> --outdir output/hsx_vacuum

Stage k runs ``combined_stage_vmex.py`` with the ladder's common + stage overrides and the
previous stage's ``input.combined_stage_final``, ``coils_final.json`` and ``xpoint_final.json``
as seeds.  ``--set`` items apply to every stage (after the ladder's common list, before the stage's own), so one
ladder serves several coil counts.  A stage with ``beta_seed: <beta>`` starts from a finite-beta copy of the previous
stage's final boundary (tools/make_beta_seed.py, p ~ 1 - s, written into the stage directory).  Stages already finished (``wout_combined_stage_final.nc`` present) are skipped, so a
re-submitted ladder resumes where it stopped.

X-line gate (user, 2026-10-21): every double-null stage after the one that introduces the X-line must hand over what
the double null is for, a periodic X-type field line outside the plasma: |tr M| >= 2, field-line residual <= 1e-2 and
X-line-boundary distance > 0 (signed, inside < 0) in its last history row.  Otherwise the ladder stops there (reason
in <stage>/XLINE_GATE_FAILED.txt, also on a resume) instead of spending the beta stages on it.  In the v3 scans 30 of
229 devices left dn_converge failing it and 3 of them recovered later (hsx_scan/c24_i0.37_A8_w1: the free X-line
converged onto an elliptic closed field line 6-8.5 cm inside the plasma).  Thresholds: ladder key ``xline_gate:``
(dict overriding XLINE_GATE, or false); a stage's own ``xline_gate: false|true`` forces it off/on; ``--no-xline-gate``.
"""
import argparse
import os
import subprocess
import sys

import yaml

import run_layout as layout

HERE = os.path.dirname(os.path.abspath(__file__))
XLINE_GATE = dict(min_trace=2.0, max_residual=1.0e-2, min_distance=0.0)
GATE_FILE = "XLINE_GATE_FAILED.txt"


def last_history_row(outdir):
    try:
        rows = yaml.safe_load(open(layout.find(outdir, "history.yaml")))
    except (OSError, yaml.YAMLError):
        return None
    return rows[-1] if isinstance(rows, list) and rows else None


def double_null_on(outdir):
    try:
        cfg = yaml.safe_load(open(layout.find(outdir, "config_used.yaml")))
    except (OSError, yaml.YAMLError):
        return False
    return bool(((cfg or {}).get("double_null") or {}).get("enabled"))


def xline_gate(outdir, gate):
    """None if the stage's final X-line passes ``gate``, else the reason."""
    row = last_history_row(outdir)
    if row is None or row.get("xpoint_trace") is None:
        return "no X-line in the last history row"
    trace, res, dist = abs(float(row["xpoint_trace"])), row.get("xline_residual_max"), row.get("xpoint_distance_min")
    why = []
    if trace < gate["min_trace"]:
        why.append(f"|tr M| = {trace:.3f} < {gate['min_trace']} (elliptic: not an X-point)")
    if res is not None and float(res) > gate["max_residual"]:
        why.append(f"field-line residual {float(res):.2e} > {gate['max_residual']:.0e} (not a periodic field line)")
    if dist is not None and float(dist) <= gate["min_distance"]:
        why.append(f"X-line-boundary distance {float(dist):+.4f} m <= {gate['min_distance']} (inside the plasma)")
    return "; ".join(why) or None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ladder")
    p.add_argument("--vmec-input", required=True, help="seed namelist for the first stage")
    p.add_argument("--outdir", required=True)
    p.add_argument("--coils", default=None, help="seed coils json for the first stage (default: the config's coil_seed)")
    p.add_argument("--xpoint", default=None, help="seed X-point json for the first stage")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override for every stage (repeatable)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--no-xline-gate", action="store_true", help="do not stop the ladder on a failed X-line gate")
    args = p.parse_args()
    ladder = yaml.safe_load(open(args.ladder))
    gate_cfg = ladder.get("xline_gate", {})
    gate = None if gate_cfg is False or args.no_xline_gate else dict(XLINE_GATE, **(gate_cfg or {}))
    dn_seen = False             # the first double-null stage introduces the X-line and is not gated
    config = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(args.ladder)), ladder["config"]))
    seeds = dict(vmec_input=os.path.abspath(args.vmec_input),
                 coils=os.path.abspath(args.coils) if args.coils else None,
                 vessel=None,
                 xpoint=os.path.abspath(args.xpoint) if args.xpoint else None)
    for k, stage in enumerate(ladder["stages"]):
        outdir = os.path.abspath(os.path.join(args.outdir, f"{k:02d}_{stage['name']}"))
        done = os.path.exists(layout.find(outdir, "wout_combined_stage_final.nc"))
        if not done and stage.get("beta_seed") is not None:
            beta_input = layout.path(outdir, f"input.beta_seed", create=True)
            if not os.path.exists(beta_input):
                print(f"\n=== stage {k}: beta {stage['beta_seed']} seed from {seeds['vmec_input']}", flush=True)
                with open(layout.path(outdir, "beta_seed.log", create=True), "a") as log:
                    code = subprocess.call([sys.executable, "-u", os.path.join(HERE, "tools", "make_beta_seed.py"), config,
                                            seeds["vmec_input"], beta_input, str(stage["beta_seed"])]
                                           + ([seeds["wout"]] if seeds.get("wout") else []),
                                           stdout=log, stderr=subprocess.STDOUT)
                if code != 0 or "SEED READY" not in open(layout.path(outdir, "beta_seed.log")).read():
                    if os.path.exists(beta_input):
                        os.remove(beta_input)
                    raise SystemExit(f"stage {k}: beta seed failed; see {layout.path(outdir, 'beta_seed.log')}")
            seeds["vmec_input"] = beta_input
        if not done:
            cmd = [sys.executable, "-u", os.path.join(HERE, "combined_stage_vmex.py"), "--config", config,
                   "--vmec-input", seeds["vmec_input"], "--outdir", outdir]
            if seeds["coils"]:
                cmd += ["--coils", seeds["coils"]]
            if seeds["xpoint"]:
                cmd += ["--xpoint", seeds["xpoint"]]
            for item in list(ladder.get("common", [])) + list(args.set) + list(stage.get("set", [])):
                cmd += ["--set", item]
            if seeds.get("wout"):
                cmd += ["--set", f"vmex.restart_wout={seeds['wout']}"]   # used only if the seed's cold solve fails
            if seeds.get("vessel"):
                cmd += ["--set", f"vessel.seed_file={seeds['vessel']}"]   # ignored unless the stage enables the vessel
            if args.smoke:
                cmd.append("--smoke")
            if os.path.exists(layout.find(outdir, "checkpoint.npz")) and os.path.exists(layout.find(outdir, "al_state.yaml")):
                cmd.append("--resume")          # unfinished stage: continue from its last outer iteration
            os.makedirs(outdir, exist_ok=True)
            print(f"\n=== stage {k}: {stage['name']} ===\n{' '.join(cmd)}", flush=True)
            with open(layout.path(outdir, "run.log", create=True), "a") as log:
                code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            if code != 0 or not os.path.exists(layout.find(outdir, "coils_final.json")):
                raise SystemExit(f"stage {k} ({stage['name']}) failed (exit {code}); see {layout.path(outdir, 'run.log')}")
        else:
            print(f"=== stage {k}: {stage['name']} already done, skipping", flush=True)
        dn = double_null_on(outdir)
        if gate is not None and dn and stage.get("xline_gate", dn_seen):
            why = xline_gate(outdir, gate)
            if why:
                with open(os.path.join(outdir, GATE_FILE), "w") as fh:
                    fh.write(f"X-line gate failed after stage {k} ({stage['name']}): {why}\n"
                             f"gate {gate}; rerun with --no-xline-gate (or ladder/stage key xline_gate: false) to "
                             f"continue anyway\n")
                raise SystemExit(f"stage {k} ({stage['name']}): X-line gate failed: {why}; ladder stopped "
                                 f"(see {os.path.join(outdir, GATE_FILE)})")
            if os.path.exists(os.path.join(outdir, GATE_FILE)):
                os.remove(os.path.join(outdir, GATE_FILE))          # stale: an earlier attempt failed it
            print(f"=== stage {k}: X-line gate passed", flush=True)
        dn_seen = dn_seen or dn
        xpoint_final = layout.find(outdir, "xpoint_final.json")
        vessel_final = layout.find(outdir, "vessel_final.json")
        seeds = dict(vmec_input=layout.find(outdir, "input.combined_stage_final"),
                     coils=layout.find(outdir, "coils_final.json"),
                     xpoint=xpoint_final if os.path.exists(xpoint_final) else None,
                     vessel=vessel_final if os.path.exists(vessel_final) else None,
                     wout=layout.find(outdir, "wout_combined_stage_final.nc"))


if __name__ == "__main__":
    main()
