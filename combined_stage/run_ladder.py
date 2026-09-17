#!/usr/bin/env python3
"""Run a continuation ladder of combined-stage optimizations, each seeded from the previous.

    ./run_ladder.py ladders/hsx_variant_vacuum.yaml --vmec-input <seed namelist> --outdir output/hsx_vacuum

Stage k runs ``combined_stage_vmex.py`` with the ladder's common + stage overrides and the
previous stage's ``input.combined_stage_final``, ``coils_final.json`` and ``xpoint_final.json``
as seeds.  Stages already finished (``wout_combined_stage_final.nc`` present) are skipped, so a
re-submitted ladder resumes where it stopped.
"""
import argparse
import os
import subprocess
import sys

import yaml

import run_layout as layout

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ladder")
    p.add_argument("--vmec-input", required=True, help="seed namelist for the first stage")
    p.add_argument("--outdir", required=True)
    p.add_argument("--coils", default=None, help="seed coils json for the first stage (default: the config's coil_seed)")
    p.add_argument("--xpoint", default=None, help="seed X-point json for the first stage")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    ladder = yaml.safe_load(open(args.ladder))
    config = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(args.ladder)), ladder["config"]))
    seeds = dict(vmec_input=os.path.abspath(args.vmec_input),
                 coils=os.path.abspath(args.coils) if args.coils else None,
                 xpoint=os.path.abspath(args.xpoint) if args.xpoint else None)
    for k, stage in enumerate(ladder["stages"]):
        outdir = os.path.abspath(os.path.join(args.outdir, f"{k:02d}_{stage['name']}"))
        done = os.path.exists(layout.find(outdir, "wout_combined_stage_final.nc"))
        if not done:
            cmd = [sys.executable, "-u", os.path.join(HERE, "combined_stage_vmex.py"), "--config", config,
                   "--vmec-input", seeds["vmec_input"], "--outdir", outdir]
            if seeds["coils"]:
                cmd += ["--coils", seeds["coils"]]
            if seeds["xpoint"]:
                cmd += ["--xpoint", seeds["xpoint"]]
            for item in list(ladder.get("common", [])) + list(stage.get("set", [])):
                cmd += ["--set", item]
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
        xpoint_final = layout.find(outdir, "xpoint_final.json")
        seeds = dict(vmec_input=layout.find(outdir, "input.combined_stage_final"),
                     coils=layout.find(outdir, "coils_final.json"),
                     xpoint=xpoint_final if os.path.exists(xpoint_final) else None)


if __name__ == "__main__":
    main()
