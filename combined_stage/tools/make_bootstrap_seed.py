#!/usr/bin/env python3
"""Seed namelist with a self-consistent bootstrap current for a combined-stage run with `bootstrap.enabled`.

usage: python make_bootstrap_seed.py <config.yaml> <seed namelist> <out namelist> [--relax 0.5] [--iter 15]

Takes the kinetic profiles from the config's `bootstrap.profiles`, replaces the seed's pressure by e ne (Te + Ti)
(as the driver does with `pressure_from_profiles`), then runs VMEX's fixed-boundary Picard iteration
(vmex.core.bootstrap.self_consistent_bootstrap: AC/CURTOR <- the current profile implied by the Redl <J.B>) and writes
the result. Starting the optimization from it matters twice: the AL bootstrap block starts near feasibility, and the AC
profile is nonzero -- with AC = 0 VMEC's shape normalization is degenerate and the AC dofs have no gradient.
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config")
    p.add_argument("seed")
    p.add_argument("out")
    p.add_argument("--relax", type=float, default=0.5)
    p.add_argument("--iter", type=int, default=15)
    p.add_argument("--degree", type=int, default=8, help="AC power-series degree of the refit")
    args = p.parse_args()

    import vmex as vj
    from vmex.core import bootstrap as bt
    from star_lite_design.utils.vmex_combined_stage import kinetic_pressure_coeffs, kinetic_profiles

    cfg = yaml.safe_load(open(args.config))
    bcfg, prof = cfg["bootstrap"], cfg["bootstrap"]["profiles"]
    helicity_n = bcfg.get("helicity_n")
    helicity_n = cfg["plasma_objective"]["helicity_n"] if helicity_n is None else helicity_n
    inp = vj.VmecInput.from_file(args.seed)
    coeffs = kinetic_pressure_coeffs(prof["ne"], prof["Te"], prof["Ti"])
    am = np.zeros(max(21, len(np.atleast_1d(inp.am))))
    am[:coeffs.size] = coeffs
    inp = replace(inp, am=am, pres_scale=1.0, pmass_type="power_series")
    profiles = kinetic_profiles(prof["ne"], prof["Te"], prof["Ti"], prof.get("Zeff", [1.0]))
    res = bt.self_consistent_bootstrap(inp, profiles, int(helicity_n), n_iter=args.iter, relax=args.relax,
                                       degree=args.degree, verbose=True)
    w = res.equilibrium.wout
    print(f"converged {res.converged} after {res.iterations} iterations; CURTOR {float(res.input.curtor):+.6e} A; "
          f"beta {float(w.betatotal):.4e}; iota axis/edge {float(w.iotaf[0]):+.4f}/{float(w.iotaf[-1]):+.4f}")
    res.input.to_indata(args.out)
    print(f"wrote {args.out}")
    if not res.converged:
        sys.exit("WARNING: Picard iteration did not converge -- inspect before seeding a run")


if __name__ == "__main__":
    main()
