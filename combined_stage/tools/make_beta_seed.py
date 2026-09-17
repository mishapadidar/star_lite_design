#!/usr/bin/env python3
"""Finite-beta copy of a VMEC namelist: p(s) = PRES_SCALE (1 - s) (AM = 1, -1), PRES_SCALE set so that VMEX's betatotal
equals the target at a combined-stage config's resolution (vmex.mpol/ntor/ns/ftol/niter). Net toroidal current stays
as in the input (NCURR = 1, CURTOR = 0 for the HSX-size seeds: no bootstrap current).

usage: make_beta_seed.py <config yaml> <namelist in> <namelist out> <beta target, e.g. 0.01>
Prints each calibration solve and "SEED READY <out>" once |beta/target - 1| < 2e-3 (a cold VMEX solve of <out>).
"""
import re
import sys
import time
from dataclasses import replace

import numpy as np
import yaml


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    src, dst, beta_target = sys.argv[2], sys.argv[3], float(sys.argv[4])
    import vmex as vj
    from vmex import optimize as opt
    v = cfg["vmex"]
    text0 = open(src).read()
    for key in ("PRES_SCALE", "AM", "PMASS_TYPE"):
        if not re.search(rf"(?m)^\s*{key}\s*=", text0):
            raise SystemExit(f"{src} has no {key} entry to replace")

    def with_pressure(scale):
        text = re.sub(r"(?m)^(\s*PRES_SCALE\s*=\s*).*$", lambda m: f"{m.group(1)}{scale:.17E}", text0)
        return re.sub(r"(?m)^(\s*AM\s*=\s*).*$",
                      lambda m: m.group(1) + "1.0E+00, -1.0E+00, 0.0E+00, 0.0E+00, 0.0E+00, 0.0E+00, ", text)

    def solve(scale):
        open(dst, "w").write(with_pressure(scale))
        inp = vj.VmecInput.from_file(dst)
        if v.get("mpol") or v.get("ntor"):
            mp, nt = int(v.get("mpol") or inp.mpol), int(v.get("ntor") or inp.ntor)
            inp = inp.change_resolution(mpol=mp, ntor=nt, ntheta=2 * mp + 6, nzeta=2 * nt + 4)
        inp = replace(inp, ns_array=np.array([v["ns"]]), ftol_array=np.array([v["ftol"]]),
                      niter_array=np.array([v["niter"]]))
        t0 = time.time()
        w = opt.solve_equilibrium(inp).wout
        return w, time.time() - t0

    scale = 2.0 * beta_target / (2.0 * 4e-7 * np.pi)      # <p> = scale / 2 for p = scale (1 - s), B0 ~ 1 T
    for it in range(5):
        try:
            w, dt = solve(scale)
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"SEED FAILED: cold solve at PRES_SCALE {scale:.1f} raised {type(e).__name__}: {e}")
        beta = float(w.betatotal)
        print(f"calibration {it}: PRES_SCALE {scale:9.1f} Pa -> betatotal {beta:.6f}, aspect {float(w.aspect):.4f}, "
              f"iota axis/edge {float(w.iotaf[0]):+.4f}/{float(w.iotaf[-1]):+.4f} ({dt:.1f}s)", flush=True)
        if abs(beta / beta_target - 1.0) < 2e-3:
            print(f"SEED READY {dst} PRES_SCALE {scale:.1f} betatotal {beta:.6f}")
            return
        scale *= beta_target / beta
    raise SystemExit(f"SEED NOT READY: betatotal {beta:.6f} after 5 calibration solves")


if __name__ == "__main__":
    main()
