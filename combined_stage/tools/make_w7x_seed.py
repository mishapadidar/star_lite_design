#!/usr/bin/env python3
"""W7-X-class QI seed namelist for the combined stage: the W7-X standard configuration boundary scaled to a chosen R0.

The boundary is simsopt's tests/test_files/input.W7-X_standard_configuration (R0 5.52 m, a 0.50 m, nfp 5), every
RBC/ZBS scaled by R0 / 5.52 and truncated to m <= 5, |n| <= 6 (the combined stage runs at mpol = ntor = 6). B0 = 1 T:
PHIEDGE = pi a^2 B0 with a = |RBC(0,1)| of the scaled boundary (positive, our convention). Vacuum, NCURR = 1, CURTOR =
0; the run settings (DELT, NS_ARRAY, PRES_SCALE/AM entries that tools/make_beta_seed.py rewrites) are those of the HSX
seeds (seeds/input.qa_nfp4_R1p2_A6).

usage: make_w7x_seed.py <R0 in m> <out namelist>
"""
import math
import re
import sys

W7X = "/data/common/raffael.wendlinger/simflare/ext/simsopt/tests/test_files/input.W7-X_standard_configuration"
TEMPLATE = "/data/common/raffael.wendlinger/simflare/runs/combined_stage/seeds/input.qa_nfp4_R1p2_A6"


def main():
    R0, out = float(sys.argv[1]), sys.argv[2]
    coef = {}
    for kind, n, m, v in re.findall(r"(RBC|ZBS)\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*=\s*([-+0-9.eEdD]+)", open(W7X).read()):
        coef[(kind, int(n), int(m))] = float(v.replace("D", "E").replace("d", "e"))
    s = R0 / coef[("RBC", 0, 0)]
    a = abs(coef[("RBC", 0, 1)]) * s
    phiedge = math.pi * a * a
    head = [l for l in open(TEMPLATE).read().splitlines()
            if l.startswith("  ") and not re.match(r"\s+(RBC|ZBS)\(", l)]
    head = [re.sub(r"NFP = .*", "NFP = 5", re.sub(r"PHIEDGE = .*", f"PHIEDGE = {phiedge:.16E}", l)) for l in head]
    body = [f"  {kind}({n},{m}) = {v * s:.16E}"
            for (kind, n, m), v in sorted(coef.items(), key=lambda t: (t[0][2], t[0][1], t[0][0]))
            if m <= 5 and abs(n) <= 6 and v != 0.0]
    hdr = ["&INDATA",
           "! W7-X-class QI seed (nfp 5) for the combined stage, written by tools/make_w7x_seed.py:",
           f"!   W7-X standard configuration boundary scaled by {s:.6f} to R0 = {R0} m, a ~ {a:.4f} m, truncated to",
           "!   m <= 5, |n| <= 6; B0 = 1 T -> PHIEDGE = pi a^2 B0; vacuum."]
    open(out, "w").write("\n".join(hdr + head + body) + "\n/\n")
    print(f"{out}: scale {s:.6f}, a {a:.4f} m, PHIEDGE {phiedge:.6f} Wb, {len(body)} boundary coefficients")


if __name__ == "__main__":
    main()
