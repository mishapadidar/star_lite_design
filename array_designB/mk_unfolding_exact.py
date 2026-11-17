#!/usr/bin/env python3
"""Exact mean of the snowflake unfolding constant C under Gaussian-process coil perturbations,
in the section at phi = 0, in every toroidal section, and averaged over the sections.

Usage:
    mk_unfolding_exact.py <design_double_final_json> [--length-scale 0.5] [--force]

Same quantity as mk_unfolding.py (X-point separation = C * sqrt(sigma) for a perturbation of
amplitude sigma of the modular coils; the aux coils are not perturbed), but without drawing
realizations: the response D_sigma of the return map is linear in the Gaussian displacement, so
it is a Gaussian 2-vector, and C is homogeneous of degree 1/2 in it, which makes E[C] a
one-dimensional integral (ExactMeanUnfoldingC). The values in the other sections follow by
transporting the split directions along the field (PhiMeanUnfoldingC). Both classes are those
of boozer_risk_opt.py, read from its source (they are defined in that script, which cannot be
imported), so this script must sit next to it.

Writes next to the design json:
    unfolding_EC_exact_profile.txt   one row per node of the full-torus grid: t (fraction of
                                     the full turn of the snowflake field line), E[C](t)
and appends to summary.txt (same 4-column format; unless already there, or --force):
    unfolding_EC_exact            E[C] in the section at phi = 0
    unfolding_EC_exact_std        standard deviation of C there
    unfolding_EC_exact_phi_mean   mean of E[C] over the sections
    unfolding_EC_exact_phi_min    smallest E[C] over the sections
    unfolding_EC_exact_phi_max    largest E[C] over the sections
Needs a simsopt with the third derivatives of the field (venv_gradgradB).
"""
import argparse
import time
from pathlib import Path

import numpy as np
from simsopt._core import load, Optimizable                       # noqa: F401 (used by the classes)
from simsopt._core.derivative import derivative_dec               # noqa: F401
from simsopt.geo import CurveLength

from star_lite_design.utils.discriminant import (CoilPerturbationField, DSigma,          # noqa: F401
                                                 SnowflakeDiscriminant, unfolding_solutions)
from star_lite_design.utils.singularperiodicfieldline import SingularPeriodicFieldline

parser = argparse.ArgumentParser(description="Exact E[C] of the snowflake, at phi = 0 and over the sections.")
parser.add_argument("design_json")
parser.add_argument("--length-scale", type=float, default=0.5, help="GP correlation length L (default 0.5)")
parser.add_argument("--force", action="store_true", help="recompute and rewrite the summary lines")
args = parser.parse_args()

p = Path(args.design_json)
summary = p.parent / 'summary.txt'
text = summary.read_text() if summary.is_file() else ""
if 'unfolding_EC_exact_phi_mean' in text and not args.force:
    raise SystemExit(f"mk_unfolding_exact: already in {summary} (use --force to recompute)")

# the classes of boozer_risk_opt.py (_WeightedPerturbation ... PhiMeanUnfoldingC)
src = (Path(__file__).resolve().parent / 'boozer_risk_opt.py').read_text()
exec(compile(src[src.index('class _WeightedPerturbation'):src.index('def _rel_vio')],
             'boozer_risk_opt.py (E[C] classes)', 'exec'), globals())

t0 = time.time()
boozer_surfaces, iota_Gs, axes, xpoints, sdf = load(p)
snowflakes = [x for x in xpoints if isinstance(x, SingularPeriodicFieldline)]
if not snowflakes:
    raise SystemExit(f"mk_unfolding_exact: no snowflake (SingularPeriodicFieldline) in {p.name}")
fl = snowflakes[-1]
bs = fl.biotsavart                      # the MODULAR field of the snowflake configuration
fl.run_code(CurveLength(fl.curve).J())  # a loaded field line has no solution state yet

ec = ExactMeanUnfoldingC(fl, bs, length_scale=args.length_scale, frame='RZ', phi0=0.0)   # noqa: F821
ec0, std0 = ec.J(), ec.std()
phi = PhiMeanUnfoldingC(ec)                                                              # noqa: F821
t, prof = phi.profile()
mean = phi.J()
print(f"exact E[C] at phi = 0: {ec0:.6f} (std of C {std0:.5f}); over the sections: mean {mean:.6f}, "
      f"min {prof.min():.6f}, max {prof.max():.6f} ({len(t)} nodes, {time.time() - t0:.0f} s)")

order = np.argsort(t)
np.savetxt(p.parent / 'unfolding_EC_exact_profile.txt', np.column_stack([t[order], prof[order]]),
           header="exact E[C] of the snowflake in the toroidal sections (mk_unfolding_exact.py)\n"
                  "t (fraction of the full turn of the snowflake field line)   E[C](t) [m^(1/2)]")

stats = [('unfolding_EC_exact', ec0), ('unfolding_EC_exact_std', std0), ('unfolding_EC_exact_phi_mean', mean),
         ('unfolding_EC_exact_phi_min', float(prof.min())), ('unfolding_EC_exact_phi_max', float(prof.max()))]
# drop the lines of a previous run (--force), then append
kept = [line for line in text.splitlines() if not line.split()[:1] or not line.split()[0].startswith('unfolding_EC_exact')]
lines = kept + [f"  {name:<30s} {val:.6e}   {'n/a':>16s}   {'n/a':>16s}" for name, val in stats]
tmp = summary.with_suffix('.txt.tmp_exact')
tmp.write_text("\n".join(lines) + "\n")
tmp.replace(summary)
print(f"wrote the unfolding_EC_exact lines to {summary}")
