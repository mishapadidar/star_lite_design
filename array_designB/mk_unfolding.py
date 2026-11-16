#!/usr/bin/env python3
"""Statistics of the snowflake unfolding constant C under Gaussian-process coil perturbations,
and their toroidal profile.

Usage:
    mk_unfolding.py <design_double_final_json> [--nsamples 128] [--nphi 16]
                    [--length-scale 0.5] [--frame RZ] [--first-seed 0]

The snowflake of a double device (its second configuration) is a second-order null: under a
perturbation of the modular coils it splits into two X-points. For a perturbation
gamma -> gamma + sigma * d, with d a unit Gaussian-process sample (length scale L) applied to
each modular coil independently (the aux coils are NOT perturbed), the two X-points separate,
to leading order, by

    distance = C * sqrt(sigma)

where C = SnowflakeUnfolding.C() (utils/discriminant.py) depends on the realization d and on
the Poincare section where the separation is measured, but not on sigma. This script evaluates
C for --nsamples realizations (seeds first-seed .. first-seed + nsamples - 1 of
CoilPerturbationField: the same unit samples, per seed, as the ones of the repair study), in
the 2D frame --frame, at --nphi sections equally spaced over one field period, the first one
at phi = 0. Each realization is built once and used for all the sections.

A realization for which the unfolding equation has no real solution has no C (nan): the
snowflake does not split into a real pair for that sign of the perturbation.

Writes next to the design json:
    unfolding_C.txt          section phi = 0: columns seed, C [m^(1/2)], D_sigma (2 components)
    unfolding_C_phi.txt      one row per seed: seed, then C at each of the sections
    unfolding_C_profile.txt  one row per section: phi/2pi, E[C], std of C, median, number of
                             realizations with a real split
and appends to summary.txt (same 4-column format; unless already there) the statistics at
phi = 0 (unfolding_C_mean / _std / _median / _min / _max / _nreal) and the extremes of the
profiles (unfolding_C_profile_mean_min / _mean_max / _std_min / _std_max).
plot_unfolding.py makes the figures. Needs a simsopt with the third derivatives of the field
(venv_gradgradB).
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
from simsopt._core import load
from simsopt.geo import CurveLength

from star_lite_design.utils.discriminant import (CoilPerturbationField, DSigma, SnowflakeDiscriminant,
                                                 unfolding_solutions)
from star_lite_design.utils.singularperiodicfieldline import SingularPeriodicFieldline

parser = argparse.ArgumentParser(description="Unfolding constant C of the snowflake under GP coil perturbations.")
parser.add_argument("design_json")
parser.add_argument("--nsamples", type=int, default=128, help="number of GP realizations (default 128)")
parser.add_argument("--nphi", type=int, default=128,
                    help="number of sections over one field period, the first at phi = 0 (default 128)")
parser.add_argument("--length-scale", type=float, default=0.5, help="GP correlation length L (default 0.5)")
parser.add_argument("--frame", default="RZ", choices=("RZ", "NB"), help="2D frame of the separation (default RZ)")
parser.add_argument("--first-seed", type=int, default=0, help="seed of the first realization (default 0)")
args = parser.parse_args()

p = Path(args.design_json)
boozer_surfaces, iota_Gs, axes, xpoints, sdf = load(p)
snowflakes = [x for x in xpoints if isinstance(x, SingularPeriodicFieldline)]
if not snowflakes:
    raise SystemExit(f"mk_unfolding: no snowflake (SingularPeriodicFieldline) in {p.name}")
fl = snowflakes[-1]
bs = fl.biotsavart                      # the MODULAR field of the snowflake configuration
fl.run_code(CurveLength(fl.curve).J())  # a loaded field line has no solution state yet
nfp = int(fl.curve.nfp)
print(f"snowflake: nfp={nfp}, {len(bs.coils)} modular coils, aux mu = {np.asarray(fl.res['mu'])}")

# sections: toroidal fractions of the full torus, over one field period
phi0s = np.linspace(0.0, 1.0 / nfp, args.nphi, endpoint=False)


class _SlicedPerturbation:
    """Stands in for a CoilPerturbationField inside a DSigma: returns the part, for that
    DSigma's own grid, of a field response dB/dsigma evaluated once on all the grids."""
    def __init__(self):
        self.dB = None

    def dB_by_dsigma(self, points):
        return self.dB


# Per section: the full-torus return-map jet K (as SnowflakeUnfolding: nfp * the per-period jet)
# and a DSigma whose perturbation is swapped for each realization (its jitted functions depend
# on the section only, so they are compiled once per section instead of once per realization).
t0 = time.time()
Ks, dsigmas, slices = [], [], []
for phi0 in phi0s:
    disc = SnowflakeDiscriminant(fl, bs, frame=args.frame, phi0=float(phi0))
    Ks.append(nfp * np.asarray(disc.jet()))
    sl = _SlicedPerturbation()
    dsigmas.append(DSigma(fl, sl, frame=args.frame, biotsavart=bs, phi0=float(phi0)))
    slices.append(sl)
grids = [np.asarray(ds.curve_full.gamma()) for ds in dsigmas]
npts = [len(g) for g in grids]
all_pts = np.concatenate(grids, axis=0)
CHUNK = 16                                    # sections per evaluation of the field response
chunk_pts = npts[0] * min(CHUNK, args.nphi)   # (every section has the same number of points)
print(f"{args.nphi} sections over one field period, {sum(npts)} grid points in all "
      f"(discriminant at phi = 0: {SnowflakeDiscriminant(fl, bs, frame=args.frame, phi0=0.0).value():.6e}), "
      f"set up in {time.time() - t0:.1f} s")

seeds = np.arange(args.first_seed, args.first_seed + args.nsamples)
C = np.full((args.nsamples, args.nphi), np.nan)
Dsig0 = np.zeros((args.nsamples, 2))
t0 = time.time()
for i, seed in enumerate(seeds):
    # one realization: its field response on the grids of all the sections, in one evaluation
    pert = CoilPerturbationField(bs, length_scale=args.length_scale, seed=int(seed))
    # evaluated by blocks of CHUNK sections: bounded memory, and one compilation per block
    # shape (all blocks have the same shape when nphi is a multiple of CHUNK)
    dB = np.concatenate([np.asarray(pert.dB_by_dsigma(all_pts[a:a + chunk_pts]))
                         for a in range(0, len(all_pts), chunk_pts)], axis=0)
    start = 0
    for k, (ds, sl, n) in enumerate(zip(dsigmas, slices, npts)):
        sl.dB = dB[start:start + n]
        start += n
        ds.recompute_bell()
        Dsig = np.asarray(ds.d_sigma())
        sols = unfolding_solutions(Ks[k], Dsig)
        C[i, k] = float(sols[0]['C']) if sols else np.nan   # nearest split pair, as SnowflakeUnfolding.C()
        if k == 0:
            Dsig0[i] = Dsig
    if (i + 1) in (1, 2, 4, 8, 16, 32, 64) or (i + 1) == args.nsamples:
        print(f"  {i + 1:4d} realizations, {time.time() - t0:7.1f} s, last C(phi=0) = {C[i, 0]:.5f}", flush=True)

header = f"GP length scale {args.length_scale}, frame {args.frame}; separation = C * sqrt(sigma)"
out0 = p.parent / 'unfolding_C.txt'
np.savetxt(out0, np.column_stack([seeds, C[:, 0], Dsig0]), fmt=['%d', '%.10e', '%.10e', '%.10e'],
           header=header + ", section phi = 0\nseed C[m^0.5] Dsigma_0 Dsigma_1")
out_phi = p.parent / 'unfolding_C_phi.txt'
np.savetxt(out_phi, np.column_stack([seeds, C]), fmt=['%d'] + ['%.10e'] * args.nphi,
           header=header + "\nsections phi/2pi: " + " ".join(f"{x:.6f}" for x in phi0s) + "\nseed C[m^0.5] at each section")

nreal = np.isfinite(C).sum(axis=0)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)   # a section where no realization splits: all nan
    mean, std, med = np.nanmean(C, axis=0), np.nanstd(C, axis=0), np.nanmedian(C, axis=0)
out_prof = p.parent / 'unfolding_C_profile.txt'
np.savetxt(out_prof, np.column_stack([phi0s, mean, std, med, nreal]), fmt=['%.8f', '%.10e', '%.10e', '%.10e', '%d'],
           header=header + f", {args.nsamples} realizations\nphi/2pi E[C] std[C] median[C] n_real_splits")
print(f"wrote {out0}, {out_phi}, {out_prof}")
print("  phi/2pi    E[C]     std[C]   real splits")
for x, m, s, n in zip(phi0s, mean, std, nreal):
    print(f"  {x:.4f}   {m:.4f}   {s:.4f}   {n:d} of {args.nsamples}")

real = C[np.isfinite(C[:, 0]), 0]
if len(real):
    print(f"at phi = 0: C over {len(real)} of {args.nsamples} realizations with a real split: mean {real.mean():.5f}, "
          f"std {real.std():.5f}, median {np.median(real):.5f}, min {real.min():.5f}, max {real.max():.5f}")
    for sigma in (1e-3, 5e-3, 1e-2, 2e-2):
        print(f"   sigma = {1e3 * sigma:4.0f} mm: mean separation E[C] sqrt(sigma) = {1e2 * real.mean() * np.sqrt(sigma):.2f} cm")

summary = p.parent / 'summary.txt'
text = summary.read_text() if summary.is_file() else ""
stats = []
if 'unfolding_C_mean' not in text:
    stats += [('nreal', float(len(real)))]
    if len(real):
        stats += [('mean', real.mean()), ('std', real.std()), ('median', float(np.median(real))),
                  ('min', real.min()), ('max', real.max())]
if 'unfolding_C_profile_mean_max' not in text and np.isfinite(mean).any():
    stats += [('profile_mean_min', np.nanmin(mean)), ('profile_mean_max', np.nanmax(mean)),
              ('profile_std_min', np.nanmin(std)), ('profile_std_max', np.nanmax(std))]
if stats:
    with open(summary, 'a') as f:
        for name, val in stats:
            f.write(f"  {'unfolding_C_' + name:<30s} {val:.6e}   {'n/a':>16s}   {'n/a':>16s}\n")
    print(f"appended unfolding_C statistics to {summary}")
