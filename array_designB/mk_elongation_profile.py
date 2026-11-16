#!/usr/bin/env python3
"""Toroidal profile of the near-axis elongation of a device.

Usage:
    mk_elongation_profile.py <design_json> [--nphi 128]

<design_json> is any final design json (design_double_final_<TAG>.json with its two
configurations, or a single-configuration design_opt_final_ / design_polished_final_ /
config1|config2 json). For each configuration, the elongation at a toroidal angle phi is the
aspect ratio of the invariant ellipse of the one-field-period return map anchored at phi, in
the plane normal to the magnetic axis (utils/tangent_map.py: TangentMap(phi=..., frame='NB')
.elongation): the quantity bounded by boozer_all.py --elongation and reported (at 32 angles)
by mk_elongation.py. It is sampled at NPHI angles equally spaced over one field period.

Writes next to the design json:
    elongation_profile.txt   columns: phi/2pi, then the elongation of each configuration
    elongation_profile.png   the profiles, with the elongation bound of the yaml if any
and appends elongation_profile_max_config<k> / _min_ / _mean_ lines to summary.txt (same
4-column format as the rest of summary.txt), unless they are already there.
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import CurveLength

import jax.numpy as jnp
from jax import jit

from star_lite_design.utils.tangent_map import TangentMap, elongation_pure

# compiled once for every angle: the Chebyshev matrix D is an argument (see the loop below)
elongation_value = jit(elongation_pure, static_argnames=('frame',))

parser = argparse.ArgumentParser(description="Toroidal profile of the near-axis elongation.")
parser.add_argument("design_json")
parser.add_argument("--nphi", type=int, default=128,
                    help="number of toroidal angles over one field period (default 128)")
parser.add_argument("--replot", action="store_true",
                    help="only redo the figure, from the elongation_profile.txt already there")
args = parser.parse_args()

p = Path(args.design_json)
out_txt = p.parent / 'elongation_profile.txt'


def compute_profiles():
    """(phis, nfp, profiles): the elongation of each configuration at args.nphi angles."""
    boozer_surfaces, iota_Gs, axes, fourth, sdf = load(p)
    nfp = axes[0].curve.nfp
    phis = np.linspace(0.0, 1.0 / nfp, args.nphi, endpoint=False)   # phi/2pi over one field period
    profiles = []
    for k, (axis, boozer_surface) in enumerate(zip(axes, boozer_surfaces)):
        # solve the axis field line so that res['length'] is populated (as mk_elongation.py does)
        axis.run_code(CurveLength(axis.curve).J())
        coils = boozer_surface.biotsavart.coils   # the configuration's own coils (aux coils included)
        bs = BiotSavart(coils)
        L = axis.res['length']
        prof = []
        for phi in phis:
            # The tangent map supplies the Chebyshev grid anchored at phi (its matrix D and the
            # axis resampled on it). The value is evaluated with the ONE jitted elongation function
            # above, D being an argument: TangentMap.elongation would compile once per angle.
            tm = TangentMap(axis, bs, 0.0, mtype='identity', phi=float(phi))
            gamma = tm.curve.gamma()
            bs.set_points(gamma)
            prof.append(float(elongation_value(bs.B(), bs.dB_by_dX(), L, gamma, jnp.asarray(tm.D), frame='NB')))
        prof = np.array(prof)
        profiles.append(prof)
        print(f"configuration {k + 1}: elongation over {args.nphi} angles: "
              f"min={prof.min():.4f} max={prof.max():.4f} mean={prof.mean():.4f} "
              f"(max at phi/2pi = {phis[prof.argmax()]:.4f})")
    return phis, nfp, np.array(profiles)


if args.replot:
    # figure only, from the profile already in elongation_profile.txt (no field evaluation)
    data = np.atleast_2d(np.loadtxt(out_txt))
    phis, profiles = data[:, 0], data[:, 1:].T
    nfp = int(round(1.0 / ((phis[1] - phis[0]) * len(phis))))
    names = [f"config{k + 1}" for k in range(len(profiles))]
else:
    phis, nfp, profiles = compute_profiles()
    names = [f"config{k + 1}" for k in range(len(profiles))]
    np.savetxt(out_txt, np.column_stack([phis] + list(profiles)),
               header="phi/2pi " + " ".join(f"elongation_{n}" for n in names))
    print(f"wrote {out_txt}")

# elongation bound of the run, if the yaml next to the json has one
bound = None
yaml_path = p.with_suffix('.yaml')
if yaml_path.is_file():
    m = re.search(r"^ELONGATION_MAX:\s*([0-9.eE+-]+)\s*$", yaml_path.read_text(), re.M)
    bound = float(m.group(1)) if m else None

# figure (Agg: no display on the cluster)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
labels = {1: ["axis"], 2: ["config 1: X-point (aux off)", "config 2: snowflake"]}.get(len(profiles), names)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for prof, label in zip(profiles, labels):
    ax.plot(phis * nfp, prof, lw=1.6, label=f"{label} (max {prof.max():.2f})")
if bound is not None:
    ax.axhline(bound, color="0.35", ls="--", lw=1.0, label=f"bound {bound:g} (enforced at 16 angles)")
ax.set_xlabel(r"toroidal angle over one field period, $\phi\, n_{fp} / 2\pi$")
ax.set_ylabel("near-axis elongation (plane normal to the axis)")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 6.0)   # the same y range on every device, so that the figures compare by eye
ax.grid(alpha=0.3)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_title(os.path.basename(str(p.parent))[-60:], fontsize=8)
fig.tight_layout()
out_png = p.parent / 'elongation_profile.png'
fig.savefig(out_png, dpi=150)
print(f"wrote {out_png}")

summary = p.parent / 'summary.txt'
done = summary.is_file() and 'elongation_profile_max' in summary.read_text()
if not done:
    with open(summary, 'a') as f:
        for n, prof in zip(names, profiles):
            for stat, val in (('max', prof.max()), ('min', prof.min()), ('mean', prof.mean())):
                f.write(f"  {'elongation_profile_' + stat + '_' + n:<30s} {val:.6e}   {'n/a':>16s}   {'n/a':>16s}\n")
    print(f"appended elongation_profile_max/min/mean per configuration to {summary}")
