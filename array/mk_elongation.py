#!/usr/bin/env python3
"""Compute the min / max / mean flux-surface elongation along the magnetic axis and
append it to summary.txt.

Usage:
    mk_elongation.py <design_json>

<design_json> is design_opt_final_<ID>.json (from boozer_all.py) or
design_polished_final_<ID>.json (from the polish). The elongation at a toroidal angle
phi is the aspect ratio of the invariant ellipse of the one-field-period RETURN MAP
anchored at phi (utils/tangent_map.py: TangentMap.elongation -- built only from the
return map R[-1], the spectrally meaningful matrix). We sample it at NPHI angles across
one field period by initializing a SEPARATE TangentMap per phi -- each integrates its
OWN variational equation over [phi, phi+1/nfp], so every value comes from a genuine
return map -- then report elongation_min / elongation_max / elongation_mean.

NOTE: a more efficient way to do this exists. Initializing one tangent map per phi
re-integrates the variational equation NPHI times. Instead one could integrate ONCE
over [0, 1/nfp] to get the propagator T(s) at every node, then obtain the return map at
each phi by the conjugation Phi(phi) = T(s) * Phi0 * T(s)^{-1} -- equivalently, propagate
the single invariant ellipse, E(phi) = T(s) * E0 -- so one solve yields the whole
profile. It is done the simple (per-phi) way here for clarity; switch to the
single-integration form if this becomes a bottleneck.
"""
import sys
import re

import numpy as np
from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import CurveLength
from pathlib import Path

from star_lite_design.utils.tangent_map import TangentMap

NPHI = 32   # number of toroidal samples across one field period [0, 1/nfp)

p = Path(sys.argv[1])

# Only run on the FINAL design jsons (design_opt_final_<ID>.json from boozer_all.py or
# design_polished_final_<ID>.json from the polish). Refuse anything else -- in
# particular the diagnostic design_opt_xpoint_deletion.json.
m = re.match(r'design_(?:opt|polished)_final_(\d+)\.json$', p.name)
if m is None:
    print(f"mk_elongation: '{p.name}' is not a design_opt_final_/design_polished_final_ "
          f"json (e.g. design_opt_xpoint_deletion.json); skipping, summary.txt unchanged.")
    sys.exit(0)
device_id = m.group(1)

dat = load(p)
boozer_surfaces, iota_Gs, axes, fourth, sdf = dat
axis = axes[0]
# Solve the axis field line so res['length'] is populated (mirrors the drivers).
axis.run_code(CurveLength(axis.curve).J())

nfp = axis.curve.nfp
phis = np.linspace(0.0, 1.0 / nfp, NPHI, endpoint=False)   # phi/2pi over one period

elongs = []
for phi in phis:
    # Fresh BiotSavart over the device coils so each tangent map's point cache is
    # isolated (the same pattern boozer_all.py uses); for the polished device these
    # coils are the combined modular+aux set. threshold/mtype are unused by .elongation;
    # phi shifts the integration window to [phi, phi+1/nfp], anchoring the return map there.
    tm = TangentMap(axis, BiotSavart(boozer_surfaces[0].biotsavart.coils),
                    0.0, mtype='identity', phi=float(phi))
    elongs.append(float(tm.elongation))

elongs = np.array(elongs)
emin, emax, emean = float(elongs.min()), float(elongs.max()), float(elongs.mean())
print(f"elongation {device_id} (NPHI={NPHI}): "
      f"min={emin:.4f} max={emax:.4f} mean={emean:.4f}")

# Append to summary.txt in the device dir (same 4-column "metric value threshold
# rel_error" format the rest of summary.txt uses, so device_browser.py parses these).
summary = p.parent / 'summary.txt'
with open(summary, 'a') as f:
    for name, val in (('elongation_min', emin),
                      ('elongation_max', emax),
                      ('elongation_mean', emean)):
        f.write(f"  {name:<30s} {val:.6e}   {'n/a':>16s}   {'n/a':>16s}\n")
print(f"appended elongation_min/max/mean to {summary}")
