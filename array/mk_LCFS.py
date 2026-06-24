#!/usr/bin/env python3
"""Compute the last closed flux surface (LCFS) for one device, plus the nested 90/80/70%
toroidal-flux surfaces.

Usage:
    mk_LCFS.py <design_json>

<design_json> is design_opt_final_<ID>.json (written by boozer_all.py) or
design_polished_final_<ID>.json (written by boozer_singular_opt.py). Both stages save the
same five-element list -- [boozer_surfaces, iota_Gs, axes, {xpoints|sing_fls}, sdf] -- so
this script only touches boozer_surfaces[0] / iota_Gs and writes the list back otherwise
unchanged (the 4th element is preserved without inspecting which it is).

The LCFS (largest-volume / lowest-aspect-ratio surface that still converges AND does not
self-intersect) and the nested frac*tf_LCFS surfaces are found by the robust VOLUME
CONTINUATION in star_lite_design.utils.lcfs (shared with boozer_all.py). Outputs, next to
the input json:
    LCFS_<ID>.json, LCFS90_<ID>.json, LCFS80_<ID>.json, LCFS70_<ID>.json
and summary.txt gets LCFS_aspect_ratio + LCFS_hits_vessel appended.
"""
import sys
import re
from pathlib import Path

import numpy as np
from simsopt._core import load, save

from star_lite_design.utils.lcfs import grow_to_lcfs, continue_to_flux, toroidal_flux, snapshot

TF_FRACS = (0.9, 0.8, 0.7)

p = Path(sys.argv[1])

# Only run on the FINAL design jsons: design_opt_final_<ID>.json (from boozer_all.py),
# design_polished_final_<ID>.json (from the polish), or design_unpolished_final_<ID>.json
# (from boozer_singular.py). Refuse anything else BEFORE loading
# -- in particular the diagnostic design_opt_xpoint_deletion.json, which is NOT a
# converged device and must never be grown into an LCFS.
m = re.match(r'design_(?:opt|polished|unpolished)_final_(\d+)\.json$', p.name)
if m is None:
    print(f"mk_LCFS: '{p.name}' is not a design_opt_final_/design_polished_final_/design_unpolished_final_ json "
          f"(e.g. design_opt_xpoint_deletion.json); skipping, no LCFS written.")
    sys.exit(0)
device_id = m.group(1)

dat = load(p)
# Same layout for both stages; we only mutate boozer_surfaces[0] and read iota_Gs[0], and
# save the list back unchanged otherwise, so the 4th element (xpoints for the init device,
# sing_fls for the polished device) is preserved without inspecting it.
boozer_surfaces, iota_Gs, axes, fourth, sdf = dat
bs = boozer_surfaces[0]

# Re-solve the loaded surface at its saved target volume (the (iota, G) guess is the saved
# iota_Gs[0], exactly how boozer_all.py / boozer_singular_opt.py re-solve a loaded
# surface), then snapshot it as the starting (optimization) surface for the flux-fraction
# continuations.
iota0, G0 = iota_Gs[0][0], iota_Gs[0][1]
bs.need_to_run_code = True
bs.run_code(iota0, G0)
ar0 = bs.surface.aspect_ratio()
orig = snapshot(bs)

# LCFS: grow the volume to the largest converging, non-self-intersecting surface.
grow_to_lcfs(bs, iota0, G0)               # leaves bs solved AT the LCFS
ar = bs.surface.aspect_ratio()
tf_lcfs = toroidal_flux(bs)
iota_lcfs = float(bs.res['iota'])         # rotational transform on the LCFS
print(f"LCFS {device_id}: aspect ratio {ar0:.4f} -> {ar:.4f}, iota = {iota_lcfs:.6f}")

out = p.parent / f'LCFS_{device_id}.json'
save([boozer_surfaces, iota_Gs, axes, fourth, sdf], str(out))
print(f"wrote {out}")

# Does the LCFS poke OUTSIDE the vacuum vessel? The magnetic axis is inside the vessel, so
# the sdf sign there marks "inside"; flag if ANY LCFS surface point has the opposite sign
# (the surface has crossed the wall). sdf is the vessel signed-distance loaded with the
# design. This mirrors mk_manifolds.py's inward-manifold/vessel test.
_axis_pt = axes[0].curve.gamma()[0]
_inside_sign = float(np.sign(sdf.eval(np.array([_axis_pt[0]]), np.array([_axis_pt[1]]),
                                      np.array([_axis_pt[2]]))[0]))
_g = bs.surface.gamma().reshape((-1, 3))
_d = np.asarray(sdf.eval(_g[:, 0], _g[:, 1], _g[:, 2]), dtype=float)
lcfs_hits_vessel = bool(_inside_sign != 0.0 and np.any(_inside_sign * _d < 0.0))
print(f"LCFS {device_id}: hits vessel = {int(lcfs_hits_vessel)}")

# Nested frac*tf_LCFS surfaces (90/80/70%): each by continuation from the OPTIMIZATION
# surface (`orig`). continue_to_flux returns (state, reached) and leaves bs at `state`; we
# only save when the target flux was actually reached to 0.1%. Best-effort -- a failure on
# one fraction just skips that surface. frac_iotas records the rotational transform on each
# surface that was reached, for summary.txt.
frac_iotas = {}
for frac in TF_FRACS:
    tag = f"LCFS{int(round(frac * 100))}"
    try:
        state, reached = continue_to_flux(bs, frac, tf_lcfs, orig)
    except Exception as e:
        print(f"{tag} {device_id}: continuation failed ({e}); {tag} not written.")
        continue
    if not reached:
        print(f"{tag} {device_id}: continuation did not reach {frac:.2f}*tf_LCFS to 0.1%; "
              f"{tag} not written.")
        continue
    frac_iotas[tag] = float(bs.res['iota'])
    print(f"{tag} {device_id}: tf/tf_LCFS = {toroidal_flux(bs)/tf_lcfs:.4f}, "
          f"aspect ratio {bs.surface.aspect_ratio():.4f}, iota = {frac_iotas[tag]:.6f}")
    out_f = p.parent / f'{tag}_{device_id}.json'
    save([boozer_surfaces, iota_Gs, axes, fourth, sdf], str(out_f))
    print(f"wrote {out_f}")

# Append the LCFS aspect ratio AND the LCFS/vessel-intersection flag to summary.txt (same
# 4-column "metric value threshold rel_error" format, so device_browser.py parses both as
# metrics). summary.txt was written by the optimizer; this runs before the render's
# mk_manifolds append (which adds inward_manifold_hits_vessel).
summary = p.parent / 'summary.txt'
with open(summary, 'a') as f:
    f.write(f"  {'LCFS_aspect_ratio':<30s} {ar:.6e}   {'n/a':>16s}   {'n/a':>16s}\n")
    f.write(f"  {'LCFS_hits_vessel':<30s} "
            f"{(1.0 if lcfs_hits_vessel else 0.0):.6e}   {'n/a':>16s}   {'n/a':>16s}\n")
    # Rotational transform on the LCFS and on each nested toroidal-flux-fraction surface
    # that was reached (90/80/70%). Surfaces that the continuation did not reach are omitted.
    f.write(f"  {'LCFS_iota':<30s} {iota_lcfs:.6e}   {'n/a':>16s}   {'n/a':>16s}\n")
    for frac in TF_FRACS:
        tag = f"LCFS{int(round(frac * 100))}"
        if tag in frac_iotas:
            f.write(f"  {tag + '_iota':<30s} {frac_iotas[tag]:.6e}   {'n/a':>16s}   {'n/a':>16s}\n")
print(f"appended LCFS_aspect_ratio + LCFS_hits_vessel + LCFS/90/80/70 iota to {summary}")
