#!/usr/bin/env python3
"""Compute the last closed flux surface (LCFS) for one device.

Usage:
    mk_LCFS.py <design_json>

<design_json> is design_opt_final_<ID>.json (written by boozer_all.py) or
design_polished_final_<ID>.json (written by boozer_singular_opt.py). Both stages save
the same five-element list -- [boozer_surfaces, iota_Gs, axes, {xpoints|sing_fls}, sdf]
-- so this script only ever touches boozer_surfaces[0] / iota_Gs and writes the list
back otherwise unchanged (the fourth element is preserved without caring which it is).

The LCFS is found by ADAPTIVE CONTINUATION on the Boozer-surface target volume: starting
from the device's converged surface, the (signed) enclosed volume is grown step by step.
After every step the surface is re-solved; if it converges with no self-intersection the
step is accepted (and the step grows a little), and if the solve fails OR the surface
self-intersects the step is halved and retried from the last good surface. This walks the
volume up to the feasibility boundary, so the surface returned is the largest-volume /
LOWEST-aspect-ratio surface that still converges and does not self-intersect.

The result is solved IN PLACE on boozer_surfaces[0] and saved next to the input as
LCFS_<ID>.json (same list layout, surface 0 at the LCFS).
"""
import sys
import re
from pathlib import Path

import numpy as np
from simsopt._core import load, save

p = Path(sys.argv[1])

# Only run on the FINAL design jsons: design_opt_final_<ID>.json (from boozer_all.py)
# or design_polished_final_<ID>.json (from the polish). Refuse anything else BEFORE
# loading -- in particular the diagnostic design_opt_xpoint_deletion.json, which is NOT
# a converged device and must never be grown into an LCFS. The device ID is the trailing
# number captured here.
m = re.match(r'design_(?:opt|polished)_final_(\d+)\.json$', p.name)
if m is None:
    print(f"mk_LCFS: '{p.name}' is not a design_opt_final_/design_polished_final_ json "
          f"(e.g. design_opt_xpoint_deletion.json); skipping, no LCFS written.")
    sys.exit(0)
device_id = m.group(1)

dat = load(p)
# Same layout for both stages; we only mutate boozer_surfaces[0] and read iota_Gs[0],
# and save the list back unchanged otherwise, so the 4th element (xpoints for the init
# device, sing_fls for the polished device) is preserved without inspecting it.
boozer_surfaces, iota_Gs, axes, fourth, sdf = dat
bs = boozer_surfaces[0]

# Self-intersection check for the continuation: test cross-sections on a FINE theta grid
# at several phi planes. The default is_self_intersecting() checks a SINGLE angle on the
# surface's own (coarse) theta grid, which can miss a fold that appears between
# quadpoints or at a different toroidal angle. Half a field period [0, 0.5/nfp) suffices
# when stellarator-symmetric (the other half is its mirror); the full field period
# [0, 1/nfp) otherwise (SN / non-stellsym).
N_SI_THETA = 1024                                                  # theta points per cross-section
_si_max = (0.5 if bs.surface.stellsym else 1.0) / bs.surface.nfp
SI_PHIS = np.linspace(0.0, _si_max, 8, endpoint=False)             # phi/2pi planes to check

# Re-solve the loaded surface at its saved target volume so res (iota, G) is populated.
# The (iota, G) initial guess is the saved iota_Gs[0] -- exactly how boozer_all.py and
# boozer_singular_opt.py re-solve a freshly loaded surface.
iota0, G0 = iota_Gs[0][0], iota_Gs[0][1]
bs.need_to_run_code = True
bs.run_code(iota0, G0)
ar0 = bs.surface.aspect_ratio()   # aspect ratio of the device's starting surface

V_old = bs.targetlabel
# The Boozer-surface volume is SIGNED (negative for some devices); keep its sign and
# grow the MAGNITUDE (a larger volume == a lower aspect ratio).
sgn = 1.0 if V_old >= 0.0 else -1.0
Vmag0 = abs(V_old)

# best = the last surface state that converged AND is non-self-intersecting.
best = {'V': V_old, 'sdofs': bs.surface.x.copy(),
        'iota': bs.res['iota'], 'G': bs.res['G']}


def _try_volume(V):
    """Warm-start from the best surface, retarget to (signed) V, re-solve. True iff it
    converges with no self-intersection. A degenerate surface can make the Newton solve
    OR is_self_intersecting() itself RAISE (at very low aspect ratio the cross-section
    'goes back on itself', so the cylindrical angle is no longer monotonic); treat any
    such failure as 'did not converge' (return False) so the continuation just backs
    off the step instead of crashing."""
    bs.surface.x = best['sdofs']
    bs.res['iota'], bs.res['G'] = best['iota'], best['G']
    bs.targetlabel = V
    bs.need_to_run_code = True
    try:
        rr = bs.run_code(best['iota'], best['G'])
        if not bool(rr['success']):
            return False
        # No cross-section may self-intersect, checked on a fine theta grid at each of
        # the SI_PHIS planes (a degenerate cross-section here raises -> caught below).
        return not any(bs.surface.is_self_intersecting(angle=float(phi), thetas=N_SI_THETA)
                       for phi in SI_PHIS)
    except Exception:
        return False


# Adaptive continuation on |volume|: accept-and-grow on success, shrink-and-retry on
# failure, until the step falls below a small fraction of the starting volume.
step = 0.10 * Vmag0          # initial continuation step: 10% of the current volume
min_step = 1e-3 * Vmag0      # resolution floor: 0.1% of the current volume
GROW, SHRINK = 1.5, 0.5
n_accept = 0
if Vmag0 > 0.0:
    while step >= min_step:
        V_try = sgn * (abs(best['V']) + step)
        if _try_volume(V_try):
            best = {'V': V_try, 'sdofs': bs.surface.x.copy(),
                    'iota': bs.res['iota'], 'G': bs.res['G']}
            n_accept += 1
            step *= GROW         # speed up while it keeps converging
        else:
            step *= SHRINK       # back off near the feasibility boundary

# Adopt the LCFS: restore + re-solve at best so the saved surface is exactly it.
bs.surface.x = best['sdofs']
bs.res['iota'], bs.res['G'] = best['iota'], best['G']
bs.targetlabel = best['V']
bs.need_to_run_code = True
bs.run_code(best['iota'], best['G'])

ar = bs.surface.aspect_ratio()
print(f"LCFS {device_id}: accepted {n_accept} continuation steps, "
      f"|V| {Vmag0:.6e} -> {abs(best['V']):.6e}, "
      f"aspect ratio {ar0:.4f} -> {ar:.4f}")

out = p.parent / f'LCFS_{device_id}.json'
save([boozer_surfaces, iota_Gs, axes, fourth, sdf], str(out))
print(f"wrote {out}")

# Does the LCFS poke OUTSIDE the vacuum vessel? The magnetic axis is inside the vessel,
# so the sdf sign there marks "inside"; flag if ANY LCFS surface point has the opposite
# sign (the surface has crossed the wall). sdf is the vessel signed-distance loaded with
# the design. This mirrors mk_manifolds.py's inward-manifold/vessel test.
_axis_pt = axes[0].curve.gamma()[0]
_inside_sign = float(np.sign(sdf.eval(np.array([_axis_pt[0]]), np.array([_axis_pt[1]]),
                                      np.array([_axis_pt[2]]))[0]))
_g = bs.surface.gamma().reshape((-1, 3))
_d = np.asarray(sdf.eval(_g[:, 0], _g[:, 1], _g[:, 2]), dtype=float)
lcfs_hits_vessel = bool(_inside_sign != 0.0 and np.any(_inside_sign * _d < 0.0))
print(f"LCFS {device_id}: hits vessel = {int(lcfs_hits_vessel)}")

# Append the LCFS aspect ratio AND the LCFS/vessel-intersection flag to summary.txt
# (same 4-column "metric value threshold rel_error" format, so device_browser.py parses
# both as metrics). summary.txt was written by the optimizer; this runs before the
# render's mk_manifolds append (which adds inward_manifold_hits_vessel).
summary = p.parent / 'summary.txt'
with open(summary, 'a') as f:
    f.write(f"  {'LCFS_aspect_ratio':<30s} {ar:.6e}   {'n/a':>16s}   {'n/a':>16s}\n")
    f.write(f"  {'LCFS_hits_vessel':<30s} "
            f"{(1.0 if lcfs_hits_vessel else 0.0):.6e}   {'n/a':>16s}   {'n/a':>16s}\n")
print(f"appended LCFS_aspect_ratio + LCFS_hits_vessel to {summary}")
