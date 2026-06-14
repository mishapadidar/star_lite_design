"""Last-closed-flux-surface (LCFS) and nested toroidal-flux-fraction surface helpers,
shared by array/mk_LCFS.py and array/boozer_all.py.

All routines drive a Boozer surface by setting its targetlabel (a SIGNED VOLUME) and
calling run_code. Toroidal flux increases monotonically with |volume| on nested surfaces,
so a flux target is reached by a robust VOLUME CONTINUATION with backtracking and a fine
self-intersection check -- never by a single large jump. Surface state is passed around as
small dicts {'V', 'sdofs', 'iota', 'G'} so a surface can be snapshotted and restored
cheaply.
"""
import numpy as np
from simsopt.field import BiotSavart
from simsopt.geo import ToroidalFlux

# Self-intersection check resolution. Kept equal to mk_manifolds.NTHETA_CROSS so a surface
# that passes is drawn identically clean on the xs figure.
N_SI_THETA = 1024
N_SI_PHI = 8


def toroidal_flux(bs):
    """Toroidal flux of the Boozer surface in its own coil field (as magneticwell.py:
    ToroidalFlux(surface, BiotSavart(coils))). A fresh BiotSavart keeps the point cache
    independent of bs.biotsavart."""
    return float(ToroidalFlux(bs.surface, BiotSavart(bs.biotsavart.coils)).J())


def _self_intersects(surface):
    """True if any cross-section self-intersects, checked on a FINE theta grid at several
    phi planes over one field period (half-period when stellarator-symmetric, since the
    other half is its mirror; full period otherwise)."""
    phimax = (0.5 if surface.stellsym else 1.0) / surface.nfp
    return any(surface.is_self_intersecting(angle=float(phi), thetas=N_SI_THETA)
               for phi in np.linspace(0.0, phimax, N_SI_PHI, endpoint=False))


def snapshot(bs):
    """Current solved state of bs as a {'V','sdofs','iota','G'} dict (V = targetlabel)."""
    return {'V': bs.targetlabel, 'sdofs': bs.surface.x.copy(),
            'iota': bs.res['iota'], 'G': bs.res['G']}


def _try_volume(bs, V, start):
    """Warm-start bs from `start`, retarget to signed volume V, re-solve. True iff it
    converges with no self-intersection. A degenerate surface can make the Newton solve
    OR is_self_intersecting() itself RAISE (at the feasibility boundary the cross-section
    'goes back on itself'); any such failure counts as 'did not converge'."""
    bs.surface.x = start['sdofs']
    bs.res['iota'], bs.res['G'] = start['iota'], start['G']
    bs.targetlabel = V
    bs.need_to_run_code = True
    try:
        rr = bs.run_code(start['iota'], start['G'])
        return bool(rr['success']) and not _self_intersects(bs.surface)
    except Exception:
        return False


def grow_to_lcfs(bs, iota0, G0):
    """Grow |volume| from bs's current surface to the largest value that still converges
    without self-intersecting -- the last closed flux surface (LCFS). bs must already hold
    the starting surface; (iota0, G0) is the initial Boozer solve guess. Adaptive step
    with backtracking (grow x1.5 on success, shrink x0.5 on failure). Leaves bs solved AT
    the LCFS and returns its state dict."""
    bs.need_to_run_code = True
    bs.run_code(iota0, G0)
    best = snapshot(bs)
    sgn = 1.0 if best['V'] >= 0.0 else -1.0
    Vmag0 = abs(best['V'])
    if Vmag0 > 0.0:
        step, min_step = 0.10 * Vmag0, 1e-3 * Vmag0
        GROW, SHRINK = 1.5, 0.5
        while step >= min_step:
            if _try_volume(bs, sgn * (abs(best['V']) + step), best):
                best = snapshot(bs)
                step *= GROW
            else:
                step *= SHRINK
    _try_volume(bs, best['V'], best)   # leave bs solved at the LCFS
    return best


def continue_to_flux(bs, frac, tf_ref, start, frac_tol=1e-3):
    """Continuation from `start` (a converged, non-self-intersecting state dict) toward
    the surface whose toroidal flux is frac*tf_ref. Marches |volume| OUTWARD (grow) if the
    target flux is outside `start`, INWARD (shrink) if inside it, with BACKTRACKING and a
    self-intersection check at every step; only converging, non-self-intersecting steps
    advance, and a failure or a flux overshoot shrinks the step.

    Returns (state, reached): `state` is the furthest-advanced good surface toward the
    target (== `start` if none advanced), and `reached` is True iff its flux is within
    `frac_tol` (relative) of frac*tf_ref. Leaves bs solved at the returned state."""
    tf_target = frac * tf_ref
    _try_volume(bs, start['V'], start)         # ensure bs is at `start`
    tf_start = toroidal_flux(bs)
    if abs(tf_start - tf_target) <= frac_tol * abs(tf_target):
        return dict(start), True               # start is already at the target flux
    sgn = 1.0 if start['V'] >= 0.0 else -1.0
    direction = 1.0 if (tf_start / tf_ref) < frac else -1.0
    anchor, reached = dict(start), False
    step, min_step = 0.10 * abs(start['V']), 1e-6 * abs(start['V'])
    GROW, SHRINK = 1.5, 0.5
    while step >= min_step:
        V_mag = abs(anchor['V']) + direction * step
        if V_mag <= 0.0:                       # cannot shrink past zero volume
            step *= SHRINK
            continue
        if _try_volume(bs, sgn * V_mag, anchor):
            try:
                tf_try = toroidal_flux(bs)
            except Exception:
                step *= SHRINK
                continue
            if abs(tf_try - tf_target) <= frac_tol * abs(tf_target):
                anchor, reached = snapshot(bs), True
                break
            f = tf_try / tf_ref                # overshoot = crossed the target this way
            overshoot = (f > frac) if direction > 0 else (f < frac)
            if overshoot:
                step *= SHRINK                 # back off; keep the anchor on the near side
            else:
                anchor = snapshot(bs)          # advance toward the target + speed up
                step *= GROW
        else:
            step *= SHRINK                     # backtrack on failure / self-intersection
    _try_volume(bs, anchor['V'], anchor)       # leave bs at the returned state
    return anchor, reached
