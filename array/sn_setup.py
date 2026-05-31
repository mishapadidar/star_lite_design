"""
Single-null (SN) setup helpers for boozer_all.py.

The double-null (DN) device loaded from convert/designA_after_scaled.json is
stellarator-symmetric: nfp=2, stellsym=True surfaces/axis and a 6-coil set built
from a stellsym expansion. For the SN variant we drop stellarator symmetry (keep
nfp=2, set stellsym=False) so the optimizer can build an up-down-asymmetric,
single-null divertor. This module rebuilds the coils and converts the axis,
surface and X-point to non-stellsym representations, and builds the (initially
mirror-image) bottom X-point as its own field line.

Everything here reproduces the original device exactly at initialization (the
symmetry-violating dofs fit to ~0); those dofs are simply freed for the
optimizer. Verified: coil rebuild and surface/axis fits round-trip to ~1e-15.
"""
import numpy as np

from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (CurveXYZFourier, CurveXYZFourierSymmetries, CurveLength,
                         RotatedCurve, SurfaceXYZTensorFourier, BoozerSurface, Volume)

from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.pillpipevessel import VesselDistance

# Newton options used when (re)solving converted field lines / surfaces,
# matching the settings boozer_all.py applies to the axis/X-point/surface.
_FL_OPTIONS = {'newton_tol': 1e-15, 'newton_maxiter': 20, 'verbose': True}
# The Boozer-surface residual floors at ~5e-16, so a 1e-16 tol runs to maxiter
# and reports success=False (true for the original stellsym surface too). We use
# an achievable 1e-14 here: the geometry is identical (residual ~5e-16) but
# res['success'] is True, so boozer_all.fun()'s success gate does not spuriously
# roll back at initialization.
_BS_OPTIONS = {'newton_tol': 1e-14, 'newton_maxiter': 20, 'verbose': True}


# The three independent coils for an nfp=2, stellsym=False expansion are the
# Y>0 representatives, one per rotation orbit: the orbits are (0,2), (1,3),
# (4,5) and the Y>0 members are 0, 3, 4. Expanding these by coils_via_symmetries
# (nfp=2, stellsym=False) reproduces the original 6-coil field exactly (each
# base's pi-rotated copy keeps the same current sign, matching coils 2, 1, 5).
SN_BASE_COIL_IDX = [0, 3, 4]


def _design_current_scale(coils):
    """The current scale factor used in designA: each base coil current is
    ScaledCurrent(Current(unscaled), scale). Read it off a base coil whose
    current_to_scale is a plain Current (coils 0 and 4), not a nested
    ScaledCurrent (coil 3 is ScaledCurrent(coil0_current, -1))."""
    for c in coils:
        cur = c.current
        cts = getattr(cur, 'current_to_scale', None)
        if isinstance(cts, Current) and getattr(cur, 'scale', None) is not None:
            return float(cur.scale)
    raise RuntimeError("could not determine designA current scale")


def rebuild_coils_single_null(coils, base_idx=SN_BASE_COIL_IDX):
    """Re-fit the independent coils onto fresh CurveXYZFourier curves and expand
    them with coils_via_symmetries(nfp=2, stellsym=False).

    Each base current is rebuilt as ScaledCurrent(Current(unscaled), scale) using
    designA's scale, so the optimization dof is the unscaled Current and
    CurrentBound (which reads .current_to_scale / .scale) keeps working. The
    three base currents are now INDEPENDENT (coil 3 no longer shares coil 0's
    dof), as required without stellsym.

    Returns (new_coils, base_curves, base_currents)."""
    # Fourier order of the underlying base CurveXYZFourier coils.
    order = None
    for c in coils:
        cv = c.curve.curve if isinstance(c.curve, RotatedCurve) else c.curve
        if isinstance(cv, CurveXYZFourier):
            order = cv.order
            break
    if order is None:
        raise RuntimeError("could not determine base CurveXYZFourier order")

    scale = _design_current_scale(coils)

    base_curves = []
    base_currents = []
    for i in base_idx:
        src = coils[i].curve
        # Fit on the source curve's own quadpoints (least_squares_fit requires
        # the target sample count to match numquadpoints).
        c = CurveXYZFourier(src.quadpoints, order)
        c.least_squares_fit(src.gamma())
        base_curves.append(c)
        # Unscaled current so that scale * unscaled == original scaled value.
        unscaled = coils[i].current.get_value() / scale
        base_currents.append(ScaledCurrent(Current(unscaled), scale))

    new_coils = coils_via_symmetries(base_curves, base_currents, 2, False)
    return new_coils, base_curves, base_currents


def to_nonstellsym_fieldline(fieldline, new_coils, options=None):
    """Convert a PeriodicFieldLine (axis or top X-point) to an nfp=2,
    stellsym=False one on the rebuilt coil field, and re-solve it.

    A new CurveXYZFourierSymmetries (stellsym=False) is least-squares fit to the
    old curve's gamma; because the initial device is stellsym, the symmetry-
    violating dofs fit to ~0, so the field line is geometrically unchanged but
    those dofs are now free. It is wrapped on a BiotSavart of new_coils and
    solved. (For the X-point, already stellsym=False, this just re-wraps it on
    the rebuilt coil field.)"""
    old = fieldline.curve
    new_curve = CurveXYZFourierSymmetries(
        old.quadpoints, old.order, old.nfp, False, ntor=old.ntor)
    new_curve.least_squares_fit(old.gamma())
    pfl = PeriodicFieldLine(BiotSavart(new_coils), new_curve,
                            options=dict(options) if options else dict(_FL_OPTIONS))
    pfl.run_code(CurveLength(new_curve).J())
    return pfl


def bottom_xpoint_vessel_penalty(sdf, bottom_xpoints, min_dist=0.01, max_dist=0.05):
    """Soft band constraint keeping each bottom X-point field line between
    min_dist (1 cm) and max_dist (5 cm) INSIDE the vessel, i.e. inward depth in
    [min_dist, max_dist]. Returns a single Optimizable = J_lower + J_upper.

    Conventions (verified): inside <=> sdf<0, inward depth = -sdf.
      >= min_dist inside : sign=-1, threshold=+min_dist  penalizes inward<min_dist
      <= max_dist inside : sign=+1, threshold=-max_dist  penalizes inward>max_dist
    Both terms are means over the bottom field-line points (re-solved each
    evaluation, so the band tracks the actual bottom X-point)."""
    bx = list(bottom_xpoints)
    n = len(bx)
    J_lower = VesselDistance(sdf, bx, np.array([-1.0] * n), float(min_dist))
    J_upper = VesselDistance(sdf, bx, np.array([+1.0] * n), -float(max_dist))
    return J_lower + J_upper


def setup_single_null(boozer_surfaces, iota_Gs, axes, xpoints):
    """Convert a loaded double-null (stellsym) design into a single-null
    (nfp=2, stellsym=False) one.

    Rebuilds the coils from the 3 independent Y>0 bases (coils 0,3,4), then for
    each configuration converts the Boozer surface, magnetic axis and (top)
    X-point to the non-stellsym field on the rebuilt coils, and builds the
    bottom X-point as its own solved field line (initialized from the top by
    the stellsym mirror). At initialization everything reproduces the original
    device; the freed dofs let the optimizer build the single-null asymmetry.

    Returns: (new_boozer_surfaces, new_axes, new_xpoints, bottom_xpoints,
              new_coils, base_curves, base_currents)."""
    coils = boozer_surfaces[0].biotsavart.coils
    new_coils, base_curves, base_currents = rebuild_coils_single_null(coils)

    new_bs, new_ax, new_xp, bot_xp = [], [], [], []
    for bsurf, (iota, G), axis, xp in zip(boozer_surfaces, iota_Gs, axes, xpoints):
        new_bs.append(to_nonstellsym_surface(bsurf, new_coils, iota, G))
        new_ax.append(to_nonstellsym_fieldline(axis, new_coils))
        new_xp.append(to_nonstellsym_fieldline(xp, new_coils))
        bot_xp.append(make_bottom_xpoint(xp, new_coils))

    return new_bs, new_ax, new_xp, bot_xp, new_coils, base_curves, base_currents


def to_nonstellsym_surface(boozer_surface, new_coils, iota, G, options=None):
    """Convert a stellsym BoozerSurface to an nfp=2, stellsym=False one on the
    rebuilt coil field, and re-solve it at the given (iota, G).

    A new SurfaceXYZTensorFourier (stellsym=False, same mpol/ntor/nfp and
    quadpoints) is least-squares fit to the old surface gamma (exact round-trip;
    the symmetry-violating dofs fit to ~0 and are freed). The volume target is
    the old surface's volume so the re-solve lands on the same surface."""
    old = boozer_surface.surface
    new_s = SurfaceXYZTensorFourier(
        mpol=old.mpol, ntor=old.ntor, nfp=old.nfp, stellsym=False,
        quadpoints_phi=old.quadpoints_phi, quadpoints_theta=old.quadpoints_theta)
    new_s.least_squares_fit(old.gamma())
    bs = BoozerSurface(BiotSavart(new_coils), new_s, Volume(new_s), old.volume())
    bs.options = dict(options) if options else dict(_BS_OPTIONS)
    bs.run_code(iota, G)
    return bs


def make_bottom_xpoint(top_xpoint, new_coils, options=None):
    """Build the bottom X-point as its OWN field line.

    At initialization the device is stellsym, so the bottom X-point is the
    stellarator-symmetric image of the top: bottom(t) = [X,-Y,-Z] of top(-t).
    [X,-Y,-Z] (a rotation by pi about X) flips the toroidal sense, so we
    evaluate the top curve at the NEGATED parameter (period 1) and transform.
    Evaluating continuously at -t (rather than reversing the samples) gives a
    forward-oriented field line (gammadash // B, so the Newton solve converges;
    verified residual ~1e-14) AND keeps g(0) on the y=0 plane -- because
    top(0) is on y=0, so is [X,-Y,-Z]*top(0). That matches the stellsym=False
    gauge (gamma[0,1]=0), so the solve does not have to roll the parametrization.
    (Negating at period 1/nfp instead would give the wrong parametrization,
    residual O(1).)

    The transform is applied ONLY here (initialization). The returned
    PeriodicFieldLine re-solves each evaluation, so once the optimizer breaks
    up-down symmetry it tracks the device's ACTUAL bottom X-point."""
    top = top_xpoint.curve
    qp = top.quadpoints
    target = top.gamma_pure(top.x, (-qp) % 1.0) * np.array([1.0, -1.0, -1.0])
    new_curve = CurveXYZFourierSymmetries(
        qp, top.order, top.nfp, False, ntor=top.ntor)
    new_curve.least_squares_fit(target)
    pfl = PeriodicFieldLine(BiotSavart(new_coils), new_curve,
                            options=dict(options) if options else dict(_FL_OPTIONS))
    pfl.run_code(CurveLength(new_curve).J())
    return pfl
