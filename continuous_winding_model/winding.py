import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.optimize import lsq_linear
from scipy.interpolate import CubicSpline

import simsoptpp as sopp
from simsopt.geo import CurveXYZFourier, RotatedCurve, FramedCurveCentroid
from simsopt.geo.curve import Curve
from simsopt.field import Coil, Current

from star_lite_design.finite_build.hex_finite_build_coil import _hex_grid_of_circles

"""
Continuous winding model of a Star_Lite coil.

A single cable is wound around the steel spine in the slots of a hexagonal lattice.
The cable is laid point by point in the order it is wound. The starting lead leaves
the first slot straight out of the pack at the start position t0. Each wind lies
snug against the wind before it and goes around the lead; it steps over to the next
slot where it meets the previous wind's step. Only the inner layer is modelled so far.
See README.md for the model and its parameters.
"""


class SampledCurve(sopp.Curve, Curve):
    """
    Closed curve known only at n equally spaced parameter values in [0, 1): its points
    ``gamma`` and tangents ``gammadash`` there. The quadrature in ``BiotSavart`` (equal
    weights 1/n) is then the trapezoid rule. The curve has no dofs.
    """

    def __init__(self, gamma, gammadash):
        self._gamma = np.asarray(gamma, dtype=float)
        self._gammadash = np.asarray(gammadash, dtype=float)
        sopp.Curve.__init__(self, np.linspace(0, 1, self._gamma.shape[0], endpoint=False))
        Curve.__init__(self, x0=np.asarray([]))

    def gamma_impl(self, gamma, quadpoints):
        gamma[:] = self._gamma

    def gammadash_impl(self, gammadash):
        gammadash[:] = self._gammadash

    def num_dofs(self):
        return 0

    def get_dofs(self):
        return np.array([])

    def set_dofs_impl(self, dofs):
        pass


class PolylineCurve(SampledCurve):
    """
    Closed polyline through the points ``nodes``. The points of the curve are the
    segment midpoints and ``gammadash`` is the segment vector times the number of
    segments, so the quadrature in ``BiotSavart`` is the midpoint rule on each
    segment.
    """

    def __init__(self, nodes):
        self.nodes = np.asarray(nodes, dtype=float)
        segments = np.roll(self.nodes, -1, axis=0) - self.nodes
        super().__init__(self.nodes + 0.5 * segments, self.nodes.shape[0] * segments)


def _split_long_segments(nodes):
    """The closed polyline ``nodes`` with segments over 3 times the median length split into straight pieces."""
    nodes = np.asarray(nodes, dtype=float)
    segments = np.roll(nodes, -1, axis=0) - nodes
    length = np.linalg.norm(segments, axis=1)
    median = np.median(length)
    pieces = np.where(length > 3 * median, np.ceil(length / median), 1).astype(int)
    return np.vstack([p + np.arange(m)[:, None] / m * d for p, d, m in zip(nodes, segments, pieces)])


def spline_curve(nodes, n):
    """
    The closed curve through the points ``nodes`` as a periodic cubic spline in arc
    length, sampled at n points equally spaced in arc length (a SampledCurve, so
    BiotSavart uses the trapezoid rule on the spline). Segments much longer than the others
    (the closing piece between the far ends of the leads) are first split into straight
    pieces, so the spline does not bulge across them.
    """
    nodes = _split_long_segments(nodes)
    P = np.vstack((nodes, nodes[:1]))
    s = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
    spline = CubicSpline(s, P, bc_type="periodic")
    t = np.linspace(0, s[-1], n, endpoint=False)
    return SampledCurve(spline(t), s[-1] * spline(t, 1))


def spline_panel_curve(nodes, h, order=4, corner_angle=np.pi / 6, bend_fraction=None):
    """
    The closed curve through the points ``nodes``, split at its corners (nodes where it
    turns by more than corner_angle) into smooth pieces, each a cubic spline in arc
    length, integrated with Gauss-Legendre panels of ``order`` points about h apart.
    The quadrature weights go into gammadash, so BiotSavart's equal weights give the
    Gauss rule on each piece, and the corners do not spoil its convergence. A SampledCurve.
    With bend_fraction, the panels are also no longer than bend_fraction times the local
    bend radius, so the tight bends near the leads get more points.
    """
    nodes = _split_long_segments(nodes)
    n = nodes.shape[0]
    t = np.roll(nodes, -1, axis=0) - nodes
    t /= np.linalg.norm(t, axis=1)[:, None]
    corners = np.where(np.sum(t * np.roll(t, 1, axis=0), axis=1) < np.cos(corner_angle))[0]
    if corners.size == 0:
        P = np.vstack((nodes, nodes[:1]))
        return spline_curve(nodes, int(np.ceil(np.linalg.norm(np.diff(P, axis=0), axis=1).sum() / h)))
    x_gauss, w_gauss = np.polynomial.legendre.leggauss(order)
    gamma, dgamma, weight = [], [], []
    for a, b in zip(corners, np.roll(corners, -1)):
        P = nodes[np.arange(a, b + 1 if b > a else b + n + 1) % n]
        s = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))))
        spline = CubicSpline(s, P)
        if bend_fraction is None:
            edges = np.linspace(0, s[-1], int(np.ceil(s[-1] / (order * h))) + 1)
        else:
            # panel length min(order h, bend_fraction / curvature), placed by equidistributing
            # the number of panels per unit length
            density = np.maximum(1 / (order * h), np.linalg.norm(spline(s, 2), axis=1) / bend_fraction)
            count = np.concatenate(([0.], np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(s))))
            edges = np.interp(np.linspace(0, count[-1], int(np.ceil(count[-1])) + 1), count, s)
        for e0, e1 in zip(edges[:-1], edges[1:]):
            q = 0.5 * (e0 + e1) + 0.5 * (e1 - e0) * x_gauss
            gamma.append(spline(q))
            dgamma.append(spline(q, 1))
            weight.append(0.5 * (e1 - e0) * w_gauss)
    weight = np.concatenate(weight)
    return SampledCurve(np.vstack(gamma), weight.size * weight[:, None] * np.vstack(dgamma))


def refine(curve, quadpoints):
    """
    Copy of ``curve`` with new quadrature points (an int or an array). ``curve`` is a
    CurveXYZFourier, possibly wrapped in RotatedCurves; the rotations are kept.
    """
    rotmats = []
    while not isinstance(curve, CurveXYZFourier):
        rotmats.append(curve.rotmat)
        curve = curve.curve
    fine = CurveXYZFourier(quadpoints, curve.order)
    fine.local_full_x = curve.local_full_x
    for rotmat in reversed(rotmats):
        fine = RotatedCurve(fine, 0., False)
        fine.rotmat = rotmat
        fine.rotmatT = rotmat.T.copy()
    return fine


def slot_sequence(n_layers, radius, directions, start_slots):
    """
    Order in which the slots of the hexagonal lattice are filled.

    The slots of each layer are sorted by angle. Layer ``l`` is filled in angular
    order starting at slot ``start_slots[l]``, counterclockwise in the (normal,
    binormal) plane if ``directions[l] = +1`` and clockwise if ``-1``.

    Returns:
        rho, theta: polar coordinates of the slots in winding order.
        layer: layer index of each slot.
    """
    shifts_normal, shifts_binormal = _hex_grid_of_circles(n_layers, radius)
    a = np.asarray(shifts_normal)
    b = np.asarray(shifts_binormal)
    rho = np.hypot(a, b)
    theta = np.mod(np.arctan2(b, a), 2*np.pi)
    # layer l of the hexagonal lattice holds 6l slots, at distances between sqrt(3) l r and 2 l r
    layer = np.rint(np.ceil(rho / (2*radius) - 1e-8)).astype(int) - 1

    order = []
    for l in range(n_layers):
        idx = np.where(layer == l)[0]
        idx = idx[np.argsort(theta[idx])]
        k = (start_slots[l] + directions[l] * np.arange(idx.size)) % idx.size
        order.extend(idx[k])
    order = np.asarray(order)
    return rho[order], theta[order], layer[order]


def smooth_above(target, lower, upper, h, bend_length, active_tol=1e-5):
    """
    Smoothing spline that stays within bounds: the minimizer of
    sum (bend_length^2 y'')^2 + (y - target)^2 subject to lower <= y <= upper, on a
    grid of spacing h. A cable pulled toward ``target`` by its tension, resisting
    bending, and kept off obstacles by the bounds.

    lsq_linear's iterative solve stops a few micrometres off the minimizer, which shows
    up as node to node jitter where y should lie on a bound. The result is polished:
    the points within active_tol of a bound are put on it and the others solved for
    exactly, until none of them leaves the bounds.
    """
    n = target.size
    if n <= 2 or (np.ptp(target) == 0 and np.all(lower <= target)):
        return target.copy()
    upper = np.maximum(upper, lower + 1e-12)
    D2 = sp.diags([1., -2., 1.], [0, 1, 2], shape=(n - 2, n)) / h**2
    A = sp.vstack((bend_length**2 * D2, sp.identity(n))).tocsc()
    rhs = np.concatenate((np.zeros(n - 2), target))
    y = lsq_linear(A, rhs, bounds=(lower, upper), lsmr_tol="auto").x
    at_lower, at_upper = y <= lower + active_tol, y >= upper - active_tol
    for _ in range(20):
        y = np.where(at_lower, lower, np.where(at_upper, upper, y))
        free = ~(at_lower | at_upper)
        if not free.any():
            break
        Af = A[:, free]
        y[free] = spsolve((Af.T @ Af).tocsc(), Af.T @ (rhs - A[:, ~free] @ y[~free]))
        if np.all((y[free] >= lower[free]) & (y[free] <= upper[free])):
            break
        at_lower |= free & (y < lower)
        at_upper |= free & (y > upper)
    return np.clip(y, lower, upper)


def taut_over(f, h, sag_radius):
    """
    A cable under tension lying over the surface f (on a grid of spacing h), pressed
    onto it gently: the lowest curve above f whose curvature nowhere exceeds
    1/sag_radius. It bridges dips narrower than that sag allows, like a taut rope.
    That is 1/(2 sag_radius) x^2 plus the upper concave hull of f - 1/(2 sag_radius) x^2.
    """
    x = h * np.arange(f.size)
    g = f - 0.5 * x**2 / sag_radius
    hull = []  # upper concave hull, monotone chain
    for i in range(f.size):
        while len(hull) >= 2 and (x[hull[-1]] - x[hull[-2]]) * (g[i] - g[hull[-2]]) >= (g[hull[-1]] - g[hull[-2]]) * (x[i] - x[hull[-2]]):
            hull.pop()
        hull.append(i)
    return np.interp(x, x[hull], g[hull]) + 0.5 * x**2 / sag_radius


def continuous_winding(curve, n_winds=3, radius=0.0108, direction=1, start_slot=0, outer_start_slot=11,
                       t0=0., twist=0., lead_angle=None, lead_length=1.0, lead_exit_radius=None,
                       lead_gap=None, bend_length=0.01, sag_radius=0.2, contact_tol=5e-4, n_per_turn=1000):
    """
    Path of a continuous cable wound around the spine ``curve`` in the hexagonal
    lattice, laid in the order it is wound (see README.md): 6 winds in the inner layer
    and 12 in the outer layer, wound in the same direction.

    The spine is sampled at ``n_per_turn`` points starting at t0, and the cable gets
    one point per spine point it passes. The position of a point in the normal plane
    is its angle psi around the spine in the centroid frame (normal, binormal),
    measured from the first slot in the winding direction, and its distance rho from
    the spine. The cable rests on what is beneath it (pulled in by the winding
    tension): rho is the smallest distance >= 2r at which it is clear of the spine and
    of the cable of the layers below. The angle is set by the contacts below.

    - Starting lead: a straight rod from the first slot at t0, perpendicular to the
      spine, in the direction ``lead_angle`` (default: straight out from the slot).
    - Contact: at each spine point the cable is pushed, in the winding direction, to
      the smallest angle where it is 2r from the cable of its own layer laid one turn
      before. So each wind lies snug against the previous one and steps over to the
      next slot where it meets the previous wind's step.
    - The lead: a cable passes the lead on the side its centre is on as it gets
      there. At or just past the lead angle (the first wind, the outer wind on top of
      it) it is pushed past it like past any other cable; just before it, it stays on
      the near side.
    - Squeezed: where the lead takes up a slot of the outer layer, an outer cable
      with no room left rides up over its neighbours.
    - Climbing to the next layer: when the last wind of the inner layer, pushed by
      the previous wind's step, has no room left in the inner layer, it climbs,
      starting to lift off 3 ``bend_length`` before that point. From there on the inner
      layer is beneath it, and it settles into the outer slot ``outer_start_slot``.
    - Bending: the cable is pulled toward its slot by the winding tension, within the
      bounds above, and resists bending. Each turn of angle is the minimizer of
      sum (bend_length^2 y'')^2 + (y - y_slot)^2 within the bounds, where
      y = 2r psi. ``bend_length`` sets how early the cable starts to move over
      before it touches an obstacle. Once a wind has been pushed more than half a
      slot forward, it is pulled toward the next slot.
    - Tension: the distance from the spine is not pulled into every dip of the
      surface beneath. The cable spans dips like a taut rope (free spans curve at
      most 1/``sag_radius``) and is then smoothed for bending like the angle.
    - Finishing lead: the last wind runs on, riding over whatever is in its way,
      until ``lead_gap`` before the starting lead. There the end of the cable curves
      out of the pack and over to the starting lead, and runs alongside it,
      touching it if ``lead_gap`` is 2r.

    The circuit is closed between the far ends of the leads.

    Args:
        curve: the spine; a CurveXYZFourier, possibly wrapped in RotatedCurves.
        n_winds: number of winds, at most 18.
        radius: cable minor radius [m].
        direction: winding direction, +1 counterclockwise in the (normal, binormal) plane.
        start_slot: first slot of the inner layer, in angular order.
        outer_start_slot: first slot of the outer layer, counted in the winding
            direction from the first inner slot, at 30 degrees per slot. The even
            slots sit on top of an inner cable, the odd ones in the groove between two.
            Default 11: the groove between the last and the first inner slot.
        t0: start position of the winding along the spine, in [0, 1).
        twist: rotation of the whole pack about the spine [rad].
        lead_angle: direction of the starting lead in the normal plane at t0 [rad].
        lead_length: length of the straight part of each lead outside the pack [m].
        lead_exit_radius: distance along the starting lead, from its slot, where the
            finishing lead comes alongside it [m]. Default 6r.
        lead_gap: distance along the spine from the starting lead to the finishing
            lead [m]. Default 2r, side by side.
        bend_length: ratio of bending stiffness to winding tension, as a length [m].
        sag_radius: smallest radius of curvature of a free span of cable, set by the
            tension against the pressure that holds it on the pack [m].
        contact_tol: cables closer than 2*radius - contact_tol are in contact [m].
        n_per_turn: number of spine points per turn.

    Returns:
        nodes: (n, 3) array, points of the closed path.
        info: dict with the index ranges of the winds and leads, and the local coordinates.
    """
    n_inner, n_outer = 6, 12
    if n_winds > n_inner + n_outer:
        raise ValueError(f"the two layers hold {n_inner + n_outer} winds")
    if lead_exit_radius is None:
        lead_exit_radius = 6 * radius
    if lead_gap is None:
        lead_gap = 2 * radius
    d0 = 2 * radius
    dmin = d0 - contact_tol

    _, theta, _ = slot_sequence(1, radius, (direction,), (start_slot,))
    theta0 = theta[0] + twist
    if lead_angle is None:
        lead_angle = theta0
    psi_outer = outer_start_slot * np.pi / 6

    def slot(k):
        """Angle of the slot of wind k."""
        return k * np.pi / 3 if k < n_inner else psi_outer + (k - n_inner) * np.pi / 6

    # spine and its centroid frame on a grid that starts at t0
    N = n_per_turn
    spine = refine(curve, np.mod(t0 + np.linspace(0, 1, N, endpoint=False), 1))
    gamma = spine.gamma()
    _, normal, binormal = FramedCurveCentroid(spine).rotated_frame()
    arclength = np.linalg.norm(spine.gammadash(), axis=1)
    s = np.concatenate(([0.], np.cumsum(0.5 * (arclength[1:] + arclength[:-1]) / N)))
    L = s[-1] + 0.5 * (arclength[-1] + arclength[0]) / N
    ds = arclength / N
    W = int(np.ceil(d0 / ds.min())) + 1  # spine points within one cable diameter

    def unit(psi):
        phi = theta0 + direction * np.asarray(psi)
        return np.stack((np.cos(phi), np.sin(phi)), axis=-1)

    def xyz(j, uv):
        return gamma[j] + uv[..., :1] * normal[j] + uv[..., 1:] * binormal[j]

    # starting lead; only its part next to the pack can touch the winds
    direction_in = np.array([np.cos(lead_angle), np.sin(lead_angle)])
    reach = lead_exit_radius - d0 + lead_length
    lam = np.linspace(0, reach, max(2, int(np.ceil(reach / 0.002))))
    lead_in_uv = d0 * unit(0.) + lam[:, None] * direction_in
    lead_xyz = xyz(0, lead_in_uv)
    lead_obstacle = lead_xyz[np.linalg.norm(lead_in_uv, axis=1) < 4 * d0]

    J_total = n_winds * N
    psi = np.zeros(J_total)
    rho = np.full(J_total, d0)
    P = np.zeros((J_total, 3))
    layer_of = np.zeros(J_total, dtype=int)  # layer each point is in
    n_laid = 0

    def obstacles(J, layer):
        """Cable laid about one turn or more before point J near spine point J: (beneath, same layer or above)."""
        idx = (J - N * np.arange(1, J // N + 2)[:, None] + np.arange(-W, W + 1)[None, :]).ravel()
        idx = idx[(idx >= 0) & (idx < n_laid) & (idx <= J - N // 2)]
        return P[idx[layer_of[idx] < layer]], P[idx[layer_of[idx] >= layer]]

    def near_lead(J):
        j = J % N
        return J >= N // 2 and min(j, N - j) <= W

    def rest(j, p, beneath):
        """Points at angles p resting on the spine and the cable beneath, and their distances from the spine."""
        e = unit(p)
        r_rest = np.full(e.shape[0], d0)
        if beneath.shape[0]:
            rel = beneath - gamma[j]
            a = e @ np.column_stack((rel @ normal[j], rel @ binormal[j])).T
            disc = a**2 - ((rel**2).sum(axis=1) - d0**2)[None, :]
            root = np.sqrt(np.maximum(disc, 0))
            lo, hi = a - root, a + root
            for _ in range(50):
                inside = (disc > 0) & (lo < r_rest[:, None] - 1e-12) & (hi > r_rest[:, None] + 1e-12)
                if not inside.any():
                    break
                r_rest = np.maximum(r_rest, np.where(inside, hi, 0.).max(axis=1))
        return xyz(j, r_rest[:, None] * e), r_rest

    def first_switch(j, p0, sign, obs, beneath, to_clear, dpsi=np.radians(0.5), n_grid=180):
        """
        Scanning from p0 in the direction ``sign``: the first angle at which the resting
        cable is clear of obs (``to_clear``), or the last clear angle before it touches
        obs (not ``to_clear``). +-inf if that does not happen within 90 degrees.
        """
        def clear(p):
            x, _ = rest(j, p, beneath)
            return np.linalg.norm(x[:, None, :] - obs[None, :, :], axis=2).min(axis=1) >= dmin
        if obs.shape[0] == 0:
            return p0 if to_clear else sign * np.inf
        if clear(np.array([p0]))[0] == to_clear:
            return p0
        grid = p0 + sign * dpsi * np.arange(1, n_grid + 1)
        hit = clear(grid) == to_clear
        if not hit.any():
            return sign * np.inf
        good = grid[np.argmax(hit)]
        bad = good - sign * dpsi
        for _ in range(12):
            mid = 0.5 * (good + bad)
            if clear(np.array([mid]))[0] == to_clear:
                good = mid
            else:
                bad = mid
        return good if to_clear else bad

    def bounds(J, layer, ref):
        """
        Free interval of angles for point J in ``layer``, around the angle ``ref`` of the
        point before it. If ref is clear, the interval runs from the cable behind to the
        cable ahead. If the cable behind has moved into ref, the cable is pushed forward
        to the first clear angle. The cable passes the lead on the side its centre is
        on: at or just past the lead angle it is pushed past it like past any other
        cable, just before it it stays on the near side.
        """
        j = J % N
        beneath, same = obstacles(J, layer)
        lead = lead_obstacle if near_lead(J) else np.zeros((0, 3))
        if ref >= 2 * np.pi * np.round(ref / (2 * np.pi)) - 1e-9:
            same, lead = np.vstack((same, lead)), np.zeros((0, 3))
        lo = first_switch(j, ref, 1, same, beneath, to_clear=True)
        if lo == ref:
            lo = first_switch(j, ref, -1, same, beneath, to_clear=False)
        elif lo > ref + np.pi / 3:
            lo = np.inf  # the nearest room is beyond the next cable: no room
        if not np.isfinite(lo) and lo > 0:
            return lo, -np.inf  # empty
        start = max(lo, ref)
        hi = first_switch(j, start, 1, same, beneath, to_clear=False)
        if lead.shape[0]:
            hi_lead = first_switch(j, start, 1, lead, beneath, to_clear=False)
            if hi_lead == start:  # start is on the lead: back off below it
                hi_lead = first_switch(j, start, -1, lead, beneath, to_clear=True)
            hi = min(hi, hi_lead)
        return lo, hi

    # first half of the first wind: nothing to touch yet
    n_laid = N // 2
    P[:n_laid] = xyz(np.arange(n_laid), d0 * unit(psi[:n_laid]))
    # then one turn at a time, from half way round to half way round
    # the last wind ends lead_gap before the starting lead
    J_end = (n_winds - 1) * N + int(np.searchsorted(s, L - lead_gap))
    layer = 0
    ref = 0.
    a = n_laid
    while a < J_end:
        b = min(a + N, J_end)
        lo_s, hi_s, home_s, layer_s = [], [], [], []
        for J in range(a, b):
            k = J // N
            home = slot(k)
            if layer > 0 and k < n_inner:
                home = slot(n_inner)  # climbing
            elif J >= k * N + N // 2 and ref > home + 0.5 * (slot(k + 1) - home):
                home = slot(k + 1)  # pushed on toward the next slot
            if layer == 0:
                lo, hi = bounds(J, 0, ref)
                if lo > hi and k == n_inner - 1:
                    # no room left in the full inner layer (squeezed by the previous wind's
                    # step, the first wind or the lead): climb to the outer layer. The
                    # cable starts lifting off a few bend lengths earlier, and from there
                    # on rests on the inner layer instead of being pinned in its slot.
                    layer = 1
                    h = ds[J % N]
                    for i in range(max(0, len(lo_s) - int(np.ceil(3 * bend_length / h))), len(lo_s)):
                        lo_s[i], hi_s[i] = bounds(a + i, 1, home_s[i])
                        layer_s[i] = 1
            layer_J = layer
            if layer == 1:
                home = slot(n_inner) if home < slot(n_inner) else home
                lo, hi = bounds(J, 1, ref)
                if lo > hi:
                    # squeezed in the outer layer (next to the lead): ride up over the neighbours
                    layer_J = 2
                    lo, hi = bounds(J, 2, ref)
            if lo > hi:
                raise RuntimeError(f"no room for the cable at spine point {J % N} of wind {k + 1} (layer {layer_J + 1}): "
                                   f"free angles {np.degrees(lo):.1f} to {np.degrees(hi):.1f}, slot {np.degrees(home):.1f}, "
                                   f"previous {np.degrees(ref):.1f} deg")
            ref = np.clip(home, lo, hi)
            lo_s.append(lo)
            hi_s.append(hi)
            home_s.append(home)
            layer_s.append(layer_J)
        y_lo, y_hi = d0 * np.array(lo_s), d0 * np.array(hi_s)
        h = ds[np.arange(a, b) % N].mean()
        psi[a:b] = smooth_above(np.clip(d0 * np.array(home_s), y_lo, y_hi), y_lo, y_hi, h, bend_length) / d0
        layer_of[a:b] = layer_s
        # distance from the spine: resting on the surface beneath, smoothed over its corners
        rho_rest = np.empty(b - a)
        for i, J in enumerate(range(a, b)):
            beneath, _ = obstacles(J, layer_of[J])
            rho_rest[i] = rest(J % N, psi[J:J+1], beneath)[1][0]
        rho[a:b] = smooth_above(taut_over(rho_rest, h, sag_radius), rho_rest, np.full(b - a, np.inf), h, bend_length)
        P[a:b] = xyz(np.arange(a, b) % N, rho[a:b, None] * unit(psi[a:b]))
        n_laid = b
        a = b
    psi, rho, P, layer_of = psi[:J_end], rho[:J_end], P[:J_end], layer_of[:J_end]
    cable_j = np.arange(J_end) % N
    cable_uv = rho[:, None] * unit(psi)

    # finishing lead: out of the pack and over to the starting lead, then alongside it:
    # parallel to it, lead_gap behind it along the spine
    i_end = cable_j[-1]
    e_out = xyz(i_end, unit(psi[-1])) - gamma[i_end]
    e_lead = (lead_xyz[-1] - lead_xyz[0]) / np.linalg.norm(lead_xyz[-1] - lead_xyz[0])
    tangent = spine.gammadash()[0] / np.linalg.norm(spine.gammadash()[0])
    q = lead_xyz[0] - lead_gap * tangent + lead_exit_radius * e_lead
    m = np.linalg.norm(q - P[-1])
    tau = np.linspace(0, 1, max(2, int(np.ceil(m / 0.002))))[:, None]
    hermite = ((2*tau**3 - 3*tau**2 + 1) * P[-1] + (tau**3 - 2*tau**2 + tau) * m * e_out
               + (-2*tau**3 + 3*tau**2) * q + (tau**3 - tau**2) * m * e_lead)
    lam_out = np.linspace(0, lead_length, max(2, int(np.ceil(lead_length / 0.002))))[1:, None]
    lead_out = np.vstack((hermite, q + lam_out * e_lead))
    rel = lead_out - gamma[i_end]
    lead_out_uv = np.column_stack((rel @ normal[i_end], rel @ binormal[i_end]))

    lead_in = lead_xyz[::-1]  # from the far end to the start of the cable
    offset = lead_in.shape[0] - 1
    path = np.vstack((lead_in[:-1], P, lead_out[1:]))
    # the closing segment from the far end of lead_out to the far end of lead_in is implied
    winds = [(k * N, min((k + 1) * N, J_end)) for k in range(n_winds)]
    sigma = np.where(s > L / 2, s - L, s)
    info = {
        "turns": [(a_ + offset, b_ + offset) for a_, b_ in winds],
        "lead_in": (0, offset),
        "lead_out": (offset + J_end - 1, path.shape[0]),
        "layer": np.array([0] * min(n_winds, n_inner) + [1] * max(n_winds - n_inner, 0)),
        "length": L,
        # local coordinates: spine grid starting at t0, its frame, and the cable on it
        "s": s,
        "sigma": sigma,
        "spine": gamma,
        "normal": normal,
        "binormal": binormal,
        "cable_j": cable_j,
        "cable_uv": cable_uv,
        "cable_psi": psi,
        "cable_rho": rho,
        "cable_layer": layer_of,
        "cable_wind": np.arange(J_end) // N,
        "theta0": theta0,
        "turn_start_sigma": np.zeros(n_winds),
        "lead_in_uv": lead_in_uv,
        "lead_out_uv": lead_out_uv,
        "sigma_lead_out": sigma[i_end],
    }
    return path, info


def continuous_winding_coils(coils, params=None, n_turns=18):
    """
    Continuous winding coils for a list of filamentary coils. Each coil is one cable
    carrying the coil current divided by the number of turns.

    Args:
        coils: list of filamentary Coils.
        params: keyword arguments of ``continuous_winding``, either one dict used for
            every coil or a list with one dict per coil (each coil is wound separately).
        n_turns: number of turns per coil.

    Returns:
        list of Coils, and a list with the ``info`` dict of each coil.
    """
    if params is None:
        params = {}
    if isinstance(params, dict):
        params = [params] * len(coils)
    coils_cw = []
    infos = []
    for c, p in zip(coils, params):
        nodes, info = continuous_winding(c.curve, **p)
        coils_cw.append(Coil(PolylineCurve(nodes), Current(c.current.get_value() / n_turns)))
        infos.append(info)
    return coils_cw, infos
