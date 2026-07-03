"""Fixed-point / X-point geometry and dynamics helpers, used by mk_perturb_manifold.py (and
mirroring the analysis in fieldline_gui_qt.py). Pure functions -- no Qt, no argparse -- so
they can be imported by scripts.

The key entry point is `invariant_directions`, which returns the X-point's invariant ray
("leg") directions in the (R, Z) plane: for a snowflake (degenerate, monodromy ~ I) it returns
the SIX legs (3 invariant lines x 2 rays) via the cubic jet `_snowflake_directions`; for a
regular hyperbolic X-point it returns the FOUR legs (the two eigenvector lines of the
monodromy); for a parabolic point the two rays of the single eigendirection.

`find_fixed_points` is a general multi-start fixed-point finder (user-chosen nfp / stellsym,
optional initial-guess curve, otherwise a circle of radius R), inspired by
find_fixedpoints.find_fixed_points.
"""
import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
from simsopt.geo import CurveRZFourier, CurveXYZFourierSymmetries, CurveLength


def evaluate_at_phi(curve, phi, tol=1e-10):
    """(xyz, accurate) of the curve at toroidal angle phi (units of phi/2pi)."""
    phi += np.ceil(-phi)

    def curve_val(theta):
        return curve.gamma_pure(curve.x, np.array([theta]))[0]

    def theta2phi(theta_in, phi0):
        xyz = curve_val(theta_in)
        angle = np.arctan2(xyz[1], xyz[0]) / (2 * np.pi) - phi0
        angle += np.ceil(-angle)
        return angle

    def fun(theta):
        if theta == 1.0:
            return 1.0 - phi_prime
        return theta2phi(theta, phi0) - phi_prime

    xyz0 = curve_val(0.0)
    phi0 = np.arctan2(xyz0[1], xyz0[0]) / (2 * np.pi)
    phi_prime = phi - phi0
    phi_prime += np.ceil(-phi_prime)
    try:
        result = root_scalar(fun, bracket=[0, 1.0])
        if not result.converged:
            return (None, False)
        xyz = curve_val(result.root)
        R = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)
        c = np.cos(2 * np.pi * phi)
        s = np.sin(2 * np.pi * phi)
        accurate = np.abs(-s * xyz[0] + c * xyz[1]) / R < tol
        return (xyz, accurate)
    except Exception:
        return (None, False)


def _on_axis_field_data(axis, magnetic_field, phi, need_hessian):
    ind = np.array([phi])
    pt = np.zeros((1, 3))
    axis.gamma_impl(pt, ind)
    magnetic_field.set_points(pt)
    xyz = pt.flatten()
    c = np.cos(2 * np.pi * phi)
    s = np.sin(2 * np.pi * phi)
    R = np.sqrt(xyz[0] ** 2 + xyz[1] ** 2)
    B = magnetic_field.B().flatten()
    GradB = magnetic_field.dB_by_dX().reshape((3, 3))
    dBdR = c * GradB[0, :] + s * GradB[1, :]
    dBdZ = GradB[2, :]
    BR = c * B[0] + s * B[1]
    Bp = -s * B[0] + c * B[1]
    BZ = B[2]
    BR_R = c * dBdR[0] + s * dBdR[1]
    BR_Z = c * dBdZ[0] + s * dBdZ[1]
    Bp_R = -s * dBdR[0] + c * dBdR[1]
    Bp_Z = -s * dBdZ[0] + c * dBdZ[1]
    BZ_R = dBdR[2]
    BZ_Z = dBdZ[2]

    def quot(N, N_R, N_Z, D, D_R, D_Z,
             N_RR=None, N_RZ=None, N_ZZ=None, D_RR=None, D_RZ=None, D_ZZ=None):
        h = N / D
        hR = (N_R * D - N * D_R) / D ** 2
        hZ = (N_Z * D - N * D_Z) / D ** 2
        if N_RR is None:
            return (h, hR, hZ, None, None, None)
        hRR = ((N_RR * D - N * D_RR) * D - 2 * (N_R * D - N * D_R) * D_R) / D ** 3
        hZZ = ((N_ZZ * D - N * D_ZZ) * D - 2 * (N_Z * D - N * D_Z) * D_Z) / D ** 3
        w = N_R * D - N * D_R
        w_Z = N_RZ * D + N_R * D_Z - N_Z * D_R - N * D_RZ
        hRZ = (w_Z * D - 2 * w * D_Z) / D ** 3
        return (h, hR, hZ, hRR, hRZ, hZZ)

    if not need_hessian:
        (h, hR, hZ, _, _, _) = quot(BR, BR_R, BR_Z, Bp, Bp_R, Bp_Z)
        (h2, h2R, h2Z, _, _, _) = quot(BZ, BZ_R, BZ_Z, Bp, Bp_R, Bp_Z)
        A = np.array([[h + R * hR, R * hZ],
                      [h2 + R * h2R, R * h2Z]])
        return (R, A, None)
    d2B = magnetic_field.d2B_by_dXdX().reshape((3, 3, 3))
    d2RR = c * c * d2B[0, 0, :] + 2 * c * s * d2B[0, 1, :] + s * s * d2B[1, 1, :]
    d2RZ = c * d2B[0, 2, :] + s * d2B[1, 2, :]
    d2ZZ = d2B[2, 2, :]
    BR_RR = c * d2RR[0] + s * d2RR[1]
    BR_RZ = c * d2RZ[0] + s * d2RZ[1]
    BR_ZZ = c * d2ZZ[0] + s * d2ZZ[1]
    Bp_RR = -s * d2RR[0] + c * d2RR[1]
    Bp_RZ = -s * d2RZ[0] + c * d2RZ[1]
    Bp_ZZ = -s * d2ZZ[0] + c * d2ZZ[1]
    BZ_RR = d2RR[2]
    BZ_RZ = d2RZ[2]
    BZ_ZZ = d2ZZ[2]
    (h, hR, hZ, hRR, hRZ, hZZ) = quot(BR, BR_R, BR_Z, Bp, Bp_R, Bp_Z,
                                      BR_RR, BR_RZ, BR_ZZ, Bp_RR, Bp_RZ, Bp_ZZ)
    (h2, h2R, h2Z, h2RR, h2RZ, h2ZZ) = quot(BZ, BZ_R, BZ_Z, Bp, Bp_R, Bp_Z,
                                            BZ_RR, BZ_RZ, BZ_ZZ, Bp_RR, Bp_RZ, Bp_ZZ)
    A = np.array([[h + R * hR, R * hZ],
                  [h2 + R * h2R, R * h2Z]])
    Hg = np.zeros((2, 2, 2))
    Hg[0, 0, 0] = 2 * hR + R * hRR
    Hg[0, 0, 1] = Hg[0, 1, 0] = hZ + R * hRZ
    Hg[0, 1, 1] = R * hZZ
    Hg[1, 0, 0] = 2 * h2R + R * h2RR
    Hg[1, 0, 1] = Hg[1, 1, 0] = h2Z + R * h2RZ
    Hg[1, 1, 1] = R * h2ZZ
    return (R, A, Hg)


def _rays_from_line_angles(line_angles):
    directions = []
    for t in line_angles:
        directions.append([np.cos(t), np.sin(t)])
        directions.append([np.cos(t + np.pi), np.sin(t + np.pi)])
    return np.array(directions)


def _snowflake_directions(T):
    a0 = T[0, 0, 0]
    a1 = 2 * T[0, 0, 1]
    a2 = T[0, 1, 1]
    b0 = T[1, 0, 0]
    b1 = 2 * T[1, 0, 1]
    b2 = T[1, 1, 1]

    def f(t):
        c = np.cos(t)
        s = np.sin(t)
        return -b0 * c ** 3 + (a0 - b1) * c ** 2 * s + (a1 - b2) * c * s ** 2 + a2 * s ** 3

    Ngrid = 4000
    grid = np.linspace(0, np.pi, Ngrid, endpoint=False)
    step = np.pi / Ngrid
    roots = []
    for t0 in grid:
        t1 = t0 + step
        f0 = f(t0)
        f1 = f(t1)
        if f0 == 0.0:
            roots.append(t0)
            continue
        if f0 * f1 < 0.0:
            lo, hi, flo = t0, t1, f0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                if flo * fm <= 0.0:
                    hi = mid
                else:
                    lo, flo = mid, fm
            roots.append(0.5 * (lo + hi))
    roots = sorted(roots)
    uniq = []
    for r in roots:
        if not any(abs(r - q) < 1e-07 or abs(abs(r - q) - np.pi) < 1e-07 for q in uniq):
            uniq.append(r)
    line_angles = np.array(uniq)
    return (line_angles, _rays_from_line_angles(line_angles))


def _monodromy(curve, magnetic_field, nfp, order=16):
    quadpoints = np.linspace(0, 1 / nfp, 2 * order + 1, endpoint=False)
    XYZ = []
    for phi in quadpoints:
        (xyz, ok) = evaluate_at_phi(curve, phi)
        if not ok:
            return (None, None)
        XYZ.append(xyz)
    cRZ = CurveRZFourier(quadpoints, order, nfp, False)
    cRZ.least_squares_fit(XYZ)
    L = 1 / nfp

    def rhs_M(t, y):
        (_, A, _) = _on_axis_field_data(cRZ, magnetic_field, t, need_hessian=False)
        return (2 * np.pi * (A @ y.reshape((2, 2)))).flatten()

    sol = solve_ivp(rhs_M, [0, L], np.eye(2).flatten(),
                    rtol=1e-12, atol=1e-12, method='RK45')
    return (cRZ, sol.y[:, -1].reshape((2, 2)))


def invariant_directions(xp_curve, magnetic_field, nfp, order=16, tol=1e-07, id_tol=1e-05):
    """Invariant ray ("leg") directions (n_rays, 2) in (R, Z) at the fixed point, or None if
    elliptic / the monodromy fit fails. Snowflake -> 6 legs; hyperbolic -> 4; parabolic -> 2.
    id_tol is the snowflake (|M - I| < id_tol) tolerance (the snowflake monodromy is the
    identity only up to integration error, ~1e-6 in practice)."""
    (cRZ, M) = _monodromy(xp_curve, magnetic_field, nfp, order)
    if M is None:
        return None
    tr = M[0, 0] + M[1, 1]
    if np.abs(M - np.eye(2)).max() < id_tol:
        L = 1 / nfp

        def rhs_full(t, y):
            (_, A, Hg) = _on_axis_field_data(cRZ, magnetic_field, t, need_hessian=True)
            Mt = y[:4].reshape((2, 2))
            dM = 2 * np.pi * (A @ Mt)
            Minv = np.linalg.inv(Mt)
            dT = np.zeros((2, 2, 2))
            for pp in range(2):
                inner = Mt.T @ Hg[pp] @ Mt
                for ii in range(2):
                    dT[ii] += np.pi * Minv[ii, pp] * inner
            return np.concatenate([dM.flatten(), dT.flatten()])

        y0 = np.concatenate([np.eye(2).flatten(), np.zeros(8)])
        sol2 = solve_ivp(rhs_full, [0, L], y0, rtol=1e-12, atol=1e-12, method='RK45')
        T = sol2.y[4:, -1].reshape((2, 2, 2))
        return _snowflake_directions(T)[1]
    if np.abs(tr) > 2 + tol:
        (evals, evecs) = np.linalg.eig(M)
        evecs = np.real(evecs)
        angs = [np.arctan2(evecs[1, j], evecs[0, j]) for j in range(2)]
        return _rays_from_line_angles(np.array(angs))
    if np.abs(np.abs(tr) - 2) <= tol:
        lam = float(np.sign(tr))
        K = M - lam * np.eye(2)
        a = K[0, 0]
        b = K[0, 1]
        cc = K[1, 0]
        d = K[1, 1]
        v = np.array([b, -a]) if abs(a) + abs(b) >= abs(cc) + abs(d) else np.array([d, -cc])
        n = np.linalg.norm(v)
        v = v / n if n > 0 else np.array([1, 0])
        return _rays_from_line_angles(np.array([np.arctan2(v[1], v[0])]))


def _integrate_T(cRZ, magnetic_field, nfp):
    """Second-order return-map tensor T (2,2,2) along the fixed-point field line over one
    field period (the quadratic jet of the return map), same integration as the snowflake
    branch of invariant_directions."""
    L = 1 / nfp

    def rhs_full(t, y):
        (_, A, Hg) = _on_axis_field_data(cRZ, magnetic_field, t, need_hessian=True)
        Mt = y[:4].reshape((2, 2))
        dM = 2 * np.pi * (A @ Mt)
        Minv = np.linalg.inv(Mt)
        dT = np.zeros((2, 2, 2))
        for pp in range(2):
            inner = Mt.T @ Hg[pp] @ Mt
            for ii in range(2):
                dT[ii] += np.pi * Minv[ii, pp] * inner
        return np.concatenate([dM.flatten(), dT.flatten()])

    y0 = np.concatenate([np.eye(2).flatten(), np.zeros(8)])
    sol = solve_ivp(rhs_full, [0, L], y0, rtol=1e-12, atol=1e-12, method='RK45')
    return sol.y[4:, -1].reshape((2, 2, 2))


def leg_info(xp_curve, magnetic_field, nfp, order=16, tol=1e-07, id_tol=1e-05):
    """Per-leg data for seeding manifold fieldlines (matches mk_manifolds.py). Returns
    (type, rays, speeds, stable):
      rays   : (n, 2) unit (dR, dZ) leg directions (6 snowflake / 4 hyperbolic / 2 parabolic),
      speeds : per-ray scalar -- the eigenvalue (hyperbolic) or the quadratic radial
               coefficient c = Q(v).v (snowflake / parabolic) that sets the fundamental
               interval,
      stable : per-ray bool (the manifold flows INTO the X-point along a stable leg, so it is
               grown by integrating BACKWARD in time).
    Returns (type, None, None, None) for an elliptic point / failed fit."""
    (cRZ, M) = _monodromy(xp_curve, magnetic_field, nfp, order)
    if M is None:
        return ('unknown', None, None, None)
    tr = M[0, 0] + M[1, 1]

    def _cs(T, rays):
        out = []
        for v in rays:
            Qv = np.array([sum(T[i, j, k] * v[j] * v[k]
                               for j in range(2) for k in range(2)) for i in range(2)])
            out.append(float(np.dot(Qv, v)))
        return np.array(out)

    if np.abs(M - np.eye(2)).max() < id_tol:
        T = _integrate_T(cRZ, magnetic_field, nfp)
        (_, rays) = _snowflake_directions(T)
        c = _cs(T, rays)
        return ('snowflake', rays, c, c < 0)
    if np.abs(tr) > 2 + tol:
        (evals, evecs) = np.linalg.eig(M)
        evals = np.real(evals)
        evecs = np.real(evecs)
        iu = int(np.argmax(np.abs(evals)))
        is_ = 1 - iu
        vu = evecs[:, iu] / np.linalg.norm(evecs[:, iu])
        vs = evecs[:, is_] / np.linalg.norm(evecs[:, is_])
        rays = _rays_from_line_angles(np.array([np.arctan2(vu[1], vu[0]),
                                                np.arctan2(vs[1], vs[0])]))
        speeds = np.array([evals[iu], evals[iu], evals[is_], evals[is_]])
        stable = np.array([False, False, True, True])
        return ('hyperbolic', rays, speeds, stable)
    if np.abs(np.abs(tr) - 2) <= tol:
        lam = float(np.sign(tr))
        K = M - lam * np.eye(2)
        (a, b, cc, d) = (K[0, 0], K[0, 1], K[1, 0], K[1, 1])
        v = np.array([b, -a]) if abs(a) + abs(b) >= abs(cc) + abs(d) else np.array([d, -cc])
        n = np.linalg.norm(v)
        v = v / n if n > 0 else np.array([1, 0])
        rays = _rays_from_line_angles(np.array([np.arctan2(v[1], v[0])]))
        c = _cs(_integrate_T(cRZ, magnetic_field, nfp), rays)
        return ('parabolic', rays, c, c < 0)
    return ('elliptic', None, None, None)


def classify(xp_curve, magnetic_field, nfp, order=16, tol=1e-07, id_tol=1e-05):
    """('snowflake'|'hyperbolic'|'parabolic'|'elliptic', trace(M)) for the fixed point."""
    (_, M) = _monodromy(xp_curve, magnetic_field, nfp, order)
    if M is None:
        return ('unknown', np.nan)
    tr = float(M[0, 0] + M[1, 1])
    if np.abs(M - np.eye(2)).max() < id_tol:
        return ('snowflake', tr)
    if np.abs(tr) > 2 + tol:
        return ('hyperbolic', tr)
    if np.abs(np.abs(tr) - 2) <= tol:
        return ('parabolic', tr)
    return ('elliptic', tr)


def find_fixed_points(field, nfp, stellsym, *, guess=None, R=0.3, Z=0.0,
                      order=16, ntor=1, n_seeds=8, search_radius=2e-2, dedupe=1e-3,
                      length=None, newton=None, classify_order=16):
    """All UNIQUE converged fixed points of `field`, found by a multi-start Newton from a
    Vogel/sunflower disk of (R, Z) shifts of an initial guess. Inspired by
    find_fixedpoints.find_fixed_points, but generalized: the fixed-point curves are solved as
    CurveXYZFourierSymmetries with the USER-SPECIFIED `nfp` and `stellsym`.

    Parameters
    ----------
    field : simsopt MagneticField
        The field to search (already carrying any auxiliary coils).
    nfp : int
        Field periodicity of the fixed-point curves to solve for.
    stellsym : bool
        Stellarator symmetry of the fixed-point curves.
    guess : simsopt Curve or None, optional
        Initial-guess curve giving the seed shape. If None, a planar circle of radius `R`
        at height `Z`. A CurveXYZFourierSymmetries guess is resampled onto the solve grid;
        any other Curve must already have 2*order+1 quadpoints.
    R, Z : float or array-like, optional
        Radius / height [m] of the default circular guess (ignored when `guess` is given).
        Each may be a scalar OR equal-length lists/arrays: in the list case every
        (R[k], Z[k]) is a SEPARATE circular guess centre and the n_seeds Vogel-disk
        multistart runs around EACH, with all unique fixed points (deduped globally) returned.
    order, ntor : int, optional
        Fourier resolution / toroidal winding of the solve curve.
    n_seeds : int, optional
        Number of Vogel-disk multistart seeds (n_seeds=1 solves only the guess itself).
    search_radius : float, optional
        Radius of the (R, Z) seed disk [m].
    dedupe : float, optional
        Merge tolerance in (R, Z) [m].
    length : float or None, optional
        Target field-line length; defaults to CurveLength of the guess curve.
    newton : dict or None, optional
        PeriodicFieldLine options (defaults newton_tol=1e-10, newton_maxiter=20).
        1e-10 (rather than 1e-12) is used because the field-line residual floors near
        1e-10 on near-degenerate / aux-coil fields, where a tighter tol spuriously
        reports non-convergence.
    classify_order : int, optional
        Order passed to classify().

    Returns
    -------
    list of (curve, (Rx, Zx), trace, type)
        One entry per unique fixed point, sorted by (R, Z) distance from the guess centre.
        `curve` is the converged CurveXYZFourierSymmetries; `type` is one of
        'snowflake' / 'hyperbolic' / 'parabolic' / 'elliptic'.
    """
    # Local import avoids a utils <-> driver import cycle and keeps this module light.
    from star_lite_design.utils.periodicfieldline import PeriodicFieldLine

    # quadpoints span ONE field period [0, 1/nfp), matching CurveXYZFourierSymmetries'
    # convention (the full curve is reconstructed from one period by symmetry).
    qp = np.linspace(0.0, 1.0 / nfp, 2 * order + 1, endpoint=False)

    # Seed CENTRES. With `guess`, a single centre from that curve (R/Z ignored). Otherwise R
    # and Z may EACH be a scalar OR equal-length lists/arrays: every (R[k], Z[k]) is a separate
    # circular guess centre, and the n_seeds Vogel-disk multistart runs around EACH; all unique
    # fixed points across every centre are returned (deduped globally).
    seed_centres = []                              # one seed-point array per centre
    if guess is None:
        R_arr = np.atleast_1d(np.asarray(R, dtype=float))
        Z_arr = np.atleast_1d(np.asarray(Z, dtype=float))
        if R_arr.size != Z_arr.size:
            raise ValueError(f"R and Z must be the same length; got {R_arr.size} and {Z_arr.size}")
        for Rc, Zc in zip(R_arr, Z_arr):
            seed_centres.append(np.column_stack([Rc * np.cos(2 * np.pi * qp),
                                                 Rc * np.sin(2 * np.pi * qp),
                                                 np.full_like(qp, Zc)]))
    elif isinstance(guess, CurveXYZFourierSymmetries):
        rs = CurveXYZFourierSymmetries(qp, guess.order, guess.nfp, guess.stellsym,
                                       ntor=guess.ntor)
        rs.x = guess.x
        seed_centres.append(np.asarray(rs.gamma()))
    else:
        seed_pts = np.asarray(guess.gamma())
        if seed_pts.shape[0] != qp.size:
            raise ValueError(
                f"guess curve has {seed_pts.shape[0]} quadpoints; need 2*order+1 = {qp.size} "
                "(or pass a CurveXYZFourierSymmetries, which is resampled automatically).")
        seed_centres.append(seed_pts)

    if newton is None:
        newton = {"newton_tol": 1e-10, "newton_maxiter": 20, "verbose": False}
    i = np.arange(n_seeds)
    # n_seeds == 1 -> solve each centre itself (no shift), as the docstring promises (handy
    # when a LIST of (R, Z) centres is supplied). n_seeds > 1 -> a golden-angle Vogel disk of
    # (R, Z) shifts of radius up to `search_radius` around each centre.
    gr = np.zeros(1) if n_seeds == 1 else search_radius * np.sqrt((i + 0.5) / n_seeds)
    gth = i * (np.pi * (3.0 - np.sqrt(5.0)))       # golden-angle Vogel disk

    found = []                                     # (curve, (Rx, Zx), trace, type)
    R0 = Z0 = None                                 # final-sort reference = the FIRST centre
    for seed_pts in seed_centres:
        base = CurveXYZFourierSymmetries(qp, order, nfp, stellsym, ntor=ntor)
        base.least_squares_fit(seed_pts)
        g = np.asarray(base.gamma())
        Rg = np.hypot(g[:, 0], g[:, 1])
        phig = np.arctan2(g[:, 1], g[:, 0])
        Zg = g[:, 2]
        if R0 is None:
            R0, Z0 = float(Rg[0]), float(Zg[0])
        clen = float(CurveLength(base).J()) if length is None else length
        for dR, dZ in zip(gr * np.cos(gth), gr * np.sin(gth)):
            pts = np.column_stack([(Rg + dR) * np.cos(phig),
                                   (Rg + dR) * np.sin(phig), Zg + dZ])
            guess_curve = CurveXYZFourierSymmetries(qp, order, nfp, stellsym, ntor=ntor)
            try:
                guess_curve.least_squares_fit(pts)
                res = PeriodicFieldLine(field, guess_curve, options=dict(newton)).run_code(clen)
            except Exception:
                continue
            if res is None or not res.get("success", False):
                continue
            x0 = np.asarray(guess_curve.gamma())[0]
            Rx, Zx = float(np.hypot(x0[0], x0[1])), float(x0[2])
            if any(abs(Rx - q[1][0]) < dedupe and abs(Zx - q[1][1]) < dedupe for q in found):
                continue
            typ, tr = classify(guess_curve, field, nfp, order=classify_order)
            found.append((guess_curve, (Rx, Zx), tr, typ))
    found.sort(key=lambda q: (q[1][0] - R0) ** 2 + (q[1][1] - Z0) ** 2)
    return found
