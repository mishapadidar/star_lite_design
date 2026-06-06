#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import argparse
from pathlib import Path
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
from simsopt._core import load
from simsopt.geo import SurfaceXYZTensorFourier, CurveRZFourier
from simsopt.field import (SurfaceClassifier, compute_fieldlines,
                           LevelsetStoppingCriterion, ToroidalTransitStoppingCriterion)
from simsopt.util import proc0_print, comm_world


def evaluate_at_phi(curve, phi, tol=1e-10):
    # map phi to [0, 1)
    phi += np.ceil(-phi)

    def curve_val(theta):
        return curve.gamma_pure(curve.x, np.array([theta]))[0]

    def theta2phi(theta_in, phi0):
        xyz = curve_val(theta_in)
        angle = np.arctan2(xyz[1], xyz[0]) / (2 * np.pi) - phi0
        angle += np.ceil(-angle)
        return angle

    def fun(theta):
        # if theta is exactly 1, then override the rest. hacky.
        if theta == 1.:
            return 1. - phi_prime
        angle = theta2phi(theta, phi0)
        return angle - phi_prime

    xyz0 = curve_val(0.)
    phi0 = np.arctan2(xyz0[1], xyz0[0]) / (2 * np.pi)
    phi_prime = phi - phi0
    phi_prime += np.ceil(-phi_prime)

    result = root_scalar(fun, bracket=[0, 1.])
    conv = result.converged

    # check that the result is accurate by dotting with something orthogonal to the angle
    xyz = curve_val(result.root)
    R = np.sqrt(xyz[0]**2 + xyz[1]**2)
    c = np.cos(2 * np.pi * phi)
    s = np.sin(2 * np.pi * phi)
    accurate = np.abs(-s * xyz[0] + c * xyz[1]) / R < tol
    success = conv and accurate

    return xyz, success


def _on_axis_field_data(axis, magnetic_field, phi, need_hessian):
    """
    Evaluate, at the axis point with curve parameter ``phi``, the quantities needed
    to linearize (and, optionally, to expand to second order) the field-line return
    map in the meridional ``(R, Z)`` plane.
    """
    ind = np.array([phi])
    pt = np.zeros((1, 3))
    axis.gamma_impl(pt, ind)
    magnetic_field.set_points(pt)
    xyz = pt.flatten()

    c = np.cos(2 * np.pi * phi)
    s = np.sin(2 * np.pi * phi)
    R = np.sqrt(xyz[0]**2 + xyz[1]**2)

    B = magnetic_field.B().flatten()
    GradB = magnetic_field.dB_by_dX().reshape((3, 3))

    # First derivatives of each Cartesian component w.r.t. (R, Z)
    dBdR = c * GradB[0, :] + s * GradB[1, :]   # dB_q/dR for q in {x, y, z}
    dBdZ = GradB[2, :]                          # dB_q/dZ

    # Cylindrical components and their first (R, Z) derivatives
    BR = c * B[0] + s * B[1]
    Bp = -s * B[0] + c * B[1]
    BZ = B[2]
    BR_R = c * dBdR[0] + s * dBdR[1]; BR_Z = c * dBdZ[0] + s * dBdZ[1]
    Bp_R = -s * dBdR[0] + c * dBdR[1]; Bp_Z = -s * dBdZ[0] + c * dBdZ[1]
    BZ_R = dBdR[2];                    BZ_Z = dBdZ[2]

    # Quotient h = N/D and its (R, Z) derivatives, up to the order we need.
    def quot(N, N_R, N_Z, D, D_R, D_Z,
             N_RR=None, N_RZ=None, N_ZZ=None, D_RR=None, D_RZ=None, D_ZZ=None):
        h = N / D
        hR = (N_R * D - N * D_R) / D**2
        hZ = (N_Z * D - N * D_Z) / D**2
        if N_RR is None:
            return h, hR, hZ, None, None, None
        hRR = ((N_RR * D - N * D_RR) * D - 2 * (N_R * D - N * D_R) * D_R) / D**3
        hZZ = ((N_ZZ * D - N * D_ZZ) * D - 2 * (N_Z * D - N * D_Z) * D_Z) / D**3
        w = N_R * D - N * D_R
        w_Z = N_RZ * D + N_R * D_Z - N_Z * D_R - N * D_RZ
        hRZ = (w_Z * D - 2 * w * D_Z) / D**3
        return h, hR, hZ, hRR, hRZ, hZZ

    if not need_hessian:
        h, hR, hZ, _, _, _ = quot(BR, BR_R, BR_Z, Bp, Bp_R, Bp_Z)
        h2, h2R, h2Z, _, _, _ = quot(BZ, BZ_R, BZ_Z, Bp, Bp_R, Bp_Z)
        A = np.array([[h + R * hR, R * hZ],
                      [h2 + R * h2R, R * h2Z]])
        return R, A, None

    # Second derivatives of each Cartesian component w.r.t. (R, Z)
    d2B = magnetic_field.d2B_by_dXdX().reshape((3, 3, 3))
    d2RR = c * c * d2B[0, 0, :] + 2 * c * s * d2B[0, 1, :] + s * s * d2B[1, 1, :]
    d2RZ = c * d2B[0, 2, :] + s * d2B[1, 2, :]
    d2ZZ = d2B[2, 2, :]

    BR_RR = c * d2RR[0] + s * d2RR[1]; BR_RZ = c * d2RZ[0] + s * d2RZ[1]; BR_ZZ = c * d2ZZ[0] + s * d2ZZ[1]
    Bp_RR = -s * d2RR[0] + c * d2RR[1]; Bp_RZ = -s * d2RZ[0] + c * d2RZ[1]; Bp_ZZ = -s * d2ZZ[0] + c * d2ZZ[1]
    BZ_RR = d2RR[2];                    BZ_RZ = d2RZ[2];                    BZ_ZZ = d2ZZ[2]

    h, hR, hZ, hRR, hRZ, hZZ = quot(BR, BR_R, BR_Z, Bp, Bp_R, Bp_Z,
                                    BR_RR, BR_RZ, BR_ZZ, Bp_RR, Bp_RZ, Bp_ZZ)
    h2, h2R, h2Z, h2RR, h2RZ, h2ZZ = quot(BZ, BZ_R, BZ_Z, Bp, Bp_R, Bp_Z,
                                          BZ_RR, BZ_RZ, BZ_ZZ, Bp_RR, Bp_RZ, Bp_ZZ)

    A = np.array([[h + R * hR, R * hZ],
                  [h2 + R * h2R, R * h2Z]])
    Hg = np.zeros((2, 2, 2))
    # g_p = R * (B_p^cyl / B_phi), so the explicit factor R produces these terms.
    Hg[0, 0, 0] = 2 * hR + R * hRR
    Hg[0, 0, 1] = Hg[0, 1, 0] = hZ + R * hRZ
    Hg[0, 1, 1] = R * hZZ
    Hg[1, 0, 0] = 2 * h2R + R * h2RR
    Hg[1, 0, 1] = Hg[1, 1, 0] = h2Z + R * h2RZ
    Hg[1, 1, 1] = R * h2ZZ
    return R, A, Hg


def _snowflake_directions(T):
    """
    Given the symmetric vector-valued quadratic form ``Q_i(v) = sum_{j,k} T[i,j,k] v_j v_k``
    that is the leading (second order) part of the return map ``v -> v + Q(v)`` about a
    fixed point whose linearization is the identity, find the invariant ray directions.
    """
    a0 = T[0, 0, 0]; a1 = 2 * T[0, 0, 1]; a2 = T[0, 1, 1]
    b0 = T[1, 0, 0]; b1 = 2 * T[1, 0, 1]; b2 = T[1, 1, 1]

    def f(t):
        c = np.cos(t); s = np.sin(t)
        return -b0 * c**3 + (a0 - b1) * c**2 * s + (a1 - b2) * c * s**2 + a2 * s**3

    # Robustly bracket-and-bisect the cubic over [0, pi) (avoids the tan(t) singularity).
    N = 4000
    grid = np.linspace(0.0, np.pi, N, endpoint=False)
    step = np.pi / N
    roots = []
    for t0 in grid:
        t1 = t0 + step
        f0 = f(t0); f1 = f(t1)
        if f0 == 0.0:
            roots.append(t0)
        elif f0 * f1 < 0.0:
            lo, hi, flo = t0, t1, f0
            for _ in range(80):
                mid = 0.5 * (lo + hi); fm = f(mid)
                if flo * fm <= 0.0:
                    hi = mid
                else:
                    lo, flo = mid, fm
            roots.append(0.5 * (lo + hi))
    # Deduplicate (lines repeating mod pi or coincident near the boundary)
    roots = sorted(roots)
    uniq = []
    for r in roots:
        if not any(abs(r - q) < 1e-7 or abs(abs(r - q) - np.pi) < 1e-7 for q in uniq):
            uniq.append(r)
    line_angles = np.array(uniq)
    return line_angles, _rays_from_line_angles(np.array(uniq))


def _rays_from_line_angles(line_angles):
    """Two opposite unit (dR, dZ) rays per invariant-line angle."""
    directions = []
    for t in line_angles:
        directions.append([np.cos(t), np.sin(t)])
        directions.append([np.cos(t + np.pi), np.sin(t + np.pi)])
    return np.array(directions)


def _snowflake_cubic_discriminant(T):
    """Discriminant of the homogeneous cubic f(cos t, sin t) whose real roots are
    the snowflake's invariant lines (see :func:`_snowflake_directions`).

    With the binary cubic  p c^3 + q c^2 s + r c s^2 + w s^3  (c=cos t, s=sin t),
        disc = q^2 r^2 - 4 p r^3 - 4 q^3 w + 18 p q r w - 27 p^2 w^2.

    The sign tells you the leg count directly:
        disc > 0 -> 3 distinct real lines -> 6 legs (genuine monkey-saddle snowflake)
        disc < 0 -> 1 real line           -> 2 legs
        disc ~ 0 -> a real double root: right at the 2<->6 leg transition.
    """
    a0 = T[0, 0, 0]; a1 = 2 * T[0, 0, 1]; a2 = T[0, 1, 1]
    b0 = T[1, 0, 0]; b1 = 2 * T[1, 0, 1]; b2 = T[1, 1, 1]
    p = -b0; q = a0 - b1; r = a1 - b2; w = a2
    return float(q**2 * r**2 - 4 * p * r**3 - 4 * q**3 * w
                 + 18 * p * q * r * w - 27 * p**2 * w**2)


def _integrate_monodromy(curve, field, nfp):
    """Integrate the (2,2) tangent (monodromy) map over one field period."""
    def rhs(t, y):
        _, A, _ = _on_axis_field_data(curve, field, t, need_hessian=False)
        return (2 * np.pi * (A @ y.reshape((2, 2)))).flatten()
    sol = solve_ivp(rhs, [0.0, 1.0 / nfp], np.eye(2).flatten(),
                    rtol=1e-12, atol=1e-12, method='RK45')
    return sol.y[:, -1].reshape((2, 2))


def _integrate_T(curve, field, nfp):
    """Second-order return-map tensor T(2,2,2) over one field period."""
    def rhs_full(t, y):
        _, A, Hg = _on_axis_field_data(curve, field, t, need_hessian=True)
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
    sol2 = solve_ivp(rhs_full, [0.0, 1.0 / nfp], y0, rtol=1e-12, atol=1e-12, method='RK45')
    return sol2.y[4:, -1].reshape((2, 2, 2))


def compute(axis, magnetic_field, tol=1e-5):
    """
    Classify the on-axis tangent (monodromy) map of a magnetic axis over one field period
    and return the corresponding invariant data. Copied from
    simsopt.field.magnetic_axis_helpers so this script has no external dependency on it.
    """
    assert type(axis) is CurveRZFourier
    nfp = axis.nfp

    M = _integrate_monodromy(axis, magnetic_field, nfp)
    tr = M[0, 0] + M[1, 1]

    # Standardized classification output; only 'type' and 'legs' are consumed
    # downstream — the rest aid reuse/inspection.
    res = {'type': None, 'M': M, 'eigenvalues': np.linalg.eig(M)[0],
           'iota': None, 'line_angles': None, 'directions': None, 'c': None, 'T': None,
           'sigma': None, 'D': None, 'discriminant': None}
    
    # ---- Classify. ----
    if np.abs(M - np.eye(2)).max() < tol:
        case = 'snowflake'
    elif np.abs(M + np.eye(2)).max() < tol:
        raise ValueError(
            "On-axis tangent map is -Identity; this degenerate case is not supported "
            "(only the +Identity 'snowflake' case is handled).")
    elif np.abs(tr) < 2.0 - tol:
        case = 'elliptic'
    elif np.abs(tr) > 2.0 + tol:
        case = 'hyperbolic'
    else:
        case = 'parabolic'
    res['type'] = case

    if case == 'elliptic':
        evals = res['eigenvalues']
        res['iota'] = np.arctan2(np.imag(evals[0]), np.real(evals[0])) * nfp / (2 * np.pi)
    elif case == 'hyperbolic':
        evals, evecs = np.linalg.eig(M)
        evals = np.real(evals)
        evecs = np.real(evecs)
        iu = int(np.argmax(np.abs(evals)))   # |lambda| > 1 : unstable manifold
        is_ = 1 - iu                          # |lambda| < 1 : stable manifold
        vu = evecs[:, iu] / np.linalg.norm(evecs[:, iu])
        vs = evecs[:, is_] / np.linalg.norm(evecs[:, is_])
        # eigenvalues ordered [unstable, stable] to match the line ordering
        res['eigenvalues'] = np.array([evals[iu], evals[is_]])
        res['line_angles'] = np.array([np.arctan2(vu[1], vu[0]),
                                       np.arctan2(vs[1], vs[0])])
        res['directions'] = _rays_from_line_angles(res['line_angles'])

    elif case == 'parabolic':
        lam = float(np.sign(tr))            # repeated eigenvalue +-1
        res['eigenvalues'] = np.array([lam, lam])
        K = M - lam * np.eye(2)             # rank 1; its kernel is the invariant direction
        a, b = K[0, 0], K[0, 1]
        c, d = K[1, 0], K[1, 1]
        if abs(a) + abs(b) >= abs(c) + abs(d):
            v = np.array([b, -a])
        else:
            v = np.array([d, -c])
        n = np.linalg.norm(v)
        v = v / n if n > 0 else np.array([1.0, 0.0])
        res['line_angles'] = np.array([np.arctan2(v[1], v[0])])
        res['directions'] = _rays_from_line_angles(res['line_angles'])
        res['T'] = _integrate_T(axis, magnetic_field, nfp)
        # Jordan/cusp data. In the frame x = xi*v + eta*u (u = unit normal to v)
        # the jet reads xi' = xi + sigma*eta + ..., eta' = eta + D*xi^2 + ...
        # (det M = 1 => K^2 = 0, so Im K is along v and sigma = v.(K u) is the
        # Jordan shear). The leading local model is the interpolating Hamiltonian
        #   H(xi, eta) = (sigma/2) eta^2 - (D/3) xi^3,
        # whose zero level set is the separatrix: the semicubical (cusp) curve
        # eta = +-sqrt(2D/(3 sigma)) xi^(3/2) on the side sign(xi) = sign(sigma*D).
        u_n = np.array([-v[1], v[0]])
        res['sigma'] = float(v @ (K @ u_n))
        Qvv = np.array([sum(res['T'][i, j, k] * v[j] * v[k]
                            for j in range(2) for k in range(2)) for i in range(2)])
        res['D'] = float(u_n @ Qvv)

    else:  # snowflake (M = +I): integrate the second-order return map.
        T = _integrate_T(axis, magnetic_field, nfp)
        line_angles, directions = _snowflake_directions(T)
        res['line_angles'] = line_angles
        res['directions'] = directions
        res['T'] = T
        # Discriminant of the invariant-line cubic: >0 => 3 lines/6 legs (true
        # monkey-saddle snowflake), <0 => 1 line/2 legs, ~0 => 2<->6 transition.
        res['discriminant'] = _snowflake_cubic_discriminant(T)

    # Second-order radial coefficient c = Q(v).v per ray (snowflake & parabolic).
    if res['T'] is not None:
        cs = []
        for v in res['directions']:
            Qv = np.array([sum(res['T'][i, j, k] * v[j] * v[k]
                               for j in range(2) for k in range(2)) for i in range(2)])
            cs.append(float(np.dot(Qv, v)))
        res['c'] = np.array(cs)

    return res


def print_fixed_point_info(name, res):
    """Print the fixed-point classification to the console, using compute()'s
    output (res) only."""
    M = res['M']
    tr = float(M[0, 0] + M[1, 1])
    det = float(np.linalg.det(M))
    print(f"==================== {name} ====================")
    print(f"  monodromy M = [[{M[0,0]:+.6f}, {M[0,1]:+.6f}], [{M[1,0]:+.6f}, {M[1,1]:+.6f}]]")
    print(f"  trace(M) = {tr:+.8f}    det(M) = {det:+.8f}")
    print(f"  eigenvalues = {np.array2string(np.asarray(res['eigenvalues']), precision=6)}")

    if res['type'] == 'snowflake':
        line_angles = res['line_angles']
        print(f"  type: SNOWFLAKE (M = I)")
        print(f"  invariant lines: {len(line_angles)}   rays (legs): {len(res['directions'])}")
        disc = res.get('discriminant')
        if disc is not None:
            verdict = ('3 real lines -> 6 legs (monkey saddle)' if disc > 0 else
                       '1 real line -> 2 legs' if disc < 0 else 'degenerate (double root)')
            print(f"  invariant-line cubic discriminant = {disc:+.4e}   ({verdict})")
        for li, t in enumerate(line_angles):
            c = res['c'][2 * li]          # c = Q(v).v for the +ray of this line
            print(f"    line {np.degrees(t):+7.2f} deg   c = {c:+.4e}   "
                  f"({'unstable' if c > 0 else 'stable'})")
    elif res['type'] == 'hyperbolic':
        lam_u, lam_s = res['eigenvalues']            # ordered [unstable, stable]
        au, as_ = np.degrees(res['line_angles'])     # ordered [unstable, stable]
        print(f"  type: HYPERBOLIC")
        print(f"  invariant lines: 2   rays (legs): 4")
        print(f"    unstable: eigenvalue {lam_u:+.6f}  angle {au:+.2f} deg")
        print(f"    stable:   eigenvalue {lam_s:+.6f}  angle {as_:+.2f} deg")
    elif res['type'] == 'parabolic':
        lam = res['eigenvalues'][0]
        ang = np.degrees(res['line_angles'][0])
        print(f"  type: PARABOLIC (repeated eigenvalue {lam:+.0f})")
        print(f"  invariant lines: 1   rays (legs): 2   angle {ang:+.2f} deg")
        sig, D = res['sigma'], res['D']
        print(f"  Jordan shear sigma = {sig:+.4e}   cusp coefficient D = {D:+.4e}")
        if sig != 0.0 and D / sig > 0:
            print(f"  separatrix eta = +-{np.sqrt(2 * D / (3 * sig)):.4e} * xi^(3/2)"
                  f"   (cusp on the xi > 0 side)")
        elif sig != 0.0:
            print(f"  separatrix eta = +-{np.sqrt(-2 * D / (3 * sig)):.4e} * (-xi)^(3/2)"
                  f"   (cusp on the xi < 0 side)")
    else:  # elliptic
        print(f"  type: ELLIPTIC")
        print(f"  rotational transform iota = {res['iota']:+.6f}")


# =========================
# Load data
# =========================

_parser = argparse.ArgumentParser(description="snowflake / x-point manifold tracer")
_parser.add_argument("data_file", help="path to the singular/design json")
_parser.add_argument("--parabolic-tol", type=float, default=0.1,
                     help="parabolic classification tolerance, RELATIVE in percent: a "
                          "fixed point is parabolic if ||tr M| - 2| / 2 <= tol/100 "
                          "(default 0.1 = 0.1%%)")
_args = _parser.parse_args()
# Convert the relative-percent tolerance to the absolute window on |tr M| - 2.
CLASSIFY_TOL = 2.0 * _args.parabolic_tol / 100.0

p = Path(_args.data_file)
dat = load(p)

[boozer_surfaces, iota_Gs, axes, xpoints, sdf] = dat
xpoint=xpoints[0]
boozer_surface = boozer_surfaces[0]

# For SN (non-stellsym surface) up-down symmetry is lost, so the stellsym
# half-period phi/2pi in [0, 0.25] is no longer representative: trace/plot over
# the full nfp=2 field period [0, 0.5]. DN keeps the half-period [0, 0.25].
PHI_MAX = 0.25 if boozer_surface.surface.stellsym else 0.5
NPHI = 9
PHIS = np.linspace(0, PHI_MAX, NPHI)   # panel grid, phi/2pi
PHIS[0] = PHIS[0] + 1e-10
PHIS_RAD = PHIS * 2 * np.pi            # radians, for compute_fieldlines
# SN is non-stellsym, so the "bottom" X-point is NOT the Z-mirror of the top:
# only trace/record the top X-point (the mirror-based bottom would be wrong).
is_sn = not boozer_surface.surface.stellsym

# Surface classifier used as the field-line stopping criterion: a torus fit
# around the magnetic axis (matches array_initial/mk_manifolds_DN.py). Field
# lines are killed once they leave this level set, so traces can't wander off to
# infinity. Built on its own quadpoint grid with the surface's symmetry.
_sc_surface = SurfaceXYZTensorFourier(
    mpol=boozer_surface.surface.mpol, ntor=boozer_surface.surface.ntor,
    quadpoints_phi=np.linspace(0, 1, 32, endpoint=False),
    quadpoints_theta=np.linspace(0, 1, 32, endpoint=False),
    nfp=boozer_surface.surface.nfp, stellsym=boozer_surface.surface.stellsym)
_sc_surface.fit_to_curve(axes[0].curve, 0.45, flip_theta=False)
surface_classifier = SurfaceClassifier(_sc_surface, h=0.05, p=2)

# convert to RZFourier
order=16
quadpoints=np.linspace(0, 0.5, 2*order+1, endpoint=False)
nfp=2
stellsym=False

XYZ = []
for phi in quadpoints:
    xyz, success = evaluate_at_phi(xpoint.curve, phi)
    assert success
    XYZ.append(xyz)
xpoint_RZ = CurveRZFourier(quadpoints, order, nfp, stellsym)
xpoint_RZ.least_squares_fit(XYZ)

res = compute(xpoint_RZ, boozer_surface.biotsavart, tol=CLASSIFY_TOL)

OUT_DIR = str(p.parent) + "/"   # write next to singular.json so plot + sync find the files
os.makedirs(OUT_DIR, exist_ok=True)

g0 = xpoint.curve.gamma()[0]
nfp = xpoint.curve.nfp


if comm_world is None or comm_world.rank == 0:
    print_fixed_point_info("X-point", res)

    np.savetxt(OUT_DIR + 'xpoint.txt', g0[None, :])
    # Single source of truth for the panel phi grid (plot_manifolds.py reads this
    # so DN half-period [0,0.25] vs SN full-period [0,0.5] always agree).
    np.savetxt(OUT_DIR + 'phis.txt', PHIS)
    # X-point classification, so plot_manifolds.py can pick the zoom window
    # (hyperbolic uses a wider zoom).
    with open(OUT_DIR + 'xpoint_type.txt', 'w') as fh:
        fh.write(f"{res['type']}\n")
    with open(OUT_DIR + 'legs.txt', 'w') as fh:
        fh.write('# k, theta, vR, vZ, kind, speed\n')
        for k, v2 in enumerate([] if res['directions'] is None else res['directions']):
            th = np.arctan2(v2[1], v2[0])
            if res['type'] == 'hyperbolic':
                speed = res['eigenvalues'][0 if k < 2 else 1]   # rays [u+, u-, s+, s-]
                stable = k >= 2
            else:                                               # snowflake / parabolic
                speed = res['c'][k]
                stable = speed < 0
            kind = 'stable' if stable else 'unstable'
            fh.write(f"{k}, {th:.6f}, {v2[0]:.6f}, {v2[1]:.6f}, {kind}, {speed:.6f}\n")
    if res['type'] == 'parabolic':
        # Leading local model at the parabolic X-point, for plot_manifolds.py:
        # in the frame xi = dx.v, eta = dx.u (v = invariant direction, u = unit
        # normal), orbits follow level sets of
        #   H(xi, eta) = (sigma/2) eta^2 - (D/3) xi^3,
        # and H = 0 is the separatrix (semicubical cusp tangent to v).
        with open(OUT_DIR + 'parabolic_cusp.txt', 'w') as fh:
            fh.write('# lam, theta, sigma, D\n')
            fh.write(f"{res['eigenvalues'][0]:.1f}, {res['line_angles'][0]:.8f}, "
                     f"{res['sigma']:.8e}, {res['D']:.8e}\n")

print("running the integration now...")

# Manifold growth (type-agnostic): integrate a fixed budget of toroidal transits,
# measure the manifold arclength in the phi=0 plane, and while it is below the
# target re-seed from the image of the outermost fundamental interval and
# integrate another budget. Stop once the target arclength is reached.
BUDGET = 6                     # toroidal transits per integration call
MANIFOLD_TARGET_LEN = 0.5   # target manifold arclength [m], measured at phi=0
MAX_GENERATIONS = 100          # cap on re-seeding iterations
TMAX_FL = 2e5


def _phi_plane_rows(hits, i, ids=None):
    """[seed_id, x, y, z] rows for every seed's hits in phi-plane i. ids maps
    seed index -> id label (default: the seed index itself)."""
    blocks = []
    for j, h in enumerate(hits):
        sel = h[h[:, 1] == i]
        if sel.size:
            sid = j if ids is None else ids[j]
            blocks.append(np.column_stack((np.full(sel.shape[0], sid), sel[:, 2], sel[:, 3], sel[:, 4])))
    return np.vstack(blocks) if blocks else np.zeros((0, 4))


def _phi_plane_rows_s(hits, i, ids, base_s):
    """[seed_id, s, x, y, z] rows for every seed's hits in phi-plane i, tagged
    with the global manifold parameter s. base_s[j] is the s-value of seed j's
    first recorded crossing this generation; successive crossings advance s by
    one return each. Ordering the manifold polyline by s (instead of by raw
    return index) is what keeps the line walking strictly outward: seeds that
    survive a different number of transits, or whose seed index does not match
    their position along the manifold, no longer fold the curve back on itself."""
    blocks = []
    for j, h in enumerate(hits):
        sel = h[h[:, 1] == i]
        if sel.size:
            s = base_s[j] + np.arange(sel.shape[0])
            sid = np.full(sel.shape[0], ids[j])
            blocks.append(np.column_stack((sid, s, sel[:, 2], sel[:, 3], sel[:, 4])))
    return np.vstack(blocks) if blocks else np.zeros((0, 5))


def _manifold_arclength(phi0, max_arclen=None):
    """Arclength of the manifold polyline (rows [id, s, x, y, z]), reordered by
    the global manifold parameter s so it walks outward from the X-point (matches
    plot_manifolds.py). With max_arclen set, the walk is cut there and
    (capped_length, kept_rows) is returned, where kept_rows is the leading
    <= max_arclen portion (original row order, so the plot's reorder reproduces
    the same outward walk on the subset)."""
    if phi0.shape[0] < 2:
        return (0.0, phi0) if max_arclen is not None else 0.0
    order = np.argsort(phi0[:, 1], kind='stable')   # walk outward by global parameter s
    R = np.hypot(phi0[order, 2], phi0[order, 3])
    Z = phi0[order, 4]
    cum = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(R), np.diff(Z)))))
    if max_arclen is None:
        return float(cum[-1])
    keep = order[cum <= max_arclen]
    return float(min(cum[-1], max_arclen)), phi0[np.sort(keep)]


def grow_manifold(signed_bfield, R0, Z0, t_seed):
    """Grow one manifold branch to MANIFOLD_TARGET_LEN by repeated fixed-budget
    integration with re-seeding. Each generation traces the current seeds for
    BUDGET transits; if the accumulated phi=0 arclength is still short, each
    surviving seed's last phi=0 crossing becomes the next generation's seed.

    Every crossing carries a global manifold parameter s = base_s + return, where
    base_s is the s-value of the seed's first recorded crossing this generation
    and t_seed[j] is seed j's offset within the initial fundamental interval (in
    return-map units). On re-seeding, base_s advances by the number of returns the
    seed actually completed, so the s-axis stays continuous and monotone across
    generations even though seeds survive different numbers of transits. Returns a
    list (length NPHI) of (M,5) arrays [seed_id, s, x, y, z] and the arclength."""
    
    phis = PHIS_RAD
    combined = [np.zeros((0, 5)) for _ in range(NPHI)]
    seeds_R = np.asarray(R0, float).copy()
    seeds_Z = np.asarray(Z0, float).copy()
    ids = np.arange(seeds_R.size)          # original-seed id, carried across generations
    base_s = np.asarray(t_seed, float).copy()   # global parameter of each seed's first crossing
    L = 0.0
    for gen in range(MAX_GENERATIONS):
        _, hits = compute_fieldlines(
            signed_bfield, seeds_R, seeds_Z, tmax=TMAX_FL, tol=1e-14, comm=comm_world,
            phis=phis, stopping_criteria=[ToroidalTransitStoppingCriterion(BUDGET, False)])
        # Append this generation's hits per phi plane, tagged with the global
        # parameter s so the manifold polyline is ordered by arclength position.
        for i in range(NPHI):
            combined[i] = np.vstack((combined[i], _phi_plane_rows_s(hits, i, ids, base_s)))
        L = _manifold_arclength(combined[0])
        proc0_print(f"    gen {gen}: {seeds_R.size} seeds, arclength {L:.4f} m")
        if L >= MANIFOLD_TARGET_LEN:
            break
        # Re-seed every surviving fieldline from its own last phi=0 crossing.
        # base_s advances by (#crossings - 1): the last crossing becomes the new
        # seed's first crossing, so s is continuous across the generation seam.
        # Seeds that did not complete a full transit (< 2 crossings) are dropped.
        nR, nZ, nid, nbase = [], [], [], []
        for j, h in enumerate(hits):
            p0 = h[h[:, 1] == 0]
            if p0.shape[0] >= 2:
                nR.append(np.hypot(p0[-1, 2], p0[-1, 3])); nZ.append(p0[-1, 4])
                nid.append(ids[j]); nbase.append(base_s[j] + (p0.shape[0] - 1))
        if not nR:
            break
        seeds_R, seeds_Z = np.array(nR), np.array(nZ)
        ids, base_s = np.array(nid), np.array(nbase)
    # The final generation overshoots the target; trim every plane to the target
    # arclength so we keep and report only the manifold up to MANIFOLD_TARGET_LEN.
    trimmed = [_manifold_arclength(c, MANIFOLD_TARGET_LEN) for c in combined]
    return [rows for (_, rows) in trimmed], trimmed[0][0]


def trace_fieldlines(bfield, g0, res):
    t1 = time.time()
    nmanif = 30
    nmanif_hyper = 30   # more seeds for the hyperbolic fundamental domain
    R_xp = np.sqrt(g0[0]**2 + g0[1]**2)
    dirs = [] if res['directions'] is None else res['directions']
    for k, v2 in enumerate(dirs):
        if res['type'] == 'hyperbolic':
            speed = res['eigenvalues'][0 if k < 2 else 1]   # rays [u+, u-, s+, s-]
            stable = k >= 2
        else:                                               # snowflake / parabolic
            speed = res['c'][k]
            stable = speed < 0
        kind = 'stable' if stable else 'unstable'
        ang = np.degrees(np.arctan2(v2[1], v2[0]))
        proc0_print(f"  tracing leg idx={k} ({kind}, angle {ang:+.2f} deg)")

        # Initial fundamental-interval seeds along the eigendirection. t_seed[j]
        # is seed j's offset (in return-map units) within the initial interval,
        # measured in the conjugacy coordinate where one phi=0 return is a unit
        # shift; it interleaves the seeds when the manifold is ordered by s.
        if res['type'] == 'hyperbolic':
            eps0 = 1e-3
            # log-uniform over one fundamental domain [eps0/ratio, eps0]; the
            # return map scales r by `ratio` per phi=0 return, so s = log_ratio(r).
            ratio = (abs(speed) if abs(speed) > 1.0 else 1.0 / abs(speed)) ** nfp
            r_vals = np.geomspace(eps0 / ratio, eps0, nmanif_hyper)
            t_seed = np.log(r_vals / r_vals.min()) / np.log(ratio)
        elif np.isfinite(speed) and abs(speed) > 1e-30:
            eps0 = 1e-3
            # snowflake/parabolic: uniform in 1/r (the quadratic drift is a 1/r
            # translation of size nfp*|c| per phi=0 return, c = res['c'][k]).
            r_max = min(np.sqrt(2 * eps0 / abs(speed)), 0.05)
            r_vals = 1.0 / np.linspace(1.0 / r_max, 1.0 / max(r_max / 100, 1e-3), nmanif)
            t_seed = (1.0 / r_vals.min() - 1.0 / r_vals) / (nfp * abs(speed))
        else:
            r_vals = np.geomspace(1e-3, 0.05, nmanif)
            t_seed = np.linspace(0.0, 1.0, r_vals.size)   # no model: order seeds only

        # parabolic (tr M = 2, single eigendirection, NOT the snowflake M = I
        # case) has no stable/unstable split, so trace each leg both forward AND
        # backward in time to capture the manifold segments flowing out of and
        # into the X-point; the backward trace is written under leg index
        # k + len(dirs) so its files stay distinct. Otherwise a single sign:
        # unstable -> forward, stable -> backward.
        if res['type'] == 'parabolic':
            sign_legs = [(+1.0, k), (-1.0, k + len(dirs))]
        else:
            sign_legs = [(+1.0 if kind == 'unstable' else -1.0, k)]
        R0 = R_xp + r_vals * v2[0]
        Z0 = g0[2] + r_vals * v2[1]

        for sign, k_out in sign_legs:
            for xp in (['top'] if is_sn else ['top', 'bot']):
                tt = sign * (1.0 if xp == 'top' else -1.0)

                Zseed = Z0 if xp == 'top' else -Z0
                combined, L = grow_manifold(tt * bfield, R0, Zseed, t_seed)
                proc0_print(f"  {time.time()-t1:.1f}s, leg {k} "
                            f"{'fwd' if sign > 0 else 'bwd'} {xp}: arclength {L:.4f} m")
                if comm_world is None or comm_world.rank == 0:
                    for i in range(NPHI):
                        np.savetxt(OUT_DIR + f'poincare_{xp}_{i}_{k_out}.txt',
                                   combined[i], comments='', delimiter=',')


def trace_interior(bfield, axis_pt, xp_pt, n_seeds=12, tmax_fl=2e5):
    """Seed field lines along the line from the magnetic axis to the X-point to
    fill the interior volume (nested flux surfaces inside the separatrix).
    Writes poincare_interior_{i}.txt, which plot_manifolds.py reads."""
    t1 = time.time()
    R_axis = np.sqrt(axis_pt[0]**2 + axis_pt[1]**2); Z_axis = axis_pt[2]
    R_xp = np.sqrt(xp_pt[0]**2 + xp_pt[1]**2);       Z_xp = xp_pt[2]
    # Skip the endpoints: the axis is a fixed point (no extra dots) and the
    # X-point sits on the separatrix.
    s = np.linspace(0.0, 1.0, n_seeds + 2)[1:-1]
    R0 = R_axis + s * (R_xp - R_axis)
    Z0 = Z_axis + s * (Z_xp - Z_axis)

    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-13, comm=comm_world,
        phis=PHIS_RAD, stopping_criteria=[ToroidalTransitStoppingCriterion(75, False),
                                          LevelsetStoppingCriterion(surface_classifier.dist)])
    proc0_print(f"  interior {time.time()-t1:.1f}s, hits/seed: " +
                ' '.join(str(h.shape[0]) for h in fieldlines_phi_hits))

    if comm_world is None or comm_world.rank == 0:
        for i in range(NPHI):
            np.savetxt(OUT_DIR + f'poincare_interior_{i}.txt',
                       _phi_plane_rows(fieldlines_phi_hits, i), comments='', delimiter=',')


def surface_cross_sections(surface):
    """Optimization-surface (R, Z) cross sections at the manifold phi values.
    surface.cross_section takes the normalized angle phi/(2*pi)."""
    for ii, phi in enumerate(PHIS):
        XYZ = surface.cross_section(phi, thetas=100)
        R, Z = np.hypot(XYZ[:, 0], XYZ[:, 1]), XYZ[:, 2]
        RZ = np.concatenate((R[:, None], Z[:, None]), axis=1)
        np.savetxt(OUT_DIR + f'surface_cross_{ii}.txt', RZ, delimiter=',')


def extract_vessel_cross_sections(sdf, nR=200, nZ=200):
    """Vessel SDF zero-level (R, Z) contours at the manifold phi values."""
    phis = PHIS_RAD
    Rg = np.linspace(0.0, 1.5, nR)
    Zg = np.linspace(-1.0, 1.0, nZ)
    Rm, Zm = np.meshgrid(Rg, Zg, indexing='xy')
    fig_tmp, ax_tmp = plt.subplots()   # only used to extract contour paths
    try:
        for i, phi in enumerate(phis):
            X = (Rm * np.cos(phi)).ravel()
            Y = (Rm * np.sin(phi)).ravel()
            Z = Zm.ravel()
            d = sdf.eval(X, Y, Z).reshape(Rm.shape)
            cs = ax_tmp.contour(Rg, Zg, d, levels=[0.0])
            paths = (cs.get_paths() if hasattr(cs, 'get_paths')
                     else [pth for c in cs.collections for pth in c.get_paths()])
            segs = []
            for path in paths:
                v = path.vertices
                if v.shape[0] < 2:
                    continue
                segs.append(v)
                segs.append(np.array([[np.nan, np.nan]]))
            data = np.vstack(segs) if segs else np.zeros((0, 2))
            np.savetxt(OUT_DIR + f'vessel_cross_{i}.txt', data, delimiter=',',
                       comments='', header='R,Z (NaN rows separate disconnected components)')
            ax_tmp.clear()
    finally:
        plt.close(fig_tmp)


def save_fixed_points(axis_curve, xp_curve):
    """Magnetic-axis and X-point (R, Z) at each manifold phi.
    row0=axis, row1=xpoint (top), row2=stellsym partner (bottom) when computable.

    For DN the two X-points are a stellarator-symmetry pair: the field is
    invariant under (R, phi, Z) -> (R, -phi, -Z), so the bottom X-point's crossing
    at +phi equals the top X-point field line evaluated at -phi with Z negated.
    For SN this no longer holds (not stellsym), so the bottom row is omitted and
    only the top X-point is recorded.

    A phi's file is written only when the axis + top evaluate_at_phi succeed; the
    bottom row is appended (DN only) when its evaluate_at_phi also succeeds."""
    for ii, phi in enumerate(PHIS):
        a_xyz, a_ok = evaluate_at_phi(axis_curve, phi)
        x_xyz, x_ok = evaluate_at_phi(xp_curve, phi)
        if not (a_ok and x_ok):
            proc0_print(f"  fixed_points: evaluate_at_phi failed at phi/2pi={phi:.4f}; skipping")
            continue
        rows = [[np.hypot(a_xyz[0], a_xyz[1]), a_xyz[2]],
                [np.hypot(x_xyz[0], x_xyz[1]), x_xyz[2]]]
        # bottom X-point = stellsym image (DN only): top curve at -phi, Z negated.
        if not is_sn:
            b_xyz, b_ok = evaluate_at_phi(xp_curve, -phi)
            if b_ok:
                rows.append([np.hypot(b_xyz[0], b_xyz[1]), -b_xyz[2]])
            else:
                proc0_print(f"  fixed_points: bottom evaluate_at_phi failed at phi/2pi={phi:.4f}")
        np.savetxt(OUT_DIR + f'fixed_points_{ii}.txt', np.array(rows), delimiter=',',
                   comments='', header='R,Z (row0=axis, row1=xpoint top, row2=xpoint bottom)')


if comm_world is None or comm_world.rank == 0:
    surface_cross_sections(boozer_surface.surface)
    extract_vessel_cross_sections(sdf)
    save_fixed_points(axes[0].curve, xpoint.curve)

trace_fieldlines(boozer_surface.biotsavart, g0, res)
trace_interior(boozer_surface.biotsavart, axes[0].curve.gamma()[0], g0)
