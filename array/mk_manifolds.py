#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from simsopt._core import load
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import CurveLength, CurveXYZFourierSymmetries, SurfaceXYZTensorFourier, CurveRZFourier
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.boozer_surface_utils import BoozerResidual, CurveBoozerSurfaceDistance
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine
from star_lite_design.utils.curve_periodicfieldline_distance import CurvesPeriodicFieldlineDistance
from star_lite_design.utils.tangent_map import TangentMap, Monodromy
from simsopt.field import (InterpolatedField, particles_to_vtk,
                           compute_fieldlines, IterationStoppingCriterion, plot_poincare_data)
import time
from simsopt.util import in_github_actions, proc0_print, comm_world
import os
from pathlib import Path
import sys
import argparse


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


def compute(axis, magnetic_field, tol=1e-7, id_tol=1e-6):
    """
    Classify the on-axis tangent (monodromy) map of a magnetic axis over one field period
    and return the corresponding invariant data. Copied from
    simsopt.field.magnetic_axis_helpers so this script has no external dependency on it.
    """
    assert type(axis) is CurveRZFourier
    nfp = axis.nfp
    L = 1.0 / nfp

    # ---- Pass 1: integrate the tangent (monodromy) map over one field period. ----
    def rhs_M(t, y):
        _, A, _ = _on_axis_field_data(axis, magnetic_field, t, need_hessian=False)
        M = y.reshape((2, 2))
        return (2 * np.pi * (A @ M)).flatten()

    sol = solve_ivp(rhs_M, [0.0, L], np.eye(2).flatten(),
                    rtol=1e-12, atol=1e-12, method='RK45')
    M = sol.y[:, -1].reshape((2, 2))
    tr = M[0, 0] + M[1, 1]

    # Standardized output across all cases. Merged fields:
    #   eigenvalues : (2,) eigenvalues of M (subsumes the old scalar 'eigenvalue'
    #                 and the unstable/stable eigenvalues)
    #   directions  : (2*n_lines, 2) invariant unit rays (subsumes the old
    #                 'invariant_direction' / 'unstable_direction' / 'stable_direction')
    #   line_angles : invariant-line angles those rays come from
    #   iota        : elliptic only;  T : snowflake only
    res = {
        'type': None,
        'M': M,
        'eigenvalues': np.linalg.eig(M)[0],
        'iota': None,
        'line_angles': None,
        'directions': None,
        'legs': None,          # per-ray {'direction', 'c', 'stable'}
        'T': None,
    }

    def _integrate_second_order_T():
        # Integrate the linear monodromy M(t) and the second-order tensor T
        # together over one field period; returns the (2, 2, 2) tensor T.
        def rhs_full(t, y):
            _, A, Hg = _on_axis_field_data(axis, magnetic_field, t, need_hessian=True)
            Mt = y[:4].reshape((2, 2))
            dM = 2 * np.pi * (A @ Mt)
            Minv = np.linalg.inv(Mt)
            dT = np.zeros((2, 2, 2))
            for p in range(2):
                inner = Mt.T @ Hg[p] @ Mt          # (M^T Hg[p] M)[j,k]
                for i in range(2):
                    dT[i] += np.pi * Minv[i, p] * inner
            return np.concatenate([dM.flatten(), dT.flatten()])
        y0 = np.concatenate([np.eye(2).flatten(), np.zeros(8)])
        sol2 = solve_ivp(rhs_full, [0.0, L], y0, rtol=1e-12, atol=1e-12, method='RK45')
        return sol2.y[4:, -1].reshape((2, 2, 2))

    # ---- Classify. ----
    if np.abs(M - np.eye(2)).max() < id_tol:
        case = 'snowflake'
    elif np.abs(M + np.eye(2)).max() < id_tol:
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
        # elliptic has no real invariant directions

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
        res['T'] = _integrate_second_order_T()

    else:  # snowflake (M = +I): integrate the second-order return map.
        T = _integrate_second_order_T()
        line_angles, directions = _snowflake_directions(T)
        res['line_angles'] = line_angles
        res['directions'] = directions
        res['T'] = T

    # ---- Classify each invariant ray into a leg. ----
    if res['directions'] is None:
        res['legs'] = None
    elif res['T'] is not None:
        # Second-order radial coefficient c = Q(v)·v, Q_i(v) = sum_jk T[i,j,k] v_j v_k;
        # stable if c < 0. Valid for the identity/unipotent linear part (snowflake
        # and the tr(M)=+2 parabolic case).
        legs = []
        for v in res['directions']:
            Qv = np.array([sum(res['T'][i, j, k] * v[j] * v[k]
                               for j in range(2) for k in range(2)) for i in range(2)])
            c = float(np.dot(Qv, v))
            legs.append({'direction': np.asarray(v), 'c': c, 'lam': None, 'stable': c < 0})
        res['legs'] = legs
    else:
        # hyperbolic: rays are [unstable+, unstable-, stable+, stable-] (the first
        # line is unstable). Classify by eigen-line; no quadratic rate (c = nan).
        # 'lam' carries the eigenvalue of each ray's line (eigenvalues are
        # ordered [unstable, stable]) for fundamental-domain seeding.
        lam_u, lam_s = res['eigenvalues']
        ray_lam = [lam_u, lam_u, lam_s, lam_s]
        legs = []
        for idx_ray, v in enumerate(res['directions']):
            legs.append({'direction': np.asarray(v), 'c': float('nan'),
                         'lam': float(np.real(ray_lam[idx_ray])), 'stable': idx_ray >= 2})
        res['legs'] = legs

    return res


def _monodromy(curve, magnetic_field, nfp, order=16):
    """Fit curve to a CurveRZFourier and integrate the (2,2) monodromy M over one
    field period. Returns (cRZ, M), or (None, None) if the fit fails."""
    quadpoints = np.linspace(0, 0.5, 2 * order + 1, endpoint=False)
    XYZ = []
    for phi in quadpoints:
        xyz, ok = evaluate_at_phi(curve, phi)
        if not ok:
            return None, None
        XYZ.append(xyz)
    cRZ = CurveRZFourier(quadpoints, order, nfp, False)
    cRZ.least_squares_fit(XYZ)
    L = 1.0 / nfp

    def rhs_M(t, y):
        _, A, _ = _on_axis_field_data(cRZ, magnetic_field, t, need_hessian=False)
        return (2 * np.pi * (A @ y.reshape((2, 2)))).flatten()

    sol = solve_ivp(rhs_M, [0.0, L], np.eye(2).flatten(), rtol=1e-12, atol=1e-12, method='RK45')
    return cRZ, sol.y[:, -1].reshape((2, 2))


def _integrate_T(cRZ, magnetic_field, nfp):
    """Second-order return-map tensor T(2,2,2) over one field period."""
    L = 1.0 / nfp

    def rhs_full(t, y):
        _, A, Hg = _on_axis_field_data(cRZ, magnetic_field, t, need_hessian=True)
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
    sol2 = solve_ivp(rhs_full, [0.0, L], y0, rtol=1e-12, atol=1e-12, method='RK45')
    return sol2.y[4:, -1].reshape((2, 2, 2))


def print_fixed_point_info(name, curve, magnetic_field, nfp, order=16, tol=1e-7, id_tol=1e-6):
    """Print a full classification of the fixed point to the console (same format
    as fieldline_gui_qt.py)."""
    print(f"==================== {name} ====================")
    cRZ, M = _monodromy(curve, magnetic_field, nfp, order)
    if M is None:
        print("  curve not evaluable at all phi; cannot classify.")
        return
    tr = float(M[0, 0] + M[1, 1])
    det = float(np.linalg.det(M))
    evals = np.linalg.eigvals(M)
    xyz, ok = evaluate_at_phi(curve, 0.0)
    if ok:
        print(f"  location (phi=0): R = {np.hypot(xyz[0], xyz[1]):.6f}, Z = {xyz[2]:.6f}")
    print(f"  monodromy M = [[{M[0,0]:+.6f}, {M[0,1]:+.6f}], [{M[1,0]:+.6f}, {M[1,1]:+.6f}]]")
    print(f"  trace(M) = {tr:+.8f}    det(M) = {det:+.8f}")
    print(f"  eigenvalues = {np.array2string(evals, precision=6)}")

    if np.abs(M - np.eye(2)).max() < id_tol:
        T = _integrate_T(cRZ, magnetic_field, nfp)
        line_angles, dirs = _snowflake_directions(T)
        print(f"  type: SNOWFLAKE (M = I)")
        print(f"  invariant lines: {len(line_angles)}   rays (legs): {len(dirs)}")
        for t in line_angles:
            v = np.array([np.cos(t), np.sin(t)])
            Qv = np.array([sum(T[i, j, k] * v[j] * v[k]
                               for j in range(2) for k in range(2)) for i in range(2)])
            c = float(np.dot(Qv, v))
            print(f"    line {np.degrees(t):+7.2f} deg   c = {c:+.4e}   "
                  f"({'unstable' if c > 0 else 'stable'})")
    elif np.abs(tr) > 2.0 + tol:
        ev, evecs = np.linalg.eig(M)
        ev = np.real(ev); evecs = np.real(evecs)
        iu = int(np.argmax(np.abs(ev))); is_ = 1 - iu
        au = np.degrees(np.arctan2(evecs[1, iu], evecs[0, iu]))
        as_ = np.degrees(np.arctan2(evecs[1, is_], evecs[0, is_]))
        print(f"  type: HYPERBOLIC")
        print(f"  invariant lines: 2   rays (legs): 4")
        print(f"    unstable: eigenvalue {ev[iu]:+.6f}  angle {au:+.2f} deg")
        print(f"    stable:   eigenvalue {ev[is_]:+.6f}  angle {as_:+.2f} deg")
    elif np.abs(np.abs(tr) - 2.0) <= tol:
        lam = float(np.sign(tr))
        K = M - lam * np.eye(2)
        a, b = K[0, 0], K[0, 1]
        cc, d = K[1, 0], K[1, 1]
        v = np.array([b, -a]) if abs(a) + abs(b) >= abs(cc) + abs(d) else np.array([d, -cc])
        ang = np.degrees(np.arctan2(v[1], v[0]))
        print(f"  type: PARABOLIC (repeated eigenvalue {lam:+.0f})")
        print(f"  invariant lines: 1   rays (legs): 2   angle {ang:+.2f} deg")
    else:
        iota = np.arctan2(np.imag(evals[0]), np.real(evals[0])) * nfp / (2 * np.pi)
        a, b = M[0, 0], M[0, 1]
        c, d = M[1, 0], M[1, 1]
        Q = np.array([[c, 0.5 * (d - a)], [0.5 * (d - a), -b]])
        if Q[0, 0] < 0:
            Q = -Q
        w, V = np.linalg.eigh(Q)
        sa = 1.0 / np.sqrt(w[0])
        sb = 1.0 / np.sqrt(w[1])
        nrm = np.sqrt(sa * sb)
        angle = np.degrees(np.arctan2(V[1, 0], V[0, 0]))
        print(f"  type: ELLIPTIC")
        print(f"  rotational transform iota = {iota:+.6f}")
        print(f"  invariant ellipse: major/minor ratio = {sa / sb:.6f}")
        print(f"    semi-major (unit area) = {sa / nrm:.6f}   semi-minor = {sb / nrm:.6f}")
        print(f"    inclination = {angle:+.2f} deg")


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

xpoint.need_to_run_code=False
xpoint.res = {'length':CurveLength(xpoint.curve).J()}

OUT_DIR = str(p.parent) + "/"   # write next to singular.json so plot + sync find the files
os.makedirs(OUT_DIR, exist_ok=True)
legs = res['legs'] or []

g0 = xpoint.curve.gamma()[0]
nfp = xpoint.curve.nfp

if comm_world is None or comm_world.rank == 0:
    np.savetxt(OUT_DIR + 'xpoint.txt', g0[None, :])
    with open(OUT_DIR + 'legs.txt', 'w') as fh:
        fh.write('# k, theta, vR, vZ, kind, sign, vK(vv)\n')
        
        for k, leg in enumerate(legs):
            v2  = leg['direction']
            th = np.arctan2(v2[1], v2[0])
            rad = leg['c']
            sign = +1.0 if not leg['stable'] else -1.0
            kind = 'stable' if leg['stable'] else 'unstable'
            fh.write(f"{k}, {th:.6f}, {v2[0]:.6f}, {v2[1]:.6f}, {kind}, {sign:+.0f}, {rad:.6f}\n")

if comm_world is None or comm_world.rank == 0:
    print_fixed_point_info("X-point", xpoint.curve, boozer_surface.biotsavart,
                           xpoint.curve.nfp, tol=CLASSIFY_TOL)
    print_fixed_point_info("O-point (magnetic axis)", axes[0].curve, boozer_surface.biotsavart,
                           axes[0].curve.nfp, tol=CLASSIFY_TOL)


print("running the integration now...")
def trace_fieldlines(bfield, g0, legs):
    tmax_fl = 1e5
    t1 = time.time()

    n_stable = sum(1 for leg in legs if leg['stable'])
    proc0_print(f"tracing {len(legs)} legs "
                f"({len(legs) - n_stable} unstable, {n_stable} stable)")

    nmanif=10
    #r_vals  = np.geomspace(1e-4, 5e-2, nmanif)   # multi-radius seeding
    for k, leg in enumerate(legs):
        rad = leg['c']
        lam = leg.get('lam')
        kind = 'stable' if leg['stable'] else 'unstable'
        r_max = 0.05                         # outer seed radius (stay in linear regime)
        if lam is not None and np.isfinite(lam) and abs(lam) not in (0.0, 1.0):
            # Hyperbolic: the map r -> lambda r is a translation in u = log r, so
            # the invariant seeding is uniform in log r (geomspace) over one
            # fundamental domain [r_max/|lambda|, r_max]. (det M = 1 => the
            # backward-traced stable leg expands by 1/|lambda_s| = |lambda_u|.)
            ratio = abs(lam) if abs(lam) > 1.0 else 1.0 / abs(lam)
            r_min = max(r_max / ratio, 1e-4)
            r_vals = np.geomspace(r_min, r_max, nmanif)
        elif np.isfinite(rad) and abs(rad) > 1e-30:
            # Snowflake/parabolic: quadratic drift r -> r + 1/2 c r^2. This map is
            # a translation in u = 1/r, so the invariant seeding is uniform in
            # 1/r (not log r). r_max set so the outer per-period step ~ target_step.
            target_step = 0.01  # desired per-return displacement (5 mm)
            r_max = min(np.sqrt(2 * target_step / abs(rad)), 0.05)
            r_min = max(r_max / 50, 1e-3)
            r_vals = 1.0 / np.linspace(1.0 / r_max, 1.0 / r_min, nmanif)
        else:
            r_min = max(r_max / 50, 1e-3)
            r_vals = np.geomspace(r_min, r_max, nmanif)

        v2  = leg['direction']
        sign = +1.0 if kind == 'unstable' else -1.0
        # v2 is the leg direction in the (R, Z) plane; seed directly in
        # cylindrical coords (compute_fieldlines launches at phi=0).
        R0 = np.sqrt(g0[0]**2 + g0[1]**2) + r_vals * v2[0]
        Z0 = g0[2] + r_vals * v2[1]

        for xp in ['top', 'bot']:
            tt = sign * (1.0 if xp == 'top' else -1.0)
            phis = np.linspace(0, 0.25, 9)*2*np.pi
            fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
                tt*sign*bfield, R0, (-1.0 if xp == 'bot' else 1.0) * Z0, tmax=tmax_fl, tol=1e-13, comm=comm_world,
                phis=phis, stopping_criteria=[IterationStoppingCriterion(int(1e5))])
            proc0_print(f"  {time.time()-t1:.1f}s, hits/seed: " +' '.join(str(h.shape[0]) for h in fieldlines_phi_hits))

            if comm_world is None or comm_world.rank == 0:
                for i in range(len(phis)):
                    data_this_phi = np.zeros((0, 4))
                    for j in range(len(fieldlines_phi_hits)):
                        toadd = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == i)[0], 2:]
                        toadd = np.concatenate( (j*np.ones((toadd.shape[0], 1)), toadd), axis=1)
                        data_this_phi = np.concatenate((data_this_phi, toadd), axis=0)
                    np.savetxt(OUT_DIR+f'poincare_{xp}_{i}_{k}.txt', data_this_phi, comments='', delimiter=',')


def trace_interior(bfield, axis_pt, xp_pt, n_seeds=12, tmax_fl=1e5):
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

    phis = np.linspace(0, 0.25, 9) * 2 * np.pi
    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
        bfield, R0, Z0, tmax=tmax_fl, tol=1e-13, comm=comm_world,
        phis=phis, stopping_criteria=[IterationStoppingCriterion(int(1e5))])
    proc0_print(f"  interior {time.time()-t1:.1f}s, hits/seed: " +
                ' '.join(str(h.shape[0]) for h in fieldlines_phi_hits))

    if comm_world is None or comm_world.rank == 0:
        for i in range(len(phis)):
            data_this_phi = np.zeros((0, 4))
            for j in range(len(fieldlines_phi_hits)):
                m = fieldlines_phi_hits[j][:, 1] == i
                toadd = fieldlines_phi_hits[j][m, 2:]
                toadd = np.concatenate((j * np.ones((toadd.shape[0], 1)), toadd), axis=1)
                data_this_phi = np.concatenate((data_this_phi, toadd), axis=0)
            np.savetxt(OUT_DIR + f'poincare_interior_{i}.txt',
                       data_this_phi, comments='', delimiter=',')


def surface_cross_sections(surface):
    """Optimization-surface (R, Z) cross sections at the manifold phi values.
    surface.cross_section takes the normalized angle phi/(2*pi)."""
    phis = np.linspace(0, 0.25, 9)
    for ii, phi in enumerate(phis):
        XYZ = surface.cross_section(phi, thetas=100)
        R, Z = np.hypot(XYZ[:, 0], XYZ[:, 1]), XYZ[:, 2]
        RZ = np.concatenate((R[:, None], Z[:, None]), axis=1)
        np.savetxt(OUT_DIR + f'surface_cross_{ii}.txt', RZ, delimiter=',')


def extract_vessel_cross_sections(sdf, nR=200, nZ=200):
    """Vessel SDF zero-level (R, Z) contours at the manifold phi values."""
    phis = np.linspace(0, 0.25, 9) * 2 * np.pi
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
    """Magnetic-axis and X-point (R, Z) at each manifold phi (row0=axis, row1=xpoint).
    A phi's file is written only when both evaluate_at_phi calls succeed."""
    phis = np.linspace(0, 0.25, 9)
    for ii, phi in enumerate(phis):
        a_xyz, a_ok = evaluate_at_phi(axis_curve, phi)
        x_xyz, x_ok = evaluate_at_phi(xp_curve, phi)
        if not (a_ok and x_ok):
            proc0_print(f"  fixed_points: evaluate_at_phi failed at phi/2pi={phi:.4f}; skipping")
            continue
        rows = np.array([[np.hypot(a_xyz[0], a_xyz[1]), a_xyz[2]],
                         [np.hypot(x_xyz[0], x_xyz[1]), x_xyz[2]]])
        np.savetxt(OUT_DIR + f'fixed_points_{ii}.txt', rows, delimiter=',',
                   comments='', header='R,Z (row0=axis, row1=xpoint)')


if comm_world is None or comm_world.rank == 0:
    surface_cross_sections(boozer_surface.surface)
    extract_vessel_cross_sections(sdf)
    save_fixed_points(axes[0].curve, xpoint.curve)

trace_fieldlines(boozer_surface.biotsavart, g0, legs)
trace_interior(boozer_surface.biotsavart, axes[0].curve.gamma()[0], g0)
