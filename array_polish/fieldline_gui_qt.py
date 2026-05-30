#!/usr/bin/env python3
"""
pyqtgraph port of the interactive fieldline tracer.

Why pyqtgraph: unlike matplotlib, equal aspect + linked (shared) axes + zero
inter-panel spacing all coexist. Panels are linked so pan/zoom on one syncs all.

Run:  python fieldline_gui_qt.py <data_file>
Needs: pip install pyqtgraph pyqt5   (a desktop Qt environment; not headless)
"""
import sys
import time
import threading
import queue

import numpy as np
from scipy.optimize import root_scalar
from scipy.integrate import solve_ivp

import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui

# White background / black foreground (pyqtgraph defaults to the opposite).
pg.setConfigOption("background", "w")
pg.setConfigOption("foreground", "k")
pg.setConfigOptions(antialias=True)

from simsopt._core import load
from simsopt.geo import CurveLength, CurveRZFourier
from simsopt.util import proc0_print, comm_world
from simsopt.field import compute_fieldlines_xyz

# ============================================================
# Configuration
# ============================================================
DATA_FILE = sys.argv[1] if len(sys.argv) > 1 else None

TOL = 1e-10
POINT_SIZE = 3
DEFAULT_TMAX = 1000.0
DEFAULT_XLIM = (0.0, 1.5)
DEFAULT_ZLIM = (-0.75, 0.75)
N_PHIS = 9
SDF_GRID_NR = 120
SDF_GRID_NZ = 120
ELLIPSE_SCALE = 0.01
N_ELLIPSES = 5

# tab20 categorical palette (RGB 0-255), used to colour dots by fieldline id.
PALETTE = [
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
]


# ============================================================
# Geometry / dynamics helpers (framework-agnostic)
# ============================================================
def evaluate_at_phi(curve, phi, tol=1e-10):
    phi += np.ceil(-phi)

    def curve_val(theta):
        return curve.gamma_pure(curve.x, np.array([theta]))[0]

    def theta2phi(theta_in, phi0):
        xyz = curve_val(theta_in)
        angle = np.arctan2(xyz[1], xyz[0]) / (2 * np.pi) - phi0
        angle += np.ceil(-angle)
        return angle

    def fun(theta):
        if theta == 1.:
            return 1. - phi_prime
        return theta2phi(theta, phi0) - phi_prime

    xyz0 = curve_val(0.)
    phi0 = np.arctan2(xyz0[1], xyz0[0]) / (2 * np.pi)
    phi_prime = phi - phi0
    phi_prime += np.ceil(-phi_prime)
    try:
        result = root_scalar(fun, bracket=[0, 1.])
        if not result.converged:
            return None, False
        xyz = curve_val(result.root)
        R = np.sqrt(xyz[0]**2 + xyz[1]**2)
        c, s = np.cos(2 * np.pi * phi), np.sin(2 * np.pi * phi)
        accurate = np.abs(-s * xyz[0] + c * xyz[1]) / R < tol
        return xyz, accurate
    except Exception:
        return None, False


def _on_axis_field_data(axis, magnetic_field, phi, need_hessian):
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
    dBdR = c * GradB[0, :] + s * GradB[1, :]
    dBdZ = GradB[2, :]

    BR = c * B[0] + s * B[1]
    Bp = -s * B[0] + c * B[1]
    BZ = B[2]
    BR_R = c * dBdR[0] + s * dBdR[1]; BR_Z = c * dBdZ[0] + s * dBdZ[1]
    Bp_R = -s * dBdR[0] + c * dBdR[1]; Bp_Z = -s * dBdZ[0] + c * dBdZ[1]
    BZ_R = dBdR[2];                    BZ_Z = dBdZ[2]

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
        A = np.array([[h + R * hR, R * hZ], [h2 + R * h2R, R * h2Z]])
        return R, A, None

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
    A = np.array([[h + R * hR, R * hZ], [h2 + R * h2R, R * h2Z]])
    Hg = np.zeros((2, 2, 2))
    Hg[0, 0, 0] = 2 * hR + R * hRR
    Hg[0, 0, 1] = Hg[0, 1, 0] = hZ + R * hRZ
    Hg[0, 1, 1] = R * hZZ
    Hg[1, 0, 0] = 2 * h2R + R * h2RR
    Hg[1, 0, 1] = Hg[1, 1, 0] = h2Z + R * h2RZ
    Hg[1, 1, 1] = R * h2ZZ
    return R, A, Hg


def _rays_from_line_angles(line_angles):
    directions = []
    for t in line_angles:
        directions.append([np.cos(t), np.sin(t)])
        directions.append([np.cos(t + np.pi), np.sin(t + np.pi)])
    return np.array(directions)


def _snowflake_directions(T):
    a0 = T[0, 0, 0]; a1 = 2 * T[0, 0, 1]; a2 = T[0, 1, 1]
    b0 = T[1, 0, 0]; b1 = 2 * T[1, 0, 1]; b2 = T[1, 1, 1]

    def f(t):
        c = np.cos(t); s = np.sin(t)
        return -b0 * c**3 + (a0 - b1) * c**2 * s + (a1 - b2) * c * s**2 + a2 * s**3

    Ngrid = 4000
    grid = np.linspace(0.0, np.pi, Ngrid, endpoint=False)
    step = np.pi / Ngrid
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
    roots = sorted(roots)
    uniq = []
    for r in roots:
        if not any(abs(r - q) < 1e-7 or abs(abs(r - q) - np.pi) < 1e-7 for q in uniq):
            uniq.append(r)
    line_angles = np.array(uniq)
    return line_angles, _rays_from_line_angles(line_angles)


def _monodromy(curve, magnetic_field, nfp, order=16):
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


def invariant_directions(xp_curve, magnetic_field, nfp, order=16, tol=1e-7, id_tol=1e-6):
    """Invariant ray directions (2*n_lines, 2) in (R, Z) at the fixed point, or
    None if elliptic / the fit fails."""
    cRZ, M = _monodromy(xp_curve, magnetic_field, nfp, order)
    if M is None:
        return None
    tr = M[0, 0] + M[1, 1]
    if np.abs(M - np.eye(2)).max() < id_tol:
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
        T = sol2.y[4:, -1].reshape((2, 2, 2))
        return _snowflake_directions(T)[1]
    elif np.abs(tr) > 2.0 + tol:
        evals, evecs = np.linalg.eig(M)
        evecs = np.real(evecs)
        angs = [np.arctan2(evecs[1, j], evecs[0, j]) for j in range(2)]
        return _rays_from_line_angles(np.array(angs))
    elif np.abs(np.abs(tr) - 2.0) <= tol:
        lam = float(np.sign(tr))
        K = M - lam * np.eye(2)
        a, b = K[0, 0], K[0, 1]
        cc, d = K[1, 0], K[1, 1]
        v = np.array([b, -a]) if abs(a) + abs(b) >= abs(cc) + abs(d) else np.array([d, -cc])
        n = np.linalg.norm(v)
        v = v / n if n > 0 else np.array([1.0, 0.0])
        return _rays_from_line_angles(np.array([np.arctan2(v[1], v[0])]))
    return None


def elliptic_ellipse(curve, magnetic_field, nfp, order=16, tol=1e-7):
    """For an elliptic fixed point: (semi_major, semi_minor, angle_rad) of the
    invariant ellipse, unit-area normalized, or None if not elliptic."""
    cRZ, M = _monodromy(curve, magnetic_field, nfp, order)
    if M is None:
        return None
    a, b = M[0, 0], M[0, 1]
    c, d = M[1, 0], M[1, 1]
    if np.abs(a + d) >= 2.0 - tol:
        return None
    Q = np.array([[c, 0.5 * (d - a)], [0.5 * (d - a), -b]])
    if Q[0, 0] < 0:
        Q = -Q
    w, V = np.linalg.eigh(Q)
    sa = 1.0 / np.sqrt(w[0])
    sb = 1.0 / np.sqrt(w[1])
    norm = np.sqrt(sa * sb)
    sa, sb = sa / norm, sb / norm
    angle = np.arctan2(V[1, 0], V[0, 0])
    return sa, sb, angle


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
    """Print a full classification of the fixed point to the console."""
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


def _1d(a):
    """Coerce to a contiguous 1D float ndarray (pyqtgraph requires this)."""
    return np.asarray(a, dtype=float).ravel()


def _ellipse_curve(Rc, Zc, sa, sb, ang, scale, n=120):
    t = np.linspace(0, 2 * np.pi, n)
    x = scale * sa * np.cos(t)
    y = scale * sb * np.sin(t)
    ca, sang = np.cos(ang), np.sin(ang)
    return Rc + ca * x - sang * y, Zc + sang * x + ca * y


# ============================================================
# GUI
# ============================================================
class FieldlineGUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.tmax_value = DEFAULT_TMAX
        self.tol_value = TOL
        self.ellipse_scale = ELLIPSE_SCALE
        self.poincare_items = []     # ScatterPlotItems across panels
        self.click_items = {}        # phi_idx -> click marker
        self.ellipse_items = []      # axis-ellipse curves + dot
        self.result_queue = queue.Queue()
        self.worker = None
        self.trace_count = 0

        self._load_data()
        self._build_ui()
        self._draw_static_overlays()
        self._draw_axis_ellipses()
        self._fit_view()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._poll_worker)
        self.timer.start(150)
        # Re-fit once the window is shown and laid out (the aspect-locked range
        # computed in __init__ is wrong until the ViewBox has its real size).
        QtCore.QTimer.singleShot(0, self._fit_view)
        self._set_status("Ready. Click any panel to trace a fieldline from that phi.")

    # --- data ----------------------------------------------------------------
    def _load_data(self):
        data = load(DATA_FILE)
        self.boozer_surface = data[0][0]
        self.magnetic_axes = data[2]
        self.xpoint = data[3][0]
        self.sdf = data[4]
        self.xpoint.need_to_run_code = False
        self.xpoint.res = {"length": CurveLength(self.xpoint.curve).J()}
        self.bs = self.boozer_surface.biotsavart
        self.curve = self.xpoint.curve
        self.axis_curve = self.magnetic_axes[0].curve
        self.nfp = self.curve.nfp
        self.phis_normalized = np.linspace(0, 0.25, N_PHIS)
        self.phis = self.phis_normalized * 2 * np.pi

        # Console summary of both fixed points.
        print_fixed_point_info("X-point", self.curve, self.bs, self.nfp)
        print_fixed_point_info("O-point (magnetic axis)", self.axis_curve, self.bs, self.nfp)

        try:
            self.invariant_dirs = invariant_directions(self.curve, self.bs, self.nfp)
        except Exception as e:
            proc0_print(f"invariant_directions failed: {e}")
            self.invariant_dirs = None
        try:
            self.axis_ellipse = elliptic_ellipse(self.axis_curve, self.bs, self.nfp)
        except Exception as e:
            proc0_print(f"elliptic_ellipse failed: {e}")
            self.axis_ellipse = None

    # --- UI ------------------------------------------------------------------
    def _build_ui(self):
        self.setWindowTitle("Fieldline tracer (pyqtgraph)")
        layout = QtWidgets.QVBoxLayout(self)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.layout.setSpacing(0)
        self.glw.ci.layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.glw, stretch=1)

        self.plots = []
        p0 = None
        for k in range(N_PHIS):
            row, col = divmod(k, 3)
            p = self.glw.addPlot(row=row, col=col)
            p.setAspectLocked(True)                 # equal aspect (coexists with linking)
            p.setMenuEnabled(False)
            p.showGrid(x=True, y=True, alpha=0.3)   # gridlines
            p.disableAutoRange()                    # don't auto-zoom when items are added
            p.setXRange(*DEFAULT_XLIM, padding=0)
            p.setYRange(*DEFAULT_ZLIM, padding=0)
            if p0 is None:
                p0 = p
            else:
                p.setXLink(p0)                      # shared axes
                p.setYLink(p0)
            # Identical axis reservation so cells align; tick values only on edges.
            for side in ("left", "bottom"):
                ax = p.getAxis(side)
                ax.setStyle(showValues=(side == "left" and col == 0) or
                                       (side == "bottom" and row == 2))
            p.getAxis("left").setWidth(46)
            p.getAxis("bottom").setHeight(26)
            txt = pg.TextItem(f"phi/2pi = {self.phis_normalized[k]:.3f}",
                              color=(40, 40, 40), anchor=(0, 0))
            txt.setPos(DEFAULT_XLIM[0], DEFAULT_ZLIM[1])
            p.addItem(txt)
            p.scene().sigMouseClicked.connect(self._on_click)
            self.plots.append(p)

        # Controls
        ctrl = QtWidgets.QHBoxLayout()
        self.ed_tmax = QtWidgets.QLineEdit(f"{DEFAULT_TMAX:g}")
        self.ed_tol = QtWidgets.QLineEdit(f"{TOL:g}")
        self.ed_escale = QtWidgets.QLineEdit(f"{ELLIPSE_SCALE:g}")
        for lbl, ed in [("tmax", self.ed_tmax), ("tol", self.ed_tol), ("ellipse", self.ed_escale)]:
            ctrl.addWidget(QtWidgets.QLabel(lbl))
            ed.setMaximumWidth(90)
            ctrl.addWidget(ed)
        self.ed_tmax.editingFinished.connect(self._on_tmax)
        self.ed_tol.editingFinished.connect(self._on_tol)
        self.ed_escale.editingFinished.connect(self._on_escale)

        btn_clear = QtWidgets.QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear)
        btn_reset = QtWidgets.QPushButton("Reset view")
        btn_reset.clicked.connect(self._on_reset)
        ctrl.addStretch(1)
        ctrl.addWidget(btn_clear)
        ctrl.addWidget(btn_reset)
        layout.addLayout(ctrl)

        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)
        self.resize(1000, 1100)

    def _set_status(self, msg):
        self.status.setText(msg)

    # --- static overlays -----------------------------------------------------
    def _draw_static_overlays(self):
        R_grid = np.linspace(DEFAULT_XLIM[0], DEFAULT_XLIM[1], SDF_GRID_NR)
        Z_grid = np.linspace(DEFAULT_ZLIM[0], DEFAULT_ZLIM[1], SDF_GRID_NZ)
        RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing="xy")
        dR = (R_grid[-1] - R_grid[0]) / (SDF_GRID_NR - 1)
        dZ = (Z_grid[-1] - Z_grid[0]) / (SDF_GRID_NZ - 1)

        for k, p in enumerate(self.plots):
            phi = self.phis[k]
            phi_n = self.phis_normalized[k]

            # surface cross-section
            try:
                cs = np.asarray(self.boozer_surface.surface.cross_section(phi_n)).reshape(-1, 3)
                cs_R = np.sqrt(cs[:, 0]**2 + cs[:, 1]**2)
                cs_Z = cs[:, 2]
                cs_R = np.append(cs_R, cs_R[0])
                cs_Z = np.append(cs_Z, cs_Z[0])
                p.addItem(pg.PlotCurveItem(_1d(cs_R), _1d(cs_Z),
                                           pen=pg.mkPen((30, 30, 30), width=1.5)))
            except Exception as e:
                proc0_print(f"surface.cross_section failed at phi/2pi={phi_n}: {e}")

            # SDF zero contour via IsocurveItem (matplotlib-free)
            try:
                D = self.sdf.eval(RR * np.cos(phi), RR * np.sin(phi), ZZ)  # (nZ, nR)
                iso = pg.IsocurveItem(data=np.asarray(D).T, level=0.0,
                                      pen=pg.mkPen((136, 136, 136), width=1))
                tr = QtGui.QTransform()
                tr.translate(R_grid[0], Z_grid[0])
                tr.scale(dR, dZ)
                iso.setTransform(tr)
                p.addItem(iso)
            except Exception as e:
                proc0_print(f"sdf.eval failed at phi/2pi={phi_n}: {e}")

            # x-point marker(s) + invariant lines (top & bottom) only at phi=0
            xyz, ok = evaluate_at_phi(self.curve, phi_n)
            if ok:
                Rx = np.sqrt(xyz[0]**2 + xyz[1]**2)
                if k == 0:
                    for Zc, zsgn in [(xyz[2], 1.0), (-xyz[2], -1.0)]:
                        p.addItem(pg.ScatterPlotItem(x=[Rx], y=[Zc], symbol="x", size=12,
                                                     pen=pg.mkPen((214, 39, 40), width=2),
                                                     brush=None))
                        if self.invariant_dirs is not None:
                            seen = set()
                            for vR, vZ in self.invariant_dirs:
                                vZc = zsgn * vZ
                                ang = round(np.arctan2(vZc, vR) % np.pi, 6)
                                if ang in seen:
                                    continue
                                seen.add(ang)
                                line = pg.InfiniteLine(pos=(Rx, Zc),
                                                       angle=np.degrees(np.arctan2(vZc, vR)),
                                                       pen=pg.mkPen((0, 0, 0, 80), width=1))
                                p.addItem(line)
                else:
                    p.addItem(pg.ScatterPlotItem(x=[Rx], y=[xyz[2]], symbol="x", size=12,
                                                 pen=pg.mkPen((214, 39, 40), width=2),
                                                 brush=None))

    def _draw_axis_ellipses(self):
        for it in self.ellipse_items:
            self.plots[0].removeItem(it)
        self.ellipse_items = []
        if self.axis_ellipse is None:
            return
        a_xyz, ok = evaluate_at_phi(self.axis_curve, self.phis_normalized[0])
        if not ok:
            return
        R_a = np.sqrt(a_xyz[0]**2 + a_xyz[1]**2)
        Z_a = a_xyz[2]
        sa, sb, ang = self.axis_ellipse
        p = self.plots[0]
        for j in range(1, N_ELLIPSES + 1):
            S = self.ellipse_scale * j / N_ELLIPSES
            ex, ez = _ellipse_curve(R_a, Z_a, sa, sb, ang, S)
            curve = pg.PlotCurveItem(_1d(ex), _1d(ez), pen=pg.mkPen((214, 39, 40), width=1))
            p.addItem(curve)
            self.ellipse_items.append(curve)
        dot = pg.ScatterPlotItem(x=[R_a], y=[Z_a], symbol="o", size=6,
                                 brush=pg.mkBrush(214, 39, 40), pen=None)
        p.addItem(dot)
        self.ellipse_items.append(dot)

    # --- controls ------------------------------------------------------------
    def _on_tmax(self):
        try:
            v = float(self.ed_tmax.text())
            assert v > 0
            self.tmax_value = v
            self._set_status(f"tmax set to {v:g}")
        except Exception:
            self._set_status("Invalid tmax")

    def _on_tol(self):
        try:
            v = float(self.ed_tol.text())
            assert v > 0
            self.tol_value = v
            self._set_status(f"tol set to {v:g}")
        except Exception:
            self._set_status("Invalid tol")

    def _on_escale(self):
        try:
            v = float(self.ed_escale.text())
            assert v > 0
            self.ellipse_scale = v
            self._draw_axis_ellipses()
            self._set_status(f"ellipse scale set to {v:g}")
        except Exception:
            self._set_status("Invalid ellipse scale")

    def _on_clear(self):
        for it in self.poincare_items:
            it.getViewBox().removeItem(it) if it.getViewBox() else None
        self.poincare_items = []
        for it in self.click_items.values():
            if it.getViewBox():
                it.getViewBox().removeItem(it)
        self.click_items = {}
        self.trace_count = 0          # restart the per-trace colour cycle
        self._set_status("Cleared traces.")

    def _fit_view(self):
        # Linked axes: setting the master's range sets them all. With aspect
        # locked, pyqtgraph expands one axis to keep 1:1, so the full window is
        # always contained.
        self.plots[0].setRange(xRange=DEFAULT_XLIM, yRange=DEFAULT_ZLIM, padding=0)

    def _on_reset(self):
        self._fit_view()
        self._set_status("View reset.")

    # --- click / tracing -----------------------------------------------------
    def _on_click(self, event):
        if self.worker is not None and self.worker.is_alive():
            self._set_status("Tracing already running.")
            return
        scene_pos = event.scenePos()
        for idx, p in enumerate(self.plots):
            if p.vb.sceneBoundingRect().contains(scene_pos):
                pt = p.vb.mapSceneToView(scene_pos)
                R, Z = float(pt.x()), float(pt.y())
                phi = self.phis[idx]
                x0, y0, z0 = R * np.cos(phi), R * np.sin(phi), Z
                if idx in self.click_items and self.click_items[idx].getViewBox():
                    self.click_items[idx].getViewBox().removeItem(self.click_items[idx])
                marker = pg.ScatterPlotItem(x=[R], y=[Z], symbol="+", size=14,
                                            pen=pg.mkPen((31, 119, 180), width=2), brush=None)
                p.addItem(marker)
                self.click_items[idx] = marker
                self._set_status(f"Tracing phi/2pi={self.phis_normalized[idx]:.3f}: "
                                 f"(R={R:.4f}, Z={Z:.4f}), tmax={self.tmax_value:g}, tol={self.tol_value:g}")
                self.worker = threading.Thread(
                    target=self._trace, args=(x0, y0, z0, self.tmax_value, self.tol_value),
                    daemon=True)
                self.worker.start()
                return

    def _trace(self, x0, y0, z0, tmax_val, tol_val):
        t0 = time.time()
        try:
            X0, Y0, Z0 = np.array([x0]), np.array([y0]), np.array([z0])
            all_hits = []
            for sign in [+1.0, -1.0]:
                _, hits = compute_fieldlines_xyz(sign * self.bs, X0, Y0, Z0,
                                                 tmax=tmax_val, tol=tol_val, comm=comm_world,
                                                 phis=list(self.phis))
                all_hits.append(hits)
            self.result_queue.put(("ok", {"hits": all_hits, "elapsed": time.time() - t0}))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def _poll_worker(self):
        try:
            kind, payload = self.result_queue.get_nowait()
        except queue.Empty:
            return
        if kind == "error":
            self._set_status(f"Tracing failed: {payload}")
            return
        # One colour per trace (per click): all dots from this IC share it,
        # and each new trace advances to the next palette colour.
        brush = pg.mkBrush(*PALETTE[self.trace_count % len(PALETTE)])
        total = 0
        for phi_idx, p in enumerate(self.plots):
            Rs, Zs = [], []
            for hits in payload["hits"]:
                for j in range(len(hits)):
                    rows = hits[j]
                    if rows is None or rows.size == 0:
                        continue
                    m = rows[:, 1] == phi_idx
                    sel = rows[m, 2:]
                    if sel.size == 0:
                        continue
                    Rs.extend(np.sqrt(sel[:, 0]**2 + sel[:, 1]**2))
                    Zs.extend(sel[:, 2])
            if not Rs:
                continue
            spi = pg.ScatterPlotItem(x=_1d(Rs), y=_1d(Zs), size=POINT_SIZE, pen=None, brush=brush)
            p.addItem(spi)
            self.poincare_items.append(spi)
            total += len(Rs)
        self.trace_count += 1
        self._set_status(f"Done in {payload['elapsed']:.2f}s, {total} Poincare dots.")


def main():
    if DATA_FILE is None:
        print("Usage: python fieldline_gui_qt.py <data_file>", file=sys.stderr)
        sys.exit(2)
    app = pg.mkQApp("Fieldline tracer")
    gui = FieldlineGUI()
    gui.show()
    pg.exec()


if __name__ == "__main__":
    main()
