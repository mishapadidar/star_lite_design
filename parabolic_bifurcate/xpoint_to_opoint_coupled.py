#!/usr/bin/env python3
"""
Convert an X-point into an O-point with a COUPLED continuation that simultaneously PINS the
existing O-point (magnetic axis).

There is a single, shared set of auxiliary (PF) coils -- one parameter vector
    mu = (I_1..I_N, r_1..r_N, Z)
-- and the device has two periodic field lines in the resulting total field B = B_modular + B_aux(mu):

  * the O-point (magnetic axis): its monodromy trace must stay CONSTANT (= its loaded value), so the
    axis is not destroyed while we reshape the field;
  * the X-point: its monodromy trace is driven down across tr = 2 by CONTINUATION, turning it into a
    second O-point.

Both fixed points are represented by SingularPeriodicFieldline objects, but we do NOT call their
solver.  We use them ONLY to build the per-point residual + Jacobian via singular_field_line_residual
(trace mode), then assemble ONE coupled system and take a custom pseudoinverse (minimum-norm) Newton
step over the joint unknown vector

    x = [ curve_O dofs, L_O,   curve_X dofs, L_X,   I_1..I_N ]   (the aux radii r_k and height Z are held)

Each point's block is underdetermined on its own (one trace equation + a square field-line block, but
N shared currents); stacking the two blocks gives one underdetermined system, solved by lstsq
(minimum-norm).  After every Newton step the SHARED currents are written back into BOTH
SingularPeriodicFieldline mu vectors.

Continuation in the X-point target trace uses an adaptive step with backtracking: a step that fails to
converge is halved and retried from the last good state.

Output: a standard BiotSavart (modular + PF coils as ordinary Coil objects) together with both fixed-
point curves -> load(out) = [BiotSavart, o_curve, x_curve].

Usage:
    xpoint_to_opoint_coupled.py <design_json> <num_aux> [target_|trace|=1.9] [initial_step=0.1]

<design_json> is the rich design layout [boozer_surfaces, iota_Gs, axes, xpoints, ...]: the O-point
is axes[0] (dat[2][0]) and the X-point is xpoints[0] (dat[3][0]), each a PeriodicFieldLine.
"""
import sys
from pathlib import Path

import numpy as np
from simsopt._core import load, save
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import CurveLength, CurveXYZFourierSymmetries, CurveXYZFourier, curves_to_vtk

from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.singularperiodicfieldline import (
    SingularPeriodicFieldline, singular_field_line_residual, _mu_names, _CURRENT_SCALE)

# --------------------------------------------------------------------------- CLI
if len(sys.argv) < 3:
    raise SystemExit(__doc__)
p = Path(sys.argv[1])
num_aux = int(sys.argv[2])
TARGET_TRACE_MAG = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
STEP0 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.1              # grid increment for target traces
STEP_MIN = 1e-3
NEWTON_TOL = 1e-10
NEWTON_MAXITER = 40
if num_aux < 3:
    raise SystemExit("num_aux must be >= 3 (two trace constraints on the shared currents + slack)")

residue = lambda tr: (2.0 - tr) / 4.0
kind = lambda tr: "O-point (elliptic)" if abs(tr) < 2.0 else "X-point (hyperbolic)"

# --------------------------------------------------------------------------- load
dat = load(str(p))
o_obj = dat[2][0]                                     # magnetic axis (O-point)
x_obj = dat[3][0]                                     # X-point
coils = getattr(o_obj, "biotsavart", getattr(x_obj, "biotsavart", None)).coils

# ----------------------------------------- project a loaded fixed-point curve onto the canonical
# CurveXYZFourierSymmetries (2*order+1 quadpoints over [0, 1/nfp) so get_stellsym_mask matches; and
# stellsym=True if the curve is in fact stellarator symmetric). Mirrors xpoint_to_opoint.py.
STELLSYM_TOL = 1e-9


def _stellsym_defect(curve):
    qp = np.asarray(curve.quadpoints)
    refl = CurveXYZFourierSymmetries(np.mod(-qp, 1.0), curve.order, curve.nfp,
                                     curve.stellsym, ntor=curve.ntor)
    refl.set_dofs(curve.get_dofs())
    return float(np.max(np.abs(curve.gamma() - np.array([1.0, -1.0, -1.0]) * refl.gamma())))


def make_fl_curve(xc):
    qp = np.linspace(0.0, 1.0 / xc.nfp, 2 * xc.order + 1, endpoint=False)
    scale = float(np.max(np.abs(xc.gamma())))
    if xc.stellsym:
        c = CurveXYZFourierSymmetries(qp, xc.order, xc.nfp, True, ntor=xc.ntor)
        c.set_dofs(xc.get_dofs())
    elif _stellsym_defect(xc) <= STELLSYM_TOL * scale:
        c = CurveXYZFourierSymmetries(qp, xc.order, xc.nfp, True, ntor=xc.ntor)
        for nm in ([f'xc({i})' for i in range(xc.order + 1)]
                   + [f'ys({i})' for i in range(1, xc.order + 1)]
                   + [f'zs({i})' for i in range(1, xc.order + 1)]):
            c.set(nm, xc.get(nm))
    else:
        c = CurveXYZFourierSymmetries(qp, xc.order, xc.nfp, xc.stellsym, ntor=xc.ntor)
        c.set_dofs(xc.get_dofs())
    return c


o_curve = make_fl_curve(getattr(o_obj, "curve", o_obj))
x_curve = make_fl_curve(getattr(x_obj, "curve", x_obj))

# ----------------------------------------- shared auxiliary-coil parameters (mirror boozer_singular.py)
stellsym_aux = True
try:
    stellsym_aux = bool(dat[0][0].surface.stellsym)
except Exception:
    pass
max_Z = np.max([np.abs(c.curve.gamma()[:, 2]) for c in coils])
AUX_COIL_DISTANCE_THRESHOLD = 0.15
radii0 = np.linspace(0.5, 1.5, num_aux)
mu0 = np.concatenate([np.zeros(num_aux), radii0, [max_Z + AUX_COIL_DISTANCE_THRESHOLD]])
nmu = len(mu0)
names = _mu_names(nmu)

# ----------------------------------------- SingularPeriodicFieldlines used ONLY to build the systems
# (we never call their .run_code / .solve; we call singular_field_line_residual directly).
SPF_OPTS = {'monodromy_constraint': 'trace', 'target_trace': 2.0, 'verbose': False}
fl_O = SingularPeriodicFieldline(BiotSavart(coils), o_curve, mu0.copy(), options=dict(SPF_OPTS),
                                 stellsym_aux=stellsym_aux)
fl_X = SingularPeriodicFieldline(BiotSavart(coils), x_curve, mu0.copy(), options=dict(SPF_OPTS),
                                 stellsym_aux=stellsym_aux)

# ----------------------------------------- joint state (curve dofs live in fl_*.curve; we set them)
mu = mu0.copy()                                       # shared aux parameters
L_O = float(CurveLength(o_curve).J())
L_X = float(CurveLength(x_curve).J())
amps = lambda: "  ".join(f"{names[k]}={mu[k] * _CURRENT_SCALE:+.3e}A" for k in range(num_aux))


def point_system(fl, length, mu, target_trace):
    """Build one point's trace-mode residual + Jacobian (NO solve). Returns (b, J_cL, J_mu, J_fl, M):
      b    : masked residual [field-line eqns (+ y=0 BC if non-stellsym), trace row];
      J_cL : d b / d(curve dofs, length)   -- all masked rows (used in the coupled system);
      J_mu : d b / d mu                    -- all masked rows, all nmu columns;
      J_fl : the FIELD-LINE system Jacobian -- rows = the field-line equations (+ the y=0 boundary
             condition when the curve is non-stellsym), columns = the masked field-line unknowns
             (curve dofs) + the length L. The trace row and the mu columns are NOT included.
      M    : monodromy matrix.
    """
    r, J, M = singular_field_line_residual(
        fl.curve, fl.curve_tm, length, fl.biotsavart, mu, fl.monodromy_fns,
        stellsym=fl.stellsym_aux, monodromy_constraint='trace', target_trace=target_trace)
    n_c = fl.curve.get_dofs().size
    row_mask = fl.get_stellsym_mask(tail=1)           # field-line rows (+ BC) + the trace row
    Jm = J[row_mask]
    fl_mask = fl.get_stellsym_mask(tail=0)            # field-line rows only (+ BC if non-stellsym)
    J_fl = J[:fl_mask.size][fl_mask][:, :n_c + 1]     # field-line eqns x (masked curve dofs + L)
    return r[row_mask], Jm[:, :n_c + 1], Jm[:, n_c + 1:], J_fl, M


def combined_newton(target_trace_X, tol=NEWTON_TOL, maxiter=NEWTON_MAXITER):
    """Custom pseudoinverse Newton on the COUPLED system: O-point trace pinned at tr0_O, X-point
    trace driven to target_trace_X, sharing the auxiliary currents I_1..I_N. Updates the global
    state (curve dofs, L_O, L_X, mu) in place; writes the shared currents into BOTH fl_*.mu each
    step. Returns (success, ||r||_inf, cond(J), iters, tr(M_O), tr(M_X))."""
    global L_O, L_X, mu
    norm, condJ, cond_fl_O, cond_fl_X, M_O, M_X = np.inf, np.nan, np.nan, np.nan, None, None
    it = 0
    while it < maxiter:
        b_O, JcL_O, Jmu_O, Jfl_O, M_O = point_system(fl_O, L_O, mu, tr0_O)            # O trace pinned
        b_X, JcL_X, Jmu_X, Jfl_X, M_X = point_system(fl_X, L_X, mu, target_trace_X)   # X trace continued
        b = np.concatenate([b_O, b_X])
        norm = np.linalg.norm(b, np.inf)
        nO, nX = JcL_O.shape[1], JcL_X.shape[1]        # (curve_O + L_O), (curve_X + L_X)
        rO, rX = b_O.shape[0], b_X.shape[0]
        # Per-point field-line system condition numbers (field-line eqns + BC if non-stellsym, vs the
        # masked field-line unknowns + L) -- see point_system; the trace row / mu columns are excluded.
        cond_fl_O = np.linalg.cond(Jfl_O)              # magnetic axis (O-point) field-line system
        cond_fl_X = np.linalg.cond(Jfl_X)              # X/O point field-line system
        # Joint unknowns: [cL_O (nO), cL_X (nX), I_1..I_N (num_aux)]. Only the currents (first
        # num_aux mu columns) are free; the radii/height are held. Assemble (and condition) J every
        # iteration -- INCLUDING the converged one -- so cond(J) is reported even when no Newton step
        # is taken (e.g. the baseline solve, which converges immediately).
        J = np.zeros((rO + rX, nO + nX + num_aux))
        J[:rO, :nO] = JcL_O
        J[:rO, nO + nX:] = Jmu_O[:, :num_aux]
        J[rO:, nO:nO + nX] = JcL_X
        J[rO:, nO + nX:] = Jmu_X[:, :num_aux]
        condJ = np.linalg.cond(J)
        if norm <= tol:
            break
        dx, *_ = np.linalg.lstsq(J, b, rcond=None)     # minimum-norm Newton step (pseudoinverse)

        xO = np.concatenate([fl_O.curve.get_dofs(), [L_O]]) - dx[:nO]
        fl_O.curve.set_dofs(xO[:-1]); L_O = float(xO[-1])
        xX = np.concatenate([fl_X.curve.get_dofs(), [L_X]]) - dx[nO:nO + nX]
        fl_X.curve.set_dofs(xX[:-1]); L_X = float(xX[-1])
        mu = mu.copy(); mu[:num_aux] -= dx[nO + nX:]   # shared currents
        fl_O.mu = mu; fl_X.mu = mu                      # write shared mu back into BOTH points
        it += 1
    trO = float(M_O[0, 0] + M_O[1, 1])
    trX = float(M_X[0, 0] + M_X[1, 1])
    return norm <= tol, norm, condJ, cond_fl_O, cond_fl_X, it, trO, trX


def snapshot():
    return (fl_O.curve.get_dofs().copy(), fl_X.curve.get_dofs().copy(), L_O, L_X, mu.copy())


def restore(snap):
    global L_O, L_X, mu
    fl_O.curve.set_dofs(snap[0]); fl_X.curve.set_dofs(snap[1])
    L_O, L_X, mu = snap[2], snap[3], snap[4].copy()
    fl_O.mu = mu; fl_X.mu = mu


# ----------------------------------------- VTK output (one set per accepted continuation step)
VTK_DIR = p.parent / f"{p.stem}_x2o_coupled_vtk"
VTK_DIR.mkdir(exist_ok=True)
_manifest = []


def _full_loop(curve):
    full = CurveXYZFourierSymmetries(np.linspace(0, 1, max(200, 8 * curve.order), endpoint=False),
                                     curve.order, curve.nfp, curve.stellsym, ntor=curve.ntor)
    full.set_dofs(curve.get_dofs())
    return full


def _combined_coils():
    """Modular + auxiliary coils for the CURRENT shared mu (aux geometry fixed; only currents move)."""
    Z = float(mu[-1])
    base_curves, base_currents = [], []
    for k in range(num_aux):
        c = CurveXYZFourier(np.linspace(0, 1, 160, endpoint=False), 1)
        c.x = c.x * 0.0
        c.set('zc(0)', Z); c.set('xc(1)', float(mu[num_aux + k])); c.set('ys(1)', float(mu[num_aux + k]))
        base_curves.append(c)
        base_currents.append(ScaledCurrent(Current(float(mu[k])), _CURRENT_SCALE))
    if stellsym_aux:
        aux = coils_via_symmetries(base_curves, base_currents, 1, True)
    else:
        aux = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
    return list(coils) + aux


def write_step(idx, trO, trX):
    cset = _combined_coils()
    abs_cur = np.concatenate([np.full(c.curve.gamma().shape[0] + 1, abs(c.current.get_value()))
                              for c in cset])
    curves_to_vtk([c.curve for c in cset], str(VTK_DIR / f"coils_{idx:03d}"),
                  close=True, extra_data={'abs_current': abs_cur})
    curves_to_vtk([_full_loop(fl_O.curve), _full_loop(fl_X.curve)],
                  str(VTK_DIR / f"fixedpoints_{idx:03d}"), close=True)
    _manifest.append(f"{idx:03d}  trO={trO:+.6f}  trX={trX:+.6f}  R_X={residue(trX):+.4f}")


def save_state(trace):
    """Save the CURRENT state as [standard BiotSavart (modular + PF coils), O-point curve,
    (former) X-point curve] to <stem>_x2o_coupled_trX<|trace|>.json. Returns the path."""
    out = p.parent / f"{p.stem}_x2o_coupled_trX{abs(trace):.3f}.json"
    save([BiotSavart(_combined_coils()), fl_O.curve, fl_X.curve], str(out))
    return out


def fp_RZ(fl):
    """(R, Z) of a fixed point at the phi=0 Poincare section, from its curve's first quadpoint."""
    g = np.asarray(fl.curve.gamma())[0]
    return float(np.hypot(g[0], g[1])), float(g[2])


# ----------------------------------------- initial monodromies (mu currents are zero => modular field)
_, _, _, _, M_O0 = point_system(fl_O, L_O, mu, 2.0)
_, _, _, _, M_X0 = point_system(fl_X, L_X, mu, 2.0)
tr0_O = float(M_O0[0, 0] + M_O0[1, 1])                # PINNED for the whole continuation
tr0_X = float(M_X0[0, 0] + M_X0[1, 1])
print(f"O-point (axis):  tr={tr0_O:+.6f}  R={residue(tr0_O):+.4f}  {kind(tr0_O)}  (held constant)")
print(f"X-point:         tr={tr0_X:+.6f}  R={residue(tr0_X):+.4f}  {kind(tr0_X)}")
if abs(tr0_O) >= 2.0:
    print(f"WARNING: the 'O-point' axis has |tr|={abs(tr0_O):.4f} >= 2 (not elliptic).")
if abs(tr0_X) <= 2.0:
    print(f"WARNING: the 'X-point' has |tr|={abs(tr0_X):.4f} <= 2 (not hyperbolic).")
target_end = float(np.sign(tr0_X)) * abs(TARGET_TRACE_MAG)
print(f"continuation: X-point tr {tr0_X:+.4f} -> {target_end:+.4f}, O-point tr pinned at {tr0_O:+.4f}"
      f"  ({num_aux} shared aux coils, stellsym_aux={stellsym_aux})\n")

# ----------------------------------------- baseline: solve at (tr0_O, tr0_X) -> currents ~ 0
# Each accepted state records [trace_X, cond_combined, cond_magnetic_axis, cond_XO_point], plus the
# (R, Z) of both fixed points -> [trace_X, R_axis, Z_axis, R_track, Z_track] (to track bifurcations).
cond_rows, pos_rows = [], []
ok, norm, condJ, cflO, cflX, it, trO, trX = combined_newton(tr0_X)
if not ok:
    raise SystemExit(f"baseline coupled solve failed (||r||_inf={norm:.2e})")
print(f"baseline  trO={trO:+.6f}  trX={trX:+.6f}  iter={it:2d}  ||r||={norm:.1e}  "
      f"cond(J)={condJ:.2e}  cond(fl_axis)={cflO:.2e}  cond(fl_X)={cflX:.2e}  {amps()}")
cond_rows.append([trX, condJ, cflO, cflX])
pos_rows.append([trX, *fp_RZ(fl_O), *fp_RZ(fl_X)])
write_step(0, trO, trX)
print(f"  >>> start (tr0); wrote {save_state(tr0_X)}")

# ----------------------------------------- continuation on a GRID of target traces (multiples of
# STEP0), e.g. STEP0=0.01, tr0=2.865 -> 2.86, 2.85, ..., 2.00, ..., target. The tr=2 bifurcation and
# the exact target are always included. A json is saved at every grid target; if a step fails to
# converge it is bisected toward the last good state (backtracking) until it does or stalls.
sgn = float(np.sign(tr0_X))
m0, mt = abs(tr0_X), abs(target_end)
k_lo = int(np.floor(mt / STEP0)) + 1            # smallest k with k*STEP0 > |target|
k_hi = int(np.ceil(m0 / STEP0)) - 1             # largest  k with k*STEP0 < |tr0|
mags = {k * STEP0 for k in range(k_lo, k_hi + 1)}        # grid multiples strictly inside (|target|, |tr0|)
if mt < 2.0 < m0:
    mags.add(2.0)                               # always stop on the tr=2 bifurcation
mags.add(mt)                                    # finish exactly on the target (may be off-grid)
grid_targets = [sgn * m for m in sorted(mags, reverse=True)]    # descending |trace| = march order

tr_curr = tr0_X
n_steps = n_back = 0
stalled = False
for tr_target in grid_targets:
    if stalled:
        break
    sub = tr_target                              # value attempted (bisected toward tr_curr on failure)
    while abs(tr_curr - tr_target) > 1e-12 and not stalled:
        snap = snapshot()
        ok, norm, condJ, cflO, cflX, it, trO, trX = combined_newton(sub)
        if ok:
            tr_curr = sub
            if abs(tr_curr - tr_target) < 1e-12:        # reached the grid target -> record + save
                n_steps += 1
                print(f"  ok    trX={trX:+.6f}  R_X={residue(trX):+.4f}  trO={trO:+.6f}  iter={it:2d}  "
                      f"||r||={norm:.1e}  cond(J)={condJ:.2e}  cond(fl_axis)={cflO:.2e}  "
                      f"cond(fl_X)={cflX:.2e}")
                cond_rows.append([trX, condJ, cflO, cflX])
                pos_rows.append([trX, *fp_RZ(fl_O), *fp_RZ(fl_X)])
                write_step(n_steps, trO, trX)
                print(f"  >>> wrote {save_state(tr_curr)}")
            sub = tr_target                              # (re)attempt the full remaining jump
        else:
            n_back += 1
            restore(snap)
            sub = 0.5 * (tr_curr + sub)                  # backtrack: bisect toward the last good state
            print(f"  fail  -> backtrack to tr_try={sub:+.6f}  (||r||={norm:.1e})")
            if abs(sub - tr_curr) < STEP_MIN:
                print(f"\nSTALLED at trX={tr_curr:+.6f} (R_X={residue(tr_curr):+.4f}); "
                      f"gap |{abs(sub - tr_curr):.1e}| < {STEP_MIN:.0e}.")
                stalled = True

# ----------------------------------------- final report
_, _, _, _, M_O = point_system(fl_O, L_O, mu, tr0_O)
_, _, _, _, M_X = point_system(fl_X, L_X, mu, tr_curr)
trO, trX = float(M_O[0, 0] + M_O[1, 1]), float(M_X[0, 0] + M_X[1, 1])
print(f"\nFINAL  O-point tr={trO:+.6f} ({kind(trO)})   X-point tr={trX:+.6f} ({kind(trX)})   "
      f"({n_steps} accepted steps, {n_back} backtracks)")
print(f"shared aux currents: {amps()}")
print("held aux geometry:   " + "  ".join(f"{names[k]}={mu[k]:+.4f}" for k in range(num_aux, nmu)))

(VTK_DIR / "manifest.txt").write_text("# idx  trO  trX  R_X\n" + "\n".join(_manifest) + "\n")
print(f"wrote {len(_manifest)} VTK step(s) to {VTK_DIR}/")

# condition numbers vs trace, one row per accepted continuation state (baseline included)
out_cond = p.parent / f"{p.stem}_x2o_coupled_cond.txt"
np.savetxt(out_cond, np.asarray(cond_rows), fmt="%.10e",
           header="trace  cond_combined  cond_magnetic_axis  cond_XO_point")
print(f"wrote {out_cond}")

# fixed-point (R, Z) positions vs the X-point trace T, one row per accepted state (for tracking
# bifurcations): trace_X, R_axis, Z_axis (pinned O-point), R_track, Z_track (continued X/O-point).
out_pos = p.parent / f"{p.stem}_x2o_coupled_positions.txt"
np.savetxt(out_pos, np.asarray(pos_rows), fmt="%.10e",
           header="trace_X  R_axis  Z_axis  R_track  Z_track")
print(f"wrote {out_pos}")

# ----------------------------------------- save: standard BiotSavart (modular + PF coils) + both curves
out = save_state(trX)
print(f"\nwrote {out}  (standard BiotSavart + O-point curve + (former) X-point curve)")
