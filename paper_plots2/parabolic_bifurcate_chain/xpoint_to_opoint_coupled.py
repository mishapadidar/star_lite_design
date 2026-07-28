#!/usr/bin/env python3
"""
Turn ONE X-point of a fixed stellarator design into an O-point by a COUPLED continuation that PINS the
trace of EVERY other fixed point (all O-points AND all other X-points) at the same time.

Input: the grouped fixed-point json written by mk_manifolds.py,

    load(in) = [BiotSavart, {type: [curve, ...]}]   (type in elliptic / hyperbolic / parabolic / ...)

The file is fixed: ./hyperbolic_SN_x2o_coupled_trX1.500_fixedpoints.json (next to this script), and all
output is written into ./output/. Its BiotSavart carries
the modular coils plus the circular PF (aux) coils; we hold the MODULAR coils as a FIXED base field and
REUSE the PF coils as the shared auxiliary set, parametrized by

    mu = (I_1..I_N, r_1..r_N, Z)                     (only the currents I_k are free; geometry held),

initialized to the loaded PF currents / radii / height (plus any extra coils at zero current) so the
baseline reproduces the loaded field exactly. Every loaded fixed point becomes a
SingularPeriodicFieldline in the total field B = B_base + B_aux(mu); we never call its solver, only build
its trace-mode residual + Jacobian via singular_field_line_residual.

ONE large coupled system is assembled over the joint unknown

    x = [ (curve_1 dofs, L_1), ..., (curve_P dofs, L_P),  I_1..I_N ]      P = #fixed points

with one block per fixed point (its field-line equations -- square in its own curve dofs + L -- plus a
single trace row) and a shared block of current columns. Each block is underdetermined on its own (the
N shared currents); stacking all P blocks gives one underdetermined system solved by lstsq (minimum
norm). The shared currents are written back into ALL points' mu after every Newton step.

Trace targets: every O-point and every NON-continued X-point is PINNED at its loaded trace (so the
elliptic O-points keep a CONSTANT trace); the single CONTINUED X-point (the hyperbolic point CLOSEST to
the magnetic axis) has its trace driven downward across tr=2 by continuation on a grid of target traces
(multiples of STEP0), one Newton solve per grid step.

Output (per grid target, and at the end): a standard BiotSavart (base + new aux coils) with the
fixed-point curves grouped by their CURRENT classification ->
    load(out) = [BiotSavart, {type: [curve, ...]}]   (re-loadable by this script or mk_manifolds.py).

Usage:
    xpoint_to_opoint_coupled.py [num_extra_PF=0] [target_|trace|=1.0] [STEP0=0.1]

The circular PF (aux) coils already in the json are REUSED (their currents / radii / height initialize
the shared mu); num_extra_PF new circular coils are added between the existing radii -- or, if too many,
at regular radii beyond the outermost -- at zero initial current. The continued X-point is always the
hyperbolic point CLOSEST to the magnetic axis.
"""
import sys
from pathlib import Path

import numpy as np
from simsopt._core import load, save
from simsopt.field import BiotSavart, Current, Coil
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import CurveLength, CurveXYZFourierSymmetries, CurveXYZFourier, curves_to_vtk

from star_lite_design.utils.singularperiodicfieldline import (
    SingularPeriodicFieldline, singular_field_line_residual, _mu_names, _CURRENT_SCALE)

# --------------------------------------------------------------------------- CLI + input json
# The design json is fixed and sits next to this script; all output goes into ./output/.
# The positional arguments tune the continuation (all optional; see the module docstring).
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)
NUM_EXTRA = int(sys.argv[1]) if len(sys.argv) > 1 else 0             # additional PF coils to add
TARGET_TRACE_MAG = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
STEP0 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1            # grid increment for target traces
NEWTON_TOL = 1e-10
NEWTON_MAXITER = 40
p = HERE / "hyperbolic_SN_x2o_coupled_trX1.500_fixedpoints.json"
if not p.exists():
    raise SystemExit(f"{p} not found")

residue = lambda tr: (2.0 - tr) / 4.0
kind = lambda tr: "O-point (elliptic)" if abs(tr) < 2.0 else "X-point (hyperbolic)"
type_of = lambda tr: ("parabolic" if abs(abs(tr) - 2.0) <= 1e-6
                      else ("elliptic" if abs(tr) < 2.0 else "hyperbolic"))

# --------------------------------------------------------------------------- load field + fixed points
# Grouped format written by mk_manifolds.py: [BiotSavart, {type: [curve, ...]}]. Roles (O vs X) are
# assigned later, by trace.
field, groups = load(str(p))
fp_curves = [c for curves in groups.values() for c in curves]     # flatten all fixed-point curves
if not fp_curves:
    raise SystemExit("no fixed-point curves found in the input")


# Split the loaded coils into the FIXED base (modular) coils and the auxiliary circular PF coils (planar,
# order-1 CurveXYZFourier). The aux coils' (current, radius, height) INITIALIZE the shared mu; the base
# coils are held fixed, so at the loaded currents the baseline reproduces the loaded field.
def _is_aux_coil(coil):
    cur = coil.curve
    if type(cur).__name__ != "CurveXYZFourier" or getattr(cur, "order", 99) != 1:
        return False
    g = np.asarray(cur.gamma())
    R = np.hypot(g[:, 0], g[:, 1])
    return float(R.std()) < 1e-6 and float(g[:, 2].std()) < 1e-6   # circular + planar (constant R and Z)


base_coils = [c for c in field.coils if not _is_aux_coil(c)]      # modular: FIXED base field
aux_coils = [c for c in field.coils if _is_aux_coil(c)]           # circular PF coils: reused as mu
if not aux_coils:
    raise SystemExit("no auxiliary circular PF coils (planar order-1) found in the loaded field")


def _aux_rzI(coil):
    g = np.asarray(coil.curve.gamma())
    return (float(np.hypot(g[:, 0], g[:, 1]).mean()), float(g[:, 2].mean()),
            float(coil.current.get_value()) / _CURRENT_SCALE)     # mu current = physical current / scale


aux_rzI = sorted((_aux_rzI(c) for c in aux_coils), key=lambda t: t[0])   # by radius
loaded_r = np.array([t[0] for t in aux_rzI])
loaded_I = np.array([t[2] for t in aux_rzI])
Zc = float(np.mean([t[1] for t in aux_rzI]))                      # common coil height
coils = base_coils                                               # SPF field + _combined_coils base


def _extra_radii(base_r, n):
    """n ADDITIONAL PF-coil radii: first the midpoints BETWEEN consecutive existing radii, then, if more
    are needed, at regular spacing BEYOND the outermost existing coil."""
    r = np.sort(np.asarray(base_r, dtype=float))
    out = list(0.5 * (r[:-1] + r[1:]))[:n]                        # between the existing coils
    step = float(np.mean(np.diff(r))) if r.size > 1 else 0.25
    k = 1
    while len(out) < n:                                           # beyond the outermost, regular spacing
        out.append(float(r[-1] + step * k)); k += 1
    return np.array(out[:n])

# ----------------------------------------- shared auxiliary-coil parameters: the loaded PF coils
# (currents + radii + common height) PLUS NUM_EXTRA new coils interleaved between / beyond them at zero
# current. Only the currents are free; radii and height are held. mu = (I_1..I_N, r_1..r_N, Z).
stellsym_aux = False                                  # loaded PF coils are stored individually (not mirrored)
num_base = len(loaded_r)
num_aux = num_base + NUM_EXTRA
extra_r = _extra_radii(loaded_r, NUM_EXTRA) if NUM_EXTRA > 0 else np.array([])
radii = np.concatenate([loaded_r, extra_r])
currents = np.concatenate([loaded_I, np.zeros(NUM_EXTRA)])   # extra coils start at zero current
mu0 = np.concatenate([currents, radii, [Zc]])
nmu = len(mu0)
names = _mu_names(nmu)
mu = mu0.copy()                                       # shared aux parameters (state)
amps = lambda: "  ".join(f"{names[k]}={mu[k] * _CURRENT_SCALE:+.3e}A" for k in range(num_aux))

SPF_OPTS = {'monodromy_constraint': 'trace', 'target_trace': 2.0, 'verbose': False}

# ----------------------------------------- one record per fixed point (SPF used ONLY to build systems)
# Each record: {curve, fl, L, role ('O'/'X'), tr0 (loaded trace, pin target), continued (bool)}.
points = []
for c in fp_curves:            # loaded curves already carry the canonical 2*order+1 quadpoints over [0, 1/nfp)
    fl = SingularPeriodicFieldline(BiotSavart(coils), c, mu0.copy(), options=dict(SPF_OPTS),
                                   stellsym_aux=stellsym_aux)
    points.append({"curve": c, "fl": fl, "L": float(CurveLength(c).J()),
                   "role": None, "tr0": None, "continued": False})
P = len(points)
if num_aux < P + 1:
    raise SystemExit(f"num_aux = {num_aux} (={num_base} loaded + {NUM_EXTRA} extra PF coils) must be "
                     f">= P+1 = {P + 1}: {P} trace constraints on the shared currents + slack. Add more "
                     f"PF coils (raise num_extra_PF).")


def point_system(rec, mu, target_trace):
    """Assemble ONE fixed point's trace-mode residual and Jacobian at the current state (NO solve).

    singular_field_line_residual returns the full per-point system, whose
      rows = [ field-line equations (+ y=0 BC if non-stellsym) , one trace row ]   (trace row LAST)
      cols = [ curve dofs (n_c) , length L (1) , shared coil params mu (nmu) ].
    Stellarator-symmetry-redundant rows are then dropped with two masks:
      row_mask = get_stellsym_mask(tail=1)  -> kept rows INCLUDING the trace row   (k = row_mask.sum())
      fl_mask  = get_stellsym_mask(tail=0)  -> kept FIELD-LINE rows only            (f = fl_mask.sum()).

    Returns (b, J_cL, J_mu, J_fl, M):
      b    : (k,)         masked residual = [kept field-line eqns (+ BC), trace-target residual].
      J_cL : (k, n_c+1)   d b / d(curve dofs, L)  -- the curve-and-length columns of the kept rows.
      J_mu : (k, nmu)     d b / d mu              -- the shared-coil columns of the kept rows.
      J_fl : (f, n_c+1)   field-line-only Jacobian (kept field-line rows x curve dofs + L; the trace
             row and the mu columns are EXCLUDED) -- used only for the per-point conditioning diagnostic.
      M    : (2, 2)       monodromy matrix; its trace tr = M[0,0]+M[1,1] classifies the point.

    combined_newton stacks b / J_cL / J_mu over all P points into the coupled system (J_cL block-diagonal,
    J_mu the shared-current columns); J_fl and M are diagnostics only.
    """
    fl = rec["fl"]
    r, J, M = singular_field_line_residual(
        fl.curve, fl.curve_tm, rec["L"], fl.biotsavart, mu, fl.monodromy_matrix_all,
        stellsym=fl.stellsym_aux, monodromy_constraint='trace', target_trace=target_trace)
    # J is the full residual Jacobian, laid out as
    #   rows = [ field-line eqns (+ y=0 BC if non-stellsym) , trace row ]   (the trace row is LAST)
    #   cols = [ curve dofs (n_c) , length L (1) , mu (nmu) ]
    n_c = fl.curve.get_dofs().size
    n_cL = n_c + 1                                    # count of curve-dof + L columns (mu columns follow)

    # coupled-system rows: stellsym mask over ALL rows, trace row INCLUDED (its length matches J's rows,
    # so it indexes J directly). These rows give the returned residual b and Jacobian blocks J_cL, J_mu.
    row_mask = fl.get_stellsym_mask(tail=1)
    Jm = J[row_mask]

    # per-point conditioning uses the FIELD-LINE block alone (trace row and mu columns excluded). Its
    # mask (tail=0) does NOT cover the trace row, so it is one row shorter than J -> slice J down to the
    # field-line block FIRST so the boolean mask lines up, THEN keep the curve-dof + L columns.
    fl_mask = fl.get_stellsym_mask(tail=0)
    n_fl = fl_mask.size                              # number of field-line rows in J (mask LENGTH, not its True-count)
    fl_block = J[:n_fl]                              # the field-line block; the trailing trace row is dropped
    J_fl = fl_block[fl_mask][:, :n_cL]
    return r[row_mask], Jm[:, :n_cL], Jm[:, n_cL:], J_fl, M


def combined_newton(cont_target, tol=NEWTON_TOL, maxiter=NEWTON_MAXITER):
    """Custom pseudoinverse Newton on the COUPLED system over ALL P fixed points: each pinned point's
    trace held at its tr0, the single continued X-point's trace driven to cont_target, sharing the
    auxiliary currents I_1..I_N. Updates each record's curve dofs + L and the global mu in place; writes
    the shared currents into EVERY point's mu each step. Returns (success, ||r||_inf, cond(J), cond_fls,
    iters, traces) with cond_fls/traces aligned with `points`."""
    global mu
    norm, condJ = np.inf, np.nan
    cond_fls = [np.nan] * P
    Ms = [None] * P
    it = 0
    ncs = None
    while it < maxiter:
        blocks = []                                   # (b, JcL, Jmu) per point
        for i, rec in enumerate(points):
            target = cont_target if rec["continued"] else rec["tr0"]
            b_i, JcL_i, Jmu_i, Jfl_i, M_i = point_system(rec, mu, target)
            blocks.append((b_i, JcL_i, Jmu_i))
            cond_fls[i] = np.linalg.cond(Jfl_i)
            Ms[i] = M_i
        b = np.concatenate([blk[0] for blk in blocks])
        norm = np.linalg.norm(b, np.inf)
        rows = [blk[0].shape[0] for blk in blocks]
        ncs = [blk[1].shape[1] for blk in blocks]     # (curve_i dofs + L_i) per point
        Ccurve = sum(ncs)
        # Joint unknowns: [curve_i+L_i blocks (block-diagonal), shared currents I_1..I_N]. Only the
        # currents (first num_aux mu columns) are free; radii/height are held. Assemble + condition J
        # every iteration (incl. the converged one) so cond(J) is reported even at the baseline.
        J = np.zeros((sum(rows), Ccurve + num_aux))
        roff = coff = 0
        for (b_i, JcL_i, Jmu_i), r_i, nc_i in zip(blocks, rows, ncs):
            J[roff:roff + r_i, coff:coff + nc_i] = JcL_i
            J[roff:roff + r_i, Ccurve:] = Jmu_i[:, :num_aux]
            roff += r_i
            coff += nc_i
        condJ = np.linalg.cond(J)
        if norm <= tol:
            break
        dx, *_ = np.linalg.lstsq(J, b, rcond=None)     # minimum-norm Newton step (pseudoinverse)
        coff = 0
        for rec, nc_i in zip(points, ncs):
            x_i = np.concatenate([rec["fl"].curve.get_dofs(), [rec["L"]]]) - dx[coff:coff + nc_i]
            rec["fl"].curve.set_dofs(x_i[:-1])
            rec["L"] = float(x_i[-1])
            coff += nc_i
        mu = mu.copy()
        mu[:num_aux] -= dx[Ccurve:]                     # shared currents
        for rec in points:
            rec["fl"].mu = mu                           # write shared mu back into EVERY point
        it += 1
    traces = [float(M[0, 0] + M[1, 1]) for M in Ms]
    return norm <= tol, norm, condJ, cond_fls, it, traces


# ----------------------------------------- VTK output (one set per accepted continuation step)
VTK_DIR = OUT_DIR / f"{p.stem}_x2o_coupled2_vtk"
VTK_DIR.mkdir(exist_ok=True)
_manifest = []


def _full_loop(curve):
    full = CurveXYZFourierSymmetries(np.linspace(0, 1, max(200, 8 * curve.order), endpoint=False),
                                     curve.order, curve.nfp, curve.stellsym, ntor=curve.ntor)
    full.set_dofs(curve.get_dofs())
    return full


def _combined_coils():
    """Base + auxiliary coils for the CURRENT shared mu (aux geometry fixed; only currents move)."""
    Z = float(mu[-1])
    base_curves, base_currents = [], []
    for k in range(num_aux):
        c = CurveXYZFourier(np.linspace(0, 1, 160, endpoint=False), 1)
        c.x = c.x * 0.0
        c.set('zc(0)', Z); c.set('xc(1)', float(mu[num_aux + k])); c.set('ys(1)', float(mu[num_aux + k]))
        base_curves.append(c)
        base_currents.append(ScaledCurrent(Current(float(mu[k])), _CURRENT_SCALE))
    aux = [Coil(c, I) for c, I in zip(base_curves, base_currents)]   # PF coils stored individually
    return list(coils) + aux


def write_step(idx, traces):
    cset = _combined_coils()
    abs_cur = np.concatenate([np.full(c.curve.gamma().shape[0] + 1, abs(c.current.get_value()))
                              for c in cset])
    curves_to_vtk([c.curve for c in cset], str(VTK_DIR / f"coils_{idx:03d}"),
                  close=True, extra_data={'abs_current': abs_cur})
    curves_to_vtk([_full_loop(rec["fl"].curve) for rec in points],
                  str(VTK_DIR / f"fixedpoints_{idx:03d}"), close=True)
    _manifest.append(f"{idx:03d}  " + "  ".join(f"tr{i}={t:+.5f}" for i, t in enumerate(traces)))


def save_state(trace, traces):
    """Save the CURRENT state as [standard BiotSavart (base + new aux coils), {type: [curve, ...]}],
    the fixed-point curves grouped by their CURRENT classification, to
    <stem>_x2o_coupled2_trX<|trace|>.json. Returns the path."""
    grouped = {}
    for rec, tr in zip(points, traces):
        grouped.setdefault(type_of(tr), []).append(rec["fl"].curve)
    out = OUT_DIR / f"{stem}_x2o_coupled2_trX{abs(trace):.3f}.json"
    save([BiotSavart(_combined_coils()), grouped], str(out))
    return out


stem = p.stem.replace("_fixedpoints", "")             # cleaner output names

# ----------------------------------------- initial traces (mu = loaded PF coils + zero-current extra
# coils => the loaded field) and role assignment: |tr|<2 -> O-point (pinned), |tr|>=2 -> X-point.
for rec in points:
    _, _, _, _, M0 = point_system(rec, mu, 2.0)
    rec["tr0"] = float(M0[0, 0] + M0[1, 1])
    rec["role"] = "O" if abs(rec["tr0"]) < 2.0 else "X"
o_curves_in = [rec for rec in points if rec["role"] == "O"]
x_curves_in = [rec for rec in points if rec["role"] == "X"]
if not x_curves_in:
    raise SystemExit("no hyperbolic X-point among the loaded fixed points to continue")

# choose the X-point to continue: the hyperbolic point CLOSEST to the magnetic axis, where the axis is
# taken as the elliptic O-point nearest the midplane (smallest |Z|). _rz(rec) is the fixed point's
# (R, Z). Its trace is then continued downward.
def _rz(rec):
    g = np.asarray(rec["fl"].curve.gamma())[0]
    return float(np.hypot(g[0], g[1])), float(g[2])


ax_R, ax_Z = _rz(min(o_curves_in, key=lambda r: abs(_rz(r)[1]))) if o_curves_in else (0.0, 0.0)
cont = min(x_curves_in, key=lambda r: (_rz(r)[0] - ax_R) ** 2 + (_rz(r)[1] - ax_Z) ** 2)   # closest X to axis
print(f"X-point closest to the magnetic axis selected for continuation: R={_rz(cont)[0]:.4f} "
      f"Z={_rz(cont)[1]:.4f}  (magnetic axis at R={ax_R:.4f} Z={ax_Z:.4f})")
cont["continued"] = True
tr0_cont = cont["tr0"]

print(f"{P} fixed points loaded ({len(o_curves_in)} O-point(s), {len(x_curves_in)} X-point(s)):")
for i, rec in enumerate(points):
    tag = " <== CONTINUED" if rec["continued"] else " (pinned)"
    print(f"  [{i}] {rec['role']}  tr={rec['tr0']:+.6f}  R={residue(rec['tr0']):+.4f}  "
          f"{kind(rec['tr0'])}{tag}")
target_end = float(np.sign(tr0_cont)) * abs(TARGET_TRACE_MAG)
print(f"continuation: continued X-point tr {tr0_cont:+.4f} -> {target_end:+.4f}; all other traces "
      f"pinned  ({num_aux} shared PF coils = {num_base} loaded + {NUM_EXTRA} extra, "
      f"stellsym_aux={stellsym_aux})\n")

# ----------------------------------------- baseline: solve at the loaded traces -> currents ~ 0
cond_rows = []                                        # [tr_cont, cond(J), cond_fl_0, ..., cond_fl_{P-1}]
ok, norm, condJ, cond_fls, it, traces = combined_newton(tr0_cont)
if not ok:
    raise SystemExit(f"baseline coupled solve failed (||r||_inf={norm:.2e})")
cont_idx = points.index(cont)
pin_drift = max((abs(t - rec["tr0"]) for rec, t in zip(points, traces) if not rec["continued"]),
                default=0.0)
print(f"baseline  trX={traces[cont_idx]:+.6f}  iter={it:2d}  ||r||={norm:.1e}  cond(J)={condJ:.2e}  "
      f"max|pin drift|={pin_drift:.1e}  {amps()}")
cond_rows.append([traces[cont_idx], condJ] + list(cond_fls))
write_step(0, traces)
print(f"  >>> start (tr0); wrote {save_state(tr0_cont, traces)}")

# ----------------------------------------- continuation on a GRID of target traces (multiples of STEP0)
# for the single continued X-point; the tr=2 bifurcation and the exact target are always included. Each
# grid target is one Newton solve from the previous converged state, and a json is saved at every one.
sgn = float(np.sign(tr0_cont))
m0, mt = abs(tr0_cont), abs(target_end)
k_lo = int(np.floor(mt / STEP0)) + 1
k_hi = int(np.ceil(m0 / STEP0)) - 1
mags = {k * STEP0 for k in range(k_lo, k_hi + 1)}
if mt < 2.0 < m0:
    mags.add(2.0)
mags.add(mt)
grid_targets = [sgn * m for m in sorted(mags, reverse=True)]

tr_curr = tr0_cont
n_steps = 0
for tr_target in grid_targets:
    ok, norm, condJ, cond_fls, it, traces = combined_newton(tr_target)
    if not ok:
        print(f"\nSTOPPED at trX={tr_curr:+.6f}: step to {tr_target:+.6f} did not converge "
              f"(||r||={norm:.1e}).")
        break
    tr_curr = tr_target
    n_steps += 1
    pin_drift = max((abs(t - rec["tr0"]) for rec, t in zip(points, traces)
                     if not rec["continued"]), default=0.0)
    print(f"  ok    trX={traces[cont_idx]:+.6f}  R_X={residue(traces[cont_idx]):+.4f}  "
          f"iter={it:2d}  ||r||={norm:.1e}  cond(J)={condJ:.2e}  "
          f"max|pin drift|={pin_drift:.1e}")
    cond_rows.append([traces[cont_idx], condJ] + list(cond_fls))
    write_step(n_steps, traces)
    print(f"  >>> wrote {save_state(tr_curr, traces)}")

# ----------------------------------------- final report
ok, norm, condJ, cond_fls, it, traces = combined_newton(tr_curr)
print(f"\nFINAL  continued X-point tr={traces[cont_idx]:+.6f} ({kind(traces[cont_idx])})   "
      f"({n_steps} accepted steps)")
for i, (rec, t) in enumerate(zip(points, traces)):
    tag = "CONTINUED" if rec["continued"] else "pinned   "
    print(f"  [{i}] {rec['role']} {tag}  tr0={rec['tr0']:+.6f} -> tr={t:+.6f}  ({kind(t)})")
print(f"shared aux currents: {amps()}")
print("held aux geometry:   " + "  ".join(f"{names[k]}={mu[k]:+.4f}" for k in range(num_aux, nmu)))

(VTK_DIR / "manifest.txt").write_text(
    "# idx  " + "  ".join(f"tr{i}" for i in range(P)) + "\n" + "\n".join(_manifest) + "\n")
print(f"wrote {len(_manifest)} VTK step(s) to {VTK_DIR}/")

# condition numbers vs the continued trace, one row per accepted state (baseline included)
out_cond = OUT_DIR / f"{stem}_x2o_coupled2_cond.txt"
np.savetxt(out_cond, np.asarray(cond_rows), fmt="%.10e",
           header="trace_cont  cond_combined  " + "  ".join(f"cond_fl_{i}" for i in range(P)))
print(f"wrote {out_cond}")

# ----------------------------------------- save: standard BiotSavart (base + aux) + grouped curves
out = save_state(traces[cont_idx], traces)
print(f"\nwrote {out}  (standard BiotSavart + fixed-point curves grouped by current type)")

