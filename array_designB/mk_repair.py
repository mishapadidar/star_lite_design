#!/usr/bin/env python3
"""Can the snowflake of a double device be repaired with its aux currents after a
Gaussian-process perturbation of the modular coils?

Usage:
    mk_repair.py <design_double_final_json> [--nsamples 128] [--sigmas-mm 1 5 10 20]
                 [--length-scale 0.5] [--first-seed 0]

For each realization (seed) the modular coils of the snowflake configuration are displaced by
sigma * d, with d the unit Gaussian-process sample of that seed (length scale L; the same unit
samples, per seed, as CoilPerturbationField, hence as mk_unfolding.py), one independent sample
per coil; the aux coils are NOT perturbed. The perturbed field has no field-period symmetry,
so the snowflake field line is re-solved on the FULL torus: SingularPeriodicFieldline on an
nfp = 1 curve, monodromy M = I over one full turn, with the aux CURRENTS as the only unknowns
besides the field line itself (the aux radii and height are held: the coils are built).

A realization is REPAIRED at a given sigma when that solve converges. The solve is continued
in sigma (1 mm -> 5 mm -> ...), each amplitude starting from the solution of the previous
one, with intermediate amplitudes inserted when a step does not converge.

With 3 aux coils the repairing currents are unique. With more, they form a family of
dimension N - 3, and the member with the smallest largest |current| is sought: from the
solution Newton lands on, steps are taken along the tangent of the family (the null space of
the Newton Jacobian) chosen by a small linear program (minimize t subject to |I + dI| <= t,
within a trust region), each followed by a Newton correction, until the largest |current| no
longer decreases. It is a local minimum, reached from the nominal currents.

Writes next to the design json:
    repair_currents.txt   one row per (seed, sigma): seed, sigma [mm], repaired (0/1), Newton
                          iterations, largest |aux current| [kA], then the N aux currents [kA]
    repair_fraction.txt   one row per current threshold [kA]: the fraction of the realizations
                          repaired with all |aux currents| below it, one column per sigma
and appends repair_frac_<I>kA_sigma<s>mm lines to summary.txt for I = 5, 10, 15, 20 kA (same
4-column format; unless already there). plot_repair.py makes the figure.
Needs a simsopt with the second derivatives of the field (venv_gradgradB).
"""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from simsopt._core import load
from simsopt.field import BiotSavart, Coil
from simsopt.geo import CurveLength, CurvePerturbed, CurveXYZFourierSymmetries

from star_lite_design.utils.discriminant import CoilPerturbationField
from star_lite_design.utils.singularperiodicfieldline import SingularPeriodicFieldline, _CURRENT_SCALE

parser = argparse.ArgumentParser(description="Repair of the snowflake with the aux currents under GP coil perturbations.")
parser.add_argument("design_json")
parser.add_argument("--nsamples", type=int, default=128, help="number of GP realizations (default 128)")
parser.add_argument("--sigmas-mm", type=float, nargs="+", default=[1.0, 5.0, 10.0, 20.0],
                    help="perturbation amplitudes in mm, in increasing order (default 1 5 10 20)")
parser.add_argument("--length-scale", type=float, default=0.5, help="GP correlation length L (default 0.5)")
parser.add_argument("--first-seed", type=int, default=0, help="seed of the first realization (default 0)")
parser.add_argument("--part", action="store_true",
                    help="only write the rows of this run (seeds first-seed .. first-seed + nsamples - 1) to "
                         "repair_parts/part_<first-seed>_<nsamples>.txt: the realizations are independent, so "
                         "a device can be split over several processes and merged afterwards")
parser.add_argument("--merge", action="store_true",
                    help="no computation: merge the repair_parts/ files into repair_currents.txt, "
                         "repair_fraction.txt and the summary lines (every seed 0 .. nsamples - 1 must be there)")
args = parser.parse_args()
sigmas = sorted(1e-3 * s for s in args.sigmas_mm)
KA = _CURRENT_SCALE / 1e3          # aux current in kA per unit of mu
THRESHOLDS = (5.0, 10.0, 15.0, 20.0)   # kA, for the summary lines
NEWTON = {'newton_tol': 1e-10, 'newton_maxiter': 25, 'verbose': False, 'monodromy_constraint': 'identity'}

p = Path(args.design_json)
PARTS = p.parent / 'repair_parts'


def finalize(rows, header):
    """Write repair_currents.txt (rows sorted by seed and sigma), repair_fraction.txt and the
    summary lines from the table of every realization. `header` is the two-line description of
    the columns (it carries N and the nominal current, read back by plot_repair.py)."""
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    n_aux = rows.shape[1] - 7
    sig_mm = np.unique(rows[:, 1])
    nreal = len(np.unique(rows[:, 0]))
    out = p.parent / 'repair_currents.txt'
    np.savetxt(out, rows, fmt=['%d', '%g', '%d', '%d', '%.6e', '%.4f', '%d'] + ['%.6e'] * n_aux, header=header)
    # fraction repaired with every |aux current| below a threshold, per sigma
    grid = np.arange(0.1, 40.0 + 1e-9, 0.1)   # kA; the exact values are in repair_currents.txt
    frac = np.zeros((len(grid), len(sig_mm)))
    for j, s in enumerate(sig_mm):
        sel = rows[np.isclose(rows[:, 1], s)]
        top = np.where(sel[:, 2] > 0, sel[:, 4], np.inf)
        frac[:, j] = [(top <= g).mean() for g in grid]
        med = np.nanmedian(sel[:, 4]) if sel[:, 2].any() else np.nan
        print(f"sigma = {s:g} mm: repaired {int(sel[:, 2].sum())} of {len(sel)} | largest |I| needed: median "
              f"{med:.2f} kA | repaired below " + ", ".join(f"{g:g} kA: {100 * (top <= g).mean():.0f}%" for g in THRESHOLDS))
    out_f = p.parent / 'repair_fraction.txt'
    np.savetxt(out_f, np.column_stack([grid, frac]), fmt=['%g'] + ['%.6f'] * len(sig_mm),
               header=f"fraction of the {nreal} realizations repaired with every |aux current| <= the threshold\n"
                      "threshold[kA] " + " ".join(f"sigma={s:g}mm" for s in sig_mm))
    print(f"wrote {out}, {out_f}")
    summary = p.parent / 'summary.txt'
    text = summary.read_text() if summary.is_file() else ""
    if 'repair_frac_' not in text:
        with open(summary, 'a') as fh:
            for j, s in enumerate(sig_mm):
                for g in THRESHOLDS:
                    name = f"repair_frac_{g:g}kA_sigma{s:g}mm"
                    fh.write(f"  {name:<30s} {frac[np.argmin(np.abs(grid - g)), j]:.6e}   {'n/a':>16s}   {'n/a':>16s}\n")
        print(f"appended repair fractions to {summary}")


if args.merge:
    files = sorted(PARTS.glob('part_*.txt'))
    if not files:
        raise SystemExit(f"mk_repair --merge: no part file in {PARTS}")
    rows = np.vstack([np.atleast_2d(np.loadtxt(f)) for f in files])
    with open(files[0]) as fh:
        header = "".join(line[2:] for line in fh if line.startswith("#")).rstrip("\n")
    seeds = np.unique(rows[:, 0]).astype(int)
    counts = [int((rows[:, 0] == s).sum()) for s in seeds]
    missing = sorted(set(range(args.nsamples)) - set(seeds.tolist()))
    if missing or len(set(counts)) != 1 or len(rows) != len(seeds) * counts[0]:
        raise SystemExit(f"mk_repair --merge: incomplete or duplicated parts in {PARTS}: {len(seeds)} seeds, "
                         f"missing {missing[:10]}{'...' if len(missing) > 10 else ''}, rows per seed {sorted(set(counts))}")
    print(f"merging {len(files)} part files: {len(seeds)} realizations x {counts[0]} amplitudes")
    finalize(rows, header)
    raise SystemExit(0)
boozer_surfaces, iota_Gs, axes, xpoints, sdf = load(p)
snowflakes = [x for x in xpoints if isinstance(x, SingularPeriodicFieldline)]
if not snowflakes:
    raise SystemExit(f"mk_repair: no snowflake (SingularPeriodicFieldline) in {p.name}")
fl = snowflakes[-1]
bs = fl.biotsavart                       # the MODULAR field of the snowflake configuration
c = fl.curve
mu0 = np.asarray(fl.mu, dtype=float)     # (I_1..I_N, r_1..r_N, Z)
N = (len(mu0) - 1) // 2
res_nom = fl.run_code(CurveLength(c).J())
print(f"snowflake: nfp={c.nfp}, {len(bs.coils)} modular coils, N={N} aux coils, nominal currents [kA] "
      f"{np.round(mu0[:N] * KA, 3)} (largest {np.abs(mu0[:N]).max() * KA:.3f})")

# The closed field line as an nfp = 1, non-stellarator-symmetric curve: same curve, fitted
# exactly (its Fourier modes are a subset of those of the nfp = 1 curve of order nfp * order).
order1 = c.order * c.nfp
q1 = np.linspace(0, 1, 2 * order1 + 1, endpoint=False)
ref = CurveXYZFourierSymmetries(q1, c.order, c.nfp, c.stellsym, ntor=c.ntor, dofs=c.dofs)
_c1 = CurveXYZFourierSymmetries(q1, order1, 1, False, ntor=c.ntor)
A = _c1.dgamma_by_dcoeff().reshape((-1, _c1.get_dofs().size))
dofs_nom, *_ = np.linalg.lstsq(A, ref.gamma().reshape(-1), rcond=None)
_c1.set_dofs(dofs_nom)
fit_err = np.abs(_c1.gamma() - ref.gamma()).max()
if fit_err > 1e-10:
    raise SystemExit(f"mk_repair: the full-torus fit of the field line is off by {fit_err:.2e}")

_shared = {}


def field_line(field, curve_dofs, mu):
    """A full-torus SingularPeriodicFieldline on `field`, the aux currents being the unknowns
    (fixed = dependent) and the aux radii / height held. The jitted monodromy function is
    compiled once and shared by every instance (same curve order, same Chebyshev grid)."""
    curve = CurveXYZFourierSymmetries(q1, order1, 1, False, ntor=c.ntor)
    curve.set_dofs(curve_dofs)
    f = SingularPeriodicFieldline(field, curve, np.array(mu, dtype=float), options=dict(NEWTON),
                                  stellsym_aux=fl.stellsym_aux)
    if 'mono' in _shared:
        f.monodromy_matrix_all = _shared['mono']
    else:
        _shared['mono'] = f.monodromy_matrix_all
    for k in range(N):
        f.fix(f'I{k + 1}')
    return f


def solve(field, state):
    """Newton solve from `state` = (curve dofs, length, mu). Returns (ok, new state, res, f)."""
    f = field_line(field, state[0], state[2])
    try:
        res = f.run_code(state[1])
    except Exception as e:   # e.g. a singular Jacobian far from a solution
        return False, state, {'iter': -1, 'error': repr(e)[:120]}, f
    ok = bool(res['success']) and np.all(np.isfinite(res['mu']))
    new = (f.curve.get_dofs().copy(), float(res['length']), np.asarray(res['mu'], dtype=float).copy())
    return ok, (new if ok else state), res, f


def minimize_largest_current(field, state, res, maxit=25):
    """N > 3: move along the (N - 3)-dimensional family of repairing currents to the member
    with the smallest largest |current| (see the module doc). Returns (state, steps taken)."""
    trust, steps = 0.25, 0
    for _ in range(maxit):
        I = state[2][:N]
        top = np.abs(I).max()
        Jm = np.asarray(res['jacobian'])               # active rows x (curve dofs, length, currents)
        _, s, Vt = np.linalg.svd(Jm)                   # full: Vt is (columns, columns)
        null = Vt[Jm.shape[0]:]                        # (k, columns): orthonormal basis of the null space
        if null.shape[0] == 0:
            break
        VI = null[:, -N:].T                            # (N, k): tangent of the family, current part
        k = VI.shape[1]
        # each coefficient is also bounded, so that a tangent direction that hardly changes the
        # currents cannot carry the field line far away
        amax = trust * top / max(np.abs(VI).max(axis=0).min(), 1e-9)
        # linear program in (a_1..a_k, t): minimize t, |I + VI a| <= t, |VI a| <= trust * top
        cost = np.r_[np.zeros(k), 1.0]
        A_ub = np.block([[VI, -np.ones((N, 1))], [-VI, -np.ones((N, 1))],
                         [VI, np.zeros((N, 1))], [-VI, np.zeros((N, 1))]])
        b_ub = np.r_[-I, I, np.full(2 * N, trust * top)]
        lp = linprog(cost, A_ub=A_ub, b_ub=b_ub, bounds=[(-amax, amax)] * k + [(0, None)], method="highs")
        if not lp.success or top - lp.x[-1] < 1e-4 * top:
            break                                      # no further first-order decrease
        step = null.T @ lp.x[:k]                       # (curve dofs, length, currents)
        nd = state[0].size
        mu_try = state[2].copy()
        mu_try[:N] += step[nd + 1:]
        trial = (state[0] + step[:nd], state[1] + step[nd], mu_try)
        ok, new, res_new, _ = solve(field, trial)
        if ok and np.abs(new[2][:N]).max() < top * (1 - 1e-6):
            state, res, steps = new, res_new, steps + 1
            trust = min(2.0 * trust, 1.0)              # the linear model held: allow a longer step
        else:
            trust *= 0.5                               # the linear model was too optimistic
            if trust < 1e-3:
                break
    return state, steps


# nominal snowflake on the full torus, unperturbed coils (must converge at once)
t0 = time.time()
ok, state_nom, res, _ = solve(BiotSavart(bs.coils), (dofs_nom, float(res_nom['length']), mu0))
if not ok:
    raise SystemExit("mk_repair: the unperturbed full-torus solve did not converge")
print(f"full-torus field line: order {order1}, fit error {fit_err:.1e}; unperturbed solve: {res['iter']} iteration(s), "
      f"currents moved by {np.abs(state_nom[2][:N] - mu0[:N]).max() * KA:.1e} kA, {time.time() - t0:.1f} s")

gamma_nom = ref.gamma().copy()


def shift_mm(state):
    """Largest displacement [mm] of the repaired snowflake field line from the nominal one:
    a small value shows that it is the same null, moved, and not another field line."""
    _c1.set_dofs(state[0])
    return 1e3 * np.linalg.norm(_c1.gamma() - gamma_nom, axis=1).max()


if N > 3:
    # Reference: with the UNPERTURBED coils, how low the largest |current| of the family goes
    # (the design is one member of it, not necessarily the one with the smallest largest current).
    t0 = time.time()
    st, steps = minimize_largest_current(BiotSavart(bs.coils), state_nom, res)
    print(f"unperturbed coils: smallest largest |I| of the family of snowflake currents = "
          f"{np.abs(st[2][:N]).max() * KA:.3f} kA (design: {np.abs(mu0[:N]).max() * KA:.3f} kA), currents [kA] "
          f"{np.round(st[2][:N] * KA, 3)}, field line moved by {shift_mm(st):.2f} mm, {steps} steps, {time.time() - t0:.1f} s")

rows = []
t0 = time.time()
for i, seed in enumerate(range(args.first_seed, args.first_seed + args.nsamples)):
    unit = CoilPerturbationField(bs, length_scale=args.length_scale, seed=seed)._unit_samples

    def perturbed_field(sigma):
        coils = []
        for coil, u in zip(bs.coils, unit):
            s = copy.copy(u)
            s._sample = [sigma * np.asarray(a) for a in u._sample]   # exactly sigma x the unit sample
            coils.append(Coil(CurvePerturbed(coil.curve, s), coil.current))
        return BiotSavart(coils)

    sig_prev, state, alive = 0.0, state_nom, True
    for sigma in sigmas:
        iters, steps = 0, 0
        if alive:
            # continuation from the last solved amplitude, halving the step when it fails
            targets, depth = [sigma], 0
            while targets:
                ok, new, res, _ = solve(perturbed_field(targets[0]), state)
                iters += max(int(res.get('iter', 0)), 0)
                if ok:
                    state, sig_prev = new, targets.pop(0)
                elif depth < 4:
                    targets.insert(0, 0.5 * (sig_prev + targets[0])); depth += 1
                else:
                    alive = False
                    break
            if alive and N > 3:
                state, steps = minimize_largest_current(perturbed_field(sigma), state, res)
        I = state[2][:N] * KA if alive else np.full(N, np.nan)
        rows.append([seed, 1e3 * sigma, int(alive), iters, np.abs(I).max() if alive else np.nan,
                     shift_mm(state) if alive else np.nan, steps] + list(I))
    if (i + 1) in (1, 2, 4, 8, 16, 32, 64) or (i + 1) == args.nsamples:
        last = rows[-len(sigmas):]
        print(f"  {i + 1:4d} realizations, {time.time() - t0:7.1f} s | last: "
              + ", ".join(f"{r[1]:g} mm: " + (f"{r[4]:.2f} kA" if r[2] else "not repaired") for r in last), flush=True)

rows = np.array(rows, dtype=float)
header = (f"GP length scale {args.length_scale}; aux currents only (radii and height held); N = {N}; "
          f"nominal largest |I| = {np.abs(mu0[:N]).max() * KA:.4f} kA\n"
          "seed sigma[mm] repaired newton_iterations largest_abs_I[kA] fieldline_shift[mm] minimax_steps "
          + " ".join(f"I{k + 1}[kA]" for k in range(N)))
if args.part:
    # one slice of the realizations of this device: written whole, then renamed, so that a
    # part file that exists is complete (mk_repair.py --merge assembles them)
    PARTS.mkdir(exist_ok=True)
    out = PARTS / f"part_{args.first_seed:04d}_{args.nsamples:04d}.txt"
    tmp = out.with_suffix('.tmp')
    np.savetxt(tmp, rows, fmt=['%d', '%g', '%d', '%d', '%.6e', '%.4f', '%d'] + ['%.6e'] * N, header=header)
    tmp.replace(out)
    print(f"wrote {out}")
else:
    finalize(rows, header)
