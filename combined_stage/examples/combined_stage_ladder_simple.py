#!/usr/bin/env python
r"""
In this example we run a whole continuation LADDER of combined stage-I/II
optimisations in one script: from a rotating-ellipse QA seed and planar
circular coils to a quasi-axisymmetric, double-null (DN) device at
beta = 1 %, with its coils.

One RUNG of the ladder is the combined stage of combined_stage_simple.py:
the VMEX boundary s and the coils c are optimised TOGETHER in one augmented-
Lagrangian (AL) solve,

    min_{s, c}  f_QS(s) + penalties(s, c)
    subject to  ((B_plasma(s) + B_coil(c))·n)_mn = 0      (B·n harmonics on the boundary)
                one field-strength condition               (fixes the coil-current scale)
                [+ the X-line is a field line, the boundary stays regular, ...]

Why a ladder and not one big solve: the full problem -- QA, an iota floor, a
magnetic well, two X-lines outside the plasma, finite beta, coil engineering
limits -- started from circles does not converge in one go. Its terms fight
each other, VMEX rejects big boundary steps, and the X-line does not exist yet
at the seed. So the problem is solved as a sequence of easier ones, each
seeded with the previous one's result (continuation):

    rung               boundary  what is ADDED on this rung
                       modes
    0 m1_iota0.15      m,|n|<=1  QS + aspect + coil engineering, B·n = 0 (vacuum), iota >= 0.15
    1 m2_iota0.25      <=2       more boundary freedom, iota >= 0.25
    2 m3_iota0.30      <=3       iota >= 0.30 (the final floor) + magnetic well >= 1 %
    3 dn_introduce     <=3       a FREE X-line (curve + length are dofs) that the AL makes a
                                 field line; hyperbolic (|tr M| >= 2.2), 4-12 cm outside the
                                 plasma, coils >= 22 cm from it; smooth boundary-regularity limit
    4 dn_converge      <=2       X-line flat (|z - <z>| <= 1 cm); B·n block much stiffer,
                                 smaller boundary steps
    5 beta0.3_G_band   <=2       beta 0.3 %: virtual casing for B_plasma, Boozer-G field
                                 strength, B·n as a BAND |h| <= 3e-3, Mercier stability
    6 beta1_G_band     <=2       beta 1 %

Hand-over between rungs: rung k writes <outdir>/<k>_<name>/ with input.final
(the VMEX boundary), coils_final.json, xpoint_final.json (DN rungs) and
wout_final.nc; rung k+1 starts from exactly these files. Nothing else is
handed over: the AL multipliers/penalties and escalated weights start fresh on
every rung (the objective changed, so the old multipliers are not meaningful).
A rung whose wout_final.nc exists is skipped, so re-running the script after a
crash or a time limit resumes at the first unfinished rung.

Inside a rung there are four kinds of mechanism (the same as in
combined_stage_vmex.py):

* AL CONSTRAINT BLOCKS -- B·n harmonics, the field-strength scalar, the X-line
  field-line equations (equalities), the boundary regularity and the B·n band
  (inequalities). Met to a tolerance at a FINITE penalty: the multipliers
  carry the force (Nocedal & Wright 17.4, utils/augmented_lagrangian.py).
* FIXED-WEIGHT OBJECTIVE TERMS -- f_QS (QS residual + aspect ratio), coil
  arclength variation, the X-line flatness "pull".
* ESCALATED PENALTIES -- one-sided limits (coil length, curvature, coil-coil
  distance, iota floor, well, Mercier, X-line hyperbolicity / distance /
  flatness, ...). A fixed weight loses against the growing AL terms, so after
  every outer iteration the weight of each limit still violated by > 1 % is
  multiplied by 10, up to a cap (array/boozer_all.py practice).
* HARD GUARDS -- trial boundaries that self-intersect / pinch (boundary guard)
  or that VMEX cannot solve are never accepted: they return a quadratic
  barrier around the last good point, so the line search backs off.

Between DN rungs an X-LINE GATE stops the ladder when the X-line it hands over
is not what a double null is for (a periodic, hyperbolic field line outside
the plasma); see run_ladder.py.

This is the c16_i0.30_A8_w1 device of tools/make_hsx_scan.py (HSX size: nfp 4,
R0 1.2 m, a 0.15 m, B0 1 T, 16 coils), with the same settings but smaller
iteration budgets. The production pipeline is run_ladder.py (one subprocess
per rung) + combined_stage_vmex.py (one rung, from a YAML config) +
ladders/hsx_scan/*.yaml. Left out here for brevity: the |B| ripple on the
axis, the conformal vessel, the bootstrap current, checkpoint/resume INSIDE a
rung and the geometry snapshots.

Run on a whole compute node (one evaluation = one VMEX solve + adjoint; the
budgets below take roughly 8-12 h, the full production budgets ~2.5 days):

    cd <run dir>
    sbatch --cpus-per-task=32 --mem=128G --time=2-00:00:00 --output=slurm_%j.out \
        <this dir>/run_example.sbatch <this dir>/combined_stage_ladder_simple.py
    # end-to-end check, 1 outer x 3 inner per rung, gate off:
    sbatch ... run_example.sbatch <this dir>/combined_stage_ladder_simple.py --smoke --no-xline-gate
"""

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import vmex
import yaml
from scipy.optimize import minimize
from vmex import optimize as vopt
from vmex.core import implicit as vmex_implicit
from simsopt._core import load, save
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (ArclengthVariation, CurveCurveDistance, CurveLength, LpCurveCurvature, MeanSquaredCurvature,
                         create_equally_spaced_curves)
from simsopt.objectives import QuadraticPenalty, Weight
from star_lite_design.utils.augmented_lagrangian import AugmentedLagrangian, BandConstraint, SubsetConstraint
from star_lite_design.utils.current_bound import CurrentBound
from star_lite_design.utils.vmex_combined_stage import (BoundaryRegularityConstraint, CoilPlasmaDistance,
                                                        CoilXLineDistance, PlasmaCoilInterface, VmexPlasma,
                                                        VmexQuasisymmetry, VmexTermCost, WeightedSum,
                                                        boundary_regularity, fix_self_symmetric_coil_parity,
                                                        interface_check, net_poloidal_current_output)
from star_lite_design.utils.vmex_double_null import (FreeXLine, FreeXLineHyperbolicity, XLineFieldLineConstraint,
                                                     XpointPlasmaDistance, free_xline_from_plasma)

# Two pieces of the production driver are reused as they are (combined_stage/ is the parent directory):
#   XLineFlatness   mean(max(|z - mean z| - threshold, 0)^2) along the X-line, with its gradient
#   jacobi_scaling  per-dof optimiser step D_j = step / ||d constraints / d x_j||, see "Scaled variables" below
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from combined_stage_vmex import XLineFlatness, jacobi_scaling  # noqa: E402

# Seed boundary: HSX-size QA rotating ellipse, nfp = 4, R0 = 1.2 m, a = 0.15 m (aspect 8), B0 = 1 T, vacuum
SEED = Path(__file__).resolve().parent / "inputs" / "input.qa_nfp4_R1p2_A8"

# Equilibrium resolution during the optimisation (boundary harmonics above it are dropped) and radial grid:
MPOL = 6
NTOR = 6
NS = 16

# Virtual-casing / interface grid on the boundary, per field period, and the virtual-casing accuracy:
VC_NPHI = 32
VC_NTHETA = 32
VC_DIGITS = 3

# Plasma objective: quasi-axisymmetry (helicity m = 1, n = 0) on 8 surfaces s in [0.1, 0.9], aspect ratio held at
# ASPECT_TARGET in the vacuum rungs (the beta rungs hold whatever aspect they start from):
QS_SURFACES = np.linspace(0.1, 0.9, 8)
ASPECT_TARGET = 8.0
WELL_FLOOR = 0.01            # magnetic well depth (V'(0) - V'(1)) / V'(0) >= 1 %, and V'' <= 0 everywhere

# Coils: NCOILS planar circles per half field period (nfp 4, stellarator symmetric -> 2 * 4 * NCOILS = 16 coils),
# CurveXYZFourier order ORDER, radius R1_FACTOR x the seed minor radius (3.5 a leaves room for the X-lines and a vessel)
NCOILS = 2
ORDER = 6
R1_FACTOR = 3.5

# Coil engineering limits, each an ESCALATED one-sided penalty: name -> (limit, starting weight)
ENGINEERING = dict(
    length=(7.0, 1e3),        # m, per coil
    curvature=(10.0, 1e-3),   # 1/m, L2 norm of the curvature above the limit
    msc=(20.0, 1e-2),         # 1/m^2, mean squared curvature
    coil_coil=(0.08, 1e7),    # m
    coil_plasma=(0.22, 1e4),  # m, to the MOVING VMEX boundary
    current=(5e5, 1e6),       # A, 4/3 of the 375 kA per coil that B0 = 1 T needs
)
ARCLENGTH_WEIGHT = 1e-3       # uniform quadrature-point spacing along each coil (fixed weight)

# Double null: one free X-line per field period (its stellarator image is the second null)
DN = dict(
    order=10,                 # CurveXYZFourierSymmetries order of the X-line
    offset=0.06,              # m: the initial X-line is the top of every boundary cross-section raised by this
    hyperbolicity_margin=0.2, hyperbolicity_weight=1e-3,   # |tr M| >= 2 + margin (X-type, not O-type)
    distance_min=0.04, distance_max=0.12, distance_weight=1e2,   # m, X-line to plasma boundary (signed: inside < 0)
    coil_distance_min=0.22, coil_distance_weight=1e4,      # m, coils to X-line (room for a conformal vessel)
    flatness_threshold=0.01, flatness_weight=1e3,          # m, max |z - mean z| along the X-line (escalated)
    flatness_pull_weight=1e2,                              # fixed pull on the whole deviation: flatter is better
)

# Escalation of the one-sided penalties after each outer iteration:
ESCALATION_FACTOR = 10.0
ESCALATION_TOL = 1e-2         # relative violation that triggers it

# Hard boundary guard: reject trial boundaries that self-intersect or whose poloidal speed |dx/dtheta| drops below
# GUARD_MIN_SPEED x its cross-section median (a pinched boundary = degenerate VMEC Jacobian), BEFORE solving VMEX
GUARD_MIN_SPEED = 0.05
# Smooth version of the same as an AL inequality (DN rungs): |dx/dtheta| / cross-section mean >= this
REGULARITY_MIN_SPEED = 0.15
BARRIER_RETREAT = 0.5

# X-line gate (run_ladder.py): after every DN rung but the first, |tr M| >= 2, field-line residual <= 1e-2 and
# X-line-boundary distance > 0
XLINE_GATE = dict(min_trace=2.0, max_residual=1e-2, min_distance=0.0)

# Settings every rung starts from; a rung below overrides only what it changes.
DEFAULTS = dict(
    max_mode=2,               # boundary modes varied: m, |n| <= max_mode
    iota_floor=None,          # one-sided floor on min |iota| (escalated)
    iota_weight=10.0,
    well=False,               # magnetic well >= WELL_FLOOR (escalated, weight 10)
    double_null=False,        # free X-line + its field-line constraint and limits
    flatness=False,           # X-line flatness (only once the X-line exists)
    regularity=False,         # boundary-regularity AL inequality
    beta=None,                # finite beta: pressure seed, virtual casing, Boozer G, B·n band, Mercier
    bn_mpol=6, bn_ntor=9,     # constrained B·n harmonics m <= bn_mpol, |n| <= bn_ntor
    outer=4,                  # AL outer iterations (budget of the rung)
    inner=50,                 # L-BFGS-B iterations per inner solve (production: 100)
    penalty=1e3,              # starting AL penalty rho of every block ...
    penalty_blocks={},        # ... or of a block by name, e.g. {"bnormal": 1e5}
    penalty_growth=2.0,       # rho *= this when a block misses its tolerance
    penalty_max=1e6,
    eta0=1e-3,                # first feasibility tolerance (|residual|_inf), scalar or per block name
    omega0=1e-2,              # first inner gradient tolerance
    ctol=1e-5,                # final feasibility tolerance, scalar or per block name
    gtol=1e-6,                # final |grad L|_inf
    plasma_factor=1.0,        # extra factor on the boundary dofs' optimiser step (< 1: smaller boundary steps)
    escalation_cap=1e4,       # an escalated weight never exceeds this x its starting value
)
REGULARITY_TOL = {"default": 1e-3, "boundary_regularity": 5e-3}   # the speed-ratio rows live at ~1e-2
STIFF = dict(penalty_max=1e8, escalation_cap=100.0)              # v3 scans: cap escalation, let rho grow further
BETA_TOL = dict(eta0={"bnormal_band": 1e-4, "net_poloidal_current": 1e-4, "fieldline": 3e-4, "boundary_regularity": 5e-3},
                ctol={"bnormal_band": 1e-4, "net_poloidal_current": 1e-5, "fieldline": 3e-4, "boundary_regularity": 5e-3})

RUNGS = [
    # vacuum climb: more boundary modes and a higher iota floor rung by rung
    dict(name="m1_iota0.15", max_mode=1, iota_floor=0.15, outer=3, ctol=1e-4),
    dict(name="m2_iota0.25", max_mode=2, iota_floor=0.25, outer=4, ctol=1e-4),
    dict(name="m3_iota0.30", max_mode=3, iota_floor=0.30, well=True, outer=4, ctol=1e-4),
    # double null: make the X-line a field line first, then flatten and converge it (fewer modes, smaller steps)
    dict(name="dn_introduce", max_mode=3, iota_floor=0.30, iota_weight=1e3, well=True, double_null=True,
         regularity=True, outer=4, eta0=REGULARITY_TOL, ctol={**REGULARITY_TOL, "default": 1e-4}),
    dict(name="dn_converge", max_mode=2, iota_floor=0.30, iota_weight=1e3, well=True, double_null=True, flatness=True,
         regularity=True, plasma_factor=0.1, penalty_blocks={"bnormal": 1e5}, outer=8, eta0=REGULARITY_TOL,
         ctol={**REGULARITY_TOL, "default": 1e-5}, **STIFF),
    # finite beta in two steps
    dict(name="beta0.3_G_band", beta=0.003, max_mode=2, iota_floor=0.30, iota_weight=1e3, well=True, double_null=True,
         flatness=True, regularity=True, bn_mpol=8, plasma_factor=0.1, penalty_blocks={"bnormal_band": 1e4}, outer=4,
         **BETA_TOL, **STIFF),
    dict(name="beta1_G_band", beta=0.01, max_mode=2, iota_floor=0.30, iota_weight=1e3, well=True, double_null=True,
         flatness=True, regularity=True, bn_mpol=8, plasma_factor=0.1, penalty_blocks={"bnormal_band": 1e4}, outer=6,
         **BETA_TOL, **STIFF),
]
BNORMAL_BAND = 3e-3           # beta rungs: |B·n harmonic| / B_ref <= this instead of = 0

#######################################################
# End of input parameters.
#######################################################


def at_resolution(inp):
    """The optimisation resolution (MPOL, NTOR, NS) applied to a VmecInput."""
    inp = inp.change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=2 * NTOR + 4)
    return replace(inp, ns_array=np.array([NS]), ftol_array=np.array([1e-11]), niter_array=np.array([20000]))


def solve(inp, restart_wout=None):
    """VMEX solve; if the cold solve fails, hot-restart from the previous rung's wout.

    A rung's seed is the previous rung's final boundary, which the optimiser reached by warm starts only; strongly
    shaped ones can raise INITIAL JACOBIAN CHANGED SIGN when solved cold, and a hot restart solves them in seconds.
    """
    try:
        return vopt.solve_equilibrium(inp)
    except Exception:  # noqa: BLE001
        if restart_wout is None:
            raise
        print(f"  cold solve failed; hot restart from {restart_wout}", flush=True)
        return vopt.solve_equilibrium(inp, restart_from=str(restart_wout))


def make_beta_seed(src, beta, dst, restart_wout):
    """Finite-beta copy of the namelist ``src``: p(s) = PRES_SCALE (1 - s), PRES_SCALE calibrated so that VMEX's
    volume-averaged beta equals ``beta`` (tools/make_beta_seed.py). The boundary is the previous rung's."""
    inp = vmex.VmecInput.from_file(str(src))
    am = np.zeros(21)
    am[:2] = [1.0, -1.0]
    scale = 2.0 * beta / (2.0 * 4e-7 * np.pi)         # <p> = PRES_SCALE / 2 for p ~ 1 - s, guess B0 ~ 1 T
    for it in range(5):
        trial = replace(inp, am=am, pres_scale=scale, pmass_type="power_series")
        b = float(solve(at_resolution(trial), restart_wout).wout.betatotal)
        print(f"  beta seed {it}: PRES_SCALE {scale:9.1f} Pa -> beta {b:.5f} (target {beta})", flush=True)
        if abs(b / beta - 1.0) < 2e-3:
            trial.to_indata(str(dst))
            return dst
        scale *= beta / b                             # beta is ~linear in the pressure scale
    raise SystemExit(f"beta seed: no PRES_SCALE gives beta {beta} within 5 solves")


def build_plasma(rung, namelist, restart_wout):
    """VmexPlasma: the boundary dofs s and the cached VMEX equilibrium + objective rows they define.

    Returns the plasma and the names of the VMEX terms that get their own escalated weight (they are excluded from
    f_QS and added as VmexTermCost terms in run_rung).
    """
    inp = at_resolution(vmex.VmecInput.from_file(str(namelist)))
    eq0 = solve(inp, restart_wout)
    w0 = eq0.wout
    print(f"  seed equilibrium: aspect {float(w0.aspect):.3f}, beta {float(w0.betatotal):.3e}, "
          f"iota (axis, edge) = ({float(w0.iotaf[0]):+.3f}, {float(w0.iotaf[-1]):+.3f})", flush=True)

    # f_QS = 0.5 |r|^2 of these VMEX (function, target, weight) terms. The term NAME (function name) is how
    # VmexTermCost and VmexQuasisymmetry(exclude=...) find a term's rows.
    aspect_target = ASPECT_TARGET if rung["beta"] is None else float(w0.aspect)
    terms = [(vopt.QuasisymmetryRatioResidual(QS_SURFACES, helicity_m=1, helicity_n=0), 0.0, 1.0),
             (vopt.aspect_ratio, aspect_target, 1.0)]
    escalated_terms = []
    if rung["iota_floor"] is not None:
        floor = float(rung["iota_floor"])

        def iota_floor(state, runtime):
            # one-sided: zero while min |iota| >= floor
            return jnp.maximum(floor - vopt.min_abs_iota(state, runtime), 0.0)
        terms.append((iota_floor, 0.0, 1.0))            # unit weight inside VMEX; the real weight is escalated outside
        escalated_terms.append(("iota_floor", rung["iota_weight"], floor))
    if rung["well"]:
        from vmex.core.optimize import _field_chain

        def magnetic_well_floor(state, runtime):
            # V'(s) must fall everywhere (V'' <= 0) and the well depth (V'(0) - V'(1)) / V'(0) must reach WELL_FLOOR
            vp = jnp.abs(jnp.asarray(_field_chain(state, runtime)[4].vp))[1:]
            v0 = 1.5 * vp[0] - 0.5 * vp[1]
            v1 = 1.5 * vp[-1] - 0.5 * vp[-2]
            depth = jnp.maximum(WELL_FLOOR - (v0 - v1) / v0, 0.0)
            slope = jnp.maximum(jnp.diff(vp) / v0 * vp.size, 0.0)
            return jnp.concatenate([depth[None], slope])
        terms.append((magnetic_well_floor, 0.0, 1.0))
        escalated_terms.append(("magnetic_well_floor", 10.0, WELL_FLOOR))
    if rung["beta"] is not None:
        from vmex.core.stability import mercier_stability_residual

        def mercier(state, runtime):
            # Mercier criterion DMerc >= 0 on the interior surfaces (smooth one-sided residual)
            return mercier_stability_residual(state, runtime, margin=0.0, smoothing=1e-3)
        terms.append((mercier, 0.0, 1.0))
        escalated_terms.append(("mercier", 1e-4, 1.0))

    finite_beta = rung["beta"] is not None
    plasma = VmexPlasma(inp, terms, max_mode=rung["max_mode"], nphi=VC_NPHI, ntheta=VC_NTHETA, vc_digits=VC_DIGITS,
                        # vacuum: B_plasma = 0 on the boundary; virtual_casing: B_plasma from the VMEX current at
                        # EVERY evaluation (the chain simsopt's VirtualCasing runs once), with its adjoint
                        plasma_field="virtual_casing" if finite_beta else "vacuum",
                        # Boozer G (VMEX rbtor) as an extra differentiable output for the field-strength condition
                        extra_outputs={"rbtor": net_poloidal_current_output()} if finite_beta else None,
                        restart_from=eq0,
                        # accept an unconverged VMEX solve only up to fsq/ftol 1e3: at the default 1e6 the X-line
                        # rungs walked onto boundaries VMEX cannot converge, and the NEXT rung could not start
                        max_fsq_ratio=1e3)
    return plasma, escalated_terms


def circular_coils(inp):
    """NCOILS planar circles per half period around the seed axis; each carries the current giving B0 at R0."""
    R0, a = float(inp.rbc[inp.ntor, 0]), float(inp.rbc[inp.ntor, 1])    # rbc[n + ntor, m]
    nfp = int(inp.nfp)
    B0 = abs(float(inp.phiedge)) / (np.pi * a ** 2)
    I0 = 2 * np.pi * R0 * B0 / (4e-7 * np.pi * 2 * nfp * NCOILS)        # Ampere: 2 pi R0 B0 = mu0 * total current
    base = create_equally_spaced_curves(NCOILS, nfp, stellsym=True, R0=R0, R1=R1_FACTOR * a, order=ORDER)
    # Current(1.0) * I0 keeps the current dofs O(1) instead of ~1e5
    coils = coils_via_symmetries(base, [Current(1.0) * I0 for _ in base], nfp, True)
    print(f"  circular coil seed: {len(coils)} coils, R1 = {R1_FACTOR * a:.3f} m, {I0 / 1e3:.1f} kA each", flush=True)
    return coils


def coil_parts(coils):
    """The base curves and currents that own the free coil dofs (not the rotated / flipped symmetry copies).

    Each current is returned with |product of its scale factors| among the coils using it, so the current limit is on
    the actual ampere-turns."""
    base_curves, seen = [], set()
    for c in coils:
        for opt in c.curve.unique_dof_lineage:
            if opt.local_dof_size > 0 and id(opt) not in seen and "Curve" in type(opt).__name__:
                seen.add(id(opt))
                base_curves.append(opt)
    # B·n harmonics see only stellarator-odd parts, so a coil that is its OWN stellarator image could drift off
    # symmetry unseen: pin its symmetry-breaking coefficients (no-op for these circles, which are not self-images)
    fix_self_symmetric_coil_parity(base_curves)
    leaves = {}
    for c in coils:
        current, scale = c.current, 1.0
        while hasattr(current, "current_to_scale"):
            scale *= float(current.scale)
            current = current.current_to_scale
        leaves[id(current)] = (current, max(leaves.get(id(current), (None, 0.0))[1], abs(scale)))
    return [c.curve for c in coils], base_curves, list(leaves.values())


def xline_gate(rung_dir):
    """None if the last history row of a DN rung hands over a usable X-line, else the reason (run_ladder.py)."""
    row = yaml.safe_load(open(rung_dir / "history.yaml"))[-1]
    why = []
    if abs(row["xpoint_trace"]) < XLINE_GATE["min_trace"]:
        why.append(f"|tr M| = {abs(row['xpoint_trace']):.3f} < 2 (elliptic: not an X-point)")
    if row["xline_residual_max"] > XLINE_GATE["max_residual"]:
        why.append(f"field-line residual {row['xline_residual_max']:.2e} (not a periodic field line)")
    if row["xpoint_distance_min"] <= XLINE_GATE["min_distance"]:
        why.append(f"X-line-boundary distance {row['xpoint_distance_min']:+.4f} m (inside the plasma)")
    return "; ".join(why) or None


def run_rung(k, rung, seed, out_dir):
    """One combined-stage AL solve. ``seed``: the files handed over by the previous rung. Returns this rung's."""
    rung_dir = out_dir / f"{k:02d}_{rung['name']}"
    handover = dict(vmec_input=rung_dir / "input.final", coils=rung_dir / "coils_final.json",
                    xline=(rung_dir / "xpoint_final.json") if rung["double_null"] else seed["xline"],
                    wout=rung_dir / "wout_final.nc")
    if handover["wout"].exists():                      # the wout is written last: its presence means "rung done"
        print(f"\n=== rung {k}: {rung['name']} already done, skipping", flush=True)
        return handover
    rung_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== rung {k}: {rung['name']} ===", flush=True)
    t_rung = time.time()

    # ---- plasma -------------------------------------------------------------------------------------------------
    namelist = seed["vmec_input"]
    if rung["beta"] is not None:
        # the previous rung's boundary with a pressure profile added (its own small continuation in PRES_SCALE)
        namelist = make_beta_seed(namelist, rung["beta"], rung_dir / "input.beta_seed", seed["wout"])
    plasma, escalated_terms = build_plasma(rung, namelist, seed["wout"])
    vmex_key = plasma.problem.metadata.get("config")   # key of VMEX's warm-start caches for this problem

    # ---- coils: the previous rung's, or circles on the first rung ------------------------------------------------
    coils = load(str(seed["coils"])) if seed["coils"] else circular_coils(plasma.inp)
    curves, base_curves, base_currents = coil_parts(coils)

    # ---- X-line: the previous rung's, or a guess above the boundary on the first DN rung -------------------------
    xline = None
    if rung["double_null"]:
        if seed["xline"]:
            saved = load(str(seed["xline"]))
            xline = FreeXLine(saved["curve"], float(saved["length"]))
        else:
            # NOT a field line yet: the curve dofs and its length are optimisation variables, and the AL
            # field-line block below makes it one. The coils have no X-point at this stage.
            xline = free_xline_from_plasma(plasma, order=DN["order"], offset=DN["offset"])

    # ---- objective: fixed-weight terms + escalated one-sided limits ---------------------------------------------
    # Every escalated limit is (Weight, starting weight, violation()): violation() is the RELATIVE amount by which the
    # limit is broken (0 when it holds); it decides the escalation and is logged.
    escalated = {}
    terms = []

    def add_limit(name, weight, term, violation):
        escalated[name] = dict(weight=Weight(weight), start=weight, violation=violation)
        terms.append((escalated[name]["weight"], term))

    # f_QS = QS residual + aspect ratio; the escalated VMEX terms are excluded here and weighted on their own
    terms.append((1.0, VmexQuasisymmetry(plasma, exclude=tuple(name for name, _, _ in escalated_terms))))
    terms.append((ARCLENGTH_WEIGHT, sum(ArclengthVariation(c) for c in base_curves)))
    for name, weight, scale in escalated_terms:
        J_term = VmexTermCost(plasma, name)               # 0.5 |rows|^2 of that one VMEX term
        add_limit(name, weight, J_term, lambda J=J_term, s=scale: float(np.sqrt(2.0 * J.J())) / s)

    thr = {key: value[0] for key, value in ENGINEERING.items()}
    wt = {key: value[1] for key, value in ENGINEERING.items()}
    add_limit("length", wt["length"], sum(QuadraticPenalty(CurveLength(c), thr["length"], "max") for c in base_curves),
              lambda: max(CurveLength(c).J() for c in base_curves) / thr["length"] - 1.0)
    add_limit("curvature", wt["curvature"], sum(LpCurveCurvature(c, 2, thr["curvature"]) for c in base_curves),
              lambda: max(float(np.max(c.kappa())) for c in base_curves) / thr["curvature"] - 1.0)
    add_limit("msc", wt["msc"], sum(QuadraticPenalty(MeanSquaredCurvature(c), thr["msc"], "max") for c in base_curves),
              lambda: max(MeanSquaredCurvature(c).J() for c in base_curves) / thr["msc"] - 1.0)
    J_cc = CurveCurveDistance(curves, thr["coil_coil"])
    add_limit("coil_coil", wt["coil_coil"], J_cc, lambda: 1.0 - J_cc.shortest_distance() / thr["coil_coil"])
    J_cp = CoilPlasmaDistance(plasma, curves, thr["coil_plasma"])      # distance to the MOVING boundary
    add_limit("coil_plasma", wt["coil_plasma"], J_cp, lambda: 1.0 - J_cp.shortest_distance() / thr["coil_plasma"])
    add_limit("current", wt["current"], sum(CurrentBound(cur, thr["current"] / s) for cur, s in base_currents),
              lambda: max(abs(float(cur.get_value())) * s for cur, s in base_currents) / thr["current"] - 1.0)

    J_hyp = J_xdist = J_flat = J_cx = xline_constraint = None
    if xline is not None:
        # every X-line term gets its OWN BiotSavart: each one moves its evaluation points (set_points)
        margin = 2.0 + DN["hyperbolicity_margin"]
        J_hyp = FreeXLineHyperbolicity(xline, BiotSavart(coils), margin=DN["hyperbolicity_margin"])
        add_limit("hyperbolicity", DN["hyperbolicity_weight"], J_hyp, lambda: 1.0 - abs(J_hyp.trace()) / margin)
        # signed distance: a free X-line can slide INSIDE the plasma (onto the axis), which must violate d_min
        J_xdist = XpointPlasmaDistance(xline, plasma, d_min=DN["distance_min"], d_max=DN["distance_max"], signed=True)

        def xdist_violation():
            d = J_xdist.distances()
            return 0.0 if d is None else max(1.0 - d.min() / DN["distance_min"], d.max() / DN["distance_max"] - 1.0)
        add_limit("xpoint_distance", DN["distance_weight"], J_xdist, xdist_violation)
        J_cx = CoilXLineDistance(curves, xline, DN["coil_distance_min"])
        add_limit("coil_xline", DN["coil_distance_weight"], J_cx,
                  lambda: 1.0 - J_cx.shortest_distance() / DN["coil_distance_min"])
        if rung["flatness"]:
            J_flat = XLineFlatness(xline, DN["flatness_threshold"])
            add_limit("xline_flatness", DN["flatness_weight"], J_flat,
                      lambda: J_flat.max_deviation() / DN["flatness_threshold"] - 1.0)
            terms.append((DN["flatness_pull_weight"], XLineFlatness(xline, 0.0)))   # threshold 0: always pulls
    # WeightedSum (not "+"): every term that needs the VMEX adjoint shares ONE pullback per gradient
    objective = WeightedSum(terms)

    # ---- constraints (AL blocks) --------------------------------------------------------------------------------
    # B·n harmonics of B_plasma + B_coil on the boundary, divided by B_ref, plus ONE field-strength scalar: B·n = 0
    # is blind to scaling every coil current by the same factor, so something must fix the scale.
    #   vacuum:       toroidal_flux        coil flux through the phi = 0 cross-section = PHIEDGE (exact in vacuum)
    #   finite beta:  net_poloidal_current coils' linked current = the equilibrium's Boozer G (exact at any beta)
    # (docs/field_strength_condition.md). The interface gets its own BiotSavart too.
    finite_beta = rung["beta"] is not None
    interface = PlasmaCoilInterface(plasma, BiotSavart(coils), mode="fourier", mpol=rung["bn_mpol"],
                                    ntor=rung["bn_ntor"], p_edge=0.0,
                                    field_strength="net_poloidal_current" if finite_beta else "toroidal_flux")
    if finite_beta:
        # B·n as a BAND |h| <= tau: an equality keeps pulling toward an unreachable zero once the coil set's floor
        # is reached and quasi-symmetry pays for it; inside the band there is no force. The field-strength scalar
        # stays an equality (SubsetConstraint = the interface without its B·n block).
        constraints = [SubsetConstraint(interface, exclude=("bnormal",))]
    else:
        constraints = [interface]
    if xline is not None:
        # gamma'/L - B/|B| = 0 at the X-line's quadrature points: the free curve IS a field line of the coils.
        xline_constraint = XLineFieldLineConstraint(xline, BiotSavart(coils))
        constraints.append(xline_constraint)
    regularity = None
    if rung["regularity"]:
        # inequality: the poloidal speed of the boundary stays >= 0.15 x its cross-section mean. Unlike the hard
        # guard it has a gradient, so the optimiser slides along the limit instead of stalling at it (the X-line
        # tends to squeeze the boundary tip into a pinch VMEX cannot cold-solve in the next rung).
        regularity = BoundaryRegularityConstraint(plasma, min_speed_ratio=REGULARITY_MIN_SPEED, nphi=8, ntheta=64)
        constraints.append(regularity)
    if finite_beta:
        constraints.append(BandConstraint(interface, "bnormal", BNORMAL_BAND))

    # starting penalty per block (block names are "<constraint index>:<key>", e.g. "0:bnormal", "1:fieldline")
    penalty = {f"{i}:{key}": float(rung["penalty"]) for i, c in enumerate(constraints) for key in c.residuals()}
    for name in penalty:
        if name.split(":", 1)[1] in rung["penalty_blocks"]:
            penalty[name] = float(rung["penalty_blocks"][name.split(":", 1)[1]])
    al = AugmentedLagrangian(objective, constraints, penalty=penalty, penalty_growth=rung["penalty_growth"],
                             penalty_max=rung["penalty_max"], eta0=rung["eta0"], omega0=rung["omega0"])

    # ---- scaled variables: x = x_seed + D * u, the optimiser works in u (starting at 0) -------------------------
    # D sets which lever L-BFGS pulls first. Jacobi scaling equilibrates the constraint-Jacobian columns,
    # D_j = 1e-2 / ||dc/dx_j|| (capped at 100 x the median), so no dof group is inert in u: with fixed steps the
    # boundary absorbed all of B·n and the coils never moved, or the other way round. One VJP per constraint row,
    # i.e. one VMEX adjoint each (~1 min in vacuum, ~20 min with virtual casing). The regularity and band blocks
    # are left out: they would dominate the column norms.
    names = list(al.dof_names)
    plasma_mask = np.array([n.startswith(plasma.name + ":") for n in names])
    x_seed = al.x.copy()
    t0 = time.time()
    D, _ = jacobi_scaling(al, [interface] + ([xline_constraint] if xline_constraint is not None else []), names,
                          1e-2, 100.0)
    # smaller boundary steps: one rejected boundary (VMEX cannot start from it) rejects the coil part of the step too
    D[plasma_mask] *= rung["plasma_factor"]
    print(f"  {x_seed.size} dofs = {plasma_mask.sum()} boundary + {(~plasma_mask).sum()} coil/X-line; "
          f"Jacobi scaling {time.time() - t0:.0f}s; AL blocks {sorted(al.constraint_norms())}", flush=True)

    # ---- L(u) with guards and a failed-solve barrier -----------------------------------------------------------
    good, best = {}, {}      # last accepted evaluation (barrier anchor) / lowest-L accepted one of this inner solve

    def vmex_state():
        """VMEX's stored solve of the current point: re-solving a boundary that was reached only through a chain of
        warm starts can fail from a different starting state, so fallback points keep theirs."""
        return dict(solve=vmex_implicit._LAST_SOLVE.get(vmex_key), hot=vmex_implicit._HOT_CACHE.get(vmex_key))

    def restore(state):
        if state.get("solve") is not None:
            vmex_implicit._LAST_SOLVE[vmex_key] = state["solve"]
        if state.get("hot") is not None:
            vmex_implicit._HOT_CACHE[vmex_key] = state["hot"]

    def fun(u):
        al.x = x_seed + D * u
        why = None
        # 1) hard guard, BEFORE any VMEX solve: a self-intersecting or pinched boundary is never evaluated
        ratio, _, crossing = boundary_regularity(plasma, 8, 512)
        if crossing or ratio < GUARD_MIN_SPEED:
            why = f"boundary guard (speed ratio {ratio:.3g}, self-intersecting {crossing})"
        else:
            try:
                J, g = al.J(), al.dJ() * D                # chain rule: dL/du = dL/dx * D
                if not plasma.accepted:
                    why = "VMEX solve rejected"
                elif not (np.isfinite(J) and np.all(np.isfinite(g))):
                    why = "non-finite L or gradient"
                elif finite_beta:
                    # virtual casing has returned garbage on many-core nodes (|B_plasma| up to 1e10 T at random
                    # points): never accept a trial whose plasma field is not small compared with |B|
                    out = plasma.outputs()
                    ratio_vc = (np.linalg.norm(np.asarray(out["B_plasma"]).reshape(3, -1), axis=0)
                                / np.sqrt(np.asarray(out["Bin_mag2"]).ravel()))
                    if not np.all(np.isfinite(ratio_vc)) or ratio_vc.max() > 1.0:
                        why = f"implausible virtual casing (max |B_plasma|/|B| {ratio_vc.max():.2e})"
            except Exception as e:  # noqa: BLE001 -- any solver failure becomes a barrier
                why = f"evaluation raised {e!r}"
        if why is None:
            good.update(u=u.copy(), J=J, g=g.copy())
            if not best or J < best["J"]:
                best.update(u=u.copy(), J=J, **vmex_state())
            return J, g
        # 2) barrier: a quadratic bowl around the last accepted point, steep enough that the line search backs off
        if not good:
            raise RuntimeError(why)
        print(f"    barrier: {why}", flush=True)
        du = u - good["u"]
        nrm2 = float(du @ du) + 1e-300
        gdu = float(good["g"] @ du)
        K = max(abs(gdu) / (BARRIER_RETREAT * nrm2), 1.0)
        return good["J"] + gdu + 0.5 * K * nrm2, good["g"] + K * du

    def report(outer, record=None):
        """One history row (history.yaml; the X-line gate reads the last one) and a log line."""
        res = interface.residuals()
        costs = plasma.term_costs()
        # f_QS = QS residual + aspect only; the escalated VMEX terms (iota floor, well, Mercier) are in `violations`
        row = dict(outer=outer, L=float(al.J()), f_QS=float(sum(v for n, v in costs.items() if n not in escalated)),
                   terms=costs,
                   bnormal_inf=float(np.max(np.abs(res["bnormal"]))), rms_Bn_over_B=float(interface.rms_bnormal_over_B()),
                   min_abs_iota=float(np.min(np.abs(plasma.equilibrium().wout.iotaf))),
                   violations={name: max(float(e["violation"]()), 0.0) for name, e in escalated.items()},
                   weights={name: float(e["weight"].value) for name, e in escalated.items()})
        if xline is not None:
            d = J_xdist.distances()
            row.update(xpoint_trace=J_hyp.trace(), xpoint_distance_min=float(d.min()),
                       xpoint_distance_max=float(d.max()),
                       xline_residual_max=float(np.max(np.abs(xline_constraint.residuals()["fieldline"]))))
            if J_flat is not None:
                row["xline_z_deviation"] = J_flat.max_deviation()
        if record is not None:
            row.update(constraint_norms=record["norms"], penalties=record["penalties"])
        with open(rung_dir / "history.yaml", "a") as fh:
            yaml.safe_dump([row], fh, sort_keys=False)
        broken = {n: round(v, 3) for n, v in row["violations"].items() if v > ESCALATION_TOL}
        print(f"  [outer {outer}] L = {row['L']:.4e}, f_QS = {row['f_QS']:.3e}, max |B·n harmonic| = "
              f"{row['bnormal_inf']:.2e}, rms(B·n/B) = {row['rms_Bn_over_B']:.2e}, min |iota| = {row['min_abs_iota']:.3f}"
              + (f", X-line |tr M| = {abs(row['xpoint_trace']):.3f} d = [{row['xpoint_distance_min']:.3f}, "
                 f"{row['xpoint_distance_max']:.3f}] m residual {row['xline_residual_max']:.1e}" if xline is not None else "")
              + (f" dz {row['xline_z_deviation'] * 1e3:.1f} mm" if "xline_z_deviation" in row else "")
              + f"; violated limits {broken or 'none'}", flush=True)

    # ---- the AL outer loop --------------------------------------------------------------------------------------
    u = np.zeros_like(x_seed)
    fun(u)                                             # the starting point must evaluate (raises otherwise)
    report(-1)
    for outer in range(rung["outer"]):
        good.clear()
        best.clear()
        start = dict(u=u.copy(), **vmex_state())
        t0 = time.time()
        # inner solve: minimise L(u; multipliers, penalties) to the current gradient tolerance omega
        res = minimize(fun, u, jac=True, method="L-BFGS-B",
                       options=dict(maxiter=rung["inner"], gtol=al.omega, ftol=1e-15))
        # L-BFGS-B returns its LAST iterate, which may be a barrier value or a trial the line search had already
        # rejected: re-evaluate it, and fall back to the lowest-L accepted point (then to this outer's start)
        best_point = dict(best)
        good.clear()
        try:
            J_end, g = fun(res.x)
            u = res.x
            if best_point and J_end > best_point["J"]:
                u = best_point["u"]
                restore(best_point)
                good.clear()
                _, g = fun(u)
        except RuntimeError:
            for fallback in (best_point, start):
                if not fallback:
                    continue
                u = fallback["u"]
                restore(fallback)
                good.clear()
                try:
                    _, g = fun(u)
                    break
                except RuntimeError:
                    continue
        # outer update (Nocedal & Wright 17.4, per block): a block within its tolerance eta gets a multiplier update
        # lambda <- lambda - rho c and a tighter eta; a block outside it gets rho *= penalty_growth
        record = al.update(float(np.max(np.abs(g))), ctol=rung["ctol"], gtol=rung["gtol"])
        # escalation: every one-sided limit still broken by > 1 % gets 10x its weight (capped)
        for name, e in escalated.items():
            if e["violation"]() > ESCALATION_TOL and e["weight"].value < rung["escalation_cap"] * e["start"]:
                e["weight"] *= ESCALATION_FACTOR
        print(f"  [outer {outer}] inner: {res.nit} it ({time.time() - t0:.0f}s), {res.message}; "
              f"actions {record['actions']}", flush=True)
        report(outer, record)
        if record["converged"]:
            print("  augmented Lagrangian converged", flush=True)
            break

    # ---- hand-over files (the wout last: it marks the rung as done) ---------------------------------------------
    al.x = x_seed + D * u
    plasma.vmec_input().to_indata(str(handover["vmec_input"]))
    save(coils, str(handover["coils"]))
    if xline is not None:
        save({"curve": xline.curve, "length": xline.length}, str(handover["xline"]))
    # reported, never optimised: pointwise B·n and pressure balance on an independent, finer virtual-casing grid
    check = interface_check(plasma, BiotSavart(coils), nphi=32, ntheta=32, digits=4, p_edge=0.0)
    print(f"  final check: rms(B·n/B) = {check['bnormal_rms']:.2e}, pressure jump rms {check['pressure_rms']:.2e}; "
          f"rung took {(time.time() - t_rung) / 3600:.2f} h", flush=True)
    vmex.write_wout(str(handover["wout"]), plasma.equilibrium().wout)
    return handover


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default="output_combined_stage_ladder")
    p.add_argument("--smoke", action="store_true", help="1 outer x 3 inner iterations per rung")
    p.add_argument("--no-xline-gate", action="store_true", help="do not stop after a DN rung with an unusable X-line")
    args = p.parse_args()
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # the first rung starts from the seed namelist and circular coils (coils=None); there is no X-line yet
    seed = dict(vmec_input=SEED, coils=None, xline=None, wout=None)
    dn_seen = False
    for k, rung in enumerate(RUNGS):
        rung = {**DEFAULTS, **rung}
        if args.smoke:
            rung.update(outer=1, inner=3)
        seed = run_rung(k, rung, seed, out_dir)
        if rung["double_null"]:
            # the first DN rung only INTRODUCES the X-line (it is rarely converged yet), so it is not gated
            if dn_seen and not args.no_xline_gate:
                why = xline_gate(out_dir / f"{k:02d}_{rung['name']}")
                if why:
                    raise SystemExit(f"rung {k} ({rung['name']}): X-line gate failed: {why}; ladder stopped")
            dn_seen = True
    print(f"\nladder done; final device in {out_dir.resolve()}/{len(RUNGS) - 1:02d}_{RUNGS[-1]['name']}/")


if __name__ == "__main__":
    main()
