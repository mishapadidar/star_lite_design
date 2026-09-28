#!/usr/bin/env python
r"""
The combined stage-I/II optimisation of combined_stage_simple.py applied to
Star_Lite designA at volume-averaged beta = 1 %.

designA was optimised in vacuum (its native equilibrium has beta ~ 3e-4). At
beta = 1 % its plasma carries a diamagnetic/Pfirsch-Schlueter field of its
own, so designA's coils no longer produce a consistent boundary: B_plasma·n
does not vanish, the boundary |B| changes, and iota drops. This example lets
the boundary and designA's coils adapt TOGETHER to the finite-beta plasma:

    boundary s --VMEX--> B_total on s --virtual casing--> B_plasma on s

    min_{s, c}  f_QS(s) + coil penalties (length, curvature, coil-coil,
                          coil-plasma to the MOVING boundary, |I| <= 60 kA)

    subject to  ((B_plasma(s) + B_coil(c))·n)_mn / B_ref = 0     m <= BN_MPOL, |n| <= BN_NTOR
                < |B_out|^2 - |B_in|^2 - 2 mu0 p_edge > / B_ref^2 = 0

in one augmented-Lagrangian solve with exact gradients (VMEX implicit adjoint
through equilibrium + virtual casing, simsopt derivatives for the coils).

Differences to combined_stage_simple.py:

* seed = designA's boundary (VMEC namelist) and its 6 coils (simsopt archive)
  instead of the Landreman-Paul QA and circular coils;
* the seed namelist is given beta = 1 %: p(s) = PRES_SCALE (1 - s), with
  PRES_SCALE calibrated by a few VMEX solves (as tools/make_beta_seed.py);
  the net toroidal current stays zero (no bootstrap current);
* coil penalties at designA's scale, including curvature (designA's coils are
  order-16 Fourier curves) and the 60 kA current limit; every threshold sits
  just outside designA's own coils, so none is active at the seed;
* an iota floor: at beta = 1 % designA's |iota| falls from ~0.24 to ~0.13.

Not included: designA's double null. The X-point field line is not tracked
here, so nothing keeps it; combined_stage_vmex.py (double_null section)
does that, see configs/designA_L2_vc_beta1.yaml for the production setup.

Run on a compute node:

    python combined_stage_designA_beta1.py
"""

import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import vmex
from vmex import optimize as vopt
from simsopt import load, save
from simsopt.field import BiotSavart
from simsopt.geo import (CurveCurveDistance, CurveLength, LpCurveCurvature, SurfaceRZFourier, curves_to_vtk)
from simsopt.objectives import QuadraticPenalty
from star_lite_design.utils.augmented_lagrangian import AugmentedLagrangian, solve_augmented_lagrangian
from star_lite_design.utils.current_bound import CurrentBound
from star_lite_design.utils.vmex_combined_stage import (CoilPlasmaDistance, PlasmaCoilInterface, VmexPlasma,
                                                        VmexQuasisymmetry, WeightedSum)

STAR_LITE = Path(__file__).resolve().parents[2]

# designA: simsopt archive (entry 0 = BoozerSurfaces, whose BiotSavart holds the coils)
# and the VMEC namelist of its optimisation surface (MPOL 11 / NTOR 10, beta ~ 3e-4):
design_archive = STAR_LITE / "convert" / "designA_after_scaled.json"
vmec_input = STAR_LITE.parents[1] / "runs" / "combined_stage" / "seeds" / "input.designA_opt_beta0p03_mpol11.fixed"

# Target volume-averaged beta:
BETA = 0.01

# Equilibrium resolution used during the optimisation. designA's boundary is
# MPOL 11 / NTOR 10; 6 / 6 drops its m >= 6 content (~0.5 mm rms) but is ~5x
# faster (README "Resolution audit": f_QS changes by ~0.1 %):
MPOL = 6
NTOR = 6
NS = 16

# Largest boundary mode number varied (m, |n| <= MAX_MODE):
MAX_MODE = 1

# Virtual-casing grid on the boundary, per field period, and its accuracy:
VC_NPHI = 24
VC_NTHETA = 20
VC_DIGITS = 3

# Constrained B·n harmonics on the plasma boundary (need 2*BN_MPOL < VC_NTHETA, 2*BN_NTOR < VC_NPHI).
# Only the stellarator-ODD (sin) harmonics are constrained: B·n of a symmetric
# plasma + coil set has no even part. The coils are kept symmetric below.
BN_MPOL = 5
BN_NTOR = 5

# Plasma objective: quasi-axisymmetry on these surfaces, the seed's aspect ratio,
# and a one-sided floor on min |iota| (below the beta = 1 % seed's value):
QS_SURFACES = np.linspace(0.1, 0.9, 8)
IOTA_FLOOR = 0.11
IOTA_WEIGHT = 10.0

# Engineering penalties (active only when violated). designA's own coils: length
# 3.03 / 3.09 m, max curvature 7.95 / 10.47 /m, coil-coil 0.149 m, coil-plasma 0.133 m,
# |I| 41 / 28 kA. The weights must hold against the augmented-Lagrangian terms,
# which grow with the multipliers: at curvature weight 1e-3 / coil-coil 1e3 the
# B·n constraints bent designA's coils to 14.8 /m and to 0.115 m apart
# within 5 outer iterations (job 2135).
LENGTH_MAX = 4.0         # m, per coil
LENGTH_WEIGHT = 1e1
CURVATURE_MAX = 11.0     # 1/m
CURVATURE_WEIGHT = 1e0
CC_THRESHOLD = 0.12      # m, coil-coil
CC_WEIGHT = 1e5
CS_THRESHOLD = 0.10      # m, coil-plasma (to the MOVING boundary)
CS_WEIGHT = 1e3
CURRENT_MAX = 60e3       # A, Star_Lite's coil current limit
CURRENT_WEIGHT = 1e6       # the penalty acts on the raw Current dof (~0.05 per 41 kA), as in config.yaml

# Augmented Lagrangian. designA's own coils cannot reach every constrained
# harmonic much below ~3e-4 at beta 1 % (README, job 874), so that is the target:
PENALTY0 = 10.0
ETA0 = 1e-2
CTOL = 3e-4
GTOL = 1e-5
OUTER_MAXITER = 10
INNER_MAXITER = 100

# Optimiser variables u: x = x_seed + D * u (see combined_stage_simple.py):
# D = PLASMA_STEP x VMEX's ESS scale for each boundary mode, COIL_STEP for each
# coil Fourier coefficient, CURRENT_STEP x |seed value| for each current dof.
PLASMA_STEP = 1e-1
COIL_STEP = 1e-1
CURRENT_STEP = 1e-1

#######################################################
# End of input parameters.
#######################################################

# Directory for output
out_dir = Path("output_combined_stage_designA_beta1")
out_dir.mkdir(parents=True, exist_ok=True)

print("""
################################################################################
### Plasma: designA at beta = 1 % (VMEX + virtual casing) ######################
################################################################################
""")
inp = vmex.VmecInput.from_file(str(vmec_input))
inp = inp.change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=2 * NTOR + 4)
inp = replace(inp, ns_array=np.array([NS]), ftol_array=np.array([1e-11]), niter_array=np.array([20000]))

# p(s) = PRES_SCALE (1 - s): calibrate PRES_SCALE so that VMEX's betatotal = BETA
# (beta is nearly linear in the pressure, so a few rescalings converge)
am = np.zeros_like(np.asarray(inp.am, dtype=float))
am[:2] = 1.0, -1.0
a_minor = abs(float(inp.rbc[inp.ntor, 1]))                      # rbc[n + ntor, m]
B0 = abs(float(inp.phiedge)) / (np.pi * a_minor ** 2)
pres_scale = 2.0 * BETA * B0 ** 2 / (2.0 * 4e-7 * np.pi)        # <p> = PRES_SCALE / 2 = beta B0^2 / (2 mu0)
for it in range(6):
    inp = replace(inp, pmass_type="power_series", am=am, pres_scale=pres_scale)
    t0 = time.time()
    eq0 = vopt.solve_equilibrium(inp)
    beta = float(eq0.wout.betatotal)
    print(f"beta calibration {it}: PRES_SCALE {pres_scale:8.2f} Pa -> beta {beta:.4%} ({time.time() - t0:.1f}s)")
    if abs(beta / BETA - 1.0) < 2e-3:
        break
    pres_scale *= BETA / beta
else:
    raise SystemExit(f"beta calibration did not converge (beta {beta:.4%})")
aspect0 = float(eq0.wout.aspect)
iota = np.asarray(eq0.wout.iotaf, dtype=float)
print(f"seed equilibrium: aspect {aspect0:.3f}, beta {beta:.3%}, iota (axis, edge) = ({iota[0]:+.3f}, {iota[-1]:+.3f}), "
      f"min |iota| {np.min(np.abs(iota)):.3f} (floor {IOTA_FLOOR})")


def iota_floor(state, runtime):
    return jnp.maximum(IOTA_FLOOR - vopt.min_abs_iota(state, runtime), 0.0)


# f_QS = 0.5 |r|^2 of these VMEX (function, target, weight) terms:
objective_terms = [
    (vopt.QuasisymmetryRatioResidual(QS_SURFACES, helicity_m=1, helicity_n=0), 0.0, 1.0),
    (vopt.aspect_ratio, aspect0, 1.0),
    (iota_floor, 0.0, IOTA_WEIGHT),
]
plasma = VmexPlasma(inp, objective_terms, max_mode=MAX_MODE, nphi=VC_NPHI, ntheta=VC_NTHETA,
                    vc_digits=VC_DIGITS, plasma_field="virtual_casing", restart_from=eq0)
f_qs = VmexQuasisymmetry(plasma)

out0 = plasma.outputs()
B_in = np.sqrt(out0["Bin_mag2"])
print(f"virtual casing: |B_plasma| / |B_in| median {np.median(np.linalg.norm(out0['B_plasma'], axis=0) / B_in):.2e}, "
      f"rms B_plasma·n / |B_in| = {np.sqrt(np.sum(out0['weights'] * (out0['Bn_plasma'] / B_in) ** 2)):.2e} "
      f"(the part of B·n the coils must cancel)")

print("""
################################################################################
### Coils: designA's own ########################################################
################################################################################
""")
coils = load(str(design_archive))[0][0].biotsavart.coils
curves = [c.curve for c in coils]
# the curves / currents that own free dofs (the others are rotated / sign-flipped copies)
base_curves, seen = [], set()
for c in coils:
    for o in c.curve.unique_dof_lineage:
        if o.local_dof_size > 0 and id(o) not in seen and "Curve" in type(o).__name__:
            seen.add(id(o))
            base_curves.append(o)
base_currents = {}          # id -> (Current, largest |scale| applied to it)
for c in coils:
    current, scale = c.current, 1.0
    while hasattr(current, "current_to_scale"):
        scale *= float(current.scale)
        current = current.current_to_scale
    prev = base_currents.get(id(current), (current, 0.0))
    base_currents[id(current)] = (current, max(prev[1], abs(scale)))
print(f"{len(coils)} coils from {len(base_curves)} base curves (order {base_curves[0].order}), "
      f"currents {sorted({round(c.current.get_value() / 1e3, 1) for c in coils})} kA")

# Keep the coil set stellarator symmetric. designA's two coils at phi = ±90 deg are
# each their own stellarator image: their base curve is symmetric only while its
# odd-parity coefficients (xs, yc, zc -- 50 of 99, exactly 0 in the archive) stay 0.
# Nothing else enforces that, and the B·n constraints above cannot see the even B·n
# a broken coil makes: with these dofs free, the coils drifted 7 mm off symmetry and
# rms B·n/B grew 1.3 % -> 1.95 %, all of it stellarator-even (jobs 2147, 2149).
for b in base_curves:
    for name, value in zip(b.local_full_dof_names, b.local_full_x):
        if value == 0.0:
            b.fix(name)


def coil_symmetry_error():
    """Largest distance from the stellarator image (x, -y, -z) of a coil point to the coil set, m."""
    g = np.array([c.curve.gamma() for c in coils])
    mirror = g * np.array([1.0, -1.0, -1.0])
    return max(float(np.min(np.linalg.norm(mirror[i][:, None, None] - g[None], axis=-1))) for i in range(len(g)))


print(f"fixed {sum(b.local_full_dof_size - b.local_dof_size for b in base_curves)} odd-parity coil coefficients; "
      f"coil symmetry error {coil_symmetry_error():.1e} m")

bs = BiotSavart(coils)      # for diagnostics / output
curves_to_vtk(curves, out_dir / "curves_init")

print("""
################################################################################
### Objective and constraints ###################################################
################################################################################
""")
# The consistency constraints get their OWN BiotSavart: they move its evaluation points.
interface = PlasmaCoilInterface(plasma, BiotSavart(coils), mode="fourier", mpol=BN_MPOL, ntor=BN_NTOR,
                                field_strength="pressure_balance", p_edge=0.0)
Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD)
Jcsdist = CoilPlasmaDistance(plasma, curves, CS_THRESHOLD)
Jcurv = sum(LpCurveCurvature(c, 2, CURVATURE_MAX) for c in base_curves)
Jcurrent = sum(CurrentBound(current, CURRENT_MAX / scale) for current, scale in base_currents.values())

# WeightedSum (not "+") so that f_QS and the coil-plasma distance share ONE VMEX adjoint
objective = WeightedSum([
    (1.0, f_qs),
    (LENGTH_WEIGHT, sum(QuadraticPenalty(J, LENGTH_MAX, "max") for J in Jls)),
    (CURVATURE_WEIGHT, Jcurv),
    (CC_WEIGHT, Jccdist),
    (CS_WEIGHT, Jcsdist),
    (CURRENT_WEIGHT, Jcurrent),
])
al = AugmentedLagrangian(objective, [interface], penalty=PENALTY0, eta0=ETA0)

names = list(al.dof_names)
plasma_mask = np.array([n.startswith(plasma.name + ":") for n in names])
current_mask = np.array([n.startswith("Current") for n in names])
x_seed = al.x.copy()
D = np.full(x_seed.size, COIL_STEP)
D[plasma_mask] = PLASMA_STEP * np.asarray(plasma.problem.scales)[plasma.dofs_free_status]
D[current_mask] = CURRENT_STEP * np.abs(x_seed[current_mask])
res0 = interface.residuals()
print(f"dofs: {x_seed.size} = {plasma_mask.sum()} plasma + {(~plasma_mask & ~current_mask).sum()} coil shape + "
      f"{current_mask.sum()} current; constraints: {res0['bnormal'].size} B·n harmonics + 1 pressure balance")
print(f"seed: f_QS = {f_qs.J():.3e} ({', '.join(f'{k} {v:.2e}' for k, v in plasma.term_costs().items())}), "
      f"max |B·n harmonic| = {np.max(np.abs(res0['bnormal'])):.3e}, rms(B·n/B) = {interface.rms_bnormal_over_B():.3e}, "
      f"pressure balance = {res0['pressure_balance'][0]:+.3e}")
print(f"seed penalties (all should be 0): length {sum(QuadraticPenalty(J, LENGTH_MAX, 'max').J() for J in Jls):.2e}, "
      f"curvature {Jcurv.J():.2e}, coil-coil {Jccdist.J():.2e}, coil-plasma {Jcsdist.J():.2e}, current {Jcurrent.J():.2e}")

# Wrapper for scipy in the scaled variables u. A trial whose VMEX solve fails
# returns a quadratic barrier around the last good point, so the line search
# backs off instead of stopping (as in array/boozer_all.py).
good = {}


def fun(u):
    al.x = x_seed + D * u
    J, grad = al.J(), al.dJ() * D
    if plasma.accepted and np.isfinite(J) and np.all(np.isfinite(grad)):
        good.update(u=u.copy(), J=J, grad=grad.copy())
        return J, grad
    if not good:
        raise RuntimeError("VMEX rejected the starting point")
    du = u - good["u"]
    nrm2 = float(du @ du) + 1e-300
    K = max(abs(float(good["grad"] @ du)) / (0.5 * nrm2), 1.0)
    return good["J"] + float(good["grad"] @ du) + 0.5 * K * nrm2, good["grad"] + K * du


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
u0 = np.zeros_like(x_seed)
np.random.seed(1)
h = np.random.uniform(size=u0.shape)
J0, dJ0 = fun(u0)
dJh = float(dJ0 @ h)
for eps in [1e-2, 1e-3, 1e-4, 1e-5]:
    J1, _ = fun(u0 + eps * h)
    J2, _ = fun(u0 - eps * h)
    print(f"eps {eps:.0e}: err {(J1 - J2) / (2 * eps) - dJh:+.3e} (relative {((J1 - J2) / (2 * eps) - dJh) / dJh:+.2e})")
fun(u0)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
t_start = time.time()


def report(record):
    al.x = x_seed + D * good["u"]
    res = interface.residuals()
    iota_now = np.asarray(plasma.equilibrium().wout.iotaf, dtype=float)
    print(f"[outer {record['outer']}] {record['inner_nit']} inner it ({time.time() - t_start:.0f}s): "
          f"L = {record['L']:.4e}, f_QS = {f_qs.J():.4e}, min|iota| = {np.min(np.abs(iota_now)):.3f}, "
          f"max|B·n harmonic| = {np.max(np.abs(res['bnormal'])):.2e}, rms(B·n/B) = {interface.rms_bnormal_over_B():.2e}, "
          f"pressure balance = {res['pressure_balance'][0]:+.2e}, lengths = {[round(float(J.J()), 3) for J in Jls]}, "
          f"max kappa = {max(float(np.max(c.kappa())) for c in base_curves):.2f}, "
          f"coil-coil = {Jccdist.shortest_distance():.3f} m, coil-plasma = {Jcsdist.shortest_distance():.3f} m, "
          f"|I| max = {max(abs(c.current.get_value()) for c in coils) / 1e3:.1f} kA, rho = {record['penalties']}",
          flush=True)


# x0: the solver works in u, which starts at 0 (not at al.x)
u, record = solve_augmented_lagrangian(al, fun=fun, x0=u0, outer_maxiter=OUTER_MAXITER, inner_maxiter=INNER_MAXITER,
                                       ctol=CTOL, gtol=GTOL, callback=report)
print("converged" if record["converged"] else f"stopped after {OUTER_MAXITER} outer iterations")
fun(u)
al.x = x_seed + D * good["u"]      # leave every object at the last accepted point

# Where the remaining B·n sits: its stellarator-odd part (what the constraints act on)
# and even part (zero for symmetric coils), rms of (B_plasma + B_coil)·n / |B_in|.
c = interface._compute()
w = plasma.outputs()["weights"]
f = c["f"] * interface.B_ref / np.sqrt(plasma.outputs()["Bin_mag2"])
f_image = np.roll(np.roll(f[::-1, ::-1], 1, axis=0), 1, axis=1)     # f(-phi, -theta) on the endpoint-free grid
print(f"rms B·n/B: stellarator-odd {np.sqrt(np.sum(w * (0.5 * (f - f_image)) ** 2)):.2e}, "
      f"even {np.sqrt(np.sum(w * (0.5 * (f + f_image)) ** 2)):.2e}; coil symmetry error {coil_symmetry_error():.1e} m")

print("""
################################################################################
### Save the results ###########################################################
################################################################################
""")
curves_to_vtk(curves, out_dir / "curves_opt")
save(coils, out_dir / "coils_opt.json")
plasma.vmec_input().to_indata(str(out_dir / "input.combined_stage_opt"))
vmex.write_wout(str(out_dir / "wout_combined_stage_opt.nc"), plasma.equilibrium().wout)

# B·n of the coils alone on the final boundary, relative to |B_coil| (at finite
# beta it should equal -B_plasma·n, not 0)
s = SurfaceRZFourier.from_vmec_input(str(out_dir / "input.combined_stage_opt"), range="full torus",
                                     nphi=64, ntheta=32)
bs.set_points(s.gamma().reshape((-1, 3)))
Bbs = bs.B().reshape(s.gamma().shape)
BdotN = np.sum(Bbs * s.unitnormal(), axis=2) / np.linalg.norm(Bbs, axis=2)
s.to_vtk(out_dir / "surf_opt", extra_data={"B_coil_N": BdotN[:, :, None]})
res = interface.residuals()
print(f"final: f_QS = {f_qs.J():.4e}, rms((B_plasma + B_coil)·n / B) = {interface.rms_bnormal_over_B():.2e}, "
      f"pressure balance = {res['pressure_balance'][0]:+.2e}; "
      f"VMEX forward/adjoint = {plasma.n_forward}/{plasma.n_backward}; outputs in {out_dir.resolve()}")
