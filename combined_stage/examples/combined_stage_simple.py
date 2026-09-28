#!/usr/bin/env python
r"""
In this example we solve a combined stage-I/II problem at finite beta: the
plasma boundary and the coils are optimised TOGETHER, in one augmented-
Lagrangian solve. It is the combined-stage counterpart of simsopt's
``examples/2_Intermediate/stage_two_optimization_finite_beta.py``.

There, the equilibrium is fixed: VMEC gives B_total on the boundary, virtual
casing splits off the plasma's own field B_plasma, and the coils are fitted
so that B_coil·n = -B_plasma·n. Here the same chain is evaluated at EVERY
iterate, because the boundary is a set of degrees of freedom too:

    boundary s --VMEX--> B_total on s --virtual casing--> B_plasma on s

and the problem is

    min_{s, c}  f_QS(s) + LENGTH_WEIGHT * Σ max(CurveLength - L_max, 0)^2
                        + CC_WEIGHT * coil-coil distance penalty
                        + CS_WEIGHT * coil-plasma distance penalty (moving plasma)

    subject to  ((B_plasma(s) + B_coil(c))·n)_mn / B_ref = 0     m <= BN_MPOL, |n| <= BN_NTOR
                < |B_out|^2 - |B_in|^2 - 2 mu0 p_edge > / B_ref^2 = 0

where s are the VMEX boundary modes, c the coil shapes and currents, f_QS
the quasi-symmetry residual (plus an aspect-ratio term) of the equilibrium,
(·)_mn the Fourier harmonics of the vacuum-side normal field on the
boundary, B_out = B_plasma + B_coil the field just outside and B_in the
VMEX field just inside. The second (pressure-balance) constraint fixes the
field strength, which B·n alone cannot see.

Compared with stage_two_optimization_finite_beta.py:

* the boundary moves, so B_plasma·n is not a fixed target but a function of
  s, and quasi-symmetry is optimised alongside the coils;
* every gradient comes from an adjoint -- VMEX's implicit adjoint through the
  equilibrium AND the virtual-casing integral for the plasma dofs, simsopt's
  analytic derivatives for the coils -- ONE VMEX adjoint per gradient;
* the consistency condition is an equality constraint enforced with an
  augmented Lagrangian (multipliers + finite penalty) instead of a
  squared-flux term with a big weight.

Caveat: the plasma-dof gradient of the virtual-casing / pressure-balance
terms differs from re-solve finite differences by ~2 % (the discrete VMEC
equilibrium is not unique along a near-null direction; see ../README.md).
The Taylor test below shows the size of it.

Seed: the Landreman & Paul (2021) QA at beta = 2.5 % with a self-consistent
bootstrap current (nfp = 2, R = 1 m), and 4 circular coils per half period.

Run on a compute node (one evaluation = one VMEX solve + virtual casing):

    python combined_stage_simple.py
"""

import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import vmex
from vmex import optimize as vopt
from simsopt import save
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (CurveCurveDistance, CurveLength, SurfaceRZFourier, curves_to_vtk,
                         create_equally_spaced_curves)
from simsopt.objectives import QuadraticPenalty
from star_lite_design.utils.augmented_lagrangian import AugmentedLagrangian, solve_augmented_lagrangian
from star_lite_design.utils.vmex_combined_stage import (CoilPlasmaDistance, PlasmaCoilInterface, VmexPlasma,
                                                        VmexQuasisymmetry, WeightedSum)

# Seed fixed-boundary VMEC namelist (from the VMEX sources next to star_lite_design in ext/):
VMEX_DATA = Path(__file__).resolve().parents[3] / "vmex" / "examples" / "data"
vmec_input = VMEX_DATA / "input.LandremanPaul2021_QA_beta2p5_bootstrap"

# Equilibrium resolution used during the optimisation (lower = faster; the
# boundary harmonics above it are dropped):
MPOL = 6
NTOR = 6
NS = 16

# Largest boundary mode number varied (m, |n| <= MAX_MODE):
MAX_MODE = 1

# Virtual-casing grid on the boundary, per field period, and its accuracy:
VC_NPHI = 24
VC_NTHETA = 20
VC_DIGITS = 3

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2 and stellarator symmetry, multiply ncoils by 2 * 2 to get the total number of coils.)
ncoils = 4

# Minor radius for the initial circular coils (their major radius is the plasma's):
R1 = 0.5

# Number of Fourier modes describing each Cartesian component of each coil:
order = 5

# Constrained B·n harmonics on the plasma boundary (need 2*BN_MPOL < VC_NTHETA, 2*BN_NTOR < VC_NPHI):
BN_MPOL = 5
BN_NTOR = 5

# Engineering penalties (active only when violated):
LENGTH_MAX = 4.0         # m, per coil
LENGTH_WEIGHT = 1e0
CC_THRESHOLD = 0.1       # m, coil-coil
CC_WEIGHT = 1e3
CS_THRESHOLD = 0.1       # m, coil-plasma (to the MOVING boundary); keep it inactive at the seed, or the
                         # cheapest way to satisfy it is to deform the plasma
CS_WEIGHT = 1e3

# Augmented Lagrangian:
PENALTY0 = 10.0          # initial penalty on every constraint block
ETA0 = 1e-2              # first feasibility tolerance (|residual|_inf)
CTOL = 1e-4              # final feasibility tolerance
GTOL = 1e-5              # final |grad L|_inf (scaled variables)
OUTER_MAXITER = 10
INNER_MAXITER = 100

# Optimiser variables u: x = x_seed + D * u, with D = PLASMA_STEP x VMEX's ESS
# scale for each boundary mode and COIL_STEP for every coil dof. This sets which
# lever L-BFGS pulls first: keeping B·n = 0 takes far more coil than plasma
# motion, so with plasma steps as large as the coil steps the boundary absorbs
# the whole mismatch (f_QS 5e-5 -> 1.9 with the coils unmoved at PLASMA_STEP 1,
# COIL_STEP 1e-2).
PLASMA_STEP = 1e-1
COIL_STEP = 1e-1

#######################################################
# End of input parameters.
#######################################################

# Directory for output
out_dir = Path("output_combined_stage_simple")
out_dir.mkdir(parents=True, exist_ok=True)

print("""
################################################################################
### Plasma: VMEX equilibrium + virtual casing ##################################
################################################################################
""")
inp = vmex.VmecInput.from_file(str(vmec_input))
inp = inp.change_resolution(mpol=MPOL, ntor=NTOR, ntheta=2 * MPOL + 6, nzeta=2 * NTOR + 4)
inp = replace(inp, ns_array=np.array([NS]), ftol_array=np.array([1e-11]), niter_array=np.array([20000]))
t0 = time.time()
eq0 = vopt.solve_equilibrium(inp)
aspect0 = float(eq0.wout.aspect)
print(f"seed equilibrium: {time.time() - t0:.1f}s, aspect {aspect0:.3f}, beta {float(eq0.wout.betatotal):.3%}, "
      f"iota (axis, edge) = ({float(eq0.wout.iotaf[0]):+.3f}, {float(eq0.wout.iotaf[-1]):+.3f})")

# f_QS = 0.5 |r|^2 of these VMEX (function, target, weight) terms:
objective_terms = [
    (vopt.QuasisymmetryRatioResidual(np.linspace(0.1, 1.0, 10), helicity_m=1, helicity_n=0), 0.0, 1.0),
    (vopt.aspect_ratio, aspect0, 1.0),
]
# plasma_field="virtual_casing": every evaluation solves VMEX and splits B_plasma
# off B_total on the boundary (the chain simsopt's VirtualCasing runs ONCE)
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
### Coils #######################################################################
################################################################################
""")
nfp = plasma.nfp
R0, a = float(inp.rbc[inp.ntor, 0]), float(inp.rbc[inp.ntor, 1])    # rbc[n + ntor, m]
B0 = abs(float(inp.phiedge)) / (np.pi * a ** 2)
# every coil carries the current that gives B0 on the axis; all currents stay free,
# the pressure-balance constraint pins the field strength
I0 = 2 * np.pi * R0 * B0 / (4e-7 * np.pi * 2 * nfp * ncoils)
base_curves = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
# Current(1.0) * I0 keeps the current dofs O(1) rather than ~ MA
base_currents = [Current(1.0) * I0 for _ in range(ncoils)]
coils = coils_via_symmetries(base_curves, base_currents, nfp, True)
curves = [c.curve for c in coils]
print(f"{len(coils)} circular coils: R0 = {R0:.3f} m, R1 = {R1:.3f} m, {I0 / 1e6:.2f} MA each")

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

# WeightedSum (not "+") so that f_QS and the coil-plasma distance share ONE VMEX adjoint
objective = WeightedSum([
    (1.0, f_qs),
    (LENGTH_WEIGHT, sum(QuadraticPenalty(J, LENGTH_MAX, "max") for J in Jls)),
    (CC_WEIGHT, Jccdist),
    (CS_WEIGHT, Jcsdist),
])
al = AugmentedLagrangian(objective, [interface], penalty=PENALTY0, eta0=ETA0)

names = list(al.dof_names)
plasma_mask = np.array([n.startswith(plasma.name + ":") for n in names])
x_seed = al.x.copy()
D = np.full(x_seed.size, COIL_STEP)
D[plasma_mask] = PLASMA_STEP * np.asarray(plasma.problem.scales)[plasma.dofs_free_status]
res0 = interface.residuals()
print(f"dofs: {x_seed.size} = {plasma_mask.sum()} plasma + {(~plasma_mask).sum()} coil; "
      f"constraints: {res0['bnormal'].size} B·n harmonics + 1 pressure balance")
print(f"seed: f_QS = {f_qs.J():.3e}, max |B·n harmonic| = {np.max(np.abs(res0['bnormal'])):.3e}, "
      f"rms(B·n/B) = {interface.rms_bnormal_over_B():.3e}, pressure balance = {res0['pressure_balance'][0]:+.3e}, "
      f"coil-plasma distance = {Jcsdist.shortest_distance():.3f} m")

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
# Plasma gradients are VMEX adjoints of the frozen residual; re-solve finite
# differences agree with them only to the solver tolerance and the ~2 %
# virtual-casing caveat above, so the error stops shrinking at small eps.
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
    print(f"[outer {record['outer']}] {record['inner_nit']} inner it ({time.time() - t_start:.0f}s): "
          f"L = {record['L']:.4e}, f_QS = {f_qs.J():.4e}, max|B·n harmonic| = {np.max(np.abs(res['bnormal'])):.2e}, "
          f"rms(B·n/B) = {interface.rms_bnormal_over_B():.2e}, pressure balance = {res['pressure_balance'][0]:+.2e}, "
          f"lengths = {[round(float(J.J()), 3) for J in Jls]}, coil-plasma = {Jcsdist.shortest_distance():.3f} m, "
          f"rho = {record['penalties']}", flush=True)


# x0: the solver works in u, which starts at 0 (not at al.x)
u, record = solve_augmented_lagrangian(al, fun=fun, x0=u0, outer_maxiter=OUTER_MAXITER, inner_maxiter=INNER_MAXITER,
                                       ctol=CTOL, gtol=GTOL, callback=report)
print("converged" if record["converged"] else f"stopped after {OUTER_MAXITER} outer iterations")
fun(u)
al.x = x_seed + D * good["u"]      # leave every object at the last accepted point

print("""
################################################################################
### Save the results ###########################################################
################################################################################
""")
curves_to_vtk(curves, out_dir / "curves_opt")
save(coils, out_dir / "coils_opt.json")
plasma.vmec_input().to_indata(str(out_dir / "input.combined_stage_opt"))
vmex.write_wout(str(out_dir / "wout_combined_stage_opt.nc"), plasma.equilibrium().wout)

# B·n of the coils alone on the final boundary, relative to |B_coil| (in the
# simsopt example's form; at finite beta it should equal -B_plasma·n, not 0)
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
