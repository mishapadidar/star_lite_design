#!/usr/bin/env python
r"""
simsopt's ``examples/2_Intermediate/stage_two_optimization_finite_beta.py``
with a Star_Lite device swapped in: a stage-II coil optimisation for
designA at volume-averaged beta = 1 %.

The equilibrium is FIXED. VMEC solves designA's boundary at beta = 1 %, and
a virtual casing calculation (one forward evaluation) splits the plasma's own
field off the total field on the boundary, giving the target normal field
B_External·n that the coils must produce. Since the plasma is not a vacuum,
this target is nonzero.

The objective is given by

    J = (1/2) ∫ |B_{BiotSavart}·n - B_{External}·n|^2 ds / (1/2) ∫ |B_{BiotSavart}|^2 ds
        + LENGTH_PENALTY * Σ max(CurveLength - LENGTH_MAX, 0)^2

Changes to the simsopt example, all for Star_Lite's scale:

* plasma: designA's boundary with p(s) = PRES_SCALE (1 - s) calibrated to
  beta = 1 % (``inputs/input.designA_beta1``, made with
  ``tools/make_beta_seed.py``; zero net toroidal current). VMEC is run here
  if its wout file does not exist yet;
* coils: 3 circular coils per half period of radius 0.3 m around designA's
  R0 = 0.5 m (W7-X's 5 of radius 1.25 m around 5.5 m);
* the length penalty caps each coil at LENGTH_MAX instead of holding it at its
  initial length: 3 circles of 1.89 m per half period are far shorter than
  designA's own coils (~3 m), and pinned there the fit stalls at
  <|B·n - B_External·n|>/|B| = 9e-3, barely below designA's own coils;
* SquaredFlux uses ``definition="normalized"``: the plain quadratic flux
  scales with |B|^2, and at designA's 0.09 T it is ~1000x smaller than at
  W7-X's 2.5 T, which would change the balance against the length penalty;
* for reference, designA's own (vacuum-optimised) coils are scored against
  the same beta = 1 % target.

The combined-stage version, where the boundary moves too and the virtual
casing is re-evaluated at every iterate, is combined_stage_designA_beta1.py.
"""

import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt import load, save
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import CurveLength, curves_to_vtk, create_equally_spaced_curves, SurfaceRZFourier
from simsopt.mhd import VirtualCasing, Vmec
from simsopt.objectives import QuadraticPenalty, SquaredFlux

# Number of unique coil shapes, i.e. the number of coils per half field period:
# (Since the configuration has nfp = 2 and stellarator symmetry, multiply ncoils by 2 * 2 to get the total number of coils.)
ncoils = 3

# Major radius for the initial circular coils:
R0 = 0.5

# Minor radius for the initial circular coils (designA's boundary reaches |Z| = 0.15 m):
R1 = 0.3

# Number of Fourier modes describing each Cartesian component of each coil:
order = 6

# Weight on the curve length penalties in the objective function, and the
# largest coil length (designA's own coils are 3.03 / 3.09 m):
LENGTH_PENALTY = 1e0
LENGTH_MAX = 3.5

# Number of iterations to perform:
MAXITER = 500

# VMEC input for the desired boundary magnetic surface (designA, beta = 1 %):
vmec_input = Path(__file__).resolve().parent / "inputs" / "input.designA_beta1"

# designA's own coils, for comparison (entry 0 of the archive = its BoozerSurfaces):
design_archive = Path(__file__).resolve().parents[2] / "convert" / "designA_after_scaled.json"

# Resolution on the plasma boundary surface:
# nphi is the number of grid points in 1/2 a field period.
nphi = 32
ntheta = 32

# Resolution for the virtual casing calculation:
vc_src_nphi = 80

#######################################################
# End of input parameters.
#######################################################

# Directory for output
out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

# Solve the equilibrium once (VMEC2000 through simsopt), unless its wout file exists.
vmec_file = out_dir / "wout_designA_beta1.nc"
if not vmec_file.is_file():
    print('Running VMEC on', vmec_input)
    equil = Vmec(str(vmec_input))
    equil.run()
    os.replace(equil.output_file, vmec_file)
print('wout file:', vmec_file)

# Once the virtual casing calculation has been run once, the results
# can be used for many coil optimizations. Therefore here we check to
# see if the virtual casing output file alreadys exists. If so, load
# the results, otherwise run the virtual casing calculation and save
# the results.
head, tail = os.path.split(vmec_file)
vc_filename = os.path.join(head, tail.replace('wout', 'vcasing'))
print('virtual casing data file:', vc_filename)
if os.path.isfile(vc_filename):
    print('Loading saved virtual casing result')
    vc = VirtualCasing.load(vc_filename)
else:
    # Virtual casing must not have been run yet.
    print('Running the virtual casing calculation')
    vc = VirtualCasing.from_vmec(str(vmec_file), src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta,
                                 filename=vc_filename)

# Initialize the boundary magnetic surface:
s = SurfaceRZFourier.from_wout(str(vmec_file), range="half period", nphi=nphi, ntheta=ntheta)
total_current = Vmec(str(vmec_file)).external_current() / (2 * s.nfp)
print(f"beta = {Vmec(str(vmec_file)).wout.betatotal:.3%}, poloidal coil current per half period = {total_current / 1e3:.1f} kA")

# Create the initial coils:
base_curves = create_equally_spaced_curves(ncoils, s.nfp, stellsym=True, R0=R0, R1=R1, order=order, numquadpoints=128)
# Since we know the total sum of currents, we only optimize for ncoils-1
# currents, and then pick the last one so that they all add up to the correct
# value.
base_currents = [Current(total_current / ncoils * 1e-4) * 1e4 for _ in range(ncoils-1)]
# Above, the factors of 1e-4 and 1e4 are included so the current
# degrees of freedom are O(1) rather than ~ 10 kA.  The optimization
# algorithm may not perform well if the dofs are scaled badly.
total_current = Current(total_current)
total_current.fix_all()
base_currents += [total_current - sum(base_currents)]

coils = coils_via_symmetries(base_curves, base_currents, s.nfp, True)
bs = BiotSavart(coils)

bs.set_points(s.gamma().reshape((-1, 3)))
curves = [c.curve for c in coils]
curves_to_vtk(curves, out_dir / "curves_init")
pointData = {"B_N": np.sum(bs.B().reshape((nphi, ntheta, 3)) * s.unitnormal(), axis=2)[:, :, None]}
s.to_vtk(out_dir / "surf_init", extra_data=pointData)

# For reference: designA's own coils against the same target.
bs_designA = BiotSavart(load(str(design_archive))[0][0].biotsavart.coils)
bs_designA.set_points(s.gamma().reshape((-1, 3)))
B_A = bs_designA.B().reshape((nphi, ntheta, 3))
BdotN_A = np.abs(np.sum(B_A * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(B_A, axis=2)
BextN = np.abs(vc.B_external_normal) / np.linalg.norm(B_A, axis=2)
print(f"target: ⟨|B_External·n|⟩/|B| = {np.mean(BextN):.1e} (max {np.max(BextN):.1e}); "
      f"designA's own coils miss it by ⟨|B·n - B_External·n|⟩/|B| = {np.mean(BdotN_A):.1e} (max {np.max(BdotN_A):.1e})")

# Define the objective function:
Jf = SquaredFlux(s, bs, target=vc.B_external_normal, definition="normalized")
Jls = [CurveLength(c) for c in base_curves]

# Form the total objective function. To do this, we can exploit the
# fact that Optimizable objects with J() and dJ() functions can be
# multiplied by scalars and added:
JF = Jf \
    + LENGTH_PENALTY * sum(QuadraticPenalty(Jls[i], LENGTH_MAX, "max") for i in range(len(base_curves)))

# We don't have a general interface in SIMSOPT for optimisation problems that
# are not in least-squares form, so we write a little wrapper function that we
# pass directly to scipy.optimize.minimize


def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    jf = Jf.J()
    Bbs = bs.B().reshape((nphi, ntheta, 3))
    BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
    BdotN_mean = np.mean(BdotN)
    BdotN_max = np.max(BdotN)
    outstr = f"J={J:.1e}, Jf={jf:.1e}, ⟨|B·n|⟩={BdotN_mean:.1e}, max(|B·n|)={BdotN_max:.1e}"
    cl_string = ", ".join([f"{J.J():.2f}" for J in Jls])
    outstr += f", Len=sum([{cl_string}])={sum(J.J() for J in Jls):.2f}"
    outstr += f", ║∇J║={np.linalg.norm(grad):.1e}"
    print(outstr)
    return J, grad


print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = JF.x
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
    J1, _ = f(dofs + eps*h)
    J2, _ = f(dofs - eps*h)
    print("err", (J1-J2)/(2*eps) - dJh)

print("""
################################################################################
### Run the optimisation #######################################################
################################################################################
""")
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor': 300, 'ftol': 1e-20, 'gtol': 1e-20}, tol=1e-20)
dofs = res.x
curves_to_vtk(curves, out_dir / "curves_opt")
save(coils, out_dir / "coils_opt.json")
Bbs = bs.B().reshape((nphi, ntheta, 3))
BdotN = np.abs(np.sum(Bbs * s.unitnormal(), axis=2) - vc.B_external_normal) / np.linalg.norm(Bbs, axis=2)
pointData = {"B_N": BdotN[:, :, None]}
s.to_vtk(out_dir / "surf_opt", extra_data=pointData)
print(f"final: ⟨|B·n - B_External·n|⟩/|B| = {np.mean(BdotN):.1e} (max {np.max(BdotN):.1e}); "
      f"designA's own coils: {np.mean(BdotN_A):.1e}")
