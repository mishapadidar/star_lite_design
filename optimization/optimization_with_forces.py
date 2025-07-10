#!/usr/bin/env python3

import os
import numpy as np
from scipy.optimize import minimize

from simsopt.field.coil import ScaledCurrent 
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt._core import load, save
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, ArclengthVariation, 
                        #  CurveBoozerSurfaceDistance,
                        #  BoozerResidual,
                         CurveRZFourier)
from star_lite_design.utils.boozer_surface_utils import BoozerResidual, CurveBoozerSurfaceDistance
from simsopt.field.selffield import regularization_circ
from simsopt.field.force import coil_force, LpCurveForce
from axis_field_strength_penalty import AxisFieldStrengthPenalty
from curve_vessel_distance import CurveVesselDistance
import pandas as pd


def find_magnetic_axis(biotsavart, r0, z0, nfp, order, stellsym):
    from scipy.spatial.distance import cdist
    from scipy.optimize import fsolve
    
    n = r0.size
    if n % 2 == 0:
        n+=1
    
    length = 2*np.pi/nfp
    points = np.linspace(0, length, n, endpoint=False).reshape((n, 1))
    oneton = np.asarray(range(0, n)).reshape((n, 1))
    fak = 2*np.pi / length
    dists = fak * cdist(points, points, lambda a, b: a-b)
    np.fill_diagonal(dists, 1e-10)  # to shut up the warning
    if n % 2 == 0:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.tan(0.5 * dists)
    else:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.sin(0.5 * dists)

    np.fill_diagonal(D, 0)
    D *= fak
    phi = points

    def build_residual(rz):
        inshape = rz.shape
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        B = biotsavart.B()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        Br = np.cos(phi)*Bx + np.sin(phi)*By
        Bphi = np.cos(phi)*By - np.sin(phi)*Bx
        residual_r = D @ r - r * Br / Bphi
        residual_z = D @ z - r * Bz / Bphi
        return np.vstack((residual_r, residual_z)).reshape(inshape)

    def build_jacobian(rz):
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        GradB = biotsavart.dB_by_dX()
        B = biotsavart.B()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        dxBx = GradB[:, 0, 0].reshape((n, 1))
        dyBx = GradB[:, 1, 0].reshape((n, 1))
        dzBx = GradB[:, 2, 0].reshape((n, 1))
        dxBy = GradB[:, 0, 1].reshape((n, 1))
        dyBy = GradB[:, 1, 1].reshape((n, 1))
        dzBy = GradB[:, 2, 1].reshape((n, 1))
        dxBz = GradB[:, 0, 2].reshape((n, 1))
        dyBz = GradB[:, 1, 2].reshape((n, 1))
        dzBz = GradB[:, 2, 2].reshape((n, 1))
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        Br = cosphi*Bx + sinphi*By
        Bphi = cosphi*By - sinphi*Bx
        drBr = cosphi*cosphi * dxBx + cosphi*sinphi*dyBx + sinphi*cosphi*dxBy + sinphi*sinphi*dyBy
        dzBr = cosphi*dzBx + sinphi*dzBy
        drBphi = cosphi*cosphi*dxBy + cosphi*sinphi*dyBy - sinphi*cosphi*dxBx - sinphi*sinphi*dyBx
        dzBphi = cosphi*dzBy - sinphi*dzBx
        drBz = cosphi * dxBz + sinphi*dyBz
        # residual_r = D @ r - r * Br / Bphi
        # residual_z = D @ z - r * Bz / Bphi
        dr_resr = D + np.diag((-Br/Bphi - r*drBr/Bphi + r*Br*drBphi/Bphi**2).reshape((n,)))
        dz_resr = np.diag((-r*dzBr/Bphi + r*Br*dzBphi/Bphi**2).reshape((n,)))
        dr_resz = np.diag((-Bz/Bphi - r*drBz/Bphi + r*Bz*drBphi/Bphi**2).reshape((n,)))
        dz_resz = D + np.diag((-r*dzBz/Bphi + r*Bz*dzBphi/Bphi**2).reshape((n,)))
        return np.block([[dr_resr, dz_resr], [dr_resz, dz_resz]])
    
    x0 = np.vstack((r0, z0))
    
    soln = fsolve(build_residual, x0, fprime=build_jacobian, xtol=1e-13)
    
    res = build_residual(soln)
    norm_res = np.sqrt(np.sum(res**2))
    ma_success = norm_res < 1e-10

    xyz = np.hstack((soln[:n, None]*np.cos(phi), soln[:n, None]*np.sin(phi), soln[n:, None]))
    quadpoints = np.linspace(0, 1/nfp, n, endpoint=False)
    ma_fp = CurveRZFourier(quadpoints, order, nfp, stellsym)
    ma_fp.least_squares_fit(xyz)

    quadpoints = np.linspace(0, nfp, nfp*n, endpoint=False)
    ma_ft = CurveRZFourier(quadpoints, order, nfp, stellsym)
    ma_ft.x = ma_fp.x

    return ma_fp, ma_ft, ma_success, soln[:n], soln[n:]


"""
The script optimizes the Star_lite device to have 3 configurations with different iota values, and low coil forces.
This script was run as a second stage of optimization, after star_lite was optimized for quasi-symmetry etc.
"""


print("Running Optimization with coil forces")
print("================================")

design = 'B' # A or B
# load the boozer surfaces (1 per Current configuration, so 3 total.)
data = load(f"../designs/design{design}_after_scaled.json")
boozer_surfaces = data[0] # BoozerSurfaces
iota_Gs = data[1] # (iota, G) pairs
axis_curves = data[2] # magnetic axis CurveRZFouriers
# x_point_curves = data[3] # X-point CurveRZFouriers

for ii, bsurf in enumerate(boozer_surfaces):
    # rebuild the boozer surface and populate the res attribute
    bsurf = BoozerSurface(bsurf.biotsavart, bsurf.surface, bsurf.label, bsurf.targetlabel, options={'newton_tol':1e-13, 'newton_maxiter':20})
    # bsurf = BoozerSurface(bsurf.biotsavart, bsurf.surface, bsurf.label, bsurf.targetlabel, constraint_weight=1, options={'newton_tol':1e-12})

    bsurf.run_code(iota_Gs[ii][0], iota_Gs[ii][1])
    boozer_surfaces[ii] = bsurf

# get the x-points
bs_list = [bsurf.biotsavart for bsurf in boozer_surfaces]
coils_list = [bs.coils for bs in bs_list]
xpoints = []
for bs in bs_list:
    N_axis=121
    curve_stellsym=False
    nfp=2
    order=16
    r0 = 0.55967771 * np.ones(N_axis)
    z0 = 0.15932784 * np.ones(N_axis)
    xp0_fp, xp0_ft, x_success, rx0, zx0= find_magnetic_axis(bs, r0, z0, nfp, order, curve_stellsym)
    xpoints.append([rx0, zx0])


# get the base curves
biotsavart = boozer_surfaces[0].biotsavart
coils = biotsavart.coils
curves = [c.curve for c in coils]
if design == "A":
    base_curve_idx = [0, 1, 4]
    base_curves = [curves[i] for i in base_curve_idx]
elif design == "B":
    base_curve_idx = [0, 2]
    base_curves = [curves[i] for i in base_curve_idx]

# target value of axis field for rescaling
B_axis_target = 0.0875

## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
mr = MajorRadius(boozer_surfaces[0])
ls = [CurveLength(c) for c in base_curves]
brs = [BoozerResidual(boozer_surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.15
CC_WEIGHT = 1000

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.15
CS_WEIGHT = 1e2

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 4.3352595043*2
CURVATURE_WEIGHT = 3e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 20.
MSC_WEIGHT = 4e-3

AL_WEIGHT = 1.
LENGTH_WEIGHT = 1e-1

IOTAS_WEIGHT=1e4
MR_WEIGHT=1e3

BR_WEIGHT=1e7

coil_minor_radius = 0.054 # 54mm
force_order = 2 
FORCE_WEIGHT = 1e-11

LENGTH_THRESHOLD = 4.0
J_major_radius = QuadraticPenalty(mr, 0.5, 'identity')  # target major radius is that computed on the initial surface
#J_iotas = sum([QuadraticPenalty(Iotas(boozer_surface), iota, 'identity') for boozer_surface, iota in zip(boozer_surfaces, [0.1882775, 0.10, 0.30])]) # target rotational transform is that computed on the initial surface
J_iotas = sum([QuadraticPenalty(Iotas(boozer_surface), boozer_surface.res['iota'], 'identity') for boozer_surface in boozer_surfaces]) # target rotational transform is that computed on the initial surface
nonQS_list = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]
print([J.J()**0.5 for J in nonQS_list])
J_nonQSRatio = (1./len(boozer_surfaces)) * sum(nonQS_list)

Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD)
Jcsdist = CurveBoozerSurfaceDistance(curves, boozer_surfaces[0], CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jal = sum(ArclengthVariation(curve) for curve in base_curves)

# coil-to-vessel distance
df = pd.read_csv("../designs/sheetmetal_chamber.csv")
X_vessel = df.values
CV_WEIGHT = 1e9
CV_THRESHOLD = 0.06 # 0.054 minimum
Jcvd = CurveVesselDistance(base_curves, X_vessel, CV_THRESHOLD)


# coil forces on all current groups
coil_force_list = []
for bbsurf in boozer_surfaces:
    # only compute force on base coils
    coil_force_list += [LpCurveForce(bbsurf.biotsavart.coils[i], bbsurf.biotsavart.coils, regularization_circ(coil_minor_radius), p=force_order) for i in base_curve_idx]
Jforce = sum(coil_force_list)

length_penalty = LENGTH_WEIGHT * sum([QuadraticPenalty(Jl, LENGTH_THRESHOLD, "max") for Jl in Jls])
cc_penalty = CC_WEIGHT * Jccdist
cs_penalty =  CS_WEIGHT * Jcsdist
curvature_penalty = CURVATURE_WEIGHT * sum(Jcs)
msc_penalty = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
Jbrs = sum(brs)

# penalty on deviation from target mean field strength
AFS_WEIGHT = 1000.0
Jafs = sum([AxisFieldStrengthPenalty(boozer_surfaces[ii].biotsavart, axis_curves[ii].gamma(), B_target=B_axis_target) for ii in range(len(boozer_surfaces))])



# sum the objectives together
JF = (J_nonQSRatio + IOTAS_WEIGHT * J_iotas + MR_WEIGHT * J_major_radius
    + length_penalty
    + cc_penalty
    + cs_penalty
    + curvature_penalty
    + msc_penalty
    + AL_WEIGHT * Jal
    + BR_WEIGHT * Jbrs
    + FORCE_WEIGHT * Jforce
    + AFS_WEIGHT * Jafs
    + CV_WEIGHT * Jcvd
    )


# rescale currents to achieve target axis field
print("")
print('Rescaling currents to achieve target axis field:', B_axis_target)
for ii, bbsurf in enumerate(boozer_surfaces):
    xyz = axis_curves[ii].gamma()

    # compute B on axis
    bbsurf.biotsavart.set_points(xyz)
    B_axis = boozer_surfaces[ii].biotsavart.B()
    B_mean = np.mean(np.linalg.norm(B_axis, axis=1))
    print('Axis field before scaling', B_mean)
    # rescale currents
    scale = B_axis_target / B_mean
    # change current
    for jj in base_curve_idx:
        bbsurf.biotsavart.coils[jj].current.unfix_all()
        bbsurf.biotsavart.coils[jj].current.x = scale * bbsurf.biotsavart.coils[jj].current.x

    B_axis = boozer_surfaces[ii].biotsavart.B()
    B_mean = np.mean(np.linalg.norm(B_axis, axis=1))
    print('Axis field after scaling', B_mean)

# fix some currents
for bbsurf in boozer_surfaces:
    bbsurf.biotsavart.coils[base_curve_idx[0]].current.fix_all()
    dn = bbsurf.biotsavart.dof_names
    print('free currents:', [c for c in dn if 'current' in c.lower() ])

print("n_dofs", len(bbsurf.x))

# Directory for output
OUT_DIR = f"./output/design{design}/force_weight_{FORCE_WEIGHT}/"
os.makedirs(OUT_DIR, exist_ok=True)

curves_to_vtk(curves, OUT_DIR + "curves_init")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_init_{idx}")

# save these as a backup in case the boozer surface Newton solve fails
res_list = [{'sdofs': boozer_surface.surface.x.copy() , 'iota': boozer_surface.res['iota'], 'G': boozer_surface.res['G'], 'rx0':rz[0].copy(), 'zx0': rz[1].copy()} for rz, boozer_surface in zip(xpoints, boozer_surfaces)]
dat_dict = {'J': JF.J(), 'dJ': JF.dJ().copy()}

def callback(dofs):
    for res, boozer_surface, rz in zip(res_list, boozer_surfaces, xpoints):
        res['sdofs'] = boozer_surface.surface.x.copy()
        res['rx0'] = rz[0].copy()
        res['zx0'] = rz[1].copy()
        res['iota'] =  boozer_surface.res['iota']
        res['G'] = boozer_surface.res['G']
    
    dat_dict['J'] = JF.J()
    dat_dict['dJ'] = JF.dJ().copy()

    outstr = f"J={dat_dict['J']:.1e}, J_nonQSRatio={J_nonQSRatio.J():.2e}, mr={mr.J():.2e} ({J_major_radius.J()*MR_WEIGHT:.1e})"
    outstr += f", Jforce={Jforce.J():.2e}"
    outstr += f", Jafs={Jafs.J():.2e} ({AFS_WEIGHT*Jafs.J():.1e})"
    outstr += f", Jcvd={Jcvd.J():.2e} ({Jcvd.shortest_distance():.4f})"
    iota_string = ", ".join([f"{res['iota']:.3f}" for res in res_list])
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    brs_string = ", ".join(f"{J.J():.1e}" for J in brs)
    xp_string = ", ".join(f"{rz[0][0]:.2e} {rz[1][0]:.2e} " for rz in xpoints)
    outstr += f", iotas=[{iota_string}] ({IOTAS_WEIGHT*J_iotas.J():.1e}), Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f} ({length_penalty.J():.1e}), ϰ=[{kap_string}] ({curvature_penalty.J():.1e}), ∫ϰ²/L=[{msc_string}] ({msc_penalty.J():.1e})"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f} ({cc_penalty.J():.1e}), C-S-Sep={Jcsdist.shortest_distance():.2f} ({cs_penalty.J():.1e})"
    outstr += f", al var={Jal.J():.2e}"
    outstr += f", brs={brs_string} {BR_WEIGHT*Jbrs.J():.2e}"
    outstr += f", xps={xp_string}"
    outstr += f", ║∇J║={np.linalg.norm(dat_dict['dJ']):.1e}"
    print("")
    print(outstr)

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    
    axis_success = []
    for res, rz, coils_local, bs_local in zip(res_list, xpoints, coils_list, bs_list):
        curve_stellsym=False
        nfp=2
        order=16
        _, _, success, rx0, zx0 = find_magnetic_axis(bs_local, res['rx0'], res['zx0'], nfp, order, curve_stellsym)
        print(f"axis solve: {success}")
        rz[0] = rx0.copy()
        rz[1] = zx0.copy()
        axis_success.append(success)

    if not np.all([boozer_surface.res['success'] and (not boozer_surface.surface.is_self_intersecting()) for boozer_surface in boozer_surfaces]) or not np.all(axis_success):
        print('failed')
        # failed, so reset back to previous surface and return a large value
        # of the objective.  The purpose is to trigger the line search to reduce
        # the step size.
        J = 1e3
        grad = -dat_dict['dJ']
        for res, boozer_surface in zip(res_list,  boozer_surfaces): 
            boozer_surface.surface.x = res['sdofs']
            boozer_surface.res['iota'] = res['iota']
            boozer_surface.res['G'] = res['G']
    return J, grad


# print("""
# ################################################################################
# ### Perform a Taylor test ######################################################
# ################################################################################
# """)
# f = fun
# dofs = JF.x.copy()
# np.random.seed(1)
# # h = np.loadtxt(OUT_DIR+'h.txt')
# h = 1.0
# J0, dJ0 = f(dofs)
# dJh = sum(dJ0 * h)
# for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
#    J1, _ = f(dofs + 2*eps*h)
#    J2, _ = f(dofs + eps*h)
#    J3, _ = f(dofs - eps*h)
#    J4, _ = f(dofs - 2*eps*h)
#    print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/np.linalg.norm(dJh))
# quit()


print("""
################################################################################
### Initial performance #######################################################
################################################################################
""")
print("Initial objective function value: %.2e"%(JF.J()))
print('J_nonQSRatio = %.2e'%(J_nonQSRatio.J()))
print("Jforce value: %.2e"%(Jforce.J()))
print("Jafs value: %.2e"%(Jafs.J()))

# print out coil forces
for iota_group, bbsurf in enumerate(boozer_surfaces):
    total = 0
    for ii in base_curve_idx:
        force = np.linalg.norm(coil_force(bbsurf.biotsavart.coils[ii], bbsurf.biotsavart.coils, regularization_circ(coil_minor_radius)), axis=1)
        # print(f"group {iota_group}; max force on coil {ii}: %.2f"%(np.max(np.abs(force))))
        print(f"group {iota_group}; max force on coil {np.max(np.abs(force)):.2f}; mean force on coil {np.mean((force)):.2f}; mean_squared force on coil {np.mean((force**2)):.2f}")
        total += np.mean((force**2))
    print(f"group {iota_group}; total mean squared force on coils {total:.2f}")

J0, dJ0 = fun(JF.x.copy())
print("Norm gradient", np.linalg.norm(dJ0))
print("Norm QS gradient", np.linalg.norm(J_nonQSRatio.dJ()))

# print the currents
for bbsurf in boozer_surfaces:
    for ii in base_curve_idx:
        print(f"Coil {ii} current: {bbsurf.biotsavart.coils[ii].current.full_x} A")


print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")
# Number of iterations to perform:
MAXITER=3000

dofs = JF.x
callback(dofs)
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor':20}, tol=1e-15, callback=callback)
print(res.message)

# recompute the axis
for ii, bbsurf in enumerate(boozer_surfaces):
    # axis
    xyz = axis_curves[ii].gamma()

    # recompute axis
    N_axis=121
    curve_stellsym=False
    nfp=2
    order=16
    r0 = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    z0 = xyz[:, 2]
    xp0_fp, xp0_ft, x_success, rx0, zx0= find_magnetic_axis(bs, r0, z0, nfp, order, curve_stellsym)
    axis_curves[ii] = xp0_fp

# rescale currents to achieve target axis field
print("")
print('Rescaling currents to achieve target axis field:', B_axis_target)
for ii, bbsurf in enumerate(boozer_surfaces):

    # axis
    xyz = axis_curves[ii].gamma()

    # compute B on axis
    bbsurf.biotsavart.set_points(xyz)
    B_axis = boozer_surfaces[ii].biotsavart.B()
    B_mean = np.mean(np.linalg.norm(B_axis, axis=1))
    print('Axis field before scaling', B_mean)
    # rescale currents
    scale = B_axis_target / B_mean
    # change current
    for jj in base_curve_idx:
        bbsurf.biotsavart.coils[jj].current.unfix_all()
        bbsurf.biotsavart.coils[jj].current.x = scale * bbsurf.biotsavart.coils[jj].current.x

    B_axis = boozer_surfaces[ii].biotsavart.B()
    B_mean = np.mean(np.linalg.norm(B_axis, axis=1))
    print('Axis field after scaling', B_mean)

# recompute the x-points curves
bs_list = [bsurf.biotsavart for bsurf in boozer_surfaces]
coils_list = [bs.coils for bs in bs_list]
xpoint_curves = []
for bs in bs_list:
    N_axis=121
    curve_stellsym=False
    nfp=2
    order=16
    r0 = 0.55967771 * np.ones(N_axis)
    z0 = 0.15932784 * np.ones(N_axis)
    xp0_fp, xp0_ft, x_success, rx0, zx0= find_magnetic_axis(bs, r0, z0, nfp, order, curve_stellsym)
    xpoint_curves.append(xp0_fp)

# get the iotas and Gs
iota_Gs = []
for ii, bbsurf in enumerate(boozer_surfaces):
    iota_Gs.append((bbsurf.res['iota'], bbsurf.res['G']))

curves_to_vtk(curves, OUT_DIR + "curves_opt")
curves_to_vtk(xpoint_curves, OUT_DIR + f"xpoint_curves_opt")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}")
save([boozer_surfaces, iota_Gs, axis_curves, xpoint_curves], OUT_DIR + f'design{design}_after_forces_opt.json')

# print the currents
for bbsurf in boozer_surfaces:
    for ii in base_curve_idx:
        print(f"Coil {ii} current: {bbsurf.biotsavart.coils[ii].current.full_x} A")

print("")
print("End of optimization")
print("================================")


print("""
################################################################################
### Final performance #######################################################
################################################################################
""")
print("Final objective function value: %.2e"%(JF.J()))
print('J_nonQSRatio = %.2e'%(J_nonQSRatio.J()))
print("Jforce value: %.2e"%(Jforce.J()))
# print out coil forces
for iota_group, bbsurf in enumerate(boozer_surfaces):
    total = 0
    for ii in base_curve_idx:
        force = np.linalg.norm(coil_force(bbsurf.biotsavart.coils[ii], bbsurf.biotsavart.coils, regularization_circ(coil_minor_radius)), axis=1)
        # print(f"group {iota_group}; max force on coil {ii}: %.2f"%(np.max(np.abs(force))))
        print(f"group {iota_group}; max force on coil {np.max(np.abs(force)):.2f}; mean force on coil {np.mean((force)):.2f}; mean_squared force on coil {np.mean((force**2)):.2f}")
        total += np.mean((force**2))
    print(f"group {iota_group}; total mean squared force on coils {total:.2f}")


# print("""
# ################################################################################
# ### Perform a Taylor test ######################################################
# ################################################################################
# """)
# dofs = res.x.copy()
# np.random.seed(1)
# h = np.random.uniform(size=dofs.shape)
# J0, dJ0 = fun(dofs)
# dJh = sum(dJ0 * h)
# for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
#     J1, _ = fun(dofs + 2*eps*h)
#     J2, _ = fun(dofs + eps*h)
#     J3, _ = fun(dofs - eps*h)
#     J4, _ = fun(dofs - 2*eps*h)
#     print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/np.linalg.norm(dJh))
# np.savetxt(OUT_DIR+'h.txt', h)

