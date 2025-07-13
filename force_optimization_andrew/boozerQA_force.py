#!/usr/bin/env python3

# code corrections:
# lb, ub
# all currents should be unfixed

import os
import numpy as np
from scipy.optimize import minimize

from simsopt.field.coil import ScaledCurrent 
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas, RotatedCurve
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt._core import load, save
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, ArclengthVariation, 
                         CurveRZFourier, CurveXYZFourierSymmetries)
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.field.selffield import regularization_circ
import pandas as pd


from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.boozer_surface_utils import BoozerResidual, CurveBoozerSurfaceDistance
from star_lite_design.utils.curve_vessel_distance import CurveVesselDistance
from star_lite_design.utils.fieldline_vessel_distance import FieldLineVesselDistance
from star_lite_design.utils.modb_on_fieldline import ModB_on_FieldLine

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
axes_RZ = data[2] # magnetic axis CurveRZFouriers
xpoints_RZ = data[3] # X-point CurveRZFouriers

axes = []
for axis_RZ, boozer_surface in zip(axes_RZ, boozer_surfaces):
    stellsym=True
    nfp = 2
    order=16
    tmp = CurveXYZFourierSymmetries(axis_RZ.quadpoints, order, nfp, stellsym)
    tmp.least_squares_fit(axis_RZ.gamma())
    quadpoints = np.linspace(0, 1/nfp, 2*order+1, endpoint=False)
    axis = CurveXYZFourierSymmetries(quadpoints, order, nfp, stellsym)
    axis.x = tmp.x
    axis_fl = PeriodicFieldLine(BiotSavart(boozer_surface.biotsavart.coils), axis)
    axis_fl.run_code(CurveLength(axis_fl.curve).J())
    axes.append(axis_fl)

xpoints = []
for xpoint_RZ, boozer_surface in zip(xpoints_RZ, boozer_surfaces):
    stellsym=False
    nfp = 2
    order=16
    tmp = CurveXYZFourierSymmetries(xpoint_RZ.quadpoints, order, nfp, stellsym)
    tmp.least_squares_fit(xpoint_RZ.gamma())
    quadpoints = np.linspace(0, 1/nfp, 2*order+1, endpoint=False)
    xpoint = CurveXYZFourierSymmetries(quadpoints, order, nfp, stellsym)
    xpoint.x = tmp.x
    xpoint_fl = PeriodicFieldLine(BiotSavart(boozer_surface.biotsavart.coils), xpoint)
    xpoint_fl.run_code(CurveLength(xpoint_fl.curve).J())
    xpoints.append(xpoint_fl)

for ii, bsurf in enumerate(boozer_surfaces):
    # rebuild the boozer surface and populate the res attribute
    bsurf = BoozerSurface(bsurf.biotsavart, bsurf.surface, bsurf.label, bsurf.targetlabel, options={'newton_tol':1e-13, 'newton_maxiter':20})
    bsurf.run_code(iota_Gs[ii][0], iota_Gs[ii][1])
    boozer_surfaces[ii] = bsurf

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

# maximum current
current_bound = 60000 # [Amps]

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
FORCE_THRESHOLD = 4e3

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

XV_WEIGHT = 1e3
Jfvs = [FieldLineVesselDistance(xpoint, X_vessel, CV_THRESHOLD) for xpoint in xpoints]


# coil forces on all current groups
coil_force_list = []
for bbsurf in boozer_surfaces:
    # only compute force on base coils
    coil_force_list += [LpCurveForce(bbsurf.biotsavart.coils[i], bbsurf.biotsavart.coils, regularization_circ(coil_minor_radius), p=force_order, threshold=FORCE_THRESHOLD) for i in base_curve_idx]
Jforce = sum(coil_force_list)

length_penalty = LENGTH_WEIGHT * sum([QuadraticPenalty(Jl, LENGTH_THRESHOLD, "max") for Jl in Jls])
cc_penalty = CC_WEIGHT * Jccdist
cs_penalty =  CS_WEIGHT * Jcsdist
curvature_penalty = CURVATURE_WEIGHT * sum(Jcs)
msc_penalty = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
Jbrs = sum(brs)

# penalty on deviation from target mean field strength
AFS_WEIGHT = 1e4
Jafs = sum([QuadraticPenalty(ModB_on_FieldLine(axis, BiotSavart(boozer_surface.biotsavart.coils)), B_axis_target, 'identity') for axis, boozer_surface in zip(axes, boozer_surfaces)])


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
    + XV_WEIGHT * sum(Jfvs)
    )

# fix some currents
for bbsurf in boozer_surfaces:
    for coil in bbsurf.biotsavart.coils:
        coil.current.unfix_all()
    dn = bbsurf.biotsavart.dof_names
    print('free currents:', [c for c in dn if 'current' in c.lower() ])

    # set lower/upper bounds on the remaining currents
    for jj in base_curve_idx:
        bbsurf.biotsavart.coils[jj].current.upper_bounds = [current_bound/bbsurf.biotsavart.coils[jj].current.scale]
        bbsurf.biotsavart.coils[jj].current.lower_bounds = [-current_bound/bbsurf.biotsavart.coils[jj].current.scale]

# make sure coils are stellarator symmetric
for ii in base_curve_idx:
    c = boozer_surfaces[0].biotsavart.coils[ii].curve
    if isinstance(c, RotatedCurve):
        c = c.curve
    for df in c.local_dof_names:
        if ('xs' in df) or ('yc' in df) or ('zc' in df):
            c.fix(df)

print("n_dofs", len(bbsurf.x))


# Directory for output
OUT_DIR = f"./output/design{design}/force_weight_{FORCE_WEIGHT}/"
os.makedirs(OUT_DIR, exist_ok=True)

curves_to_vtk(curves, OUT_DIR + "curves_init")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_init_{idx}")

# save these as a backup in case the boozer surface Newton solve fails
res_list = [{'sdofs': boozer_surface.surface.x.copy() , 'iota': boozer_surface.res['iota'], 'G': boozer_surface.res['G']} for boozer_surface in boozer_surfaces]
axes_res_list = [{'adofs': axis.curve.x.copy() , 'length': axis.res['length']} for axis in axes]
xpoints_res_list = [{'xdofs': xpoint.curve.x.copy() , 'length': xpoint.res['length']} for xpoint in xpoints]
dat_dict = {'J': JF.J(), 'dJ': JF.dJ().copy()}

def callback(dofs):
    for res, boozer_surface in zip(res_list, boozer_surfaces):
        res['sdofs'] = boozer_surface.surface.x.copy()
        res['iota'] =  boozer_surface.res['iota']
        res['G'] = boozer_surface.res['G']
    for res, axis in zip(axes_res_list, axes):
        res['adofs'] = axis.curve.x.copy()
        res['length'] =  axis.res['length']
    for res, axis in zip(xpoints_res_list, xpoints):
        res['adofs'] = axis.curve.x.copy()
        res['length'] =  axis.res['length']
    
    dat_dict['J'] = JF.J()
    dat_dict['dJ'] = JF.dJ().copy()
    
    currents_list = [np.abs(coil.current.get_value()) for boozer_surface in boozer_surfaces for coil in boozer_surface.biotsavart.coils]
    
    max_force = -1
    # print out coil forces
    for iota_group, bbsurf in enumerate(boozer_surfaces):
        for ii in base_curve_idx:
            force = np.linalg.norm(coil_force(bbsurf.biotsavart.coils[ii], bbsurf.biotsavart.coils, regularization_circ(coil_minor_radius)), axis=1)
            max_force = max([max_force, np.max(np.abs(force))])

    outstr = f"J={dat_dict['J']:.1e}, J_nonQSRatio={J_nonQSRatio.J():.2e}, mr={mr.J():.2e} ({J_major_radius.J()*MR_WEIGHT:.1e})"
    outstr += f", Jforce={FORCE_WEIGHT*Jforce.J():.2e}, Maximum Force {max_force:.2e} (N)"
    outstr += f", Jafs={Jafs.J():.2e} ({AFS_WEIGHT*Jafs.J():.1e})"
    outstr += f", Jcvd={Jcvd.J():.2e} ({Jcvd.shortest_distance():.4f})"
    outstr += f", Jfvs={sum(Jfvs).J():.2e} ({min([Jfv.shortest_distance() for Jfv in Jfvs]):.4f})"
    iota_string = ", ".join([f"{res['iota']:.3f}" for res in res_list])
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    brs_string = ", ".join(f"{J.J():.1e}" for J in brs)
    outstr += f", iotas=[{iota_string}] ({IOTAS_WEIGHT*J_iotas.J():.1e}), Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f} ({length_penalty.J():.1e}), ϰ=[{kap_string}] ({curvature_penalty.J():.1e}), ∫ϰ²/L=[{msc_string}] ({msc_penalty.J():.1e})"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f} ({cc_penalty.J():.1e}), C-S-Sep={Jcsdist.shortest_distance():.2f} ({cs_penalty.J():.1e})"
    outstr += f", al var={Jal.J():.2e}"
    outstr += f", brs={brs_string} {BR_WEIGHT*Jbrs.J():.2e}"
    outstr += f", currents={min(currents_list):.2e} (A), {max(currents_list):.2e} (A)"
    outstr += f", ║∇J║={np.linalg.norm(dat_dict['dJ']):.1e}"
    print("")

    curves_to_vtk(curves, OUT_DIR + "curves_tmp")
    curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_tmp")
    curves_to_vtk([axis.curve for axis in axes], OUT_DIR + "ma_tmp")
    for idx, boozer_surface in enumerate(boozer_surfaces):
        boozer_surface.surface.to_vtk(OUT_DIR + f"surf_tmp_{idx}")
    save([boozer_surfaces, iota_Gs, axes, xpoints], OUT_DIR + f'design{design}_after_forces_tmp.json')
    print(outstr)

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    
    for axis in axes+xpoints:
        axis.run_code(axis.res['length']) 
    
    axis_success = [axis.res['success'] for axis in axes+xpoints]
    if (not np.all([boozer_surface.res['success'] \
            and (not boozer_surface.surface.is_self_intersecting()) for boozer_surface in boozer_surfaces])) \
            or not np.all(axis_success):
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
        for res, axis in zip(axes_res_list+xpoints_res_list,  axes+xpoints): 
            axis.curve.x = res['adofs']
            axis.res['length'] = res['length']

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
MAXITER=5000

lb = JF.lower_bounds
ub = JF.upper_bounds
bounds = np.vstack((lb, ub)).T

dofs = JF.x
callback(dofs)
res = minimize(fun, dofs, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': MAXITER, 'maxcor':100}, tol=1e-15, callback=callback)
print(res.message)

curves_to_vtk(curves, OUT_DIR + "curves_opt")
curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_opt")
curves_to_vtk([axis.curve for axis in axes], OUT_DIR + "ma_opt")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}")
save([boozer_surfaces, iota_Gs, axes, xpoints], OUT_DIR + f'design{design}_after_forces_opt.json')

