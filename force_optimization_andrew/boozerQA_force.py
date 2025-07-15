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
from simsopt.objectives import Weight
from simsopt._core import load, save
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, ArclengthVariation, 
                         CurveRZFourier, CurveXYZFourierSymmetries)
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.field.selffield import regularization_circ
import pandas as pd
from rich.console import Console
from rich.table import Column, Table

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
MODB_TARGET = 0.0875

# maximum current
current_bound = 60000 # [Amps]

## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
mr = MajorRadius(boozer_surfaces[0])
ls = [CurveLength(c) for c in base_curves]
brs = [BoozerResidual(boozer_surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]

# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.15
CC_WEIGHT = Weight(1e3)

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.15
CS_WEIGHT = Weight(1e2)

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 4.3352595043*2
CURVATURE_WEIGHT = Weight(3e-6)

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 20.
MSC_WEIGHT = Weight(4e-3)

AL_WEIGHT = Weight(1.)
LENGTH_WEIGHT = Weight(1e-1)

IOTAS_WEIGHT=Weight(1e4)
MR_WEIGHT=Weight(1e3)

BR_WEIGHT=Weight(1e7)

coil_minor_radius = 0.054 # 54mm
force_order = 2
FORCE_WEIGHT = Weight(1e-9)
FORCE_THRESHOLD = 4e3

LENGTH_THRESHOLD = 4.0
MR_TARGET = 0.5
J_major_radius = QuadraticPenalty(mr, MR_TARGET, 'identity')  # target major radius is that computed on the initial surface

IOTAS_LIST = [Iotas(boozer_surface) for boozer_surface in boozer_surfaces]
IOTAS_TARGET  = [boozer_surface.res['iota'] for boozer_surface in boozer_surfaces]
J_iotas = sum([QuadraticPenalty(IOTAS, IOTAS_TARGET, 'identity') for IOTAS, IOTAS_TARGET in zip(IOTAS_LIST, IOTAS_TARGET)]) # target rotational transform is that computed on the initial surface
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
CV_WEIGHT = Weight(1e9)
CV_THRESHOLD = 0.06 # 0.054 minimum
Jcvd = CurveVesselDistance(base_curves, X_vessel, CV_THRESHOLD)

XV_THRESHOLD = 0.06 # 0.054 minimum
XV_WEIGHT = Weight(1e6)
Jxvs = [FieldLineVesselDistance(xpoint, X_vessel, XV_THRESHOLD) for xpoint in xpoints]
xv_penalty = sum(Jxvs)

# coil forces on all current groups
coil_force_list = []
for bbsurf in boozer_surfaces:
    # only compute force on base coils
    coil_force_list += [LpCurveForce(bbsurf.biotsavart.coils[i], bbsurf.biotsavart.coils, regularization_circ(coil_minor_radius), p=force_order, threshold=FORCE_THRESHOLD) for i in base_curve_idx]
Jforce = sum(coil_force_list)

length_penalty = sum([QuadraticPenalty(Jl, LENGTH_THRESHOLD, "max") for Jl in Jls])
curvature_penalty = sum(Jcs)
msc_penalty = sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
Jbrs = sum(brs)

# penalty on deviation from target mean field strength
MODB_WEIGHT = Weight(1e4)
modBs = [ModB_on_FieldLine(axis, BiotSavart(boozer_surface.biotsavart.coils)) for axis, boozer_surface in zip(axes, boozer_surfaces)]
JmodB = sum([QuadraticPenalty(modB, MODB_TARGET, 'identity') for modB, axis, boozer_surface in zip(modBs, axes, boozer_surfaces)])


# sum the objectives together
JF = (J_nonQSRatio 
    + IOTAS_WEIGHT * J_iotas 
    + MR_WEIGHT * J_major_radius
    + LENGTH_WEIGHT * length_penalty
    + CC_WEIGHT * Jccdist
    + CS_WEIGHT * Jcsdist
    + CURVATURE_WEIGHT * curvature_penalty
    + MSC_WEIGHT * msc_penalty
    + AL_WEIGHT * Jal
    + BR_WEIGHT * Jbrs
    + FORCE_WEIGHT * Jforce
    + MODB_WEIGHT * JmodB
    + CV_WEIGHT * Jcvd
    + XV_WEIGHT * xv_penalty
    )

penalties = {'nonQS': J_nonQSRatio,
        'iotas':IOTAS_WEIGHT * J_iotas,
        'length':LENGTH_WEIGHT * length_penalty,
        'coil-to-coil': CC_WEIGHT * Jccdist,
        'coil-to-surface':CS_WEIGHT * Jcsdist,
        'curvature':CURVATURE_WEIGHT * curvature_penalty,
        'mean-squared curvature': MSC_WEIGHT * msc_penalty,
        'arclength':AL_WEIGHT * Jal,
        'Boozer residual': BR_WEIGHT * Jbrs,
        'force weight': FORCE_WEIGHT * Jforce,
        'modB': MODB_WEIGHT * JmodB,
        'coil-to-vessel':CV_WEIGHT * Jcvd,
        'x-point-to-vessel':XV_WEIGHT * xv_penalty
        }

states = {
        'iotas': IOTAS_LIST,
        'modB': modBs,
        'lengths':Jls,
        'major radius': [MajorRadius(boozer_surface) for boozer_surface in boozer_surfaces],
        'Boozer residuals': brs,
        'mean-squared curvature': Jmscs
        }

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
    
    currents_list = [np.abs(boozer_surface.biotsavart.coils[idx].current.get_value()) for boozer_surface in boozer_surfaces for idx in base_curve_idx]
    
    forces = []
    for bbsurf in boozer_surfaces:
        for idx in base_curve_idx:
            forces += [np.linalg.norm(coil_force(bbsurf.biotsavart.coils[idx], bbsurf.biotsavart.coils, regularization_circ(coil_minor_radius)), axis=1).max()]
    max_force = np.max(forces)
    kappas = [np.max(c.kappa()) for c in base_curves]

    console = Console(width=250)
    table1 = Table(show_header=False)
    table1.add_row(*['J', 'dJ'])
    table1.add_row(*[f'{dat_dict["J"]:.2e}', f'{np.max(np.abs(dat_dict["dJ"])):.2e}'])
    console.print(table1)

    console = Console(width=250)
    table1 = Table(expand=True, show_header=False)
    table1.add_row(*[f"{v}" for v in penalties.keys()])
    table1.add_row(*[f"{v.J():.4e}" for v in penalties.values()])
    console.print(table1)
    
    table2 = Table(expand=True, show_header=False) 
    for k in states.keys():
        table2.add_row(k, ' '.join([f'{J.J():.4e}' for J in states[k]]))
    table2.add_row('max force', f'{max_force:.3e}')
    table2.add_row('forces', ' '.join([f'{f:.3e}' for f in forces]))
    table2.add_row('currents', ' '.join([f'{curr:.3e}' for curr in currents_list]))
    table2.add_row('curvatures', ' '.join([f'{curv:.3e}' for curv in kappas]))
    table2.add_row('minimum X-point-to-vessel distance', ' '.join([f'{Jxv.shortest_distance():.3e}' for Jxv in Jxvs]))
    console.print(table2)

    curves_to_vtk(curves, OUT_DIR + "curves_tmp")
    curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_tmp")
    curves_to_vtk([axis.curve for axis in axes], OUT_DIR + "ma_tmp")
    for idx, boozer_surface in enumerate(boozer_surfaces):
        boozer_surface.surface.to_vtk(OUT_DIR + f"surf_tmp_{idx}")
    save([boozer_surfaces, iota_Gs, axes, xpoints], OUT_DIR + f'design{design}_after_forces_tmp.json')
    #print(outstr)

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
force = []
# print out coil forces
for bbsurf in boozer_surfaces:
    for coil in bbsurf.biotsavart.coils:
        force += [np.linalg.norm(coil_force(coil, bbsurf.biotsavart.coils, regularization_circ(coil_minor_radius)), axis=1)]
max_force = np.max(force)
print(f"max force on coils {max_force:.2f}")

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
MAXITER=1e3

lb = JF.lower_bounds
ub = JF.upper_bounds
bounds = np.vstack((lb, ub)).T

dofs = JF.x
callback(dofs)

for j in range(5):
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': MAXITER, 'maxcor':100}, tol=1e-15, callback=callback)
    dofs = res.x.copy()
    print(res.message)
    

#JF = (J_nonQSRatio 
#    + IOTAS_WEIGHT * J_iotas 
#    + MR_WEIGHT * J_major_radius
#    + LENGTH_WEIGHT * length_penalty
#    + CC_WEIGHT * Jccdist
#    + CS_WEIGHT * Jcsdist
#    + CURVATURE_WEIGHT * curvature_penalty
#    + MSC_WEIGHT * msc_penalty
#    + AL_WEIGHT * Jal
#    + BR_WEIGHT * Jbrs
#    + FORCE_WEIGHT * Jforce
#    + AFS_WEIGHT * Jafs
#    + CV_WEIGHT * Jcvd
#    + XV_WEIGHT * fv_penalty
#    )
    
    iota_err = max([np.abs(IOTAS.J() - IOTAS_TARGET)/np.abs(IOTAS_TARGET) for IOTAS, IOTAS_TARGET in zip(IOTAS_LIST, IOTAS_TARGET)])
    mr_err = np.abs(mr.J()-MR_TARGET)/MR_TARGET
    clen_err = max([max(Jl.J() - LENGTH_THRESHOLD, 0)/np.abs(LENGTH_THRESHOLD) for Jl in Jls])

    cc_err = max(CC_THRESHOLD-Jccdist.shortest_distance(), 0)/np.abs(CC_THRESHOLD)
    cs_err = max(CS_THRESHOLD-Jcsdist.shortest_distance(), 0)/np.abs(CS_THRESHOLD)
    xv_err = max([max(XV_THRESHOLD-Jxvdist.shortest_distance(), 0)/np.abs(XV_THRESHOLD) for Jxvdist in Jxvs])

    msc = [J.J() for J in Jmscs]
    msc_err = max(np.max(msc) - MSC_THRESHOLD, 0)/np.abs(MSC_THRESHOLD)

    curv_err = max(max([np.max(c.kappa()) for c in base_curves]) - CURVATURE_THRESHOLD, 0)/np.abs(CURVATURE_THRESHOLD)
    alen_err = np.max([ArclengthVariation(c).J() for c in base_curves])

    force = []
    for bbsurf in boozer_surfaces:
        for coil in bbsurf.biotsavart.coils:
            force += [np.linalg.norm(coil_force(coil, bbsurf.biotsavart.coils, regularization_circ(coil_minor_radius)), axis=1).max()]
    force_err = np.max([max(f-FORCE_THRESHOLD, 0)/FORCE_THRESHOLD for f in force])

    modB_err = max([np.abs(modB.J()-MODB_TARGET)/MODB_TARGET for modB in modBs])
    
    # check which constraints are violated and increase weight if violated by more than 0.1%
    if iota_err > 0.001:
        IOTAS_WEIGHT*=10
        print("IOTA ERROR", iota_err)
    if mr_err > 0.001:
        MR_WEIGHT*=10
        print("MR ERROR", mr_err)
    if clen_err > 0.001:
        LENGTH_WEIGHT*=10
        print("COIL LENGTH ERROR", clen_err)
    if cc_err > 0.001:
        CC_WEIGHT*=10
        print("COIL TO COIL ERROR", cc_err)
    if cs_err > 0.001:
        CS_WEIGHT*=10
        print("COIL TO SURFACE ERROR", cs_err)
    if xv_err > 0.001:
        XV_WEIGHT*=10
        print("XPOINT TO VESSEL ERROR", xv_err)
    if force_err > 0.001:
        FORCE_WEIGHT*=10
        print("FORCE ERROR", force_err)
    if modB_err > 0.001:
        MODB_WEIGHT*=10
        print("MODB ERROR", modB_err)
    if msc_err > 0.001:
        MSC_WEIGHT*=10
        print("MEAN SQUARED ERROR", msc_err)
    if curv_err > 0.001:
        CURVATURE_WEIGHT*=10
        print("CURVATURE ERROR", curv_err)
    if alen_err > 0.001:
        AL_WEIGHT*=10
        print("ARCLENGTH ERROR", alen_err)

curves_to_vtk(curves, OUT_DIR + "curves_opt")
curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_opt")
curves_to_vtk([axis.curve for axis in axes], OUT_DIR + "ma_opt")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}")
save([boozer_surfaces, iota_Gs, axes, xpoints], OUT_DIR + f'design{design}_after_forces_opt.json')

