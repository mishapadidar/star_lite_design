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
                         CurveRZFourier, CurveXYZFourierSymmetries, CurveXYZFourier)
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.field.selffield import regularization_circ
import pandas as pd
from rich.console import Console
from rich.table import Column, Table

from star_lite_design.utils.magneticwell import MagneticWell
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.boozer_surface_utils import BoozerResidual, CurveBoozerSurfaceDistance
from star_lite_design.utils.curve_vessel_distance import CurveVesselDistance
from star_lite_design.utils.fieldline_vessel_distance import FieldLineVesselDistance
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine
import yaml

"""
The script optimizes the Star_lite device to have 3 configurations with different iota values, and low coil forces.
This script was run as a second stage of optimization, after star_lite was optimized for quasi-symmetry etc.
"""


print("Running Optimization with coil well")
print("================================")

design = 'A' # A or B
# load the boozer surfaces (1 per Current configuration, so 3 total.)
data = load(f"../designs/design{design}_after_scaled.json")
boozer_surfaces = data[0] # BoozerSurfaces
iota_Gs = data[1] # (iota, G) pairs
axes_RZ = data[2] # magnetic axis CurveRZFouriers
xpoints_RZ = data[3] # X-point CurveRZFouriers
config = yaml.safe_load(open(f"./config_{design}.yaml",'r'))


# get the base curves
biotsavart = boozer_surfaces[0].biotsavart
coils = biotsavart.coils
curves = [c.curve for c in coils]
if design == "A":
    base_curve_idx = [0, 4]
    base_curves = [curves[i] for i in base_curve_idx]
elif design == "B":
    base_curve_idx = [0, 2]
    base_curves = [curves[i] for i in base_curve_idx]


# fix some currents
for bbsurf in boozer_surfaces:
    bbsurf.biotsavart.fix_all()
    for coil in bbsurf.biotsavart.coils:
        coil.current.unfix_all()

trim_coils = []

ncoils = 1
R0 = 1.2
R1 = 0.47746482927
order = 16

base_trim_curves = create_equally_spaced_curves(ncoils, 2, stellsym=True, R0=R0, R1=R1, order=order)

base_trim_curves[0].x *= 0.
base_trim_curves[0].set('xc(0)', R0)
base_trim_curves[0].set('yc(1)', R1)
base_trim_curves[0].set('zs(1)', R1)
g = base_trim_curves[0].gamma().copy()
angle = np.pi / 4
R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
g = g@R.T
base_trim_curves[0].least_squares_fit(g)

#ss_trim = CurveXYZFourier(np.linspace(0, 1, 160, endpoint=False), 16)
#ss_trim.x *= 0.
#for name in ss_trim.dof_names:
#    if 'xs' in name or 'yc' in name or 'zc' in name:
#        idx = name.index(':')
#        ss_trim.set(name[idx+1:], 0)
#        ss_trim.fix(name[idx+1:])
#ss_trim.set('xc(0)', R0+0.3)
#ss_trim.set('xc(1)', R1)
#ss_trim.set('zs(1)', R1)
#angle = np.pi/2
#c1 = RotatedCurve(ss_trim,  angle, False)


# repeat the trim geometries but use new currents here
trim_coils = []
for config_idx in range(3):
    coils = coils_via_symmetries(base_trim_curves, [ScaledCurrent(Current(0.), (1/4/np.pi)*1e7)], 2, True)
    #coils += coils_via_symmetries([c1], [ScaledCurrent(Current(0.), (1/4/np.pi)*1e7)], 2, False)
    
    trim_coils.append(coils)

for ii, bsurf in enumerate(boozer_surfaces):
    biotsavart_new = BiotSavart(bsurf.biotsavart.coils + trim_coils[ii])
    # rebuild the boozer surface and populate the res attribute
    bsurf = BoozerSurface(biotsavart_new, bsurf.surface, bsurf.label, bsurf.targetlabel, options={'newton_tol':1e-13, 'newton_maxiter':20})
    #bsurf = BoozerSurface(bsurf.biotsavart, bsurf.surface, bsurf.label, bsurf.targetlabel, options={'newton_tol':1e-13, 'newton_maxiter':20})
    bsurf.run_code(iota_Gs[ii][0], iota_Gs[ii][1])
    boozer_surfaces[ii] = bsurf

#base_trim_curves += [c1]
base_curves += base_trim_curves
curves = [c.curve for c in boozer_surfaces[0].biotsavart.coils]

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



# load all the target, and threshold quantities along with their associated penalty weights
CURRENT_THRESHOLD = config['CURRENT_THRESHOLD']
MODB_TARGET = config['MODB_TARGET']
MR_TARGET = config['MAJOR_RADIUS_TARGET']
COIL_TO_COIL_THRESHOLD = config['COIL_TO_COIL_THRESHOLD']
CV_THRESHOLD = config['COIL_TO_VESSEL_THRESHOLD']
CV_THRESHOLD = 5.13e-02
print("CV THRESHOLD CHANGED")

XPOINT_TO_VESSEL_THRESHOLD = config['XPOINT_TO_VESSEL_THRESHOLD']
CURVATURE_THRESHOLD = config['CURVATURE_THRESHOLD']
MEAN_SQUARED_CURVATURE_THRESHOLD = config['MEAN_SQUARED_CURVATURE_THRESHOLD']
LENGTH_THRESHOLD = config['LENGTH_THRESHOLD']
IOTAS_TARGET  = config['IOTAS_TARGET']
COIL_MINOR_RADIUS = config['COIL_MINOR_RADIUS']

COIL_TO_COIL_WEIGHT = Weight(config['COIL_TO_COIL_WEIGHT'])
CURVATURE_WEIGHT = Weight(config['CURVATURE_WEIGHT'])
MEAN_SQUARED_CURVATURE_WEIGHT = Weight(config['MEAN_SQUARED_CURVATURE_WEIGHT'])
LENGTH_WEIGHT = Weight(config['LENGTH_WEIGHT'])
IOTAS_WEIGHT=Weight(config['IOTAS_WEIGHT'])
MAJOR_RADIUS_WEIGHT=Weight(config['MAJOR_RADIUS_WEIGHT'])
BOOZER_RESIDUAL_WEIGHT=Weight(config['BOOZER_RESIDUAL_WEIGHT'])
COIL_TO_VESSEL_WEIGHT = Weight(config['COIL_TO_VESSEL_WEIGHT'])
XPOINT_TO_VESSEL_WEIGHT = Weight(config['XPOINT_TO_VESSEL_WEIGHT'])
MODB_WEIGHT = Weight(config['MODB_WEIGHT'])
ARCLENGTH_WEIGHT = Weight(config['ARCLENGTH_WEIGHT'])

FORCE_WEIGHT = Weight(0.)
FORCE_THRESHOLD = 5e3 # 5216.75

WELL_WEIGHT = Weight(config['WELL_WEIGHT'])
WELL_THRESHOLD = float(config['WELL_THRESHOLD'])

## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
magnetic_wells = [MagneticWell(boozer_surface) for boozer_surface in boozer_surfaces]
#magnetic_wells = [MagneticWell(axes[ii], boozer_surfaces[ii]) for ii in range(len(boozer_surfaces))]
J_wells = sum([QuadraticPenalty(Jw, WELL_THRESHOLD, "max") for Jw in magnetic_wells])

mr = MajorRadius(boozer_surfaces[0])
brs = [BoozerResidual(boozer_surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]
J_major_radius = QuadraticPenalty(mr, MR_TARGET, 'identity')  # target major radius is that computed on the initial surface

IOTAS_LIST = [Iotas(boozer_surface) for boozer_surface in boozer_surfaces]
J_iotas = sum([QuadraticPenalty(IOTAS, IOTAS_TARGET, 'identity') for IOTAS, IOTAS_TARGET in zip(IOTAS_LIST, IOTAS_TARGET)]) # target rotational transform is that computed on the initial surface
nonQS_list = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]
print([J.J()**0.5 for J in nonQS_list])
J_nonQSRatio = (1./len(boozer_surfaces)) * sum(nonQS_list)

Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, COIL_TO_COIL_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jal = sum(ArclengthVariation(curve) for curve in base_curves)

# coil-to-vessel distance
df = pd.read_csv("../designs/sheetmetal_chamber.csv")
X_vessel = df.values
Jcvd = CurveVesselDistance(base_curves, X_vessel, CV_THRESHOLD)

Jxvs = [FieldLineVesselDistance(xpoint, X_vessel, XPOINT_TO_VESSEL_THRESHOLD) for xpoint in xpoints]
xv_penalty = sum(Jxvs)

length_penalty = sum([QuadraticPenalty(Jl, LENGTH_THRESHOLD, "max") for Jl in Jls])
curvature_penalty = sum(Jcs)
msc_penalty = sum(QuadraticPenalty(J, MEAN_SQUARED_CURVATURE_THRESHOLD, "max") for J in Jmscs)
Jbrs = sum(brs)

# penalty on deviation from target mean field strength
MODB_WEIGHT = Weight(1e4)
modBs = [ModBOnFieldLine(axis, BiotSavart(boozer_surface.biotsavart.coils)) for axis, boozer_surface in zip(axes, boozer_surfaces)]
JmodB = sum([QuadraticPenalty(modB, MODB_TARGET, 'identity') for modB, axis, boozer_surface in zip(modBs, axes, boozer_surfaces)])

if design == 'A':
    #base_curve_idx += [6, 10] 
    base_curve_idx += [6] 
else:
    print('design B not finished..')
    quit()


for bbsurf in boozer_surfaces:
    for jj in base_curve_idx:
        bbsurf.biotsavart.coils[jj].current.upper_bounds = [CURRENT_THRESHOLD/bbsurf.biotsavart.coils[jj].current.scale]
        bbsurf.biotsavart.coils[jj].current.lower_bounds = [-CURRENT_THRESHOLD/bbsurf.biotsavart.coils[jj].current.scale]


# sum the objectives together
JF = (J_nonQSRatio 
    + IOTAS_WEIGHT * J_iotas 
    + MAJOR_RADIUS_WEIGHT * J_major_radius
    + LENGTH_WEIGHT * length_penalty
    + COIL_TO_COIL_WEIGHT * Jccdist
    + CURVATURE_WEIGHT * curvature_penalty
    + MEAN_SQUARED_CURVATURE_WEIGHT * msc_penalty
    + ARCLENGTH_WEIGHT * Jal
    + BOOZER_RESIDUAL_WEIGHT * Jbrs
    + MODB_WEIGHT * JmodB
    + COIL_TO_VESSEL_WEIGHT * Jcvd
    + XPOINT_TO_VESSEL_WEIGHT * xv_penalty
    + WELL_WEIGHT * J_wells
    )
    #+ FORCE_WEIGHT * Jforce

penalties = {'nonQS': J_nonQSRatio,
        'iotas':IOTAS_WEIGHT * J_iotas,
        'length':LENGTH_WEIGHT * length_penalty,
        'coil-to-coil': COIL_TO_COIL_WEIGHT * Jccdist,
        'curvature':CURVATURE_WEIGHT * curvature_penalty,
        'mean-squared curvature': MEAN_SQUARED_CURVATURE_WEIGHT * msc_penalty,
        'arclength':ARCLENGTH_WEIGHT * Jal,
        'Boozer residual': BOOZER_RESIDUAL_WEIGHT * Jbrs,
        'modB': MODB_WEIGHT * JmodB,
        'coil-to-vessel':COIL_TO_VESSEL_WEIGHT * Jcvd,
        'x-point-to-vessel':XPOINT_TO_VESSEL_WEIGHT * xv_penalty,
        'magnetic well': WELL_WEIGHT*J_wells
        }
        #'force': FORCE_WEIGHT * Jforce,

states = {
        'well': magnetic_wells,
        'iotas': IOTAS_LIST,
        'modB': modBs,
        'lengths':Jls,
        'major radius': [MajorRadius(boozer_surface) for boozer_surface in boozer_surfaces],
        'Boozer residuals': brs,
        'mean-squared curvature': Jmscs
        }

print("n_dofs", len(JF.x))


# Directory for output
OUT_DIR = f"./output/design{design}/well/"
os.makedirs(OUT_DIR, exist_ok=True)

curves_to_vtk(curves, OUT_DIR + "curves_init")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_init_{idx}")

# save these as a backup in case the boozer surface Newton solve fails
res_list = [{'sdofs': boozer_surface.surface.x.copy() , 'iota': boozer_surface.res['iota'], 'G': boozer_surface.res['G']} for boozer_surface in boozer_surfaces]
axes_res_list = [{'adofs': axis.curve.x.copy() , 'length': axis.res['length']} for axis in axes]
xpoints_res_list = [{'xdofs': xpoint.curve.x.copy() , 'length': xpoint.res['length']} for xpoint in xpoints]
dat_dict = {'iter':0, 'J': JF.J(), 'dJ': JF.dJ().copy()}

def callback(dofs):
    for res, boozer_surface in zip(res_list, boozer_surfaces):
        res['sdofs'] = boozer_surface.surface.x.copy()
        res['iota'] =  boozer_surface.res['iota']
        res['G'] = boozer_surface.res['G']
    for res, axis in zip(axes_res_list, axes):
        res['adofs'] = axis.curve.x.copy()
        res['length'] =  axis.res['length']
    for res, axis in zip(xpoints_res_list, xpoints):
        res['xdofs'] = axis.curve.x.copy()
        res['length'] =  axis.res['length']
    
    dat_dict['J'] = JF.J()
    dat_dict['dJ'] = JF.dJ().copy()
    
    currents_list = [np.abs(boozer_surface.biotsavart.coils[idx].current.get_value()) for boozer_surface in boozer_surfaces for idx in base_curve_idx]
    kappas = [np.max(c.kappa()) for c in base_curves]

    console = Console(width=250)
    table1 = Table(show_header=False)
    table1.add_row(*['iter', 'J', 'dJ'])
    table1.add_row(*[f'{dat_dict["iter"]}', f'{dat_dict["J"]:.2e}', f'{np.max(np.abs(dat_dict["dJ"])):.2e}'])
    console.print(table1)

    console = Console(width=250)
    table1 = Table(expand=True, show_header=False)
    table1.add_row(*[f"{v}" for v in penalties.keys()])
    table1.add_row(*[f"{v.J():.4e}" for v in penalties.values()])
    console.print(table1)
    
    table2 = Table(expand=True, show_header=False) 
    for k in states.keys():
        table2.add_row(k, ' '.join([f'{J.J():.4e}' for J in states[k]]))
    table2.add_row('currents', ' '.join([f'{curr:.3e}' for curr in currents_list]))
    table2.add_row('curvatures', ' '.join([f'{curv:.3e}' for curv in kappas]))
    table2.add_row('minimum X-point-to-vessel distance', ' '.join([f'{Jxv.shortest_distance():.3e}' for Jxv in Jxvs]))
    table2.add_row('minimum coil-to-vessel distance', f'{Jcvd.shortest_distance():.3e}')
    table2.add_row('minimum coil-to-coil distance', f'{Jccdist.shortest_distance():.3e}')
    console.print(table2)

    curves_to_vtk(curves, OUT_DIR + "curves_tmp")
    curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_tmp")
    curves_to_vtk([axis.curve for axis in axes], OUT_DIR + "ma_tmp")
    for idx, boozer_surface in enumerate(boozer_surfaces):
        boozer_surface.surface.to_vtk(OUT_DIR + f"surf_tmp_{idx}")
    save([boozer_surfaces, iota_Gs, axes, xpoints], OUT_DIR + f'design{design}_after_well_tmp.json')
    dat_dict["iter"] += 1

def fun(dofs):
    solver_fail = False

    JF.x = dofs
    try:
        J = JF.J()
        grad = JF.dJ()
    except: # the objective function evaluation failed unfortunately
        solver_fail = True
    
    fieldline_success = [fieldline.res['success'] for fieldline in axes+xpoints]
    if (not np.all([boozer_surface.res['success'] \
            and (not boozer_surface.surface.is_self_intersecting()) for boozer_surface in boozer_surfaces])) \
            or not np.all(fieldline_success)\
            or solver_fail:
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
        for res, axis in zip(axes_res_list,  axes): 
            axis.curve.x = res['adofs']
            axis.res['length'] = res['length']
        for res, xp in zip(xpoints_res_list,  xpoints): 
            xp.curve.x = res['xdofs']
            xp.res['length'] = res['length']

    return J, grad


#print("""
#################################################################################
#### Perform a Taylor test ######################################################
#################################################################################
#""")
#f = fun
#dofs = JF.x.copy()
##callback(dofs)
#np.random.seed(1)
#h = np.random.rand(dofs.size)
#h*=0
#h[0] = 1.
#h[1] = 1.
#J0, dJ0 = f(dofs)
#dJh = sum(dJ0 * h)
#for eps in [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
#   J1, _ = f(dofs + 2*eps*h)
#   J2, _ = f(dofs + eps*h)
#   J3, _ = f(dofs - eps*h)
#   J4, _ = f(dofs - 2*eps*h)
#   dd = (J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps
#   print("err", (dd - dJh)/np.linalg.norm(dJh), dd, dJh)
#quit()

dofs = JF.x
callback(dofs)


print("""
################################################################################
### Initial performance #######################################################
################################################################################
""")
#force = []
## print out coil forces
#for bbsurf in boozer_surfaces:
#    for coil in bbsurf.biotsavart.coils:
#        force += [np.linalg.norm(coil_force(coil, bbsurf.biotsavart.coils, regularization_circ(COIL_MINOR_RADIUS)), axis=1)]
#max_force = np.max(force)
#print(f"max force on coils {max_force:.2f}")

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

for j in range(10):
    dat_dict["iter"] = 0
    res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)
    #res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor':500, "ftol": 1e-22, "gtol": 1e-12}, tol=1e-15, callback=callback)
    #res = minimize(fun, dofs, jac=True, method='L-BFGS-B', bounds=bounds, options={'maxiter': MAXITER, 'maxcor':200}, tol=1e-15, callback=callback)
    dofs = res.x.copy()
    callback(dofs)
    print(res.message)
    
#JF = (J_nonQSRatio 
#    + IOTAS_WEIGHT * J_iotas 
#    + MAJOR_RADIUS_WEIGHT * J_major_radius
#    + LENGTH_WEIGHT * length_penalty
#    + COIL_TO_COIL_WEIGHT * Jccdist
#    + CURVATURE_WEIGHT * curvature_penalty
#    + MEAN_SQUARED_CURVATURE_WEIGHT * msc_penalty
#    + ARCLENGTH_WEIGHT * Jal
#    + BOOZER_RESIDUAL_WEIGHT * Jbrs
#    + FORCE_WEIGHT * Jforce
#    + AFS_WEIGHT * Jafs
#    + COIL_TO_VESSEL_WEIGHT * Jcvd
#    + XPOINT_TO_VESSEL_WEIGHT * fv_penalty
#    )
    well_err = max([max(Jl.J() - WELL_THRESHOLD, 0)/np.abs(WELL_THRESHOLD) for Jl in magnetic_wells])
    iota_err = max([np.abs(IOTAS.J() - IOTAS_TARGET)/np.abs(IOTAS_TARGET) for IOTAS, IOTAS_TARGET in zip(IOTAS_LIST, IOTAS_TARGET)])
    mr_err = np.abs(mr.J()-MR_TARGET)/MR_TARGET
    clen_err = max([max(Jl.J() - LENGTH_THRESHOLD, 0)/np.abs(LENGTH_THRESHOLD) for Jl in Jls])

    cc_err = max(COIL_TO_COIL_THRESHOLD-Jccdist.shortest_distance(), 0)/np.abs(COIL_TO_COIL_THRESHOLD)
    xv_err = max([max(XPOINT_TO_VESSEL_THRESHOLD-Jxvdist.shortest_distance(), 0)/np.abs(XPOINT_TO_VESSEL_THRESHOLD) for Jxvdist in Jxvs])

    msc = [J.J() for J in Jmscs]
    msc_err = max(np.max(msc) - MEAN_SQUARED_CURVATURE_THRESHOLD, 0)/np.abs(MEAN_SQUARED_CURVATURE_THRESHOLD)

    curv_err = max(max([np.max(c.kappa()) for c in base_curves]) - CURVATURE_THRESHOLD, 0)/np.abs(CURVATURE_THRESHOLD)
    alen_err = np.max([ArclengthVariation(c).J() for c in base_curves])

    #force = []
    #for bbsurf in boozer_surfaces:
    #    for coil in bbsurf.biotsavart.coils:
    #        force += [np.linalg.norm(coil_force(coil, bbsurf.biotsavart.coils, regularization_circ(COIL_MINOR_RADIUS)), axis=1).max()]
    #force_err = np.max([max(f-FORCE_THRESHOLD, 0)/FORCE_THRESHOLD for f in force])

    modB_err = max([np.abs(modB.J()-MODB_TARGET)/MODB_TARGET for modB in modBs])
    
    # check which constraints are violated and increase weight if violated by more than 0.1%
    if iota_err > 0.001:
        IOTAS_WEIGHT*=10
        print("IOTA ERROR", iota_err)
    if mr_err > 0.001:
        MAJOR_RADIUS_WEIGHT*=10
        print("MR ERROR", mr_err)
    if clen_err > 0.001:
        LENGTH_WEIGHT*=10
        print("COIL LENGTH ERROR", clen_err)
    if cc_err > 0.001:
        COIL_TO_COIL_WEIGHT*=10
        print("COIL TO COIL ERROR", cc_err)
    if xv_err > 0.001:
        XPOINT_TO_VESSEL_WEIGHT*=10
        print("XPOINT TO VESSEL ERROR", xv_err)
    #if force_err > 0.001:
    #    FORCE_WEIGHT*=10
    #    print("FORCE ERROR", force_err)
    if modB_err > 0.001:
        MODB_WEIGHT*=10
        print("MODB ERROR", modB_err)
    if msc_err > 0.001:
        MEAN_SQUARED_CURVATURE_WEIGHT*=10
        print("MEAN SQUARED ERROR", msc_err)
    if curv_err > 0.001:
        CURVATURE_WEIGHT*=10
        print("CURVATURE ERROR", curv_err)
    if alen_err > 0.001:
        ARCLENGTH_WEIGHT*=10
        print("ARCLENGTH ERROR", alen_err)
    if well_err > 0.001:
        WELL_WEIGHT*=10
        print("WELL ERROR", well_err)

    curves_to_vtk(curves, OUT_DIR + f"curves_opt_{j}")
    curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_opt_{j}")
    curves_to_vtk([axis.curve for axis in axes], OUT_DIR + f"ma_opt_{j}")
    for idx, boozer_surface in enumerate(boozer_surfaces):
        boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_{j}")
    save([boozer_surfaces, iota_Gs, axes, xpoints], OUT_DIR + f'design{design}_after_well_opt_{j}.json')
    
    # save the weights in a yaml file
    config['CURRENT_THRESHOLD'] = CURRENT_THRESHOLD
    config['COIL_TO_COIL_WEIGHT'] = COIL_TO_COIL_WEIGHT.value
    config['CURVATURE_WEIGHT'] = CURVATURE_WEIGHT.value
    config['MEAN_SQUARED_CURVATURE_WEIGHT'] = MEAN_SQUARED_CURVATURE_WEIGHT.value
    config['LENGTH_WEIGHT'] = LENGTH_WEIGHT.value
    config['IOTAS_WEIGHT'] = IOTAS_WEIGHT.value
    config['MAJOR_RADIUS_WEIGHT'] = MAJOR_RADIUS_WEIGHT.value
    config['BOOZER_RESIDUAL_WEIGHT'] = BOOZER_RESIDUAL_WEIGHT.value
    config['COIL_TO_VESSEL_WEIGHT'] = COIL_TO_VESSEL_WEIGHT.value
    config['XPOINT_TO_VESSEL_WEIGHT'] = XPOINT_TO_VESSEL_WEIGHT.value
    config['MODB_WEIGHT'] = MODB_WEIGHT.value
    config['ARCLENGTH_WEIGHT'] = ARCLENGTH_WEIGHT.value
    # Save to YAML
    with open(OUT_DIR + f'design{design}_after_well_opt_{j}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
