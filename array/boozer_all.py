#!/usr/bin/env python3
import argparse
import os
import re
import zlib

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from scipy.optimize import minimize

from simsopt._core import load, save
from simsopt.field import BiotSavart
from simsopt.geo import (
    ArclengthVariation,
    BoozerSurface,
    CurveCurveDistance,
    CurveLength,
    LpCurveCurvature,
    MajorRadius,
    MeanSquaredCurvature,
    NonQuasiSymmetricRatio,
    Iotas,
    RotatedCurve,
    curves_to_vtk,
)
from simsopt.objectives import QuadraticPenalty, Weight

from star_lite_design.utils.tangent_map import TangentMap, Monodromy
from star_lite_design.utils.boozer_surface_utils import BoozerResidual
from star_lite_design.utils.current_bound import CurrentBound
from star_lite_design.utils.displacement import FieldLineMeanZ
from star_lite_design.utils.magneticwell import MagneticWell
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine
from star_lite_design.utils.pillpipevessel import RennaissanceSDF, PillPipeSDF, TorusSDF, VesselDistance

import sn_setup


parser = argparse.ArgumentParser()
parser.add_argument("--margin", type=float)
parser.add_argument("--well", type=str, default='OFF')
parser.add_argument("--Z", type=int)
parser.add_argument("--distance", type=int)
parser.add_argument("--on-vessel", type=int)
parser.add_argument("--config", type=int)
parser.add_argument("--vessel-id", type=int)
parser.add_argument("--mono", type=int)
# This script always produces the unpolished (num_aux=0) device; --num-aux is
# accepted for a uniform CLI with the polish step and to keep the folder name
# consistent. --attempt indexes the 4 perturbed restarts per parameter combo;
# it (via the folder name) sets the perturbation seed.
parser.add_argument("--num-aux", type=int, default=0)
parser.add_argument("--attempt", type=int, default=0)
# Null type: DN = double-null (current stellsym behavior); SN = single-null
# (drop stellsym, rebuild as nfp=2/stellsym=False, push the bottom X-point to
# the lower wall).
parser.add_argument("--null", type=str, default='DN', choices=['DN', 'SN'])

args = parser.parse_args()

if args.well == "OFF":
    well_target = 0.0
    well_active = False
else:
    well_target = float(args.well)
    well_active = True

on_vessel = args.on_vessel
margin_target = args.margin
Z_weight = args.Z
distance_weight = args.distance
config_id = args.config
vessel_id = args.vessel_id
mono = args.mono
num_aux = args.num_aux
attempt = args.attempt
null_type = args.null



margin_str = f"{margin_target:.2f}".replace(".", "p")
if args.well == "OFF":
    well_str = "OFF"
else:
    well_str = str(float(args.well))
# The folder name carries every task parameter, num_aux (0 here), and the
# attempt index. DEVICE_ID is a deterministic hash of that name so the random
# coil perturbation below is reproducible and device_browser can recompute the
# same ID from the folder name (it uses the identical zlib.crc32 convention).
TASK_NAME = (f"margin={margin_str}_well={well_str}_Z={Z_weight}_onvessel={on_vessel}"
             f"_distance={distance_weight}_configID={config_id}_vesselID={vessel_id}"
             f"_mono={mono}_null={null_type}_num_aux={num_aux}_attempt={attempt}")
DEVICE_ID = zlib.crc32(TASK_NAME.encode())
OUT_DIR = f"./output/{TASK_NAME}/"
print(f"Task: {TASK_NAME}")
print(f"Device ID: {DEVICE_ID}")

"""
The script optimizes the Star_lite device to have 3 configurations with different iota values, and low coil forces.
This script was run as a second stage of optimization, after star_lite was optimized for quasi-symmetry etc.
"""


print("Running ALL IN ONE Optimization")
print("================================")


# load the boozer surfaces (1 per Current configuration, so 3 total.)
config = yaml.safe_load(open(f"../convert/designA_after_scaled.yaml",'r'))
data = load(f"../convert/designA_after_scaled.json")
boozer_surfaces = data[0][config_id:config_id+1] # BoozerSurfaces
iota_Gs = data[1][config_id:config_id+1] # (iota, G) pairs
axes = data[2][config_id:config_id+1] # magnetic axis CurveRZFouriers
xpoints = data[3][config_id:config_id+1] # X-point CurveRZFouriers

# Single-null setup (SN): drop stellarator symmetry. Rebuild the coils from the
# 3 independent (Y>0) bases as nfp=2/stellsym=False, convert the surface, axis
# and top X-point to non-stellsym on the rebuilt coils, and build the bottom
# X-point as its own solved field line (initially the stellsym mirror of the
# top). Everything is wired to the objectives below exactly as in DN; the freed
# dofs let the optimizer build the single-null asymmetry. bottom_xpoints stays
# None for DN.
bottom_xpoints = None
if null_type == 'SN':
    (boozer_surfaces, axes, xpoints, bottom_xpoints,
     _sn_coils, _sn_base_curves, _sn_base_currents) = sn_setup.setup_single_null(
        boozer_surfaces, iota_Gs, axes, xpoints)

#boozer_surfaces = [boozer_surfaces[1]]
#axes = [axes[1]]
#xpoints = [xpoints[1]]

for ii, (boozer_surface, axis, xpoint) in enumerate(zip(boozer_surfaces, axes, xpoints)):
    axis.options = {'newton_tol':1e-15, 'newton_maxiter':20, 'verbose':True}
    xpoint.options = {'newton_tol':1e-15, 'newton_maxiter':20, 'verbose':True}
    boozer_surface.options = {'newton_tol':1e-16 if null_type == 'DN' else 1e-14, 'newton_maxiter':20, 'verbose':True}
    axis.run_code(CurveLength(axis.curve).J())
    xpoint.run_code(CurveLength(xpoint.curve).J())
    boozer_surface.run_code(iota_Gs[ii][0], iota_Gs[ii][1])

# get the base curves
biotsavart = boozer_surfaces[0].biotsavart
coils = biotsavart.coils
curves = [c.curve for c in coils]
# DN: 2 independent base coils (stellsym expansion gives [0,4]). SN: the rebuilt
# coils_via_symmetries(3 bases, nfp=2, stellsym=False) set has the 3 independent
# base coils at indices [0,1,2] (rotated copies at 3,4,5).
base_curve_idx = [0, 1, 2] if null_type == 'SN' else [0, 4]
base_curves = [curves[i] for i in base_curve_idx]

if vessel_id == 0:
    bx = 0.5
    by = 0.55
    r = 0.5
    rr = 0.2
    sdf = PillPipeSDF(bx, by, r, rr)
elif vessel_id == 1:
    b1 = 0.35
    b2 = 0.45
    rr = 0.2
    sdf = RennaissanceSDF(b1, b2, rr)
elif vessel_id == 2:
    r = 0.2
    R = 0.5
    sdf = TorusSDF(r, R)
else:
    raise Exception('vessel not implemented')

config['COIL_ON_VESSEL_THRESHOLD'] = -0.001
COIL_ON_VESSEL_THRESHOLD = config['COIL_ON_VESSEL_THRESHOLD']

# leave enough space for trims
config['COIL_CLEARANCE_THRESHOLD'] = 0.12
COIL_CLEARANCE_THRESHOLD = config['COIL_CLEARANCE_THRESHOLD']

config['COIL_ON_VESSEL_WEIGHT'] = 1e-1 if on_vessel else 0.0
config['COIL_CLEARANCE_WEIGHT'] = 0.0 if on_vessel else config['COIL_TO_VESSEL_WEIGHT']

COIL_ON_VESSEL_WEIGHT = Weight(config['COIL_ON_VESSEL_WEIGHT'])
COIL_CLEARANCE_WEIGHT = Weight(config['COIL_CLEARANCE_WEIGHT'])


PLASMA_VESSEL_MARGIN_THRESHOLD = margin_target
config['PLASMA_VESSEL_MARGIN_THRESHOLD'] = PLASMA_VESSEL_MARGIN_THRESHOLD

# load all the target, and threshold quantities along with their associated penalty weights
CURRENT_THRESHOLD = config['CURRENT_THRESHOLD']
CURRENT_WEIGHT = Weight(config['CURRENT_WEIGHT'])

config['WELL_THRESHOLD'] = well_target
config['WELL_WEIGHT'] = 1e-9 if well_active else 0.0
WELL_THRESHOLD = float(config['WELL_THRESHOLD'])
WELL_WEIGHT = Weight(config['WELL_WEIGHT'])

config['FIELDLINE_MEANZ_WEIGHT'] = 10.0 * Z_weight
config['FIELDLINE_MEANZ_THRESHOLD'] = 0.001
FIELDLINE_MEANZ_WEIGHT = Weight(config['FIELDLINE_MEANZ_WEIGHT'])
FIELDLINE_MEANZ_THRESHOLD = config['FIELDLINE_MEANZ_THRESHOLD']

config['FIELDLINE_MEANDIST_WEIGHT'] = 10.0 * distance_weight
config['FIELDLINE_MEANDIST_THRESHOLD'] = 0.001
FIELDLINE_MEANDIST_WEIGHT = Weight(config['FIELDLINE_MEANDIST_WEIGHT'])
FIELDLINE_MEANDIST_THRESHOLD = config['FIELDLINE_MEANDIST_THRESHOLD']



MODB_TARGET = config['MODB_TARGET']
MR_TARGET = config['MAJOR_RADIUS_TARGET']
COIL_TO_COIL_THRESHOLD = config['COIL_TO_COIL_THRESHOLD']
CURVATURE_THRESHOLD = config['CURVATURE_THRESHOLD']
MEAN_SQUARED_CURVATURE_THRESHOLD = config['MEAN_SQUARED_CURVATURE_THRESHOLD']
LENGTH_THRESHOLD = config['LENGTH_THRESHOLD']
IOTAS_TARGET  = config['IOTAS_TARGET'][config_id:config_id+1]
COIL_MINOR_RADIUS = config['COIL_MINOR_RADIUS']

COIL_TO_COIL_WEIGHT = Weight(config['COIL_TO_COIL_WEIGHT'])
CURVATURE_WEIGHT = Weight(config['CURVATURE_WEIGHT'])
MEAN_SQUARED_CURVATURE_WEIGHT = Weight(config['MEAN_SQUARED_CURVATURE_WEIGHT'])
LENGTH_WEIGHT = Weight(config['LENGTH_WEIGHT'])
IOTAS_WEIGHT=Weight(config['IOTAS_WEIGHT'])
MAJOR_RADIUS_WEIGHT=Weight(config['MAJOR_RADIUS_WEIGHT']/1000.)

BOOZER_RESIDUAL_WEIGHT=Weight(config['BOOZER_RESIDUAL_WEIGHT'])


PLASMA_VESSEL_MARGIN_WEIGHT = Weight(config['COIL_TO_VESSEL_WEIGHT'])
MODB_WEIGHT = Weight(config['MODB_WEIGHT'])
ARCLENGTH_WEIGHT = Weight(config['ARCLENGTH_WEIGHT'])

## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
mr = MajorRadius(boozer_surfaces[0])
ls = [CurveLength(c) for c in base_curves]
brs = [BoozerResidual(boozer_surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]
J_major_radius = QuadraticPenalty(mr, MR_TARGET, 'identity')  # target major radius is that computed on the initial surface

IOTAS_LIST = [Iotas(boozer_surface) for boozer_surface in boozer_surfaces]
J_iotas = sum([QuadraticPenalty(IOTAS, IOT_TARGET, 'identity') for IOTAS, IOT_TARGET in zip(IOTAS_LIST, IOTAS_TARGET)]) # target rotational transform is that computed on the initial surface
nonQS_list = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]
print([J.J()**0.5 for J in nonQS_list])
J_nonQSRatio = (1./len(boozer_surfaces)) * sum(nonQS_list)

Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, COIL_TO_COIL_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jal = sum(ArclengthVariation(curve) for curve in base_curves)

length_penalty = sum([QuadraticPenalty(Jl, LENGTH_THRESHOLD, "max") for Jl in Jls])
curvature_penalty = sum(Jcs)
msc_penalty = sum(QuadraticPenalty(J, MEAN_SQUARED_CURVATURE_THRESHOLD, "max") for J in Jmscs)
Jbrs = sum(brs)

# penalty on deviation from target mean field strength
modBs = [ModBOnFieldLine(axis, BiotSavart(boozer_surface.biotsavart.coils)) for axis, boozer_surface in zip(axes, boozer_surfaces)]
JmodB = sum([QuadraticPenalty(modB, MODB_TARGET, 'identity') for modB, axis, boozer_surface in zip(modBs, axes, boozer_surfaces)])

J_curr = None
for boozer_surface in boozer_surfaces:
    for idx in base_curve_idx:
        curr = boozer_surface.biotsavart.coils[idx].current
        if J_curr is None:
            J_curr = CurrentBound(curr.current_to_scale, CURRENT_THRESHOLD/curr.scale)
        else:
            J_curr += CurrentBound(curr.current_to_scale, CURRENT_THRESHOLD/curr.scale)

plasma_boundary_entities = xpoints + boozer_surfaces
plasma_boundary_signs = np.array([-1.0 for _ in xpoints] + [-1.0 for _ in boozer_surfaces])
J_plasma_to_vessel_margin = VesselDistance(sdf, plasma_boundary_entities, plasma_boundary_signs, PLASMA_VESSEL_MARGIN_THRESHOLD)

coil_on_vessel_entities = base_curves + base_curves
coil_on_vessel_signs = np.array(
    [-1.0 for _ in base_curves] + [1.0 for _ in base_curves]
)
J_coil_on_vessel = VesselDistance(
    sdf,
    coil_on_vessel_entities,
    coil_on_vessel_signs,
    COIL_ON_VESSEL_THRESHOLD,
)

coil_clearance_entities = base_curves
coil_clearance_signs = np.array([1.0 for _ in base_curves])
J_coil_clearance = VesselDistance(
    sdf,
    coil_clearance_entities,
    coil_clearance_signs,
    COIL_CLEARANCE_THRESHOLD,
)

magnetic_wells = [MagneticWell(axis, boozer_surface, WELL_THRESHOLD) for axis, boozer_surface in zip(axes, boozer_surfaces)]
J_wells = sum(magnetic_wells)

meanzs = [FieldLineMeanZ(xpoint, FIELDLINE_MEANZ_THRESHOLD) for xpoint in xpoints]
J_meanz = sum(meanzs)

fieldline_mean_distance_signs = np.array([-1.0 for _ in xpoints])
fieldline_mean_distance_entities = xpoints

J_fieldline_mean_distance = VesselDistance(
    sdf,
    fieldline_mean_distance_entities,
    fieldline_mean_distance_signs,
    FIELDLINE_MEANDIST_THRESHOLD,
    metric='distance',
)

config['MONODROMY_THRESHOLD'] = 0.1
MONODROMY_THRESHOLD = config['MONODROMY_THRESHOLD']
MONODROMY_WEIGHT = Weight(1e-4) if mono > 0 else Weight(0.)
tmos = [TangentMap(xp, BiotSavart(boozer_surface.biotsavart.coils), MONODROMY_THRESHOLD, mtype='identity' if mono == 1 else 'jordan') for xp, boozer_surface in zip(xpoints, boozer_surfaces)]
MONODROMY_LIST = [Monodromy(tmo) for tmo in tmos]
J_monodromy = sum(MONODROMY_LIST) # target rotational transform is that computed on the initial surface

# Bottom X-point band (SN only): keep each bottom X-point field line between
# BOTTOM_XPOINT_MIN (1 cm) and BOTTOM_XPOINT_MAX (5 cm) inside the vessel. The
# top X-point's "stay inside" constraint is already in J_plasma_to_vessel_margin.
J_bottom_xpoint = None
if null_type == 'SN':
    config['BOTTOM_XPOINT_MIN'] = 0.01
    config['BOTTOM_XPOINT_MAX'] = 0.05
    config['BOTTOM_XPOINT_WEIGHT'] = 1e-1
    BOTTOM_XPOINT_MIN = config['BOTTOM_XPOINT_MIN']
    BOTTOM_XPOINT_MAX = config['BOTTOM_XPOINT_MAX']
    BOTTOM_XPOINT_WEIGHT = Weight(config['BOTTOM_XPOINT_WEIGHT'])
    J_bottom_xpoint = sn_setup.bottom_xpoint_vessel_penalty(
        sdf, bottom_xpoints, BOTTOM_XPOINT_MIN, BOTTOM_XPOINT_MAX)
else:
    BOTTOM_XPOINT_WEIGHT = Weight(0.0)

# sum the objectives together
JF = (J_nonQSRatio 
    + MONODROMY_WEIGHT * J_monodromy 
    + IOTAS_WEIGHT * J_iotas 
    + MAJOR_RADIUS_WEIGHT * J_major_radius
    + LENGTH_WEIGHT * length_penalty
    + COIL_TO_COIL_WEIGHT * Jccdist
    + CURVATURE_WEIGHT * curvature_penalty
    + MEAN_SQUARED_CURVATURE_WEIGHT * msc_penalty
    + ARCLENGTH_WEIGHT * Jal
    + BOOZER_RESIDUAL_WEIGHT * Jbrs
    + MODB_WEIGHT * JmodB
    + PLASMA_VESSEL_MARGIN_WEIGHT * J_plasma_to_vessel_margin
    + COIL_ON_VESSEL_WEIGHT * J_coil_on_vessel
    + COIL_CLEARANCE_WEIGHT * J_coil_clearance
    + CURRENT_WEIGHT * J_curr
    + WELL_WEIGHT * J_wells
    + FIELDLINE_MEANZ_WEIGHT * J_meanz
    + FIELDLINE_MEANDIST_WEIGHT * J_fieldline_mean_distance
    )

# Bottom X-point band term (SN only).
if J_bottom_xpoint is not None:
    JF = JF + BOTTOM_XPOINT_WEIGHT * J_bottom_xpoint


penalties = {'nonQS': J_nonQSRatio,
        'monodromy' : MONODROMY_WEIGHT * J_monodromy,
        'iotas':IOTAS_WEIGHT * J_iotas,
        'length':LENGTH_WEIGHT * length_penalty,
        'coil-to-coil': COIL_TO_COIL_WEIGHT * Jccdist,
        'curvature':CURVATURE_WEIGHT * curvature_penalty,
        'mean-squared curvature': MEAN_SQUARED_CURVATURE_WEIGHT * msc_penalty,
        'arclength':ARCLENGTH_WEIGHT * Jal,
        'Boozer residual': BOOZER_RESIDUAL_WEIGHT * Jbrs,
        'modB': MODB_WEIGHT * JmodB,
        'plasma-boundary-to-vessel': PLASMA_VESSEL_MARGIN_WEIGHT * J_plasma_to_vessel_margin,
        'coil-on-vessel': COIL_ON_VESSEL_WEIGHT * J_coil_on_vessel,
        'coil-clearance': COIL_CLEARANCE_WEIGHT * J_coil_clearance,
        'current':CURRENT_WEIGHT * J_curr,
        'well':WELL_WEIGHT * J_wells,
        'fieldline meanz': FIELDLINE_MEANZ_WEIGHT * J_meanz,
        'fieldline mean dist':FIELDLINE_MEANDIST_WEIGHT * J_fieldline_mean_distance,
        'major radius': MAJOR_RADIUS_WEIGHT * J_major_radius,
        }
if J_bottom_xpoint is not None:
    penalties['bottom xpoint'] = BOTTOM_XPOINT_WEIGHT * J_bottom_xpoint

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

# make sure coils are stellarator symmetric (DN only). For SN we deliberately
# keep the base coils general (the device is no longer stellsym), so we do NOT
# fix the stellsym-violating dofs.
if null_type == 'DN':
    for ii in [base_curve_idx[-1]]:
        c = boozer_surfaces[0].biotsavart.coils[ii].curve
        if isinstance(c, RotatedCurve):
            c = c.curve
        for df in c.local_dof_names:
            if ('xs' in df) or ('yc' in df) or ('zc' in df):
                c.fix(df)

# ---- per-attempt perturbation of the two base modular coils -------------
# Jitter the order-0, 1 and 2 Fourier coefficients of each base modular coil
# (base_curve_idx) by N(0, 1cm) so each of the 4 attempts starts the optimizer
# from a different point. This runs AFTER the stellsym dofs are fixed above and
# only touches FREE dofs (we set only names present in local_dof_names), so the
# enforced symmetry is preserved. Seeded by DEVICE_ID for reproducibility.
#
# After perturbing, the magnetic axis, Boozer surface and X-point are re-solved
# with the perturbed coils. If any solve fails or the surface self-intersects,
# the axis/surface/X-point are rolled back to their unperturbed dofs and a
# fresh perturbation is drawn, repeating until all three converge to a
# non-self-intersecting state. A generous cap turns a pathological case into a
# task failure rather than an indefinite hang.
PERTURB_STD = 0.005 if null_type == 'SN' else 0.01   # metres (SN: 0.5 cm, DN: 1 cm)
PERTURB_ORDERS = (0, 1, 2)   # Fourier harmonics to perturb
MAX_PERTURB_TRIES = 100
pert_rng = np.random.default_rng(DEVICE_ID)

# The two base coils named by base_curve_idx (unwrap RotatedCurve, as above).
perturb_curves = []
for ii in base_curve_idx:
    cc = boozer_surfaces[0].biotsavart.coils[ii].curve
    if isinstance(cc, RotatedCurve):
        cc = cc.curve
    perturb_curves.append(cc)

# Fourier dof names at orders 0/1/2 (order 0 has only the cosine term).
def _perturb_dof_names():
    names = []
    for order in PERTURB_ORDERS:
        for coord in ('x', 'y', 'z'):
            if order == 0:
                names.append(f'{coord}c(0)')
            else:
                names.append(f'{coord}s({order})')
                names.append(f'{coord}c({order})')
    return names


_PERTURB_NAMES = _perturb_dof_names()

# Capture the converged, unperturbed reference state to roll back to.
coil_ref = [(cc, cc.x.copy()) for cc in perturb_curves]
axes_ref = [(ax, ax.curve.x.copy()) for ax in axes]
xpoints_ref = [(xp, xp.curve.x.copy()) for xp in xpoints]
surf_ref = [(bs, bs.surface.x.copy(), bs.res['iota'], bs.res['G'])
            for bs in boozer_surfaces]


def _restore_unperturbed():
    for cc, xref in coil_ref:
        cc.x = xref
    for ax, xref in axes_ref:
        ax.curve.x = xref
        ax.need_to_run_code = True
    for xp, xref in xpoints_ref:
        xp.curve.x = xref
        xp.need_to_run_code = True
    for bs, sref, _iota, _G in surf_ref:
        bs.surface.x = sref
        bs.need_to_run_code = True


def _apply_perturbation():
    for cc in perturb_curves:
        free = set(cc.local_dof_names)   # only perturb unfixed dofs
        for nm in _PERTURB_NAMES:
            if nm in free:
                cc.set(nm, cc.get(nm) + pert_rng.normal(0.0, PERTURB_STD))


def _resolve_perturbed():
    """Re-solve axis, X-point and Boozer surface with the perturbed coils.
    Returns True iff all three converge and no surface self-intersects."""
    for ax, _ in axes_ref:
        ax.need_to_run_code = True
        ax.run_code(CurveLength(ax.curve).J())
    for xp, _ in xpoints_ref:
        xp.need_to_run_code = True
        xp.run_code(CurveLength(xp.curve).J())
    for bs, _s, iota, G in surf_ref:
        bs.need_to_run_code = True
        bs.run_code(iota, G)
    return (all(ax.res['success'] for ax in axes)
            and all(xp.res['success'] for xp in xpoints)
            and all(bs.res['success'] for bs in boozer_surfaces)
            and not any(bs.surface.is_self_intersecting() for bs in boozer_surfaces))


_perturb_solved = False
for _ktry in range(MAX_PERTURB_TRIES):
    _restore_unperturbed()
    _apply_perturbation()
    try:
        ok = _resolve_perturbed()
    except Exception as e:
        print(f"perturbation try {_ktry}: solve raised: {e}")
        ok = False
    if ok:
        print(f"perturbation converged after {_ktry + 1} tr{'y' if _ktry == 0 else 'ies'}")
        _perturb_solved = True
        break
    # Shrink the perturbation on failure so subsequent resamples are gentler and
    # more likely to converge (_apply_perturbation reads PERTURB_STD each call).
    PERTURB_STD = PERTURB_STD / 2.0
    print(f"perturbation try {_ktry}: failed (non-convergence or self-intersection); "
          f"halving std to {PERTURB_STD:.2e} and resampling")

if not _perturb_solved:
    raise RuntimeError(
        f"could not find a converged base-coil perturbation in {MAX_PERTURB_TRIES} tries "
        f"(device {DEVICE_ID}, task {TASK_NAME})")

print(JF.dof_names, JF.x.size)
#import ipdb;ipdb.set_trace()
#quit()



# Directory for output
os.makedirs(OUT_DIR, exist_ok=True)

curves_to_vtk(curves, OUT_DIR + "curves_init")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_init_{idx}")

# save these as a backup in case the boozer surface Newton solve fails
res_list = [{'sdofs': boozer_surface.surface.x.copy() , 'iota': boozer_surface.res['iota'], 'G': boozer_surface.res['G']} for boozer_surface in boozer_surfaces]
axes_res_list = [{'adofs': axis.curve.x.copy() , 'length': axis.res['length']} for axis in axes]
xpoints_res_list = [{'xdofs': xpoint.curve.x.copy() , 'length': xpoint.res['length']} for xpoint in xpoints]
dat_dict = {'iter':0, 'J': JF.J(), 'dJ': JF.dJ().copy(), 'x': JF.x.copy()}

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
    
    dist = np.linalg.norm(dofs - dat_dict['x'])

    dat_dict['x'] = JF.x.copy()
    dat_dict['J'] = JF.J()
    dat_dict['dJ'] = JF.dJ().copy()
    
    currents_list = [np.abs(boozer_surface.biotsavart.coils[idx].current.get_value()) for boozer_surface in boozer_surfaces for idx in base_curve_idx]
    kappas = [np.max(c.kappa()) for c in base_curves]
    
    console = Console(width=250)
    table1 = Table(show_header=False)
    table1.add_row(*['iter', 'J', 'dJ', 'dist'])
    table1.add_row(*[f'{dat_dict["iter"]}', f'{dat_dict["J"]:.2e}', f'{np.max(np.abs(dat_dict["dJ"])):.2e}', f'{dist:.2e}'])
    console.print(table1)

    console = Console(width=250)
    table1 = Table(expand=False, show_header=False)
    table1.add_row(*[f"{v}" for v in penalties.keys()])
    table1.add_row(*[f"{v.J():.4e}" for v in penalties.values()])
    console.print(table1)
    
    table2 = Table(expand=False, show_header=False) 
    for k in states.keys():
        table2.add_row(k, ' '.join([f'{J.J():.4e}' for J in states[k]]))
    table2.add_row('xpoint(0)', ' '.join([f'{np.array2string(xpoint.curve.gamma()[0])}' for xpoint in xpoints]))
    table2.add_row('monodromys', ' '.join([f'{np.array2string(d.matrix)}' for d in tmos]))
    table2.add_row('trace(monodromys)', ' '.join([f'{float(np.trace(d.matrix)):.4e}' for d in tmos]))
    table2.add_row('well', ' '.join([f'{w.well().max():.3e}' for w in magnetic_wells]))
    table2.add_row('currents', ' '.join([f'{curr:.3e}' for curr in currents_list]))
    table2.add_row('curvatures', ' '.join([f'{curv:.3e}' for curv in kappas]))
    #table2.add_row('minimum X-point-to-vessel distance', ' '.join([f'{Jxv.shortest_distance():.3e}' for Jxv in Jxvs]))
    _, min_xpoint_to_vessel, min_boozer_surface_to_vessel = J_plasma_to_vessel_margin.shortest_distance()
    table2.add_row('minimum X-point-to-vessel distance', f'{min_xpoint_to_vessel:.3e}')
    table2.add_row('minimum Boozer surface-to-vessel distance', f'{min_boozer_surface_to_vessel:.3e}')
    
    min_coil_on_vessel_distance, _, _ = J_coil_on_vessel.shortest_distance()
    min_coil_clearance_distance, _, _ = J_coil_clearance.shortest_distance()
    table2.add_row('minimum coil-on-vessel distance', f'{min_coil_on_vessel_distance:.3e}')
    table2.add_row('minimum coil-clearance distance', f'{min_coil_clearance_distance:.3e}')

    table2.add_row('vessel dimensions', ' '.join([f'{name}={sdf.local_full_x[ii]:.6e} ' for ii, name in enumerate(sdf.local_dof_names)]))
    table2.add_row('minimum coil-to-coil distance', f'{Jccdist.shortest_distance():.3e}')
    
    table2.add_row('fieldline mean-z', ' '.join([f'{Jfl.max_distance():.3e}' for Jfl in meanzs]))
    _, max_fieldline_mean_distance, _ = J_fieldline_mean_distance.longest_distance()
    table2.add_row('fieldline mean distance', f'{max_fieldline_mean_distance:.3e}')
 
    console.print(table2)
    
    if dat_dict["iter"] % 10 == 0:
        sdf.to_vtk(OUT_DIR + 'vessel_tmp')
        curves_to_vtk(curves, OUT_DIR + "curves_tmp")
        curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_tmp")
        curves_to_vtk([axis.curve for axis in axes], OUT_DIR + "ma_tmp")
        for idx, boozer_surface in enumerate(boozer_surfaces):
            boozer_surface.surface.to_vtk(OUT_DIR + f"surf_tmp_{idx}")
        save([boozer_surfaces, iota_Gs, axes, xpoints, sdf], OUT_DIR + f'design_tmp.json')
    
    dat_dict["iter"] += 1

def _restore_state():
    """Roll boozer surfaces, axis, xpoint curves, and JF dofs back to the
    last successful callback. Called after a failed fun() evaluation so the
    next call warm-starts from a good point."""
    for res, bs in zip(res_list, boozer_surfaces):
        bs.surface.x = res['sdofs']
        bs.res['iota'] = res['iota']
        bs.res['G'] = res['G']
    for res, ax in zip(axes_res_list, axes):
        ax.curve.x = res['adofs']
        ax.res['length'] = res['length']
    for res, xp in zip(xpoints_res_list, xpoints):
        xp.curve.x = res['xdofs']
        xp.res['length'] = res['length']
    JF.x = dat_dict['x']


def fun(dofs):
    JF.x = dofs

    eval_failed = False
    try:
        J = JF.J()
        grad = JF.dJ()
    except Exception:
        eval_failed = True

    surfaces_ok = all(
        bs.res['success'] and not bs.surface.is_self_intersecting()
        for bs in boozer_surfaces
    )
    fieldlines_ok = all(fl.res['success'] for fl in axes + xpoints)

    if eval_failed or not surfaces_ok or not fieldlines_ok:
        print('failed — rolling back to last good state')
        _restore_state()
        # Return the previous J and the negated previous gradient so BFGS's
        # line search shrinks the step back toward the last good point.
        # in case the penalty method is failing, this will always return
        # larger by 1e3
        return 1e3 + dat_dict['J'], -dat_dict['dJ']

    return J, grad

#print("""
#################################################################################
#### Perform a Taylor test ######################################################
#################################################################################
#""")
#f = fun
#dofs = JF.x.copy()
#np.random.seed(1)
#h = np.random.rand(dofs.size)
#J0, dJ0 = f(dofs)
#dJh = sum(dJ0 * h)
#fd_order = 6
#
#fd_stencils = {
#    2: (np.array([-1,  1]), np.array([-1/2,  1/2])),
#    4: (np.array([-2, -1,  1,  2]), np.array([ 1/12, -8/12,  8/12, -1/12])),
#    6: (np.array([-3, -2, -1,  1,  2,  3]), np.array([-1/60, 3/20,  -3/4, 3/4,  -3/20, 1/60])),
#    8: (np.array([-4, -3, -2, -1,  1,  2,  3,  4]), np.array([ 1/280, -4/105,  1/5, -4/5,  4/5, -1/5,  4/105, -1/280])),
#}
#
#shifts, w = fd_stencils[fd_order]
#eps_prev = -1
#err_prev = -1
#for eps in [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
#   fd = 0.0
#   for s, wi in zip(shifts, w):
#      Ji, _ = f(dofs + s * eps * h)
#      fd += wi * Ji
#   fd /= eps
#   err =  (fd - dJh)/np.linalg.norm(dJh)
#   print("err", err, fd, dJh, np.log(err_prev/err)/np.log(eps_prev/eps))
#   eps_prev = eps
#   err_prev = err
#quit()

print("""
################################################################################
### Initial performance #######################################################
################################################################################
""")

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

dofs = JF.x.copy()
callback(dofs)

for j in range(10):
    dat_dict["iter"] = 0

    try:
        res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)
        dofs = res.x.copy()
        msg = res.message
    except Exception as e:
        dofs = dat_dict['x'].copy()
        msg = f'caught exception: {e}, restarting from last successful callback.'
    
    print(msg)
    J0, dJ0 = fun(dofs)
    callback(dofs)
    
    currents_list = [np.abs(boozer_surface.biotsavart.coils[idx].current.get_value()) for boozer_surface in boozer_surfaces for idx in base_curve_idx]
    curr_err = max([max([c-CURRENT_THRESHOLD, 0.])/CURRENT_THRESHOLD for c in currents_list])
    iota_err = max([np.abs(IOTAS.J() - IOT_TARGET)/np.abs(IOT_TARGET) for IOTAS, IOT_TARGET in zip(IOTAS_LIST, IOTAS_TARGET)])
    mr_err = np.abs(mr.J()-MR_TARGET)/MR_TARGET
    clen_err = max([max(Jl.J() - LENGTH_THRESHOLD, 0)/np.abs(LENGTH_THRESHOLD) for Jl in Jls])

    cc_err = max(COIL_TO_COIL_THRESHOLD-Jccdist.shortest_distance(), 0)/np.abs(COIL_TO_COIL_THRESHOLD)
    _, min_xpoint_to_vessel, min_boozer_surface_to_vessel = J_plasma_to_vessel_margin.shortest_distance()
    plasma_vessel_margin_err = max(PLASMA_VESSEL_MARGIN_THRESHOLD-np.min([min_xpoint_to_vessel, min_boozer_surface_to_vessel]), 0)/np.abs(PLASMA_VESSEL_MARGIN_THRESHOLD)

    min_coil_on_vessel_distance, _, _ = J_coil_on_vessel.shortest_distance()
    coil_on_vessel_err = (
        max(COIL_ON_VESSEL_THRESHOLD - min_coil_on_vessel_distance, 0)
        / np.abs(COIL_ON_VESSEL_THRESHOLD)
    )
    
    min_coil_clearance_distance, _, _ = J_coil_clearance.shortest_distance()
    coil_clearance_err = (
        max(COIL_CLEARANCE_THRESHOLD - min_coil_clearance_distance, 0)
        / np.abs(COIL_CLEARANCE_THRESHOLD)
    )

    vessel_shape_err = 0.
    if vessel_id == 0:
        bx, by, r, rr = sdf.local_full_x.copy()
        vessel_shape_err = max([r-bx, r-by, 0])

    msc = [J.J() for J in Jmscs]
    msc_err = max(np.max(msc) - MEAN_SQUARED_CURVATURE_THRESHOLD, 0)/np.abs(MEAN_SQUARED_CURVATURE_THRESHOLD)

    curv_err = max(max([np.max(c.kappa()) for c in base_curves]) - CURVATURE_THRESHOLD, 0)/np.abs(CURVATURE_THRESHOLD)
    alen_err = np.max([ArclengthVariation(c).J() for c in base_curves])

    modB_err = max([np.abs(modB.J()-MODB_TARGET)/MODB_TARGET for modB in modBs])

    well_err = 0.
    if WELL_THRESHOLD != 0. and WELL_WEIGHT.value != 0.:
        well_err = max([max(Jl.well().max() - WELL_THRESHOLD, 0)/np.abs(WELL_THRESHOLD) for Jl in magnetic_wells])
    elif WELL_THRESHOLD == 0. and WELL_WEIGHT.value != 0.:  # take absolute error in case of WELL_THRESHOLD == 0
        well_err = max([max(Jl.well().max() - WELL_THRESHOLD, 0) for Jl in magnetic_wells])

    meanz_err = max(max([Jfl.max_distance() for Jfl in meanzs]) - FIELDLINE_MEANZ_THRESHOLD, 0) / np.abs(FIELDLINE_MEANZ_THRESHOLD)
    _, max_fieldline_mean_distance, _ = J_fieldline_mean_distance.longest_distance()
    fieldline_mean_distance_err = (
        max(max_fieldline_mean_distance - FIELDLINE_MEANDIST_THRESHOLD, 0)
        / np.abs(FIELDLINE_MEANDIST_THRESHOLD)
    )
    if mono <= 1:
        monodromy_err = (np.max([np.abs(d.matrix - np.eye(d.matrix.shape[0])) for d in tmos])-MONODROMY_THRESHOLD)/MONODROMY_THRESHOLD
    else:
        monodromy_err = (np.max([np.abs(np.trace(d.matrix) - 2) for d in tmos]) - MONODROMY_THRESHOLD) / MONODROMY_THRESHOLD

    # check which constraints are violated and increase weight if violated by more than 0.1%
    if curr_err > 0.001 and CURRENT_WEIGHT.value != 0.:
        CURRENT_WEIGHT*=10
    if (plasma_vessel_margin_err > 0.001 or vessel_shape_err > 0.001) and PLASMA_VESSEL_MARGIN_WEIGHT.value != 0.:
        PLASMA_VESSEL_MARGIN_WEIGHT*=10
        print("PLASMA-VESSEL MARGIN ERROR", plasma_vessel_margin_err)
    if (on_vessel and coil_on_vessel_err > 0.001) and COIL_ON_VESSEL_WEIGHT.value != 0.:
        COIL_ON_VESSEL_WEIGHT *= 10
        print("COIL-ON-VESSEL ERROR", coil_on_vessel_err)
    if ((not on_vessel) and coil_clearance_err > 0.001) and COIL_CLEARANCE_WEIGHT.value != 0.:
        COIL_CLEARANCE_WEIGHT *= 10
        print("COIL-CLEARANCE ERROR", coil_clearance_err)
    if iota_err > 0.001 and IOTAS_WEIGHT.value != 0.:
        IOTAS_WEIGHT*=10
        print("IOTA ERROR", iota_err)
    if mr_err > 0.001 and MAJOR_RADIUS_WEIGHT.value !=0.:
        MAJOR_RADIUS_WEIGHT*=10
        print("MR ERROR", mr_err)
    if clen_err > 0.001 and LENGTH_WEIGHT.value !=0.:
        LENGTH_WEIGHT*=10
        print("COIL LENGTH ERROR", clen_err)
    if cc_err > 0.001 and COIL_TO_COIL_WEIGHT.value != 0.:
        COIL_TO_COIL_WEIGHT*=10
        print("COIL TO COIL ERROR", cc_err)
    if modB_err > 0.001 and MODB_WEIGHT.value != 0.:
        MODB_WEIGHT*=10
        print("MODB ERROR", modB_err)
    if msc_err > 0.001 and MEAN_SQUARED_CURVATURE_WEIGHT.value != 0.:
        MEAN_SQUARED_CURVATURE_WEIGHT*=10
        print("MEAN SQUARED ERROR", msc_err)
    if curv_err > 0.001 and CURVATURE_WEIGHT.value !=0.:
        CURVATURE_WEIGHT*=10
        print("CURVATURE ERROR", curv_err)
    if alen_err > 0.001 and ARCLENGTH_WEIGHT.value != 0.:
        ARCLENGTH_WEIGHT*=10
        print("ARCLENGTH ERROR", alen_err)
    if well_err > 0.001 and WELL_WEIGHT.value != 0.:
        WELL_WEIGHT*=10
        print("WELL ERROR", well_err)
    if meanz_err > 0.001 and FIELDLINE_MEANZ_WEIGHT.value != 0.:
        print("MEANZ ERROR", meanz_err)
        FIELDLINE_MEANZ_WEIGHT*=10
    if fieldline_mean_distance_err > 0.001 and FIELDLINE_MEANDIST_WEIGHT.value !=0.:
        print("FIELDLINE MEAN DIST ERROR", fieldline_mean_distance_err)
        FIELDLINE_MEANDIST_WEIGHT*=10
    if monodromy_err > 0.001 and MONODROMY_WEIGHT.value != 0.:
        MONODROMY_WEIGHT*=10
        print("MONODROMY ERROR", monodromy_err)

    # SN bottom X-point band: inward depth of every bottom field-line point must
    # lie in [BOTTOM_XPOINT_MIN, BOTTOM_XPOINT_MAX]. Relative violation vs the
    # respective bound; escalate the weight until satisfied to 0.1%.
    bottom_err = 0.
    if null_type == 'SN':
        for bx in bottom_xpoints:
            inward = -np.asarray(sdf.pure(bx.curve.gamma(), sdf.local_full_x, 1.0))
            lo = max(BOTTOM_XPOINT_MIN - inward.min(), 0.) / BOTTOM_XPOINT_MIN
            hi = max(inward.max() - BOTTOM_XPOINT_MAX, 0.) / BOTTOM_XPOINT_MAX
            bottom_err = max(bottom_err, lo, hi)
        if bottom_err > 0.001 and BOTTOM_XPOINT_WEIGHT.value != 0.:
            BOTTOM_XPOINT_WEIGHT *= 10
            print("BOTTOM XPOINT ERROR", bottom_err)

    sdf.to_vtk(OUT_DIR + f'vessel_opt_{j}', nx=40, ny=40, nz=40)
    curves_to_vtk(curves, OUT_DIR + f"curves_opt_{j}")
    curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_opt_{j}")
    curves_to_vtk([axis.curve for axis in axes], OUT_DIR + f"ma_opt_{j}")
    for idx, boozer_surface in enumerate(boozer_surfaces):
        boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_{j}")
    save([boozer_surfaces, iota_Gs, axes, xpoints, sdf], OUT_DIR + f'design_opt_{j}.json')
    
    # save the weights in a yaml file
    config['CURRENT_WEIGHT'] = CURRENT_WEIGHT.value
    config['COIL_TO_COIL_WEIGHT'] = COIL_TO_COIL_WEIGHT.value
    config['CURVATURE_WEIGHT'] = CURVATURE_WEIGHT.value
    config['MEAN_SQUARED_CURVATURE_WEIGHT'] = MEAN_SQUARED_CURVATURE_WEIGHT.value
    config['LENGTH_WEIGHT'] = LENGTH_WEIGHT.value
    config['IOTAS_WEIGHT'] = IOTAS_WEIGHT.value
    config['MAJOR_RADIUS_WEIGHT'] = MAJOR_RADIUS_WEIGHT.value
    config['BOOZER_RESIDUAL_WEIGHT'] = BOOZER_RESIDUAL_WEIGHT.value
    config['PLASMA_VESSEL_MARGIN_WEIGHT'] = PLASMA_VESSEL_MARGIN_WEIGHT.value
    config['MODB_WEIGHT'] = MODB_WEIGHT.value
    config['ARCLENGTH_WEIGHT'] = ARCLENGTH_WEIGHT.value
    config['COIL_ON_VESSEL_WEIGHT'] = COIL_ON_VESSEL_WEIGHT.value
    config['COIL_CLEARANCE_WEIGHT'] = COIL_CLEARANCE_WEIGHT.value
    config['WELL_WEIGHT'] = WELL_WEIGHT.value
    config['FIELDLINE_MEANZ_WEIGHT'] = FIELDLINE_MEANZ_WEIGHT.value
    config['FIELDLINE_MEANDIST_WEIGHT'] = FIELDLINE_MEANDIST_WEIGHT.value
    config['MONODROMY_WEIGHT'] = MONODROMY_WEIGHT.value
    if null_type == 'SN':
        config['BOTTOM_XPOINT_WEIGHT'] = BOTTOM_XPOINT_WEIGHT.value

    # Save to YAML
    with open(OUT_DIR + f'design_opt_{j}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

sdf.to_vtk(OUT_DIR + f'vessel_opt_final', nx=40, ny=40, nz=40)
curves_to_vtk(curves, OUT_DIR + f"curves_opt_final")
curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_opt_final")
curves_to_vtk([axis.curve for axis in axes], OUT_DIR + f"ma_opt_final")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_final")
with open(OUT_DIR + f'design_opt_final.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
save([boozer_surfaces, iota_Gs, axes, xpoints, sdf], OUT_DIR + f'design_opt_final.json')

# ---- final summary of constraints, values, and relative errors ----
final_nonqs_pct = 100. * J_nonQSRatio.J()**0.5

# Recompute all error metrics one more time with final dofs
final_metrics = {
    'nonQS_percent':              (final_nonqs_pct,                                None,                          None),
    'current':                    (max(currents_list),                             CURRENT_THRESHOLD,             curr_err if CURRENT_WEIGHT.value != 0. else 0.0),
    'iotas':                      (max(IOTAS.J() for IOTAS in IOTAS_LIST),         max(IOTAS_TARGET),             iota_err if IOTAS_WEIGHT.value != 0. else 0.0),
    'major_radius':               (mr.J(),                                         MR_TARGET,                     mr_err if MAJOR_RADIUS_WEIGHT.value != 0. else 0.0),
    'coil_length':                (max(Jl.J() for Jl in Jls),                      LENGTH_THRESHOLD,              clen_err if LENGTH_WEIGHT.value != 0. else 0.0),
    'coil_to_coil':               (Jccdist.shortest_distance(),                    COIL_TO_COIL_THRESHOLD,        cc_err if COIL_TO_COIL_WEIGHT.value != 0. else 0.0),
    'plasma_vessel_margin':       (min(min_xpoint_to_vessel, min_boozer_surface_to_vessel),
                                                                                   PLASMA_VESSEL_MARGIN_THRESHOLD, plasma_vessel_margin_err if PLASMA_VESSEL_MARGIN_WEIGHT.value != 0. else 0.0),
    'coil_on_vessel':             (min_coil_on_vessel_distance,                    COIL_ON_VESSEL_THRESHOLD,      coil_on_vessel_err if COIL_ON_VESSEL_WEIGHT.value != 0. else 0.0),
    'coil_clearance':             (min_coil_clearance_distance,                    COIL_CLEARANCE_THRESHOLD,      coil_clearance_err if COIL_CLEARANCE_WEIGHT.value != 0. else 0.0),
    'msc':                        (max(msc),                                       MEAN_SQUARED_CURVATURE_THRESHOLD, msc_err if MEAN_SQUARED_CURVATURE_WEIGHT.value != 0. else 0.0),
    'curvature':                  (max(np.max(c.kappa()) for c in base_curves),    CURVATURE_THRESHOLD,           curv_err if CURVATURE_WEIGHT.value != 0. else 0.0),
    'arclength':                  (alen_err,                                       0.0,                           alen_err if ARCLENGTH_WEIGHT.value != 0. else 0.0),
    'modB':                       (max(modB.J() for modB in modBs),                MODB_TARGET,                   modB_err if MODB_WEIGHT.value != 0. else 0.0),
    'well':                       (max(w.well().max() for w in magnetic_wells),    WELL_THRESHOLD,                well_err if WELL_WEIGHT.value != 0. else 0.0),
    'fieldline_meanz':            (max(Jfl.max_distance() for Jfl in meanzs),      FIELDLINE_MEANZ_THRESHOLD,     meanz_err if FIELDLINE_MEANZ_WEIGHT.value != 0. else 0.0),
    'fieldline_meandist':         (max_fieldline_mean_distance,                    FIELDLINE_MEANDIST_THRESHOLD,  fieldline_mean_distance_err if FIELDLINE_MEANDIST_WEIGHT.value != 0. else 0.0),
    'monodromy':                  (np.max([np.abs(d.matrix - np.eye(d.matrix.shape[0])).max() for d in tmos]) if mono <= 1
                                   else np.max([np.abs(np.trace(d.matrix) - 2) for d in tmos]),
                                                                                   MONODROMY_THRESHOLD,           monodromy_err if MONODROMY_WEIGHT.value != 0. else 0.0),
}

# also record the full 2x2 monodromy (tangent) matrix per X-point so the
# device browser can display it; entries are descriptive (no threshold/error).
for ti, d in enumerate(tmos):
    Mm = np.asarray(d.matrix)
    for a in range(2):
        for b in range(2):
            final_metrics[f'monodromy_M{a}{b}_idx{ti}'] = (float(Mm[a, b]), None, None)

# bottom X-point band (SN only): achieved inward-depth range and the relative
# band violation against [BOTTOM_XPOINT_MIN, BOTTOM_XPOINT_MAX].
if null_type == 'SN':
    inw_min = min(float((-np.asarray(sdf.pure(bx.curve.gamma(), sdf.local_full_x, 1.0))).min())
                  for bx in bottom_xpoints)
    inw_max = max(float((-np.asarray(sdf.pure(bx.curve.gamma(), sdf.local_full_x, 1.0))).max())
                  for bx in bottom_xpoints)
    final_metrics['bottom_xpoint_inward_min'] = (
        inw_min, BOTTOM_XPOINT_MIN, max(BOTTOM_XPOINT_MIN - inw_min, 0.) / BOTTOM_XPOINT_MIN)
    final_metrics['bottom_xpoint_inward_max'] = (
        inw_max, BOTTOM_XPOINT_MAX, max(inw_max - BOTTOM_XPOINT_MAX, 0.) / BOTTOM_XPOINT_MAX)

# write a human-readable summary
with open(OUT_DIR + 'summary.txt', 'w') as f:
    f.write(f"# {'metric':<30s} {'value':>16s} {'threshold':>16s} {'rel_error':>16s}\n")
    for name, (value, threshold, rel_err) in final_metrics.items():
        thr_str = f"{threshold:.6e}" if threshold is not None else "n/a"
        err_str = f"{rel_err:.6e}"   if rel_err   is not None else "n/a"
        f.write(f"  {name:<30s} {value:.6e}   {thr_str:>16s}   {err_str:>16s}\n")

# write a machine-readable max relative error for the prefix.sh check
max_rel_err = max(abs(v[2]) for v in final_metrics.values() if v[2] is not None)
with open(OUT_DIR + 'max_rel_error.txt', 'w') as f:
    f.write(f"{max_rel_err:.18e}\n")

# keep nonQS.txt for backwards compatibility
np.savetxt(OUT_DIR + f'nonQS.txt', np.array([final_nonqs_pct]))
