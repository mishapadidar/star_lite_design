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
    LinkingNumber,
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
from star_lite_design.utils.xpoint_surface_distance import XpointSurfaceDistance
from star_lite_design.utils.curve_periodicfieldline_distance import CurvesPeriodicFieldlineDistance


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
# --AR: aspect-ratio knob. 0 = leave the aspect ratio as-is. 1 = after each BFGS
# run, try to lower the plasma aspect ratio to ~5 by retargeting the Boozer-surface
# volume (torus estimate V = 2*pi^2*R*r^2 with r = R/5); see the AR block in the
# optimization loop. Reverts if the re-solve fails / self-intersects; stops once it
# succeeds once.
parser.add_argument("--AR", type=int, default=0, choices=[0, 1])

args = parser.parse_args()

# aspect-ratio reduction enabled? (ar_done flips True once a lower-AR surface has
# converged, so it is only attempted until it first succeeds.)
ar_enabled = bool(args.AR)
ar_done = False

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
             f"_mono={mono}_null={null_type}_num_aux={num_aux}_attempt={attempt}"
             f"_AR={args.AR}")
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
# Record the well ON/OFF state explicitly so the downstream polish can load it
# (well OFF -> WELL_ACTIVE False and WELL_WEIGHT 0.0; both are written to the
# design_opt_final.yaml).
config['WELL_ACTIVE'] = bool(well_active)
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
# Record, for the downstream singular polish, which configuration this design
# is for and which monodromy constraint to enforce (mono 1 -> M=I 'identity',
# mono 2 -> tr(M)=2 'trace'; mono 0 -> no polish). The polish reads these from
# the saved yaml so it needs no command-line arguments.
config['CONFIG_ID'] = config_id
config['MONODROMY_CONSTRAINT'] = {1: 'identity', 2: 'trace'}.get(mono, None)
tmos = [TangentMap(xp, BiotSavart(boozer_surface.biotsavart.coils), MONODROMY_THRESHOLD, mtype='identity' if mono == 1 else 'jordan') for xp, boozer_surface in zip(xpoints, boozer_surfaces)]
MONODROMY_LIST = [Monodromy(tmo) for tmo in tmos]
J_monodromy = sum(MONODROMY_LIST) # target rotational transform is that computed on the initial surface

# X-point-to-surface inequality constraint (SN only): the BOTTOM X-point must
# stay at least XPOINT_SURFACE_THRESHOLD (10 cm) AWAY from its Boozer surface
# (kind='min'). One penalty per configuration.
J_bot_surf = None
bot_surf_penalties = []
if null_type == 'SN':
    config['XPOINT_SURFACE_THRESHOLD'] = 0.10
    config['BOTTOM_XPOINT_SURFACE_WEIGHT'] = 1e3
    XPOINT_SURFACE_THRESHOLD = config['XPOINT_SURFACE_THRESHOLD']
    BOTTOM_XPOINT_SURFACE_WEIGHT = Weight(config['BOTTOM_XPOINT_SURFACE_WEIGHT'])
    bot_surf_penalties = [XpointSurfaceDistance(bx, bsurf, XPOINT_SURFACE_THRESHOLD, kind='min')
                          for bx, bsurf in zip(bottom_xpoints, boozer_surfaces)]
    J_bot_surf = sum(bot_surf_penalties)
else:
    BOTTOM_XPOINT_SURFACE_WEIGHT = Weight(0.0)

# Coil-clearance inequality constraint (SN only): the BOTTOM X-point must stay
# at least BOTTOM_XPOINT_COIL_THRESHOLD (6 cm) away from the coils.
J_bot_coil = None
bot_coil_penalties = []
if null_type == 'SN':
    config['BOTTOM_XPOINT_COIL_THRESHOLD'] = 0.10
    config['BOTTOM_XPOINT_COIL_WEIGHT'] = 1e3
    BOTTOM_XPOINT_COIL_THRESHOLD = config['BOTTOM_XPOINT_COIL_THRESHOLD']
    BOTTOM_XPOINT_COIL_WEIGHT = Weight(config['BOTTOM_XPOINT_COIL_WEIGHT'])
    bot_coil_penalties = [CurvesPeriodicFieldlineDistance(curves, bx, BOTTOM_XPOINT_COIL_THRESHOLD)
                          for bx in bottom_xpoints]
    J_bot_coil = sum(bot_coil_penalties)
else:
    BOTTOM_XPOINT_COIL_WEIGHT = Weight(0.0)

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

# JF_base excludes the SN bottom-X-point penalties. For SN the ACTIVE objective
# JF adds the bottom X-point-to-surface and X-point-to-coil penalties, and BFGS
# defaults to it; once the X-point-surface penalty reaches 0 we switch back to
# JF_base and stop tracking the bottom X-point (see the optimization loop).
JF_base = JF
track_bottom = (null_type == 'SN')
if track_bottom:
    JF = JF_base + BOTTOM_XPOINT_SURFACE_WEIGHT * J_bot_surf + BOTTOM_XPOINT_COIL_WEIGHT * J_bot_coil


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
if J_bot_surf is not None:
    penalties['bottom xpoint-surface'] = BOTTOM_XPOINT_SURFACE_WEIGHT * J_bot_surf
if J_bot_coil is not None:
    penalties['bottom xpoint-coil'] = BOTTOM_XPOINT_COIL_WEIGHT * J_bot_coil

# Weight object backing each penalty above, keyed identically (used by the
# post-perturbation pre-scaling below). 'nonQS' has an implicit weight of 1 (no
# Weight object) and is the primary objective, so it is omitted and never
# scaled. The bottom-X-point penalties are special drop-when-satisfied
# constraints (not in the in-loop escalation), so they are likewise omitted.
penalty_weights = {
        'monodromy': MONODROMY_WEIGHT,
        'iotas': IOTAS_WEIGHT,
        'length': LENGTH_WEIGHT,
        'coil-to-coil': COIL_TO_COIL_WEIGHT,
        'curvature': CURVATURE_WEIGHT,
        'mean-squared curvature': MEAN_SQUARED_CURVATURE_WEIGHT,
        'arclength': ARCLENGTH_WEIGHT,
        'Boozer residual': BOOZER_RESIDUAL_WEIGHT,
        'modB': MODB_WEIGHT,
        'plasma-boundary-to-vessel': PLASMA_VESSEL_MARGIN_WEIGHT,
        'coil-on-vessel': COIL_ON_VESSEL_WEIGHT,
        'coil-clearance': COIL_CLEARANCE_WEIGHT,
        'current': CURRENT_WEIGHT,
        'well': WELL_WEIGHT,
        'fieldline meanz': FIELDLINE_MEANZ_WEIGHT,
        'fieldline mean dist': FIELDLINE_MEANDIST_WEIGHT,
        'major radius': MAJOR_RADIUS_WEIGHT,
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
# (base_curve_idx) by N(0, 1cm). attempt 0 is the UNPERTURBED reference device
# (no jitter); attempts >= 1 each start the optimizer from a different perturbed
# point. This runs AFTER the stellsym dofs are fixed above and
# only touches FREE dofs (we set only names present in local_dof_names), so the
# enforced symmetry is preserved. Seeded by DEVICE_ID for reproducibility.
#
# After perturbing, the magnetic axis, Boozer surface and X-point are re-solved
# with the perturbed coils. If any solve fails or the surface self-intersects,
# the axis/surface/X-point are rolled back to their unperturbed dofs and a
# fresh perturbation is drawn, repeating until all three converge to a
# non-self-intersecting state. A generous cap turns a pathological case into a
# task failure rather than an indefinite hang.
PERTURB_STD = 0.01   # metres (2 cm, both DN and SN)
PERTURB_ORDERS = (0, 1, 2)   # Fourier harmonics to perturb
MAX_PERTURB_TRIES = 10
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
bottom_ref = [(bx, bx.curve.x.copy()) for bx in (bottom_xpoints or [])]


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
    for bx, xref in bottom_ref:
        bx.curve.x = xref
        bx.need_to_run_code = True


def _apply_perturbation():
    for cc in perturb_curves:
        free = set(cc.local_dof_names)   # only perturb unfixed dofs
        for nm in _PERTURB_NAMES:
            if nm in free:
                cc.set(nm, cc.get(nm) + pert_rng.normal(0.0, PERTURB_STD))


def _resolve_perturbed():
    """Re-solve axis, X-point, bottom X-point (SN) and Boozer surface with the
    perturbed coils. Returns True iff all converge and no surface
    self-intersects."""
    for ax, _ in axes_ref:
        ax.need_to_run_code = True
        ax.run_code(CurveLength(ax.curve).J())
    for xp, _ in xpoints_ref:
        xp.need_to_run_code = True
        xp.run_code(CurveLength(xp.curve).J())
    for bx, _ in bottom_ref:
        bx.need_to_run_code = True
        bx.run_code(CurveLength(bx.curve).J())
    for bs, _s, iota, G in surf_ref:
        bs.need_to_run_code = True
        bs.run_code(iota, G)
    return (all(ax.res['success'] for ax in axes)
            and all(xp.res['success'] for xp in xpoints)
            and all(bx.res['success'] for bx in (bottom_xpoints or []))
            and all(bs.res['success'] for bs in boozer_surfaces)
            and not any(bs.surface.is_self_intersecting() for bs in boozer_surfaces))


if attempt == 0:
    # attempt 0 is the UNPERTURBED reference device: keep the base coils exactly
    # as loaded (already solved above) and skip the perturbation entirely.
    print("attempt 0: no base-coil perturbation (unperturbed reference device)")
else:
    # SIMSOPT Gauss linking number over the full modular-coil set: 0 if no pair
    # is interlocked, >=1 otherwise. The unperturbed coils are unlinked; a
    # perturbation that interlocks them is rejected and resampled below.
    J_linking_number = LinkingNumber(curves)
    print(f"unperturbed modular-coil linking number: {int(J_linking_number.J())}")

    _perturb_solved = False
    for _ktry in range(MAX_PERTURB_TRIES):
        _restore_unperturbed()
        _apply_perturbation()
        # Reject (and resample with a smaller amplitude) any perturbation that
        # interlocks the modular coils, BEFORE paying for the re-solve.
        _ln = int(J_linking_number.J())
        if _ln != 0:
            PERTURB_STD = PERTURB_STD / 2.0
            print(f"perturbation try {_ktry}: modular coils interlink (linking number "
                  f"{_ln}); halving std to {PERTURB_STD:.2e} and resampling")
            continue
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

    # The yaml weights were tuned for the unperturbed device; a perturbation can
    # push a penalty's weighted contribution weight*penalty.J() above 1, letting
    # it dominate the objective. Mirror boozer_singular_opt.py: after the
    # perturbation, shrink any weight whose CURRENT product exceeds 1 -- by
    # repeated factors of 10 -- so each weighted term starts below 1. The in-loop
    # x10 escalation can raise it back as the constraints demand. Modifying each
    # Weight in place is seen by JF/JF_base (same objects).
    print("Pre-scaling penalty weights so each weighted term starts below 1 after the perturbation:")
    for _name, _scaled in penalties.items():
        _w = penalty_weights.get(_name)
        if _w is None or _w.value == 0.:
            continue
        _prod = _scaled.J()
        if not np.isfinite(_prod) or _prod <= 1.0:
            continue
        _bare = _prod / _w.value          # the unweighted penalty value (dofs fixed here)
        while _w.value * _bare > 1.0:
            _w *= 0.1
        print(f"  {_name}: weight -> {_w.value:.3e}  (penalty={_bare:.3e}, product {_prod:.3e} -> {_w.value * _bare:.3e})")

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
# The bottom X-points (SN) are re-solved each evaluation but their curve dofs are
# NOT optimization variables, so they must be snapshotted/restored explicitly
# like the axis/top-xpoint; otherwise a corrupted bottom field line poisons every
# later gradient (modB->0). Empty for DN (bottom_xpoints is None).
bottom_res_list = [{'xdofs': bx.curve.x.copy(), 'length': bx.res['length']} for bx in (bottom_xpoints or [])]
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
    for res, bx in zip(bottom_res_list, bottom_xpoints or []):
        res['xdofs'] = bx.curve.x.copy()
        res['length'] = bx.res['length']
    
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
    # Split the penalties over two stacked tables so the many columns don't get
    # truncated ("…") to fit the console width.
    pen_items = list(penalties.items())
    half = (len(pen_items) + 1) // 2
    for chunk in (pen_items[:half], pen_items[half:]):
        table1 = Table(expand=False, show_header=False)
        table1.add_row(*[k for k, _ in chunk])
        table1.add_row(*[f"{v.J():.4e}" for _, v in chunk])
        console.print(table1)
    
    table2 = Table(expand=False, show_header=False) 
    for k in states.keys():
        table2.add_row(k, ' '.join([f'{J.J():.4e}' for J in states[k]]))
    table2.add_row('xpoint_top(0)', ' '.join([f'{np.array2string(xpoint.curve.gamma()[0])}' for xpoint in xpoints]))
    if bottom_xpoints is not None:
        table2.add_row('xpoint_bottom(0)', ' '.join([f'{np.array2string(bx.curve.gamma()[0])}' for bx in bottom_xpoints]))
    table2.add_row('monodromys', ' '.join([f'{np.array2string(d.matrix)}' for d in tmos]))
    table2.add_row('trace(monodromys)', ' '.join([f'{float(np.trace(d.matrix)):.4e}' for d in tmos]))
    table2.add_row('well', ' '.join([f'{w.well().max():.3e}' for w in magnetic_wells]))
    table2.add_row('currents', ' '.join([f'{curr:.3e}' for curr in currents_list]))
    table2.add_row('curvatures', ' '.join([f'{curv:.3e}' for curv in kappas]))
    #table2.add_row('minimum X-point-to-vessel distance', ' '.join([f'{Jxv.shortest_distance():.3e}' for Jxv in Jxvs]))
    _, min_xpoint_to_vessel, min_boozer_surface_to_vessel = J_plasma_to_vessel_margin.shortest_distance()
    table2.add_row('minimum X-point-to-vessel distance', f'{min_xpoint_to_vessel:.3e}')
    table2.add_row('minimum Boozer surface-to-vessel distance', f'{min_boozer_surface_to_vessel:.3e}')
    # X-point-to-surface and X-point-to-coil distances (SN): bottom must stay
    # beyond each threshold (closest approach).
    if bottom_xpoints is not None and bot_surf_penalties:
        table2.add_row('bottom X-point-surface min dist',
                       ' '.join([f'{p.min_distance():.3e}' for p in bot_surf_penalties]))
    if bottom_xpoints is not None and bot_coil_penalties:
        table2.add_row('bottom X-point-coil min dist',
                       ' '.join([f'{p.shortest_distance():.3e}' for p in bot_coil_penalties]))
    
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
        curves_to_vtk([xpoint.curve for xpoint in xpoints] + [bx.curve for bx in (bottom_xpoints or [])], OUT_DIR + f"xpoint_curves_tmp")
        curves_to_vtk([axis.curve for axis in axes], OUT_DIR + "ma_tmp")
        for idx, boozer_surface in enumerate(boozer_surfaces):
            boozer_surface.surface.to_vtk(OUT_DIR + f"surf_tmp_{idx}")
        save([boozer_surfaces, iota_Gs, axes, xpoints, sdf], OUT_DIR + f'design_tmp.json')
    
    dat_dict["iter"] += 1

    # Once the bottom X-point-to-surface penalty has reached 0 at an ACCEPTED
    # step, break out of BFGS (caught in the loop) so we can drop the bottom
    # penalties and stop tracking the bottom X-point. dat_dict['x'] was just
    # updated to this accepted point, so the restart is consistent.
    if track_bottom and float(J_bot_surf.J()) <= 0.0:
        # Snapshot the full state at the moment the bottom X-point is dropped.
        # Filenames carry the "xpoint_deletion" tag, and the bottom X-point is
        # included (in the xpoint VTK and the saved xpoints list) so it is
        # captured before it stops being tracked.
        sdf.to_vtk(OUT_DIR + 'vessel_opt_xpoint_deletion', nx=40, ny=40, nz=40)
        curves_to_vtk(curves, OUT_DIR + "curves_opt_xpoint_deletion")
        curves_to_vtk([xp.curve for xp in xpoints] + [bx.curve for bx in (bottom_xpoints or [])],
                      OUT_DIR + "xpoint_curves_opt_xpoint_deletion")
        curves_to_vtk([ax.curve for ax in axes], OUT_DIR + "ma_opt_xpoint_deletion")
        for idx, bsurf in enumerate(boozer_surfaces):
            bsurf.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_xpoint_deletion")
        save([boozer_surfaces, iota_Gs, axes, xpoints + (bottom_xpoints or []), sdf],
             OUT_DIR + 'design_opt_xpoint_deletion.json')
        raise _XpointSurfaceSatisfied()

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
    # Restore the bottom X-point curves too (SN) so a corrupted bottom field line
    # does not poison the next gradient; force a re-solve from the good guess.
    for res, bx in zip(bottom_res_list, bottom_xpoints or []):
        bx.curve.x = res['xdofs']
        bx.res['length'] = res['length']
        bx.need_to_run_code = True
    JF.x = dat_dict['x']


class _XpointSurfaceSatisfied(Exception):
    """Raised inside the BFGS objective when the bottom X-point-to-surface
    penalty reaches 0, to break out of the solver so the caller can drop the
    bottom-X-point penalties and stop tracking the bottom X-point."""
    pass


# Failed-evaluation barrier: aim the 1-D parabola minimum at ~this fraction of
# the failed step, so the line search retreats to a short feasible step rather
# than all the way to zero (which would stall BFGS).
BARRIER_RETREAT = 0.1


def fun(dofs):
    JF.x = dofs

    fail_reasons = []
    try:
        J = JF.J()
        grad = JF.dJ()
    except Exception as e:
        fail_reasons.append(f'JF.J()/dJ() raised: {e!r}')

    for i, bs in enumerate(boozer_surfaces):
        if not bs.res['success']:
            fail_reasons.append(f'boozer_surface[{i}] Newton solve did not converge')
        # is_self_intersecting() can itself raise; treat a raise as self-intersecting.
        try:
            if bs.surface.is_self_intersecting():
                fail_reasons.append(f'boozer_surface[{i}] surface is self-intersecting')
        except Exception as e:
            fail_reasons.append(f'boozer_surface[{i}] is_self_intersecting() raised: {e!r} (assumed self-intersecting)')
    for i, fl in enumerate(axes):
        if not fl.res['success']:
            fail_reasons.append(f'axis[{i}] Newton solve did not converge')
    for i, fl in enumerate(xpoints):
        if not fl.res['success']:
            fail_reasons.append(f'xpoint[{i}] Newton solve did not converge')

    if fail_reasons:
        print('failed — quadratic barrier toward last good point: ' + '; '.join(fail_reasons))
        _restore_state()
        # The trial step left the solvable region (Newton solves diverged /
        # surface self-intersects). Return a SMOOTH, CONSISTENT quadratic barrier
        # anchored at the last good point (x0, f0, g0):
        #     f(x) = f0 + g0.(x-x0) + (K/2)||x-x0||^2 ,   g(x) = g0 + K (x-x0)
        # Because g == grad f exactly and, along the BFGS search direction, f is a
        # convex parabola, the Wolfe line search backtracks to a short step just
        # inside the feasible region instead of failing on a flat/inconsistent
        # wall. K places the parabola minimum at ~BARRIER_RETREAT of this failed
        # step (scale-aware, floored to stay strongly convex).
        x0, f0, g0 = dat_dict['x'], dat_dict['J'], dat_dict['dJ']
        dx = dofs - x0
        nrm2 = float(dx @ dx) + 1e-300
        gdx = float(g0 @ dx)
        K = max(abs(gdx) / (BARRIER_RETREAT * nrm2), 1.0)
        return f0 + gdx + 0.5 * K * nrm2, g0 + K * dx

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

def _drop_bottom_penalties():
    """Bottom X-point-surface penalty hit 0: drop both bottom-X-point penalties,
    switch BFGS to the base objective, and stop tracking the bottom X-point."""
    global JF, bottom_xpoints, track_bottom
    JF = JF_base
    bottom_xpoints = None
    track_bottom = False
    penalties.pop('bottom xpoint-surface', None)
    penalties.pop('bottom xpoint-coil', None)


dofs = JF.x.copy()
# The sentinel can fire on the very first callback if the bottom
# X-point-surface constraint is ALREADY satisfied at initialization; handle it
# the same way as inside the loop instead of crashing the whole task.
try:
    callback(dofs)
except _XpointSurfaceSatisfied:
    _drop_bottom_penalties()
    dofs = dat_dict['x'].copy()
    print('XpointSurfaceDistance already 0 at initialization: dropped bottom '
          'X-point penalties, switched to base objective')


def _attempt_ar_reduction(j, dofs):
    """Aspect-ratio reduction (--AR 1), run before each BFGS step (see the call site).
    Raise the Boozer-surface target volume toward the AR=5 torus estimate
    V = 2*pi^2*R*r^2 (R = current major radius, r = R/5). If the full AR=5 target
    converges with no self-intersection, adopt it and return True (caller then stops
    retrying). Otherwise BISECT (at most 5 times) between the current known-good
    volume and the AR=5 target, adopt the LARGEST volume that still converges without
    self-intersecting, and return False (retry next outer iteration once BFGS has
    adapted the coils). On any adopted volume change, refresh the _restore_state
    snapshot, downweight any penalty whose weighted value exceeds 1, and refresh the
    failed-evaluation barrier anchor (dat_dict)."""
    bs = boozer_surfaces[0]
    R_cur = mr.J()
    V_new = 2.0 * np.pi**2 * R_cur * (R_cur / 5.0)**2   # AR=5 torus volume
    V_old = bs.targetlabel
    # the current (known-good) surface is the fallback "best".
    best = {'V': V_old, 'sdofs': bs.surface.x.copy(),
            'iota': bs.res['iota'], 'G': bs.res['G']}

    def _try_volume(V):
        """Warm-start from the best working surface, retarget to V, re-solve;
        True iff it converges with no self-intersection."""
        bs.surface.x = best['sdofs']
        bs.res['iota'], bs.res['G'] = best['iota'], best['G']
        bs.targetlabel = V
        bs.need_to_run_code = True
        rr = bs.run_code(best['iota'], best['G'])
        return bool(rr['success']) and not bs.surface.is_self_intersecting()

    def _snapshot(V):
        return {'V': V, 'sdofs': bs.surface.x.copy(),
                'iota': bs.res['iota'], 'G': bs.res['G']}

    done = False
    if V_new <= V_old:
        # current volume already >= AR=5 volume (AR already <= 5): nothing to do.
        print(f"[AR] j={j}: current AR already <= 5 (V={V_old:.6e} >= V_AR5={V_new:.6e}); done")
        done = True
    elif _try_volume(V_new):
        best = _snapshot(V_new)
        done = True
        print(f"[AR] j={j}: AR=5 reached on first try (V {V_old:.6e} -> {V_new:.6e}); won't retry")
    else:
        # bisect for the LARGEST working volume in (V_old, V_new]; lo works, hi fails.
        lo, hi = V_old, V_new
        for _bi in range(5):
            mid = 0.5 * (lo + hi)
            if _try_volume(mid):
                best = _snapshot(mid)
                lo = mid
            else:
                hi = mid
        _ar_best = R_cur / (best['V'] / (2.0 * np.pi**2 * R_cur))**0.5
        print(f"[AR] j={j}: AR=5 (V={V_new:.6e}) did not converge; bisected (<=5x) to "
              f"largest working volume V={best['V']:.6e} (AR~{_ar_best:.3f}); retry next iteration")

    # adopt the chosen volume: restore that surface state, retarget, re-solve.
    bs.surface.x = best['sdofs']
    bs.res['iota'], bs.res['G'] = best['iota'], best['G']
    bs.targetlabel = best['V']
    bs.need_to_run_code = True
    bs.run_code(best['iota'], best['G'])
    if best['V'] != V_old:
        print("[AR] adopting new volume; re-scaling penalty weights so none start above 1:")
        # refresh the _restore_state warm-start snapshot to the adopted surface
        for _res, _bs in zip(res_list, boozer_surfaces):
            _res['sdofs'] = _bs.surface.x.copy()
            _res['iota'] = _bs.res['iota']
            _res['G'] = _bs.res['G']
        # downweight any penalty whose CURRENT weighted contribution exceeds 1
        for _name, _scaled in penalties.items():
            _w = penalty_weights.get(_name)
            if _w is None or _w.value == 0.:
                continue
            _prod = _scaled.J()
            if not np.isfinite(_prod) or _prod <= 1.0:
                continue
            _bare = _prod / _w.value
            while _w.value * _bare > 1.0:
                _w *= 0.1
            print(f"  {_name}: weight -> {_w.value:.3e}  (product was {_prod:.3e})")
        J0, dJ0 = fun(dofs)
        dat_dict['x'] = dofs.copy()
        dat_dict['J'] = J0
        dat_dict['dJ'] = dJ0.copy()
    return done


for j in range(10):
    dat_dict["iter"] = 0

    # Aspect-ratio reduction (--AR 1): before the BFGS run, push the volume toward
    # AR=5 (bisecting for the largest feasible volume). See _attempt_ar_reduction.
    if ar_enabled and not ar_done:
        ar_done = _attempt_ar_reduction(j, dofs)

    try:
        res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)
        msg = res.message
    except _XpointSurfaceSatisfied:
        _drop_bottom_penalties()
        msg = 'XpointSurfaceDistance reached 0: dropped bottom X-point penalties, switched to base objective'
    except Exception as e:
        msg = f'caught exception: {e}, restarting from last successful callback.'

    # Resume from the last ACCEPTED state (dat_dict), restoring the live
    # surface/axis/xpoint warm-start to match it. res.x can be left on a different
    # (e.g. self-intersecting) branch -- BFGS may stop on a feasible line-search
    # probe, not a _restore_state -- so re-solving fun(res.x) from that stale
    # warm-start can pick a different surface solution. Reset dofs to the accepted
    # point and _restore_state() so the post-min eval and the next run start
    # consistent and feasible. (Safe after _drop_bottom_penalties: _restore_state
    # iterates `bottom_xpoints or []`.)
    dofs = dat_dict['x'].copy()
    _restore_state()

    print(msg)
    J0, dJ0 = fun(dofs)
    # Same hazard as the initial callback: the penalty can reach 0 exactly at
    # the post-minimize state (e.g. BFGS stopped for its own reasons on an
    # iterate that satisfies the constraint).
    try:
        callback(dofs)
    except _XpointSurfaceSatisfied:
        _drop_bottom_penalties()
        print('XpointSurfaceDistance reached 0 at post-minimize callback: dropped '
              'bottom X-point penalties, switched to base objective')
    
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

    # SN X-point-to-surface constraint: the bottom must stay BEYOND threshold
    # (its closest approach > threshold). Relative violation vs the threshold;
    # escalate the weight until satisfied to 0.1%.
    bot_surf_err = 0.
    bot_coil_err = 0.
    if null_type == 'SN' and bottom_xpoints is not None:
        bot_surf_err = max(max(XPOINT_SURFACE_THRESHOLD - p.min_distance(), 0.) / XPOINT_SURFACE_THRESHOLD
                           for p in bot_surf_penalties)
        if bot_surf_err > 0.001 and BOTTOM_XPOINT_SURFACE_WEIGHT.value != 0.:
            BOTTOM_XPOINT_SURFACE_WEIGHT *= 10
            print("BOTTOM XPOINT-SURFACE ERROR", bot_surf_err)
        bot_coil_err = max(max(BOTTOM_XPOINT_COIL_THRESHOLD - p.shortest_distance(), 0.) / BOTTOM_XPOINT_COIL_THRESHOLD
                           for p in bot_coil_penalties)
        if bot_coil_err > 0.001 and BOTTOM_XPOINT_COIL_WEIGHT.value != 0.:
            BOTTOM_XPOINT_COIL_WEIGHT *= 10
            print("BOTTOM XPOINT-COIL ERROR", bot_coil_err)

    # Refresh the failed-evaluation barrier anchor with the JUST-ESCALATED weights.
    # callback() above captured dat_dict['J']/['dJ'] BEFORE these weight bumps, so on
    # the NEXT BFGS run's first line search the barrier (anchored at dat_dict) would
    # be built from STALE, pre-escalation weights -- inconsistent with the new-weight
    # phi(0)/phi'(0) the line search measures at this same point, which makes the
    # barrier the "inconsistent wall" it was designed to avoid (every probe fails ->
    # line search gives up with no accepted step). Re-evaluating fun() here at the
    # unchanged dofs rebuilds (J, dJ) under the new weights so the barrier is again a
    # faithful local model. fun() reuses the warm-started (feasible) solves, so this
    # is one cheap extra evaluation per outer iteration.
    J0, dJ0 = fun(dofs)
    dat_dict['x'] = dofs.copy()
    dat_dict['J'] = J0
    dat_dict['dJ'] = dJ0.copy()

    sdf.to_vtk(OUT_DIR + f'vessel_opt_{j}', nx=40, ny=40, nz=40)
    curves_to_vtk(curves, OUT_DIR + f"curves_opt_{j}")
    curves_to_vtk([xpoint.curve for xpoint in xpoints] + [bx.curve for bx in (bottom_xpoints or [])], OUT_DIR + f"xpoint_curves_opt_{j}")
    curves_to_vtk([axis.curve for axis in axes], OUT_DIR + f"ma_opt_{j}")
    for idx, boozer_surface in enumerate(boozer_surfaces):
        boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_{j}")
    save([boozer_surfaces, iota_Gs, axes, xpoints + (bottom_xpoints or []), sdf], OUT_DIR + f'design_opt_{j}.json')

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
        config['BOTTOM_XPOINT_SURFACE_WEIGHT'] = BOTTOM_XPOINT_SURFACE_WEIGHT.value
        config['BOTTOM_XPOINT_COIL_WEIGHT'] = BOTTOM_XPOINT_COIL_WEIGHT.value

    # Save to YAML
    with open(OUT_DIR + f'design_opt_{j}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

sdf.to_vtk(OUT_DIR + f'vessel_opt_final', nx=40, ny=40, nz=40)
curves_to_vtk(curves, OUT_DIR + f"curves_opt_final")
curves_to_vtk([xpoint.curve for xpoint in xpoints] + [bx.curve for bx in (bottom_xpoints or [])], OUT_DIR + f"xpoint_curves_opt_final")
curves_to_vtk([axis.curve for axis in axes], OUT_DIR + f"ma_opt_final")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_final")
# The final design json (and its paired weights yaml) carry the device ID in the
# name: design_opt_final_<DEVICE_ID>.json. The yaml shares the basename because the
# polish step reads it as the json's sibling (splitext(json)[0] + '.yaml').
with open(OUT_DIR + f'design_opt_final_{DEVICE_ID}.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
save([boozer_surfaces, iota_Gs, axes, xpoints, sdf], OUT_DIR + f'design_opt_final_{DEVICE_ID}.json')

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
# Greene's residue R = (2 - tr(M))/4 is recorded for every X-point in all cases.
for ti, d in enumerate(tmos):
    Mm = np.asarray(d.matrix)
    for a in range(2):
        for b in range(2):
            final_metrics[f'monodromy_M{a}{b}_idx{ti}'] = (float(Mm[a, b]), None, None)
    final_metrics[f'greene_residue_idx{ti}'] = (
        (2.0 - float(np.trace(Mm))) / 4.0, None, None)

# X-point-to-surface distance (SN only): bottom beyond threshold (closest
# approach), with relative violation.
if null_type == 'SN' and bottom_xpoints is not None:
    bot_min = min(p.min_distance() for p in bot_surf_penalties)
    final_metrics['bottom_xpoint_surface_mindist'] = (
        bot_min, XPOINT_SURFACE_THRESHOLD,
        max(XPOINT_SURFACE_THRESHOLD - bot_min, 0.) / XPOINT_SURFACE_THRESHOLD)
    bot_coil_min = min(p.shortest_distance() for p in bot_coil_penalties)
    final_metrics['bottom_xpoint_coil_mindist'] = (
        bot_coil_min, BOTTOM_XPOINT_COIL_THRESHOLD,
        max(BOTTOM_XPOINT_COIL_THRESHOLD - bot_coil_min, 0.) / BOTTOM_XPOINT_COIL_THRESHOLD)

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
