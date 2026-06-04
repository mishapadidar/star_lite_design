#!/usr/bin/env python3
"""boozer_singular_opt.py

Continue a boozer_all.py optimization WITH the singular polish in the loop.

Takes as input a design json produced by boozer_all.py
([boozer_surfaces, iota_Gs, axes, xpoints, sdf]).  For each X-point a
SingularPeriodicFieldline_diff is created with --num-aux planar circular
auxiliary coils:

  1) initially ALL aux currents and radii are DEPENDENT (fixed dofs, solved by
     the under-determined pinv Newton polish; z is independent/held);
  2) then the partition is re-drawn: everything is unfixed and only the
     dependent currents are fixed -- 1 current for the 'trace' system, 3 for
     'identity' -- giving a square LU Newton polish.  The remaining mu
     (radii, z, other currents) become free dofs, i.e. design variables.

The optimization then adds inequality constraints on the aux currents:
  * dependent currents (via the DependentMu optimizable + QuadraticPenalty):
        -EPS <= I_dep <= EPS,  EPS = 1e-3 (mu units; *_CURRENT_SCALE -> A)
  * independent currents: same bound as the modular coil currents
    (|I| <= CURRENT_THRESHOLD), via a CurrentBound-analogue on the mu dof,
    accumulated into J_curr so it shares CURRENT_WEIGHT and its escalation.
"""
import argparse
import os

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from scipy.optimize import minimize

from simsopt._core import Optimizable, load, save
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    ArclengthVariation,
    BoozerSurface,
    CurveCurveDistance,
    CurveLength,
    CurveXYZFourier,
    CurveXYZFourierSymmetries,
    Volume,
    LpCurveCurvature,
    MajorRadius,
    MeanSquaredCurvature,
    NonQuasiSymmetricRatio,
    Iotas,
    RotatedCurve,
    curves_to_vtk,
)
from simsopt.objectives import QuadraticPenalty, Weight

from star_lite_design.utils.boozer_surface_utils import BoozerResidual
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.current_bound import CurrentBound
from star_lite_design.utils.displacement import FieldLineMeanZ
from star_lite_design.utils.magneticwell import MagneticWell
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine
from star_lite_design.utils.pillpipevessel import RennaissanceSDF, PillPipeSDF, TorusSDF, VesselDistance
from star_lite_design.utils.SingularPeriodicFieldline_diff import (
    SingularPeriodicFieldline_diff, DependentMu, _mu_names, _CURRENT_SCALE)


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True,
                    help="design json from a boozer_all.py run "
                         "([boozer_surfaces, iota_Gs, axes, xpoints, sdf])")
parser.add_argument("--num-aux", type=int, required=True,
                    help="number of planar circular auxiliary coils (>= 1)")
parser.add_argument("--constraint", type=str, required=True, choices=['trace', 'identity'],
                    help="monodromy constraint of the singular polish")
parser.add_argument("--margin", type=float)
parser.add_argument("--well", type=str, default='OFF')
parser.add_argument("--Z", type=int)
parser.add_argument("--distance", type=int)
parser.add_argument("--on-vessel", type=int)
parser.add_argument("--config", type=int)
parser.add_argument("--outdir", type=str, default=None,
                    help="output directory (default: ./output_singular_opt/<task name>/)")

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
num_aux = args.num_aux
mon_constraint = args.constraint
if num_aux < 1:
    raise SystemExit(f"--num-aux must be >= 1, got {num_aux}")
# 1 dependent current for the trace system, 3 for identity.
N_DEP_CURRENTS = 1 if mon_constraint == 'trace' else 3
if num_aux < N_DEP_CURRENTS:
    raise SystemExit(f"--num-aux must be >= {N_DEP_CURRENTS} for the "
                     f"{mon_constraint!r} constraint, got {num_aux}")

margin_str = f"{margin_target:.2f}".replace(".", "p")
if args.well == "OFF":
    well_str = "OFF"
else:
    well_str = str(float(args.well))
_in_tag = os.path.basename(os.path.dirname(os.path.abspath(args.input))) or \
    os.path.splitext(os.path.basename(args.input))[0]
TASK_NAME = (f"{_in_tag}_singopt_margin={margin_str}_well={well_str}_Z={Z_weight}"
             f"_onvessel={on_vessel}_distance={distance_weight}_configID={config_id}"
             f"_numaux={num_aux}_constraint={mon_constraint}")
OUT_DIR = (os.path.join(args.outdir, '') if args.outdir is not None
           else f"./output_singular_opt/{TASK_NAME}/")
print(f"Task: {TASK_NAME}")
print(f"Output dir: {OUT_DIR}")

print("Running SINGULAR-POLISH Optimization")
print("================================")

# load the optimized design produced by boozer_all.py. The lists are already
# sliced to the configuration(s) that run optimized; the vessel sdf is the one
# the design was optimized against. The weights/thresholds come from the yaml
# boozer_all.py saved NEXT TO the input json (design_opt_final.yaml), so this
# run continues from that run's final (escalated) weights.
_yaml_path = os.path.splitext(args.input)[0] + '.yaml'
config = yaml.safe_load(open(_yaml_path, 'r'))
data = load(args.input)
boozer_surfaces = data[0]   # BoozerSurfaces
iota_Gs = data[1]           # (iota, G) pairs
axes = data[2]              # magnetic axis field lines
xpoints = data[3]           # X-point field lines
sdf = data[4]               # vessel signed-distance function

for ii, (boozer_surface, axis, xpoint) in enumerate(zip(boozer_surfaces, axes, xpoints)):
    axis.options = {'newton_tol': 1e-15, 'newton_maxiter': 20, 'verbose': True}
    xpoint.options = {'newton_tol': 1e-15, 'newton_maxiter': 20, 'verbose': True}
    boozer_surface.options = {'newton_tol': 1e-16 if boozer_surface.surface.stellsym else 1e-14,
                              'newton_maxiter': 20, 'verbose': True}
    axis.run_code(CurveLength(axis.curve).J())
    xpoint.run_code(CurveLength(xpoint.curve).J())
    boozer_surface.run_code(iota_Gs[ii][0], iota_Gs[ii][1])

# get the base curves
biotsavart = boozer_surfaces[0].biotsavart
coils = biotsavart.coils
curves = [c.curve for c in coils]
# DN (stellsym surface): 2 independent base coils -> [0, 4]. SN: 3 -> [0, 1, 2].
base_curve_idx = [0, 4] if boozer_surfaces[0].surface.stellsym else [0, 1, 2]
base_curves = [curves[i] for i in base_curve_idx]

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

print("""
################################################################################
### Singular polish: SingularPeriodicFieldline_diff per X-point ###############
################################################################################
""")
# Auxiliary-coil parameters mu = (I_1..I_N, r_1..r_N, Z) become local dofs of
# the SingularPeriodicFieldline_diff. FIXED mu dofs are DEPENDENT (solved by
# the Newton polish), FREE mu dofs are INDEPENDENT design variables.
max_Z = np.max([np.abs(c.curve.gamma()[:, 2]) for c in boozer_surfaces[0].biotsavart.coils])
radii0 = np.linspace(0.25, 1.25, num_aux + 1)[1:]
MU_NAMES = _mu_names(2 * num_aux + 1)
DEP_CURRENT_NAMES = [f'I{k+1}' for k in range(N_DEP_CURRENTS)]
INDEP_CURRENT_NAMES = [f'I{k+1}' for k in range(N_DEP_CURRENTS, num_aux)]

sing_fls = []
for idx, (xpoint, boozer_surface) in enumerate(zip(xpoints, boozer_surfaces)):
    mu0 = np.concatenate([np.zeros(num_aux), radii0, [max_Z + 0.15]])
    # The polish gets its OWN copy of the X-point curve: the plain `xpoint`
    # field line keeps re-solving its curve every objective evaluation, so
    # sharing dofs would have the two Newton solvers fight each other.
    xc = xpoint.curve
    fl_curve = CurveXYZFourierSymmetries(xc.quadpoints, xc.order, xc.nfp, xc.stellsym, ntor=xc.ntor)
    fl_curve.set_dofs(xc.get_dofs())
    fl = SingularPeriodicFieldline_diff(
        BiotSavart(boozer_surface.biotsavart.coils), fl_curve, mu0,
        options={'newton_tol': 1e-12, 'newton_maxiter': 10, 'verbose': True,
                 'monodromy_constraint': mon_constraint},
        stellsym_aux=boozer_surface.surface.stellsym)

    # (1) initial polish: ALL aux currents and radii are DEPENDENT (fixed),
    # z is independent (held at max_Z + 0.15). Under-determined -> pinv Newton.
    for k in range(num_aux):
        fl.fix(f'I{k+1}')
        fl.fix(f'r{k+1}')
    res = fl.run_code(CurveLength(fl_curve).J())
    print(f"[idx={idx}] stage-1 polish (all I, r dependent; pinv): success={res['success']}  "
          f"iter={res['iter']}  square={res['square']}  "
          f"||r||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e}")
    if not res['success']:
        raise SystemExit(f"idx={idx}: stage-1 singular polish failed")

    # (2) re-partition: unfix the currents, radii and z, then fix only the
    # dependent currents (1 for trace, 3 for identity). Square -> LU Newton.
    fl.unfix_all()
    for name in DEP_CURRENT_NAMES:
        fl.fix(name)
    fl.need_to_run_code = True
    res = fl.run_code(res['length'])
    print(f"[idx={idx}] stage-2 polish (dependent: {DEP_CURRENT_NAMES}; LU): success={res['success']}  "
          f"iter={res['iter']}  square={res['square']}  "
          f"||r||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e}")
    if not (res['success'] and res['square']):
        raise SystemExit(f"idx={idx}: stage-2 (square) singular polish failed")
    print(f"[idx={idx}] mu: " + '  '.join(
        f"{nm}={v * _CURRENT_SCALE:+.4e}A" if k < num_aux else f"{nm}={v:+.4f}"
        for k, (nm, v) in enumerate(zip(MU_NAMES, fl.mu))))
    sing_fls.append(fl)

# From here on, the POLISHED singular field lines stand in for the X-points
# everywhere downstream (vessel penalties, meanZ, tangent maps, callback,
# snapshots/restores, VTK dumps and saves). Their res['PLU']/res['vjp']
# adjoints propagate sensitivities to the modular coils and the independent
# mu design variables.
xpoints = sing_fls


class MuBound(Optimizable):
    """CurrentBound analogue for one named (independent / free) mu dof of a
    SingularPeriodicFieldline_diff:  J = max(|mu_k| - threshold, 0)^2."""

    def __init__(self, fl, name, threshold):
        Optimizable.__init__(self, depends_on=[fl])
        self.fl = fl
        self.name = name
        self._idx = _mu_names(fl.local_full_dof_size).index(name)
        self.threshold = threshold

    def J(self):
        v = float(self.fl.mu[self._idx])
        return max(abs(v) - self.threshold, 0.) ** 2

    @derivative_dec
    def dJ(self):
        v = float(self.fl.mu[self._idx])
        ex = max(abs(v) - self.threshold, 0.)
        g = np.zeros(self.fl.local_full_dof_size)
        g[self._idx] = 2.0 * ex * np.sign(v)
        return Derivative({self.fl: g})


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

# INDEPENDENT aux currents: bounded the same way as the modular coil currents
# (|I| <= CURRENT_THRESHOLD), accumulated into J_curr so they share
# CURRENT_WEIGHT and its escalation. mu currents are in scaled units, hence
# the threshold CURRENT_THRESHOLD/_CURRENT_SCALE (parallel to the
# CURRENT_THRESHOLD/curr.scale used for the ScaledCurrents above).
for fl in sing_fls:
    for name in INDEP_CURRENT_NAMES:
        J_curr += MuBound(fl, name, CURRENT_THRESHOLD / _CURRENT_SCALE)

# DEPENDENT aux currents: inequality constraint -EPS <= I_dep <= EPS on the
# DependentMu optimizable (mu units: EPS * _CURRENT_SCALE ~ 796 A).
config['DEP_AUX_CURRENT_EPS'] = 1e-6
config['DEP_AUX_CURRENT_WEIGHT'] = 1e4
DEP_AUX_CURRENT_EPS = config['DEP_AUX_CURRENT_EPS']
DEP_AUX_CURRENT_WEIGHT = Weight(config['DEP_AUX_CURRENT_WEIGHT'])
dependent_mus = [DependentMu(fl, name) for fl in sing_fls for name in DEP_CURRENT_NAMES]
J_dep_aux = sum(QuadraticPenalty(dm, +DEP_AUX_CURRENT_EPS, 'max')
                + QuadraticPenalty(dm, -DEP_AUX_CURRENT_EPS, 'min')
                for dm in dependent_mus)

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

# sum the objectives together (the bare-field monodromy/tangent-map penalty is
# gone: the singular polish enforces the monodromy constraint exactly).
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
    + PLASMA_VESSEL_MARGIN_WEIGHT * J_plasma_to_vessel_margin
    + COIL_ON_VESSEL_WEIGHT * J_coil_on_vessel
    + COIL_CLEARANCE_WEIGHT * J_coil_clearance
    + CURRENT_WEIGHT * J_curr
    + WELL_WEIGHT * J_wells
    + FIELDLINE_MEANZ_WEIGHT * J_meanz
    + FIELDLINE_MEANDIST_WEIGHT * J_fieldline_mean_distance
    + DEP_AUX_CURRENT_WEIGHT * J_dep_aux
    )


penalties = {'nonQS': J_nonQSRatio,
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
        'aux dependent current': DEP_AUX_CURRENT_WEIGHT * J_dep_aux,
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

# make sure coils are stellarator symmetric (DN only). For SN designs the base
# coils are deliberately general (the device is no longer stellsym).
if boozer_surfaces[0].surface.stellsym:
    for ii in [base_curve_idx[-1]]:
        c = boozer_surfaces[0].biotsavart.coils[ii].curve
        if isinstance(c, RotatedCurve):
            c = c.curve
        for df in c.local_dof_names:
            if ('xs' in df) or ('yc' in df) or ('zc' in df):
                c.fix(df)

print(JF.dof_names, JF.x.size)


# Directory for output
os.makedirs(OUT_DIR, exist_ok=True)

curves_to_vtk(curves, OUT_DIR + "curves_init")
curves_to_vtk([fl.curve for fl in sing_fls], OUT_DIR + "xpoint_singular_curves_init")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_init_{idx}")

# save these as a backup in case the boozer surface Newton solve fails
res_list = [{'sdofs': boozer_surface.surface.x.copy() , 'iota': boozer_surface.res['iota'], 'G': boozer_surface.res['G']} for boozer_surface in boozer_surfaces]
axes_res_list = [{'adofs': axis.curve.x.copy() , 'length': axis.res['length']} for axis in axes]
xpoints_res_list = [{'xdofs': xpoint.curve.x.copy() , 'length': xpoint.res['length']} for xpoint in xpoints]
# the singular-polish field lines re-solve their own curve copies and the
# dependent mu each evaluation; snapshot/restore them like the axis/x-point.
sing_res_list = [{'xdofs': fl.curve.x.copy(), 'mu': fl.mu.copy(), 'length': fl.res['length']} for fl in sing_fls]
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
    for res, fl in zip(sing_res_list, sing_fls):
        res['xdofs'] = fl.curve.x.copy()
        res['mu'] = fl.mu.copy()
        res['length'] = fl.res['length']

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
    # auxiliary-coil parameters mu, with currents scaled up to Amperes.
    for fi, fl in enumerate(sing_fls):
        mu_str = '  '.join(
            f"{nm}={v * _CURRENT_SCALE:+.4e}A" if k < num_aux else f"{nm}={v:+.4f}"
            for k, (nm, v) in enumerate(zip(MU_NAMES, fl.mu)))
        table2.add_row(f'aux mu [{fi}]', mu_str)
    table2.add_row('aux dependent currents [A]',
                   ' '.join([f'{dm.J() * _CURRENT_SCALE:+.4e}' for dm in dependent_mus]))
    table2.add_row('singular polish monodromy', ' '.join(
        [f'{np.array2string(np.asarray(fl.res["monodromy_matrix"]))}' for fl in sing_fls]))
    table2.add_row('well', ' '.join([f'{w.well().max():.3e}' for w in magnetic_wells]))
    table2.add_row('currents', ' '.join([f'{curr:.3e}' for curr in currents_list]))
    table2.add_row('curvatures', ' '.join([f'{curv:.3e}' for curv in kappas]))
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
        curves_to_vtk([fl.curve for fl in sing_fls], OUT_DIR + "xpoint_singular_curves_tmp")
        curves_to_vtk([axis.curve for axis in axes], OUT_DIR + "ma_tmp")
        for idx, boozer_surface in enumerate(boozer_surfaces):
            boozer_surface.surface.to_vtk(OUT_DIR + f"surf_tmp_{idx}")
        # standard 5-entry layout, with the singular-polish field lines in the
        # x-point slot (xpoints is sing_fls).
        save([boozer_surfaces, iota_Gs, axes, sing_fls, sdf], OUT_DIR + f'design_tmp.json')

    dat_dict["iter"] += 1

def _restore_state():
    """Roll boozer surfaces, axis, xpoint and singular-polish curves, and JF
    dofs back to the last successful callback. Called after a failed fun()
    evaluation so the next call warm-starts from a good point."""
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
    for res, fl in zip(sing_res_list, sing_fls):
        fl.curve.x = res['xdofs']
        fl.mu = res['mu']
        fl.res['length'] = res['length']
        fl.need_to_run_code = True
    # set JF.x LAST: it owns the independent mu (free dofs of the sing_fls), so
    # this re-imposes the good design vector after the mu reset above.
    JF.x = dat_dict['x']


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
    for i, fl in enumerate(sing_fls):
        if not fl.res.get('success', False):
            fail_reasons.append(f'singular polish[{i}] Newton solve did not converge')
        elif not fl.res.get('square', False):
            fail_reasons.append(f'singular polish[{i}] system is not square (bad mu partition)')

    if fail_reasons:
        print('failed — rolling back to last good state: ' + '; '.join(fail_reasons))
        _restore_state()
        # Return the previous J and the negated previous gradient so BFGS's
        # line search shrinks the step back toward the last good point.
        # in case the penalty method is failing, this will always return
        # larger by 1e3
        return 1e3 + dat_dict['J'], -dat_dict['dJ']

    return J, grad

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
    # the INDEPENDENT aux currents are bounded like the modular currents:
    # include them (in Amperes) so curr_err escalates CURRENT_WEIGHT for both.
    indep_aux_amps = [np.abs(float(fl.mu[k])) * _CURRENT_SCALE
                      for fl in sing_fls for k in range(N_DEP_CURRENTS, num_aux)]
    currents_list = currents_list + indep_aux_amps
    curr_err = max([max([c-CURRENT_THRESHOLD, 0.])/CURRENT_THRESHOLD for c in currents_list])
    # DEPENDENT aux currents: violation of -EPS <= I_dep <= EPS (mu units).
    dep_aux_vals = [float(dm.J()) for dm in dependent_mus]
    dep_aux_err = max([max(abs(v) - DEP_AUX_CURRENT_EPS, 0.)/DEP_AUX_CURRENT_EPS for v in dep_aux_vals])
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

    # check which constraints are violated and increase weight if violated by more than 0.1%
    if curr_err > 0.001 and CURRENT_WEIGHT.value != 0.:
        CURRENT_WEIGHT*=10
        print("CURRENT ERROR (modular + independent aux)", curr_err)
    if dep_aux_err > 0.001 and DEP_AUX_CURRENT_WEIGHT.value != 0.:
        DEP_AUX_CURRENT_WEIGHT *= 10
        print("DEPENDENT AUX CURRENT ERROR", dep_aux_err)
    if plasma_vessel_margin_err > 0.001 and PLASMA_VESSEL_MARGIN_WEIGHT.value != 0.:
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

    sdf.to_vtk(OUT_DIR + f'vessel_opt_{j}', nx=40, ny=40, nz=40)
    curves_to_vtk(curves, OUT_DIR + f"curves_opt_{j}")
    curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_opt_{j}")
    curves_to_vtk([fl.curve for fl in sing_fls], OUT_DIR + f"xpoint_singular_curves_opt_{j}")
    curves_to_vtk([axis.curve for axis in axes], OUT_DIR + f"ma_opt_{j}")
    for idx, boozer_surface in enumerate(boozer_surfaces):
        boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_{j}")
    # standard 5-entry layout, with the singular-polish field lines in the
    # x-point slot (xpoints is sing_fls).
    save([boozer_surfaces, iota_Gs, axes, sing_fls, sdf], OUT_DIR + f'design_opt_{j}.json')

    # save the weights in a yaml file
    config['CURRENT_WEIGHT'] = CURRENT_WEIGHT.value
    config['DEP_AUX_CURRENT_WEIGHT'] = DEP_AUX_CURRENT_WEIGHT.value
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

    # Save to YAML
    with open(OUT_DIR + f'design_opt_{j}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

sdf.to_vtk(OUT_DIR + f'vessel_opt_final', nx=40, ny=40, nz=40)
curves_to_vtk(curves, OUT_DIR + f"curves_opt_final")
curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_opt_final")
curves_to_vtk([fl.curve for fl in sing_fls], OUT_DIR + f"xpoint_singular_curves_opt_final")
curves_to_vtk([axis.curve for axis in axes], OUT_DIR + f"ma_opt_final")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_final")
with open(OUT_DIR + f'design_opt_final.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
# design_opt_final.json is written by the FINALIZE step at the end of the
# script (boozer surfaces and axes re-solved on the COMBINED modular+aux coil
# set), so downstream tracing (mk_manifolds.py) sees the polished field.

# ---- final summary of constraints, values, and relative errors ----
final_nonqs_pct = 100. * J_nonQSRatio.J()**0.5

# Recompute all error metrics one more time with final dofs
final_metrics = {
    'nonQS_percent':              (final_nonqs_pct,                                None,                          None),
    'current':                    (max(currents_list),                             CURRENT_THRESHOLD,             curr_err if CURRENT_WEIGHT.value != 0. else 0.0),
    'aux_dependent_current_A':    (max(abs(v) for v in dep_aux_vals) * _CURRENT_SCALE,
                                                                                   DEP_AUX_CURRENT_EPS * _CURRENT_SCALE,
                                                                                   dep_aux_err if DEP_AUX_CURRENT_WEIGHT.value != 0. else 0.0),
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
}

# record the solved auxiliary-coil parameters (currents in Amperes) per fl.
for fi, fl in enumerate(sing_fls):
    for k, nm in enumerate(MU_NAMES):
        v = float(fl.mu[k])
        if k < num_aux:
            v *= _CURRENT_SCALE
        final_metrics[f'aux_{nm}_idx{fi}'] = (v, None, None)

# record the full 2x2 monodromy matrix of the SINGULAR POLISH per X-point so
# the device browser can display it; entries are descriptive (no threshold/error).
for ti, fl in enumerate(sing_fls):
    Mm = np.asarray(fl.res['monodromy_matrix'])
    for a in range(2):
        for b in range(2):
            final_metrics[f'monodromy_M{a}{b}_idx{ti}'] = (float(Mm[a, b]), None, None)

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

print("""
################################################################################
### Finalize: rebuild aux coils + re-solve on the COMBINED coil set ###########
################################################################################
""")
# Mirror boozer_singular.py: build simsopt coils from each fl's solved mu and
# re-solve the Boozer surface and magnetic axis with modular+aux coils, so the
# saved design is self-consistent for downstream tracing -- mk_manifolds.py
# uses boozer_surface.biotsavart, which must contain the aux field.
out_boozer_surfaces = []
out_iota_Gs = []
out_axes = []
for idx, (fl, boozer_surface, ax) in enumerate(zip(sing_fls, boozer_surfaces, axes)):
    mu_sol = fl.mu
    Naux = (len(mu_sol) - 1) // 2
    Zc = float(mu_sol[-1])
    aux_base_curves = []
    aux_base_currents = []
    for k in range(Naux):
        Ik = float(mu_sol[k])
        rk = float(mu_sol[Naux + k])
        c = CurveXYZFourier(np.linspace(0, 1, 160, endpoint=False), 1)
        c.x = c.x * 0.
        c.set('zc(0)', Zc)
        c.set('xc(1)', rk)
        c.set('ys(1)', rk)
        aux_base_curves.append(c)
        aux_base_currents.append(ScaledCurrent(Current(Ik), _CURRENT_SCALE))
    if fl.stellsym_aux:
        aux_coils = coils_via_symmetries(aux_base_curves, aux_base_currents, 1, True)
    else:
        aux_coils = [Coil(c, I) for c, I in zip(aux_base_curves, aux_base_currents)]
    combined_coils = boozer_surface.biotsavart.coils + aux_coils

    bs_out = BoozerSurface(BiotSavart(combined_coils), boozer_surface.surface,
                           Volume(boozer_surface.surface), boozer_surface.surface.volume())
    bs_res = bs_out.run_code(boozer_surface.res['iota'], boozer_surface.res['G'])
    if not bs_res['success'] or bs_out.surface.is_self_intersecting():
        print(f"ERROR: idx={idx}: BoozerSurface re-solve on the combined coil set failed")
        print("ABORT: design_opt_final.json will not be written.")
        raise SystemExit(1)

    new_ax = PeriodicFieldLine(BiotSavart(combined_coils), ax.curve)
    ax_res = new_ax.run_code(CurveLength(ax.curve).J())
    if not ax_res['success']:
        print(f"ERROR: idx={idx}: magnetic-axis re-solve on the combined coil set failed")
        print("ABORT: design_opt_final.json will not be written.")
        raise SystemExit(1)

    out_boozer_surfaces.append(bs_out)
    out_iota_Gs.append([bs_res['iota'], bs_res['G']])
    out_axes.append(new_ax)
    curves_to_vtk([coil.curve for coil in aux_coils], OUT_DIR + f'aux_coils_{idx}')
    bs_out.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_final")   # refresh with the re-solved surface

# standard 5-entry layout consumed by mk_manifolds.py: the boozer surfaces and
# axes carry the COMBINED (modular + aux) coils, and the singular-polish field
# lines sit in the x-point slot.
save([out_boozer_surfaces, out_iota_Gs, out_axes, sing_fls, sdf], OUT_DIR + f'design_opt_final.json')
print(f"wrote {OUT_DIR}design_opt_final.json (combined modular+aux coil set)")
