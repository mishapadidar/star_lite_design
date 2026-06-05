#!/usr/bin/env python3
"""boozer_singular_opt.py

Continue a boozer_all.py optimization WITH the singular polish in the loop.

Takes as input a design json produced by boozer_all.py
([boozer_surfaces, iota_Gs, axes, xpoints, sdf]).  For each X-point a
SingularPeriodicFieldline is created with --num-aux planar circular
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

It also keeps the aux coils clear of the modular coils: AuxCoilDistance adds a
one-sided quadratic clearance penalty (minimum separation
AUX_COIL_DISTANCE_THRESHOLD, weight AUX_COIL_DISTANCE_WEIGHT escalated after each
BFGS run) between every modular coil and the planar circular aux coils. The aux
coils are mu dofs (not Curves), so the penalty reads the aux radii/Z from fl.mu
and its gradient lands on both the modular-coil dofs and the aux (mu) dofs.
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
from star_lite_design.utils.singularperiodicfieldline import (
    SingularPeriodicFieldline, DependentMu, AuxCoilDistance, _mu_names, _CURRENT_SCALE)
from star_lite_design.utils.singularbiotsavart import SingularBiotSavart


# The boozer_all design json is the only required input; everything else
# (weights, thresholds, monodromy constraint, config id, well state, ...) is
# read from the sibling yaml of the same basename in the same directory. The
# number of auxiliary coils is given on the command line (prefix.sh passes it),
# defaulting to NUM_AUX_DEFAULT. Outputs are written in place, next to the input.
NUM_AUX_DEFAULT = 10

parser = argparse.ArgumentParser(
    description="Singular polish + coil optimization, continuing a boozer_all "
                "design. Reads all run parameters from the design json's sibling "
                ".yaml; writes its outputs next to the input.")
parser.add_argument("design_json",
                    help="path to a boozer_all design_opt_final.json (the matching "
                         ".yaml in the same directory supplies all parameters)")
parser.add_argument("--num-aux", type=int, default=NUM_AUX_DEFAULT,
                    help=f"number of planar circular auxiliary coils (default {NUM_AUX_DEFAULT})")
args = parser.parse_args()

_in = os.path.abspath(args.design_json)
_yaml_path = os.path.splitext(_in)[0] + '.yaml'
config = yaml.safe_load(open(_yaml_path, 'r'))
data = load(_in)

# run parameters: num_aux from the command line, everything else from the yaml
config_id = int(config['CONFIG_ID'])
mon_constraint = config['MONODROMY_CONSTRAINT']
if mon_constraint not in ('trace', 'identity'):
    raise SystemExit(f"config MONODROMY_CONSTRAINT must be 'trace' or 'identity', "
                     f"got {mon_constraint!r}")
num_aux = int(args.num_aux)
config['NUM_AUX'] = num_aux   # record it in the (output) yaml
# 1 dependent current for the trace system, 3 for identity.
N_DEP_CURRENTS = 1 if mon_constraint == 'trace' else 3
if num_aux < N_DEP_CURRENTS:
    raise SystemExit(f"NUM_AUX must be >= {N_DEP_CURRENTS} for the "
                     f"{mon_constraint!r} constraint, got {num_aux}")

# write outputs in place (next to the input design json)
OUT_DIR = os.path.join(os.path.dirname(_in), '')
TASK_NAME = os.path.basename(os.path.dirname(_in)) or os.path.basename(_in)
print(f"Task: {TASK_NAME}")
print(f"Input: {_in}")
print(f"Output dir: {OUT_DIR}")
print(f"constraint={mon_constraint}  num_aux={num_aux}  config_id={config_id}")

print("Running SINGULAR-POLISH Optimization")
print("================================")
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

# ALL thresholds AND weights are read STRAIGHT from the config yaml (boozer_all
# wrote them); the polish does not set its own values. This includes the well
# state: WELL_ACTIVE False / WELL_WEIGHT == 0 means the well penalty was OFF in
# the boozer_all run and is loaded as off here.
COIL_ON_VESSEL_THRESHOLD = config['COIL_ON_VESSEL_THRESHOLD']
COIL_CLEARANCE_THRESHOLD = config['COIL_CLEARANCE_THRESHOLD']
COIL_ON_VESSEL_WEIGHT = Weight(config['COIL_ON_VESSEL_WEIGHT'])
COIL_CLEARANCE_WEIGHT = Weight(config['COIL_CLEARANCE_WEIGHT'])

PLASMA_VESSEL_MARGIN_THRESHOLD = config['PLASMA_VESSEL_MARGIN_THRESHOLD']

CURRENT_THRESHOLD = config['CURRENT_THRESHOLD']
CURRENT_WEIGHT = Weight(config['CURRENT_WEIGHT'])

well_active = bool(config['WELL_ACTIVE'])
WELL_THRESHOLD = float(config['WELL_THRESHOLD'])
WELL_WEIGHT = Weight(config['WELL_WEIGHT'])
print(f"WELL loaded from config: active={well_active}  threshold={WELL_THRESHOLD}  weight={WELL_WEIGHT.value}")

FIELDLINE_MEANZ_WEIGHT = Weight(config['FIELDLINE_MEANZ_WEIGHT'])
FIELDLINE_MEANZ_THRESHOLD = config['FIELDLINE_MEANZ_THRESHOLD']

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
MAJOR_RADIUS_WEIGHT=Weight(config['MAJOR_RADIUS_WEIGHT'])

BOOZER_RESIDUAL_WEIGHT=Weight(config['BOOZER_RESIDUAL_WEIGHT'])


PLASMA_VESSEL_MARGIN_WEIGHT = Weight(config['PLASMA_VESSEL_MARGIN_WEIGHT'])
MODB_WEIGHT = Weight(config['MODB_WEIGHT'])
ARCLENGTH_WEIGHT = Weight(config['ARCLENGTH_WEIGHT'])

print("""
################################################################################
### Singular polish: SingularPeriodicFieldline per X-point ###############
################################################################################
""")
# Auxiliary-coil parameters mu = (I_1..I_N, r_1..r_N, Z) become local dofs of
# the SingularPeriodicFieldline. FIXED mu dofs are DEPENDENT (solved by
# the Newton polish), FREE mu dofs are INDEPENDENT design variables.
max_Z = np.max([np.abs(c.curve.gamma()[:, 2]) for c in boozer_surfaces[0].biotsavart.coils])
# Aux-coil clearance threshold, read now because it also sets the INITIAL aux-coil
# height Z = max_Z + AUX_COIL_DISTANCE_THRESHOLD (one clearance above the modular
# coils). NEW key (boozer_all did not write it); an existing yaml may override it.
config.setdefault('AUX_COIL_DISTANCE_THRESHOLD', 0.15)
AUX_COIL_DISTANCE_THRESHOLD = config['AUX_COIL_DISTANCE_THRESHOLD']
radii0 = np.linspace(0.25, 1.25, num_aux + 1)[1:]
MU_NAMES = _mu_names(2 * num_aux + 1)
DEP_CURRENT_NAMES = [f'I{k+1}' for k in range(N_DEP_CURRENTS)]
INDEP_CURRENT_NAMES = [f'I{k+1}' for k in range(N_DEP_CURRENTS, num_aux)]

sing_fls = []
for idx, (xpoint, boozer_surface) in enumerate(zip(xpoints, boozer_surfaces)):
    mu0 = np.concatenate([np.zeros(num_aux), radii0, [max_Z + AUX_COIL_DISTANCE_THRESHOLD]])
    # The polish gets its OWN copy of the X-point curve: the plain `xpoint`
    # field line keeps re-solving its curve every objective evaluation, so
    # sharing dofs would have the two Newton solvers fight each other.
    xc = xpoint.curve
    fl_curve = CurveXYZFourierSymmetries(xc.quadpoints, xc.order, xc.nfp, xc.stellsym, ntor=xc.ntor)
    fl_curve.set_dofs(xc.get_dofs())
    fl = SingularPeriodicFieldline(
        BiotSavart(boozer_surface.biotsavart.coils), fl_curve, mu0,
        options={'newton_tol': 1e-12, 'newton_maxiter': 10, 'verbose': True,
                 'monodromy_constraint': mon_constraint},
        stellsym_aux=boozer_surface.surface.stellsym)

    # (1) initial polish: ALL aux currents and radii are DEPENDENT (fixed),
    # z is independent (held at max_Z + AUX_COIL_DISTANCE_THRESHOLD). Under-determined -> pinv Newton.
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

# Recompute the Boozer surfaces and magnetic axes in the TOTAL field (modular +
# auxiliary): wrap each polished singular field line in a SingularBiotSavart and
# REPLACE the loaded (modular-field) boozer_surfaces / axes with ones solved on
# the total field. Every field-dependent objective below is then built against the
# total field too. Each object that drives its own set_points gets its OWN
# SingularBiotSavart(fl) wrapper (all sharing fl), mirroring the original
# per-objective BiotSavart(coils) so their point caches stay independent.
new_boozer_surfaces, new_axes = [], []
for boozer_surface, axis, fl in zip(boozer_surfaces, axes, sing_fls):
    nbs = BoozerSurface(SingularBiotSavart(fl), boozer_surface.surface,
                        boozer_surface.label, boozer_surface.targetlabel,
                        constraint_weight=boozer_surface.constraint_weight,
                        options=boozer_surface.options)
    nbs.run_code(boozer_surface.res['iota'], boozer_surface.res['G'])
    new_boozer_surfaces.append(nbs)
    nax = PeriodicFieldLine(SingularBiotSavart(fl), axis.curve, options=axis.options)
    nax.run_code(axis.res['length'])
    new_axes.append(nax)
boozer_surfaces, axes = new_boozer_surfaces, new_axes


class MuBound(Optimizable):
    """CurrentBound analogue for one named (independent / free) mu dof of a
    SingularPeriodicFieldline:  J = max(|mu_k| - threshold, 0)^2."""

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
brs = [BoozerResidual(boozer_surface, SingularBiotSavart(fl)) for boozer_surface, fl in zip(boozer_surfaces, sing_fls)]
J_major_radius = QuadraticPenalty(mr, MR_TARGET, 'identity')  # target major radius is that computed on the initial surface

IOTAS_LIST = [Iotas(boozer_surface) for boozer_surface in boozer_surfaces]
J_iotas = sum([QuadraticPenalty(IOTAS, IOT_TARGET, 'identity') for IOTAS, IOT_TARGET in zip(IOTAS_LIST, IOTAS_TARGET)]) # target rotational transform is that computed on the initial surface
nonQS_list = [NonQuasiSymmetricRatio(boozer_surface, SingularBiotSavart(fl)) for boozer_surface, fl in zip(boozer_surfaces, sing_fls)]
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
modBs = [ModBOnFieldLine(axis, SingularBiotSavart(fl)) for axis, fl in zip(axes, sing_fls)]
JmodB = sum([QuadraticPenalty(modB, MODB_TARGET, 'identity') for modB, axis, boozer_surface in zip(modBs, axes, boozer_surfaces)])

J_curr = None
for boozer_surface in boozer_surfaces:
    for idx in base_curve_idx:
        curr = boozer_surface.biotsavart.coils[idx].current
        if J_curr is None:
            J_curr = CurrentBound(curr.current_to_scale, CURRENT_THRESHOLD/curr.scale)
        else:
            J_curr += CurrentBound(curr.current_to_scale, CURRENT_THRESHOLD/curr.scale)


# AUX coil currents: BOTH dependent and independent aux currents are constrained
# to -AUX_CURRENT_EPS <= I <= AUX_CURRENT_EPS, sharing ONE penalty weight
# (AUX_CURRENT_WEIGHT) in a single objective J_aux_current. Dependent currents go
# through DependentMu (their value is an implicit function of the design vars, so
# the gradient must flow through the adjoint); independent currents are free mu
# dofs handled directly by MuBound. (mu units: EPS/_CURRENT_SCALE; EPS is in A.)
config['AUX_CURRENT_EPS'] = 1e3  # Amperes
config['AUX_CURRENT_WEIGHT'] = 1e2
AUX_CURRENT_EPS = config['AUX_CURRENT_EPS']
AUX_CURRENT_WEIGHT = Weight(config['AUX_CURRENT_WEIGHT'])
_aux_eps_mu = AUX_CURRENT_EPS / _CURRENT_SCALE   # bound in mu units
dependent_mus = [DependentMu(fl, name) for fl in sing_fls for name in DEP_CURRENT_NAMES]
J_aux_current = sum(QuadraticPenalty(dm, +_aux_eps_mu, 'max')
                    + QuadraticPenalty(dm, -_aux_eps_mu, 'min')
                    for dm in dependent_mus)
for fl in sing_fls:
    for name in INDEP_CURRENT_NAMES:
        J_aux_current += MuBound(fl, name, _aux_eps_mu)


# AUX coil <-> MODULAR coil clearance: keep every modular coil at least
# AUX_COIL_DISTANCE_THRESHOLD away from the planar circular aux coils, sharing one
# escalating weight (AUX_COIL_DISTANCE_WEIGHT) via J_aux_coil_distance. The aux
# coils are mu dofs (not Curves), so AuxCoilDistance reads their radii/Z from
# fl.mu directly (closed-form point-to-circle distance) and returns gradients to
# BOTH the modular-coil dofs and the aux (mu) dofs. AUX_COIL_DISTANCE_THRESHOLD is
# set above (it also fixes the initial aux-coil Z); here we just default/read the
# starting weight, letting an existing yaml override so escalation survives a restart.
config.setdefault('AUX_COIL_DISTANCE_WEIGHT', 1e5)
AUX_COIL_DISTANCE_WEIGHT = Weight(config['AUX_COIL_DISTANCE_WEIGHT'])
aux_coil_distances = [AuxCoilDistance(fl, curves, AUX_COIL_DISTANCE_THRESHOLD) for fl in sing_fls]
J_aux_coil_distance = sum(aux_coil_distances)





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
    + AUX_CURRENT_WEIGHT * J_aux_current
    + AUX_COIL_DISTANCE_WEIGHT * J_aux_coil_distance
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
        'aux current': AUX_CURRENT_WEIGHT * J_aux_current,
        'aux coil distance': AUX_COIL_DISTANCE_WEIGHT * J_aux_coil_distance,
        }

# Weight object backing each penalty above, keyed identically. 'nonQS' has an
# implicit weight of 1 (no Weight object) and is the primary objective, so it is
# omitted and never pre-scaled.
penalty_weights = {
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
        'aux current': AUX_CURRENT_WEIGHT,
        'aux coil distance': AUX_COIL_DISTANCE_WEIGHT,
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
    # auxiliary-coil parameters, per fl: summarize the currents (scaled to
    # Amperes) by min/max, separated into DEPENDENT (solved, bounded to
    # +-AUX_CURRENT_EPS) and INDEPENDENT (free design vars); list the radii + z.
    for fi, fl in enumerate(sing_fls):
        mu = np.asarray(fl.mu)
        dep_I = np.abs(mu[:N_DEP_CURRENTS]) * _CURRENT_SCALE
        indep_I = np.abs(mu[N_DEP_CURRENTS:num_aux]) * _CURRENT_SCALE
        radii = mu[num_aux:2 * num_aux]
        z = float(mu[-1])
        table2.add_row(f'aux dep |I| [{fi}] [A]',
                       f'min={dep_I.min():.4e}  max={dep_I.max():.4e}')
        table2.add_row(f'aux indep |I| [{fi}] [A]',
                       (f'min={indep_I.min():.4e}  max={indep_I.max():.4e}'
                        if indep_I.size else '(none)'))
        table2.add_row(f'aux r,z [{fi}]',
                       '  '.join([f'r{k+1}={radii[k]:+.4f}' for k in range(num_aux)] + [f'z={z:+.4f}']))
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
    table2.add_row('minimum aux-to-modular coil distance',
                   ' '.join([f'{a.shortest_distance():.3e}' for a in aux_coil_distances]))
    table2.add_row('minimum aux-to-aux coil distance',
                   ' '.join([f'{a.shortest_aux_distance():.3e}' for a in aux_coil_distances]))

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
        save([boozer_surfaces, iota_Gs, axes, sing_fls, sdf], OUT_DIR + f'design_polished_tmp.json')

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
    for i, fl in enumerate(sing_fls):
        if not fl.res.get('success', False):
            fail_reasons.append(f'singular polish[{i}] Newton solve did not converge')
        elif not fl.res.get('square', False):
            fail_reasons.append(f'singular polish[{i}] system is not square (bad mu partition)')

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

# The weights were loaded from the yaml for the (modular-field) boozer_all run and
# may be too large now that every penalty is evaluated in the TOTAL (modular + aux)
# field. Before optimizing, shrink any weight whose CURRENT contribution
# weight*penalty.J() exceeds 1 -- by repeated factors of 10 -- so each weighted
# term starts below 1. The in-loop x10 escalation can raise it back as the
# constraints demand. Modifying each Weight in place is seen by JF (same objects).
print("Pre-scaling penalty weights so each weighted term starts below 1 in the total field:")
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

    # MODULAR coil currents (A): bounded by CURRENT_THRESHOLD; weight CURRENT_WEIGHT.
    modular_currents = [np.abs(boozer_surface.biotsavart.coils[idx].current.get_value())
                        for boozer_surface in boozer_surfaces for idx in base_curve_idx]
    curr_err = max([max(c - CURRENT_THRESHOLD, 0.) / CURRENT_THRESHOLD for c in modular_currents])
    # AUX currents (A): BOTH dependent and independent are bounded by AUX_CURRENT_EPS
    # and share ONE weight (AUX_CURRENT_WEIGHT) via J_aux_current. dm.J() is the mu
    # value -> *_CURRENT_SCALE for A; independent currents are read straight from fl.mu.
    dep_aux_amps = [np.abs(float(dm.J())) * _CURRENT_SCALE for dm in dependent_mus]
    indep_aux_amps = [np.abs(float(fl.mu[k])) * _CURRENT_SCALE
                      for fl in sing_fls for k in range(N_DEP_CURRENTS, num_aux)]
    aux_amps = dep_aux_amps + indep_aux_amps
    aux_err = (max([max(c - AUX_CURRENT_EPS, 0.) / AUX_CURRENT_EPS for c in aux_amps])
               if aux_amps else 0.)
    iota_err = max([np.abs(IOTAS.J() - IOT_TARGET)/np.abs(IOT_TARGET) for IOTAS, IOT_TARGET in zip(IOTAS_LIST, IOTAS_TARGET)])
    mr_err = np.abs(mr.J()-MR_TARGET)/MR_TARGET
    clen_err = max([max(Jl.J() - LENGTH_THRESHOLD, 0)/np.abs(LENGTH_THRESHOLD) for Jl in Jls])

    cc_err = max(COIL_TO_COIL_THRESHOLD-Jccdist.shortest_distance(), 0)/np.abs(COIL_TO_COIL_THRESHOLD)
    aux_coil_mod_err = max([max(AUX_COIL_DISTANCE_THRESHOLD - a.shortest_distance(), 0)
                            / np.abs(AUX_COIL_DISTANCE_THRESHOLD) for a in aux_coil_distances])
    aux_coil_aux_err = max([max(AUX_COIL_DISTANCE_THRESHOLD - a.shortest_aux_distance(), 0)
                            / np.abs(AUX_COIL_DISTANCE_THRESHOLD) for a in aux_coil_distances])
    # one shared weight drives BOTH the modular<->aux and aux<->aux clearance
    aux_coil_dist_err = max(aux_coil_mod_err, aux_coil_aux_err)
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
        print("MODULAR CURRENT ERROR", curr_err)
    if aux_err > 0.001 and AUX_CURRENT_WEIGHT.value != 0.:
        AUX_CURRENT_WEIGHT *= 10
        print("AUX CURRENT ERROR (dependent + independent)", aux_err)
    if plasma_vessel_margin_err > 0.001 and PLASMA_VESSEL_MARGIN_WEIGHT.value != 0.:
        PLASMA_VESSEL_MARGIN_WEIGHT*=10
        print("PLASMA-VESSEL MARGIN ERROR", plasma_vessel_margin_err)
    # on-vessel vs clearance is encoded by which weight is nonzero (the other is
    # 0 in the config), so the weight check alone selects the active constraint.
    if coil_on_vessel_err > 0.001 and COIL_ON_VESSEL_WEIGHT.value != 0.:
        COIL_ON_VESSEL_WEIGHT *= 10
        print("COIL-ON-VESSEL ERROR", coil_on_vessel_err)
    if coil_clearance_err > 0.001 and COIL_CLEARANCE_WEIGHT.value != 0.:
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
    if aux_coil_dist_err > 0.001 and AUX_COIL_DISTANCE_WEIGHT.value != 0.:
        AUX_COIL_DISTANCE_WEIGHT*=10
        print("AUX COIL DISTANCE ERROR", aux_coil_dist_err)
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
    save([boozer_surfaces, iota_Gs, axes, sing_fls, sdf], OUT_DIR + f'design_polished_{j}.json')

    # save the weights in a yaml file
    config['CURRENT_WEIGHT'] = CURRENT_WEIGHT.value
    config['AUX_CURRENT_WEIGHT'] = AUX_CURRENT_WEIGHT.value
    config['COIL_TO_COIL_WEIGHT'] = COIL_TO_COIL_WEIGHT.value
    config['AUX_COIL_DISTANCE_WEIGHT'] = AUX_COIL_DISTANCE_WEIGHT.value
    config['AUX_COIL_DISTANCE_THRESHOLD'] = AUX_COIL_DISTANCE_THRESHOLD
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
    with open(OUT_DIR + f'design_polished_{j}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

sdf.to_vtk(OUT_DIR + f'vessel_opt_final', nx=40, ny=40, nz=40)
curves_to_vtk(curves, OUT_DIR + f"curves_opt_final")
curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_opt_final")
curves_to_vtk([fl.curve for fl in sing_fls], OUT_DIR + f"xpoint_singular_curves_opt_final")
curves_to_vtk([axis.curve for axis in axes], OUT_DIR + f"ma_opt_final")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_final")
with open(OUT_DIR + f'design_polished_final.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
# design_polished_final.json is written by the FINALIZE step at the end of the
# script (boozer surfaces and axes re-solved on the COMBINED modular+aux coil
# set), so downstream tracing (mk_manifolds.py) sees the polished field.

# ---- final summary of constraints, values, and relative errors ----
final_nonqs_pct = 100. * J_nonQSRatio.J()**0.5

# Recompute all error metrics one more time with final dofs
final_metrics = {
    'nonQS_percent':              (final_nonqs_pct,                                None,                          None),
    'current':                    (max(modular_currents),                          CURRENT_THRESHOLD,             curr_err if CURRENT_WEIGHT.value != 0. else 0.0),
    'aux_current_A':              (max(aux_amps) if aux_amps else 0.0,             AUX_CURRENT_EPS,               aux_err if AUX_CURRENT_WEIGHT.value != 0. else 0.0),
    'iotas':                      (max(IOTAS.J() for IOTAS in IOTAS_LIST),         max(IOTAS_TARGET),             iota_err if IOTAS_WEIGHT.value != 0. else 0.0),
    'major_radius':               (mr.J(),                                         MR_TARGET,                     mr_err if MAJOR_RADIUS_WEIGHT.value != 0. else 0.0),
    'coil_length':                (max(Jl.J() for Jl in Jls),                      LENGTH_THRESHOLD,              clen_err if LENGTH_WEIGHT.value != 0. else 0.0),
    'coil_to_coil':               (Jccdist.shortest_distance(),                    COIL_TO_COIL_THRESHOLD,        cc_err if COIL_TO_COIL_WEIGHT.value != 0. else 0.0),
    'aux_coil_distance':          (min(a.shortest_distance() for a in aux_coil_distances), AUX_COIL_DISTANCE_THRESHOLD, aux_coil_mod_err if AUX_COIL_DISTANCE_WEIGHT.value != 0. else 0.0),
    'aux_coil_to_aux_distance':   (min(a.shortest_aux_distance() for a in aux_coil_distances), AUX_COIL_DISTANCE_THRESHOLD, aux_coil_aux_err if AUX_COIL_DISTANCE_WEIGHT.value != 0. else 0.0),
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

    # Save the polished design on an EXPLICIT combined modular+aux BiotSavart
    # (real CurveXYZFourier aux coils), NOT SingularBiotSavart: the total field is
    # identical (~1e-17) but pure-C++, so downstream tracing (mk_manifolds.py's
    # compute_fieldlines) runs at native speed instead of calling back into
    # Python/jax per integration step. The optimization above still uses
    # SingularBiotSavart; only the saved/finalized field is the combined coil set.
    field = BiotSavart(combined_coils)
    bs_out = BoozerSurface(field, boozer_surface.surface,
                           Volume(boozer_surface.surface), boozer_surface.surface.volume())
    bs_res = bs_out.run_code(boozer_surface.res['iota'], boozer_surface.res['G'])
    if not bs_res['success'] or bs_out.surface.is_self_intersecting():
        print(f"ERROR: idx={idx}: BoozerSurface re-solve on the combined coil set failed")
        print("ABORT: design_polished_final.json will not be written.")
        raise SystemExit(1)

    new_ax = PeriodicFieldLine(field, ax.curve)
    ax_res = new_ax.run_code(CurveLength(ax.curve).J())
    if not ax_res['success']:
        print(f"ERROR: idx={idx}: magnetic-axis re-solve on the combined coil set failed")
        print("ABORT: design_polished_final.json will not be written.")
        raise SystemExit(1)

    out_boozer_surfaces.append(bs_out)
    out_iota_Gs.append([bs_res['iota'], bs_res['G']])
    out_axes.append(new_ax)
    curves_to_vtk([coil.curve for coil in aux_coils], OUT_DIR + f'aux_coils_{idx}')
    bs_out.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_final")   # refresh with the re-solved surface

# standard 5-entry layout consumed by mk_manifolds.py: the boozer surfaces and
# axes carry the COMBINED (modular + aux) coils, and the singular-polish field
# lines sit in the x-point slot.
save([out_boozer_surfaces, out_iota_Gs, out_axes, sing_fls, sdf], OUT_DIR + f'design_polished_final.json')
print(f"wrote {OUT_DIR}design_polished_final.json (combined modular+aux coil set)")
