#!/usr/bin/env python3
"""boozer_singular.py

Compute the auxiliary coils for a boozer_all.py design and STOP -- no coil
optimization. This is the "unpolished, with-aux" sibling of
boozer_singular_opt.py: it runs the identical input load + SingularPeriodicFieldline
construction + two-stage Newton polish that FIRST computes the auxiliary-coil
currents/radii/Z (with the trace/identity monodromy enforced EXACTLY), writes the
resulting device json, and exits. The escalating-penalty BFGS optimization that
boozer_singular_opt.py runs afterwards is skipped entirely.

Takes as input a design json produced by boozer_all.py
([boozer_surfaces, iota_Gs, axes, xpoints, sdf]).  For each X-point a
SingularPeriodicFieldline is created with --num-aux planar circular auxiliary
coils:

  1) initially ALL aux currents and radii are DEPENDENT (fixed dofs, solved by
     the under-determined pinv Newton polish; z is independent/held);
  2) then the partition is re-drawn: everything is unfixed and only the
     dependent currents are fixed -- 1 current for the 'trace' system, 3 for
     'identity' -- giving a square LU Newton polish that pins the monodromy.

That square polish is where the auxiliary currents are FIRST computed. We then
re-solve the Boozer surfaces and magnetic axes in the TOTAL (modular + aux)
field, rebuild the aux coils as explicit CurveXYZFourier coils on a combined
modular+aux BiotSavart (so downstream tracing runs at native speed, exactly like
boozer_singular_opt.py's finalize step), save the device, and exit.

Outputs are written to a NEW directory next to the input whose name is the input
device folder with '_unpolished' appended, so the unpolished device sits beside
(not on top of) the polished one. The saved files mirror the polished pipeline's
names with the 'unpolished' tag: design_unpolished_final_<DEVICE_ID>.json (+ .yaml),
the *_opt_final VTK files consumed by mk_paraview.py, and a descriptive summary.txt.
"""
import argparse
import os
import re
import zlib

import numpy as np
import yaml

from simsopt._core import load, save
from simsopt.field import BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    BoozerSurface,
    CurveLength,
    CurveXYZFourier,
    CurveXYZFourierSymmetries,
    Volume,
    NonQuasiSymmetricRatio,
    curves_to_vtk,
)

from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.singularperiodicfieldline import (
    SingularPeriodicFieldline, _mu_names, _CURRENT_SCALE)
from star_lite_design.utils.singularbiotsavart import SingularBiotSavart


# The boozer_all design json is the only required input; the monodromy constraint
# and config id are read from the sibling yaml of the same basename in the same
# directory (the weights/thresholds the polish optimization would read are NOT
# needed here, since no optimization happens). The number of auxiliary coils is
# given on the command line, defaulting to NUM_AUX_DEFAULT.
NUM_AUX_DEFAULT = 10

parser = argparse.ArgumentParser(
    description="Compute auxiliary coils for a boozer_all design (trace/identity "
                "monodromy enforced exactly) and save the device WITHOUT any coil "
                "optimization. Reads the monodromy constraint from the design json's "
                "sibling .yaml; writes outputs to a sibling '<folder>_unpolished' dir.")
parser.add_argument("design_json",
                    help="path to a boozer_all design_opt_final.json (the matching "
                         ".yaml in the same directory supplies the monodromy constraint)")
parser.add_argument("--num-aux", type=int, default=NUM_AUX_DEFAULT,
                    help=f"number of planar circular auxiliary coils (default {NUM_AUX_DEFAULT})")
args = parser.parse_args()

_in = os.path.abspath(args.design_json)
_yaml_path = os.path.splitext(_in)[0] + '.yaml'
config = yaml.safe_load(open(_yaml_path, 'r'))
data = load(_in)

# run parameters: num_aux from the command line, the monodromy constraint from the yaml
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

# Outputs go to a NEW directory, a sibling of the input: the input device folder name
# with its num_aux token set to the ACTUAL number of auxiliary coils we just added (the
# input is a num_aux=0 boozer_all device, but this device carries `num_aux` of them, so
# leaving num_aux=0 in the name would be wrong and would mislead device_browser), then
# '_unpolished' appended. The device ID is the crc32 of that (unpolished) folder name --
# matching boozer_all.py / device_browser.py's folder-name -> ID convention -- and is
# embedded in the output json/yaml names.
_in_dir = os.path.dirname(_in)
_src_name = os.path.basename(_in_dir) or os.path.basename(_in)
TASK_NAME = re.sub(r'num_aux=\d+', f'num_aux={num_aux}', _src_name) + '_unpolished'
OUT_DIR = os.path.join(os.path.dirname(_in_dir), TASK_NAME, '')
DEVICE_ID = zlib.crc32(TASK_NAME.encode())
print(f"Task: {TASK_NAME}")
print(f"Input: {_in}")
print(f"Output dir: {OUT_DIR}")
print(f"constraint={mon_constraint}  num_aux={num_aux}")

print("Running SINGULAR aux-coil computation (NO optimization)")
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

print("""
################################################################################
### Singular polish: SingularPeriodicFieldline per X-point ###############
################################################################################
""")
# Auxiliary-coil parameters mu = (I_1..I_N, r_1..r_N, Z) become local dofs of
# the SingularPeriodicFieldline. FIXED mu dofs are DEPENDENT (solved by
# the Newton polish), FREE mu dofs are INDEPENDENT (held here, since we never
# optimize them).
max_Z = np.max([np.abs(c.curve.gamma()[:, 2]) for c in boozer_surfaces[0].biotsavart.coils])
# Aux-coil clearance threshold, read now because it also sets the INITIAL aux-coil
# height Z = max_Z + AUX_COIL_DISTANCE_THRESHOLD (one clearance above the modular
# coils). NEW key (boozer_all did not write it); an existing yaml may override it.
config.setdefault('AUX_COIL_DISTANCE_THRESHOLD', 0.15)
AUX_COIL_DISTANCE_THRESHOLD = config['AUX_COIL_DISTANCE_THRESHOLD']
radii0 = np.linspace(0.5, 1.5, num_aux)
MU_NAMES = _mu_names(2 * num_aux + 1)
DEP_CURRENT_NAMES = [f'I{k+1}' for k in range(N_DEP_CURRENTS)]

# A loaded X-point is always parametrized stellsym=False, but it may in fact be
# stellarator symmetric, i.e. gamma(t) = diag(1, -1, -1) gamma(-t) for all t. If
# so we project it onto a stellsym=True curve below so the singular field line is
# exactly stellarator symmetric. STELLSYM_TOL is the defect threshold (relative
# to the curve scale); the separation between a truly symmetric X-point (defect
# ~ machine eps) and a single-null one (defect ~ O(0.1 m)) is enormous, so the
# exact value is not delicate.
STELLSYM_TOL = 1e-9


def _stellsym_defect(curve):
    """max_t | gamma(t) - diag(1, -1, -1) gamma(-t) |, which is ~0 iff the curve
    is stellarator symmetric. gamma is 1-periodic in the parametrization, so
    gamma(-t) is evaluated on a copy of the curve at the reflected quadpoints."""
    qp = np.asarray(curve.quadpoints)
    refl = CurveXYZFourierSymmetries(np.mod(-qp, 1.0), curve.order, curve.nfp,
                                     curve.stellsym, ntor=curve.ntor)
    refl.set_dofs(curve.get_dofs())
    flip = np.array([1.0, -1.0, -1.0])
    return float(np.max(np.abs(curve.gamma() - flip * refl.gamma())))


sing_fls = []
for idx, (xpoint, boozer_surface) in enumerate(zip(xpoints, boozer_surfaces)):
    mu0 = np.concatenate([np.zeros(num_aux), radii0, [max_Z + AUX_COIL_DISTANCE_THRESHOLD]])
    # The polish gets its OWN copy of the X-point curve: the plain `xpoint`
    # field line keeps re-solving its curve every objective evaluation, so
    # sharing dofs would have the two Newton solvers fight each other.
    #
    # The loaded X-point is always stellsym=False. If it is in fact stellarator
    # symmetric, project it onto a stellsym=True curve so the singular field line
    # is exactly stellarator symmetric: the stellsym=True parametrization keeps
    # only the symmetric Fourier coefficients (xc, ys, zs); copying those by name
    # drops the antisymmetric ones (xs, yc, zc), which are zero up to `defect`.
    xc = xpoint.curve
    defect = _stellsym_defect(xc)
    scale = float(np.max(np.abs(xc.gamma())))
    if defect <= STELLSYM_TOL * scale:
        fl_curve = CurveXYZFourierSymmetries(xc.quadpoints, xc.order, xc.nfp, True, ntor=xc.ntor)
        sym_names = ([f'xc({i})' for i in range(xc.order + 1)]
                     + [f'ys({i})' for i in range(1, xc.order + 1)]
                     + [f'zs({i})' for i in range(1, xc.order + 1)])
        for nm in sym_names:
            fl_curve.set(nm, xc.get(nm))
        print(f"[idx={idx}] X-point IS stellarator symmetric "
              f"(defect={defect:.2e} <= {STELLSYM_TOL:.0e}*scale={STELLSYM_TOL * scale:.2e}); "
              f"projected onto a stellsym=True singular field line")
    else:
        fl_curve = CurveXYZFourierSymmetries(xc.quadpoints, xc.order, xc.nfp, xc.stellsym, ntor=xc.ntor)
        fl_curve.set_dofs(xc.get_dofs())
        print(f"[idx={idx}] X-point is NOT stellarator symmetric "
              f"(defect={defect:.2e} > {STELLSYM_TOL:.0e}*scale={STELLSYM_TOL * scale:.2e}); "
              f"keeping the stellsym=False parametrization")
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
    # This is where the auxiliary currents are FIRST computed.
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
# (they carry the aux coils and the exact monodromy).
xpoints = sing_fls

# Recompute the Boozer surfaces and magnetic axes in the TOTAL field (modular +
# auxiliary): wrap each polished singular field line in a SingularBiotSavart and
# REPLACE the loaded (modular-field) boozer_surfaces / axes with ones solved on
# the total field. This both validates the total-field surfaces (fail fast on a
# non-converged or self-intersecting surface) and warm-starts the combined-coil
# re-solve in the finalize step below.
new_boozer_surfaces, new_axes = [], []
for idx, (boozer_surface, axis, fl) in enumerate(zip(boozer_surfaces, axes, sing_fls)):
    nbs = BoozerSurface(SingularBiotSavart(fl), boozer_surface.surface,
                        boozer_surface.label, boozer_surface.targetlabel,
                        constraint_weight=boozer_surface.constraint_weight,
                        options=boozer_surface.options)
    bs_res = nbs.run_code(boozer_surface.res['iota'], boozer_surface.res['G'])
    # fail fast: the total-field surface must converge AND not self-intersect.
    if not bs_res['success']:
        print(f"ERROR: idx={idx}: total-field BoozerSurface re-solve did not converge")
        raise SystemExit(1)
    if nbs.surface.is_self_intersecting():
        print(f"ERROR: idx={idx}: total-field BoozerSurface is self-intersecting")
        raise SystemExit(1)
    new_boozer_surfaces.append(nbs)

    nax = PeriodicFieldLine(SingularBiotSavart(fl), axis.curve, options=axis.options)
    ax_res = nax.run_code(axis.res['length'])
    # fail fast: the total-field magnetic axis must converge.
    if not ax_res['success']:
        print(f"ERROR: idx={idx}: total-field magnetic-axis re-solve did not converge")
        raise SystemExit(1)
    new_axes.append(nax)
boozer_surfaces, axes = new_boozer_surfaces, new_axes

print(f"non-QS ratio (total field): {[NonQuasiSymmetricRatio(bs, SingularBiotSavart(fl)).J()**0.5 for bs, fl in zip(boozer_surfaces, sing_fls)]}")

os.makedirs(OUT_DIR, exist_ok=True)

print("""
################################################################################
### Finalize: rebuild aux coils + re-solve on the COMBINED coil set ###########
################################################################################
""")
# Mirror boozer_singular_opt.py's finalize step: build simsopt coils from each
# fl's solved mu and re-solve the Boozer surface and magnetic axis with
# modular+aux coils, so the saved design is self-consistent for downstream
# tracing -- mk_manifolds.py uses boozer_surface.biotsavart, which must contain
# the aux field. The aux coils are saved as explicit CurveXYZFourier coils on a
# pure-C++ BiotSavart (NOT SingularBiotSavart): the total field is identical
# (~1e-17) but tracing runs at native speed instead of calling back into
# Python/jax per integration step.
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

    field = BiotSavart(combined_coils)
    bs_out = BoozerSurface(field, boozer_surface.surface,
                           Volume(boozer_surface.surface), boozer_surface.surface.volume())
    bs_res = bs_out.run_code(boozer_surface.res['iota'], boozer_surface.res['G'])
    if not bs_res['success'] or bs_out.surface.is_self_intersecting():
        print(f"ERROR: idx={idx}: BoozerSurface re-solve on the combined coil set failed")
        print(f"ABORT: design_unpolished_final_{DEVICE_ID}.json will not be written.")
        raise SystemExit(1)

    new_ax = PeriodicFieldLine(field, ax.curve)
    ax_res = new_ax.run_code(CurveLength(ax.curve).J())
    if not ax_res['success']:
        print(f"ERROR: idx={idx}: magnetic-axis re-solve on the combined coil set failed")
        print(f"ABORT: design_unpolished_final_{DEVICE_ID}.json will not be written.")
        raise SystemExit(1)

    out_boozer_surfaces.append(bs_out)
    out_iota_Gs.append([bs_res['iota'], bs_res['G']])
    out_axes.append(new_ax)
    curves_to_vtk([coil.curve for coil in aux_coils], OUT_DIR + f'aux_coils_{idx}')

# VTK files consumed by mk_paraview.py (device views). Names match the polished
# pipeline's *_opt_final layout so run_render.sh / mk_paraview.py need no change.
sdf.to_vtk(OUT_DIR + 'vessel_opt_final', nx=40, ny=40, nz=40)
curves_to_vtk(curves, OUT_DIR + "curves_opt_final")
curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + "xpoint_curves_opt_final")
curves_to_vtk([fl.curve for fl in sing_fls], OUT_DIR + "xpoint_singular_curves_opt_final")
curves_to_vtk([ax.curve for ax in out_axes], OUT_DIR + "ma_opt_final")
for idx, bs_out in enumerate(out_boozer_surfaces):
    bs_out.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_final")

# standard 5-entry layout consumed by mk_manifolds.py: the boozer surfaces and
# axes carry the COMBINED (modular + aux) coils, and the singular-polish field
# lines sit in the x-point slot.
save([out_boozer_surfaces, out_iota_Gs, out_axes, sing_fls, sdf],
     OUT_DIR + f'design_unpolished_final_{DEVICE_ID}.json')
with open(OUT_DIR + f'design_unpolished_final_{DEVICE_ID}.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"wrote {OUT_DIR}design_unpolished_final_{DEVICE_ID}.json (combined modular+aux coil set, unpolished)")

# ---- descriptive summary of the unpolished device (no thresholds/errors) ----
# These mirror the descriptive (no-threshold) metrics of the polished summary so
# the device shows up with its key numbers in device_browser. There is no
# optimization here, so no constraint relative-errors and no max_rel_error.txt.
final_metrics = {}
final_metrics['nonQS_percent'] = (
    100. * (sum(NonQuasiSymmetricRatio(bs, bs.biotsavart) for bs in out_boozer_surfaces).J()
            / len(out_boozer_surfaces))**0.5, None, None)

# mirror ratio max|B|/min|B| on the (first) magnetic surface, in the TOTAL field,
# via a fresh BiotSavart on the combined coils so no live point cache is disturbed.
_bs_mirror = BiotSavart(out_boozer_surfaces[0].biotsavart.coils)
_bs_mirror.set_points(out_boozer_surfaces[0].surface.gamma().reshape((-1, 3)))
_modB_surf = _bs_mirror.AbsB()
final_metrics['mirror_ratio'] = (float(np.max(_modB_surf) / np.min(_modB_surf)), None, None)
final_metrics['aspect_ratio'] = (out_boozer_surfaces[0].surface.aspect_ratio(), None, None)

# record the solved auxiliary-coil parameters (currents in Amperes) per fl.
for fi, fl in enumerate(sing_fls):
    for k, nm in enumerate(MU_NAMES):
        v = float(fl.mu[k])
        if k < num_aux:
            v *= _CURRENT_SCALE
        final_metrics[f'aux_{nm}_idx{fi}'] = (v, None, None)

# record the full 2x2 monodromy matrix of the SINGULAR POLISH per X-point so the
# device browser can display it; entries are descriptive (no threshold/error).
# Greene's residue R = (2 - tr(M))/4 is recorded for every X-point.
for ti, fl in enumerate(sing_fls):
    Mm = np.asarray(fl.res['monodromy_matrix'])
    for a in range(2):
        for b in range(2):
            final_metrics[f'monodromy_M{a}{b}_idx{ti}'] = (float(Mm[a, b]), None, None)
    final_metrics[f'greene_residue_idx{ti}'] = (
        (2.0 - float(np.trace(Mm))) / 4.0, None, None)

with open(OUT_DIR + 'summary.txt', 'w') as f:
    f.write(f"# {'metric':<30s} {'value':>16s} {'threshold':>16s} {'rel_error':>16s}\n")
    for name, (value, threshold, rel_err) in final_metrics.items():
        thr_str = f"{threshold:.6e}" if threshold is not None else "n/a"
        err_str = f"{rel_err:.6e}"   if rel_err   is not None else "n/a"
        f.write(f"  {name:<30s} {value:.6e}   {thr_str:>16s}   {err_str:>16s}\n")
print(f"wrote {OUT_DIR}summary.txt")
