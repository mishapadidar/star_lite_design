#!/usr/bin/env python3

import sys
import os
import numpy as np
from scipy.optimize import minimize, root_scalar

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
from simsopt.field.selffield import regularization_circ
import pandas as pd
from rich.console import Console
from rich.table import Column, Table

from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.singularperiodicfieldline import SingularPeriodicFieldLine
from star_lite_design.utils.boozer_surface_utils import BoozerResidual, CurveBoozerSurfaceDistance
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine
from star_lite_design.utils.curve_periodicfieldline_distance import CurvesPeriodicFieldlineDistance
from star_lite_design.utils.tangent_map import TangentMap, Monodromy
import yaml
from pathlib import Path


def evaluate_at_phi(curve, phi, tol=1e-10):
    # map phi to [0, 1)
    phi += np.ceil(-phi)

    def curve_val(theta):
        return curve.gamma_pure(curve.x, np.array([theta]))[0]

    def theta2phi(theta_in, phi0):
        xyz = curve_val(theta_in)
        angle = np.arctan2(xyz[1], xyz[0]) / (2 * np.pi) - phi0
        angle += np.ceil(-angle)
        return angle

    def fun(theta):
        # if theta is exactly 1, then override the rest. hacky.
        if theta == 1.:
            return 1. - phi_prime
        angle = theta2phi(theta, phi0)
        return angle - phi_prime

    xyz0 = curve_val(0.)
    phi0 = np.arctan2(xyz0[1], xyz0[0]) / (2 * np.pi)
    phi_prime = phi - phi0
    phi_prime += np.ceil(-phi_prime)

    result = root_scalar(fun, bracket=[0, 1.])
    conv = result.converged

    # check that the result is accurate by dotting with something orthogonal to the angle
    xyz = curve_val(result.root)
    R = np.sqrt(xyz[0]**2 + xyz[1]**2)
    c = np.cos(2 * np.pi * phi)
    s = np.sin(2 * np.pi * phi)
    accurate = np.abs(-s * xyz[0] + c * xyz[1]) / R < tol
    success = conv and accurate

    return xyz, success


"""
The script optimizes the Star_lite device to have 3 configurations with different iota values, and low coil forces.
This script was run as a second stage of optimization, after star_lite was optimized for quasi-symmetry etc.
"""


print("================================")

if len(sys.argv) < 4:
    print("Usage: boozer_singular.py <input_json> <'identity'|'trace'> <ncoils>", file=sys.stderr)
    sys.exit(2)

p = Path(sys.argv[1])
monodromy_constraint = sys.argv[2]
if monodromy_constraint not in ('identity', 'trace'):
    print(f"ERROR: constraint must be 'identity' or 'trace', got {monodromy_constraint!r}",
          file=sys.stderr)
    sys.exit(2)
ncoils = int(sys.argv[3])
if ncoils < 1:
    print(f"ERROR: ncoils must be >= 1, got {ncoils}", file=sys.stderr)
    sys.exit(2)
print(f"Input: {p}")
print(f"Monodromy constraint: {monodromy_constraint}")
print(f"ncoils: {ncoils}")

dat = load(p)
[boozer_surfaces, iota_Gs, axes, xpoints, sdf] = dat

for xp in xpoints:
    xp.run_code(CurveLength(xp.curve).J())

tmos = [TangentMap(xp, BiotSavart(boozer_surface.biotsavart.coils), 0.) for xp, boozer_surface in zip(xpoints, boozer_surfaces)]
M = [tmo.matrix for tmo in tmos][0]
print(M)

max_Z = np.max([np.abs(c.curve.gamma()[:, 2]) for c in boozer_surfaces[0].biotsavart.coils])

_CURRENT_SCALE = 1.0e7 / (4.0 * np.pi)   # matches _CURRENT_SCALE in singularperiodicfieldline.py.

# Polish each xpoint and recompute its Boozer surface + magnetic axis, all with
# the same auxiliary coil set. ncoils is fixed for this run (the scan over
# ncoils is done externally, one value per disBatch task). If any of the three
# solves fails, abort without writing singular.json. Radii are
# np.linspace(0, 1, ncoils+1)[1:]; only Z is fixed during the
# SingularPeriodicFieldLine Newton solve (currents and radii are free).
radii = np.linspace(0, 1, ncoils + 1)[1:]

sing_xpoints = []
out_boozer_surfaces = []
out_iota_Gs = []
out_axes = []

for idx, (xpoint, boozer_surface, (iota, G), ax) in enumerate(
        zip(xpoints, boozer_surfaces, iota_Gs, axes)):
    mu0 = np.concatenate([np.zeros(ncoils), radii, [max_Z + 0.15]])
    fixed_mu_arg = ('z',)

    # (1) Polish the xpoint.
    xpoint_fl = SingularPeriodicFieldLine(
        BiotSavart(boozer_surface.biotsavart.coils), xpoint.curve,
        options={'newton_tol': 1e-9, 'newton_maxiter': 20, 'verbose': True,
                 'use_lstsq': True, 'monodromy_constraint': monodromy_constraint},
    )
    xpoint_fl.need_to_run_code = True
    res = xpoint_fl.run_code(CurveLength(xpoint.curve).J(),
                             mu=mu0.copy(), fixed_mu=fixed_mu_arg)
    if not res['success']:
        print(f"ERROR: idx={idx}, ncoils={ncoils}: SingularPeriodicFieldLine Newton failed")
        print("ABORT: singular.json will not be written.")
        sys.exit(1)

    # Build aux simsopt coils from the solved mu.
    mu_sol = xpoint_fl.res['mu']
    nmu = len(mu_sol)
    N = (nmu - 1) // 2
    Z = float(mu_sol[-1])
    base_curves = []
    base_currents = []
    for k in range(N):
        Ik = float(mu_sol[k])
        rk = float(mu_sol[N + k])
        c = CurveXYZFourier(np.linspace(0, 1, 160, endpoint=False), 1)
        c.x = c.x * 0.
        c.set('zc(0)', Z)
        c.set('xc(1)', rk)
        c.set('ys(1)', rk)
        base_curves.append(c)
        base_currents.append(ScaledCurrent(Current(Ik), _CURRENT_SCALE))
    if xpoint_fl.stellsym_aux:
        aux_coils = coils_via_symmetries(base_curves, base_currents, 1, True)
    else:
        aux_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]
    combined_coils = boozer_surface.biotsavart.coils + aux_coils

    # (2) Recompute the BoozerSurface with the combined coil set.
    bs_out = BoozerSurface(BiotSavart(combined_coils), boozer_surface.surface,
                           Volume(boozer_surface.surface), boozer_surface.surface.volume())
    bs_res = bs_out.run_code(iota, G)
    if not bs_res['success']:
        print(f"ERROR: idx={idx}, ncoils={ncoils}: BoozerSurface re-solve failed")
        print("ABORT: singular.json will not be written.")
        sys.exit(1)

    # Reject a self-intersecting polished surface (checked at phi=0).
    if bs_out.surface.is_self_intersecting():
        print(f"ERROR: idx={idx}, ncoils={ncoils}: polished Boozer surface is self-intersecting")
        print("ABORT: singular.json will not be written.")
        sys.exit(1)

    # (3) Recompute the magnetic axis with the combined coil set.
    new_ax = PeriodicFieldLine(BiotSavart(bs_out.biotsavart.coils), ax.curve)
    ax_res = new_ax.run_code(CurveLength(ax.curve).J())
    if not ax_res['success']:
        print(f"ERROR: idx={idx}, ncoils={ncoils}: magnetic-axis re-solve failed")
        print("ABORT: singular.json will not be written.")
        sys.exit(1)

    # Validate the polished xpoint and axis are evaluable at all phi: the
    # downstream RZ fit in mk_manifolds_STAR_RZ asserts this, so fail fast here.
    _qp = np.linspace(0, 0.5, 2 * 16 + 1, endpoint=False)
    if not all(evaluate_at_phi(xpoint_fl.curve, _phi)[1] for _phi in _qp) or \
       not all(evaluate_at_phi(new_ax.curve, _phi)[1] for _phi in _qp):
        print(f"ERROR: idx={idx}, ncoils={ncoils}: polished xpoint/axis not evaluable at all phi")
        print("ABORT: singular.json will not be written.")
        sys.exit(1)

    # All three converged — record results and emit per-xpoint VTKs.
    sing_xpoints.append(xpoint_fl)
    out_boozer_surfaces.append(bs_out)
    out_iota_Gs.append([bs_res['iota'], bs_res['G']])
    out_axes.append(new_ax)

    curves_to_vtk([coil.curve for coil in aux_coils],
                  str(p.parent / f'aux_coils_{idx}'))
    bs_out.surface.to_vtk(str(p.parent / f'surf_opt_{idx}_final'))

    print(f"xpoint idx={idx}: all three solves converged with ncoils={ncoils}")

# VTK files in the legacy naming used by mk_paraview.py. curves_opt_final.vtu
# contains the combined coil set for idx 0 (original coils + idx-0 aux coils).
curves_to_vtk([c.curve for c in out_boozer_surfaces[0].biotsavart.coils],
              str(p.parent / 'curves_opt_final'))
curves_to_vtk([ax.curve for ax in out_axes], str(p.parent / 'ma_opt_final'))
curves_to_vtk([xp.curve for xp in sing_xpoints], str(p.parent / 'xpoint_curves_opt_final'))
sdf.to_vtk(str(p.parent / 'vessel_opt_final'))

save([out_boozer_surfaces, out_iota_Gs, out_axes, sing_xpoints, sdf], p.parent / 'singular.json')

# Sanity check: with the combined coil set (originals + auxiliary), the
# periodic field-line + monodromy residual at the recovered xpoint should be
# small. No aux contribution and no Newton iteration here.
print("Verification (residual at recovered xpoint with combined field):")
for idx, (xp_fl, bs_out) in enumerate(zip(sing_xpoints, out_boozer_surfaces)):
    new_bs = BiotSavart(bs_out.biotsavart.coils)
    r_check, M_check = xp_fl.residual_norm_no_aux(new_bs)
    print(f"  [idx={idx}] ||r||_inf = {np.linalg.norm(r_check, ord=np.inf):.3e}  "
          f"tr(M) = {float(M_check[0, 0] + M_check[1, 1]):+.6f}")
