#!/usr/bin/env python3

import sys
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

"""
The script optimizes the Star_lite device to have 3 configurations with different iota values, and low coil forces.
This script was run as a second stage of optimization, after star_lite was optimized for quasi-symmetry etc.
"""


print("================================")

p = Path(sys.argv[1])
dat = load(p)
[boozer_surfaces, iota_Gs, axes, xpoints, sdf] = dat

for xp in xpoints:
    xp.run_code(CurveLength(xp.curve).J())

tmos = [TangentMap(xp, BiotSavart(boozer_surface.biotsavart.coils), 0.) for xp, boozer_surface in zip(xpoints, boozer_surfaces)]
M = [tmo.matrix for tmo in tmos][0]
print(M)

max_Z = np.max([np.abs(c.curve.gamma()[:, 2]) for c in boozer_surfaces[0].biotsavart.coils])

# mu = (I1..IN, r1..rN, Z)
radii = np.array([0.5, 1.0])
N = len(radii)
mu0 = np.concatenate([np.zeros(N), np.asarray(radii), [max_Z + 0.15]])
fixed_mu_arg = ('z',)


#radii = np.array([1.])
#N = len(radii)
#mu0 = np.concatenate([np.zeros(N), np.asarray(radii), [max_Z + 0.15]])
#fixed_mu_arg = ('z', 'r1')

#radii = np.array([1.0])
#N = len(radii)
#mu0 = np.concatenate([np.zeros(N), np.asarray(radii), [max_Z + 0.15]])
#fixed_mu_arg = ('z',)


sing_xpoints = []
for xpoint, boozer_surface in zip(xpoints, boozer_surfaces):
    xpoint_fl = SingularPeriodicFieldLine(BiotSavart(boozer_surface.biotsavart.coils), xpoint.curve,
            options={'newton_tol':1e-9, 'newton_maxiter':20, 'verbose':True, 'use_lstsq':True, 'monodromy_constraint':'trace'})
    xpoint_fl.need_to_run_code = True
    res = xpoint_fl.run_code(CurveLength(xpoint.curve).J(), mu=mu0.copy(), fixed_mu=fixed_mu_arg)
    sing_xpoints.append(xpoint_fl)

assert res['success']

# Convert the solved auxiliary parameters into simsopt coils, recompute the
# Boozer surface with the new auxiliary coils, and write VTK files.
_CURRENT_SCALE = 1.0e7 / (4.0 * np.pi)   # matches _CURRENT_SCALE in singularperiodicfieldline.py.

out_boozer_surfaces = []
out_iota_Gs = []

for idx, (xp_fl, boozer_surface, (iota, G)) in enumerate(zip(sing_xpoints, boozer_surfaces, iota_Gs)):
    mu_sol = xp_fl.res['mu']
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

    if xp_fl.stellsym_aux:
        aux_coils = coils_via_symmetries(base_curves, base_currents, 1, True)
    else:
        aux_coils = [Coil(c, I) for c, I in zip(base_curves, base_currents)]

    # Recompute the BoozerSurface with the new aux coils added to the original coil set.
    combined_coils = boozer_surface.biotsavart.coils + aux_coils
    bs_out = BoozerSurface(BiotSavart(combined_coils), boozer_surface.surface,
                           Volume(boozer_surface.surface), boozer_surface.surface.volume())
    bs_res = bs_out.run_code(iota, G)
    out_boozer_surfaces.append(bs_out)
    out_iota_Gs.append([bs_res['iota'], bs_res['G']])

    # VTK output for this configuration.
    curves_to_vtk([coil.curve for coil in aux_coils], f'aux_coils_{idx}')
    curves_to_vtk([xp_fl.curve], f'polished_xp_{idx}')
    bs_out.surface.to_vtk(f'polished_boozer_surface_{idx}')

# Original coils + sdf — written once at the end.
curves_to_vtk([c.curve for c in boozer_surfaces[0].biotsavart.coils], 'coils')
sdf.to_vtk('sdf')

save([out_boozer_surfaces, out_iota_Gs, axes, sing_xpoints, sdf], p.parent / 'singular.json')

# Sanity check: with the combined coil set (originals + auxiliary), the
# periodic field-line + monodromy residual at the recovered xpoint should be
# small. No aux contribution and no Newton iteration here.
print("Verification (residual at recovered xpoint with combined field):")
for idx, (xp_fl, bs_out) in enumerate(zip(sing_xpoints, out_boozer_surfaces)):
    new_bs = BiotSavart(bs_out.biotsavart.coils)
    r_check, M_check = xp_fl.residual_norm_no_aux(new_bs)
    print(f"  [idx={idx}] ||r||_inf = {np.linalg.norm(r_check, ord=np.inf):.3e}  "
          f"tr(M) = {float(M_check[0, 0] + M_check[1, 1]):+.6f}")
