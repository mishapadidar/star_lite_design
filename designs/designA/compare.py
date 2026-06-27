#!/usr/bin/env python

import sys
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveVesselDistance, ArclengthVariation, CurveXYZFourierSymmetries,
                         CurveXYZFourier, CurveRZFourier, curves_to_vtk, MajorRadius)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty
from simsopt._core import load, save
from simsopt.field.force import coil_force
from simsopt.field.selffield import regularization_circ
from prettytable import PrettyTable
from scipy.optimize import fsolve, least_squares, root_scalar
from star_lite_design.utils.find_magnetic_axis import find_magnetic_axis
from star_lite_design.utils.find_x_point import find_x_point

#dat_orig = load('../../../direct_multiple/direct_A/designA.json')
#bs_orig = BiotSavart(dat_orig[2])
[boozer_surfaces, iotaGs, magnetic_axes, xpoints] = load('0104183_symmetrized.json')

biotsavart = boozer_surfaces[0].biotsavart
xyz = xpoints[0].gamma()
r0 = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
z0 = xyz[:, 2]

nfp = xpoints[0].nfp
order = xpoints[0].order

c0l = biotsavart.x[0]
c1l = biotsavart.x[1]

c0r = boozer_surfaces[2].biotsavart.x[0]
c1r = boozer_surfaces[2].biotsavart.x[1]

for r in np.linspace(0, 1, 200):
    xnew = biotsavart.x.copy()
    xnew[0] = c0l * (1-r) + c0r * r
    xnew[1] = c1l * (1-r) + c1r * r
    biotsavart.x = xnew
    xp_fp, xp_ft, xp_success = find_x_point(biotsavart, r0, z0, nfp, order)

    xyz = xp_fp.gamma()
    r0 = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    z0 = xyz[:, 2]
    
    print(xp_success, r)

[boozer_surfaces, iotaGs, magnetic_axes, xpoints] = load('0104183_symmetrized.json')
biotsavart = boozer_surfaces[2].biotsavart
xyz = xp_fp.gamma()
r0 = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
z0 = xyz[:, 2]
xp_fp, xp_ft, xp_success = find_x_point(biotsavart, r0, z0, nfp, order)

print(xp_success)
