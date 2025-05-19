#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

def stage_2_currents_only(surf, biotsavart):
    """
    Stage-II optimization over coil currents only. Minimize the squared flux over the surface,
    by varying the coil currents.

    NOTE: It is assumed that only the coil currents are set to be free. Fix all other dofs
    of the biotsavart object!

    Parameters
    ----------
    surf : SurfaceRZFourier
        The surface to optimize over.
    biotsavart : BiotSavart
        The BiotSavart object containing the coils and their currents.
    
    Returns
    -------
    biotsavart : BiotSavart
        The BiotSavart object with the optimized currents.
    """
    # objective
    Jf = SquaredFlux(surf, biotsavart)
    def fun(dofs):
        # use only biotsavart dofs
        biotsavart.x = dofs
        J = Jf.J()
        grad = Jf.dJ()
        return J, grad
    
    dofs = biotsavart.x

    
    # solve
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 2000, 'maxcor': 300, 'gtol':1e-16, 'ftol':1e-16}, tol=1e-15)

    biotsavart.x = res.x

    return biotsavart, res