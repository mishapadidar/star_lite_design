#!/usr/bin/env python

import numpy as np
from scipy.optimize import minimize
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, CurveSurfaceDistance, LinkingNumber)
from simsopt.objectives import Weight, SquaredFlux, QuadraticPenalty

def stage_2_normal_field_only(surf, biotsavart):
    """
    Stage-II optimization of the normal field error only

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


def stage_2_with_distance_penalties(surf, biotsavart, cc_dist = 0.15, cs_dist = 0.15):
    """
    Stage-II optimization of the normal field error and distance penalties, 
        min quadratic_flux + coil_coil_distance + coil_surface_distance + linking_number

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
    coils = biotsavart.coils
    curves = [c.curve for c in coils]
    Jccdist = CurveCurveDistance(curves, cc_dist)
    Jcsdist = CurveSurfaceDistance(curves, surf, cs_dist)
    Jf = SquaredFlux(surf, biotsavart)

    # TODO: remove linking number and use bounds/linear constraints instead!
    # Jlink = LinkingNumber(curves)

    # objective
    # Jtotal = Jf + 10*(Jccdist + Jcsdist + Jlink)
    Jtotal = Jf + 10*(Jccdist + Jcsdist)
    def fun(dofs):
        # use only biotsavart dofs
        biotsavart.x = dofs
        J = Jtotal.J()
        grad = Jtotal.dJ()
        return Jtotal.J(), Jtotal.dJ()
    
    dofs = biotsavart.x

    # solve
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': 2000, 'maxcor': 300, 'gtol':1e-16, 'ftol':1e-16}, tol=1e-15)

    biotsavart.x = res.x

    return biotsavart, res