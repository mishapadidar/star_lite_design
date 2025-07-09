import numpy as np
from simsopt._core import load
from simsopt.geo import (SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual,
                         Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas, CurveCurveDistance)
from simsopt.field.selffield import regularization_circ
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.field import BiotSavart


"""
Print out some relevant features of the configurations that were optimized to have low magnetic forces with
optimization_with_forces.py.
"""

# target values
B_target = 0.0875 # [T]
cc_dist = 0.15 # [m]
major_radius = 0.5 # [m]
coil_minor_radius = 0.054 # 54mm

# load the NEW optimized design
[boozer_surfaces, iota_Gs, axis_curves, xpoint_curves] = load("./output/designB_after_force_opt.json")

for ii, bsurf in enumerate(boozer_surfaces):
    print("")
    print(f"Group {ii}")

    bs = bsurf.biotsavart
    surf = bsurf.surface
    coils = bs.coils
    curves = [c.curve for c in coils]

    # rebuild the boozer surface and populate the res attribute
    bsurf = BoozerSurface(bs, surf, bsurf.label, bsurf.targetlabel, options={'newton_tol':1e-13, 'newton_maxiter':20})
    bsurf.run_code(iota_Gs[ii][0], iota_Gs[ii][1])

    # check magnetic axis field strength
    axis = axis_curves[ii]
    xyz = axis.gamma()
    bs.set_points(xyz)
    B_axis = bs.B()
    B_norm = np.linalg.norm(B_axis, axis=1)
    mean_B_norm = np.mean(B_norm)
    err = mean_B_norm - B_target
    print(f"Axis field strength: {mean_B_norm:.6f} T")
    
    # check coil2coil distance, 
    Jccdist = CurveCurveDistance(curves, cc_dist)
    print(f"Coil-to-coil distance: {Jccdist.shortest_distance():.6f} m")

    # check device major radius,
    print("major radius:", surf.major_radius())

    # check forces
    print("coil forces:")
    for jj, c in enumerate(coils):
        force = np.linalg.norm(coil_force(c, coils, regularization_circ(coil_minor_radius)), axis=1)
        print(f"  {jj}) max force: %.2f, mean force: %.2f"%(np.max(np.abs(force)), np.mean(np.abs(force))))

    # check QS
    Jqs = NonQuasiSymmetricRatio(bsurf, BiotSavart(bsurf.biotsavart.coils))
    print("QS err", np.sqrt(Jqs.J()))

    # iota
    print("iota", bsurf.res['iota'])

    # check coil-curvature
    msc_max = np.max([np.mean(c.kappa()**2) for c in curves])
    kappa_max = max([np.max(c.kappa()) for c in curves])
    print("largest mean-square curvature", msc_max)
    print("max coil curvature", kappa_max)

    # check coil-length
    length_max = np.max([np.mean(np.linalg.norm(c.gammadash(),axis=-1)) for c in curves])
    print("max coil length", length_max)