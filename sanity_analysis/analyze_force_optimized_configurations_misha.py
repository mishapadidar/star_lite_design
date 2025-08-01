import numpy as np
from simsopt._core import load
from simsopt.geo import (SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual,
                         Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas, CurveCurveDistance)
from simsopt.field.selffield import regularization_circ
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.field import BiotSavart
import pandas as pd
from star_lite_design.utils.curve_vessel_distance import CurveVesselDistance


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
# force_weight=1e-11
# indir = f"./output/designB/force_weight_{force_weight}/"
# [boozer_surfaces, iota_Gs, axis_curves, xpoint_curves] = load(indir + "designB_after_forces_opt.json")
[boozer_surfaces, iota_Gs, axis_curves, xpoint_curves] = load('../designs/designB_after_forces_opt_19.json')

# or load original design
# [boozer_surfaces, iota_Gs, axis_curves, xpoint_curves] = load("../designs/designB_after_scaled.json")


# vacuum vessel points
df = pd.read_csv("../designs/sheetmetal_chamber.csv")
X_vessel = df.values

for ii, bsurf in enumerate(boozer_surfaces):
    print("")
    print(f"Group {ii}")


    bs = bsurf.biotsavart
    surf = bsurf.surface
    coils = bs.coils
    curves = [c.curve for c in coils]
    currents = [c.current for c in coils]

    # rebuild the boozer surface and populate the res attribute
    bsurf = BoozerSurface(bs, surf, bsurf.label, bsurf.targetlabel, options={'newton_tol':1e-13, 'newton_maxiter':20})
    bsurf.run_code(iota_Gs[ii][0], iota_Gs[ii][1])

    # check magnetic axis field strength
    axis = axis_curves[ii].curve
    xyz = axis.gamma()
    bs.set_points(xyz)
    B_axis = bs.B()
    B_norm = np.linalg.norm(B_axis, axis=1)
    mean_B_norm = np.mean(B_norm)
    err = mean_B_norm - B_target
    print(f"Axis field strength: {mean_B_norm:.6f} T")
    if np.abs(err)/B_target > 0.01:
        print("     WARNING: axis field strength is off by more than 1%")

    # check coil currents
    currents = np.array([c.get_value() for c in currents])
    print("Currents:", currents)
    if np.any(np.abs(currents) > 60000):
        print("     WARNING: currents are too large")
    
    # check coil2coil distance, 
    Jccdist = CurveCurveDistance(curves, cc_dist)
    if Jccdist.shortest_distance() <= 0.15:
        print("     WARNING: coil2coil distance is too small")
    print(f"Coil-to-coil distance: {Jccdist.shortest_distance():.6f} m")

    # check device major radius,
    print("major radius:", surf.major_radius())
    if np.abs(surf.major_radius() - 0.5) > 0.01:
        print("     WARNING: major radius is off")

    # check forces
    print("coil forces:")
    for jj, c in enumerate(coils):
        force = np.linalg.norm(coil_force(c, coils, regularization_circ(coil_minor_radius)), axis=1)
        print(f"  {jj}) max force: %.2f, mean force: %.2f"%(np.max(np.abs(force)), np.mean(np.abs(force))))

    # check QS
    Jqs = NonQuasiSymmetricRatio(bsurf, BiotSavart(bsurf.biotsavart.coils))
    qs_err = np.sqrt(Jqs.J())
    print("QS err", qs_err)
    if qs_err > 0.1:
        print("     WARNING: non-quasi-symmetric ratio is too large")

    # iota
    print("iota", bsurf.res['iota'])

    # check coil-curvature
    msc_max = np.max([np.mean(c.kappa()**2) for c in curves])
    kappa_max = max([np.max(c.kappa()) for c in curves])
    print("largest mean-square curvature", msc_max) # threshold 20.
    print("max coil curvature", kappa_max) # threshold 1/22 mm
    if msc_max > 20 or kappa_max > 1/(22e-3):
        print("     WARNING: coil curvature is too large")

    # check coil-length
    length_max = np.max([np.mean(np.linalg.norm(c.gammadash(),axis=-1)) for c in curves])
    print("max coil length", length_max) # threshold 4
    if length_max > 4:
        print("     WARNING: coil length is too large")


    # coil-vessel distance
    Jcv = CurveVesselDistance(curves, X_vessel, coil_minor_radius)
    print("min coil-vessel distance:", Jcv.shortest_distance()) 
    if Jcv.shortest_distance() < coil_minor_radius: # 0.054 minimum
        print("     WARNING: coil-vessel distance is too small")

    # X-point to vessel distance
    x_curve = xpoint_curves[ii].curve
    Jcv = CurveVesselDistance([x_curve], X_vessel, 0.0)
    print("min Xpoint-to-vessel distance:", Jcv.shortest_distance())
    # curves_to_vtk([x_curve], indir + f"xpoint_{ii}")
    if Jcv.shortest_distance() < coil_minor_radius:
        print("     WARNING: X-point to vessel distance is too small")

    # check if perfectly nfp=2 and stellsym
    angle = 2*np.pi/surf.nfp
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    pt = np.array([[1.357683246, 1.4125674574, 0.213231]])
    Rpt = pt@R
    pts = np.concatenate((pt, Rpt), axis=0)
    bs.set_points(pts)
    B_cyl = bs.B_cyl()
    print(B_cyl)
    assert np.linalg.norm(B_cyl[0] - B_cyl[1]) <1e-16, "symmetry is not satisfied"

    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    pt = np.array([[1.357683246, 1.4125674574, 0.213231]])
    Rpt = pt@R
    pts = np.concatenate((pt, Rpt), axis=0)
    bs.set_points(pts)
    B_cyl = bs.B_cyl()
    print(B_cyl)
    assert np.linalg.norm(B_cyl[0, 0] + B_cyl[1, 0]) <1e-16, "symmetry is not satisfied"
    assert np.linalg.norm(B_cyl[0, 1:] - B_cyl[1, 1:]) <1e-16, "symmetry is not satisfied"

    # # check magnetic axis is computed correctly
    # from star_lite_design.utils.find_x_point import find_x_point
    # xyz = axis.gamma()
    # r0 = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    # z0 = xyz[:, 2]
    # ma_fp, ma_ft, ma_success=  find_x_point(bs, r0, z0, 2, 8)
    # xyz2 = ma_fp.gamma()
    # assert np.max(np.abs(xyz - xyz2)), "axis does not match"
