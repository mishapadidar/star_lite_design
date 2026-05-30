#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from simsopt._core import load
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import CurveLength, CurveXYZFourierSymmetries, SurfaceXYZTensorFourier
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.boozer_surface_utils import BoozerResidual, CurveBoozerSurfaceDistance
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine
from star_lite_design.utils.curve_periodicfieldline_distance import CurvesPeriodicFieldlineDistance
from star_lite_design.utils.tangent_map import TangentMap, Monodromy
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk,
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data)
import time
from simsopt.util import in_github_actions, proc0_print, comm_world
import os
from pathlib import Path
import sys

# =========================
# Load data
# =========================
p = Path(sys.argv[1])
data = load(p)
[boozer_surfaces, iota_Gs, axes, xpoints, sdf] = data

# Pick first configuration
bs = boozer_surfaces[0].biotsavart
R_a = axes[0].curve.gamma()[0][0]
Z_a = axes[0].curve.gamma()[0][2]

R_xp = xpoints[0].curve.gamma()[0][0]
Z_xp = xpoints[0].curve.gamma()[0][2]
nfp = xpoints[0].curve.nfp

surface = boozer_surfaces[0].surface
surf = SurfaceXYZTensorFourier(mpol=surface.mpol, ntor=surface.ntor, quadpoints_phi=np.linspace(0, 1, 32, endpoint=False), 
        quadpoints_theta=np.linspace(0, 1, 32, endpoint=False), nfp=nfp, stellsym=surface.stellsym)
surf.fit_to_curve(axes[0].curve, 0.45, flip_theta=False)

sc_fieldline = SurfaceClassifier(surf, h=0.05, p=2)
if comm_world is None or comm_world.rank == 0:
    surf.to_vtk(str(p.parent /  'sc'))
    sc_fieldline.to_vtk(str(p.parent /  'sc_sdf'), h=0.05)

def surface_cross_sections(surface):
    phis = np.linspace(0, 0.25, 9)
    
    for ii, phi in enumerate(phis):
        XYZ = surface.cross_section(phi, thetas=100)
        R, Z = np.hypot(XYZ[:, 0], XYZ[:, 1]), XYZ[:, 2]
        RZ = np.concatenate((R[:, None], Z[:, None]), axis=1)
        np.savetxt(p.parent / f'surface_cross_{ii}.txt', RZ, delimiter=',') 

def trace_fieldlines(bfield, R_a, Z_a, R_xp, Z_xp):
    print("running the integration now...")
    phis = np.linspace(0, 0.25, 9)*2*np.pi
    dirs = np.linspace(0, 2*np.pi, 8, endpoint=False)
    tmax_fl = 1e4

    r = 0.05
    nmanif = 12

    # Trace from a "disk" of seeds around each fixed point: the loaded top
    # xpoint at (R_xp, +Z_xp) and its stellsym partner at (R_xp, -Z_xp). For
    # each fixed point we sample along 8 radial directions, nmanif points per
    # direction, traced in both +/- field directions.
    fixed_points = [('top', R_xp, Z_xp), ('bot', R_xp, -Z_xp)]

    for label, R_fp, Z_fp in fixed_points:
        proc0_print(f"== {label} fixed point disk (R={R_fp:.4f}, Z={Z_fp:.4f}) ==")
        for ii, dd in enumerate(dirs):
            for sign in [1., -1.]:

                R0 = np.linspace(R_fp, R_fp + r * np.cos(dd), nmanif)[1:]
                Z0 = np.linspace(Z_fp, Z_fp + r * np.sin(dd), nmanif)[1:]

                t1 = time.time()
                try:
                    fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
                        sign*bfield, R0, Z0, tmax=tmax_fl, tol=1e-9, comm=comm_world,
                        phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
                except Exception as e:
                    proc0_print(f"  skipped {label} ii={ii} sign={sign}: {type(e).__name__}: {e}")
                    continue

                t2 = time.time()
                proc0_print(f"  {label} ii={ii} sign={sign} {t2-t1:.1f}s, hits/seed: " +
                            ' '.join(str(h.shape[0]) for h in fieldlines_phi_hits))

                if comm_world is None or comm_world.rank == 0:
                    for i in range(len(phis)):
                        data_this_phi = np.zeros((0, 4))
                        for j in range(len(fieldlines_phi_hits)):
                            toadd = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == i)[0], 2:]
                            toadd = np.concatenate((j*np.ones((toadd.shape[0], 1)), toadd), axis=1)
                            data_this_phi = np.concatenate((data_this_phi, toadd), axis=0)
                        ss = '+' if sign > 0 else '-'
                        np.savetxt(p.parent / f'poincare{i}_{label}_{ii}_{ss}.txt',
                                   data_this_phi, comments='', delimiter=',')
    proc0_print("divertor poincare done")
    
    # interior: axis -> X-point
    tmax_fl = 1e4
    nmanif = 12
    R0 = np.linspace(R_a, R_xp, nmanif)
    Z0 = np.linspace(Z_a, Z_xp, nmanif)

    try:
        t1 = time.time()
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-9, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        t2 = time.time()
        proc0_print(f"  axis->xp {t2-t1:.1f}s, hits/seed: " +' '.join(str(h.shape[0]) for h in fieldlines_phi_hits))

        if comm_world is None or comm_world.rank == 0:
            for i in range(len(phis)):
                data_this_phi = np.zeros((0, 4))
                for j in range(len(fieldlines_phi_hits)):
                    toadd = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == i)[0], 2:]
                    toadd = np.concatenate( (j*np.ones((toadd.shape[0], 1)), toadd), axis=1)
                    data_this_phi = np.concatenate((data_this_phi, toadd), axis=0)
                np.savetxt(p.parent / f'poincare_interior_{i}.txt', data_this_phi, comments='', delimiter=',')
    except Exception as e:
        proc0_print(f"  skipped interior (axis->xp): {type(e).__name__}: {e}")

    # interior: axis -> origin (R=0, Z=0)
    R0 = np.linspace(R_a, 1e-3, nmanif)   # avoid R=0 exactly (cylindrical singularity)
    Z0 = np.linspace(Z_a, 0.0, nmanif)

    try:
        t1 = time.time()
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R0, Z0, tmax=tmax_fl, tol=1e-9, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        t2 = time.time()
        proc0_print(f"  axis->origin {t2-t1:.1f}s, hits/seed: " +' '.join(str(h.shape[0]) for h in fieldlines_phi_hits))

        if comm_world is None or comm_world.rank == 0:
            for i in range(len(phis)):
                data_this_phi = np.zeros((0, 4))
                for j in range(len(fieldlines_phi_hits)):
                    toadd = fieldlines_phi_hits[j][np.where(fieldlines_phi_hits[j][:, 1] == i)[0], 2:]
                    toadd = np.concatenate( (j*np.ones((toadd.shape[0], 1)), toadd), axis=1)
                    data_this_phi = np.concatenate((data_this_phi, toadd), axis=0)
                np.savetxt(p.parent / f'poincare_inward_{i}.txt', data_this_phi, comments='', delimiter=',')
    except Exception as e:
        proc0_print(f"  skipped interior (axis->origin): {type(e).__name__}: {e}")

    proc0_print("interior poincare done")

# --- vessel cross-sections at the manifold phi values ---
def extract_vessel_cross_sections(sdf, nR=100, nZ=100):
    
    """Sample the SDF on (R, Z) grids at each phi and save zero-level contours."""
    from skimage import measure

    phis = np.linspace(0, 0.25, 9)*2*np.pi
    Rg = np.linspace(0, 1.5, nR)
    Zg = np.linspace(-1., 1., nZ)
    Rm, Zm = np.meshgrid(Rg, Zg, indexing="xy")

    for i, phi in enumerate(phis):
        X = (Rm * np.cos(phi)).ravel()
        Y = (Rm * np.sin(phi)).ravel()
        Z = Zm.ravel()

        d = sdf.eval(X, Y, Z).reshape(Rm.shape)

        # Extract zero-level contours directly without plotting
        contours = measure.find_contours(d, level=0.0)

        segs = []

        for contour in contours:
            if contour.shape[0] < 2:
                continue

            # contour coords are in array index space -> map to physical R,Z
            rows = contour[:, 0]
            cols = contour[:, 1]

            Rvals = np.interp(cols, np.arange(nR), Rg)
            Zvals = np.interp(rows, np.arange(nZ), Zg)

            segs.append(np.column_stack((Rvals, Zvals)))
            segs.append(np.array([[np.nan, np.nan]]))

        if segs:
            data = np.vstack(segs)
        else:
            data = np.zeros((0, 2))
            proc0_print(f"  warning: empty vessel cross-section at phi={phi:.4f}")

        np.savetxt(
            p.parent / f"vessel_cross_{i}.txt",
            data,
            delimiter=",",
            comments="",
            header="R,Z (NaN rows separate disconnected components)",
        )
    proc0_print("vessel cross-sections written")

if comm_world is None or comm_world.rank == 0:
    surface_cross_sections(surface)
    extract_vessel_cross_sections(sdf)

trace_fieldlines(bs, R_a, Z_a, R_xp, Z_xp)
