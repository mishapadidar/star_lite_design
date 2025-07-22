import time
import os
import numpy as np
from simsopt.field import (InterpolatedField, SurfaceClassifier, particles_to_vtk,
                           compute_fieldlines, LevelsetStoppingCriterion, plot_poincare_data)
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.util import proc0_print, comm_world
from simsopt._core import load


# If we're in the CI, make the run a bit cheaper:
nfieldlines = 50
tmax = 40000
degree = 2
tol = 1e-12

# Load in the boundary surface:
infile = "../designs/designB_after_forces_opt_19.json"
[boozer_surfaces, iota_Gs, axis_curves, xpoint_curves] = load(infile)


# Directory for output
outdir = "./output/"
os.makedirs(outdir, exist_ok=True)

for ii_config in range(len(boozer_surfaces)):
    print(f"========================================")
    print(f"Configuration {ii_config+1} of {len(boozer_surfaces)}")

    surface = boozer_surfaces[ii_config].surface

    # make the full torus
    surf = SurfaceXYZTensorFourier(nfp=surface.nfp, stellsym=surface.stellsym,
                                   mpol = surface.mpol, ntor = surface.ntor,
                                   quadpoints_phi = np.linspace(0, 1, 64),
                                   quadpoints_theta = np.linspace(0, 1, 64)
                                   )
    surf.unfix_all()
    surface.unfix_all()
    surf.x = surface.x.copy()

    bs = boozer_surfaces[ii_config].biotsavart
    ma = axis_curves[ii_config].curve
    nfp = surf.nfp

    surf.to_vtk(outdir + 'surface')

    sc_fieldline = SurfaceClassifier(surf, h=0.03, p=2)
    sc_fieldline.to_vtk(outdir + 'levelset', h=0.02)

    # compute R,Z of the axis
    xyz = ma.gamma()[0] # (3,)
    R_axis = np.sqrt(xyz[0]**2 + xyz[1]**2)
    Z_axis = xyz[2]

    print("Axis position: R=", R_axis, "Z=", Z_axis)

    # boundary R,Z
    xyz = surf.gamma()[0][0] # (3,)
    R_boundary = np.sqrt(xyz[0]**2 + xyz[1]**2)
    Z_boundary = xyz[2]

    # linearly space points 
    R = np.linspace(R_axis, R_boundary, nfieldlines)
    Z = np.linspace(Z_axis, Z_boundary, nfieldlines)


    def trace_fieldlines(bfield, label):
        t1 = time.time()
        phis = [(i/4)*(2*np.pi/nfp) for i in range(4)]
        fieldlines_tys, fieldlines_phi_hits = compute_fieldlines(
            bfield, R, Z, tmax=tmax, tol=tol, comm=comm_world,
            phis=phis, stopping_criteria=[LevelsetStoppingCriterion(sc_fieldline.dist)])
        t2 = time.time()
        proc0_print(f"Time for fieldline tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in fieldlines_tys])//nfieldlines}", flush=True)
        if comm_world is None or comm_world.rank == 0:
            particles_to_vtk(fieldlines_tys, outdir + f'fieldlines_{label}')
            plot_poincare_data(fieldlines_phi_hits, phis, outdir + f'poincare_fieldline_{label}.png', dpi=150)

        for ph in fieldlines_phi_hits:
            for row in ph:
                if row[1] < 0:
                    print(row[0])

    # uncomment this to run tracing using the biot savart field (very slow!)
    # trace_fieldlines(bs, 'bs')


    # Bounds for the interpolated magnetic field chosen so that the surface is
    # entirely contained in it
    n = 20
    rs = np.linalg.norm(surf.gamma()[:, :, 0:2], axis=2)
    zs = surf.gamma()[:, :, 2]
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2*np.pi/nfp, n*2)
    # exploit stellarator symmetry and only consider positive z values:
    zrange = (0, np.max(zs), n//2)


    def skip(rs, phis, zs):
        # The RegularGrindInterpolant3D class allows us to specify a function that
        # is used in order to figure out which cells to be skipped.  Internally,
        # the class will evaluate this function on the nodes of the regular mesh,
        # and if *all* of the eight corners are outside the domain, then the cell
        # is skipped.  Since the surface may be curved in a way that for some
        # cells, all mesh nodes are outside the surface, but the surface still
        # intersects with a cell, we need to have a bit of buffer in the signed
        # distance (essentially blowing up the surface a bit), to avoid ignoring
        # cells that shouldn't be ignored
        rphiz = np.asarray([rs, phis, zs]).T.copy()
        dists = sc_fieldline.evaluate_rphiz(rphiz)
        skip = list((dists < -0.05).flatten())
        # proc0_print("Skip", sum(skip), "cells out of", len(skip), flush=True)
        return skip


    proc0_print('Initializing InterpolatedField')
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True, skip=skip
    )
    proc0_print('Done initializing InterpolatedField.')

    # bsh.set_points(surf.gamma().reshape((-1, 3)))
    # bs.set_points(surf.gamma().reshape((-1, 3)))
    # Bh = bsh.B()
    # B = bs.B()
    # proc0_print("Mean(|B|) on plasma surface =", np.mean(bs.AbsB()))

    # proc0_print("|B-Bh| on surface:", np.sort(np.abs(B-Bh).flatten()))

    proc0_print('Beginning field line tracing')
    trace_fieldlines(bsh, f'config_{ii_config}')

    proc0_print("========================================")

