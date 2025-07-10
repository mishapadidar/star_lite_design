#!/usr/bin/env python3

"""
This examples demonstrate how to use SIMSOPT to compute guiding center
trajectories of particles in cylindrical coordinates for a vacuum field.

This example takes advantage of MPI if you launch it with multiple
processes (e.g. by mpirun -n or srun), but it also works on a single
process.
"""

import time
import os
import logging
import sys
import numpy as np

from simsopt._core import load
from simsopt.configs import get_ncsx_data
from simsopt.field import (BiotSavart, InterpolatedField, coils_via_symmetries, trace_particles_starting_on_curve,
                           SurfaceClassifier, LevelsetStoppingCriterion, plot_poincare_data, particles_to_vtk)
from simsopt.geo import SurfaceRZFourier, SurfaceXYZTensorFourier, curves_to_vtk, CurveRZFourier
from simsopt.util import in_github_actions, proc0_print, comm_world
from simsopt.util.constants import ELEMENTARY_CHARGE, ONE_EV
ELECTRON_MASS = 9.1093837139e-31 #kg

logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

proc0_print("Running 1_Simple/tracing_particle.py", flush=True)
proc0_print("====================================", flush=True)

# If we're in the CI, make the run a bit cheaper:
nparticles = 50
degree = 3


indir = f"./output/designB/force_weight_1e-11/"
[boozer_surfaces, iota_Gs, ma_list, xpoint_curves] = load(indir + "designB_after_forces_opt.json")

for config, (ma, boozer_surface) in enumerate(zip(ma_list, boozer_surfaces)):
    # Directory for output
    OUT_DIR = indir + f"tracing_particle_{config}/"
    os.makedirs(OUT_DIR, exist_ok=True)


    surface=boozer_surface.surface
    nfp = surface.nfp
    stellsym=boozer_surface.surface.stellsym
    biotsavart=boozer_surface.biotsavart
    coils = biotsavart.coils

    modB = np.mean(np.linalg.norm(biotsavart.set_points(ma.gamma()).B(), axis=1))
    target_modB = 87.5*1e-3
    modB = np.mean(np.linalg.norm(biotsavart.set_points(ma.gamma()).B(), axis=1))
    curves = [c.curve for c in coils]
    bs = BiotSavart(coils)
    proc0_print("Mean(|B|) on axis =", modB, flush=True)
    proc0_print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)), flush=True)
    # curves_to_vtk(curves + [ma], OUT_DIR + 'coils')
    
    ntor=surface.ntor
    mpol=surface.mpol
    stellsym=surface.stellsym
    s = SurfaceXYZTensorFourier.from_nphi_ntheta(mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp,
                                          range="full torus", nphi=64, ntheta=24)
    s.x = surface.x
    
    sc_particle = SurfaceClassifier(s, h=0.05, p=2)
    if comm_world is None or comm_world.rank == 0:
        sc_particle.to_vtk(OUT_DIR + 'sc_particle', h=0.05)
    
    n = 64
    rs = np.linalg.norm(s.gamma()[:, :, 0:2], axis=2)
    zs = s.gamma()[:, :, 2]
    
    rrange = (np.min(rs), np.max(rs), n)
    phirange = (0, 2*np.pi/nfp, n*2)
    # exploit stellarator symmetry and only consider positive z values:
    zrange = (0, np.max(zs), n//2)
    bsh = InterpolatedField(
        bs, degree, rrange, phirange, zrange, True, nfp=nfp, stellsym=True
    )
    
    # sample on a finer magnetic axis grid
    quadpoints = np.linspace(0, 1/ma.nfp, 128, endpoint=False)
    ma_fine = CurveRZFourier(quadpoints, ma.order, ma.nfp, ma.stellsym)
    ma_fine.x = ma.x
    
    def trace_particles(bfield, label, mode='gc_vac'):
        tmax=0.2 # seconds

        t1 = time.time()
        gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
            ma_fine, bfield, nparticles, tmax=tmax, seed=1, mass=ELECTRON_MASS, charge=-ELEMENTARY_CHARGE,
            Ekin=2.86e3*ONE_EV, umin=-1, umax=+1, comm=comm_world, tol=1e-9,
            stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
            forget_exact_path=True)
        t2 = time.time()
                
        if comm_world is None or comm_world.rank == 0:
            import matplotlib.pyplot as plt
            loss_times = np.array([dat[1,0] for dat in gc_tys])
            np.savetxt(OUT_DIR + f'losses_{config}.txt', loss_times)

            if np.any(loss_times < tmax):

                loss_times = loss_times[loss_times<tmax]
                sort_loss_times = np.sort(loss_times)
                cumulative_fraction = np.arange(1, len(sort_loss_times) + 1) / nparticles
                sort_loss_times = np.concatenate(([0], sort_loss_times, [tmax]))
                cumulative_fraction = np.concatenate(([0], cumulative_fraction, [cumulative_fraction[-1]]))
                plt.semilogx(sort_loss_times, 100*cumulative_fraction)
                
                plt.xlabel('time [s]')
                plt.ylabel('loss fraction')
                plt.savefig(OUT_DIR+f'loss_fraction_{config}.png')
                plt.clf()
                
            proc0_print(f"Time for particle tracing={t2-t1:.3f}s. Num steps={sum([len(l) for l in gc_tys])//nparticles}", flush=True)

            #particles_to_vtk(gc_tys, OUT_DIR + f'particles_{label}_{mode}')
            #plot_poincare_data(gc_phi_hits, phis, OUT_DIR + f'poincare_particle_{config}_{label}_loss.png', mark_lost=True, surf=surface)
            #plot_poincare_data(gc_phi_hits, phis, OUT_DIR + f'poincare_particle_{config}_{label}.png', mark_lost=False, surf=surface)
    
    
    proc0_print('Error in B', bsh.estimate_error_B(1000), flush=True)
    proc0_print('Error in AbsB', bsh.estimate_error_GradAbsB(1000), flush=True)
    #trace_particles(bs, 'bs', 'gc_vac')
    trace_particles(bsh, 'bsh', 'gc_vac')
    
    proc0_print("End of 1_Simple/tracing_particle.py", flush=True)
    proc0_print("====================================", flush=True)
