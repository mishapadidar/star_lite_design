#!/usr/bin/env python3
import time
import os
import logging
import sys
import numpy as np
from pathlib import Path

from simsopt._core import load
from simsopt.field import BiotSavart, trace_particles_starting_on_curve, SurfaceClassifier, LevelsetStoppingCriterion
from simsopt.geo import SurfaceXYZTensorFourier, curves_to_vtk, CurveRZFourier
from simsopt.util import proc0_print, comm_world
from simsopt.util.constants import ELEMENTARY_CHARGE, ONE_EV
ELECTRON_MASS = 9.1093837139e-31 #kg

logging.basicConfig()
logger = logging.getLogger('simsopt.field.tracing')
logger.setLevel(1)

# If we're in the CI, make the run a bit cheaper:
nparticles = 1000
degree = 7


fname = sys.argv[1]
config = int(sys.argv[2])

proc0_print(f"Running tracing_particle.py config={config}", flush=True)
proc0_print("============================================", flush=True)



[boozer_surfaces, res_list, ma_list, xp_list] = load(fname)

ma = ma_list[config]
boozer_surface = boozer_surfaces[config]

# Directory for output
OUT_DIR = Path("output") / Path(fname).stem
OUT_DIR.mkdir(parents=True, exist_ok=True)

surface=boozer_surface.surface
nfp = surface.nfp
stellsym=boozer_surface.surface.stellsym
biotsavart=boozer_surface.biotsavart
coils = biotsavart.coils

modB = np.mean(np.linalg.norm(biotsavart.set_points(ma.gamma()).B(), axis=1))
curves = [c.curve for c in coils]
bs = BiotSavart(coils)
proc0_print("Mean(|B|) on axis =", modB, flush=True)
proc0_print("Mean(Axis radius) =", np.mean(np.linalg.norm(ma.gamma(), axis=1)), flush=True)

# LOAD THE SURFACE FOR ORBIT CLASSIFICATION (LOST/CONFINED)
ntor_sc=boozer_surfaces[config].surface.ntor
mpol_sc=boozer_surfaces[config].surface.mpol
stellsym_sc=boozer_surfaces[config].surface.stellsym
nfp_sc=boozer_surfaces[config].surface.nfp
s = SurfaceXYZTensorFourier.from_nphi_ntheta(mpol=mpol_sc, ntor=ntor_sc, stellsym=stellsym_sc, nfp=nfp_sc,
                                      range="full torus", nphi=128, ntheta=128)
s.x = boozer_surfaces[config].surface.x
proc0_print(f'loss surface has AR={s.aspect_ratio()}\n')
##########################################################

ot1 = time.time()
sc_particle = SurfaceClassifier(s, h=0.01, p=2)

# sample on a finer magnetic axis grid
quadpoints = np.linspace(0, 1/ma.nfp, 1000, endpoint=False)
ma_fine = CurveRZFourier(quadpoints, ma.order, ma.nfp, ma.stellsym)
ma_fine.x = ma.x

if comm_world is None or comm_world.rank == 0:
    sc_particle.to_vtk(str(OUT_DIR / f'sc_particle_{config}'), h=0.05)
    curves_to_vtk([ma_fine], str(OUT_DIR / f'ma_{config}'))
    s.to_vtk(str(OUT_DIR / f'sc_{config}'))

def trace_particles(bfield, label, mode='gc_vac'):
    tmax=2e-2

    t1 = time.time()
    gc_tys, gc_phi_hits = trace_particles_starting_on_curve(
        ma_fine, bfield, nparticles, tmax=tmax, seed=1, mass=ELECTRON_MASS, charge=-ELEMENTARY_CHARGE,
        Ekin=2.86e3*ONE_EV, umin=-1, umax=+1, comm=comm_world, tol=1e-9,
        stopping_criteria=[LevelsetStoppingCriterion(sc_particle.dist)], mode=mode,
        forget_exact_path=True)
    t2 = time.time()
            
    if comm_world is None or comm_world.rank == 0:
        loss_times = np.array([dat[1, 0] for dat in gc_tys])
        np.savetxt(str(OUT_DIR + f'losses_{config}.txt'), loss_times)
        proc0_print(f"Time for particle tracing={t2-t1:.3f}s.", flush=True)

ot2 = time.time()
proc0_print(f"Time for initial set up={ot2-ot1:.3f}s", flush=True)
trace_particles(bs, 'bs', 'gc_vac')

proc0_print("End of 1_Simple/tracing_particle.py", flush=True)
proc0_print("====================================", flush=True)
