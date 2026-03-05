import numpy as np
from trace_simsopt import rescale_device, TraceBoozer
from simsopt.util.mpi import MpiPartition
from simsopt.mhd import Vmec, vmec_compute_geometry
from simsopt.mhd import QuasisymmetryRatioResidual
from simsopt.util import proc0_print
from simsopt._core import load
from mpi4py import MPI
import os
import pickle
ELECTRON_MASS = 9.1093837139e-31 # kg
ONE_EV = 1.602176634e-19  # J
ELEMENTARY_CHARGE = 1.602176634e-19  # C

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

"""
A script for tracing electrons in Boozer coordinates at low and high energies.

"""


# Ars
energy_type = "slow"
current_group = 0
design_file = "../designs/designA_after_scaled.json"
#design_file = "./designs/serial0104183_iota.json"

# ======================================

# load the surface
data = load(design_file)
bsurf = data[0][current_group] # BoozerSurfaces
iota = data[1][current_group][0]
surf = bsurf.surface

proc0_print("\nArgs:")
proc0_print("design_file", design_file)
proc0_print("current_group:",current_group)
proc0_print("iota:", iota)
proc0_print("energy_type:", energy_type)

if energy_type =="fast":
    energy = 2.86e3 * ONE_EV
    speed = np.sqrt(2 * energy / ELECTRON_MASS)
else:
    energy = 20 * ONE_EV
    speed = np.sqrt(2 * energy / ELECTRON_MASS)


mpi = MpiPartition(1)
vmec = Vmec("./input.vacuum_template", mpi=mpi, keep_all_files=False, verbose=False)
vmec.boundary = surf

vmec = rescale_device(vmec, target_minor_radius=0.075, target_volavgB=0.0875)
vmec.run()

# Configure quasisymmetry objective:
qa_loss = QuasisymmetryRatioResidual(vmec,
                                np.linspace(0,1,10),  # Radii to target
                                helicity_m=1, helicity_n=0).total()  # (M, N) you want in |B|
proc0_print("\nQA error", qa_loss)
proc0_print('minor radius', vmec.boundary.minor_radius())
proc0_print('volavgB', vmec.wout.volavgB)
proc0_print('mean_iota', vmec.iota_edge())

# constraint violation
tmax = 1e-3
n_particles = 100
s_label = 0.05
tracing_tol=1e-8

tracer = TraceBoozer(
    vmec,
    tracing_tol=tracing_tol,
    interpolant_degree=3,
    interpolant_level=16,
    bri_mpol=16,
    bri_ntor=16,
)
tracer.sync_seeds()

stz_inits,_ = tracer.sample_surface(n_particles, s_label)
vpar_inits = speed * np.random.uniform(-1,1,size=(n_particles))

res_tys, c_times = tracer.compute_trajectories(stz_inits, vpar_inits, tmax, energy=energy, mass=ELECTRON_MASS,charge=ELEMENTARY_CHARGE)
#c_times = tracer.compute_confinement_times(stz_inits, vpar_inits, tmax, energy=energy, mass=ELECTRON_MASS,charge=ELEMENTARY_CHARGE)

proc0_print("loss frac", np.mean(c_times < tmax))
proc0_print("c_times/tmax", c_times/tmax)

if mpi.proc0_world:
    # save data 
    data = {}
    data['stz_inits'] = stz_inits
    data['vpar_inits'] = vpar_inits
    data['c_times'] = c_times
    data['res_tys'] = res_tys
    data['tmax'] = tmax
    data['s_label'] = s_label
    data['design_file'] = design_file
    data['current_group'] = current_group
    data['iota'] = iota
    data['energy_type'] = energy_type
    data['energy'] = energy
    outdir = "./output"
    tag = os.path.basename(design_file).removesuffix(".json")
    if not os.path.exists(outdir):
       os.makedirs(outdir)
    fname = os.path.join(outdir, f"tracing_data_{tag}_iota_{iota}.pkl")
    print("\nDumping to ", fname)
    pickle.dump(data, open(fname, "wb"))
    
