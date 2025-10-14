import numpy as np
from simsopt._core import load
from simsopt.mhd import Vmec, vmec_compute_geometry, Quasisymmetry, QuasisymmetryRatioResidual,Boozer
import sys, os

"""
Plot the result of the free boundary computation e.g.
    python generate_plot_data.py design beta Imin

First, run vmec_free_boundary.py and save the wout files in output/.

Args
----
design (str): A or B to denote design A or design B.
beta (float): The non-dimensionalized pressure as a percent. Should be in [0, 5] typically. Use 0.0
    for vacuum.
Imin (float): minimum of the toroidal current. Should be non-positive. Typical values are on [0, -5].
    Use 0.0 for vacuum.
"""

design = str(sys.argv[1])
beta = float(sys.argv[2])
Imin = float(sys.argv[3])

print("")
print("Running with parameters", flush=True)
print("design", design, flush=True)
print("beta [%]", beta, flush=True)
print("Imin [A]", Imin, flush=True)

# load the device
indir = "./output/"
if design == "A":
    infile = "../designs/designA_after_scaled.json"
else:
    infile = "../designs/designB_after_forces_opt_19.json"

# load the boozer surfaces (1 per Current configuration, so 3 total.)
data = load(f"../designs/design{design}_after_scaled.json")
bsurf = data[0][0] # BoozerSurfaces
surf = bsurf.surface
biotsavart = bsurf.biotsavart
ma = data[2][0] # magnetic axis CurveRZFourier

free_boundary_file = indir + f"wout_design_{design}_beta_{beta}_Imin_{Imin}_free_boundary.nc"
eq_free = Vmec(free_boundary_file)

nfp = surf.nfp
s1d = np.linspace(1e-6, 1, 32)
theta1d = np.linspace(0, 2 * np.pi, 32, endpoint=False)
phi1d = np.linspace(0, 2 * np.pi / nfp, 32, endpoint=False)
data_free = vmec_compute_geometry(eq_free, s=s1d, theta=theta1d, phi=phi1d)

helicity_n = 0
s_qs = np.logspace(-2, 0, 64)
boozer = Boozer(eq_free, mpol=32, ntor=32)
qs_err = Quasisymmetry(boozer, s_qs, helicity_m=1, helicity_n=helicity_n, normalization="symmetric", weight="stellopt_ornl").J()

plot_data = {}
plot_data['iota_free'] = -data_free.iota
plot_data['s_iota_free'] = s1d
plot_data['qs_err_free'] = qs_err
plot_data['s_qs_err_free'] = s_qs
plot_data['design'] = design
plot_data['beta'] = beta
plot_data['Imin'] = Imin

import pickle
outdir = "./plot_data/"
os.makedirs(outdir, exist_ok=True)
with open(outdir + f'plot_data_design_{design}_beta_{beta}_Imin_{Imin}.pkl', 'wb') as f:
    pickle.dump(plot_data, f)