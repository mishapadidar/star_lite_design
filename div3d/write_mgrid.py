#!/usr/bin/env python

import numpy as np
from simsopt.mhd import Vmec, ProfilePolynomial
from simsopt.util import MpiPartition
from simsopt._core import load
from simsopt.util import proc0_print
import os, sys
from simsopt.geo import ToroidalFlux


"""
Write an mgrid file for div3d.
The mgrid file should we wide enough to enclose the plasma and X-point.

Run with e.g.
    >>> mpiexec -n 1 python vmec_free_boundary.py design

Args
----
design (str): A or B to denote design A or design B.

"""

debug = False

design = str(sys.argv[1])

proc0_print("")
proc0_print("Running with parameters", flush=True)
proc0_print("design", design, flush=True)

if design == "A":
    infile = "../designs/designA_after_scaled.json"
else:
    infile = "../designs/designB_after_forces_opt_19.json"

mpi = MpiPartition(ngroups=1)

# load the boozer surfaces (1 per Current configuration, so 3 total.)
data = load(f"../designs/design{design}_after_scaled.json")
bsurf = data[0][0] # BoozerSurfaces
surf = bsurf.surface
biotsavart = bsurf.biotsavart
ma = data[2][0] # magnetic axis CurveRZFourier
xpoint = data[3][0] # X-point CurveRZFouriers


# Get the plasma boundary surface
xyz = surf.gamma()
r = np.sqrt(xyz[:, :, 0]**2 + xyz[:, :, 1]**2)
z = xyz[:, :, 2]
proc0_print(f"Plasma bounds:", flush=True)
proc0_print(f"rmin = {np.min(r)}, rmax = {np.max(r)}, zmin = {np.min(z)}, zmax = {np.max(z)}", flush=True)

# get the x-point bounds
r_xpoint = np.sqrt(xpoint.gamma()[:, 0]**2 + xpoint.gamma()[:, 1]**2)
z_xpoint = xpoint.gamma()[:, 2]
proc0_print(f"X-point bounds:", flush=True)
proc0_print(f"rmin = {np.min(r_xpoint)}, rmax = {np.max(r_xpoint)}, zmin = {np.min(z_xpoint)}, zmax = {np.max(z_xpoint)}", flush=True)

# expand grid slightly
r = np.concatenate((r.flatten(), r_xpoint))
z = np.concatenate((z.flatten(), z_xpoint))
expansion_factor = 0.2
rmin = (1 - expansion_factor) * np.min(r)
rmax = (1 + expansion_factor) * np.max(r)
zmin = (1 + expansion_factor) * np.min(z)
zmax = (1 + expansion_factor) * np.max(z)

proc0_print(f"Expanded grid to:", flush=True)
proc0_print(f"rmin = {rmin}, rmax = {rmax}, zmin = {zmin}, zmax = {zmax}", flush=True)

# Grid density; nphi is points per field period
# using values for Hudson et al.
if debug:
    nphi = 32
    nr = 32
    nz = 32
else:
    nphi = 512
    nr   = 512 
    nz   = 512 

# write the mgrid
outdir = "./output/"
os.makedirs(outdir, exist_ok=True)
tag = f"design{design}_inlucing_xpoint"
mgrid_file = outdir + "mgrid." + tag + ".nc"
proc0_print(f"Writing.mgrid file", flush=True)
if mpi.proc0_world:
    biotsavart.to_mgrid(
        mgrid_file,
        nr=nr,
        nz=nz,
        nphi=nphi,
        rmin=rmin,
        rmax=rmax,
        zmin=zmin,
        zmax=zmax,
        nfp=surf.nfp,
    )

proc0_print(f"Done", flush=True)
