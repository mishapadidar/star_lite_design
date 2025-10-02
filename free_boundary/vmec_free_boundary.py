#!/usr/bin/env python

import numpy as np
from simsopt.mhd import Vmec, ProfilePolynomial
from simsopt.util import MpiPartition
from simsopt._core import load
from simsopt.util import proc0_print
import os, sys
from simsopt.geo import ToroidalFlux


"""
Compute the star-lite configurations with non-zero pressure/current using 
free-boundary VMEC.

We specify the profiles,
    p(r) = p0 + p1 * s
    I(r) = I1 * s + I2 * s^2.
we choose p0 and I2 so that the current and pressure are zero on the boundary,
    p0 = -p1.
    I2 = -I1.
For the current profile to be a convex quadratic, I1 should be negative. The minimum of the current
profile is always at s=1/2, with value Imin = I1/4.

This script takes two inputs, beta and Imin. beta is the non-dimensionalized pressure. We then specify the 
profiles via,
    I1 = -I2 = 4Imin
    p0 = -p1 = beta * B0^2 / mu0

Run with e.g.
    >>> mpiexec -n 1 python vmec_free_boundary.py beta Imin

NOTE: Make sure you are using the mp_mgrid_fix branch of SIMSOPT.

Args
----
design (str): A or B to denote design A or design B.
beta (float): The non-dimensionalized pressure as a percent. Should be in [0, 5] typically. Use 0.0
    for vacuum.
Imin (float): minimum of the toroidal current. Should be non-positive. Typical values are on [0, -5].
    Use 0.0 for vacuum.

"""

debug = False

design = str(sys.argv[1])
beta = float(sys.argv[2])
Imin = float(sys.argv[3])

proc0_print("")
proc0_print("Running with parameters", flush=True)
proc0_print("design", design, flush=True)
proc0_print("beta [%]", beta, flush=True)
proc0_print("Imin [A]", Imin, flush=True)


"""
Create VMEC input file
"""

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

# get the profile parameters
I1 = 4 * Imin
I2 = - I1
xyz = ma.gamma()
biotsavart.set_points(xyz)
modB = biotsavart.AbsB()
B0 = np.mean(modB)
mu0 = np.pi * 4e-7
beta_not_percent = beta / 100
p0 = beta_not_percent * B0**2 / mu0
p1 = - p0
proc0_print('p0', p0)
proc0_print('p1', p1)
proc0_print('I1', I1)
proc0_print('I2', I2)


# Create a VMEC object from an input file:
outdir = "./output/"
os.makedirs(outdir, exist_ok=True)
tag = f"design_{design}_beta_{beta}_Imin_{Imin}"
vmec_input = outdir + f"input." + tag

vmec = Vmec(mpi=mpi, verbose=True)
vmec.boundary = surf

tflux = ToroidalFlux(surf, biotsavart)
phiedge = tflux.J()
minor_radius = surf.minor_radius()

# speed/fidelity parameters
vmec.indata.delt = 0.7
if debug:
    vmec.indata.ns_array[:2] = [16, 0]
    vmec.indata.niter_array[:2] = [2000, 0]
    vmec.indata.ftol_array[:2] = [1e-10, 0]
else:
    vmec.indata.ns_array[:3] = [16, 51, 101]
    vmec.indata.niter_array[:3] = [2000, 3000, 15000]
    vmec.indata.ftol_array[:3] = [1e-15, 1e-15, 1e-13]
# how often to print
vmec.indata.nstep = 200
# grid parameters
vmec.indata.mpol = surf.mpol
vmec.indata.ntor = surf.ntor
vmec.indata.nfp = surf.nfp
vmec.indata.lasym = not surf.stellsym
vmec.indata.phiedge = phiedge
vmec.indata.ntheta = 0 # automatically choose theta discretization
vmec.indata.nzeta = 0 # automatically choose zeta discretization

# pressure profile
vmec.indata.pmass_type = "power_series"
vmec.indata.am[:2] = [p0, p1] # power series coeffs in s
vmec.indata.pres_scale = 1.0

# current profile
mu0 = np.pi * 4e-7
curtor = I1 / 6 # integral of I_T(s)
vmec.indata.curtor = curtor # total current I(s=1)
vmec.indata.ac[:3] = [0.0, I1, I2] # power series coeffs in s
vmec.indata.pcurr_type = "power_series"
vmec.indata.ncurr = 1 # use current profile (not iota)
# current_profile = ProfilePolynomial([0.0, coeff, -coeff])
# vmec.current_profile = current_profile


proc0_print("Writing VMEC input file", flush=True)
if mpi.proc0_world:
    vmec.write_input(vmec_input)
proc0_print("Wrote", vmec_input, flush=True)

# make sure all ranks have the same input
vmec = Vmec(vmec_input, mpi=mpi, verbose=True)

mpi.comm_world.Barrier()
# vmec.boundary.to_vtk(input_file = indir + tag)


# run the fixed boundary
vmec.run()

""" 
The mgrid is a grid in cylindrical coordinates (R, phi, Z) on which 
the magnetic field is calculated. The grid should enclose the plasma.
nphi is the number of points per field period. The VMEC code uses this
grid to 'deform' the plasma boundary in a iterative sense. The grid
density should be chosen appropriately to resolve the plasma boundary.
A good rule of thumb is nphi >= 4*ntor. To get centimeter accuracy make sure
the grid density is a centimeter or less.
"""
# Get the plasma boundary surface
xyz = vmec.boundary.gamma()
r = np.sqrt(xyz[:, :, 0]**2 + xyz[:, :, 1]**2)
proc0_print(f"Plasma bounds:", flush=True)
proc0_print(f"rmin = {np.min(r)}, rmax = {np.max(r)}, zmin = {np.min(xyz[:, :, 2])}, zmax = {np.max(xyz[:, :, 2])}", flush=True)

# expand grid slightly to enclose the plasma
expansion_factor = 0.4
rmin = (1 - expansion_factor) * np.min(r)
rmax = (1 + expansion_factor) * np.max(r)
zmin = (1 + expansion_factor) * np.min(xyz[:, :, 2])
zmax = (1 + expansion_factor) * np.max(xyz[:, :, 2])

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
mpi.comm_world.Barrier()
proc0_print(f"Done", flush=True)



# That input file was for fixed-boundary. We need to change some of 
# the vmec input parameters for a free-boundary calculation:
vmec.indata.lfreeb = True
vmec.indata.mgrid_file = mgrid_file
vmec.indata.nzeta = nphi
# All the coils are written into a single "current group", so we only need to
# set a single entry in vmec's "extcur" array:
vmec.indata.extcur[0] = 1.0

# for better axis refinement
vmec.indata.lforbal = True


if debug:
    # low res, for testing
    vmec.indata.mpol = 6
    vmec.indata.ntor = 6
    vmec.indata.ns_array[:2] = [16, 0]
    vmec.indata.niter_array[:2] = [2000, 0]
    vmec.indata.ftol_array[:2] =[1e-10, 1e-10]
else:
    vmec.indata.ns_array[:3] = [32, 84, 201]
    vmec.indata.niter_array[:3] = [2000, 5000, 30000]
    vmec.indata.ftol_array[:3] =[1e-14, 1e-13, 5e-14]

proc0_print("")
proc0_print("Running free-boundary VMEC", flush=True)

vmec.need_to_run_code = True
vmec.run()

if mpi.proc0_world:
    save_to = outdir + f"input." + tag + "_free_boundary_solution"
    proc0_print("Saving boundary shape to", save_to)
    vmec.write_input(save_to)



