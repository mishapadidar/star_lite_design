#!/usr/bin/env python

import os
import numpy as np
from simsopt.mhd import Vmec,j_dot_B_Redl, VmecRedlBootstrapMismatch, RedlGeomVmec
from simsopt.objectives import LeastSquaresProblem
from simsopt.solve import least_squares_mpi_solve
from simsopt.util import MpiPartition, proc0_print
from simsopt.mhd.profiles import ProfilePolynomial, ProfilePressure, ProfileScaled
from simsopt.util.constants import ELEMENTARY_CHARGE
import sys

"""
Optimize a VMEC equilibirium to have self-consistent bootstrap current.

The plasma boundary is fixed during the optimization.
The current profile is free to vary.

Use density and temperature profiles are fixed to be,
    n(s) = n0 * (1-s^5)
    T(s) = T0 * (1-s)
so that the pressure profile p(s) = n(s) * T(s).

We assume that n_i = n_e = n. T_i and T_e are left to be specified.

NOTE: prior to running this script, you should first generate the vacuum input files
    for design A and design B using e.g.
    >>> mpiexec -n 1 python vmec_free_boundary.py A 0.0 0.0

Args
-----
design (str): A or B to denote design A or design B.
n0 (float): density in units of 10^17 m^-3. Default to 5.
Ti0 (float): ion temperature in eV. Default to 1 eV.
Te0 (float): electron temperature in eV. Default to 10 eV.

Run with e.g.
    >>> mpiexec -n 1 python optimize_bootstrap.py A 5 1 10
"""

design = str(sys.argv[1])
n0 = float(sys.argv[2])
Ti0 = float(sys.argv[3])
Te0 = float(sys.argv[4])

proc0_print("")
proc0_print("Running with parameters", flush=True)
proc0_print("design", design, flush=True)
proc0_print(f"n0: {n0} x 10^17 [1/m^3]", flush=True)
proc0_print(f"Ti0: {Ti0} [eV]", flush=True)
proc0_print(f"Te0: {Te0} [eV]", flush=True)

# load the vacuum equilibrium file
mpi = MpiPartition()
vmec = Vmec(f"./output/input.design_{design}_beta_0.0_Imin_0.0", mpi=mpi, verbose=False)

# accelerate optimization
vmec.indata.ns_array[:3] = [16, 51, 0]
vmec.indata.niter_array[:3] = [2000, 3000, 0]
vmec.indata.ftol_array[:3] = [1e-10, 1e-11, 0]

# build the profiles
ne_raw = ProfilePolynomial(n0 * np.array([1, 0, 0, 0, 0, -1.0])) # n0 * (1 - s^5)
ne = ProfileScaled(ne_raw, 1e17) # convert to m^-3
ni = ne
Te_raw = ProfilePolynomial(Te0 * np.array([1, -1.0])) # Te0 * (1 - s) # in eV
Ti_raw = ProfilePolynomial(Ti0 * np.array([1, -1.0])) # Ti0 * (1 - s) # in eV
Te = ProfileScaled(Te_raw, ELEMENTARY_CHARGE) # convert to Joules
Ti = ProfileScaled(Ti_raw, ELEMENTARY_CHARGE) # convert to Joules
pressure_profile = ProfilePressure(ne, Te, ni, Ti)  # p = ne * Te + ni * Ti
vmec.pressure_profile = pressure_profile

# set the current profile
current_profile = ProfilePolynomial(np.array([0, 5, -5, 0])) # I(s) = 0
vmec.current_profile = current_profile

# compute the plasma beta
pavg = np.mean([pressure_profile(s) for s in np.linspace(0, 1, 100)])
B0 = 0.0875 #T
mu0 = 4e-7 * np.pi
beta = 2 * mu0 * pavg / B0**2
proc0_print("\nbeta =", beta * 100, "%")

# pressure.plot()
# quit()

# only free the current profile
vmec.boundary.fix_all()
pressure_profile.fix_all()
current_profile.unfix_all()

# bootstrap current objective
s1d = np.linspace(0, 1, 11, endpoint=False)[1:]  # exclude s=0 to avoid singularities
geom = RedlGeomVmec(vmec,surfaces=s1d)
obj = VmecRedlBootstrapMismatch(geom, ne, Te_raw, Ti_raw, Zeff=1, helicity_n=0)

mean_iota = vmec.mean_iota()
proc0_print('mean_iota', mean_iota)
J = obj.J()
proc0_print('obj', J)


# Define objective function
prob = LeastSquaresProblem.from_tuples([(obj.residuals, 0.0, 1)])

# Make sure all procs participate in computing the objective:
prob.objective()

least_squares_mpi_solve(prob, mpi, grad=True, rel_step=1e-5, abs_step=1e-8, max_nfev=30)

mean_iota = vmec.mean_iota()
proc0_print('mean_iota', mean_iota)
J = obj.J()
proc0_print('obj', J)

# compare the bootstrap current computations
jb_redl, _ = j_dot_B_Redl(ne, Te_raw, Ti_raw, Zeff=1, helicity_n=0, geom=geom, plot=False)
from scipy.interpolate import interp1d
interp = interp1d(vmec.s_full_grid, vmec.wout.jdotb)
s_fine = np.linspace(0, 1, 100)
jb_vmec = interp(s_fine)
if mpi.proc0_world:
    import matplotlib.pyplot as plt
    plt.plot(s_fine, jb_vmec, label='VMEC')
    plt.plot(s1d, jb_redl, 'o', label='REDL')
    plt.legend(loc='upper right')
    plt.savefig(f'optimized_bootstrap_current_design_{design}_n0_{n0}_Ti0_{Ti0}_Te0_{Te0}.pdf', format='pdf')
    # plt.show()
