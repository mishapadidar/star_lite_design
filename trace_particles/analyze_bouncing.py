import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nfft import nfft,ndft_adjoint
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
from scipy.integrate import quad
ELECTRON_MASS = 9.1093837139e-31 # kg


"""
A script for analysis the bounce motion of particles.

Args:
    FILENAME (str): the path of a pickle file which contains a dictionary
    with the following features:
        c_times = data['c_times']
        tmax = data['tmax']
        trajectories = data["res_tys"]
        energy = data['energy']
    Generate such a file by running trace_electrons.py

"""


# Set these values directly:
FILENAME = "./output/tracing_data_designA_after_scaled_iota_0.18824534785229502_slow.pkl"
# FILENAME = "./output/tracing_data_designA_after_scaled_iota_0.18824534785229502_fast.pkl"

infile = Path(FILENAME)

with infile.open("rb") as f:
    data = pickle.load(f)

c_times = data['c_times']
tmax = data['tmax']
trajectories = data["res_tys"]
energy_type = data['energy_type']
iota = data['iota']
energy = data['energy']

vmax = np.sqrt(2 * data['energy']/ELECTRON_MASS)


fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(figsize=(15,10), nrows=2,ncols=3)

for i in range(len(trajectories)):
    # if c_times[i] < tmax - 1e-12:
    #     continue

    # look at lost particles only
    if c_times[i] >= tmax:
        continue

    traj = np.asarray(trajectories[i])
    t = traj[:, 0] #/ tmax
    s = traj[:, 1]
    theta = traj[:, 2] #% (2*np.pi)
    phi = traj[:, 3] #% (2*np.pi)
    vpar = traj[:, 4]

    spl_vpar = CubicSpline(t, vpar)
    spl_s = CubicSpline(t, s)
    spl_theta = CubicSpline(t, theta)

    # get left bounce bound
    idx_b1 = (vpar[:-1] >= 0) & (vpar[1:] < 0)
    idx_b2 = (vpar[:-1] <= 0) & (vpar[1:] > 0)
    idx_lb = np.logical_or((idx_b1), (idx_b2))
    idx_lb = np.where(idx_lb)[0]
    # right bounce bound
    idx_rb = idx_lb+1

    # compute bounce times
    n_bounce_pts = len(idx_lb)
    bounce_times = np.zeros(n_bounce_pts)
    for jj in range(n_bounce_pts):
        sol = root_scalar(spl_vpar, x0=t[idx_lb[jj]],xtol=1e-14)
        bounce_times[jj] = sol.root
    
    # compute state at bounce
    bounce_vpar = spl_vpar(bounce_times)
    bounce_s = spl_s(bounce_times)
    bounce_theta = spl_theta(bounce_times)

    # compute BAD
    bad_half = np.diff(bounce_s) # on half bounce
    if n_bounce_pts > 2:
        bad_full = bounce_s[2:] - bounce_s[:-2] # on half bounce

    # compute J_parallel
    if n_bounce_pts > 2:
        """
        Compute 
            J_parallel = int vpar dl = int vpar^2 dt
        where the integral is taken forward and back between 
        two bounce points.

        We compute the normalizied J_parallel, 
            J = J_parallel / vmax
        for numerical reasons.
        """
        J_parallel = np.zeros(n_bounce_pts-2)
        for jj in range(n_bounce_pts-2):
            tb0 = bounce_times[jj]
            tb1 = bounce_times[jj+2]
            # normalized j_parallel
            fint = lambda t: (spl_vpar(t))**2 / vmax
            sol = quad(fint, tb0, tb1, epsabs= 1e-13)
            J_parallel[jj] = sol[0]
        # times at which we measure J_parallel
        J_parallel_times = bounce_times[:-2]


    ax1.plot(np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), lw=0.8, alpha=0.9)
    ax1.scatter(np.sqrt(bounce_s) * np.cos(bounce_theta), np.sqrt(bounce_s) * np.sin(bounce_theta), marker='.', alpha=0.9)
    ax2.plot(t, s, lw=0.8, alpha=0.9)
    ax2.scatter(bounce_times, bounce_s, marker='.', alpha=0.9)
    ax3.plot(t, theta, lw=0.8, alpha=0.9)
    ax3.scatter(bounce_times, bounce_theta, marker='.', alpha=0.9)
    p=ax4.plot(t, vpar, lw=0.8, alpha=0.9)
    ax4.scatter(bounce_times, bounce_vpar, marker='.', alpha=0.9)
    if n_bounce_pts > 2:
        # plot diff(J_parallel)
        ax5.scatter(J_parallel_times[:-1],np.diff(J_parallel) / np.mean(J_parallel),
                    lw=0.8, alpha=0.9, color=p[-1].get_color())


circ = plt.Circle((0.0,0.0), 1.0, fill=False)
ax1.scatter([0],[0], color='k')
ax1.add_artist(circ)
ax1.set_xlim(-1.1,1.1)
ax1.set_ylim(-1.1,1.1)
ax1.set_xlabel("$\sqrt{s}\cos(\\theta)$")
ax1.set_ylabel("$\sqrt{s}\sin(\\theta)$")
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("$t$ [sec]")
ax2.set_ylabel("$s$")
ax2.grid(True, alpha=0.3)
ax3.set_xlabel("$t$ [sec]")
ax3.set_ylabel("$\\theta$")
ax3.grid(True, alpha=0.3)
ax4.set_xlabel("$t$ [sec]")
ax4.set_ylabel("$v_{par}$")
ax4.grid(True, alpha=0.3)
ax5.set_xlabel("$t$ [sec]")
ax5.set_ylabel("$\Delta J_{\parallel} / \langle J_{\parallel}\\rangle$")
ax5.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
