import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nfft import nfft,ndft_adjoint
from scipy.interpolate import CubicSpline

# Set these values directly:
FILENAME = "./output/tracing_data_designA_after_scaled_iota_0.18824534785229502_slow.pkl"
FILENAME = "./output/tracing_data_designA_after_scaled_iota_0.18824534785229502_fast.pkl"

infile = Path(FILENAME)

with infile.open("rb") as f:
    data = pickle.load(f)

c_times = data['c_times']
tmax = data['tmax']
trajectories = data["res_tys"]
energy_type = data['energy_type']
iota = data['iota']


fig,(ax1,ax2) = plt.subplots(figsize=(12,6), ncols=2)

for i in range(len(trajectories)):
    # if c_times[i] < tmax - 1e-12:
    #     continue
    if c_times[i] >= tmax:
        continue
    traj = np.asarray(trajectories[i])
    t = traj[:, 0] #/ tmax
    s = traj[:, 1]
    theta = traj[:, 2]
    theta = theta % (2*np.pi)
    phi = traj[:, 3]

    # spl = CubicSpline(t, s)
    # n = len(t)
    # t_unif = np.linspace(np.min(t), np.max(t), n, endpoint=False)
    # s_unif = spl(t_unif)

    # fhat = np.fft.fft(s_unif)
    
    # # get principal mode
    # largest_real_mode = np.max(fhat[1:n//2].real)
    # largest_imag_mode = np.max(fhat[1:n//2].imag)
    # largest_mode = max(largest_real_mode, largest_imag_mode)
    # # account for doubling, 1/n factor
    # largest_mode *= 2 / n
    # print(largest_mode)
    # # get freq
    # kn = np.fft.fftfreq(n, (np.max(t) - np.min(t)) / n)

    # idx_max = np.argmax(np.abs(fhat[1:])) + 1
    # f_mean = fhat[0].real / n
    # f_max = fhat[idx_max]
    # fhat = fhat * 0
    # fhat[idx_max] = f_max
    # s_test = np.fft.ifft(fhat)

    # plt.plot(np.abs(fhat[1:]))
    # plt.plot(fhat[:round(len())].imag / len(t_unif))

    ax1.plot(t, s, lw=0.8, alpha=0.9)
    ax2.plot(np.sqrt(s) * np.cos(theta), np.sqrt(s) * np.sin(theta), lw=0.8, alpha=0.9)

    # plt.plot(t, theta % (2*np.pi), lw=0.8, alpha=0.9)
    # plt.plot(t, phi, lw=0.8, alpha=0.9)

    # # plt.plot(t_unif, s_unif, lw=0.8, alpha=0.9)
    # plt.plot(t_unif, f_mean+s_test, lw=0.8, alpha=0.9)

    # plt.show()
    # if i == 2:
    #     quit()

circ = plt.Circle((0.0,0.0), 1.0, fill=False)
ax2.scatter([0],[0], color='k')
ax2.add_artist(circ)
ax2.set_xlim(-1.1,1.1)
ax2.set_ylim(-1.1,1.1)
ax2.set_xlabel("$\sqrt{s}\cos(\\theta)$")
ax2.set_ylabel("$\sqrt{s}\sin(\\theta)$")
ax2.grid(True, alpha=0.3)
ax2.set_title(f"Orbits of lost particle; $\iota$ = {iota:.2f}; energy = {energy_type}")

ax1.set_xlabel("$t$ [sec]")
ax1.set_ylabel("$s$")
ax1.grid(True, alpha=0.3)
ax1.set_title(f"Radial coordinate of lost particles; $\iota$ = {iota:.2f}; energy = {energy_type}")
plt.tight_layout()

plt.show()
