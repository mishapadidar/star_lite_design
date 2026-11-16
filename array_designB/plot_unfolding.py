#!/usr/bin/env python3
"""Histogram of the snowflake unfolding constant C written by mk_unfolding.py.

Usage:
    plot_unfolding.py <device_dir>

Reads <device_dir>/unfolding_C.txt (columns: seed, C, D_sigma) and writes
<device_dir>/unfolding_C_hist.png: the histogram of C over the Gaussian-process realizations
with a real split, its mean E[C], and on a second axis the X-point separation C * sqrt(sigma)
that the mean corresponds to at sigma = 1 mm, 5 mm, 1 cm and 2 cm. Only needs numpy and
matplotlib (venv_matplotlib).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1].rstrip("/")
rows = np.atleast_2d(np.loadtxt(os.path.join(d, "unfolding_C.txt")))
C = rows[:, 1]
real = C[np.isfinite(C)]
if not len(real):
    sys.exit(f"plot_unfolding: no realization with a real split in {d}/unfolding_C.txt")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.hist(real, bins=max(8, int(np.sqrt(len(real)) * 2)), color="#4878a8", edgecolor="white", linewidth=0.6)
ax.axvline(real.mean(), color="0.15", lw=1.4, label=f"mean E[C] = {real.mean():.3f}")
ax.axvline(np.median(real), color="0.15", lw=1.0, ls="--", label=f"median = {np.median(real):.3f}")
ax.set_xlabel(r"unfolding constant $C$  [m$^{1/2}$]   (X-point separation $= C\,\sqrt{\sigma}$)")
ax.set_ylabel("number of GP realizations")
ax.grid(alpha=0.3, axis="y")
text = "\n".join([f"{len(real)} of {len(C)} realizations split into a real pair",
                  f"std {real.std():.3f},  min {real.min():.3f},  max {real.max():.3f}",
                  "mean separation: " + ",  ".join(
                      f"{100 * real.mean() * np.sqrt(s):.1f} cm at {1000 * s:g} mm" for s in (1e-3, 5e-3, 1e-2, 2e-2))])
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title(os.path.basename(d)[-70:] + "\n" + text, fontsize=8)
fig.tight_layout()
out = os.path.join(d, "unfolding_C_hist.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(f"wrote {out}")

# Toroidal profiles of the mean and of the standard deviation of C (mk_unfolding.py --nphi):
# unfolding_C_profile.txt has one row per section: phi/2pi, E[C], std[C], median, n_real.
prof_path = os.path.join(d, "unfolding_C_profile.txt")
if os.path.isfile(prof_path):
    prof = np.atleast_2d(np.loadtxt(prof_path))
    phi, mean, std, nreal = prof[:, 0], prof[:, 1], prof[:, 2], prof[:, 4]
    x = phi / (phi[1] - phi[0]) / len(phi) if len(phi) > 1 else phi   # fraction of the field period
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 5.6), sharex=True)
    ms = 4 if len(phi) <= 32 else 0   # markers only for a coarse profile
    a1.plot(x, mean, "o-", color="#4878a8", lw=1.6, ms=ms)
    a1.set_ylabel(r"$E[C]$  [m$^{1/2}$]")
    a1.grid(alpha=0.3)
    a1.set_title(os.path.basename(d)[-70:] + f"\nsnowflake unfolding constant over {int(nreal.max())} GP realizations, "
                 f"{len(phi)} sections (X-point separation $= C\\,\\sqrt{{\\sigma}}$)", fontsize=8)
    a2.plot(x, std, "s-", color="#b5651d", lw=1.6, ms=ms)
    a2.set_ylabel(r"standard deviation of $C$  [m$^{1/2}$]")
    a2.set_xlabel(r"toroidal angle of the section over one field period, $\phi\, n_{fp} / 2\pi$")
    a2.set_xlim(0.0, 1.0)
    a2.grid(alpha=0.3)
    for a in (a1, a2):
        a.set_ylim(bottom=0.0)
    fig.tight_layout()
    out = os.path.join(d, "unfolding_C_profile.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
