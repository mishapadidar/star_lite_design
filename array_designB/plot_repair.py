#!/usr/bin/env python3
"""Figure of the snowflake repair study written by mk_repair.py.

Usage:
    plot_repair.py <device_dir>

Reads <device_dir>/repair_fraction.txt (threshold on the largest |aux current| [kA], then the
fraction of the Gaussian-process realizations repaired below it, one column per perturbation
amplitude sigma) and repair_currents.txt (one row per realization and sigma), and writes
<device_dir>/repair_fraction.png: for each sigma, the fraction of the realizations whose
snowflake is recovered with every aux current below the abscissa. A curve that levels off
below 1 means that the remaining realizations could not be repaired at any current.
Only needs numpy and matplotlib (venv_matplotlib).
"""
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1].rstrip("/")
path = os.path.join(d, "repair_fraction.txt")
with open(path) as f:
    header = [line for line in f if line.startswith("#")]
sigmas = re.findall(r"sigma=([0-9.eE+-]+)mm", header[-1])
data = np.atleast_2d(np.loadtxt(path))
grid, frac = data[:, 0], data[:, 1:]

cur = np.atleast_2d(np.loadtxt(os.path.join(d, "repair_currents.txt")))
with open(os.path.join(d, "repair_currents.txt")) as f:
    first = f.readline()
m = re.search(r"N = (\d+); nominal largest \|I\| = ([0-9.]+) kA", first)
n_aux, nominal = (int(m.group(1)), float(m.group(2))) if m else (None, None)
nsamples = len(np.unique(cur[:, 0]))

colors = ["#2b6a99", "#3a9a6b", "#d08a1c", "#b23a3a", "#6a4c93", "#555555"]
xmax = 40.0   # kA: the same current range on every device, so that the figures compare by eye
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for j, s in enumerate(sigmas):
    sel = cur[np.isclose(cur[:, 1], float(s))]
    nrep = int(sel[:, 2].sum())
    # Continuous curve: the empirical distribution of the largest |current| needed, drawn
    # through the realizations themselves (one point per repaired realization, joined by
    # straight lines) instead of a staircase on a grid of thresholds; it starts at 0 at the
    # smallest current needed and stays flat beyond the largest one.
    need = np.sort(sel[sel[:, 2] > 0, 4])
    if len(need):
        x = np.r_[need[0], need, max(xmax, need[-1])]
        y = np.r_[0.0, np.arange(1, len(need) + 1) / len(sel), len(need) / len(sel)]
        ax.plot(x, y, lw=1.8, color=colors[j % len(colors)],
                label=f"$\\sigma$ = {float(s):g} mm  ({nrep} of {len(sel)} repaired)")
    else:
        ax.plot([], [], lw=1.8, color=colors[j % len(colors)], label=f"$\\sigma$ = {float(s):g} mm  (none repaired)")
if nominal is not None:
    ax.axvline(nominal, color="0.35", ls="--", lw=1.0)
    ax.text(nominal, 1.02, f"nominal {nominal:.1f} kA", ha="center", va="bottom", fontsize=8, color="0.25")
ax.set_xlabel("largest |aux current| allowed  [kA]")
ax.set_ylabel("fraction of realizations repaired")
ax.set_xlim(0.0, xmax)
ax.set_ylim(0.0, 1.08)
ax.grid(alpha=0.3)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title(os.path.basename(d)[-70:] + f"\nsnowflake recovered with the aux currents alone"
             + (f" (N = {n_aux})" if n_aux else "") + f", {nsamples} GP perturbations of the modular coils", fontsize=8)
fig.tight_layout()
out = os.path.join(d, "repair_fraction.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
