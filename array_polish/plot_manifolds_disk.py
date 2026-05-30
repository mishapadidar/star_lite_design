#!/usr/bin/env python3
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys

p = Path(sys.argv[1])

# Categorical colour scheme: each integer label (first column of the poincare
# files) maps to a fixed colour, so dots of the same label share a colour
# consistently across every scatter call and panel.
_CAT_CMAP = plt.get_cmap('tab20')
def _cat_colors(labels):
    return _CAT_CMAP(np.asarray(labels).astype(int) % _CAT_CMAP.N)

phis  = np.linspace(0, 0.25, 9)
nphi  = len(phis)

fig, axes = plt.subplots(3, 3, figsize=(8.25, 10), sharex=True, sharey=True, gridspec_kw={"wspace": 0, "hspace": 0})

for i, ax in enumerate(axes.flat):
    # manifolds
    files = (glob.glob(str(p.parent / f"poincare{i}_*.txt"))
             + glob.glob(str(p.parent / f"poincare_*_{i}_*.txt")))
    for f in sorted(files):
        d = np.loadtxt(f, delimiter=",")
        if d.size:
            if d.ndim == 1: d = d[None, :]
            # drop first hit per fieldline id (first column)
            _, first_idx = np.unique(d[:, 0], return_index=True)
            mask = np.ones(d.shape[0], dtype=bool)
            mask[first_idx] = False
            d = d[mask]
            if d.size == 0:
                continue
            R, Z = np.hypot(d[:,1], d[:,2]), d[:,3]
            ax.scatter(R, Z, s=0.1, color=_cat_colors(d[:, 0]), rasterized=True)

    # interior surfaces
    f_int = p.parent / f"poincare_interior_{i}.txt"
    if os.path.exists(f_int):
        d = np.loadtxt(f_int, delimiter=",")
        if d.size:
            if d.ndim == 1: d = d[None, :]
            _, first_idx = np.unique(d[:, 0], return_index=True)
            mask = np.ones(d.shape[0], dtype=bool)
            mask[first_idx] = False
            d = d[mask]
            if d.size:
                R, Z = np.hypot(d[:,1], d[:,2]), d[:,3]
                ax.scatter(R, Z, s=0.1, color=_cat_colors(d[:,0]), rasterized=True)

    # inward axis->origin fieldlines
    f_inw = p.parent / f"poincare_inward_{i}.txt"
    if os.path.exists(f_inw):
        d = np.loadtxt(f_inw, delimiter=",")
        if d.size:
            if d.ndim == 1: d = d[None, :]
            _, first_idx = np.unique(d[:, 0], return_index=True)
            mask = np.ones(d.shape[0], dtype=bool)
            mask[first_idx] = False
            d = d[mask]
            if d.size:
                R, Z = np.hypot(d[:,1], d[:,2]), d[:,3]
                ax.scatter(R, Z, s=0.1, color=_cat_colors(d[:,0]), rasterized=True)

    # vessel cross-section
    f_v = p.parent / f"vessel_cross_{i}.txt"
    if os.path.exists(f_v):
        v = np.loadtxt(f_v, delimiter=',', skiprows=1)
        if v.size:
            if v.ndim == 1: v = v[None, :]
            ax.plot(v[:, 0], v[:, 1], 'k-', lw=1.0, label='vessel')

    # optimization surface cross-section
    f_v = p.parent / f"surface_cross_{i}.txt"
    if os.path.exists(f_v):
        v = np.loadtxt(f_v, delimiter=',', skiprows=1)
        if v.size:
            if v.ndim == 1: v = v[None, :]
            ax.plot(v[:, 0], v[:, 1], 'r--', lw=1.0, label='optimization surface')
    # magnetic axis (row 0) and X-point (row 1)
    f_fp = p.parent / f"fixed_points_{i}.txt"
    if os.path.exists(f_fp):
        fp = np.loadtxt(f_fp, delimiter=',', skiprows=1)
        if fp.size:
            fp = np.atleast_2d(fp)
            ax.plot(fp[0, 0], fp[0, 1], marker='o', color='tab:green',
                    ms=4, ls='none', label='axis')
            if fp.shape[0] > 1:
                ax.plot(fp[1, 0], fp[1, 1], marker='X', color='tab:red',
                        ms=6, ls='none', label='X-point')

    # invariant-direction rays through the X-point, only at phi=0
    if i == 0:
        f_legs = p.parent / "legs.txt"
        f_fp0 = p.parent / "fixed_points_0.txt"
        if os.path.exists(f_legs) and os.path.exists(f_fp0):
            fp0 = np.atleast_2d(np.loadtxt(f_fp0, delimiter=',', skiprows=1))
            if fp0.shape[0] > 1:
                R_xp, Z_xp = fp0[1, 0], fp0[1, 1]          # row1 = X-point
                # legs.txt cols: k, theta, vR, vZ, kind, sign, c (skip string col)
                L = np.atleast_2d(np.loadtxt(f_legs, delimiter=',', usecols=(2, 3)))
                seen = set()
                for vR, vZ in L:
                    ang = round(np.arctan2(vZ, vR) % np.pi, 6)   # dedupe +-v lines
                    if ang in seen:
                        continue
                    seen.add(ang)
                    ax.axline((R_xp, Z_xp), (R_xp + vR, Z_xp + vZ),
                              color='k', lw=0.5, alpha=0.3, zorder=0)

    ax.text(
        0.5, 0.98,
        rf"$\phi/2\pi = {phis[i]:.4f}$",
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=8
    )
    ax.set_aspect('equal')

axes[0,0].set_xlim([0, 1])
axes[0,0].set_ylim([-0.5, 0.5])

for ax in axes[-1, :]: ax.set_xlabel('R')
for ax in axes[:, 0]:  ax.set_ylabel('Z')

h, l = [], []
for a in axes.flat:
    hh, ll = a.get_legend_handles_labels()
    h += hh; l += ll
uniq = dict(zip(l, h))
fig.legend(
    uniq.values(),
    uniq.keys(),
    loc='center left',
    bbox_to_anchor=(0.5, 0.95),
    frameon=False,
    markerscale=12
)

parent_name = p.parent.name
grandparent_name = p.parent.parent.name
plt.savefig(p.parent / f'xs_{grandparent_name}_{parent_name}.png', dpi=300)
