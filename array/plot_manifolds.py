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

def _scatter_file(ax, path, dot_size):
    """Plot a poincare file's dots (drop the first hit per fieldline id)."""
    if not os.path.exists(path):
        return
    d = np.loadtxt(path, delimiter=",")
    if not d.size:
        return
    if d.ndim == 1:
        d = d[None, :]
    _, first_idx = np.unique(d[:, 0], return_index=True)
    mask = np.ones(d.shape[0], dtype=bool)
    mask[first_idx] = False
    d = d[mask]
    if d.size == 0:
        return
    R, Z = np.hypot(d[:, 1], d[:, 2]), d[:, 3]
    ax.scatter(R, Z, s=dot_size, color=_cat_colors(d[:, 0]), rasterized=True)


def draw_panel(ax, i, dot_size=0.1, leg_alpha=0.3, leg_lw=0.5, leg_zorder=0):
    """Draw all overlays (manifolds, interior, vessel, surface, fixed points,
    invariant lines) for phi index i onto ax. The invariant-line style
    (leg_alpha/leg_lw/leg_zorder) is faint+behind by default; the zoom inset
    passes a prominent style so the lines aren't buried under the leg dots."""
    # manifolds (both naming conventions)
    for f in sorted(glob.glob(str(p.parent / f"poincare{i}_*.txt"))
                    + glob.glob(str(p.parent / f"poincare_*_{i}_*.txt"))):
        _scatter_file(ax, f, dot_size)
    _scatter_file(ax, p.parent / f"poincare_interior_{i}.txt", dot_size)
    _scatter_file(ax, p.parent / f"poincare_inward_{i}.txt", dot_size)

    f_v = p.parent / f"vessel_cross_{i}.txt"
    if os.path.exists(f_v):
        v = np.atleast_2d(np.loadtxt(f_v, delimiter=',', skiprows=1))
        if v.size:
            ax.plot(v[:, 0], v[:, 1], 'k-', lw=1.0, label='vessel')

    f_v = p.parent / f"surface_cross_{i}.txt"
    if os.path.exists(f_v):
        v = np.atleast_2d(np.loadtxt(f_v, delimiter=',', skiprows=1))
        if v.size:
            ax.plot(v[:, 0], v[:, 1], 'r--', lw=1.0, label='optimization surface')

    f_fp = p.parent / f"fixed_points_{i}.txt"
    if os.path.exists(f_fp):
        fp = np.atleast_2d(np.loadtxt(f_fp, delimiter=',', skiprows=1))
        if fp.size:
            ax.plot(fp[0, 0], fp[0, 1], marker='o', color='tab:green',
                    ms=4, ls='none', label='axis')
            if fp.shape[0] > 1:
                ax.plot(fp[1, 0], fp[1, 1], marker='X', color='tab:red',
                        ms=3, ls='none', label='X-point')

    # invariant-direction lines through the X-point, only at phi=0
    if i == 0:
        f_legs = p.parent / "legs.txt"
        f_fp0 = p.parent / "fixed_points_0.txt"
        if os.path.exists(f_legs) and os.path.exists(f_fp0):
            fp0 = np.atleast_2d(np.loadtxt(f_fp0, delimiter=',', skiprows=1))
            if fp0.shape[0] > 1:
                R_xp, Z_xp = fp0[1, 0], fp0[1, 1]
                L = np.atleast_2d(np.loadtxt(f_legs, delimiter=',', usecols=(2, 3)))
                seen = set()
                for vR, vZ in L:
                    ang = round(np.arctan2(vZ, vR) % np.pi, 6)   # dedupe +-v lines
                    if ang in seen:
                        continue
                    seen.add(ang)
                    ax.axline((R_xp, Z_xp), (R_xp + vR, Z_xp + vZ),
                              color='k', lw=leg_lw, alpha=leg_alpha, zorder=leg_zorder)


for i, ax in enumerate(axes.flat):
    draw_panel(ax, i)
    ax.text(0.5, 0.98, rf"$\phi/2\pi = {phis[i]:.4f}$",
            transform=ax.transAxes, ha='center', va='top', fontsize=8)
    ax.set_aspect('equal')

axes[0, 0].set_xlim([0, 1])
axes[0, 0].set_ylim([-0.5, 0.5])

for ax in axes[-1, :]: ax.set_xlabel('R')
for ax in axes[:, 0]:  ax.set_ylabel('Z')

# Zoom inset on the X-point in every phi panel to reveal the fixed-point structure.
ZOOM_HALF = 0.01
for i, ax in enumerate(axes.flat):
    f_fp = p.parent / f"fixed_points_{i}.txt"
    if not os.path.exists(f_fp):
        continue
    fp = np.atleast_2d(np.loadtxt(f_fp, delimiter=',', skiprows=1))
    if fp.shape[0] < 2:
        continue
    R_xp, Z_xp = fp[1, 0], fp[1, 1]
    axins = ax.inset_axes([0.66, 0.66, 0.32, 0.32])
    # In the zoom, draw the invariant lines prominently and on top of the dots.
    draw_panel(axins, i, dot_size=1.0, leg_alpha=0.8, leg_lw=1.0, leg_zorder=5)
    axins.set_xlim(R_xp - ZOOM_HALF, R_xp + ZOOM_HALF)
    axins.set_ylim(Z_xp - ZOOM_HALF, Z_xp + ZOOM_HALF)
    axins.set_aspect('equal')
    axins.set_xticks([]); axins.set_yticks([])
    ax.indicate_inset_zoom(axins, edgecolor='0.4', lw=0.8)

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
    markerscale=2,
)

parent_name = p.parent.name
grandparent_name = p.parent.parent.name
plt.savefig(p.parent / f'xs_{grandparent_name}_{parent_name}.png', dpi=300)
