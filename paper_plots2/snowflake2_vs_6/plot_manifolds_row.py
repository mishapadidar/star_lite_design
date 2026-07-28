#!/usr/bin/env python3
"""Plot two mk_manifolds.py datasets side by side in a 1x2 row with NO space between panels and ALL
axes shared, using the same per-panel formatting as plot_manifolds.py. Neither panel has a zoom inset.

Usage:
    plot_manifolds_row.py <left.txt> <right.txt>    (each a *_allmanifolds.txt or a design json)

Writes <left_stem>_row.png next to the left file.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.spatial import cKDTree

# reuse the exact formatting (parsers, colours, marker table, sizes) from the single-panel plotter, so
# each panel matches plot_manifolds.py.
from plot_manifolds import (_headers, _KIND_COLOR, LEG_COLOR, SURF_COLOR, DOT_SIZE, O_MARKER_SCALE,
                            TICK_STEP, _FP_MARKER)

TRACKED_COLOR = "tab:purple"                                # marker colour for the tracked fixed point
FP_SCALE = 1.5                                              # enlarge all fixed-point markers by this factor
WIN_XLIM = (0.2724126483084716, 0.7320413706766207)        # shared main-window R (x) limits
WIN_YLIM = (-0.13278252602760246, 0.3223837815602734)      # shared main-window Z (y) limits

# Density filter: escaped / wandering orbits leave isolated scattered dots that clutter the panels,
# while the separatrix legs and nested O-surfaces are dense curves. We keep a dot only if enough of its
# own kind sit within DENS_RADIUS. Inspired by paper_plots2/parabolic_example/run_all.py's
# density_filter, but radius-based (rather than a fixed histogram grid) so it never punches gaps in the
# curves the way a grid does when a dense curve straddles a cell boundary.
DENSITY_FILTER = True                                      # set False to draw every dot (old behaviour)
# per-panel (radius [m], min_count) for the density test -- tune each panel independently. A smaller
# radius and/or larger min_count filters more aggressively (see _density_mask).
DENS_A = (0.010, 2)                                        # panel A (left)
DENS_B = (0.006, 1)                                        # panel B (right)


def _density_mask(pts, radius, min_count):
    """Boolean keep-mask over pts (an (N, 2) array of (R, Z)): True where at least min_count OTHER
    points lie within radius. Distances are Euclidean in (R, Z), valid because the panels use equal
    aspect. Dense curves survive; isolated scattered dots are dropped."""
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)
    tree = cKDTree(pts)
    counts = tree.query_ball_point(pts, radius, return_length=True) - 1   # exclude self
    return counts >= min_count


def _resolve(arg):
    s = str(arg)
    if s.endswith(".txt"):
        return Path(s)
    cand = Path(s + "_allmanifolds.txt")                    # a design-json stem
    return cand if cand.exists() else Path(s + ".txt")      # else an *_allmanifolds stem


def _draw_fn(txt, tracked_type=None, dens=DENS_A):
    """Return draw(ax, fp_scale) rendering txt's manifold dots + fixed-point markers (the exact
    plot_manifolds.py rendering) into any axes. If tracked_type is given ('hyperbolic' / 'elliptic'),
    the TRACKED fixed point -- the lowest-Z point of that type, excluding the magnetic axis (the point
    nearest the header ref) -- is coloured TRACKED_COLOR. dens = (radius, min_count) is this panel's
    density-filter setting (see _density_mask)."""
    fps, legkinds, ref = _headers(txt)
    arr = np.atleast_2d(np.loadtxt(txt, delimiter=","))
    if not arr.size:
        raise SystemExit(f"{txt} has no dots")

    # index of the tracked fixed point (drawn purple): lowest Z of the requested type, skipping the axis
    tracked_idx = None
    if tracked_type is not None and fps:
        axis_i = (min(range(len(fps)), key=lambda i: (fps[i][0] - ref[0]) ** 2 + (fps[i][1] - ref[1]) ** 2)
                  if ref is not None else None)
        cand = [i for i, fp in enumerate(fps) if fp[2] == tracked_type and i != axis_i]
        if cand:
            tracked_idx = min(cand, key=lambda i: fps[i][1])       # bottom-most (lowest Z)

    def color(lid):                                         # legs red; volumetric dots a single muted gray
        return _KIND_COLOR.get(legkinds.get(lid, ""), SURF_COLOR)

    is_leg = np.array([legkinds.get(int(i), "") in _KIND_COLOR for i in arr[:, 0]])

    # thin out isolated scattered dots so only the dense curves remain (see _density_mask). Each class
    # is filtered against itself, so a stray red leg dot sitting near a grey O-surface is still dropped.
    if DENSITY_FILTER:
        radius, min_count = dens
        keep = np.ones(len(arr), dtype=bool)
        keep[is_leg] = _density_mask(arr[is_leg, 1:3], radius, min_count)
        keep[~is_leg] = _density_mask(arr[~is_leg, 1:3], radius, min_count)
        arr, is_leg = arr[keep], is_leg[keep]

    def draw(ax, fp_scale=1.0):
        # grey volumetric dots underneath, red manifold legs always on top
        ax.scatter(arr[~is_leg, 1], arr[~is_leg, 2], s=DOT_SIZE,
                   c=[color(int(i)) for i in arr[~is_leg, 0]],
                   edgecolors="none", rasterized=True, zorder=3)
        ax.scatter(arr[is_leg, 1], arr[is_leg, 2], s=DOT_SIZE, c=LEG_COLOR,
                   edgecolors="none", rasterized=True, zorder=4)
        for i, (R, Z, typ, _tr) in enumerate(fps):
            mk, ms, fc = _FP_MARKER.get(typ, ("+", 9, "black"))
            if typ == "elliptic":
                ms *= O_MARKER_SCALE
            if i == tracked_idx:                            # the tracked fixed point -> purple
                fc = TRACKED_COLOR
            ax.plot([R], [Z], marker=mk, ms=ms * fp_scale, mfc=fc, mec="white",
                    mew=0.7 * fp_scale, ls="none", zorder=6)

    return draw


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    plt.rcParams.update({"font.size": 15})              # larger axis labels, ticks, and text
    left, right = (_resolve(a) for a in sys.argv[1:3])
    for fpath in (left, right):
        if not fpath.exists():
            raise SystemExit(f"{fpath} not found")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.9, 5), sharex=True, sharey=True,
                                   gridspec_kw={"wspace": 0.0})       # no space between panels
    _draw_fn(left, tracked_type="hyperbolic", dens=DENS_A)(axL, fp_scale=FP_SCALE)    # panel A
    _draw_fn(right, tracked_type="elliptic", dens=DENS_B)(axR, fp_scale=FP_SCALE)     # panel B

    # shared window WIN_XLIM/WIN_YLIM (sharex/sharey propagate it to both panels)
    axL.set_xlim(*WIN_XLIM)
    axL.set_ylim(*WIN_YLIM)

    for ax in (axL, axR):
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MultipleLocator(TICK_STEP))
        ax.yaxis.set_major_locator(MultipleLocator(TICK_STEP))
        ax.set_axisbelow(True)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("R [m]")
    axL.set_ylabel("Z [m]")

    # panel labels at the same top-left height on each panel
    lab_kw = dict(va="top", ha="left", fontsize=18, fontweight="bold", zorder=8,
                  bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
    axL.text(0.03, 0.97, "A", transform=axL.transAxes, **lab_kw)
    axR.text(0.03, 0.97, "B", transform=axR.transAxes, **lab_kw)

    out = left.with_name(left.stem + "_row.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    print(f"wrote {out}")
    plt.show()
    plt.close(fig)
