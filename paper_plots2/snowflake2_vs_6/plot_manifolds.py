#!/usr/bin/env python3
"""Plot the manifolds written by O2X/mk_manifolds.py for the fixed points of a design's xpoints list.

Separatrix MANIFOLDS of hyperbolic X-points are drawn (stable = red, unstable = blue) and the NESTED
SURFACES of elliptic O-points are coloured inner -> outer; the fixed points are marked (X-points 'X',
O-points green 'o').

Usage:
    plot_manifolds.py <design_json>     (or the <stem>_allmanifolds.txt directly)

Writes <stem>_allmanifolds.png next to the data file. Also importable as plot_manifolds(path).
"""
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

_XP_RE = re.compile(r"xpoint_RZ=([-0-9.eE+]+);([-0-9.eE+]+)")
_FP_RE = re.compile(r"fixedpt R=([-0-9.eE+]+) Z=([-0-9.eE+]+) type=(\w+)(?:\s+trace=([-0-9.eEnaN+]+))?")
_LEG_RE = re.compile(r"^# leg (\d+) kind=(\w+)")
_LEGDIR_RE = re.compile(r"legdir Rx=([-0-9.eE+]+) Zx=([-0-9.eE+]+)")
LEG_COLOR = "tab:red"                                      # separatrix manifold legs (stable + unstable)
SURF_COLOR = "0.6"                                         # single muted gray for the volumetric dots
_KIND_COLOR = {"stable": LEG_COLOR, "unstable": LEG_COLOR}  # manifold legs (both branches): red
DOT_SIZE = 3.0
O_MARKER_SCALE = 0.5                                        # shrink factor for O-point (elliptic) markers
TICK_STEP = 0.1                                             # axis ticks on multiples of 0.1 m
_FP_MARKER = {"snowflake": ("*", 16.0, "black"), "hyperbolic": ("X", 8.0, "black"),
              "parabolic": ("s", 9.0, "black"), "elliptic": ("o", 10.0, "tab:blue"),
              "unknown": ("+", 9.0, "black")}
WIN_XLIM = (0.2172991358185795, 0.6769278581867286)        # main-window R (x) limits
WIN_YLIM = (-0.1303862863541289, 0.32478002123374694)      # main-window Z (y) limits
INSET_RECT = [0.12, 0.464, 0.24, 0.516]                    # zoom-inset position (axes fraction; top,
                                                           # just right of the panel label)
INSET_XLIM = (0.5919247567697767, 0.641866125964457)       # inset R (x) zoom range
INSET_YLIM = (0.18052843608453437, 0.2877554934731127)     # inset Z (y) zoom range


def _headers(path):
    """(fixedpts, legkinds, ref_point) from the '#' header lines. Each fixedpt is (R, Z, type, trace);
    trace is NaN for older files that didn't record it. Manifold legs that belong to a PARABOLIC fixed
    point are re-tagged kind='parabolic' so they colour like the O-point surface dots (muted grey): the
    legdir lines are emitted one per manifold leg in leg_id order, so legdir[i] is leg i's fixed point."""
    fps, kinds, ref, legdirs = [], {}, None, []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            if ref is None and (m := _XP_RE.search(line)):
                ref = (float(m.group(1)), float(m.group(2)))
            if m := _FP_RE.search(line):
                fps.append((float(m.group(1)), float(m.group(2)), m.group(3),
                            float(m.group(4)) if m.group(4) else float("nan")))
            if m := _LEG_RE.match(line):
                kinds[int(m.group(1))] = m.group(2)
            if m := _LEGDIR_RE.search(line):
                legdirs.append((float(m.group(1)), float(m.group(2))))
    par_xy = [(R, Z) for R, Z, typ, _tr in fps if typ == "parabolic"]
    for lid, (Rx, Zx) in enumerate(legdirs):                # legdir order == manifold-leg id order
        if any(abs(Rx - pr) < 1e-6 and abs(Zx - pz) < 1e-6 for pr, pz in par_xy):
            kinds[lid] = "parabolic"                         # -> SURF_COLOR via color() (grey)
    return fps, kinds, ref


def annotate_traces(ax, traces, cx, cy, half=0.25, half_y=None, pad=None, fontsize=14, sym=r"\mathrm{tr}"):
    """Label EVERY found fixed point with its monodromy trace (traces = the (R, Z, type, trace)
    fixed-point list from _headers), to the NORTH-WEST of each point with an arrow back. Labels are
    placed in a vertical stack whose ORDER MATCHES the points' Z order, so the leader arrows preserve
    the vertical ordering and therefore never cross; a minimum y-gap (>= a label-box height) keeps the
    boxes from overlapping and the stack is shifted back inside the window. The window half-widths
    (half in R, half_y in Z -- they differ for the anisotropic parabolic inset) set the offsets, pad
    and gap so the same routine works in the tiny zoom inset. NaN-trace points are skipped."""
    if half_y is None:
        half_y = half
    off_x, off_y = 0.40 * half, 0.40 * half_y              # NW label offsets (scale with each half-width)
    if pad is None:
        pad = 0.12 * min(half, half_y)                     # keep label boxes inside the window edges
    lo_x, hi_x, lo_y, hi_y = cx - half + pad, cx + half - pad, cy - half_y + pad, cy + half_y - pad
    # desired NW label position for each finite-trace fixed point, clamped to the window
    L = []
    for R, Z, typ, tr in traces:
        if tr is None or not np.isfinite(tr):
            continue
        tx = min(max(R - off_x, lo_x), hi_x)
        ty = min(max(Z + off_y, lo_y), hi_y)
        L.append([R, Z, tr, tx, ty])
    if not L:
        return
    # Assign label y-slots in the SAME vertical order as the target points (sort by point Z, descending).
    # A monotone point->label mapping means an arrow to a higher point starts from a higher label, so the
    # arrows cannot cross. Enforce a minimum gap top->bottom, then shift the whole stack back inside.
    gap = min(0.15 * (hi_y - lo_y), (hi_y - lo_y) / max(len(L) - 1, 1))
    L.sort(key=lambda e: -e[1])                            # by POINT Z (not the clamped label y)
    for i in range(1, len(L)):
        L[i][4] = min(L[i][4], L[i - 1][4] - gap)
    if L[-1][4] < lo_y:
        shift = lo_y - L[-1][4]
        for e in L:
            e[4] = min(e[4] + shift, hi_y)
    for R, Z, tr, tx, ty in L:
        ax.annotate(rf"${sym}={tr:.3f}$", xy=(R, Z), xytext=(tx, ty),
                    fontsize=fontsize, ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.7", alpha=0.85),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="black"))


def plot_manifolds(path, out=None, show=True):
    """Plot the manifold / O-point-surface dots written by mk_manifolds.py.

    path : a <stem>_allmanifolds.txt, or the design json (then <stem>_allmanifolds.txt next to it).
    out  : output image path (default <txt>.png; pass out=False to skip saving).
    show : call plt.show() after drawing.
    Returns the matplotlib Figure.
    """
    plt.rcParams.update({"font.size": 15})              # larger axis labels, ticks, and text
    p = Path(path)
    f = p if p.suffix == ".txt" else p.parent / f"{p.stem}_allmanifolds.txt"
    if not f.exists():
        raise SystemExit(f"{f} not found (run mk_manifolds.py first)")

    fps, legkinds, ref = _headers(f)            # ref = magnetic axis (xp_list[0]) from the header
    arr = np.atleast_2d(np.loadtxt(f, delimiter=","))           # data rows: leg_id, R, Z
    if not arr.size:
        raise SystemExit("no dots to plot.")

    # Per-leg colour: separatrix manifold legs red; everything else (O-point nested surfaces, etc.) a
    # single muted gray.
    def color(lid):
        return _KIND_COLOR.get(legkinds.get(lid, ""), SURF_COLOR)

    is_leg = np.array([legkinds.get(int(i), "") in _KIND_COLOR for i in arr[:, 0]])

    def draw(ax, fp_scale=1.0):
        # grey volumetric dots underneath, red manifold legs always on top
        ax.scatter(arr[~is_leg, 1], arr[~is_leg, 2], s=DOT_SIZE,
                   c=[color(int(i)) for i in arr[~is_leg, 0]],
                   edgecolors="none", rasterized=True, zorder=3)
        ax.scatter(arr[is_leg, 1], arr[is_leg, 2], s=DOT_SIZE, c=LEG_COLOR,
                   edgecolors="none", rasterized=True, zorder=4)
        for R, Z, typ, _tr in fps:
            mk, ms, fc = _FP_MARKER.get(typ, ("+", 9, "black"))
            if typ == "elliptic":
                ms *= O_MARKER_SCALE                                           # smaller O-point dots (axis + new O)
            ax.plot([R], [Z], marker=mk, ms=ms * fp_scale, mfc=fc, mec="white",
                    mew=0.7 * fp_scale, ls="none", zorder=6)

    # Window: explicit R (x) and Z (y) limits.
    fig, ax = plt.subplots(figsize=(6, 6))
    draw(ax, fp_scale=1.0)
    ax.set_xlim(*WIN_XLIM)
    ax.set_ylim(*WIN_YLIM)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(TICK_STEP))
    ax.yaxis.set_major_locator(MultipleLocator(TICK_STEP))
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3)

    # Zoom inset (top-left) over the requested R/Z window.
    axins = ax.inset_axes(INSET_RECT)
    draw(axins, fp_scale=1.0)
    axins.set_xlim(*INSET_XLIM)
    axins.set_ylim(*INSET_YLIM)
    axins.set_aspect("equal")
    axins.set_xticks([])
    axins.set_yticks([])
    for s in axins.spines.values():
        s.set_edgecolor("0.4")
    ind = ax.indicate_inset_zoom(axins, edgecolor="0.4", alpha=0.8, lw=1.0)
    conns = getattr(ind, "connectors", None)               # mpl>=3.10: single artist; older: (rect, conns)
    if conns is None:
        _rect, conns = ind
    for c in conns:                                        # hide the connector lines (box on ax is enough)
        c.set_visible(False)

    fig.tight_layout()
    if out is not False:
        out = Path(out) if out else f.with_suffix(".png")
        fig.savefig(out, dpi=400)
        print(f"wrote {out}  ({arr.shape[0]} dots, {len(fps)} fixed points)")
    if show:
        plt.show()
    plt.close(fig)
    return fig


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    plot_manifolds(sys.argv[1])
