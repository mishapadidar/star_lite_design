#!/usr/bin/env python3
import os, glob, re
import numpy as np
import matplotlib
# In batch/headless runs (no DISPLAY, e.g. run_render.sh on a compute node) pin the
# non-interactive Agg backend, so importing pyplot / plt.show() can never select a
# GUI backend (which would hang waiting on a window that never appears). On a
# workstation with a display, leave the default backend so plt.show() still works.
if not os.environ.get('DISPLAY'):
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
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

# Panel phi grid: single source of truth is phis.txt written by mk_manifolds
# (DN half-period [0,0.25] vs SN full-period [0,0.5]). Fall back to folder-name
# detection for older outputs that predate phis.txt.
_phis_file = p.parent / 'phis.txt'
if _phis_file.exists():
    phis = np.atleast_1d(np.loadtxt(_phis_file))
else:
    phis = np.linspace(0, 0.5 if 'null=SN' in p.parent.name else 0.25, 9)
nphi  = len(phis)

# X-point classification (from mk_manifolds) drives hyperbolic-only styling:
# colour stable manifolds red and unstable manifolds blue.
_type_file = p.parent / 'xpoint_type.txt'
_xpoint_type = _type_file.read_text().strip() if _type_file.exists() else ''
# Map leg index k -> is-stable? legs.txt: col0=k, col4=kind ('stable'/'unstable').
_leg_is_stable = {}
_f_legs = p.parent / 'legs.txt'
if _f_legs.exists():
    _la = np.atleast_2d(np.loadtxt(_f_legs, delimiter=',', usecols=(0, 4), dtype=str, skiprows=1))
    _leg_is_stable = {int(kk): (kind.strip() == 'stable') for kk, kind in _la}

# Parabolic local model (written by mk_manifolds for parabolic X-points): in the
# frame xi = dx.v, eta = dx.u at the X-point, orbits follow level sets of
#   H(xi, eta) = (sigma/2) eta^2 - (D/3) xi^3,
# and H = 0 is the separatrix (a semicubical cusp tangent to v).
_cusp = None
_f_cusp = p.parent / 'parabolic_cusp.txt'
if _f_cusp.exists():
    _lam, _th, _sig, _D = np.loadtxt(_f_cusp, delimiter=',', skiprows=1)
    if _lam > 0:        # the local model is derived for the +1 repeated eigenvalue
        _cusp = (float(_th), float(_sig), float(_D))
_CUSP_HALF = 0.02       # half-width of the (R, Z) window for the level-set overlay

# Snowflake invariant-line cubic discriminant (written by mk_manifolds for
# snowflake X-points). Sign gives the leg count: >0 -> 6 legs (monkey saddle),
# <0 -> 2 legs, ~0 -> the 2<->6 leg transition. Surfaced in the figure legend.
_snow_disc = None
_f_disc = p.parent / 'snowflake_discriminant.txt'
if _xpoint_type == 'snowflake' and _f_disc.exists():
    _dd = np.atleast_1d(np.loadtxt(_f_disc, delimiter=',', skiprows=1))
    _snow_disc = float(_dd[0])

fig, axes = plt.subplots(3, 3, figsize=(8.25, 8.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0, "hspace": 0})

def _scatter_file(ax, path, dot_size, color=None, label=None, alpha=None):
    """Plot a poincare file's dots (drop the first hit per fieldline id). If
    color is given, all dots use it (with an optional legend label); otherwise
    they are coloured per fieldline id. alpha sets the dot transparency."""
    if not os.path.exists(path):
        return
    d = np.loadtxt(path, delimiter=",")
    if not d.size:
        return
    if d.ndim == 1:
        d = d[None, :]
    _, first_idx = np.unique(d[:, 0], return_index=True)
    mask = np.ones(d.shape[0], dtype=bool)
    #mask[first_idx] = False
    d = d[mask]
    if d.size == 0:
        return
    # manifold files are [id, s, x, y, z]; interior files are [id, x, y, z].
    xi = 2 if d.shape[1] >= 5 else 1
    R, Z = np.hypot(d[:, xi], d[:, xi + 1]), d[:, xi + 2]
    c = _cat_colors(d[:, 0]) if color is None else color
    # edgecolors='none'/linewidths=0: no edge stroke, so the marker is a solid
    # filled disc (otherwise tiny markers render as hollow/white-centred specks).
    ax.scatter(R, Z, s=dot_size, color=c, rasterized=True, label=label, alpha=alpha,
               edgecolors='none', linewidths=0)


def _line_file(ax, path, color=None, label=None, alpha=None, lw=0.6, leg_k=0, zorder=None, do_print=False):
    """Plot one leg's poincare file as a continuous manifold LINE (not dots).

    New files store [id, s, x, y, z] where s is the global manifold parameter
    (arclength position in return-map units); ordering by s walks the line
    strictly outward from the X-point even when seeds survive a different number
    of transits. Legacy 4-column files [id, x, y, z] fall back to the old
    return-major reorder. If color is None the leg is coloured by leg_k."""
    if not os.path.exists(path):
        return
    d = np.loadtxt(path, delimiter=",")
    if not d.size:
        return
    if d.ndim == 1:
        d = d[None, :]
    if d.shape[1] >= 5:                       # [id, s, x, y, z]: order by parameter s
        order = np.argsort(d[:, 1], kind='stable')
        R = np.hypot(d[order, 2], d[order, 3])
        Z = d[order, 4]
    else:                                     # legacy [id, x, y, z]: return-major reorder
        df = pd.DataFrame(d[:, :4], columns=['id', 'x', 'y', 'z'])
        df['ret'] = df.groupby('id').cumcount()   # return index within each seed
        df = df.sort_values(['ret', 'id'])        # return-major: walk arc-by-arc
        R = np.hypot(df['x'].to_numpy(), df['y'].to_numpy())
        Z = df['z'].to_numpy()
    # Report the arclength of the plotted manifold (truncation to the target is
    # done upstream in mk_manifolds.py, so the full file is drawn as-is).
    if do_print:
        arclen = float(np.hypot(np.diff(R), np.diff(Z)).sum()) if R.size > 1 else 0.0
        print(f"  manifold {os.path.basename(path)}: arclength {arclen:.4f} m ({R.size} pts)")
    c = color if color is not None else _cat_colors([leg_k])[0]
    ax.plot(R, Z, color=c, lw=lw, label=label, alpha=alpha, rasterized=True, zorder=zorder)


def draw_panel(ax, i, interior_dot_size=1.0, leg_alpha=1.0, leg_lw=0.5, leg_zorder=0, dot_alpha=None, manif_lw=1.5, do_print=False):
    """Draw all overlays (manifolds, interior, vessel, surface, fixed points,
    invariant lines) for phi index i onto ax. The invariant-line style
    (leg_alpha/leg_lw/leg_zorder) is faint+behind by default; the zoom inset
    passes a prominent style so the lines aren't buried under the leg dots.
    dot_alpha sets the Poincaré-dot transparency (used to declutter the zoom)."""
    # Interior / inward Poincaré dots first, so the manifolds draw ON TOP of them
    # (matching plot_manifolds_DN.py's draw order). Always opaque (solid dots),
    # even in the zoom inset where the manifold lines use dot_alpha.
    _scatter_file(ax, p.parent / f"poincare_interior_{i}.txt", interior_dot_size, alpha=1.0)
    # Manifolds, drawn on top of the interior dots (leg index k is the trailing
    # number in poincare_{top|bot}_{i}_{k}.txt). All classified X-point types are
    # drawn as Poincaré scatter DOTS (no solid manifold lines):
    #   hyperbolic -> red (stable) / blue (unstable) dots,
    #   snowflake  -> red (stable) / blue (unstable) dots,
    #   parabolic  -> black dots (no stable/unstable structure; the single
    #                 eigendirection is drawn below).
    for f in sorted(glob.glob(str(p.parent / f"poincare_*_{i}_*.txt"))):
        m = re.search(r'_(\d+)\.txt$', os.path.basename(f))
        leg_k = int(m.group(1)) if m is not None else 0
        if _xpoint_type in ('parabolic', 'snowflake', 'hyperbolic'):   # dots only
            lbl = None
            if _xpoint_type == 'parabolic':
                col = 'black'
            elif leg_k in _leg_is_stable:   # snowflake & hyperbolic: red=stable/blue=unstable
                stable = _leg_is_stable[leg_k]
                col = 'red' if stable else 'blue'
                lbl = 'stable manifold' if stable else 'unstable manifold'
            else:
                col = None   # unclassified leg: per-fieldline-id categorical colour
            _scatter_file(ax, f, 2.8125, color=col, label=lbl)   # 5 * 0.75^2: 25% smaller diameter
            continue
        # Fallback for an unknown/unclassified X-point type (xpoint_type.txt
        # missing): draw the raw manifold files as categorical lines.
        _line_file(ax, f, color=None, label=None, alpha=dot_alpha, lw=manif_lw,
                   leg_k=leg_k, zorder=leg_zorder + 1, do_print=do_print)

    f_v = p.parent / f"vessel_cross_{i}.txt"
    if os.path.exists(f_v):
        v = np.atleast_2d(np.loadtxt(f_v, delimiter=',', skiprows=1))
        if v.size:
            ax.plot(v[:, 0], v[:, 1], 'k-', lw=1.0, label='vessel')

    f_v = p.parent / f"surface_cross_{i}.txt"
    if os.path.exists(f_v):
        v = np.atleast_2d(np.loadtxt(f_v, delimiter=',', skiprows=1))
        if v.size:
            ax.plot(v[:, 0], v[:, 1], 'k--', lw=1.0, label='optimization surface')

    # Red dots where the modular coils cross this phi-plane (written by
    # mk_manifolds for every device), showing where the coils sit in the section.
    f_c = p.parent / f"coil_cross_{i}.txt"
    if os.path.exists(f_c):
        v = np.atleast_2d(np.loadtxt(f_c, delimiter=',', skiprows=1))
        if v.size:
            ax.plot(v[:, 0], v[:, 1], 'o', color='red', ms=4, ls='none',
                    zorder=6, label='coil')

    f_fp = p.parent / f"fixed_points_{i}.txt"
    if os.path.exists(f_fp):
        fp = np.atleast_2d(np.loadtxt(f_fp, delimiter=',', skiprows=1))
        if fp.size:
            ax.plot(fp[0, 0], fp[0, 1], marker='o', color='tab:green',
                    ms=4, ls='none', label='axis')
            if fp.shape[0] > 1:
                ax.plot(fp[1, 0], fp[1, 1], marker='X', color='tab:red',
                        ms=3, ls='none', label='X-point')
            # row2 = stellsym-partner (bottom) X-point, when present.
            if fp.shape[0] > 2:
                ax.plot(fp[2, 0], fp[2, 1], marker='X', color='tab:red',
                        ms=3, ls='none')

    # invariant-direction lines through the X-points, only at phi=0
    if i == 0:
        f_legs = p.parent / "legs.txt"
        f_fp0 = p.parent / "fixed_points_0.txt"
        if os.path.exists(f_legs) and os.path.exists(f_fp0):
            fp0 = np.atleast_2d(np.loadtxt(f_fp0, delimiter=',', skiprows=1))
            L = np.atleast_2d(np.loadtxt(f_legs, delimiter=',', usecols=(2, 3)))

            def _draw_legs(R0, Z0, flipZ):
                for vR, vZ in L:
                    # bottom X-point is the stellsym mirror of the top: the
                    # invariant directions reflect in Z (slope flips sign).
                    vZ = -vZ if flipZ else vZ
                    ax.axline((R0, Z0), (R0 + vR, Z0 + vZ),
                              color='k', lw=leg_lw, alpha=leg_alpha, zorder=leg_zorder)

            if fp0.shape[0] > 1:            # top X-point: directions as-is
                _draw_legs(fp0[1, 0], fp0[1, 1], flipZ=False)
            if fp0.shape[0] > 2:            # bottom X-point: stellsym-reflected dirs
                _draw_legs(fp0[2, 0], fp0[2, 1], flipZ=True)

            # Parabolic local model: level sets of H near the X-point, with the
            # separatrix H = 0 (the semicubical cusp) drawn prominently. The
            # bottom X-point uses the Z-mirrored frame, like the legs above.
            if _cusp is not None:
                th, sig, D = _cusp
                v = np.array([np.cos(th), np.sin(th)])
                u = np.array([-v[1], v[0]])
                g = np.linspace(-_CUSP_HALF, _CUSP_HALF, 401)

                def _draw_cusp(R0, Z0, flipZ):
                    dR, dZ = np.meshgrid(g, g, indexing='xy')
                    dZs = -dZ if flipZ else dZ
                    xi = dR * v[0] + dZs * v[1]
                    eta = dR * u[0] + dZs * u[1]
                    H = 0.5 * sig * eta**2 - (D / 3.0) * xi**3
                    Hs = float(np.abs(H).max())
                    if Hs == 0.0:
                        return
                    ax.contour(R0 + g, Z0 + g, H, levels=[0.0], colors='m',
                               linewidths=1.0, linestyles='--', zorder=leg_zorder + 2)
                    ax.contour(R0 + g, Z0 + g, H,
                               levels=Hs * np.array([-0.1, -0.03, -0.01, -1e-3, 1e-3, 0.01, 0.03, 0.1]),
                               colors='0.5', linewidths=0.4, alpha=0.7, zorder=leg_zorder)

                ax.plot([], [], 'm--', lw=1.0, label='parabolic separatrix (local model)')
                if fp0.shape[0] > 1:
                    _draw_cusp(fp0[1, 0], fp0[1, 1], flipZ=False)
                if fp0.shape[0] > 2:
                    _draw_cusp(fp0[2, 0], fp0[2, 1], flipZ=True)


for i, ax in enumerate(axes.flat):
    draw_panel(ax, i, do_print=(i == 0))   # report manifold arclengths for phi=0 only
    ax.text(0.5, 0.98, rf"$\phi/2\pi = {phis[i]:.4f}$",
            transform=ax.transAxes, ha='center', va='top', fontsize=8)
    ax.set_aspect('equal')

# Panel limits: contain the full vessel cross-section (the outermost geometry —
# manifolds, surface and fixed points all sit inside it) across all phi panels so
# it is never clipped. mk_manifolds samples the vessel over its parameter-derived
# extent, so the union of the vessel_cross_*.txt data IS the vessel's (R, Z)
# bounding box. Use a SQUARE window (equal R/Z span) centred on that box so the
# shared equal-aspect, wspace=hspace=0 grid stays gap-free. Fall back to the old
# fixed window if no vessel data is present.
_vR, _vZ = [], []
for _fv in glob.glob(str(p.parent / "vessel_cross_*.txt")):
    _vd = np.atleast_2d(np.loadtxt(_fv, delimiter=',', skiprows=1))
    if _vd.size:
        _vR.append(_vd[:, 0]); _vZ.append(_vd[:, 1])
if _vR:
    _R = np.concatenate(_vR); _Z = np.concatenate(_vZ)
    _R = _R[np.isfinite(_R)]; _Z = _Z[np.isfinite(_Z)]
    _cx = 0.5 * (_R.min() + _R.max())
    _cy = 0.5 * (_Z.min() + _Z.max())
    _half = 0.5 * max(_R.max() - _R.min(), _Z.max() - _Z.min()) * 1.05   # 5% margin
    axes[0, 0].set_xlim([_cx - _half, _cx + _half])
    axes[0, 0].set_ylim([_cy - _half, _cy + _half])
else:
    axes[0, 0].set_xlim([0, 1])
    axes[0, 0].set_ylim([-0.5, 0.5])

# With wspace=0 the panels abut, so the boundary R-tick labels (R=0 of one panel,
# R=1 of its neighbour) overlap. Prune the edge ticks so only interior R labels
# show (shared x-axis, so setting it on one panel applies to all).
axes[0, 0].xaxis.set_major_locator(MaxNLocator(nbins=5, prune='both'))

for ax in axes[-1, :]: ax.set_xlabel('R')
for ax in axes[:, 0]:  ax.set_ylabel('Z')

# Zoom insets on the X-points in every phi panel to reveal the fixed-point
# structure: top X-point box top-right, bottom (stellsym-partner) box bottom-right.
# Hyperbolic X-points get a wider zoom window (zoomed out a bit) since their
# manifold structure spreads further than the snowflake/parabolic case.
ZOOM_HALF = 0.025 if _xpoint_type == 'hyperbolic' else 0.01
print(f"X-point type: {_xpoint_type or '(unknown: xpoint_type.txt missing — rerun mk_manifolds.py)'}"
      f"  ->  ZOOM_HALF = {ZOOM_HALF}")

def _add_zoom(ax, i, center, rect):
    R0, Z0 = center
    axins = ax.inset_axes(rect)
    # In the zoom, draw the invariant lines + manifold dots prominently and on top.
    draw_panel(axins, i, leg_zorder=5, dot_alpha=0.5)
    axins.set_xlim(R0 - ZOOM_HALF, R0 + ZOOM_HALF)
    axins.set_ylim(Z0 - ZOOM_HALF, Z0 + ZOOM_HALF)
    axins.set_aspect('equal')
    axins.set_xticks([]); axins.set_yticks([])
    ind = ax.indicate_inset_zoom(axins, edgecolor='0.4', lw=0.8)
    # Keep the indicator rectangle but drop the lines linking it to the inset.
    conns = getattr(ind, 'connectors', None)
    if conns is None and isinstance(ind, tuple):
        conns = ind[1]
    for c in (conns or []):
        c.set_visible(False)

for i, ax in enumerate(axes.flat):
    f_fp = p.parent / f"fixed_points_{i}.txt"
    if not os.path.exists(f_fp):
        continue
    fp = np.atleast_2d(np.loadtxt(f_fp, delimiter=',', skiprows=1))
    if fp.shape[0] > 1:                                              # top X-point
        _add_zoom(ax, i, (fp[1, 0], fp[1, 1]), [0.66, 0.66, 0.32, 0.32])
    if fp.shape[0] > 2:                                              # bottom X-point
        _add_zoom(ax, i, (fp[2, 0], fp[2, 1]), [0.66, 0.02, 0.32, 0.32])

h, l = [], []
for a in axes.flat:
    hh, ll = a.get_legend_handles_labels()
    h += hh; l += ll
uniq = dict(zip(l, h))
fig.legend(
    uniq.values(),
    uniq.keys(),
    loc='lower center',
    bbox_to_anchor=(0.5, 0.875),
    ncol=min(len(uniq), 4),
    frameon=False,
    markerscale=2,
)

# Snowflake invariant-line cubic discriminant as a caption at the bottom of the
# figure, with the leg-count verdict its sign implies.
if _snow_disc is not None:
    _verdict = ('6 legs' if _snow_disc > 0 else
                '2 legs' if _snow_disc < 0 else 'degenerate (double root)')
    fig.text(0.5, 0.05,
             f'snowflake invariant-line cubic discriminant = {_snow_disc:+.2e}  ({_verdict})',
             ha='center', va='top', fontsize=9)

parent_name = p.parent.name
grandparent_name = p.parent.parent.name
plt.savefig(p.parent / f'xs_{grandparent_name}_{parent_name}.png', dpi=600, bbox_inches='tight')
plt.show()
