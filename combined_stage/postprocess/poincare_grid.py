#!/usr/bin/env python3
"""Poincaré sections of a device folder's coil field at nine toroidal angles.

usage: poincare_grid.py <device dir> [--out-dir <run>/figures] [--lines-mid 28] [--lines-x 14] [--tmax 3000]

The angles span the symmetry-reduced range, both ends included: phi/2pi in [0, 1/(2 nfp)] for a stellarator-symmetric
device (the other half period is its mirror image) and [0, 1/nfp] otherwise. Field lines start on phi = 0 along the
segment from the magnetic axis to 15 % beyond the outboard LCFS point (the optimization surface if there is no LCFS)
and, for a double null, along the axis-to-X-point segment from 55 % to 135 % of its length. They are traced in an
InterpolatedField of the coils over one field period. Overlays: VMEX boundary (the input.combined_stage_* copied into
the device folder), optimization (Boozer) surface, LCFS, magnetic axis, X-points and coil crossings.
Writes <out dir>/poincare_grid.png (default: the device folder).
"""
import argparse
import glob
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from simsopt._core import load  # noqa: E402
from simsopt.field import BiotSavart, InterpolatedField, compute_fieldlines  # noqa: E402
from simsopt.field.tracing import (MaxRStoppingCriterion, MaxZStoppingCriterion, MinRStoppingCriterion,  # noqa: E402
                                   MinZStoppingCriterion)
from simsopt.geo import CurveXYZFourierSymmetries, SurfaceRZFourier  # noqa: E402

TAB20 = plt.get_cmap("tab20")


def curve_rz(curve, npts=4096):
    """Function phi [rad] -> (R, Z) where a closed one-turn curve (e.g. an archived axis / X-point line whose quadrature
    points cover one field period only) crosses the plane of cylindrical angle phi."""
    full = CurveXYZFourierSymmetries(np.linspace(0, 1, npts, endpoint=False), curve.order, curve.nfp, curve.stellsym,
                                     ntor=curve.ntor)
    full.x = curve.x
    g = full.gamma()
    ph = np.unwrap(np.arctan2(g[:, 1], g[:, 0]))
    order = np.argsort(ph)
    ph, R, Z = ph[order], np.hypot(g[order, 0], g[order, 1]), g[order, 2]
    ph_c, R_c, Z_c = np.r_[ph, ph[0] + 2 * np.pi], np.r_[R, R[0]], np.r_[Z, Z[0]]

    def at(phi):
        x = ph[0] + np.mod(phi - ph[0], 2 * np.pi)
        return float(np.interp(x, ph_c, R_c)), float(np.interp(x, ph_c, Z_c))
    return at


def coil_crossings(coils, phi_frac):
    """(R, Z) points where the coils pierce the plane phi/2pi = phi_frac (as array/mk_manifolds.coil_cross_sections)."""
    rows = []
    for c in coils:
        g = np.asarray(c.curve.gamma())
        d = np.mod(np.mod(np.arctan2(g[:, 1], g[:, 0]) / (2 * np.pi), 1.0) - phi_frac + 0.5, 1.0) - 0.5
        for i in range(g.shape[0]):
            j = (i + 1) % g.shape[0]
            if d[i] * d[j] < 0.0 and abs(d[i] - d[j]) < 0.5:
                q = g[i] + d[i] / (d[i] - d[j]) * (g[j] - g[i])
                rows.append((np.hypot(q[0], q[1]), q[2]))
    return np.array(rows) if rows else np.zeros((0, 2))


def section(surface, phi_frac):
    try:
        xyz = surface.cross_section(phi_frac, thetas=512)
        return np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2]
    except Exception:  # noqa: BLE001 -- a surface that "goes back on itself" at this angle
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("device_dir")
    ap.add_argument("--lines-mid", type=int, default=28)
    ap.add_argument("--lines-x", type=int, default=14)
    ap.add_argument("--tmax", type=float, default=3000.0)
    ap.add_argument("--out-dir", default=None, help="where poincare_grid.png goes (default: the device folder)")
    args = ap.parse_args()
    dev = os.path.abspath(args.device_dir)

    boozer_surfaces, _, axes, xpoints, _ = load(sorted(glob.glob(os.path.join(dev, "design_opt_final_*.json")))[0])
    surf = boozer_surfaces[0].surface
    coils = boozer_surfaces[0].biotsavart.coils
    nfp, stellsym = surf.nfp, surf.stellsym
    phis = np.linspace(0.0, (0.5 if stellsym else 1.0) / nfp, 9)
    lcfs_files = sorted(glob.glob(os.path.join(dev, "LCFS_*.json")))
    lcfs = load(lcfs_files[0])[0][0].surface if lcfs_files else None
    nml = sorted(glob.glob(os.path.join(dev, "input.combined_stage_*")))
    vmex = (SurfaceRZFourier.from_vmec_input(nml[0], quadpoints_phi=phis, quadpoints_theta=np.linspace(0, 1, 256))
            if nml else None)
    sec_opt = [section(surf, p) for p in phis]
    sec_lcfs = [section(lcfs, p) for p in phis] if lcfs is not None else [None] * 9
    sec_vmex = ([(np.hypot(g[:, 0], g[:, 1]), g[:, 2]) for g in vmex.gamma()] if vmex is not None else [None] * 9)
    axis_at = curve_rz(axes[0].curve)
    xp_at = curve_rz(xpoints[0].curve) if xpoints else None
    ax_rz = [axis_at(2 * np.pi * p) for p in phis]
    xp_rz = []
    for p in phis:
        if xp_at is None:
            xp_rz.append([])
            continue
        top = xp_at(2 * np.pi * p)
        pts = [top]
        if stellsym:                                  # lower null = stellarator-symmetric image: (R, -phi, -Z)
            R_b, Z_b = xp_at(-2 * np.pi * p)
            pts.append((R_b, -Z_b))
        xp_rz.append(pts)

    # extent of the drawn plasma region (outermost closed surface + X-points), square window as plot_manifolds.py
    outer = [s for s in (sec_lcfs if lcfs is not None else sec_opt) if s is not None]
    Rall = np.concatenate([s[0] for s in outer] + [np.array([q[0] for pts in xp_rz for q in pts])])
    Zall = np.concatenate([s[1] for s in outer] + [np.array([q[1] for pts in xp_rz for q in pts])])
    cx, cz = 0.5 * (Rall.min() + Rall.max()), 0.5 * (Zall.min() + Zall.max())
    half = 0.5 * max(np.ptp(Rall), np.ptp(Zall)) * 1.15

    # interpolated coil field over one field period, on a box around that window
    rmin, rmax, zmax = cx - 1.3 * half, cx + 1.3 * half, abs(cz) + 1.3 * half
    t0 = time.time()
    field = InterpolatedField(BiotSavart(coils), 4, (rmin, rmax, 60), (0, 2 * np.pi / nfp, 64),
                              (0, zmax, 32) if stellsym else (-zmax, zmax, 64), True, nfp=nfp, stellsym=stellsym)
    print(f"interpolated field built ({time.time() - t0:.0f}s)")
    stops = [MinRStoppingCriterion(rmin + 0.01), MaxRStoppingCriterion(rmax - 0.01),
             MinZStoppingCriterion(-zmax + 0.01), MaxZStoppingCriterion(zmax - 0.01)]

    R_ax, Z_ax = ax_rz[0]
    outer0 = sec_lcfs[0] if sec_lcfs[0] is not None else sec_opt[0]
    k_out = int(np.argmax(outer0[0]))
    R_out, Z_out = outer0[0][k_out], outer0[1][k_out]
    s = np.linspace(0.01, 1.15, args.lines_mid)
    seeds = list(zip(R_ax + s * (R_out - R_ax), Z_ax + s * (Z_out - Z_ax)))
    if xp_rz[0]:
        R_x, Z_x = xp_rz[0][0]
        s = np.linspace(0.55, 1.35, args.lines_x)
        seeds += list(zip(R_ax + s * (R_x - R_ax), Z_ax + s * (Z_x - Z_ax)))
    t0 = time.time()
    hits, failed = [], 0
    for R0, Z0 in seeds:                              # one line at a time: a single bad line must not kill the plot
        try:
            _, h = compute_fieldlines(field, [R0], [Z0], tmax=args.tmax, tol=1e-9, phis=list(2 * np.pi * phis),
                                      stopping_criteria=stops)
            hits.append(np.asarray(h[0]))
        except ValueError:
            hits.append(np.zeros((0, 5)))
            failed += 1
    print(f"traced {len(seeds)} lines, {failed} failed ({time.time() - t0:.0f}s)")

    fig, axs = plt.subplots(3, 3, figsize=(8.25, 8.3), sharex=True, sharey=True, gridspec_kw={"wspace": 0, "hspace": 0})
    for i, ax in enumerate(axs.flat):
        for k, h in enumerate(hits):
            sel = h[h[:, 1] == i] if h.size else h
            if sel.size:
                ax.scatter(np.hypot(sel[:, 2], sel[:, 3]), sel[:, 4], s=0.5, color=TAB20(k % 20), rasterized=True,
                           edgecolors="none", linewidths=0)
        if sec_vmex[i] is not None:
            ax.plot(*sec_vmex[i], "-", color="#2a78d6", lw=1.1, label="VMEX boundary", zorder=4)
        if sec_opt[i] is not None:
            ax.plot(*sec_opt[i], "k--", lw=0.9, label="optimization surface", zorder=4)
        if sec_lcfs[i] is not None:
            ax.plot(*sec_lcfs[i], "-", color="tab:purple", lw=1.1, label="LCFS", zorder=5)
        cc = coil_crossings(coils, phis[i])
        inside = (np.abs(cc[:, 0] - cx) < half) & (np.abs(cc[:, 1] - cz) < half) if cc.size else np.zeros(0, bool)
        if inside.any():
            ax.plot(cc[inside, 0], cc[inside, 1], "o", color="magenta", ms=4, ls="none", label="coil", zorder=6)
        ax.plot(*ax_rz[i], "o", color="tab:green", ms=4, ls="none", label="axis", zorder=6)
        for q in xp_rz[i]:
            ax.plot(*q, "X", color="tab:red", ms=5, ls="none", label="X-point", zorder=6)
        ax.text(0.5, 0.98, rf"$\phi/2\pi = {phis[i]:.4f}$", transform=ax.transAxes, ha="center", va="top", fontsize=8)
        ax.set_aspect("equal")
    axs[0, 0].set_xlim(cx - half, cx + half)
    axs[0, 0].set_ylim(cz - half, cz + half)
    # the panels abut (wspace = hspace = 0): keep ticks clear of the panel edges so neighbouring labels do not collide
    for lo, set_ticks in ((cx - half, axs[0, 0].set_xticks), (cz - half, axs[0, 0].set_yticks)):
        hi = lo + 2 * half
        t = MaxNLocator(nbins=5).tick_values(lo, hi)
        set_ticks(t[(t > lo + 0.1 * (hi - lo)) & (t < hi - 0.1 * (hi - lo))])
    for ax in axs[-1, :]:
        ax.set_xlabel("R [m]")
    for ax in axs[:, 0]:
        ax.set_ylabel("Z [m]")
    handles = {}
    for ax in axs.flat:
        for hdl, lbl in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lbl, hdl)
    fig.legend(handles.values(), handles.keys(), loc="lower center", bbox_to_anchor=(0.5, 0.885), ncol=4,
               frameon=False, markerscale=1.5)
    rng = "stellarator-symmetric half period" if stellsym else "full field period"
    fig.suptitle(f"{os.path.basename(os.path.dirname(dev))}: Poincaré sections, nfp = {nfp}, {rng}", y=0.985,
                 fontsize=10)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else dev
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "poincare_grid.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
