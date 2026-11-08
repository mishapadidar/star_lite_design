#!/usr/bin/env python3
"""Step 3/3 of the free-boundary Poincaré check: trace coils-only vs coils + plasma field.

usage: fb_poincare.py <prepare.npz> <fmm.npz> <wout.nc> <coils.json> <xpoint.json> <out dir> [--tmax 2000]
                      [--nproc 8] [--no-vc-check]

The plasma field on the (R, phi, Z) grid of fb_field_prepare.py is B_K outside the VMEX boundary, B_K + B_VMEX inside,
and in the skin |d| < delta the linear blend of those two evaluated at p -/+ delta n (the plasma's own field is smooth
across a p_edge = 0 boundary). Validation printed first: the jump B_K(+delta) - B_K(-delta) must reproduce B_VMEX on
the boundary, and B_K(+delta) the on-surface virtual-casing plasma field (vmex, digits 4) to O(delta).

Both fields are traced in a simsopt InterpolatedField over one field period. Seeds at phi = 0 (outboard, Z = 0 by
stellarator symmetry): on VMEX surfaces s = 0.2 ... 1, outside the boundary up to the X-line radius, and along the
axis-to-X-point segment. Per seed on a VMEX surface the in-plane distance of its Poincaré hits to that surface (rms,
max) measures how well the field keeps the equilibrium's surfaces; for seeds outside, the fraction of hits that land
inside the VMEX boundary. Writes poincare_fb.png and fb_summary.yaml into <out dir>.
"""
import argparse
import multiprocessing as mp
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.path import Path  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from simsopt._core import load  # noqa: E402
from simsopt.field import BiotSavart, InterpolatedField, MagneticField, compute_fieldlines  # noqa: E402
from simsopt.field.tracing import (MaxRStoppingCriterion, MaxZStoppingCriterion, MinRStoppingCriterion,  # noqa: E402
                                   MinZStoppingCriterion)

from fb_field_prepare import assemble_plasma_field, cyl2cart_vec, read_wout_nc, synth  # noqa: E402

FIELDS = {}


class GridField(MagneticField):
    """Cubic-spline field from cylindrical components on a uniform (phi, R, Z) grid over one field period."""

    def __init__(self, Pg, Rg, Zg, Bcyl, nfp):
        MagneticField.__init__(self)
        self.nfp = nfp
        self.o = np.array([Pg[0], Rg[0], Zg[0]])
        self.h = np.array([Pg[1] - Pg[0], Rg[1] - Rg[0], Zg[1] - Zg[0]])
        self.coef = [ndimage.spline_filter(Bcyl[..., i], order=3, mode="nearest") for i in range(3)]

    def _B_impl(self, B):
        x = self.get_points_cart_ref()
        phi = np.arctan2(x[:, 1], x[:, 0])
        idx = np.vstack([(np.mod(phi, 2 * np.pi / self.nfp) - self.o[0]) / self.h[0],
                         (np.hypot(x[:, 0], x[:, 1]) - self.o[1]) / self.h[1], (x[:, 2] - self.o[2]) / self.h[2]])
        Bc = np.column_stack([ndimage.map_coordinates(c, idx, order=3, mode="nearest", prefilter=False)
                              for c in self.coef])
        B[:] = cyl2cart_vec(Bc, phi)


def trace_one(args):
    name, R0, Z0, tmax, phis = args
    field, stops = FIELDS[name]
    try:
        _, h = compute_fieldlines(field, [R0], [Z0], tmax=tmax, tol=1e-9, phis=phis, stopping_criteria=stops)
        return np.asarray(h[0])
    except ValueError:
        return np.zeros((0, 5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("prepare", "fmm", "wout", "coils", "xpoint", "outdir"):
        ap.add_argument(a)
    ap.add_argument("--tmax", type=float, default=2000.0)
    ap.add_argument("--nproc", type=int, default=8)
    ap.add_argument("--no-vc-check", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    d, f = np.load(args.prepare), np.load(args.fmm)
    w = read_wout_nc(args.wout)
    nfp, ns = int(d["nfp"]), int(w.ns)
    Pg, Rg, Zg = d["Pg"], d["Rg"], d["Zg"]
    nP, nR, nZ = Pg.size, Rg.size, Zg.size
    summary = {}

    # ---- validation ----------------------------------------------------------------------------
    nv = d["val_xyz"].shape[0] // 2
    BKm, BKp, Bt = f["BK_val"][:nv], f["BK_val"][nv:], d["val_Btot"]
    jump = np.linalg.norm(BKp - BKm - Bt, axis=1) / np.linalg.norm(Bt, axis=1)
    summary["check_jump_rel"] = dict(median=float(np.median(jump)), max=float(jump.max()))
    print(f"jump B_K(+d) - B_K(-d) vs B_VMEX: relative error median {np.median(jump):.2e} max {jump.max():.2e}")
    # ---- assemble the plasma field on the grid ----------------------------------------------------
    Bcyl = assemble_plasma_field(d, f)
    print(f"plasma field on the grid: |B_plasma| median {np.median(np.linalg.norm(Bcyl, axis=-1)):.2e} T, "
          f"max {np.linalg.norm(Bcyl, axis=-1).max():.2e} T", flush=True)

    # ---- fields ---------------------------------------------------------------------------------
    coils = load(args.coils)
    bs = BiotSavart(coils)
    grid = GridField(Pg, Rg, Zg, Bcyl, nfp)
    rr = (Rg[2], Rg[-3], 2 * nR // 3)
    pr = (0, 2 * np.pi / nfp, 2 * nP // 3)
    zr = (0, Zg[-3], nZ // 3)
    t0 = time.time()
    FIELDS["vac"] = InterpolatedField(bs, 4, rr, pr, zr, True, nfp=nfp, stellsym=True)
    FIELDS["full"] = InterpolatedField(bs + grid, 4, rr, pr, zr, True, nfp=nfp, stellsym=True)
    stops = [MinRStoppingCriterion(rr[0] + 0.005), MaxRStoppingCriterion(rr[1] - 0.005),
             MinZStoppingCriterion(-zr[1] + 0.005), MaxZStoppingCriterion(zr[1] - 0.005)]
    for key in FIELDS:
        FIELDS[key] = (FIELDS[key], stops)
    print(f"interpolated fields built ({time.time() - t0:.0f}s)", flush=True)

    # ---- seeds ---------------------------------------------------------------------------------
    s_seed = [0.2, 0.4, 0.6, 0.8, 0.9, 1.0]
    sv = np.linspace(0, 1, ns)
    th0 = np.array([0.0])
    seeds = []
    for s in s_seed:
        j = int(np.argmin(np.abs(sv - s)))
        R, Z = synth(w, j, th0, np.array([0.0]))[:2]
        seeds.append(dict(kind="surface", s=float(sv[j]), j=j, R=float(R[0, 0]), Z=0.0))
    Rb = seeds[-1]["R"]
    xl = load(args.xpoint)["curve"]
    g = xl.gamma()
    phx = np.mod(np.arctan2(g[:, 1], g[:, 0]), 2 * np.pi)
    kx = int(np.argmin(np.minimum(phx, 2 * np.pi - phx)))
    Rx, Zx = float(np.hypot(g[kx, 0], g[kx, 1])), float(g[kx, 2])
    R_ax = synth(w, 0, th0, np.array([0.0]))[0][0, 0]
    for dR in (0.003, 0.006, 0.01, 0.015, 0.02, 0.03, 0.04):
        if Rb + dR < rr[1] - 0.01:
            seeds.append(dict(kind="outside", dR=dR, R=Rb + dR, Z=0.0))
    for sx in np.linspace(0.8, 1.2, 9):
        seeds.append(dict(kind="xline", frac=float(sx), R=float(R_ax + sx * (Rx - R_ax)), Z=float(sx * Zx)))
    phis_frac = np.array([0.0, 1 / 16, 1 / 8]) * (4 / nfp)          # phi/2pi: 0, quarter and half of a half period
    phis = list(2 * np.pi * phis_frac)

    jobs = [(name, s["R"], s["Z"], args.tmax, phis) for name in ("vac", "full") for s in seeds]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.nproc) as pool:
        hits = pool.map(trace_one, jobs, chunksize=1)
    print(f"traced {len(jobs)} lines ({time.time() - t0:.0f}s)", flush=True)
    H = {"vac": hits[:len(seeds)], "full": hits[len(seeds):]}

    # ---- VMEX contours per plane + metrics ---------------------------------------------------------
    thf = np.linspace(0, 2 * np.pi, 1024)
    cont = {}
    for ip, p in enumerate(phis):
        for j in sorted({s["j"] for s in seeds if s["kind"] == "surface"} | {ns // 4, ns // 2, 3 * ns // 4, ns - 1}):
            R, Z = synth(w, j, thf, np.array([p]))[:2]
            cont[(ip, j)] = np.column_stack([R[0], Z[0]])
    rows = []
    for name in ("vac", "full"):
        for s, h in zip(seeds, H[name]):
            sel = h[h[:, 1] >= 0] if h.size else h
            nturn = int(np.sum(sel[:, 1] == 0)) if sel.size else 0
            stopped = bool(h.size and (h[:, 1] < 0).any())
            row = dict(field=name, seed=s["kind"], R0=round(s["R"], 4), hits_phi0=nturn, lost=stopped)
            if s["kind"] == "surface":
                ds = []
                for ip in range(len(phis)):
                    q = sel[sel[:, 1] == ip]
                    if q.size:
                        ds.append(cKDTree(cont[(ip, s["j"])]).query(np.column_stack([np.hypot(q[:, 2], q[:, 3]),
                                                                                    q[:, 4]]))[0])
                ds = np.concatenate(ds) if ds else np.array([np.nan])
                row.update(s=round(s["s"], 3), dist_rms_mm=round(1e3 * float(np.sqrt(np.mean(ds ** 2))), 2),
                           dist_max_mm=round(1e3 * float(np.max(ds)), 2))
            else:
                ins = []
                for ip in range(len(phis)):
                    q = sel[sel[:, 1] == ip]
                    if q.size:
                        ins.append(Path(cont[(ip, ns - 1)]).contains_points(
                            np.column_stack([np.hypot(q[:, 2], q[:, 3]), q[:, 4]])))
                ins = np.concatenate(ins) if ins else np.array([False])
                row.update(frac_inside_vmex=round(float(np.mean(ins)), 3))
                row.update({k: round(v, 4) for k, v in s.items() if k in ("dR", "frac")})
            rows.append(row)
            print(row, flush=True)
    if not args.no_vc_check:   # JAX: only after the fork pool is done
        from vmex.core import virtual_casing as vc
        from vmex.core.wout import read_wout
        sd = vc.surface_field_data_from_wout(read_wout(args.wout), nphi=64, ntheta=64)
        Bs = np.moveaxis(np.asarray(vc.plasma_field_on_boundary(sd, digits=4)), 0, -1)[d["val_iv"], d["val_jv"]]
        e = np.linalg.norm(BKp - Bs, axis=1)
        summary["check_vc_onsurface_T"] = dict(median=float(np.median(e)), max=float(e.max()),
                                               Bplasma_median=float(np.median(np.linalg.norm(Bs, axis=1))))
        print(f"B_K(+d) vs on-surface virtual casing: |diff| median {np.median(e):.2e} T max {e.max():.2e} T "
              f"(|B_plasma| median {np.median(np.linalg.norm(Bs, axis=1)):.2e} T)", flush=True)

    summary["lines"] = rows
    summary["tmax_m"] = args.tmax
    yaml.safe_dump(summary, open(os.path.join(args.outdir, "fb_summary.yaml"), "w"), sort_keys=False)

    # ---- figure --------------------------------------------------------------------------------
    allp = np.concatenate([c for c in cont.values()])
    pts = [np.column_stack([np.hypot(h[:, 2], h[:, 3]), h[:, 4]]) for hh in H.values() for h in hh if h.size]
    pts = np.concatenate(pts + [allp])
    cx, cz = 0.5 * (allp[:, 0].min() + allp[:, 0].max()), 0.0
    half = 1.12 * max(np.abs(pts[:, 0] - cx).max(), np.abs(pts[:, 1]).max())
    fig, axs = plt.subplots(len(phis), 2, figsize=(7.2, 3.5 * len(phis)), sharex=True, sharey=True,
                            gridspec_kw={"wspace": 0.03, "hspace": 0.03})
    cmap = plt.get_cmap("tab20")
    for c, name in enumerate(("vac", "full")):
        for ip in range(len(phis)):
            ax = axs[ip, c]
            for kk, h in enumerate(H[name]):
                q = h[h[:, 1] == ip] if h.size else h
                if q.size:
                    ax.scatter(np.hypot(q[:, 2], q[:, 3]), q[:, 4], s=0.4, color=cmap(kk % 20), edgecolors="none",
                               rasterized=True)
            for (jp, j), cc in cont.items():
                if jp == ip and j in (ns // 4, ns // 2, 3 * ns // 4, ns - 1):
                    ax.plot(cc[:, 0], cc[:, 1], "-", color="k" if j == ns - 1 else "0.45",
                            lw=0.9 if j == ns - 1 else 0.5, label="VMEX boundary" if j == ns - 1 else "VMEX surfaces")
            pf = phis[ip]
            kx = int(np.argmin(np.abs(np.mod(phx - pf + np.pi, 2 * np.pi) - np.pi)))
            ax.plot(np.hypot(g[kx, 0], g[kx, 1]), g[kx, 2], "X", color="tab:red", ms=6, label="optimizer X-line")
            kb = int(np.argmin(np.abs(np.mod(phx + pf + np.pi, 2 * np.pi) - np.pi)))  # lower null: (R, -phi, -Z)
            ax.plot(np.hypot(g[kb, 0], g[kb, 1]), -g[kb, 2], "X", color="tab:red", ms=6)
            ax.set_aspect("equal")
            ax.text(0.03, 0.97, rf"$\phi/2\pi={phis_frac[ip]:.4f}$", transform=ax.transAxes, va="top", fontsize=8)
        axs[0, c].set_title("coils only" if name == "vac" else "coils + plasma (virtual casing)", fontsize=9)
    axs[0, 0].set_xlim(cx - half, cx + half)
    axs[0, 0].set_ylim(-half, half)
    for ax in axs[-1]:
        ax.set_xlabel("R [m]")
    for ax in axs[:, 0]:
        ax.set_ylabel("Z [m]")
    hd = {}
    for ax in axs.flat:
        for a, b in zip(*ax.get_legend_handles_labels()):
            hd.setdefault(b, a)
    fig.canvas.draw()
    fig.legend(hd.values(), hd.keys(), loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, axs[0, 0].get_position().y1 + 0.02))
    out = os.path.join(args.outdir, "poincare_fb.png")
    fig.savefig(out, dpi=250, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
