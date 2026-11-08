#!/usr/bin/env python3
"""Step 1/3 of the free-boundary Poincaré check: sources and targets for the plasma's own field.

usage: fb_field_prepare.py <wout.nc> <xpoint.json> <out.npz> [--src 512 512] [--h 0.005] [--nphi-grid 48]
                           [--delta 0.012] [--margin 0.04]
       fb_field_prepare.py <wout.nc> none <out.npz> --grid-mgrid <mgrid.nc> [--src 512 512] [--delta 0.012]

The plasma's field OUTSIDE its boundary is that of the surface current mu0 K = n x B_VMEX on the VMEX boundary (virtual
casing; continuous across a boundary with p_edge = 0). Inside, the same surface current gives B_K = -B_ext, so the
plasma's field there is B_K + B_VMEX. This script writes:
  - the source sheet (full torus, ``--src`` points per field period x poloidally): position and mu0 K dA,
  - a uniform (R, phi, Z) target grid over one field period (phi end points included) covering the VMEX boundary and
    the X-line with ``--margin`` to spare, each node classified inside/outside on its phi plane; with ``--grid-mgrid``
    instead the grid of that mgrid (phi end point excluded, as mgrid stores it), so the plasma field can be written as
    an extra field group of it (simflare workflows/flare_analysis/plasma_mgrid.py),
  - for nodes within ``--delta`` of the boundary (in the phi plane): the nearest boundary point p and the two points
    p -/+ delta n where the smooth plasma field is evaluated instead (the direct quadrature of the sheet is
    near-singular close to it); fb_field_assemble in fb_poincare.py blends linearly between them,
  - B_VMEX at every inside node and inner blend point (Delaunay-linear in each phi plane from the wout, s >= 0),
  - validation points boundary -/+ delta n (true 3-D normal) with the on-surface virtual-casing field for comparison.
The FMM step (fb_field_fmm.py, cadmag env) evaluates B_K at all target points.
"""
import argparse
import time

import numpy as np
from matplotlib.path import Path
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree


def read_wout_nc(path):
    """The wout arrays synth() needs, read with netCDF4 (no vmex / JAX import, so callers may fork afterwards)."""
    import types

    import netCDF4
    with netCDF4.Dataset(path) as nc:
        w = types.SimpleNamespace(**{k: np.array(nc[k][:]) for k in ("xm", "xn", "xm_nyq", "xn_nyq", "rmnc", "zmns",
                                                                      "bsupumnc", "bsupvmnc")})
        w.nfp, w.ns = int(nc["nfp"][:]), int(nc["ns"][:])
        w.lasym = bool(nc["lasym__logical__"][:]) if "lasym__logical__" in nc.variables else False
    return w


def synth(w, j, theta, phi):
    """R, Z, their theta/phi derivatives and B^theta, B^phi on full-mesh surface j: arrays (nphi, ntheta)."""
    xm, xn = np.asarray(w.xm, float), np.asarray(w.xn, float)
    xmn, xnn = np.asarray(w.xm_nyq, float), np.asarray(w.xn_nyq, float)
    rmnc, zmns = np.asarray(w.rmnc, float)[j], np.asarray(w.zmns, float)[j]
    ang = theta[None, :, None] * xm - phi[:, None, None] * xn
    c, s = np.cos(ang), np.sin(ang)
    R, Z = c @ rmnc, s @ zmns
    Ru, Rv = s @ (-rmnc * xm), s @ (rmnc * xn)
    Zu, Zv = c @ (zmns * xm), c @ (-zmns * xn)
    bu_h, bv_h = np.asarray(w.bsupumnc, float), np.asarray(w.bsupvmnc, float)
    ns = bu_h.shape[0]
    if j == ns - 1:
        bu, bv = 1.5 * bu_h[-1] - 0.5 * bu_h[-2], 1.5 * bv_h[-1] - 0.5 * bv_h[-2]
    elif j == 0:
        bu, bv = 0.0 * bu_h[1], bv_h[1]
    else:
        bu, bv = 0.5 * (bu_h[j] + bu_h[j + 1]), 0.5 * (bv_h[j] + bv_h[j + 1])
    angn = theta[None, :, None] * xmn - phi[:, None, None] * xnn
    cn = np.cos(angn)
    return R, Z, Ru, Rv, Zu, Zv, cn @ bu, cn @ bv


def cyl_B(R, Ru, Rv, Zu, Zv, bu, bv):
    """Cylindrical (B_R, B_phi, B_Z) of B = B^theta e_theta + B^phi e_phi."""
    return np.stack([bu * Ru + bv * Rv, bv * R, bu * Zu + bv * Zv], axis=-1)


def cart2cyl_vec(B, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([B[:, 0] * c + B[:, 1] * s, -B[:, 0] * s + B[:, 1] * c, B[:, 2]])


def cyl2cart_vec(B, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.column_stack([B[:, 0] * c - B[:, 1] * s, B[:, 0] * s + B[:, 1] * c, B[:, 2]])


def assemble_plasma_field(d, f):
    """The plasma's own field on the target grid of prepare ``d`` from the FMM result ``f``: (nP, nR, nZ, 3) cylindrical.

    B_K outside, B_K + B_VMEX inside, and in the skin the linear blend of those two evaluated at p -/+ delta n.
    """
    Pg, Rg, Zg, delta = d["Pg"], d["Rg"], d["Zg"], float(d["delta"])
    nP, nR, nZ = Pg.size, Rg.size, Zg.size
    PP = np.repeat(Pg, nR * nZ)
    Bp = f["BK_grid"].copy()                                           # Cartesian, (nP*nR*nZ, 3)
    flat = d["vin_k"] * nR * nZ + d["vin_i"]
    Bp[flat] += cyl2cart_vec(d["vin_B"], PP[flat])
    k, i = d["near_idx"][:, 0], d["near_idx"][:, 1]
    flat = k * nR * nZ + i
    Bm = f["BK_near_m"] + cyl2cart_vec(d["vmex_near_m"], d["near_m_phi"])
    Bo = f["BK_near_p"]
    wgt = ((d["near_d"] + delta) / (2 * delta))[:, None]
    Bp[flat] = Bm + wgt * (Bo - Bm)
    return cart2cyl_vec(Bp, PP).reshape(nP, nR, nZ, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("wout")
    ap.add_argument("xpoint")
    ap.add_argument("out")
    ap.add_argument("--src", type=int, nargs=2, default=(512, 512), help="sheet points per period (phi) and theta")
    ap.add_argument("--h", type=float, default=0.005, help="target grid spacing in R and Z [m]")
    ap.add_argument("--nphi-grid", type=int, default=48, help="grid intervals per field period")
    ap.add_argument("--delta", type=float, default=0.012, help="half width of the blended skin [m]")
    ap.add_argument("--margin", type=float, default=0.04, help="grid margin beyond boundary and X-line [m]")
    ap.add_argument("--grid-mgrid", default=None, help="use this mgrid's (R, phi, Z) grid as the target grid")
    args = ap.parse_args()

    from simsopt._core import load
    t0 = time.time()
    w = read_wout_nc(args.wout)
    nfp, ns = int(w.nfp), int(w.ns)
    if bool(getattr(w, "lasym", False)):
        raise SystemExit("lasym wout not supported")

    # ---- source sheet: full torus ---------------------------------------------------------------
    nps, nts = args.src
    th = np.linspace(0, 2 * np.pi, nts, endpoint=False)
    ph = np.linspace(0, 2 * np.pi, nfp * nps, endpoint=False)
    R, Z, Ru, Rv, Zu, Zv, bu, bv = synth(w, ns - 1, th, ph)
    cp, sp = np.cos(ph)[:, None], np.sin(ph)[:, None]

    def cart(vR, vP, vZ):
        return np.stack([vR * cp - vP * sp, vR * sp + vP * cp, vZ], axis=-1)
    gamma = cart(R, 0 * R, Z)
    e_t, e_p = cart(Ru, 0 * R, Zu), cart(Rv, R, Zv)
    B_tot = bu[..., None] * e_t + bv[..., None] * e_p
    dA = np.cross(e_t, e_p) * (2 * np.pi / nts) * (2 * np.pi / (nfp * nps))
    # outward: compare with the direction from the axis (j = 0) to the boundary
    R0, Z0 = synth(w, 0, th, ph)[:2]
    radial = cart(R - R0, 0 * R, Z - Z0)
    if np.mean(np.sum(dA * radial, axis=-1)) < 0:
        dA = -dA
    mu0KdA = np.cross(dA, B_tot)                        # (n x B) |dA|
    bn = np.abs(np.sum(dA * B_tot, -1)) / np.linalg.norm(dA, axis=-1) / np.linalg.norm(B_tot, axis=-1)
    print(f"sheet {gamma.shape[0]}x{gamma.shape[1]} points, max |B.n|/|B| of VMEX on it {bn.max():.1e} "
          f"({time.time() - t0:.0f}s)", flush=True)

    # ---- validation points: boundary +- delta n (3-D normal) at a few hundred sheet points -----------
    rng = np.random.default_rng(0)
    # on the 64 x 64 sub-grid of the first period, where fb_poincare.py compares with the on-surface virtual casing
    iv, jv = (nps // 64) * rng.integers(0, 64, 200), (nts // 64) * rng.integers(0, 64, 200)
    nhat = dA[iv, jv] / np.linalg.norm(dA[iv, jv], axis=-1, keepdims=True)
    val_pts = np.concatenate([gamma[iv, jv] - args.delta * nhat, gamma[iv, jv] + args.delta * nhat])

    # ---- target grid ------------------------------------------------------------------------------
    if args.grid_mgrid:
        import netCDF4
        with netCDF4.Dataset(args.grid_mgrid) as nc:
            g = {k: nc[k][:].item() for k in ("ir", "jz", "kp", "nfp", "rmin", "rmax", "zmin", "zmax")}
        if int(g["nfp"]) != nfp:
            raise SystemExit(f"mgrid nfp {g['nfp']} != wout nfp {nfp}")
        Rg = np.linspace(g["rmin"], g["rmax"], int(g["ir"]))
        Zg = np.linspace(g["zmin"], g["zmax"], int(g["jz"]))
        Pg = np.linspace(0, 2 * np.pi / nfp, int(g["kp"]), endpoint=False)
        if R.min() - args.delta < Rg[0] or R.max() + args.delta > Rg[-1] or \
                Z.min() - args.delta < Zg[0] or Z.max() + args.delta > Zg[-1]:
            raise SystemExit("VMEX boundary (+ delta) does not fit inside the mgrid box")
    else:
        xl = load(args.xpoint)["curve"].gamma()
        Rx, Zx = np.hypot(xl[:, 0], xl[:, 1]), np.abs(xl[:, 2])
        Rlo = min(R.min(), Rx.min()) - args.margin
        Rhi = max(R.max(), Rx.max()) + args.margin
        Zhi = max(np.abs(Z).max(), Zx.max()) + args.margin
        Rg = np.linspace(Rlo, Rhi, int(np.ceil((Rhi - Rlo) / args.h)) + 1)
        Zg = np.linspace(-Zhi, Zhi, int(np.ceil(2 * Zhi / args.h)) + 1)
        Pg = np.linspace(0, 2 * np.pi / nfp, args.nphi_grid + 1)
    nR, nZ = Rg.size, Zg.size
    print(f"grid R [{Rg[0]:.3f}, {Rg[-1]:.3f}] x Z [{Zg[0]:.3f}, {Zg[-1]:.3f}] x phi [0, 2pi/{nfp}]: {nR}x{nZ}x{Pg.size} "
          f"= {nR * nZ * Pg.size} nodes", flush=True)
    RR, ZZ = np.meshgrid(Rg, Zg, indexing="ij")
    flat_RZ = np.column_stack([RR.ravel(), ZZ.ravel()])

    nth_fine = 2048
    thf = np.linspace(0, 2 * np.pi, nth_fine, endpoint=False)
    sj = np.arange(ns)
    inside = np.zeros((Pg.size, nR, nZ), bool)
    dist = np.full((Pg.size, nR, nZ), np.inf)            # signed in-plane distance to the boundary (+ outside)
    near_idx, near_d, near_m, near_p = [], [], [], []     # node index, d, inner point, outer point (cylindrical R,Z)
    vmex_inside = {}                                       # (k, flat idx) -> B_VMEX cylindrical
    vmex_near_m = []
    for k, p in enumerate(Pg):
        Rb, Zb, Rbu, _, Zbu, _, _, _ = synth(w, ns - 1, thf, np.array([p]))
        Rb, Zb, Rbu, Zbu = Rb[0], Zb[0], Rbu[0], Zbu[0]
        poly = np.column_stack([Rb, Zb])
        ins = Path(poly).contains_points(flat_RZ)
        d, ii = cKDTree(poly).query(flat_RZ)
        sd = np.where(ins, -d, d)
        inside[k] = ins.reshape(nR, nZ)
        dist[k] = sd.reshape(nR, nZ)
        # in-plane outward normal at the nearest boundary point (tangent (Ru, Zu) rotated)
        tR, tZ = Rbu[ii], Zbu[ii]
        nn = np.column_stack([tZ, -tR]) / np.hypot(tR, tZ)[:, None]
        cen = np.array([Rb.mean(), Zb.mean()])
        flip = np.sign(np.sum((poly[ii] - cen) * nn, axis=1))
        nn *= flip[:, None]
        sel = np.flatnonzero(np.abs(sd) < args.delta)
        near_idx.append(np.column_stack([np.full(sel.size, k), sel]))
        near_d.append(sd[sel])
        pm = poly[ii[sel]] - args.delta * nn[sel]
        pp = poly[ii[sel]] + args.delta * nn[sel]
        near_m.append(np.column_stack([pm, np.full(sel.size, p)]))
        near_p.append(np.column_stack([pp, np.full(sel.size, p)]))
        # VMEX field in this plane on all full-mesh surfaces -> Delaunay-linear interpolant
        pts, Bs = [], []
        for j in sj:
            Rj, Zj, Ruj, Rvj, Zuj, Zvj, buj, bvj = synth(w, int(j), th[::2], np.array([p]))
            Bc = cyl_B(Rj, Ruj, Rvj, Zuj, Zvj, buj, bvj)[0]
            if j == 0:
                pts.append(np.array([[Rj[0, 0], Zj[0, 0]]]))
                Bs.append(Bc[:1])
            else:
                pts.append(np.column_stack([Rj[0], Zj[0]]))
                Bs.append(Bc)
        interp = LinearNDInterpolator(np.concatenate(pts), np.concatenate(Bs))
        sel_in = np.flatnonzero(ins)
        vmex_inside[k] = (sel_in, interp(flat_RZ[sel_in]))
        vmex_near_m.append(interp(pm))
    near_idx = np.concatenate(near_idx)
    near_d = np.concatenate(near_d)
    near_m = np.concatenate(near_m)
    near_p = np.concatenate(near_p)
    vmex_near_m = np.concatenate(vmex_near_m)
    vin_k = np.concatenate([np.full(v[0].size, k) for k, v in vmex_inside.items()])
    vin_i = np.concatenate([v[0] for v in vmex_inside.values()])
    vin_B = np.concatenate([v[1] for v in vmex_inside.values()])
    bad = ~np.isfinite(vin_B).all(1)
    if bad.any():   # inside the polygon but outside the hull of the VMEX sample (theta[::2] chords): nearest value
        good = ~bad
        tree = cKDTree(np.column_stack([vin_k, vin_i])[good].astype(float))
        vin_B[bad] = vin_B[good][tree.query(np.column_stack([vin_k, vin_i])[bad].astype(float))[1]]
    badm = ~np.isfinite(vmex_near_m).all(1)
    print(f"inside nodes {vin_i.size} ({bad.sum()} hull fallbacks), skin nodes {near_d.size} "
          f"({badm.sum()} inner points outside the VMEX hull) ({time.time() - t0:.0f}s)", flush=True)
    if badm.any():
        raise SystemExit("inner blend points outside the VMEX sample hull; lower --delta or refine")

    def cyl2cart(q):
        return np.column_stack([q[:, 0] * np.cos(q[:, 2]), q[:, 0] * np.sin(q[:, 2]), q[:, 1]])
    PP, RRR, ZZZ = np.meshgrid(Pg, Rg, Zg, indexing="ij")
    grid_xyz = np.column_stack([(RRR * np.cos(PP)).ravel(), (RRR * np.sin(PP)).ravel(), ZZZ.ravel()])
    np.savez(args.out, nfp=nfp, Rg=Rg, Zg=Zg, Pg=Pg, inside=inside, dist=dist, delta=args.delta,
             src_xyz=gamma.reshape(-1, 3), src_mu0KdA=mu0KdA.reshape(-1, 3),
             grid_xyz=grid_xyz, near_idx=near_idx, near_d=near_d,
             near_m_xyz=cyl2cart(near_m), near_p_xyz=cyl2cart(near_p), near_m_phi=near_m[:, 2],
             near_p_phi=near_p[:, 2], vmex_near_m=vmex_near_m,
             vin_k=vin_k, vin_i=vin_i, vin_B=vin_B,
             val_xyz=val_pts, val_iv=iv // (nps // 64), val_jv=jv // (nts // 64), val_Btot=B_tot[iv, jv])
    print(f"wrote {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
