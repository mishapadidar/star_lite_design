#!/usr/bin/env python3
"""Lower the Fourier resolution of a VMEC boundary without changing its shape: refit instead of truncating.

usage: refit_boundary_resolution.py <namelist in> <namelist out> --mpol M --ntor N [--nphi 32] [--ntheta 256]
                                    [--point-weight 1e-3]

VmecInput.change_resolution (what combined_stage_vmex.py does for vmex.mpol / vmex.ntor) keeps the coefficients with
m < M, |n| <= N and drops the rest, so the boundary moves by whatever the dropped modes carried. Part of a high-m tail
can be poloidal-angle parametrization (points sliding along the surface) rather than shape. This tool fits all m < M,
|n| <= N coefficients to the original surface by least squares along the original normal (tangential sliding is free
to first order; a small pointwise term keeps the angle tame), starting from the truncation, and writes the namelist at
the reduced resolution. Reports, for plain truncation and for the refit, the geometric distance to the original
surface (nearest point per cross-section, 32 planes of the half period x 4096 points densified x 8, both directions:
max and rms) and the largest dropped coefficient.
"""
import argparse
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree


def cross_sections(inp, nplanes=32, nth=4096):
    rbc, zbs = np.asarray(inp.rbc, dtype=float), np.asarray(inp.zbs, dtype=float)
    nt, mp, nfp = (rbc.shape[0] - 1) // 2, rbc.shape[1], int(inp.nfp)
    m = np.arange(mp)[None, None, :]
    n = (np.arange(2 * nt + 1) - nt)[None, :, None]
    th = np.linspace(0, 2 * np.pi, nth, endpoint=False)[:, None, None]
    out = []
    for phi in np.linspace(0, np.pi / nfp, nplanes, endpoint=False):
        ang = m * th - n * nfp * phi
        out.append(np.column_stack([np.einsum("tnm,nm->t", np.cos(ang), rbc), np.einsum("tnm,nm->t", np.sin(ang), zbs)]))
    return out


def densify(c, k=8):
    c2 = np.roll(c, -1, axis=0)
    return (c[:, None, :] + np.linspace(0, 1, k, endpoint=False)[None, :, None] * (c2 - c)[:, None, :]).reshape(-1, 2)


def geometric_distance(ref_inp, cand_inp):
    """(max, rms) nearest-point distance between two boundaries over 32 cross-sections, both directions, metres."""
    worst, sq, cnt = 0.0, 0.0, 0
    for a, b in zip(cross_sections(ref_inp), cross_sections(cand_inp)):
        d1 = cKDTree(densify(a)).query(b)[0]
        d2 = cKDTree(densify(b)).query(a)[0]
        worst = max(worst, float(d1.max()), float(d2.max()))
        sq += float(np.sum(d1 ** 2) + np.sum(d2 ** 2))
        cnt += d1.size + d2.size
    return worst, float(np.sqrt(sq / cnt))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("namelist_in")
    ap.add_argument("namelist_out")
    ap.add_argument("--mpol", type=int, required=True, help="poloidal modes m = 0 .. mpol-1 (VMEC convention)")
    ap.add_argument("--ntor", type=int, required=True)
    ap.add_argument("--nphi", type=int, default=32)
    ap.add_argument("--ntheta", type=int, default=256)
    ap.add_argument("--point-weight", type=float, default=1e-3)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    import vmex as vj
    from scipy.optimize import minimize
    jax.config.update("jax_enable_x64", True)

    inp = vj.VmecInput.from_file(args.namelist_in)
    nfp = int(inp.nfp)
    rbc0, zbs0 = np.asarray(inp.rbc, dtype=float), np.asarray(inp.zbs, dtype=float)
    nt0 = (rbc0.shape[0] - 1) // 2
    low = inp.change_resolution(mpol=args.mpol, ntor=args.ntor)
    rbcL, zbsL = np.asarray(low.rbc, dtype=float), np.asarray(low.zbs, dtype=float)
    nt = args.ntor

    dropped = [(abs(rbc0[i, j]), "rbc", j, i - nt0) for i in range(rbc0.shape[0]) for j in range(rbc0.shape[1])
               if j >= args.mpol or abs(i - nt0) > nt] + \
              [(abs(zbs0[i, j]), "zbs", j, i - nt0) for i in range(zbs0.shape[0]) for j in range(zbs0.shape[1])
               if j >= args.mpol or abs(i - nt0) > nt]
    dropped.sort(reverse=True)
    print(f"{args.namelist_in}: MPOL {inp.mpol} NTOR {inp.ntor} -> mpol {args.mpol} (m <= {args.mpol - 1}), ntor {nt}; "
          f"largest dropped coefficients [mm]: " + ", ".join(f"{f}(m={m},n={n}) {v * 1e3:.3f}" for v, f, m, n in dropped[:6]),
          flush=True)

    phi = jnp.linspace(0.0, np.pi / nfp, args.nphi, endpoint=False)[:, None, None, None]
    th = jnp.linspace(0.0, 2 * np.pi, args.ntheta, endpoint=False)[None, :, None, None]

    def tables(shape, ntor):
        mm = jnp.arange(shape[1], dtype=float)[None, None, None, :]
        nn = (jnp.arange(shape[0], dtype=float) - ntor)[None, None, :, None]
        ang = mm * th - nn * nfp * phi
        return jnp.cos(ang), jnp.sin(ang), mm

    C0, S0, m0 = tables(rbc0.shape, nt0)
    R0 = jnp.einsum("ptnm,nm->pt", C0, rbc0)
    Z0 = jnp.einsum("ptnm,nm->pt", S0, zbs0)
    Rt0 = jnp.einsum("ptnm,nm->pt", -S0 * m0, rbc0)
    Zt0 = jnp.einsum("ptnm,nm->pt", C0 * m0, zbs0)
    sp0 = jnp.hypot(Rt0, Zt0)
    nR, nZ = Zt0 / sp0, -Rt0 / sp0
    CL, SL, _ = tables(rbcL.shape, nt)

    maskR = np.ones(rbcL.shape, dtype=bool)
    maskR[:nt, 0] = False                       # m = 0, n < 0 duplicates
    maskZ = maskR.copy()
    maskZ[nt, 0] = False                        # zbs(0, 0) multiplies sin(0)
    iR, iZ = np.nonzero(maskR), np.nonzero(maskZ)
    kR = len(iR[0])

    def unpack(d):
        return jnp.zeros(rbcL.shape).at[iR].set(d[:kR]), jnp.zeros(zbsL.shape).at[iZ].set(d[kR:])

    def objective(d):
        rbc, zbs = unpack(d)
        R = jnp.einsum("ptnm,nm->pt", CL, rbc)
        Z = jnp.einsum("ptnm,nm->pt", SL, zbs)
        normal = jnp.mean(((R - R0) * nR + (Z - Z0) * nZ) ** 2) / 1e-6
        point = jnp.mean((R - R0) ** 2 + (Z - Z0) ** 2) / 1e-6
        return normal + args.point_weight * point

    vg = jax.jit(jax.value_and_grad(objective))
    x0 = np.concatenate([rbcL[iR], zbsL[iZ]])
    f0 = float(vg(jnp.asarray(x0))[0])
    res = minimize(lambda v: tuple(np.asarray(a, dtype=float) for a in vg(jnp.asarray(v))), x0, jac=True,
                   method="L-BFGS-B", options=dict(maxiter=5000, gtol=1e-14, ftol=1e-16))
    rbc, zbs = (np.asarray(a) for a in unpack(jnp.asarray(res.x)))
    fit = replace(low, rbc=rbc, zbs=zbs)
    fit.to_indata(args.namelist_out)
    print(f"refit: objective (mm^2, normal + {args.point_weight:g} x pointwise) {f0:.4e} -> {res.fun:.4e} in {res.nit} it "
          f"({res.message}); wrote {args.namelist_out}", flush=True)
    for label, cand in (("truncation", low), ("refit", fit)):
        mx, rms = geometric_distance(inp, cand)
        print(f"geometric distance to the MPOL {inp.mpol} / NTOR {inp.ntor} boundary, {label:10s}: max {mx * 1e3:.3f} mm, "
              f"rms {rms * 1e3:.3f} mm", flush=True)


if __name__ == "__main__":
    main()
