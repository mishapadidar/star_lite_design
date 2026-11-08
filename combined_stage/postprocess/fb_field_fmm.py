#!/usr/bin/env python3
"""Step 2/3 of the free-boundary Poincaré check: Biot-Savart of the virtual-casing sheet by FMM.

usage (cadmag env, OpenMP fmm3dpy): fb_field_fmm.py <prepare.npz> <out.npz> [--eps 1e-7]

B_K(x) = curl A, A(x) = (1/4 pi) sum_j (mu0 K dA)_j / |x - y_j| -- three Laplace charge densities, gradients at the
targets: the grid nodes, the skin blend points (inner / outer) and the validation points of fb_field_prepare.py.
Writes B_K (Cartesian) for each target set.
"""
import argparse
import time

import fmm3dpy
import numpy as np


def biot_savart_sheet(src, c, targets, eps):
    out = fmm3dpy.lfmm3d(eps=eps, sources=np.asfortranarray(src.T), charges=np.asfortranarray(c.T),
                         targets=np.asfortranarray(targets.T), pg=0, pgt=2, nd=3)
    g = np.asarray(out.gradtarg).reshape(3, 3, -1)      # g[i, k] = d A_i / d x_k (FMM3D's kernel is 1/(4 pi r))
    return np.stack([g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]], axis=-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prepare")
    ap.add_argument("out")
    ap.add_argument("--eps", type=float, default=1e-7)
    args = ap.parse_args()
    d = np.load(args.prepare)
    sets = {k: d[k] for k in ("grid_xyz", "near_m_xyz", "near_p_xyz", "val_xyz")}
    allt = np.concatenate(list(sets.values()))
    t0 = time.time()
    B = biot_savart_sheet(d["src_xyz"], d["src_mu0KdA"], allt, args.eps)
    print(f"FMM: {d['src_xyz'].shape[0]} sources, {allt.shape[0]} targets, {time.time() - t0:.0f}s", flush=True)
    res, i = {}, 0
    for k, v in sets.items():
        res["BK_" + k.replace("_xyz", "")] = B[i:i + v.shape[0]]
        i += v.shape[0]
    np.savez(args.out, **res)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
