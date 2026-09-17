#!/usr/bin/env python3
"""Remove a degenerate poloidal parametrization (collapsing |dx/dtheta|, a self-intersecting loop) from a VMEC boundary
with the smallest shape change.

usage: repair_boundary.py <namelist in> <namelist out> [--max-mode 2 | --all-modes] [--objective point|normal]
                          [--min-speed-ratio 0.15] [--nphi 16] [--ntheta 720]

Minimizes a displacement measure over a (phi, theta) grid of one field period (half period if stellarator symmetric)
subject to |dx/dtheta| >= min_speed_ratio x its cross-section median (quadratic penalty, escalated until met).
  --objective point   mean |x_new(theta, phi) - x_old(theta, phi)|^2 at equal theta (counts tangential sliding)
  --objective normal  mean ((x_new - x_old) . n_old)^2 with n_old the original in-plane normal, points where the original
                      speed is collapsed (< 0.3 x median) excluded; a 1e-3 x point term keeps the parametrization tame
Varied modes: vmex.core.optimize._dof_modes(inp, max_mode). With --max-mode 2 the repaired boundary differs from the
input only in the modes a vmex.max_mode = 2 run varies, so combined_stage_vmex.py --plasma-target reaches it from a seed
with the same fixed modes; --all-modes varies every mode (the result must then solve cold and serve as the seed itself).
Reports the geometric change (nearest-point distance per cross-section, 32 planes x 4096 points: max, max outside the
original collapsed region, rms; volume change), and the min speed ratio and simsopt is_self_intersecting of input/output.
"""
import argparse

import numpy as np


def cross_sections(path, nfp, nplanes=32, nth=4096, stellsym=True):
    from simsopt.geo import SurfaceRZFourier
    span = (0.5 if stellsym else 1.0) / nfp
    out = []
    for p in np.linspace(0.0, span, nplanes, endpoint=False):
        s = SurfaceRZFourier.from_vmec_input(path, quadpoints_phi=[p], quadpoints_theta=np.linspace(0, 1, nth, endpoint=False))
        g = s.gamma()[0]
        out.append((p, np.column_stack([np.hypot(g[:, 0], g[:, 1]), g[:, 2]]), s.is_self_intersecting(angle=p, thetas=nth)))
    return out


def densify(c, k=8):
    c2 = np.roll(c, -1, axis=0)
    return (c[:, None, :] + np.linspace(0, 1, k, endpoint=False)[None, :, None] * (c2 - c)[:, None, :]).reshape(-1, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("namelist_in")
    ap.add_argument("namelist_out")
    ap.add_argument("--max-mode", type=int, default=2)
    ap.add_argument("--all-modes", action="store_true")
    ap.add_argument("--objective", choices=("point", "normal"), default="normal")
    ap.add_argument("--min-speed-ratio", type=float, default=0.15)
    ap.add_argument("--nphi", type=int, default=16)
    ap.add_argument("--ntheta", type=int, default=720)
    args = ap.parse_args()

    from dataclasses import replace

    import jax
    import jax.numpy as jnp
    import vmex as vj
    from scipy.optimize import minimize
    from scipy.spatial import cKDTree
    from simsopt.geo import SurfaceRZFourier
    from vmex.core.optimize import _dof_modes
    jax.config.update("jax_enable_x64", True)

    inp = vj.VmecInput.from_file(args.namelist_in)
    nfp, ntor, stellsym = int(inp.nfp), int(inp.ntor), not bool(inp.lasym)
    rbc0, zbs0 = np.asarray(inp.rbc, dtype=float), np.asarray(inp.zbs, dtype=float)
    max_mode = max(rbc0.shape[1], ntor) if args.all_modes else args.max_mode
    modes = [(m, n) for m, n in _dof_modes(inp, max_mode) if m < rbc0.shape[1] and abs(n) <= ntor]
    idx = (np.array([n + ntor for m, n in modes]), np.array([m for m, n in modes]))
    m = jnp.arange(rbc0.shape[1], dtype=float)[None, None, None, :]
    n = (jnp.arange(rbc0.shape[0], dtype=float) - ntor)[None, None, :, None]
    span = (0.5 if stellsym else 1.0) * 2 * np.pi / nfp
    phi = jnp.linspace(0.0, span, args.nphi, endpoint=False)[:, None, None, None]
    theta = jnp.linspace(0.0, 2 * np.pi, args.ntheta, endpoint=False)[None, :, None, None]
    ang = m * theta - n * nfp * phi
    C, S = jnp.cos(ang), jnp.sin(ang)
    k = len(modes)

    def geometry(dofs):
        rbc = jnp.asarray(rbc0).at[idx].set(dofs[:k])
        zbs = jnp.asarray(zbs0).at[idx].set(dofs[k:])
        R = jnp.einsum("ptnm,nm->pt", C, rbc)
        Z = jnp.einsum("ptnm,nm->pt", S, zbs)
        Rt = jnp.einsum("ptnm,nm->pt", -S * m, rbc)
        Zt = jnp.einsum("ptnm,nm->pt", C * m, zbs)
        return R, Z, Rt, Zt

    x0 = np.concatenate([rbc0[idx], zbs0[idx]])
    R0, Z0, Rt0, Zt0 = geometry(jnp.asarray(x0))
    sp0 = jnp.hypot(Rt0, Zt0)
    nR, nZ = Zt0 / jnp.maximum(sp0, 1e-300), -Rt0 / jnp.maximum(sp0, 1e-300)
    wmask = (sp0 >= 0.3 * jnp.median(sp0, axis=1, keepdims=True)).astype(float)

    def objective(dofs, weight):
        R, Z, Rt, Zt = geometry(dofs)
        sp = jnp.hypot(Rt, Zt)
        point = jnp.mean((R - R0) ** 2 + (Z - Z0) ** 2) / 1e-6
        if args.objective == "normal":
            disp = jnp.sum(wmask * ((R - R0) * nR + (Z - Z0) * nZ) ** 2) / jnp.sum(wmask) / 1e-6 + 1e-3 * point
        else:
            disp = point
        viol = jnp.maximum(args.min_speed_ratio - sp / jax.lax.stop_gradient(jnp.median(sp, axis=1, keepdims=True)), 0.0)
        return disp + weight * jnp.mean(viol ** 2)

    val_grad = jax.jit(jax.value_and_grad(objective))
    x, weight = x0.copy(), 1e4
    print(f"objective {args.objective}, {k} modes x 2 families varied ({'all modes' if args.all_modes else f'max_mode {max_mode}'}); "
          f"input min speed ratio {float(jnp.min(sp0 / jnp.median(sp0, axis=1, keepdims=True))):.3e}", flush=True)
    for _ in range(12):
        res = minimize(lambda v: tuple(np.asarray(a, dtype=float) for a in val_grad(jnp.asarray(v), weight)), x,
                       jac=True, method="L-BFGS-B", options=dict(maxiter=1000, gtol=1e-12))
        x = res.x
        _, _, Rt, Zt = geometry(jnp.asarray(x))
        sp = jnp.hypot(Rt, Zt)
        worst = float(jnp.min(sp / jnp.median(sp, axis=1, keepdims=True)))
        print(f"  weight {weight:.0e}: min speed ratio {worst:.4f} ({res.nit} it)", flush=True)
        if worst >= 0.98 * args.min_speed_ratio:
            break
        weight *= 10.0
    rbc, zbs = rbc0.copy(), zbs0.copy()
    rbc[idx], zbs[idx] = x[:k], x[k:]
    replace(inp, rbc=rbc, zbs=zbs).to_indata(args.namelist_out)
    dx = np.concatenate([rbc[idx] - rbc0[idx], zbs[idx] - zbs0[idx]])
    print(f"wrote {args.namelist_out}; largest coefficient changes: "
          + ", ".join(f"{'rbc' if i < k else 'zbs'}({modes[i % k][0]},{modes[i % k][1]}) {dx[i]:+.2e}"
                      for i in np.argsort(np.abs(dx))[::-1][:6]), flush=True)

    cin, cout = cross_sections(args.namelist_in, nfp, stellsym=stellsym), cross_sections(args.namelist_out, nfp, stellsym=stellsym)
    haus, outside, rms, worst_out = [], [], [], np.inf
    for (p, o, _), (_, r, _) in zip(cin, cout):
        d_ro = cKDTree(densify(o)).query(r)[0]
        d_or = cKDTree(densify(r)).query(o)[0]
        spo = np.hypot(*np.gradient(o, axis=0).T)
        loop = spo < 0.1 * np.median(spo)
        haus.append(max(d_ro.max(), d_or.max()))
        outside.append(d_or[~loop].max())
        rms.append(np.sqrt(np.mean(np.r_[d_ro, d_or] ** 2)))
        spr = np.hypot(*np.gradient(r, axis=0).T)
        worst_out = min(worst_out, float(spr.min() / np.median(spr)))
    i = int(np.argmax(haus))
    vo = SurfaceRZFourier.from_vmec_input(args.namelist_in).volume()
    vr = SurfaceRZFourier.from_vmec_input(args.namelist_out).volume()
    print(f"geometric change: max {max(haus) * 1e3:.3f} mm at phi*nfp/2pi {cin[i][0] * nfp:.3f}, max outside the original "
          f"collapsed region {max(outside) * 1e3:.3f} mm, mean rms {np.mean(rms) * 1e3:.3f} mm; volume {(vr / vo - 1) * 100:+.3f} %")
    print(f"input: self-intersecting at phi*nfp/2pi {[round(p * nfp, 3) for p, _, si in cin if si] or 'none'}")
    print(f"output: min speed ratio {worst_out:.4f}; self-intersecting at phi*nfp/2pi {[round(p * nfp, 3) for p, _, si in cout if si] or 'none'}")


if __name__ == "__main__":
    main()
