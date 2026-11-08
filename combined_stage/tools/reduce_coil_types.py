#!/usr/bin/env python3
"""Seed coils with fewer coil types (base coils per half period) from a run's stellarator-symmetric coils.

usage: reduce_coil_types.py <coils json> <n_new> <out json> [--plane-coil] [--vmec-input <namelist>] [--xpoint <xpoint json>]

The old base coils (the curves owning dofs) sit at centroid angles phi_i in (0, pi/nfp). New coil j goes to
phi_j = (j + 1/2) pi / (nfp n_new). Its points are the pointwise blend (1 - w) R(phi_j - phi_a) gamma_a(t) +
w R(phi_j - phi_b) gamma_b(t) of the two old base coils bracketing phi_j, each rotated about z onto phi_j (w linear in
angle), fitted by a CurveXYZFourier of the same order. The pointwise blend assumes the old coils share their
parametrization origin (true for coils grown from create_equally_spaced_curves; printed as a check). Its current is
the same blend of the old currents times n_old / n_new, so the total poloidal current, hence the toroidal field, is
unchanged. Prints lengths, curvature, coil-coil distance and, with --vmec-input, rms B.n/B on that boundary and the
coil-plasma distance, and, with --xpoint, the X-line's field-line residual, for the old and the new coil set.

--plane-coil adds one coil per field period ON the phi = 0 symmetry plane, its own stellarator image, so it is copied
only by the nfp rotations: nfp (2 n_new + 1) coils in all (nfp 4: n_new 1 -> 12 coils, n_new 0 -> 4), equally spaced,
the generic coils at phi_j = j 2 pi / (nfp (2 n_new + 1)), j = 1..n_new. The plane coil is the old coil nearest phi = 0
rotated onto it and symmetrized, ((x, y, z)(t) + (x, -y, -z)(-t)) / 2 with t = 0 moved onto the symmetry line, and its
odd-parity coefficients (xs, yc, zc) set to exactly zero, so the driver's fix_self_symmetric_coil_parity pins them.
"""
import argparse

import numpy as np
from simsopt._core import load, save
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (CurveCurveDistance, CurveLength, CurveXYZFourier, MeanSquaredCurvature, SurfaceRZFourier)


def rot_z(a):
    return np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])


def base_coils(coils):
    """(curve, current value, centroid angle) of the coils whose curves own their dofs, sorted by angle."""
    out = []
    for c in coils:
        if type(c.curve).__name__ == "RotatedCurve":
            continue
        cen = c.curve.gamma().mean(axis=0)
        out.append((c.curve, float(c.current.get_value()), float(np.arctan2(cen[1], cen[0]))))
    return sorted(out, key=lambda b: b[2])


def report(label, coils, nml=None, xpoint=None):
    curves = [c.curve for c in coils]
    msg = (f"{label}: {len(coils)} coils, length max {max(CurveLength(c).J() for c in curves):.3f} m, curvature max "
           f"{max(np.max(c.kappa()) for c in curves):.2f} /m, msc max {max(MeanSquaredCurvature(c).J() for c in curves):.2f}, "
           f"coil-coil min {CurveCurveDistance(curves, 0.0).shortest_distance() * 100:.2f} cm, "
           f"|I| max {max(abs(c.current.get_value()) for c in coils) / 1e3:.1f} kA, sum |I| "
           f"{sum(abs(c.current.get_value()) for c in coils) / 1e6:.3f} MA")
    if nml:
        s = SurfaceRZFourier.from_vmec_input(nml, quadpoints_phi=np.linspace(0, 1, 256, endpoint=False),
                                             quadpoints_theta=np.linspace(0, 1, 64, endpoint=False))
        pts, n = s.gamma().reshape(-1, 3), s.normal().reshape(-1, 3)
        bs = BiotSavart(coils)
        bs.set_points(pts)
        B = bs.B()
        w = np.linalg.norm(n, axis=1)
        bn = np.sum(B * n, axis=1) / (w * np.linalg.norm(B, axis=1))
        dmin = min(np.min(np.linalg.norm(c.gamma()[:, None, :] - pts[None, ::7, :], axis=-1)) for c in curves)
        msg += f"; on the boundary: rms B.n/B {np.sqrt(np.sum(w * bn ** 2) / np.sum(w)):.3e}, max {np.max(np.abs(bn)):.3e}, coil-plasma min {dmin * 100:.1f} cm"
    if xpoint:
        from star_lite_design.utils.periodicfieldline import field_line_residual
        seed = load(xpoint)
        r = field_line_residual(seed["curve"], float(seed["length"]), BiotSavart(coils))[0]
        msg += f"; X-line field-line residual max {np.max(np.abs(r)):.3e}"
    print(msg, flush=True)


def plane_coil(curve, phi_old, n_shift=4096):
    """Old coil ``curve`` (centroid at ``phi_old``) rotated onto phi = 0 and made its own image under (x, -y, -z),
    parametrized with t = 0 on the symmetry line (outboard crossing), as a CurveXYZFourier of the same order."""
    order, q = curve.order, curve.quadpoints
    rot = rot_z(-phi_old)
    work = CurveXYZFourier(q, order)
    work.local_full_x = curve.local_full_x

    def at(t):                                   # rotated old coil at parameters t
        c = CurveXYZFourier(np.mod(t, 1.0), order)
        c.x = work.x
        return c.gamma() @ rot.T

    S = np.array([1.0, -1.0, -1.0])
    g = at(q)
    # the image of the curve, traversed backwards, is the curve itself shifted: S g(s - t) ~ g(t); fixed points t* = s/2
    shifts = np.linspace(0, 1, n_shift, endpoint=False)
    err = [np.mean(np.linalg.norm(at(sh - q) * S - g, axis=1)) for sh in shifts]
    s0 = shifts[int(np.argmin(err))]
    cands = [s0 / 2, s0 / 2 + 0.5]
    t0 = max(cands, key=lambda t: at(np.array([t]))[0, 0])          # outboard fixed point (larger x)
    h = 0.5 * (at(q + t0) + at(t0 - q) * S)
    new = CurveXYZFourier(q, order)
    new.least_squares_fit(h)
    x = new.x.copy()
    for i, name in enumerate(new.local_full_dof_names):
        if name.startswith(("xs(", "yc(", "zc(")):
            x[i] = 0.0
    new.x = x
    print(f"plane coil from the coil at {np.degrees(phi_old):.2f} deg: symmetry mismatch before {min(err) * 1e3:.2f} mm "
          f"(mean), fit residual {np.max(np.linalg.norm(new.gamma() - h, axis=1)) * 1e3:.3f} mm")
    return new


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("coils")
    ap.add_argument("n_new", type=int)
    ap.add_argument("out")
    ap.add_argument("--plane-coil", action="store_true", help="add a coil on the phi = 0 symmetry plane per period")
    ap.add_argument("--vmec-input", default=None)
    ap.add_argument("--xpoint", default=None)
    args = ap.parse_args()
    coils = load(args.coils)
    old = base_coils(coils)
    nfp = int(round(len(coils) / (2 * len(old))))
    half = np.pi / nfp
    print(f"old: {len(old)} coil types, nfp {nfp}, centroid phi [deg] {[round(np.degrees(b[2]), 2) for b in old]}")
    for curve, I, phi in old:
        g0 = rot_z(-phi) @ curve.gamma()[0]
        print(f"  parametrization origin of the coil at {np.degrees(phi):6.2f} deg, local (x, y, z) = "
              f"({g0[0]:.3f}, {g0[1]:+.3f}, {g0[2]:+.3f})")
    angles = np.array([b[2] for b in old])
    new_curves, new_currents = [], []
    n_total_old = len(coils)
    n_total_new = nfp * (2 * args.n_new + (1 if args.plane_coil else 0))
    for j in range(args.n_new):
        phi = ((j + 1) * 2 * half / (2 * args.n_new + 1)) if args.plane_coil else ((j + 0.5) * half / args.n_new)
        if not angles[0] <= phi <= angles[-1]:
            raise SystemExit(f"target angle {np.degrees(phi):.2f} deg lies outside the old coils' span "
                             f"[{np.degrees(angles[0]):.2f}, {np.degrees(angles[-1]):.2f}] deg (needs symmetry images)")
        b = int(np.searchsorted(angles, phi))
        a = max(b - 1, 0)
        w = 0.0 if a == b else (phi - angles[a]) / (angles[b] - angles[a])
        (ca, Ia, pa), (cb, Ib, pb) = old[a], old[b]
        pts = (1 - w) * ca.gamma() @ rot_z(phi - pa).T + w * cb.gamma() @ rot_z(phi - pb).T
        curve = CurveXYZFourier(ca.quadpoints, ca.order)
        curve.least_squares_fit(pts)
        I = ((1 - w) * Ia + w * Ib) * n_total_old / n_total_new
        print(f"new coil {j}: phi {np.degrees(phi):.2f} deg = {1 - w:.3f} x coil at {np.degrees(pa):.2f} + {w:.3f} x coil at "
              f"{np.degrees(pb):.2f}; current {I / 1e3:.1f} kA; fit residual "
              f"{np.max(np.linalg.norm(curve.gamma() - pts, axis=1)) * 1e3:.3f} mm")
        new_curves.append(curve)
        new_currents.append(Current(1.0) * I)
    new = coils_via_symmetries(new_curves, new_currents, nfp, True) if new_curves else []
    if args.plane_coil:
        c0, I0, p0 = old[0]                     # the old coil nearest phi = 0
        I = I0 * n_total_old / n_total_new
        print(f"plane coil: current {I / 1e3:.1f} kA")
        new = new + coils_via_symmetries([plane_coil(c0, p0)], [Current(1.0) * I], nfp, False)
    report("old", coils, args.vmec_input, args.xpoint)
    report("new", new, args.vmec_input, args.xpoint)
    save(new, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
