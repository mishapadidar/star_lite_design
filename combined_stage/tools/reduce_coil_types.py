#!/usr/bin/env python3
"""Seed coils with fewer coil types (base coils per half period) from a run's stellarator-symmetric coils.

usage: reduce_coil_types.py <coils json> <n_new> <out json> [--vmec-input <namelist>] [--xpoint <xpoint json>]

The old base coils (the curves owning dofs) sit at centroid angles phi_i in (0, pi/nfp). New coil j goes to
phi_j = (j + 1/2) pi / (nfp n_new). Its points are the pointwise blend (1 - w) R(phi_j - phi_a) gamma_a(t) +
w R(phi_j - phi_b) gamma_b(t) of the two old base coils bracketing phi_j, each rotated about z onto phi_j (w linear in
angle), fitted by a CurveXYZFourier of the same order. The pointwise blend assumes the old coils share their
parametrization origin (true for coils grown from create_equally_spaced_curves; printed as a check). Its current is
the same blend of the old currents times n_old / n_new, so the total poloidal current, hence the toroidal field, is
unchanged. Prints lengths, curvature, coil-coil distance and, with --vmec-input, rms B.n/B on that boundary and the
coil-plasma distance, and, with --xpoint, the X-line's field-line residual, for the old and the new coil set.
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("coils")
    ap.add_argument("n_new", type=int)
    ap.add_argument("out")
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
    for j in range(args.n_new):
        phi = (j + 0.5) * half / args.n_new
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
        I = ((1 - w) * Ia + w * Ib) * len(old) / args.n_new
        print(f"new coil {j}: phi {np.degrees(phi):.2f} deg = {1 - w:.3f} x coil at {np.degrees(pa):.2f} + {w:.3f} x coil at "
              f"{np.degrees(pb):.2f}; current {I / 1e3:.1f} kA; fit residual "
              f"{np.max(np.linalg.norm(curve.gamma() - pts, axis=1)) * 1e3:.3f} mm")
        new_curves.append(curve)
        new_currents.append(Current(1.0) * I)
    new = coils_via_symmetries(new_curves, new_currents, nfp, True)
    report("old", coils, args.vmec_input, args.xpoint)
    report("new", new, args.vmec_input, args.xpoint)
    save(new, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
