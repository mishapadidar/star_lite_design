#!/usr/bin/env python3
"""Uniformly scaled copy of a device: simsopt design archive (coils, surfaces, axes, X-point lines) + VMEC namelist.

usage: scale_device.py <design json> <namelist> <L> <out json> <out namelist> [--current-factor r] [--config-id 0]

Lengths x L, coil currents x r (default r = L: the same |B|, so the vacuum field lines -- magnetic axis, X-points, flux
surfaces -- are exactly the original ones scaled by L). The archive goes through workflows/scaling/scale_design.py; the
namelist boundary and axis are multiplied by L and PHIEDGE by L r (flux ~ B x area). The pressure is left alone (same B,
same beta; tools/make_beta_seed.py recalibrates it anyway).
Prints original vs scaled with the expected ratio: |B| on the magnetic axis, rms B.n/B and coil flux / PHIEDGE on the
namelist boundary, the X-point Newton re-solve on the coils (success, tr M, field-line length, distance to the
boundary), and the coil metrics the driver's ranges use (max length, max curvature, max mean-squared curvature,
coil-coil and coil-boundary distance, max |current|) -- the numbers to set a scaled run's thresholds from.
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np
from scipy.spatial import cKDTree
from simsopt._core import load
from simsopt.field import BiotSavart
from simsopt.geo import CurveCurveDistance, CurveLength, MeanSquaredCurvature, SurfaceRZFourier

from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.vmex_double_null import XpointHyperbolicity

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "workflows", "scaling"))
from scale_design import scale_design  # noqa: E402


def boundary_surface(inp, phis, ntheta):
    """SurfaceRZFourier of a VmecInput boundary (rbc/zbs[n + ntor, m], cos/sin(m u - n nfp v)) on the given phis."""
    rbc, zbs = np.asarray(inp.rbc), np.asarray(inp.zbs)
    nt, mp = (rbc.shape[0] - 1) // 2, rbc.shape[1]
    s = SurfaceRZFourier(nfp=int(inp.nfp), stellsym=True, mpol=mp - 1, ntor=nt, quadpoints_phi=np.asarray(phis),
                         quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))
    for m in range(mp):
        for n in range(0 if m == 0 else -nt, nt + 1):
            if m == 0:   # cos(-n v) = cos(n v), sin(-n v) = -sin(n v): fold VMEC's negative-n m = 0 entries
                rc = rbc[nt + n, 0] + (rbc[nt - n, 0] if n > 0 else 0.0)
                zs = zbs[nt + n, 0] - (zbs[nt - n, 0] if n > 0 else 0.0)
            else:
                rc, zs = rbc[nt + n, m], zbs[nt + n, m]
            s.set_rc(m, n, rc)
            if not (m == 0 and n == 0):
                s.set_zs(m, n, zs)
    return s


def base_curves(coils):
    out, seen = [], set()
    for c in coils:            # curves owning free dofs (not RotatedCurve copies), as the driver
        for o in c.curve.unique_dof_lineage:
            if o.local_dof_size > 0 and id(o) not in seen and "Curve" in type(o).__name__:
                seen.add(id(o))
                out.append(o)
    return out


def measure(design, namelist, config_id, margin=0.2):
    import vmex as vj
    data = load(design)
    coils = data[0][config_id].biotsavart.coils
    inp = vj.VmecInput.from_file(namelist)
    nfp = int(inp.nfp)
    full = boundary_surface(inp, np.linspace(0, 1, 64 * nfp, endpoint=False), 256)
    period = boundary_surface(inp, np.linspace(0, 1 / nfp, 64, endpoint=False), 128)
    row = {}

    bs = BiotSavart(coils)
    axis = data[2][config_id]
    bs.set_points(np.ascontiguousarray(getattr(axis, "curve", axis).gamma().reshape(-1, 3)))
    row["|B| on axis [T]"] = float(np.mean(bs.AbsB()))

    bs.set_points(period.gamma().reshape(-1, 3))
    B = bs.B()
    bn = np.sum(B * period.unitnormal().reshape(-1, 3), axis=1) / np.linalg.norm(B, axis=1)
    row["rms B.n/B"] = float(np.sqrt(np.mean(bn ** 2)))

    xs = boundary_surface(inp, [0.0], 2048)            # coil flux through the phi = 0 cross-section = oint A.dl
    bs.set_points(xs.gamma().reshape(-1, 3))
    flux = float(np.sum(np.sum(bs.A() * xs.gammadash2().reshape(-1, 3), axis=1)) / 2048)
    row["coil flux / PHIEDGE"] = flux / float(inp.phiedge)
    row["PHIEDGE [Wb]"] = float(inp.phiedge)

    entry = data[3][config_id]
    xp = PeriodicFieldLine(BiotSavart(coils), entry.curve,
                           options={"newton_tol": 1e-13, "newton_maxiter": 40, "verbose": False})
    res = xp.run_code(float(CurveLength(entry.curve).J()))
    row["X-point Newton success"] = float(bool(res["success"]))
    row["X-point tr M"] = float(XpointHyperbolicity(xp, BiotSavart(coils), margin=margin).trace()) \
        if res["success"] else float("nan")
    row["X-point field-line length [m]"] = float(res["length"]) if res["success"] else float("nan")
    d = cKDTree(full.gamma().reshape(-1, 3)).query(entry.curve.gamma().reshape(-1, 3))[0]
    row["X-point distance min [m]"], row["X-point distance max [m]"] = float(d.min()), float(d.max())

    curves, bases = [c.curve for c in coils], base_curves(coils)
    tree = cKDTree(full.gamma().reshape(-1, 3))
    row["coils (base curves)"] = float(len(coils)) + len(bases) / 1000.0
    row["max coil length [m]"] = max(float(CurveLength(c).J()) for c in bases)
    row["max curvature [1/m]"] = max(float(np.max(c.kappa())) for c in bases)
    row["max mean-sq curvature [1/m^2]"] = max(float(MeanSquaredCurvature(c).J()) for c in bases)
    row["min coil-coil distance [m]"] = float(CurveCurveDistance(curves, 0.0).shortest_distance())
    row["min coil-boundary distance [m]"] = min(float(tree.query(c.gamma())[0].min()) for c in curves)
    row["max |coil current| [kA]"] = max(abs(float(c.current.get_value())) for c in coils) / 1e3
    row["aspect (boundary major/minor)"] = float(full.major_radius() / full.minor_radius())
    row["(iota, G) at data[1]"] = float(data[1][config_id][1]) if len(data) > 1 else float("nan")
    row["Boozer targetlabel"] = float(data[0][config_id].targetlabel)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("design")
    ap.add_argument("namelist")
    ap.add_argument("L", type=float)
    ap.add_argument("out_design")
    ap.add_argument("out_namelist")
    ap.add_argument("--current-factor", type=float, default=None, help="current multiplier (default L: same |B|)")
    ap.add_argument("--config-id", type=int, default=0)
    a = ap.parse_args()
    import vmex as vj
    L = a.L
    r = L if a.current_factor is None else a.current_factor

    summary = scale_design(a.design, a.out_design, L=L, current_factor=r)
    print(f"archive: R {summary['major_radius']:.4f} m, a {summary['minor_radius']:.4f} m, A {summary['aspect_ratio']:.3f}; "
          f"B0 {summary['B0_base']:.5f} -> {summary['B0_final']:.5f} T (current factor {r:g}) -> {a.out_design}")
    inp = vj.VmecInput.from_file(a.namelist)
    changes = dict(rbc=np.asarray(inp.rbc) * L, zbs=np.asarray(inp.zbs) * L, phiedge=float(inp.phiedge) * L * r)
    for name in ("raxis_c", "zaxis_s", "raxis_s", "zaxis_c"):
        if getattr(inp, name, None) is not None:
            changes[name] = np.asarray(getattr(inp, name)) * L
    replace(inp, **changes).to_indata(a.out_namelist)
    print(f"namelist: boundary and axis x {L:g}, PHIEDGE x {L * r:g} -> {a.out_namelist}")

    before = measure(a.design, a.namelist, a.config_id)
    after = measure(a.out_design, a.out_namelist, a.config_id)
    expected = {"|B| on axis [T]": r / L, "rms B.n/B": 1.0, "coil flux / PHIEDGE": 1.0, "PHIEDGE [Wb]": L * r,
                "X-point Newton success": 1.0, "X-point tr M": 1.0, "X-point field-line length [m]": L,
                "X-point distance min [m]": L, "X-point distance max [m]": L, "coils (base curves)": 1.0,
                "max coil length [m]": L, "max curvature [1/m]": 1 / L, "max mean-sq curvature [1/m^2]": 1 / L ** 2,
                "min coil-coil distance [m]": L, "min coil-boundary distance [m]": L, "max |coil current| [kA]": r,
                "aspect (boundary major/minor)": 1.0, "(iota, G) at data[1]": r, "Boozer targetlabel": L ** 3}
    print(f"\n{'quantity':34s} {'original':>14s} {'scaled':>14s} {'ratio':>10s} {'expected':>9s}")
    worst = 0.0
    for key, v0 in before.items():
        v1 = after[key]
        ratio = v1 / v0 if v0 else float("nan")
        exp = expected.get(key, float("nan"))
        if np.isfinite(ratio) and np.isfinite(exp) and key not in ("rms B.n/B",):
            worst = max(worst, abs(ratio / exp - 1.0))
        print(f"{key:34s} {v0:14.6g} {v1:14.6g} {ratio:10.6f} {exp:9.4g}")
    print(f"\nlargest relative deviation from the expected ratios: {worst:.2e}  "
          f"({'SCALED OK' if worst < 1e-3 else 'CHECK'})")


if __name__ == "__main__":
    main()
