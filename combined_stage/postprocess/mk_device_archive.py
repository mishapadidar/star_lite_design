#!/usr/bin/env python3
"""Pack a combined-stage checkpoint (VMEX + augmented Lagrangian) into an array-campaign device folder.

The array post-processing (array/mk_LCFS.py, mk_manifolds.py, plot_manifolds.py, mk_elongation.py, mk_paraview.py)
and array/device_browser.py read a design archive [boozer_surfaces, iota_Gs, axes, xpoints, sdf] plus VTK files and
a summary.txt. A combined-stage run has coils, a VMEX boundary namelist and (double null) an X-line instead, so this
script builds the same objects on the run's coils:

  * Boozer surface  -- SurfaceXYZTensorFourier (mpol = ntor = 10, exact Boozer, Volume label) fitted to the VMEX
                       boundary and Newton-solved in the coil field; if no exact solve converges from that fit, a
                       Boozer least-squares solve first, then the exact Newton polish from its solution
  * magnetic axis   -- PeriodicFieldLine (order 16) Newton-solved from the boundary cross-section centroids
  * X-point         -- PeriodicFieldLine (order 16) Newton-solved from the run's X-line (double-null runs only)
  * sdf             -- None: these runs have no vacuum vessel, so every vessel metric is omitted

usage: mk_device_archive.py <run dir> <tag: final | outerNNN> <device dir> [--surface-guess <archive json>[:index]]

--surface-guess starts the exact Boozer solves from an archived Boozer surface (Boozer angles, e.g. the design the run
started from) instead of the VMEC-angle fit, shrinking the volume from the VMEX boundary's until a converged,
non-self-intersecting surface is found (a checkpoint whose coils do not reproduce the VMEX boundary may have no clean
flux surface there; mk_LCFS.py then grows it to the coil field's LCFS).

Writes into <device dir>: design_opt_final_<ID>.json (ID = crc32 of the device name, see device_info.yaml),
summary.txt, max_rel_error.txt, a copy of the VMEX namelist, curves_opt_final.vtu, surf_opt_0_final.vts,
ma_opt_final.vtu, xpoint_curves_opt_final.vtu. Run on a compute node.
"""
import argparse
import os
import shutil
import sys
import time
import zlib

import numpy as np
import yaml
from simsopt._core import load, save
from simsopt.field import BiotSavart, compute_fieldlines
from simsopt.geo import (BoozerSurface, CurveCurveDistance, CurveLength, CurveXYZFourierSymmetries, MajorRadius,
                         MeanSquaredCurvature, NonQuasiSymmetricRatio, SurfaceRZFourier, SurfaceXYZTensorFourier,
                         Volume, curves_to_vtk)
from star_lite_design.utils.displacement import FieldLineMeanZ
from star_lite_design.utils.lcfs import _self_intersects
from star_lite_design.utils.magneticwell import MagneticWell
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine, ModBRippleOnFieldLine
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.tangent_map import AxisIota, TangentMap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_layout as layout  # noqa: E402

MPOL = NTOR = 10            # designA's archived Boozer surfaces
FL_ORDER = 16               # designA's archived axis / X-point curves
FL_OPTIONS = {"verbose": False, "newton_tol": 1e-13, "newton_maxiter": 40}
BS_OPTIONS = {"verbose": False, "newton_tol": 1e-13, "newton_maxiter": 40}
LS_OPTIONS = {"verbose": False, "bfgs_tol": 1e-10, "bfgs_maxiter": 1500, "newton_tol": 1e-11, "newton_maxiter": 40}
MAX_OFFSET = 0.2            # a solved Boozer surface must stay within this fraction of the minor radius of the boundary
MU0 = 4e-7 * np.pi


def history_row(run, tag):
    rows = [r for r in (yaml.safe_load(open(os.path.join(run, "history.yaml"))) or []) if r["outer"] >= 0]
    if tag == "final":
        return rows[-1]
    k = int(tag[len("outer"):])
    return [r for r in rows if r["outer"] == k][-1]


def boundary(namelist, qp_phi, ntheta=128):
    return SurfaceRZFourier.from_vmec_input(namelist, quadpoints_phi=qp_phi,
                                            quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))


def solve_fieldline(coils, points, nfp, stellsym):
    qp = np.linspace(0, 1 / nfp, 2 * FL_ORDER + 1, endpoint=False)
    curve = CurveXYZFourierSymmetries(qp, FL_ORDER, nfp, stellsym, ntor=1)
    curve.least_squares_fit(points)
    line = PeriodicFieldLine(BiotSavart(coils), curve, options=dict(FL_OPTIONS))
    try:
        res = line.run_code(CurveLength(curve).J())
        return line if res["success"] else None
    except Exception as e:  # noqa: BLE001
        print(f"  field-line Newton raised {e!r}")
        return None


def plane_points(R, Z, qp):
    ph = 2 * np.pi * qp
    return np.column_stack([R * np.cos(ph), R * np.sin(ph), Z])


def magnetic_axis(coils, namelist, nfp):
    qp = np.linspace(0, 1 / nfp, 2 * FL_ORDER + 1, endpoint=False)
    g = boundary(namelist, qp).gamma()
    R, Z = np.hypot(g[..., 0], g[..., 1]).mean(axis=1), g[..., 2].mean(axis=1)
    a = 0.5 * (np.hypot(g[0, :, 0], g[0, :, 1]).max() - np.hypot(g[0, :, 0], g[0, :, 1]).min())
    for it in range(4):
        axis = solve_fieldline(coils, plane_points(R, Z, qp), nfp, True)
        if axis is not None:
            ga = axis.curve.gamma()
            if np.max(np.hypot(np.hypot(ga[:, 0], ga[:, 1]) - R, ga[:, 2] - Z)) < a:
                print(f"magnetic axis solved (guess refinement {it})")
                return axis
        # refine the guess: centroid of one field line's crossings of the quadrature planes
        _, hits = compute_fieldlines(BiotSavart(coils), [R[0]], [Z[0]], tmax=60 * 2 * np.pi * R.mean(), tol=1e-9,
                                     phis=list(2 * np.pi * qp))
        h = np.asarray(hits[0])
        h = h[h[:, 1] >= 0]
        Rh, Zh = np.hypot(h[:, 2], h[:, 3]), h[:, 4]
        idx = h[:, 1].astype(int)
        R = np.array([Rh[idx == i].mean() for i in range(qp.size)])
        Z = np.array([Zh[idx == i].mean() for i in range(qp.size)])
    raise SystemExit("magnetic axis did not solve")


def x_point(coils, path, nfp):
    if not os.path.exists(path):
        return None
    seed = load(path)["curve"]
    qp = np.linspace(0, 1 / nfp, 2 * FL_ORDER + 1, endpoint=False)
    xp = solve_fieldline(coils, seed.gamma_pure(seed.x, qp), nfp, False)
    if xp is None:
        raise SystemExit(f"X-point from {path} did not solve")
    print(f"X-point solved: length {xp.res['length']:.4f} m, (R, Z) at phi=0 "
          f"({np.hypot(*xp.curve.gamma()[0, :2]):.4f}, {xp.curve.gamma()[0, 2]:+.4f}) m")
    return xp


def boozer_surface(coils, namelist, nfp, iota_axis, axis, guess=None):
    qp_phi = np.linspace(0, 1 / nfp, 2 * NTOR + 1, endpoint=False)
    qp_theta = np.linspace(0, 1, 2 * MPOL + 1, endpoint=False)
    surf = SurfaceXYZTensorFourier(mpol=MPOL, ntor=NTOR, nfp=nfp, stellsym=True,
                                   quadpoints_phi=qp_phi, quadpoints_theta=qp_theta)
    surf.least_squares_fit(SurfaceRZFourier.from_vmec_input(namelist, quadpoints_phi=qp_phi,
                                                            quadpoints_theta=qp_theta).gamma())
    x0 = surf.x.copy()
    ref = boundary(namelist, np.linspace(0, 1 / nfp, 64, endpoint=False)).gamma().reshape(-1, 3)
    tol = MAX_OFFSET * SurfaceRZFourier.from_vmec_input(namelist).minor_radius()
    G_abs = MU0 * sum(abs(c.current.get_value()) for c in coils)
    V0 = Volume(surf).J()
    # sign guesses from the field itself: iota from the on-axis return map, G from the toroidal field direction
    f = BiotSavart(coils)
    p0 = axis.curve.gamma()[:1]
    f.set_points(p0)
    B_phi = float(f.B()[0] @ np.array([-p0[0, 1], p0[0, 0], 0.0]))
    s_iota0, s_G0 = np.sign(iota_axis) or 1.0, np.sign(B_phi) or 1.0
    iotas = sorted(set(np.round(np.r_[abs(iota_axis), np.arange(0.10, 0.625, 0.025)], 4)),
                   key=lambda v: abs(v - abs(iota_axis)))
    signs = [(s_iota0, s_G0), (s_iota0, -s_G0), (-s_iota0, s_G0), (-s_iota0, -s_G0)]

    def offset():
        pts = surf.gamma().reshape(-1, 3)
        return float(np.max(np.min(np.linalg.norm(pts[:, None, :] - ref[None, :, :], axis=-1), axis=1)))

    best = {}          # converged, non-self-intersecting full-volume solve closest to the VMEX boundary

    def attempt(start, frac, iota, G, constraint_weight=None, enforce_tol=True):
        surf.x = start
        options = dict(LS_OPTIONS if constraint_weight else BS_OPTIONS)
        bs = BoozerSurface(BiotSavart(coils), surf, Volume(surf), frac * V0, constraint_weight=constraint_weight,
                           options=options)
        try:
            res = bs.run_code(iota, G)
            # a degenerate trial surface can make the self-intersection check itself raise ("goes back on itself"),
            # as in utils/lcfs._try_volume: that is a rejection too
            if not res["success"] or _self_intersects(surf):
                return None
        except Exception:  # noqa: BLE001
            return None
        off = offset()
        if frac == 1.0 and constraint_weight is None and ("off" not in best or off < best["off"]):
            best.update(off=off, x=surf.x.copy(), iota=res["iota"], G=res["G"])
        if enforce_tol and off > tol * (1.0 if frac == 1.0 else 1.75):
            return None
        return bs, res, off

    def done(found, frac, how):
        bs, res, off = found
        print(f"Boozer surface solved ({how}): volume fraction {frac}, iota {res['iota']:+.5f}, G {res['G']:+.5f}, "
              f"max offset from the VMEX boundary {off * 100:.2f} cm (limit {tol * 100:.2f} cm)")
        return bs, frac, off, tol

    # 0) exact Newton from an archived Boozer surface, shrinking the volume until the surface is clean
    if guess is not None:
        gsurf, g_iota, g_G = guess
        if (gsurf.x.size == surf.x.size and np.allclose(gsurf.quadpoints_phi, qp_phi)
                and np.allclose(gsurf.quadpoints_theta, qp_theta)):
            xg = gsurf.x.copy()
            V0 = np.sign(Volume(gsurf).J()) * abs(V0)
            for frac in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5):
                for iota in sorted({round(float(g_iota), 4), round(float(iota_axis), 4)} | set(np.round(iotas[:8], 4)),
                                   key=lambda v: abs(v - g_iota)):
                    found = attempt(xg, frac, np.sign(g_iota) * abs(iota), g_G, enforce_tol=(frac == 1.0))
                    if found:
                        return done(found, frac, "exact from the archived surface")
            print("  archived surface guess: no clean converged surface down to half the volume")
        else:
            print("  archived surface guess skipped: its resolution / quadrature points differ from this surface")
    # 1) exact Newton from the VMEC-angle fit
    for iota_abs in iotas:
        for s_i, s_G in signs:
            found = attempt(x0, 1.0, s_i * iota_abs, s_G * G_abs)
            if found:
                return done(found, 1.0, "exact")
    # 1b) exact solves converged but none reproduces the VMEX boundary: the coils do not produce that boundary (B.n
    #     mismatch at this checkpoint), which least squares cannot change -- use the closest coil-field surface
    if best:
        print(f"  no exact Boozer surface within {tol * 100:.2f} cm of the VMEX boundary; the closest converged one is "
              f"{best['off'] * 100:.2f} cm away (the coils do not reproduce the boundary) -- using it")
        found = attempt(best["x"], 1.0, best["iota"], best["G"], enforce_tol=False)
        if found:
            return done(found, 1.0, "exact, closest to the VMEX boundary, beyond the offset limit")
    # 2) Boozer least squares from the same fit (BFGS absorbs the VMEC -> Boozer angle change), then exact Newton
    for iota_abs in iotas[:6]:
        for s_i, s_G in signs:
            ls = attempt(x0, 1.0, s_i * iota_abs, s_G * G_abs, constraint_weight=100.0)
            if ls is None:
                continue
            print(f"  least-squares Boozer surface: iota {ls[1]['iota']:+.5f}, offset {ls[2] * 100:.2f} cm")
            found = attempt(surf.x.copy(), 1.0, ls[1]["iota"], ls[1]["G"])
            if found:
                return done(found, 1.0, "least squares, exact polish")
    # 3) exact Newton on slightly smaller volumes
    for frac in (0.95, 0.9, 0.8):
        for iota_abs in iotas:
            for s_i, s_G in signs:
                found = attempt(x0, frac, s_i * iota_abs, s_G * G_abs)
                if found:
                    return done(found, frac, "exact")
    raise SystemExit("no Boozer surface converged near the VMEX boundary")


def write_summary(path, metrics):
    with open(path, "w") as f:
        f.write(f"# {'metric':<30s} {'value':>16s} {'threshold':>16s} {'rel_error':>16s}\n")
        for name, (value, threshold, rel_err) in metrics.items():
            thr_str = f"{threshold:.6e}" if threshold is not None else "n/a"
            err_str = f"{rel_err:.6e}" if rel_err is not None else "n/a"
            f.write(f"  {name:<30s} {value:.6e}   {thr_str:>16s}   {err_str:>16s}\n")
    rel = [abs(v[2]) for v in metrics.values() if v[2] is not None]
    with open(os.path.join(os.path.dirname(path), "max_rel_error.txt"), "w") as f:
        f.write(f"{(max(rel) if rel else 0.0):.18e}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("tag")
    ap.add_argument("device_dir")
    ap.add_argument("--surface-guess", default=None, help="archive json[:index] holding a Boozer surface guess")
    args = ap.parse_args()
    run, tag, dev = os.path.abspath(args.run), args.tag, os.path.abspath(args.device_dir)
    guess = None
    if args.surface_guess:
        path, _, index = args.surface_guess.partition(":")
        arch = load(path)
        guess = (arch[0][int(index or 0)].surface, *arch[1][int(index or 0)])
    os.makedirs(dev, exist_ok=True)
    cfg = yaml.safe_load(open(os.path.join(run, "config_used.yaml")))
    row = history_row(run, tag)
    namelist = layout.find(run, f"input.combined_stage_{tag}")
    nfp = SurfaceRZFourier.from_vmec_input(namelist).nfp
    coils = load(layout.find(run, f"coils_{tag}.json"))
    null = "DN" if os.path.exists(layout.find(run, f"xpoint_{tag}.json")) else "none"
    name = f"run={os.path.relpath(run, os.path.dirname(os.path.dirname(run))).replace(os.sep, '__')}_nfp={nfp}_null={null}_tag={tag}"
    device_id = zlib.crc32(name.encode())
    t0 = time.time()

    axis = magnetic_axis(coils, namelist, nfp)
    iota_axis = float(AxisIota(TangentMap(axis, BiotSavart(coils), 0.0, mtype="identity")).J())
    print(f"on-axis iota {iota_axis:+.5f}")
    xp = x_point(coils, layout.find(run, f"xpoint_{tag}.json"), nfp)
    bs, vol_frac, offset, offset_limit = boozer_surface(coils, namelist, nfp, iota_axis, axis, guess)
    print(f"objects built in {time.time() - t0:.0f}s")

    xpoints = [xp] if xp is not None else []
    save([[bs], [[float(bs.res["iota"]), float(bs.res["G"])]], [axis], xpoints, None],
         os.path.join(dev, f"design_opt_final_{device_id}.json"))
    shutil.copy(namelist, os.path.join(dev, os.path.basename(namelist)))
    yaml.safe_dump(dict(device_name=name, device_id=device_id, run=run, tag=tag, nfp=nfp, null=null),
                   open(os.path.join(dev, "device_info.yaml"), "w"), sort_keys=False)
    curves = [c.curve for c in coils]
    curves_to_vtk(curves, os.path.join(dev, "curves_opt_final"))
    curves_to_vtk([axis.curve], os.path.join(dev, "ma_opt_final"))
    if xp is not None:
        curves_to_vtk([xp.curve], os.path.join(dev, "xpoint_curves_opt_final"))
    bs.surface.to_vtk(os.path.join(dev, "surf_opt_0_final"))

    kc = cfg["coils"]
    m = {}

    def add(name, fn, threshold=None, rel=None):
        try:
            value = float(fn())
        except Exception as e:  # noqa: BLE001
            print(f"  metric {name} skipped: {e!r}")
            return
        m[name] = (value, threshold, None if rel is None else float(rel(value)))

    above = lambda thr: (lambda v: max(v / thr - 1.0, 0.0))   # noqa: E731
    below = lambda thr: (lambda v: max(1.0 - v / thr, 0.0))   # noqa: E731
    add("nonQS_percent", lambda: 100.0 * NonQuasiSymmetricRatio(bs, BiotSavart(coils)).J() ** 0.5)

    def mirror():
        f = BiotSavart(coils)
        f.set_points(bs.surface.gamma().reshape(-1, 3))
        B = f.AbsB()
        return np.max(B) / np.min(B)
    add("mirror_ratio", mirror)
    add("aspect_ratio", bs.surface.aspect_ratio)
    add("current", lambda: max(abs(c.current.get_value()) for c in coils), kc["current_threshold"],
        above(kc["current_threshold"]))
    add("iotas", lambda: bs.res["iota"])
    add("on_axis_iota", lambda: iota_axis)
    add("major_radius", lambda: MajorRadius(bs).J())
    add("coil_length", lambda: max(CurveLength(c).J() for c in curves), kc["length_threshold"],
        above(kc["length_threshold"]))
    add("coil_to_coil", lambda: CurveCurveDistance(curves, kc["coil_coil_threshold"]).shortest_distance(),
        kc["coil_coil_threshold"], below(kc["coil_coil_threshold"]))

    def coil_plasma():
        Y = boundary(namelist, np.linspace(0, 1, 64 * nfp, endpoint=False), 64).gamma().reshape(-1, 3)
        return min(np.min(np.linalg.norm(c.gamma()[:, None, :] - Y[None, :, :], axis=-1)) for c in curves)
    add("coil_plasma", coil_plasma, kc["coil_plasma_threshold"], below(kc["coil_plasma_threshold"]))
    add("msc", lambda: max(MeanSquaredCurvature(c).J() for c in curves), kc["msc_threshold"],
        above(kc["msc_threshold"]))
    add("curvature", lambda: max(np.max(c.kappa()) for c in curves), kc["curvature_threshold"],
        above(kc["curvature_threshold"]))
    add("modB", lambda: ModBOnFieldLine(axis, BiotSavart(coils)).J())
    add("modB_axis_ripple", lambda: ModBRippleOnFieldLine(axis, BiotSavart(coils), 1e-3).max_deviation())
    add("well", lambda: MagneticWell(axis, bs, 0.0).well().max())
    if xp is not None:
        add("fieldline_meanz", lambda: FieldLineMeanZ(xp, 1e-3).max_distance())
        M = np.asarray(TangentMap(xp, BiotSavart(coils), 0.1, mtype="jordan").matrix)
        m["monodromy"] = (float(np.abs(M - np.eye(2)).max()), None, None)
        for a in range(2):
            for b in range(2):
                m[f"monodromy_M{a}{b}_idx0"] = (float(M[a, b]), None, None)
        m["greene_residue_idx0"] = ((2.0 - float(np.trace(M))) / 4.0, None, None)
    # combined-stage quantities at this checkpoint (history.yaml of the run)
    m["optimization_surface_volume_fraction"] = (vol_frac, None, None)
    m["optimization_surface_offset_m"] = (offset, offset_limit, max(offset / offset_limit - 1.0, 0.0))
    m["vmex_f_QS"] = (float(row["f_QS"]), None, None)
    m["bnormal_max_harmonic"] = (float(row["bnormal_inf"]), None, None)
    m["rms_Bn_over_B"] = (float(row["rms_Bn_over_B"]), None, None)
    if row.get("field_strength") is not None:
        m["toroidal_flux_residual"] = (float(row["field_strength"]), None, None)
    if xp is not None and row.get("xpoint_distance_min") is not None:
        d = cfg["double_null"]
        m["xpoint_boundary_distance_min"] = (float(row["xpoint_distance_min"]), float(d["distance_min"]),
                                             max(1.0 - row["xpoint_distance_min"] / d["distance_min"], 0.0))
        m["xpoint_boundary_distance_max"] = (float(row["xpoint_distance_max"]), float(d["distance_max"]),
                                             max(row["xpoint_distance_max"] / d["distance_max"] - 1.0, 0.0))
        m["xpoint_trace_M"] = (float(row["xpoint_trace"]), None, None)
    if row.get("xline_residual_max") is not None:
        m["xline_residual_max"] = (float(row["xline_residual_max"]), None, None)
    m["al_outer_iteration"] = (float(row["outer"]), None, None)
    write_summary(os.path.join(dev, "summary.txt"), m)
    print(f"wrote {dev} ({name}, ID {device_id}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
