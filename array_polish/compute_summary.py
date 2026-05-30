#!/usr/bin/env python3
"""
Recompute the summary metrics on a POLISHED device and write summary.txt and
max_rel_error.txt next to singular.json.

Usage:
    compute_summary.py <singular_json> <previous_summary_txt>

Design:
  * Field-dependent metrics (nonQS, iotas, major_radius, plasma_vessel_margin,
    modB, well, fieldline_meanz, fieldline_meandist, monodromy) are RECOMPUTED
    on the polished objects loaded from singular.json.
  * Modular-coil geometry metrics (current, coil_length, coil_to_coil,
    coil_on_vessel, coil_clearance, msc, curvature, arclength) are COPIED
    verbatim from the previous summary.txt and re-labelled with a "_modular"
    suffix (the polish step does not change the modular coils).
  * The same geometry metrics are computed for the auxiliary planar circular
    coils and written with a "_planar" suffix, plus the coils' Z height and
    radii.
  * Thresholds are reused from the previous summary.txt (the polish targets the
    same limits); only the values/errors are recomputed.
"""
import sys
from pathlib import Path

import numpy as np

from simsopt._core import load
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (CurveXYZFourier, CurveLength, CurveCurveDistance,
                         MeanSquaredCurvature, ArclengthVariation,
                         MajorRadius, NonQuasiSymmetricRatio)

from star_lite_design.utils.magneticwell import MagneticWell
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine

# Must match _CURRENT_SCALE in singularperiodicfieldline.py / boozer_singular.py.
_CURRENT_SCALE = 1.0e7 / (4.0 * np.pi)

# Geometry metrics copied (modular) / recomputed (planar), in the order they
# appear in summary.txt.
GEOMETRY_METRICS = [
    'current', 'coil_length', 'coil_to_coil', 'coil_on_vessel',
    'coil_clearance', 'msc', 'curvature', 'arclength',
]


def parse_summary(path):
    """Return {name: (value, threshold, rel_error)}; threshold/err may be None."""
    out = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        name = parts[0]

        def f(tok):
            try:
                return float(tok)
            except ValueError:
                return None
        out[name] = (f(parts[1]), f(parts[2]), f(parts[3]))
    return out


def rel_abs(v, t):
    """|v-t|/|t| (used for targets like iotas, major_radius, modB)."""
    return abs(v - t) / abs(t) if t not in (None, 0.0) else abs(v - t)


def rel_below(v, t):
    """max(t-v, 0)/|t|: penalize being below a minimum (distances)."""
    if t in (None, 0.0):
        return max(t - v, 0.0) if t is not None else 0.0
    return max(t - v, 0.0) / abs(t)


def rel_above(v, t):
    """max(v-t, 0)/|t|: penalize exceeding a maximum (length, curvature, ...)."""
    if t in (None, 0.0):
        return max(v - t, 0.0) if t is not None else 0.0
    return max(v - t, 0.0) / abs(t)


def main():
    if len(sys.argv) < 3:
        print("Usage: compute_summary.py <singular_json> <previous_summary_txt>",
              file=sys.stderr)
        sys.exit(2)

    p = Path(sys.argv[1])
    prev = parse_summary(sys.argv[2])

    boozer_surfaces, iota_Gs, axes, xpoints, sdf = load(p)

    # BoozerSurface.res and PeriodicFieldLine.res are NOT serialized, but
    # objectives like NonQuasiSymmetricRatio / MagneticWell read them. Re-solve
    # each surface from the cached polished (iota, G) and each axis from its
    # length to repopulate .res; both start at the solution so this is cheap.
    for bs, (iota, G) in zip(boozer_surfaces, iota_Gs):
        bs.run_code(iota, G)
    for ax in axes:
        ax.run_code(CurveLength(ax.curve).J())

    def thr(name):
        return prev[name][1] if name in prev else None

    rows = []   # (name, value, threshold, rel_error)  threshold/err may be None

    # ---------------- field-dependent metrics (recomputed) -----------------
    # nonQS
    nonqs = [NonQuasiSymmetricRatio(bs, BiotSavart(bs.biotsavart.coils)) for bs in boozer_surfaces]
    nonqs_pct = 100.0 * ((1.0 / len(nonqs)) * sum(nonqs)).J() ** 0.5
    rows.append(('nonQS_percent', nonqs_pct, None, None))

    # iotas (from the cached polished iota_Gs)
    t = thr('iotas')
    iota_val = max(abs(ig[0]) for ig in iota_Gs)
    rows.append(('iotas', iota_val, t, rel_abs(iota_val, t) if t is not None else None))

    # major radius
    t = thr('major_radius')
    mr_val = MajorRadius(boozer_surfaces[0]).J()
    rows.append(('major_radius', mr_val, t, rel_abs(mr_val, t) if t is not None else None))

    # plasma-vessel margin: min signed distance (sign=-1) over xpoint curves + surfaces
    t = thr('plasma_vessel_margin')
    margins = []
    for xp in xpoints:
        margins.append(sdf.pure(xp.curve.gamma(), sdf.local_full_x, -1.0).min())
    for bs in boozer_surfaces:
        margins.append(sdf.pure(bs.surface.gamma().reshape((-1, 3)), sdf.local_full_x, -1.0).min())
    pvm_val = float(np.min(margins))
    rows.append(('plasma_vessel_margin', pvm_val, t, rel_below(pvm_val, t) if t is not None else None))

    # modB on axis
    t = thr('modB')
    modB_val = max(ModBOnFieldLine(ax, BiotSavart(bs.biotsavart.coils)).J()
                   for ax, bs in zip(axes, boozer_surfaces))
    rows.append(('modB', modB_val, t, rel_abs(modB_val, t) if t is not None else None))

    # magnetic well
    t = thr('well')
    well_val = max(MagneticWell(ax, bs, t if t is not None else 0.0).well().max()
                   for ax, bs in zip(axes, boozer_surfaces))
    rows.append(('well', well_val, t, rel_above(well_val, t) if t is not None else None))

    # fieldline mean-Z spread
    t = thr('fieldline_meanz')
    meanz_val = 0.0
    for xp in xpoints:
        z = xp.curve.gamma()[:, 2]
        meanz_val = max(meanz_val, np.abs(z - z.mean()).max())
    rows.append(('fieldline_meanz', meanz_val, t, rel_above(meanz_val, t) if t is not None else None))

    # fieldline mean distance to vessel (spread of signed distance, sign=-1)
    t = thr('fieldline_meandist')
    meandist_val = 0.0
    for xp in xpoints:
        sd = sdf.pure(xp.curve.gamma(), sdf.local_full_x, -1.0)
        meandist_val = max(meandist_val, np.abs(sd - sd.mean()).max())
    rows.append(('fieldline_meandist', meandist_val, t, rel_above(meandist_val, t) if t is not None else None))

    # monodromy (achieved constraint on the polished, serialized monodromy)
    t = thr('monodromy')
    constraint = xpoints[0].options.get('monodromy_constraint', 'identity')
    if constraint == 'trace':
        mono_val = max(abs(np.trace(np.asarray(xp.monodromy_matrix)) - 2.0) for xp in xpoints)
    else:
        mono_val = max(np.abs(np.asarray(xp.monodromy_matrix) - np.eye(2)).max() for xp in xpoints)
    rows.append(('monodromy', mono_val, t, rel_above(mono_val, t) if t is not None else None))

    # ---------------- modular-coil geometry (copied verbatim) --------------
    for name in GEOMETRY_METRICS:
        if name in prev:
            v, tt, e = prev[name]
            rows.append((f'{name}_modular', v, tt, e))

    # ---------------- planar (auxiliary) coil geometry ---------------------
    # Build the aux coils from each xpoint's solved mu and aggregate.
    all_base_curves = []
    aux_currents = []
    radii_per_xp = []
    Z_per_xp = []
    for xp in xpoints:
        mu = np.asarray(xp.mu)
        N = (len(mu) - 1) // 2
        Z = float(mu[-1])
        Z_per_xp.append(Z)
        radii_per_xp.append([float(r) for r in mu[N:2 * N]])
        for k in range(N):
            rk = float(mu[N + k])
            c = CurveXYZFourier(np.linspace(0, 1, 160, endpoint=False), 1)
            c.x = c.x * 0.
            c.set('zc(0)', Z)
            c.set('xc(1)', rk)
            c.set('ys(1)', rk)
            all_base_curves.append(c)
            aux_currents.append(abs(float(mu[k])) * _CURRENT_SCALE)

    if all_base_curves:
        planar_len = max(CurveLength(c).J() for c in all_base_curves)
        planar_msc = max(MeanSquaredCurvature(c).J() for c in all_base_curves)
        planar_curv = max(float(np.max(c.kappa())) for c in all_base_curves)
        planar_alen = max(ArclengthVariation(c).J() for c in all_base_curves)
        planar_curr = max(aux_currents)
        # coil-to-coil among aux coils (only meaningful with >1 coil)
        planar_cc = (CurveCurveDistance(all_base_curves, thr('coil_to_coil') or 0.0).shortest_distance()
                     if len(all_base_curves) > 1 else None)
        # min clearance of aux coils to vessel (sign=+1, as for modular clearance)
        planar_clear = min(sdf.pure(c.gamma(), sdf.local_full_x, 1.0).min() for c in all_base_curves)

        rows.append(('current_planar', planar_curr, thr('current'),
                     rel_above(planar_curr, thr('current')) if thr('current') is not None else None))
        # length/msc/curvature thresholds were defined for the modular coils;
        # they don't meaningfully apply to the planar circles, so report
        # value-only (no threshold / rel_error).
        rows.append(('coil_length_planar', planar_len, None, None))
        if planar_cc is not None:
            rows.append(('coil_to_coil_planar', planar_cc, thr('coil_to_coil'),
                         rel_below(planar_cc, thr('coil_to_coil')) if thr('coil_to_coil') is not None else None))
        rows.append(('coil_clearance_planar', planar_clear, thr('coil_clearance'),
                     rel_below(planar_clear, thr('coil_clearance')) if thr('coil_clearance') is not None else None))
        rows.append(('msc_planar', planar_msc, None, None))
        rows.append(('curvature_planar', planar_curv, None, None))
        rows.append(('arclength_planar', planar_alen, thr('arclength'),
                     planar_alen if thr('arclength') is not None else None))

    # ---------------- planar coil geometry descriptors --------------------
    # Z height and radii per xpoint (descriptive: no threshold / error).
    for idx, (Z, radii) in enumerate(zip(Z_per_xp, radii_per_xp)):
        rows.append((f'planar_coil_Z_idx{idx}', Z, None, None))
        for j, r in enumerate(radii):
            rows.append((f'planar_coil_r{j+1}_idx{idx}', r, None, None))

    # ---------------- write outputs ----------------------------------------
    out_dir = p.parent
    with open(out_dir / 'summary.txt', 'w') as f:
        f.write(f"# {'metric':<30s} {'value':>16s} {'threshold':>16s} {'rel_error':>16s}\n")
        for name, value, threshold, rel_err in rows:
            thr_str = f"{threshold:.6e}" if threshold is not None else "n/a"
            err_str = f"{rel_err:.6e}" if rel_err is not None else "n/a"
            f.write(f"  {name:<30s} {value:.6e}   {thr_str:>16s}   {err_str:>16s}\n")

    max_rel_err = max((abs(r[3]) for r in rows if r[3] is not None), default=0.0)
    with open(out_dir / 'max_rel_error.txt', 'w') as f:
        f.write(f"{max_rel_err:.18e}\n")

    print(f"Wrote summary.txt, max_rel_error.txt to {out_dir}")
    print(f"  max relative error = {max_rel_err:.3e}")


if __name__ == "__main__":
    main()
