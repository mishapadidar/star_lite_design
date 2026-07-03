#!/usr/bin/env python3
"""Generate separatrix manifolds / nested surfaces for the fixed points of a design, using the field
EXACTLY as loaded (no coil-geometry or current perturbation). Two input formats are accepted:

  * [BiotSavart, curve, ...]                  -- the standard-coil output of xpoint_to_opoint; the
                                                 BiotSavart already carries the modular + PF/aux coils.
  * the rich design [.., .., .., xpoints, ..] -- dat[3][0] is a SingularPeriodicFieldline whose
                                                 auxiliary coils define the total field (modular + aux).

The fixed points are the entries of the loaded xpoints list (used AS-IS, no re-fit) PLUS any new ones
from a multistart-Newton search near SEARCH_RZ (the X/O point by default) via
fixed_point_analysis.find_fixed_points. Each is classified and then:

  * HYPERBOLIC X-point / degenerate SNOWFLAKE -> separatrix MANIFOLDS: a geometric fundamental
    interval of seeds along each eigenvector leg, traced in the manifold's natural time-direction
    (unstable -> forward, stable -> backward).
  * PARABOLIC point (e.g. the tr=2 bifurcation) -> a FAN of seeds along the eigendirection, each
    traced BOTH forward and backward (its parabolic sectors flow toward the point on one side of a
    ray and away on the other, so one direction alone would miss the complementary branch).
  * ELLIPTIC O-point -> a few NESTED SURFACES: fieldlines at increasing radii from the O-point (out
    to a fraction of the distance to the nearest X-point), traced forward.

The fixed-point CURVES are saved (curves only), grouped by type, to <jsonstem>_fixedpoints.json as
load(out) = [BiotSavart, {type: [curve, ...]}] (type in hyperbolic / elliptic / parabolic / snowflake
/ ...). The phi=PHI Poincare hits go to <jsonstem>_allmanifolds.txt (columns seed_id=leg index, R, Z)
under header lines giving the fixed points, per-leg kinds (stable/unstable/surface) and leg
directions. The result is then auto-plotted (proc 0) with plot_manifolds.py.

Usage:
    mk_manifolds.py <design_json> [search_R search_Z]
"""
import sys
from pathlib import Path

import numpy as np
from simsopt._core import load, save
from simsopt.field import compute_fieldlines_xyz, ToroidalTransitStoppingCriterion, BiotSavart
from simsopt.util import comm_world

from star_lite_design.utils.singularperiodicfieldline import SingularPeriodicFieldline
from star_lite_design.utils.singularbiotsavart import SingularBiotSavart
from star_lite_design.utils.fixed_point_analysis import (leg_info, classify,
                                                         find_fixed_points as fp_search)

NMANIF = 10                              # fundamental-interval seeds per hyperbolic leg
EPS0 = 1e-3                              # innermost seed offset from the fixed point, metres
# Per-fixed-point-class transit budgets and time caps: each class (HYPERBOLIC / PARABOLIC /
# SNOWFLAKE X-points -> manifolds; ELLIPTIC O-points -> nested surfaces) has its OWN budget and
# tmax so they can be tuned independently.
BUDGET_HYP = 100                         # toroidal transits per HYPERBOLIC-manifold-seed trace
TMAX_HYP = 2000                          # per-call time cap for HYPERBOLIC-manifold traces

TMAX_PAR = 1000                          # per-call time cap for PARABOLIC-manifold traces
BUDGET_PAR = 1000                         # toroidal transits per PARABOLIC-manifold-seed trace

TMAX_SNOW = 500                         # per-call time cap for SNOWFLAKE-manifold traces
BUDGET_SNOW = 10                        # toroidal transits per SNOWFLAKE-manifold-seed trace

BUDGET_O = 100                           # toroidal transits per ELLIPTIC O-point nested-surface trace
TMAX_O = 5000                            # per-call time cap for ELLIPTIC O-point surface traces

N_SURF = 6                              # nested surfaces per O-point
SURF_FRAC = 0.7                         # outermost surface radius / distance to nearest X-point
SURF_RMAX = 1e-1                        # fallback outer surface radius (m) if no X-point nearby

PAR_RMAX = 2e-2                         # PARABOLIC: outer radius of the eigendirection seed fan (m)
N_PAR = 10                              # PARABOLIC: number of seeds in the eigendirection fan

TOL = 1e-8                             # field-line integration tolerance
PHI = 1e-10                             # Poincare section plane (radians); not exactly 0
DEDUPE_M = 1e-5                         # treat search hits within 1e-5 m of a known point as duplicates
# SEARCH_RZ (extra fixed-point search seed) is set after the load -- it defaults to the X/O fixed
# point's location (see below).


def make_bar(total, prefix):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=prefix, unit="line", leave=True)
    except ImportError:
        class _Bar:
            def __init__(self):
                self.n = 0
            def update(self, k=1):
                self.n += k
                sys.stdout.write(f"\r{prefix} {self.n}/{total}"); sys.stdout.flush()
            def close(self):
                sys.stdout.write("\n"); sys.stdout.flush()
        return _Bar()


p = Path(sys.argv[1])
dat = load(str(p))
if isinstance(dat[0], BiotSavart):
    # New format from xpoint_to_opoint: [standard BiotSavart, fixed-point curve]. The BiotSavart
    # already carries the modular + PF/aux coils as ordinary coils, so no SingularBiotSavart wrap.
    field = dat[0]
    xp_list = list(dat[1:])                           # every curve after the field (O-point, X-point, ...)
else:
    # Rich design format: [boozer_surfaces, iota_Gs, axes, xpoints, (sdf)]; dat[3][0] is a
    # SingularPeriodicFieldline that supplies the auxiliary coils (total field = modular + aux(mu)).
    fl = dat[3][0]
    assert isinstance(fl, SingularPeriodicFieldline), "dat[3][0] is not a SingularPeriodicFieldline"
    fl.need_to_run_code = False                       # use the loaded (modified) currents AS-IS
    field = SingularBiotSavart(fl)
    xp_list = dat[3]                                  # fixed points to process: the X and O point(s)

ref_curve = getattr(xp_list[0], "curve", xp_list[0])   # reference point (origin for sorting/header)
_g0 = np.asarray(ref_curve.gamma())[0]
REF_RX, REF_ZX = float(np.hypot(_g0[0], _g0[1])), float(_g0[2])
print(f"reference fixed point at R={REF_RX:.6f}, Z={REF_ZX:.6f}")

# Extra fixed-point search seed: default to the X/O fixed point (the LAST xpoints entry -- the one
# being converted), NOT the magnetic axis at xp_list[0]. Override with CLI args 2 and 3.
if len(sys.argv) > 3:
    SEARCH_RZ = (float(sys.argv[2]), float(sys.argv[3]))
else:
    _gxo = np.asarray(getattr(xp_list[-1], "curve", xp_list[-1]).gamma())[0]
    SEARCH_RZ = (float(np.hypot(_gxo[0], _gxo[1])), float(_gxo[2]))
print(f"extra fixed-point search seeded at R={SEARCH_RZ[0]:.6f}, Z={SEARCH_RZ[1]:.6f}")


def classify_fixed_points(field, xp_list):
    """Classify each fixed point in the loaded json's xpoints list using its curve AS-IS -- its own
    nfp/stellsym parametrization. classify()/leg_info() sample the curve only via evaluate_at_phi over
    [0, 1/nfp), and _fundamental_interval raises the leg eigenvalue to **nfp, so a field-period curve
    is handled correctly. Returns [(curve, (Rx, Zx), trace, type), ...] sorted by distance to the
    reference point."""
    found = []
    for obj in xp_list:
        curve = getattr(obj, "curve", obj)           # field-line wrapper, or a bare curve (new format)
        x0 = np.asarray(curve.gamma())[0]
        Rx, Zx = float(np.hypot(x0[0], x0[1])), float(x0[2])
        typ, tr = classify(curve, field, curve.nfp)
        found.append((curve, (Rx, Zx), tr, typ))
    found.sort(key=lambda f: (f[1][0] - REF_RX)**2 + (f[1][1] - REF_ZX)**2)
    print(f"using {len(found)} fixed point(s) from the xpoints list:")
    for curve, (Rx, Zx), tr, typ in found:
        extra = ""
        if typ == "elliptic":
            # Rotational transform of the O-point, same convention as tangent_map.iota_pure:
            # theta = arctan2(sqrt(4 - tr^2), tr) is the rotation over one field period (tr is the
            # field-period trace from classify), and iota = nfp*theta/(2*pi) per toroidal transit.
            theta = np.arctan2(np.sqrt(max(4.0 - tr * tr, 0.0)), tr)
            extra = f"  iota={curve.nfp * theta / (2.0 * np.pi):.5f}"
        print(f"  R={Rx:.6f} Z={Zx:.6f}  {typ:<10s} trace(M)={tr:+.4f}{extra}")
    return found


def trace_seed(field, x, y, z, sign, budget, tmax):
    """phi=PHI Poincare hits (M, 2) of (R, Z) for a single seed, traced in time-direction `sign`
    in one call capped at `budget` toroidal transits (ToroidalTransitStoppingCriterion) and `tmax`."""
    try:
        _, hits = compute_fieldlines_xyz(
            sign * field, np.array([x]), np.array([y]), np.array([z]),
            tmax=tmax, tol=TOL, comm=comm_world, phis=[PHI],
            stopping_criteria=[ToroidalTransitStoppingCriterion(budget, False)])
    except Exception as e:
        print(f"\n  seed trace failed (sign={sign:+.0f}): {e}")
        return np.empty((0, 2))
    h = hits[0]
    if h is None or np.asarray(h).size == 0:
        return np.empty((0, 2))
    hh = np.atleast_2d(np.asarray(h))
    return np.column_stack([np.hypot(hh[:, 2], hh[:, 3]), hh[:, 4]])


def _leg_seeds(field, curves, leg_id0, budget, tmax, r_vals_fn, both_dirs=False):
    """Shared per-leg manifold seeding. For each curve, leg_info gives the eigenvector legs; for each
    leg, r_vals_fn(speed, nfp) sets the seed radii along the ray and the seeds are traced in the
    manifold's natural direction (stable -> backward, unstable -> forward). With both_dirs=True each
    seed is emitted TWICE (forward AND backward): a parabolic sector flows toward the fixed point on
    one side of a ray and away on the other, so a single time-direction would miss the complementary
    branch. Returns (seeds, legdirs, kinds, next_leg_id) with seeds = [(x, y, z, sign, leg_id, budget,
    tmax)]."""
    seeds, legdirs, kinds = [], [], {}
    leg_id = leg_id0
    for xc in curves:
        _typ, rays, speeds, stable = leg_info(xc, field, xc.nfp)
        if rays is None:
            continue
        x0 = np.asarray(xc.gamma())[0]
        Rx, Zx = float(np.hypot(x0[0], x0[1])), float(x0[2])
        for k, v2 in enumerate(rays):
            legdirs.append((Rx, Zx, float(v2[0]), float(v2[1])))
            r_vals = r_vals_fn(float(speeds[k]), xc.nfp)
            Rl, Zl = Rx + r_vals * v2[0], Zx + r_vals * v2[1]
            sgn = -1.0 if stable[k] else +1.0           # unstable -> forward, stable -> backward
            signs = (+1.0, -1.0) if both_dirs else (sgn,)
            kinds[leg_id] = "unstable" if sgn > 0 else "stable"
            for R, Z in zip(Rl, Zl):
                for s in signs:
                    seeds.append((float(R), 0.0, float(Z), s, leg_id, budget, tmax))
            leg_id += 1
    return seeds, legdirs, kinds, leg_id


def _fundamental_interval(speed, nfp):
    """Geometric fundamental interval [EPS0/ratio, EPS0] with ratio = |eigenvalue|**nfp (or, for
    snowflakes, the quadratic radial coefficient used the same way)."""
    ratio = (abs(speed) if abs(speed) > 1.0 else 1.0 / abs(speed)) ** nfp
    return np.geomspace(EPS0 / ratio, EPS0, NMANIF)


def hyperbolic_seeds(field, curves, leg_id0):
    """Separatrix-manifold seeds for HYPERBOLIC X-points: NMANIF seeds on the geometric fundamental
    interval along each of the 4 eigenvector legs (BUDGET_HYP / TMAX_HYP)."""
    return _leg_seeds(field, curves, leg_id0, BUDGET_HYP, TMAX_HYP, _fundamental_interval)


def snowflake_seeds(field, curves, leg_id0):
    """Separatrix-manifold seeds for degenerate SNOWFLAKES (6 legs); same geometric fundamental
    interval as hyperbolic, with the quadratic radial coefficient in place of the eigenvalue
    (BUDGET_SNOW / TMAX_SNOW)."""
    return _leg_seeds(field, curves, leg_id0, BUDGET_SNOW, TMAX_SNOW, _fundamental_interval)


def parabolic_seeds(field, curves, leg_id0):
    """Seeds for PARABOLIC points (eigenvalue 1): the dynamics is ALGEBRAIC, not exponential, so the
    geometric (eigenvalue) fundamental interval is meaningless and degenerates when |c|~1. Instead
    seed a FAN of N_PAR points along the eigendirection out to PAR_RMAX and trace each for many
    (BUDGET_PAR) transits -- the Poincare dots then fill the parabolic invariant curves / separatrix
    (the standard way to plot a section near a parabolic point). Each seed is traced BOTH forward and
    backward (both_dirs=True), since a parabolic sector flows toward the point on one side of a ray
    and away on the other, so one time-direction alone would miss the complementary branch."""
    return _leg_seeds(field, curves, leg_id0, BUDGET_PAR, TMAX_PAR,
                      lambda speed, nfp: np.linspace(EPS0, PAR_RMAX, N_PAR), both_dirs=True)


def opoint_seeds(o_curves, hyp_xy, leg_id0):
    """Nested-surface seeds for each elliptic O-point: N_SURF fieldlines at increasing radii out to
    SURF_FRAC * (distance to the nearest X-point) (or SURF_RMAX). Returns (seeds, kinds), one
    leg_id per nested surface, kinds[leg_id] = 'surface'."""
    seeds, kinds = [], {}
    leg_id = leg_id0
    for oc in o_curves:
        x0 = np.asarray(oc.gamma())[0]
        Rx, Zx = float(np.hypot(x0[0], x0[1])), float(x0[2])
        if hyp_xy:
            d = min(np.hypot(Rx - hx, Zx - hz) for hx, hz in hyp_xy)
            r_max = min(SURF_FRAC * d, SURF_RMAX)
        else:
            r_max = SURF_RMAX
        for r in np.linspace(EPS0, r_max, N_SURF):
            seeds.append((Rx + r, 0.0, Zx, +1.0, leg_id, BUDGET_O, TMAX_O))
            kinds[leg_id] = "surface"
            leg_id += 1
    return seeds, kinds


found = classify_fixed_points(field, xp_list)

# Additionally search for fixed point(s) near the SEARCH_RZ seed (the X/O fixed point by default)
# with the shared multistart-Newton searcher fixed_point_analysis.find_fixed_points; add any new ones.
for fp in fp_search(field, ref_curve.nfp, False, R=SEARCH_RZ[0], Z=SEARCH_RZ[1],
                    order=ref_curve.order, ntor=ref_curve.ntor, n_seeds=64, search_radius=5e-2):
    _, (eRx, eZx), etr, etyp = fp
    if any(abs(eRx - q[0]) < DEDUPE_M and abs(eZx - q[1]) < DEDUPE_M for _, q, _, _ in found):
        continue                                          # already in the list
    found.append(fp)
    print(f"  searched fixed point (seed {SEARCH_RZ}): R={eRx:.6f} Z={eZx:.6f}  "
          f"{etyp:<10s} trace(M)={etr:+.4f}")

fixed_points = [(q, tr, typ) for _, q, tr, typ in found]                  # ((Rx, Zx), trace, type)

# Save the field + the fixed-point CURVES grouped by type (curves only): load(out) =
# [BiotSavart, {type: [curve, ...]}] with type in hyperbolic / elliptic / parabolic / snowflake / ...
if comm_world is None or comm_world.rank == 0:
    fp_by_type = {}
    for curve, _, _, typ in found:
        fp_by_type.setdefault(typ, []).append(curve)
    json_out = p.parent / f"{p.stem}_fixedpoints.json"
    save([field, fp_by_type], str(json_out))
    print(f"wrote {json_out.name}  ("
          + ", ".join(f"{len(v)} {k}" for k, v in fp_by_type.items()) + ")")

# Separatrix MANIFOLDS per fixed-point class (each has its own seeder): hyperbolic X-points,
# PARABOLIC points (e.g. the tr=2 bifurcation), and degenerate snowflakes. Elliptic O-points get
# nested surfaces instead.
hyp_curves = [c for c, _, _, typ in found if typ == "hyperbolic"]
par_curves = [c for c, _, _, typ in found if typ == "parabolic"]
snow_curves = [c for c, _, _, typ in found if typ == "snowflake"]
o_curves = [c for c, _, _, typ in found if typ == "elliptic"]
bdry_xy = [q for _, q, _, typ in found if typ in ("hyperbolic", "parabolic")]   # island boundaries

hseeds, hlegdirs, hkinds, nid = hyperbolic_seeds(field, hyp_curves, 0)
pseeds, plegdirs, pkinds, nid = parabolic_seeds(field, par_curves, nid)
sseeds, slegdirs, skinds, nid = snowflake_seeds(field, snow_curves, nid)
oseeds, okinds = opoint_seeds(o_curves, bdry_xy, nid)
seeds = hseeds + pseeds + sseeds + oseeds
legdirs = hlegdirs + plegdirs + slegdirs
kinds = {**hkinds, **pkinds, **skinds, **okinds}
n_legs = (max(s[4] for s in seeds) + 1) if seeds else 0
print(f"{len(hyp_curves)} hyperbolic, {len(par_curves)} parabolic, {len(snow_curves)} snowflake "
      f"fixed point(s); {len(o_curves)} elliptic O-point(s); {n_legs} legs and surfaces, "
      f"{len(seeds)} seeds")

out = p.parent / f"{p.stem}_allmanifolds.txt"
ndots = 0
with open(out, "w") as f:
    f.write(f"# seed_id,R,Z   phi={PHI}, n_fixedpoints={len(fixed_points)}, n_legs={n_legs}, "
            f"n_seeds={len(seeds)}, xpoint_RZ={REF_RX:.10e};{REF_ZX:.10e}\n")
    for (Rx, Zx), tr, typ in fixed_points:
        f.write(f"# fixedpt R={Rx:.10e} Z={Zx:.10e} type={typ} trace={tr:.10e}\n")
    for lid in sorted(kinds):
        f.write(f"# leg {lid} kind={kinds[lid]}\n")
    for (Rx, Zx, vR, vZ) in legdirs:
        f.write(f"# legdir Rx={Rx:.10e} Zx={Zx:.10e} vR={vR:.10e} vZ={vZ:.10e}\n")
    f.flush()
    bar = make_bar(len(seeds), "tracing")
    for (x, y, z, sgn, leg_id, budget, tmax) in seeds:
        rz = trace_seed(field, x, y, z, sgn, budget, tmax)
        if rz.size:
            np.savetxt(f, np.column_stack([np.full(rz.shape[0], leg_id), rz]),
                       fmt=["%d", "%.10e", "%.10e"], delimiter=",")
            f.flush()
            ndots += rz.shape[0]
        bar.update(1)
    bar.close()
print(f"{ndots} dots -> {out.name}")

# Plot the manifold / O-point dots straight from the JSON we just traced (proc 0 only).
if comm_world is None or comm_world.rank == 0:
    try:
        from plot_manifolds import plot_manifolds
        plot_manifolds(out, show=False)
    except Exception as e:
        print(f"plotting failed ({e}); plot manually with plot_manifolds.py {out.name}")
