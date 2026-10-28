from star_lite_design.continuous_winding_model.field_models import (
    load_design, closed_field_line, full_torus, filament_coils, hex_build_coils, continuous_coils, scale_to_axis_field,
    design_surface_full_torus, boozer, continuation, qs_error, surface_distance, outdir, r, n_turns)
from star_lite_design.continuous_winding_model.winding import PolylineCurve, refine
from star_lite_design.finite_build.hex_finite_build_coil import create_hexagonal_filament_grid
from simsopt.field import Coil, Current
import numpy as np
import argparse, json, time

"""
Effect of the continuous winding on the magnetic field of design A: magnetic axis,
X-point, rotational transform and QS error, for the coil models of field_models.py
(filament, hex build, continuous) and
- hex build (polyline): the hex build's 18 filaments drawn as polylines with the same
  spine grid as the continuous winding (1000 points per turn, midpoint rule), so that
  its difference with the continuous winding is only the winding (steps, climb, leads),
  not the discretization.
The currents of each model are scaled so that the mean |B| on its own axis is 0.0875 T.

The continuous winding breaks the stellarator symmetry and the field period symmetry,
so everything is computed on the full torus: the axis and the X-point are closed field
lines over one toroidal turn (find_x_point with nfp=1), and the Boozer surface is an
exact Boozer surface with nfp=1 and no stellarator symmetry, at the volume of the
design's surface, started from the design's surface. For the continuous winding it is
found by continuation from the hex build (polyline): the current is moved over to the
continuous winding in steps, each solve started from the last. Only the last step, with
all the current in the continuous winding, is the result.

Resolution: mpol = k, ntor = 2k on the full torus (k per field period).
resolution_study.py shows iota converged to 1e-9 from k = 11. At k = 10, the design's
resolution, the collocation equations have a second, spurious solution next to the true
one: the design's stored surface. Started from it, the solve stays on it.

Needs output/designA/coil{i}.npz from wind_coils.py. Writes output/designA/analysis_config{c}_k{k}.json.

Run with
    python analyze_winding.py --config 0 --k 12
"""

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=int, default=0, help="current configuration (0, 1, 2)")
parser.add_argument("--k", type=int, default=12, help="resolution of the Boozer surfaces, mpol = ntor per field period")
args = parser.parse_args()

bsurf, (iota0, G0), ma, xp = load_design(args.config)
s, label = bsurf.surface, bsurf.targetlabel
coils = bsurf.biotsavart.coils
mpol, ntor = args.k, 2 * args.k


def hex_build_polyline(coils, n=1000):
    out = []
    for c in coils:
        for f in create_hexagonal_filament_grid(refine(c.curve, np.linspace(0, 1, n, endpoint=False)), n_layers=2, radius=r):
            out.append(Coil(PolylineCurve(f.gamma()), Current(c.current.get_value() / n_turns)))
    return out


def solve(bs, x, iota, G):
    return boozer(bs, x, iota, G, mpol, ntor, label)


x0 = design_surface_full_torus(s, mpol, ntor).x
results = {}
for name, build in [("filament", filament_coils), ("hex build", hex_build_coils),
                    ("hex build (polyline)", hex_build_polyline), ("continuous", continuous_coils)]:
    t = time.time()
    bs, axis, ok_axis, factor = scale_to_axis_field(build(coils), ma)
    xpoint, ok_xp = closed_field_line(bs, xp)
    bs.set_points(full_torus(ma))
    modB = bs.AbsB()[:, 0].copy()
    if name == "continuous":
        h = results["hex build (polyline)"]
        last, path = continuation(h["bs"].coils, bs.coils, h["x"], h["iota"], h["G"], solve)
        print("    continuation through current fractions " + ", ".join(f"{lam:.3g}" for lam in path), flush=True)
        assert last is not None and path[-1] == 1, "continuation did not reach the continuous winding"
        res, surf = last
    else:
        res, surf = solve(bs, x0, iota0, G0 * factor)
    qs = qs_error(surf, bs, sDIM=20 * s.nfp)  # same resolution per field period as the symmetric solve
    results[name] = dict(bs=bs, axis=axis, xpoint=xpoint, modB=modB, iota=res["iota"], G=res["G"], qs=qs, x=surf.x.copy(),
                         factor=factor, success=dict(axis=bool(ok_axis), xpoint=bool(ok_xp), boozer=bool(res["success"])))
    print(f"{name:20s}: iota {res['iota']:.6f}, QS error {qs:.4e}, currents scaled by {factor:.6f}, "
          f"converged: axis {ok_axis}, X-point {ok_xp}, Boozer {res['success']} ({time.time()-t:.0f} s)", flush=True)
print(f"design's stored surface (a spurious solution at mpol = ntor = 10): iota {iota0:.6f}")


def compare(a, b):
    ra, rb = results[a], results[b]
    return {
        "max axis shift [mm]": 1e3 * np.linalg.norm(ra["axis"] - rb["axis"], axis=1).max(),
        "max X-point shift [mm]": 1e3 * np.linalg.norm(ra["xpoint"] - rb["xpoint"], axis=1).max(),
        "max |B| change on the design axis (relative)": np.abs(ra["modB"] - rb["modB"]).max() / rb["modB"].mean(),
        "iota change": ra["iota"] - rb["iota"],
        "QS error change (relative)": (ra["qs"] - rb["qs"]) / rb["qs"],
        "max distance between the Boozer surfaces [mm]": 1e3 * surface_distance(ra["x"], rb["x"], mpol, ntor),
    }


table = {f"{a} vs {b}": compare(a, b) for a, b in [("hex build", "filament"), ("hex build (polyline)", "hex build"),
                                                    ("continuous", "hex build (polyline)"), ("continuous", "filament")]}
print()
for k, v in table.items():
    print(k)
    for q, x in v.items():
        print(f"    {q:46s} {x:+.3e}")
summary = {name: {"iota": res["iota"], "QS error": res["qs"], "current scale factor": res["factor"],
                  "success": res["success"]} for name, res in results.items()}
summary["resolution"] = {"mpol": mpol, "ntor (full torus)": ntor}
with open(outdir + f"analysis_config{args.config}_k{args.k}.json", "w") as f:
    json.dump({"models": summary, "differences": table}, f, indent=2)
