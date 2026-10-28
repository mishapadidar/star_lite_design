from star_lite_design.continuous_winding_model.field_models import (
    load_design, closed_field_line, filament_coils, hex_build_coils, continuous_coils, path_lengths, outdir)
from simsopt.field import BiotSavart
import numpy as np
import json, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
Convergence of the Biot-Savart quadrature with the number of points on the coils.

The continuous winding is laid on a fine polyline (1000 points per turn, 3 mm apart),
which is needed to wind the cable but maybe not to compute its field. Each coil model
is discretized with n points per turn and its field compared with the field at a much
finer discretization (n_ref per turn), on the filament model's magnetic axis and on its
X-point (closed field lines):
- filament: the design's Fourier curves on n points per coil (trapezoid rule),
- hex build: its 18 filaments per coil on n points each (trapezoid rule),
- continuous, one spline: a periodic cubic spline through the winding's polyline
  (spline_curve), sampled uniformly at n points per mean turn length, leads included,
  trapezoid rule,
- continuous, split at corners: the spline split at the corners of the path (where the
  leads leave the pack at 90 degrees, and at the far ends of the leads), each smooth
  piece integrated with 4-point Gauss-Legendre panels, n points per mean turn length
  (spline_panel_curve),
- continuous, adaptive: the same with shorter panels in the tight bends near the leads.
Also shown: the polyline itself (midpoint rule), used by analyze_winding.py, against the
finest split spline. The x axis is the number of points actually used per turn: all
points over the number of turns (for the continuous winding the leads count as a
fraction of a turn).

Error: max over the points of |B_n - B_ref| / |B_ref|. The currents are the design's
(the error is relative, so the scaling to 0.0875 T on the axis does not matter).

Needs output/designA/coil{i}.npz from wind_coils.py. Writes output/designA/quadrature_study.png and .json.

Run with
    python quadrature_study.py
"""

config = 0
n_values = [8, 11, 16, 23, 32, 45, 64, 90, 128, 160, 256, 362, 512, 724, 1024, 1448, 2048]
n_ref = 8192
style = {"filament": dict(color="#2a78d6", ls="-", mfc="#2a78d6"),
         "hex build": dict(color="#eb6834", ls="-", mfc="#eb6834"),
         "continuous, one spline (trapezoid rule)": dict(color="#1baf7a", ls=":", mfc="white"),
         "continuous, split at corners (Gauss panels)": dict(color="#1baf7a", ls="--", mfc="white"),
         "continuous, adaptive (Gauss panels)": dict(color="#1baf7a", ls="-", mfc="#1baf7a")}
builders = {"filament": filament_coils, "hex build": hex_build_coils,
            "continuous, one spline (trapezoid rule)": lambda coils, n: continuous_coils(coils, n, rule="trapezoid"),
            "continuous, split at corners (Gauss panels)": lambda coils, n: continuous_coils(coils, n, rule="panels"),
            "continuous, adaptive (Gauss panels)": lambda coils, n: continuous_coils(coils, n, rule="adaptive")}

bsurf, _, ma, xp = load_design(config)
coils = bsurf.biotsavart.coils
bs = BiotSavart(filament_coils(coils))
points = {}
for name, guess in [("magnetic axis", ma), ("X-point", xp)]:
    points[name], ok = closed_field_line(bs, guess)
    assert ok, name
print("closed field lines of the filament model found", flush=True)
turns = {"filament": len(coils), "hex build": 18 * len(coils)}
turns_continuous = sum(np.divide(*path_lengths(np.load(outdir + f"coil{i}.npz"))) for i in range(len(coils)))


def field(coil_list):
    out = {}
    b = BiotSavart(coil_list)
    for name, xyz in points.items():
        b.set_points(xyz)
        out[name] = b.B().copy()
    return out


def error(B, B_ref):
    return {k: float(np.max(np.linalg.norm(B[k] - B_ref[k], axis=1) / np.linalg.norm(B_ref[k], axis=1))) for k in B}


def per_turn(model, coil_list):
    return sum(len(c.curve.quadpoints) for c in coil_list) / turns.get(model, turns_continuous)


results = {}
for model, build in builders.items():
    t = time.time()
    B_ref = field(build(coils, n_ref))
    results[model] = []
    for n in n_values:
        coil_list = build(coils, n)
        results[model].append(dict(n=n, points_per_turn=per_turn(model, coil_list), **error(field(coil_list), B_ref)))
    print(f"{model} ({time.time()-t:.0f} s)")
    for e in results[model]:
        print(f"    n {e['n']:5d}, {e['points_per_turn']:7.1f} points per turn: max relative error of B "
              f"on the axis {e['magnetic axis']:.2e}, on the X-point {e['X-point']:.2e}")
    if model == "continuous, split at corners (Gauss panels)":
        coil_list = continuous_coils(coils)
        polyline = dict(points_per_turn=per_turn(model, coil_list), **error(field(coil_list), B_ref))
        print(f"    polyline, {polyline['points_per_turn']:.0f} points per turn, midpoint rule: "
              f"axis {polyline['magnetic axis']:.2e}, X-point {polyline['X-point']:.2e}")

with open(outdir + "quadrature_study.json", "w") as f:
    json.dump({"reference n": n_ref, "max relative error of B": results, "continuous polyline (midpoint rule)": polyline}, f, indent=2)

fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, where in zip(axs, ["magnetic axis", "X-point"]):
    ax.axvline(160, color="0.6", lw=1, ls=":")
    ax.text(160 * 1.06, 0.97, "design: 160 per coil", color="0.35", fontsize=8, va="top", transform=ax.get_xaxis_transform())
    for model, res in results.items():
        ax.loglog([e["points_per_turn"] for e in res], [max(e[where], 1e-16) for e in res], marker="o", lw=2, ms=5,
                  label=model, **style[model])
    ax.loglog([polyline["points_per_turn"]], [polyline[where]], "*", color="#1baf7a", ms=14, mec="k", mew=0.6,
              label="continuous, polyline used so far (midpoint rule)")
    ax.set_xlabel("quadrature points per turn")
    ax.set_title(f"B on the filament model's {where}", fontsize=11)
    ax.grid(True, which="major", color="0.9", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].set_ylabel(f"max |B - B$_{{ref}}$| / |B$_{{ref}}$|")
axs[0].legend(fontsize=8, frameon=False, loc="lower left")
fig.suptitle("Biot-Savart quadrature convergence, design A, current configuration 0 (reference: the same rule with "
             f"n = {n_ref} per turn)\nfilament, hex build: trapezoid rule on their Fourier curves; "
             "continuous: a spline through the wound cable", fontsize=10)
fig.tight_layout()
fig.savefig(outdir + "quadrature_study.png", dpi=130)
