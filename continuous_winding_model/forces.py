from star_lite_design.continuous_winding_model.winding import PolylineCurve
from simsopt.field import BiotSavart, Coil, Current
import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

"""
Magnetic (J x B) forces on the cables of the continuously wound coils of design A,
in current configuration 0, from the field of all 6 coils.

The force per unit length on the cable at each point is (I/18) t x B. It is split into
- the mean over the 18 cables at that spine point: the force on the pack as a whole,
  which pushes the pack against the spine (and is carried by the spine),
- the rest: what differs from cable to cable, mostly the pinch of the pack's own
  current, which squeezes the cables together toward the spine.
The radial component (along the line from the spine to the cable, positive outward)
says whether a cable is pressed toward the spine or pulled away from it.

B is evaluated at the polyline nodes. The two segments meeting at a node add nothing
there, so the self-field of the cable is finite; what remains of its own local field
(up to about 7 N/m) is small compared with the pack's field.

Needs output/designA/coil{i}.npz from wind_coils.py. Writes output/designA/forces.png
and forces.json.

Run with
    python forces.py
"""

outdir = "./output/designA/"
r = 0.0108
n_turns = 18
lead_region = 0.25  # |sigma| below this counts as near the leads [m]
weight = 78.0 * 9.81 / 55.7  # whole coil (cable and spine) per metre of cable [N/m], elastic_finite_build/README.md

data = [np.load(outdir + f"coil{i}.npz") for i in range(6)]
bs = BiotSavart([Coil(PolylineCurve(d["path"]), Current(float(d["current"]) / n_turns)) for d in data])

summary = {}
fig, axs = plt.subplots(2, 2, figsize=(15, 11), gridspec_kw=dict(width_ratios=[2, 1]))
for row, c in enumerate([0, 4]):
    d = data[c]
    a = d["turns"][0][0]
    j = d["cable_j"]
    X = d["path"][a:a + j.size]  # the cable in winding order, without the leads
    t = np.gradient(X, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    bs.set_points(X)
    B = bs.B()
    I = float(d["current"]) / n_turns
    F = I * np.cross(t, B)  # [N/m]

    # local frame of each cable point: radial direction from the spine
    n_, b_ = d["normal"][j], d["binormal"][j]
    phi = float(d["theta0"]) + d["cable_psi"]
    e_rho = np.cos(phi)[:, None] * n_ + np.sin(phi)[:, None] * b_
    sigma = d["sigma"]
    N = sigma.size
    count = np.bincount(j, minlength=N)
    F_pack = np.zeros((N, 3))
    np.add.at(F_pack, j, F)
    F_pack /= count[:, None]  # mean over the cables at each spine point
    F_rel = F - F_pack[j]
    F_rho = np.sum(F * e_rho, axis=1)
    F_rel_rho = np.sum(F_rel * e_rho, axis=1)
    far = np.abs(sigma[j]) > lead_region
    inner = d["cable_rho"] < 3 * r
    pack_force = n_turns * np.linalg.norm(F_pack, axis=1)
    seg = np.linalg.norm(np.diff(X, axis=0, append=X[:1]), axis=1)
    net = (F * seg[:, None]).sum(axis=0)

    s = {
        "cable current [A]": I,
        "force per cable, median |F| [N/m]": float(np.median(np.linalg.norm(F, axis=1))),
        "force per cable, max |F| [N/m]": float(np.linalg.norm(F, axis=1).max()),
        "force on the pack, max per length [N/m]": float(pack_force.max()),
        "force on the pack, mean per length [N/m]": float(pack_force.mean()),
        "net force on the coil [N]": net.tolist(),
        "pinch, inner layer, mean radial [N/m]": float(F_rel_rho[inner & far].mean()),
        "pinch, outer layer, mean radial [N/m]": float(F_rel_rho[~inner & far].mean()),
        "radial force, most inward [N/m]": float(F_rho.min()),
        "radial force, most outward [N/m]": float(F_rho.max()),
        "fraction of cable pulled away from the spine": float(np.mean(F_rho > 0)),
        "fraction of cable pulled away from the spine, near the leads": float(np.mean(F_rho[~far] > 0)),
        "weight of the coil per length of cable [N/m]": weight,
    }
    summary[f"coil {c}"] = s
    print(f"coil {c}:")
    for k, v in s.items():
        print(f"    {k:62s} {np.round(v, 1) if isinstance(v, list) else f'{v:.3g}'}")

    # along the spine: force on the pack, and the radial force on the cables
    order = np.argsort(sigma)
    ax = axs[row, 0]
    ax.plot(100 * sigma[order], pack_force[order] / 1e3, color="k", label="force on the pack (18 cables) [kN/m]")
    rho_max = np.full(N, -np.inf)
    rho_min = np.full(N, np.inf)
    np.maximum.at(rho_max, j, F_rho)
    np.minimum.at(rho_min, j, F_rho)
    ax.plot(100 * sigma[order], rho_max[order] / 1e3, color="tab:red", label="most outward radial force on a cable [kN/m]")
    ax.plot(100 * sigma[order], rho_min[order] / 1e3, color="tab:blue", label="most inward radial force on a cable [kN/m]")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("$\\sigma$, arc length along the spine from the leads [cm]")
    ax.set_title(f"coil {c}, current configuration 0, {I:.0f} A per cable")
    ax.legend(fontsize=8)

    # cross section where the force on the pack is largest, forces as arrows in the normal plane
    jm = int(np.argmax(np.where(np.abs(sigma) > lead_region, pack_force, 0)))
    ax = axs[row, 1]
    ax.add_patch(Circle((0, 0), r, color="0.6"))
    m = np.where(j == jm)[0]
    uv = d["cable_uv"][m]
    Fn, Fb = np.sum(F[m] * n_[m], axis=1), np.sum(F[m] * b_[m], axis=1)
    scale = 2 * r / np.abs(np.concatenate((Fn, Fb))).max()
    for (u, v), fn, fb, fr in zip(uv, Fn, Fb, F_rho[m]):
        ax.add_patch(Circle((u, v), r, facecolor="tab:red" if fr > 0 else "tab:blue", alpha=0.3, edgecolor="k"))
        ax.arrow(u, v, scale * fn, scale * fb, width=0.001, color="k", length_includes_head=True)
    ax.set_xlim(-6 * r, 6 * r)
    ax.set_ylim(-6 * r, 6 * r)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"force on each cable at $\\sigma$ = {100*sigma[jm]:+.0f} cm\n(largest force on the pack; "
                 f"red: pulled away from the spine)", fontsize=9)
fig.tight_layout()
fig.savefig(outdir + "forces.png", dpi=130)
with open(outdir + "forces.json", "w") as f:
    json.dump(summary, f, indent=2)
