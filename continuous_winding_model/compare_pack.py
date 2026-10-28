from star_lite_design.continuous_winding_model.winding import refine
from simsopt._core import load
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
How different is the continuously wound pack from the idealized hexagonal pack of the
finite build model (finite_build/hex_finite_build_coil.py)?

At every spine point, each cable is compared with the nearest slot of the ideal
hexagon (2r for the inner layer; 4r on top and 2 sqrt(3) r in a groove for the outer
layer, in the same centroid frame). The largest distance over the cables at that
spine point is plotted along the spine, together with the curvature of the spine, to
see whether the pack bunches up where the spine bends sharply or only near the leads.

Needs output/designA/coil{i}.npz from wind_coils.py. Writes output/designA/pack_deviation.png.

Run with
    python compare_pack.py
"""

infile = "../designs/designA/designA_after_scaled.json"
outdir = "./output/designA/"
r = 0.0108
lead_region = 0.25  # |sigma| below this counts as near the leads [m]

coils = load(infile)[0][0].biotsavart.coils
fig, axs = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
for c, ax in zip([0, 4], axs):
    d = np.load(outdir + f"coil{c}.npz")
    cable_j, uv, sigma = d["cable_j"], d["cable_uv"], d["sigma"]
    N = sigma.size

    # ideal slots in the (normal, binormal) plane, in the same frame and angle convention
    theta0 = float(d["theta0"])
    psi_inner = np.arange(6) * np.pi / 3
    psi_outer = np.arange(12) * np.pi / 6
    rho_outer = np.where(np.arange(12) % 2 == 0, 4 * r, 2 * np.sqrt(3) * r)
    slots = np.vstack((2 * r * np.column_stack((np.cos(theta0 + psi_inner), np.sin(theta0 + psi_inner))),
                       rho_outer[:, None] * np.column_stack((np.cos(theta0 + psi_outer), np.sin(theta0 + psi_outer)))))
    dev = np.linalg.norm(uv[:, None, :] - slots[None, :, :], axis=2).min(axis=1)
    max_dev = np.zeros(N)
    np.maximum.at(max_dev, cable_j, dev)

    # curvature of the spine on the same grid
    spine = refine(coils[c].curve, np.mod(np.linspace(0, 1, N, endpoint=False), 1))
    kappa = spine.kappa()

    far = np.abs(sigma) > lead_region
    order = np.argsort(sigma)
    print(f"coil {c}: largest distance of a cable from its ideal slot: {1e3*max_dev[far].max():.3f} mm "
          f"more than {100*lead_region:.0f} cm from the leads, {1e3*max_dev.max():.1f} mm overall "
          f"(at sigma = {100*sigma[np.argmax(max_dev)]:+.1f} cm). Spine curvature up to {kappa.max():.1f} 1/m "
          f"(radius {100/kappa.max():.1f} cm) at sigma = {100*sigma[np.argmax(kappa)]:+.1f} cm")
    ax.semilogy(100 * sigma[order], 1e3 * np.maximum(max_dev[order], 1e-6), color="tab:blue")
    ax.set_ylabel("largest distance of a cable\nfrom its ideal slot [mm]", color="tab:blue")
    ax.axvspan(-100 * lead_region, 100 * lead_region, color="0.9")
    ax2 = ax.twinx()
    ax2.plot(100 * sigma[order], kappa[order], color="tab:red", lw=1)
    ax2.set_ylabel("spine curvature [1/m]", color="tab:red")
    ax.set_title(f"coil {c}: shaded, within {100*lead_region:.0f} cm of the leads")
axs[-1].set_xlabel("$\\sigma$, arc length along the spine from the leads [cm]")
fig.tight_layout()
fig.savefig(outdir + "pack_deviation.png", dpi=130)
