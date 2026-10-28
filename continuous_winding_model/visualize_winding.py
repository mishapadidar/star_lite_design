from star_lite_design.continuous_winding_model.winding import continuous_winding
from simsopt._core import load
from scipy.spatial import cKDTree
from pyevtk.hl import polyLinesToVTK, pointsToVTK
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Circle
import argparse, os

"""
Pictures of the continuous winding of one coil, to check the model by eye.

The interesting region is near the start position t0, where the turns step to the
next slot and the leads come out. sigma is the arc length along the spine from t0
(negative = before t0).

- slice_sweep.mp4: the cross section of the pack sweeping along the spine through
  the region where the turns start. Numbers are the turn index (winding order); the
  cable current comes out of the page; normal to the right, binormal up.
- cross_sections.png: snapshots of that sweep.
- winding.mp4: the coil being wound.
- overlap.png: where two cables come closer than 2r in 3D (pass through each other).
- ParaView files: cable.vtu (one line per turn and lead, cell data "turn" 1-18 and 0
  for leads, "layer", "kind" 0 turn / 1 starting lead / 2 finishing lead),
  cable_layer1.vtu and cable_layer2.vtu (the starting lead is drawn as part of wind 1
  and the finishing lead as part of the last wind), leads.vtu, spine.vtu, overlaps.vtu.
  Use a Tube filter with radius 0.0108 to see the cable thickness.
- winding.npz: the path and the local coordinates, for analysis.

Run with
    module load gcc openmpi ffmpeg
    source ../../env_star_lite_design/bin/activate
    export PYTHONPATH="${PYTHONPATH}:/mnt/home/mpadidar/code/star_lite"
    python visualize_winding.py --coil 0 --directions 1 1
"""

parser = argparse.ArgumentParser()
parser.add_argument("--coil", type=int, default=0)
parser.add_argument("--n_winds", type=int, default=18)
parser.add_argument("--direction", type=int, default=1)
parser.add_argument("--start_slot", type=int, default=0)
parser.add_argument("--bend_length", type=float, default=0.01)
parser.add_argument("--lead_angle", type=float, default=None)
parser.add_argument("--no_video", action="store_true")
args = parser.parse_args()

infile = "../designs/designA/designA_after_scaled.json"
r = 0.0108 # cable minor radius [m]
lead_length = 0.3 # short leads, so the pictures stay readable

data = load(infile)
coil = data[0][0].biotsavart.coils[args.coil]
params = dict(n_winds=args.n_winds, radius=r, direction=args.direction, start_slot=args.start_slot,
              bend_length=args.bend_length, lead_angle=args.lead_angle, lead_length=lead_length)
path, info = continuous_winding(coil.curve, **params)

tag = f"coil{args.coil}_{args.n_winds}winds_dir{args.direction:+d}_start{args.start_slot}_bend{100*args.bend_length:g}cm"
outdir = f"./output/{tag}/"
os.makedirs(outdir, exist_ok=True)
print("writing to", outdir)

sigma, spine = info["sigma"], info["spine"]
layer = info["layer"]
n_turns = len(info["turns"])
cable_j, cable_uv, cable_wind = info["cable_j"], info["cable_uv"], info["cable_wind"]
cmap = plt.get_cmap("turbo")
turn_color = [cmap(k / (n_turns - 1)) for k in range(n_turns)]
layer_desc = f"inner layer, {'ccw' if args.direction > 0 else 'cw'} from slot {args.start_slot}"
starts = info["turn_start_sigma"]
print(f"finishing lead at sigma = {100*info['sigma_lead_out']:.1f} cm")

# region where the turns start and the leads come out
lo = min(info["sigma_lead_out"], -0.14) - 0.06
hi = 0.06

def nearest_j(sig):
    return int(np.argmin(np.abs(sigma - sig)))

def draw_slice(ax, sig):
    j = nearest_j(sig)
    ax.add_patch(Circle((0, 0), r, color="0.6"))
    ax.text(0, 0, "spine", ha="center", va="center", fontsize=6)
    for i in np.where(cable_j == j)[0]:
        u, v = cable_uv[i]
        k = cable_wind[i]
        ax.add_patch(Circle((u, v), r, facecolor=turn_color[k], edgecolor="k", alpha=0.6, lw=0.8))
        ax.text(u, v, str(k + 1), ha="center", va="center", fontsize=8, weight="bold")
    for s_lead, uv, name in [(0., info["lead_in_uv"], "start lead"), (info["sigma_lead_out"], info["lead_out_uv"], "end lead")]:
        if abs(sigma[j] - s_lead) < r:
            ax.plot(uv[:, 0], uv[:, 1], color="k", lw=8, alpha=0.4, solid_capstyle="round")
            ax.text(*uv[min(len(uv) - 1, 30)], name, fontsize=7)
    if info["sigma_lead_out"] + r < sigma[j] < 0:  # between the leads: the last wind has left as the end lead
        u, v = info["lead_out_uv"][0]
        ax.add_patch(Circle((u, v), r, fill=False, edgecolor="k", ls="--", lw=0.8))
        ax.text(u, v, f"({n_turns})", ha="center", va="center", fontsize=7)
    lim = 8 * r
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

np.savez(outdir + "winding.npz", path=path, **{k: v for k, v in info.items() if k != "turns"}, turns=np.array(info["turns"]))

# ---------------------------------------------------------------- cross sections
sigmas = list(np.linspace(lo, hi, 13)) + [0.25 * info["length"], 0.5 * info["length"]]
fig, axs = plt.subplots(3, 5, figsize=(18, 11.5))
for ax, sig in zip(axs.flat, sigmas):
    draw_slice(ax, sig)
    ax.set_title(f"$\\sigma$ = {100*sigma[nearest_j(sig)]:+.1f} cm", fontsize=10)
fig.suptitle(f"Coil {args.coil} cross sections along the spine ($\\sigma$ = arc length from the start t0), "
             f"cable current out of the page.\nNumbers: turn index in winding order. {layer_desc}. "
             f"Bend length {100*args.bend_length:g} cm. The last two panels are far from the start.\n"
             f"Between the end lead ($\\sigma$ = {100*info['sigma_lead_out']:.1f} cm) and the start lead ($\\sigma$ = 0) "
             f"only {n_turns - 1} cables cross the section: wind {n_turns} has left as the end lead (dashed).", fontsize=11)
fig.tight_layout()
fig.savefig(outdir + "cross_sections.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- unrolled: angle around the spine vs arc length
# top: each wind as a band one cable diameter wide around the spine (2r/rho in angle);
# bottom: distance from the spine (2r inner layer, 2sqrt3 r in a groove of the outer layer, 4r on top)
psi_deg, rho_c = np.degrees(info["cable_psi"]), info["cable_rho"]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
for k in range(n_turns):
    m = np.where(cable_wind == k)[0]
    sig_k, p_k, w_k, rho_k = 100 * sigma[cable_j[m]], psi_deg[m], np.degrees(r / rho_c[m]), rho_c[m] / r
    # split where sigma wraps from +L/2 to -L/2
    for piece in np.split(np.arange(m.size), np.where(np.diff(sig_k) < 0)[0] + 1):
        ax.fill_between(sig_k[piece], p_k[piece] - w_k[piece], p_k[piece] + w_k[piece], color=turn_color[k], alpha=0.35, lw=0)
        ax.plot(sig_k[piece], p_k[piece], color=turn_color[k], lw=1.5, label=f"wind {k + 1}" if piece[0] == 0 else None)
        ax2.plot(sig_k[piece], rho_k[piece], color=turn_color[k], lw=1.5)
for p_lead in (0, 360):
    ax.add_patch(plt.Rectangle((-100 * r, p_lead - 3), 200 * r, 6, facecolor="k", alpha=0.4))
    ax.text(0, p_lead + 5, "start lead", ha="center", fontsize=7)
ax.plot(100 * info["sigma_lead_out"], psi_deg[-1], "kX", ms=12)
ax.text(100 * info["sigma_lead_out"], psi_deg[-1] + 5, "end lead", ha="center", fontsize=7)
for q in range(13):
    ax.axhline(30 * q, color="0.7", lw=0.5, ls="--" if q % 2 == 0 else ":")
for q in (2, 2 * np.sqrt(3), 4):
    ax2.axhline(q, color="0.7", lw=0.5, ls="--")
ax.set_xlim(100 * lo, 100 * hi)
ax.set_ylim(-20, max(390, psi_deg.max() + 20))
ax.set_ylabel("angle around the spine $\\psi$ [deg]\n(dashed: inner slots and outer slots on top, dotted: outer grooves)")
ax2.set_ylabel("distance from spine / r")
ax2.set_xlabel("$\\sigma$, arc length along the spine from t0 [cm]")
ax.set_title(f"Coil {args.coil} unrolled: each wind as a band one cable wide; bands touch where the winds touch. {layer_desc}", fontsize=10)
ax.legend(fontsize=8, loc="upper left", ncol=2)
fig.tight_layout()
fig.savefig(outdir + "unrolled.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- overlaps: cables closer than 2r in 3D
labels = np.full(path.shape[0], -1)
for k, (a, b) in enumerate(info["turns"]):
    labels[a:b] = k
seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
arc = np.concatenate(([0.], np.cumsum(seg_len)))
tree = cKDTree(path)
pairs = np.array(sorted(tree.query_pairs(2*r - 5e-4)))
# ignore pairs that are close along the cable itself
if pairs.size:
    pairs = pairs[np.abs(arc[pairs[:, 0]] - arc[pairs[:, 1]]) > 3 * r]
dist = np.linalg.norm(path[pairs[:, 0]] - path[pairs[:, 1]], axis=1) if pairs.size else np.array([])
print(f"pairs of cable points closer than 2r - 0.5 mm: {len(dist)}" + (f", max overlap {1e3*np.max(2*r - dist):.1f} mm" if len(dist) else ""))
fig, ax = plt.subplots(figsize=(12, 5))
if len(dist):
    off = info["turns"][0][0]
    sig_p = np.array([sigma[cable_j[i - off]] if 0 <= i - off < cable_j.size else 0. for i in pairs[:, 0]])
    sc = ax.scatter(100*sig_p, 1e3*(2*r - dist), c=labels[pairs[:, 0]] + 1, cmap=cmap, vmin=0, vmax=n_turns, s=8)
    plt.colorbar(sc, ax=ax, label="turn (0 = lead)")
ax.set_xlabel("$\\sigma$, arc length from t0 [cm]")
ax.set_ylabel("overlap 2r - distance [mm]")
ax.set_title(f"Coil {args.coil}: where two cables are closer than 2r (pass through each other)")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(outdir + "overlap.png", dpi=130)
plt.close(fig)

# ---------------------------------------------------------------- ParaView files
segments = list(info["turns"]) + [(info["lead_in"][0], info["lead_in"][1] + 1), (info["lead_out"][0], info["lead_out"][1])]
turn_id = list(range(1, n_turns + 1)) + [0, 0]
layer_id = list(layer + 1) + [0, 0]
kind = [0] * n_turns + [1, 2]

def write_lines(name, segs, cell_data):
    pts = np.vstack([path[a:b] for a, b in segs])
    polyLinesToVTK(outdir + name, *[np.ascontiguousarray(pts[:, i]) for i in range(3)],
                   pointsPerLine=np.array([b - a for a, b in segs]), cellData=cell_data)

write_lines("cable", segments, {"turn": np.array(turn_id, dtype=float), "layer": np.array(layer_id, dtype=float), "kind": np.array(kind, dtype=float)})
# the leads are the same cable: the starting lead continues into wind 1 and the last wind into the finishing lead
with_leads = list(info["turns"])
with_leads[0] = (info["lead_in"][0], with_leads[0][1])
with_leads[-1] = (with_leads[-1][0], info["lead_out"][1])
for l in np.unique(layer):
    ks = [k for k in range(n_turns) if layer[k] == l]
    write_lines(f"cable_layer{l + 1}", [with_leads[k] for k in ks], {"turn": np.array([k + 1 for k in ks], dtype=float)})
write_lines("leads", [(info["lead_in"][0], info["lead_in"][1] + 1), (info["lead_out"][0], info["lead_out"][1])],
            {"kind": np.array([1., 2.])})
sp = np.vstack((spine, spine[:1]))
polyLinesToVTK(outdir + "spine", *[np.ascontiguousarray(sp[:, i]) for i in range(3)], pointsPerLine=np.array([sp.shape[0]]))
if len(dist):
    xo = 0.5 * (path[pairs[:, 0]] + path[pairs[:, 1]])
    pointsToVTK(outdir + "overlaps", *[np.ascontiguousarray(xo[:, i]) for i in range(3)], data={"overlap": np.ascontiguousarray(2*r - dist)})

if args.no_video:
    raise SystemExit

# ---------------------------------------------------------------- video: slice sweep
sweep = np.linspace(lo, hi, int(round((hi - lo) / 0.003)))
fig, ax = plt.subplots(figsize=(7, 7.5))
def update_slice(i):
    ax.clear()
    draw_slice(ax, sweep[i])
    ax.set_title(f"coil {args.coil}, $\\sigma$ = {100*sigma[nearest_j(sweep[i])]:+.1f} cm from the start (current out of page)\n{layer_desc}", fontsize=9)
anim = FuncAnimation(fig, update_slice, frames=len(sweep))
anim.save(outdir + "slice_sweep.mp4", writer=FFMpegWriter(fps=12), dpi=110)
plt.close(fig)

# ---------------------------------------------------------------- video: the coil being wound
fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(projection="3d")
lo3, hi3 = path.min(axis=0), path.max(axis=0)
ax.set_xlim(lo3[0], hi3[0]); ax.set_ylim(lo3[1], hi3[1]); ax.set_zlim(lo3[2], hi3[2])
ax.set_box_aspect(hi3 - lo3)
ax.plot(*np.vstack((spine, spine[:1])).T, color="0.5", lw=1)
lines = [ax.plot([], [], [], color=turn_color[k], lw=2)[0] for k in range(n_turns)]
lead_lines = [ax.plot([], [], [], color="k", lw=3)[0] for _ in range(2)]
head, = ax.plot([], [], [], "ko", ms=6)
title = ax.set_title("")
steps_per_turn = 24
frames = n_turns * steps_per_turn + 30
a_in, b_in = info["lead_in"]
def update_wind(f):
    lead_lines[0].set_data_3d(*path[a_in:b_in+1].T)
    done = min(f / steps_per_turn, n_turns)
    for k, (a, b) in enumerate(info["turns"]):
        frac = np.clip(done - k, 0, 1)
        m = int(frac * (b - a))
        lines[k].set_data_3d(*path[a:a+m].T)
        if 0 < frac < 1:
            head.set_data_3d(*path[a+m-1:a+m].T)
    if done >= n_turns:
        a, b = info["lead_out"]
        lead_lines[1].set_data_3d(*path[a:b].T)
        head.set_data_3d([], [], [])
    ax.view_init(elev=25, azim=-60 + 360 * f / frames)
    title.set_text(f"coil {args.coil}: turn {min(int(done) + 1, n_turns)} of {n_turns} (layer {layer[min(int(done), n_turns-1)] + 1})\n{layer_desc}")
    return lines
anim = FuncAnimation(fig, update_wind, frames=frames)
anim.save(outdir + "winding.mp4", writer=FFMpegWriter(fps=20), dpi=100)
plt.close(fig)
