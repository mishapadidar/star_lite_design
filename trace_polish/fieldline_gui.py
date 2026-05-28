#!/usr/bin/env python3
import os
import sys
import time
import threading
import queue
import numpy as np
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from simsopt._core import load
from simsopt.geo import CurveLength
from star_lite_design.utils.tangent_map import TangentMap
from simsopt.util import proc0_print, comm_world
from simsopt.field import compute_fieldlines_xyz

# ============================================================
# Configuration
# ============================================================
DATA_FILE = sys.argv[1]
OUT_DIR = "./output_manifold/"
os.makedirs(OUT_DIR, exist_ok=True)

TOL = 1e-10
POINT_SIZE = 8
CLICK_MARKER_SIZE = 50
DEFAULT_TMAX = 1000.0
DEFAULT_XLIM = (0.0, 1.5)
DEFAULT_ZLIM = (-0.75, 0.75)
N_PHIS = 9
SDF_GRID_NR = 60
SDF_GRID_NZ = 60

# ============================================================
# Modern theme
# ============================================================
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444",
    "axes.linewidth": 0.6,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.labelcolor": "#222",
    "xtick.color": "#444",
    "ytick.color": "#444",
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "font.family": "sans-serif",
    "figure.facecolor": "#fafafa",
    "axes.facecolor": "#ffffff",
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 9,
})

COLOR_XPOINT     = "#d62728"
COLOR_CLICK      = "#1f77b4"
COLOR_SURFACE    = "#222"
COLOR_SDF        = "#888"
COLOR_BTN_BG     = "#e8eef5"
COLOR_BTN_HOVER  = "#cad8e8"
COLOR_BTN_FG     = "#222"
COLOR_STATUS_BG  = "#f1f3f5"
COLOR_STATUS_FG  = "#222"


# ============================================================
# Helpers
# ============================================================
def evaluate_at_phi(curve, phi, tol=1e-10):
    """Find xyz on a closed curve at a given normalized phi (= true phi / (2pi)).

    Adapted from star_lite_design/scan/mk_manifolds.py.
    """
    phi += np.ceil(-phi)   # map phi to [0, 1)

    def curve_val(theta):
        return curve.gamma_pure(curve.x, np.array([theta]))[0]

    xyz0 = curve_val(0.)
    phi0 = np.arctan2(xyz0[1], xyz0[0]) / (2 * np.pi)
    phi_prime = phi - phi0
    phi_prime += np.ceil(-phi_prime)

    def theta2phi(theta_in, _phi0):
        xyz = curve_val(theta_in)
        angle = np.arctan2(xyz[1], xyz[0]) / (2 * np.pi) - _phi0
        angle += np.ceil(-angle)
        return angle

    def fun(theta):
        if theta == 1.:
            return 1. - phi_prime
        return theta2phi(theta, phi0) - phi_prime

    try:
        result = root_scalar(fun, bracket=[0, 1.])
        if not result.converged:
            return None, False
        xyz = curve_val(result.root)
        R = np.sqrt(xyz[0]**2 + xyz[1]**2)
        c, s = np.cos(2 * np.pi * phi), np.sin(2 * np.pi * phi)
        accurate = np.abs(-s * xyz[0] + c * xyz[1]) / R < tol
        return xyz, accurate
    except Exception:
        return None, False


class FieldlineGUI:
    def __init__(self):
        self._load_data()
        self._setup_plot()
        self.tmax_value = DEFAULT_TMAX

        self.click_artists = {}            # phi_idx -> matplotlib artist
        self.poincare_artists = []          # flat list across panels
        self.worker = None
        self.result_queue = queue.Queue()
        self.last_seed = None
        self.trace_count = 0

        self._connect_events()
        self._set_status("Ready. Click any subplot to launch a fieldline trace from that phi.")

    # --- Data / physics setup ------------------------------------------------
    def _load_data(self):
        data = load(DATA_FILE)
        self.boozer_surfaces = data[0]
        self.xpoints = data[3]
        self.sdf = data[4]

        for xp in self.xpoints:
            xp.need_to_run_code = False
            xp.res = {"length": CurveLength(xp.curve).J()}

        self.boozer_surface = self.boozer_surfaces[0]
        self.xpoint = self.xpoints[0]
        self.bs = self.boozer_surface.biotsavart

        self.tangent_map = TangentMap(self.xpoint, self.bs, 0.0)
        self.monodromy = self.tangent_map.matrix
        print("Monodromy matrix:")
        print(self.monodromy)
        print(np.trace(self.monodromy))

        self.curve = self.xpoint.curve
        self.nfp = self.curve.nfp

        self.phis_normalized = np.linspace(0, 0.25, N_PHIS)
        self.phis = self.phis_normalized * 2 * np.pi

    # --- GUI setup -----------------------------------------------------------
    def _setup_plot(self):
        self.fig = plt.figure(figsize=(12, 15))
        try:
            self.fig.canvas.manager.set_window_title("Fieldline tracer")
        except Exception:
            pass

        # Title bar
        self.fig.text(0.5, 0.965, "Interactive fieldline tracing",
                      ha="center", va="center",
                      fontsize=17, fontweight="bold", color="#222")
        self.fig.text(0.5, 0.94,
                      "Click any panel to launch a Poincare trace from that phi.",
                      ha="center", va="center", fontsize=10, color="#666")

        # 3x3 subplot grid with panels touching.
        gs = self.fig.add_gridspec(
            3, 3,
            left=0.07, right=0.97, top=0.90, bottom=0.18,
            wspace=0, hspace=0,
        )
        self.axes = []
        for k in range(N_PHIS):
            row, col = divmod(k, 3)
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(*DEFAULT_XLIM)
            ax.set_ylim(*DEFAULT_ZLIM)

            # Title inside the panel, centered near the top.
            ax.text(0.5, 0.97, f"phi / 2pi = {self.phis_normalized[k]:.3f}",
                    ha="center", va="top", transform=ax.transAxes,
                    fontsize=9, color="#333",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="none", alpha=0.85))

            # Only the edge rows / columns show tick labels and axis labels.
            if col == 0:
                ax.set_ylabel("Z")
            else:
                ax.tick_params(labelleft=False)
            if row == 2:
                ax.set_xlabel("R")
            else:
                ax.tick_params(labelbottom=False)

            self.axes.append(ax)

        # Static overlays: surface cross-section, SDF zero contour, x-point.
        R_grid = np.linspace(DEFAULT_XLIM[0], DEFAULT_XLIM[1], SDF_GRID_NR)
        Z_grid = np.linspace(DEFAULT_ZLIM[0], DEFAULT_ZLIM[1], SDF_GRID_NZ)
        RR, ZZ = np.meshgrid(R_grid, Z_grid, indexing="xy")
        for k, ax in enumerate(self.axes):
            phi = self.phis[k]
            phi_n = self.phis_normalized[k]

            # Surface cross-section (closed loop in (R, Z)).
            try:
                cs = np.asarray(self.boozer_surface.surface.cross_section(phi_n))
                cs = cs.reshape(-1, 3)
                cs_R = np.sqrt(cs[:, 0]**2 + cs[:, 1]**2)
                cs_Z = cs[:, 2]
                cs_R = np.concatenate([cs_R, [cs_R[0]]])
                cs_Z = np.concatenate([cs_Z, [cs_Z[0]]])
                ax.plot(cs_R, cs_Z, color=COLOR_SURFACE, linewidth=1.2)
            except Exception as e:
                proc0_print(f"surface.cross_section failed at phi/2pi={phi_n}: {e}")

            # Vessel SDF zero contour.
            try:
                XX = RR * np.cos(phi)
                YY = RR * np.sin(phi)
                D = self.sdf.eval(XX, YY, ZZ)
                ax.contour(RR, ZZ, D, levels=[0], colors=COLOR_SDF, linewidths=1.0)
            except Exception as e:
                proc0_print(f"sdf.eval failed at phi/2pi={phi_n}: {e}")

            # x-point marker via evaluate_at_phi.
            xyz, ok = evaluate_at_phi(self.curve, phi_n)
            if ok:
                Rx = np.sqrt(xyz[0]**2 + xyz[1]**2)
                ax.plot(Rx, xyz[2], marker="x", markersize=10, markeredgewidth=2.0,
                        color=COLOR_XPOINT, linestyle="none")

        # Status bar
        self.status_ax = self.fig.add_axes([0.07, 0.015, 0.90, 0.04])
        self.status_ax.set_facecolor(COLOR_STATUS_BG)
        self.status_ax.set_xticks([])
        self.status_ax.set_yticks([])
        for spine in self.status_ax.spines.values():
            spine.set_visible(False)
        self.status_text = self.status_ax.text(0.01, 0.5, "", ha="left", va="center",
                                                color=COLOR_STATUS_FG, fontsize=10)

        # Controls row
        ax_tmax = self.fig.add_axes([0.16, 0.075, 0.16, 0.040])
        self.text_tmax = TextBox(ax_tmax, "tmax  ",
                                  initial=f"{DEFAULT_TMAX:g}",
                                  color=COLOR_BTN_BG, hovercolor=COLOR_BTN_HOVER)
        self.text_tmax.on_submit(self._on_tmax_submit)
        self.text_tmax.label.set_color(COLOR_BTN_FG)
        self.text_tmax.label.set_fontsize(10)

        ax_clear = self.fig.add_axes([0.55, 0.075, 0.16, 0.040])
        self.btn_clear = Button(ax_clear, "Clear",
                                color=COLOR_BTN_BG, hovercolor=COLOR_BTN_HOVER)
        self.btn_clear.on_clicked(self._on_clear)
        self.btn_clear.label.set_fontsize(10)
        self.btn_clear.label.set_color(COLOR_BTN_FG)

        ax_reset = self.fig.add_axes([0.73, 0.075, 0.21, 0.040])
        self.btn_reset = Button(ax_reset, "Reset view",
                                 color=COLOR_BTN_BG, hovercolor=COLOR_BTN_HOVER)
        self.btn_reset.on_clicked(self._on_reset_view)
        self.btn_reset.label.set_fontsize(10)
        self.btn_reset.label.set_color(COLOR_BTN_FG)

    def _connect_events(self):
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        timer = self.fig.canvas.new_timer(interval=150)
        timer.add_callback(self._poll_worker)
        timer.start()
        self._timer = timer

    # --- UI actions ----------------------------------------------------------
    def _set_status(self, msg):
        self.status_text.set_text(msg)
        self.fig.canvas.draw_idle()

    def _on_reset_view(self, _event):
        for ax in self.axes:
            ax.set_xlim(*DEFAULT_XLIM)
            ax.set_ylim(*DEFAULT_ZLIM)
        self._set_status(
            f"View reset on all panels: X in [{DEFAULT_XLIM[0]}, {DEFAULT_XLIM[1]}], "
            f"Z in [{DEFAULT_ZLIM[0]}, {DEFAULT_ZLIM[1]}]."
        )

    def _on_clear(self, _event):
        for artist in self.click_artists.values():
            artist.remove()
        self.click_artists.clear()
        for artist in self.poincare_artists:
            artist.remove()
        self.poincare_artists.clear()
        self._set_status("Cleared click markers and Poincare dots.")
        self.fig.canvas.draw_idle()

    def _on_tmax_submit(self, text):
        try:
            val = float(text)
            if val <= 0:
                raise ValueError
            self.tmax_value = val
            self._set_status(f"tmax set to {val:g}")
        except Exception:
            self._set_status("Invalid tmax value")

    def _on_click(self, event):
        # If the navigation toolbar's pan/zoom tool is active, let it handle the
        # click (zoom rectangle, pan drag, etc.) instead of launching a trace.
        try:
            if self.fig.canvas.manager.toolbar.mode:
                return
        except AttributeError:
            pass

        # which subplot was clicked?
        phi_idx = None
        for k, ax in enumerate(self.axes):
            if event.inaxes is ax:
                phi_idx = k
                break
        if phi_idx is None:
            return
        if self.worker is not None and self.worker.is_alive():
            self._set_status("Tracing already running. Wait for completion.")
            return
        if event.xdata is None or event.ydata is None:
            return

        R = float(event.xdata)
        Z = float(event.ydata)
        phi = self.phis[phi_idx]
        x0 = R * np.cos(phi)
        y0 = R * np.sin(phi)
        z0 = Z
        self.last_seed = (x0, y0, z0)

        if phi_idx in self.click_artists:
            self.click_artists[phi_idx].remove()
        self.click_artists[phi_idx] = self.axes[phi_idx].scatter(
            [R], [Z], s=CLICK_MARKER_SIZE, marker="X",
            color=COLOR_CLICK, edgecolors="white", linewidths=1.5,
            zorder=5,
        )
        self.fig.canvas.draw_idle()

        self._set_status(
            f"Tracing seed at phi/2pi = {self.phis_normalized[phi_idx]:.3f}: "
            f"(X={x0:.6f}, Y={y0:.6f}, Z={z0:.6f}), tmax={self.tmax_value:g}."
        )
        self.worker = threading.Thread(
            target=self._trace_from_seed,
            args=(x0, y0, z0, self.trace_count, self.tmax_value),
            daemon=True,
        )
        self.worker.start()
        self.trace_count += 1

    # --- Fieldline tracing ---------------------------------------------------
    def _trace_from_seed(self, x0, y0, z0, trace_id, tmax_val):
        t_start = time.time()
        try:
            X0 = np.array([x0])
            Y0 = np.array([y0])
            Z0 = np.array([z0])

            proc0_print(f"Starting trace {trace_id} from seed: {(x0, y0, z0)}")
            all_phi_hits = []
            for sign in [+1.0, -1.0]:
                _, fieldlines_phi_hits = compute_fieldlines_xyz(
                    sign * self.bs,
                    X0, Y0, Z0,
                    tmax=tmax_val,
                    tol=TOL,
                    comm=comm_world,
                    phis=list(self.phis),
                )
                all_phi_hits.append(fieldlines_phi_hits)
            elapsed = time.time() - t_start
            result = {
                "trace_id": trace_id,
                "seed": (x0, y0, z0),
                "phi_hits": all_phi_hits,
                "elapsed": elapsed,
            }
            self.result_queue.put(("ok", result))
        except Exception as exc:
            self.result_queue.put(("error", str(exc)))

    def _poll_worker(self):
        try:
            kind, payload = self.result_queue.get_nowait()
        except queue.Empty:
            return
        if kind == "error":
            self._set_status(f"Tracing failed: {payload}")
            return
        self._handle_trace_result(payload)

    def _handle_trace_result(self, result):
        phi_hits = result["phi_hits"]
        total_pts = 0

        for phi_idx in range(len(self.phis)):
            pts = []
            for direction_hits in phi_hits:
                for rows in direction_hits:
                    if rows is None or rows.size == 0:
                        continue
                    mask = rows[:, 1] == phi_idx
                    hits = rows[mask, 2:]   # columns: X, Y, Z
                    if hits.size == 0:
                        continue
                    R = np.sqrt(hits[:, 0]**2 + hits[:, 1]**2)
                    Z = hits[:, 2]
                    pts.extend(zip(R, Z))

            if not pts:
                continue
            xy = np.asarray(pts)
            artist = self.axes[phi_idx].scatter(
                xy[:, 0], xy[:, 1],
                s=POINT_SIZE, alpha=0.75, edgecolors="none",
            )
            self.poincare_artists.append(artist)
            total_pts += len(xy)

            out = np.column_stack([phi_idx * np.ones(len(xy)), xy[:, 0], xy[:, 1]])
            np.savetxt(
                OUT_DIR + f"poincare_trace_{result['trace_id']}_phi{phi_idx}.txt",
                out,
                comments="",
                delimiter=",",
            )

        sx, sy, sz = result["seed"]
        self._set_status(
            f"Done in {result['elapsed']:.2f}s.  Seed=({sx:.6f}, {sy:.6f}, {sz:.6f}),  "
            f"Poincare dots across 9 panels = {total_pts}."
        )
        self.fig.canvas.draw_idle()

    # --- Run -----------------------------------------------------------------
    def show(self):
        plt.show()


def main():
    gui = FieldlineGUI()
    gui.show()


if __name__ == "__main__":
    main()
