#!/usr/bin/env python3
"""
Stellarator Device Browser  (Streamlit edition)
===============================================

Browse stellarator-design output folders, view their rendering PNGs
side-by-side, and inspect the metrics stored in ``summary.txt``.

Each device folder is named with ``key=value`` parameters, e.g.::

    margin=0p12_well=0.0_Z=0_onvessel=1_distance=1_configID=2_vesselID=2_mono=0_num_aux=0_attempt=0/
        xs_output_*.png
        scene_top.png
        scene_left.png
        summary.txt
        nonQS.txt           (optional)
        max_rel_error.txt   (optional)

Parameter meanings
------------------
    margin     Optimization surface / X-point to vessel distance.
    Z          X-point has a constant Z coordinate.
    onvessel   The coils lie on the vessel.
    distance   X-point is a constant distance from the vessel.

Legends
-------
``mono``::

    0   trace(M) > 2
    1   trace(M) ≈ 2  and  M ≈ I
    2   trace(M) ≈ 2     (trace condition only)

``configID``::

    0   iota = 0.17
    1   iota = 0.13
    2   iota = 0.26
    3   iota = 0.37

``vesselID``::

    0   pill pipe
    1   renaissance
    2   torus

Favorites
---------
Devices can be flagged as favorites with the ⭐ toggle in the right pane.
The selection is persisted to ``.device_favorites.json`` in the scanned
root folder (one JSON list of paths relative to the root), so favorites
survive restarts and travel with the dataset.

Run with::

    pip install streamlit pandas pillow
    streamlit run device_browser.py
"""

from __future__ import annotations

import json
import os
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


def device_id(name: str) -> int:
    """Deterministic integer ID for a device folder name. MUST match the
    convention used in boozer_all.py (zlib.crc32 of the folder name) so the ID
    shown here equals the one used to seed that device's coil perturbation."""
    return zlib.crc32(name.encode())


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

PARAM_ORDER = [
    "margin", "well", "Z", "onvessel",
    "distance", "configID", "vesselID", "mono", "null", "num_aux", "attempt",
]

MONO_LEGEND = {
    "0": "trace(M) > 2",
    "1": "trace(M) ≈ 2  and  M ≈ I",
    "2": "trace(M) ≈ 2",
}

CONFIG_LEGEND = {
    "0": "iota = 0.17",
    "1": "iota = 0.13",
    "2": "iota = 0.26",
    "3": "iota = 0.37",
}

VESSEL_LEGEND = {
    "0": "pill pipe",
    "1": "renaissance",
    "2": "torus",
}

# Short tooltip help text for each parameter (shown as the ? icon on
# the selectbox).  Longer legends with multiple values are rendered as
# captions instead — see the parameter-filter loop below.
PARAM_HELP = {
    "margin":   "Optimization surface / X-point to vessel distance.",
    "Z":        "X-point has a constant Z coordinate.",
    "onvessel": "The coils lie on the vessel.",
    "distance": "X-point is a constant distance from the vessel.",
    "null":     "DN = double-null (stellsym); SN = single-null (bottom X-point pushed to the lower wall).",
}

DEFAULT_MAX_REL_ERR = 0.10   # percent  (i.e. 0.001 in fraction)

NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

FAVORITES_FILENAME = ".device_favorites.json"


# ─────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class Metric:
    name: str
    value: Optional[float]
    threshold: Optional[float]
    rel_error: Optional[float]


@dataclass
class Device:
    path: Path
    name: str
    params: dict[str, str] = field(default_factory=dict)

    xs_png:       Optional[Path] = None
    scene_top:    Optional[Path] = None
    scene_left:   Optional[Path] = None
    summary_path: Optional[Path] = None

    nonqs_pct:     Optional[float] = None   # percent
    max_rel_error: Optional[float] = None   # fraction
    metrics: list[Metric] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────

def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def parse_folder_params(name: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for token in name.split("_"):
        if "=" in token:
            k, v = token.split("=", 1)
            out[k] = v
    # Devices from before the polish step (no auxiliary planar coils) have no
    # num_aux token in their folder name; treat them as num_aux=0. Likewise
    # default attempt=0 for any folder missing that token.
    out.setdefault("num_aux", "0")
    out.setdefault("attempt", "0")
    # Devices predating the DN/SN split have no null token; treat them as DN.
    out.setdefault("null", "DN")
    return out


def parse_summary(path: Path) -> list[Metric]:
    metrics: list[Metric] = []
    try:
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            metrics.append(Metric(
                name=parts[0],
                value=_safe_float(parts[1]),
                threshold=_safe_float(parts[2]),
                rel_error=_safe_float(parts[3]),
            ))
    except OSError:
        pass
    return metrics


def read_first_float(path: Path) -> Optional[float]:
    try:
        m = NUMERIC_RE.search(path.read_text())
        return float(m.group(0)) if m else None
    except (OSError, ValueError):
        return None


def param_sort_key(value: str):
    """Sort parameter values numerically when possible (handles ``0p10``)."""
    s = value.replace("p", ".")
    try:
        return (0, float(s), value)
    except ValueError:
        return (1, 0.0, value)


# ─────────────────────────────────────────────────────────────────────────
# Favorites persistence
# ─────────────────────────────────────────────────────────────────────────

def device_key(device: Device, root: Path) -> str:
    """Stable identifier for a device: its path relative to the scan root."""
    try:
        return str(device.path.relative_to(root))
    except ValueError:
        return str(device.path)


def load_favorites(root: Path) -> set[str]:
    """Read favorited device keys from ``<root>/.device_favorites.json``."""
    p = root / FAVORITES_FILENAME
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return {str(x) for x in data}
    except (OSError, json.JSONDecodeError):
        pass
    return set()


def save_favorites(root: Path, favs: set[str]) -> None:
    """Persist favorites to ``<root>/.device_favorites.json``."""
    p = root / FAVORITES_FILENAME
    try:
        p.write_text(json.dumps(sorted(favs), indent=2))
    except OSError as e:
        st.warning(f"Could not save favorites to `{p}`: {e}")


# ─────────────────────────────────────────────────────────────────────────
# Scanning
# ─────────────────────────────────────────────────────────────────────────

def find_devices(root: Path, progress=None) -> list[Device]:
    """Recursively find every device folder under ``root``.

    A device folder contains at least one of ``xs*.png``,
    ``scene_top.png``, ``scene_left.png``.  Device folders never contain
    nested device folders, so we don't descend into them once detected,
    which keeps the scan fast even with hundreds of devices.
    """
    devices: list[Device] = []

    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        fset = set(filenames)

        xs_files = sorted(
            f for f in filenames
            if f.lower().startswith("xs") and f.lower().endswith(".png")
        )
        has_top  = "scene_top.png"  in fset
        has_left = "scene_left.png" in fset
        if not (xs_files or has_top or has_left):
            continue

        folder = Path(dirpath)
        device = Device(
            path=folder,
            name=folder.name,
            params=parse_folder_params(folder.name),
            xs_png      = folder / xs_files[0]       if xs_files else None,
            scene_top   = folder / "scene_top.png"   if has_top  else None,
            scene_left  = folder / "scene_left.png"  if has_left else None,
            summary_path= folder / "summary.txt"     if "summary.txt" in fset else None,
        )

        if device.summary_path is not None:
            device.metrics = parse_summary(device.summary_path)
            for m in device.metrics:
                if m.name.lower() == "nonqs_percent" and m.value is not None:
                    device.nonqs_pct = m.value
                    break

        if device.nonqs_pct is None and "nonQS.txt" in fset:
            device.nonqs_pct = read_first_float(folder / "nonQS.txt")

        if "max_rel_error.txt" in fset:
            device.max_rel_error = read_first_float(folder / "max_rel_error.txt")
        elif device.metrics:
            errs = [m.rel_error for m in device.metrics if m.rel_error is not None]
            if errs:
                device.max_rel_error = max(errs)

        devices.append(device)
        dirnames[:] = []  # device folders never contain other devices

        if progress is not None and len(devices) % 25 == 0:
            progress(len(devices))

    devices.sort(key=lambda d: (
        d.nonqs_pct is None,
        d.nonqs_pct if d.nonqs_pct is not None else float("inf"),
        d.name,
    ))
    if progress is not None:
        progress(len(devices))
    return devices


def unique_values(records: list[Device], key: str) -> list[str]:
    return sorted({r.params[key] for r in records if key in r.params},
                  key=param_sort_key)


# ─────────────────────────────────────────────────────────────────────────
# Streamlit caching: scan once per root folder, reuse across reruns
# ─────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def cached_scan(root_str: str) -> list[Device]:
    placeholder = st.empty()
    def progress(n: int):
        placeholder.info(f"Scanning… {n} devices found so far")
    devices = find_devices(Path(root_str), progress=progress)
    placeholder.empty()
    return devices


# ─────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Stellarator Device Browser",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Sidebar: root folder + filters ──────────────────────────────────────
with st.sidebar:
    st.title("⚙️  Device Browser")

    default_root = st.session_state.get("root_dir", str(Path.cwd()))
    root_input = st.text_input("Root folder", value=default_root,
                               help="Absolute or relative path to scan")
    cols = st.columns([1, 1])
    scan_clicked  = cols[0].button("🔍 Scan",   use_container_width=True,
                                   type="primary")
    rescan_clicked = cols[1].button("🔄 Rescan", use_container_width=True)

    if scan_clicked:
        st.session_state["root_dir"] = root_input
    if rescan_clicked:
        cached_scan.clear()
        st.session_state["root_dir"] = root_input
        # Force favorites to reload from disk on rescan
        for k in list(st.session_state.keys()):
            if k.startswith("favorites::"):
                del st.session_state[k]

if "root_dir" not in st.session_state or not st.session_state["root_dir"]:
    st.info("Choose a root folder in the sidebar and press **Scan**.")
    st.stop()

root_path = Path(st.session_state["root_dir"]).expanduser()
if not root_path.exists():
    st.error(f"Folder does not exist:\n`{root_path}`")
    st.stop()

with st.spinner(f"Scanning `{root_path}`…"):
    records = cached_scan(str(root_path))

if not records:
    st.warning("No device folders found under that root.")
    st.stop()


# ── Load favorites for this root (cached in session state) ─────────────
fav_state_key = f"favorites::{root_path}"
if fav_state_key not in st.session_state:
    st.session_state[fav_state_key] = load_favorites(root_path)
favorites: set[str] = st.session_state[fav_state_key]


# ── Sidebar: thresholds and parameter filters ───────────────────────────
with st.sidebar:
    st.markdown("---")
    st.subheader("Thresholds")

    show_all_rerr = st.checkbox("Show all (ignore rel-err threshold)", value=False)
    max_rerr_pct = st.number_input(
        "Max constraint relative error (%)",
        min_value=0.0, value=DEFAULT_MAX_REL_ERR, step=0.01,
        format="%.3f",
        disabled=show_all_rerr,
    )

    fav_only = st.checkbox(
        f"⭐ Favorites only ({len(favorites)})",
        value=False,
        help=f"Stored in `{FAVORITES_FILENAME}` under the scan root.",
    )

    st.markdown("---")
    st.subheader("Parameter filters")

    available_keys = [k for k in PARAM_ORDER
                      if any(k in r.params for r in records)]
    extras = sorted({k for r in records for k in r.params
                     if k not in available_keys})
    keys = available_keys + extras

    selections: dict[str, str] = {}
    for key in keys:
        values = ["Any"] + unique_values(records, key)
        selections[key] = st.selectbox(
            key, values, key=f"sel_{key}",
            help=PARAM_HELP.get(key),
        )
        if key == "mono":
            st.caption(
                "**mono** values:\n\n"
                + "\n".join(f"- **{k}**: {v}" for k, v in MONO_LEGEND.items())
            )
        elif key == "configID":
            st.caption(
                "**configID** values:\n\n"
                + "\n".join(f"- **{k}**: {v}" for k, v in CONFIG_LEGEND.items())
            )
        elif key == "vesselID":
            st.caption(
                "**vesselID** values:\n\n"
                + "\n".join(f"- **{k}**: {v}" for k, v in VESSEL_LEGEND.items())
            )


# ── Apply filters ───────────────────────────────────────────────────────
max_rerr_thr = None if show_all_rerr else float(max_rerr_pct) / 100.0

def keep(r: Device) -> bool:
    for k, choice in selections.items():
        if choice != "Any" and r.params.get(k) != choice:
            return False
    if (max_rerr_thr is not None and r.max_rel_error is not None
            and r.max_rel_error > max_rerr_thr):
        return False
    if fav_only and device_key(r, root_path) not in favorites:
        return False
    return True

filtered = [r for r in records if keep(r)]
filtered.sort(key=lambda d: (
    d.nonqs_pct is None,
    d.nonqs_pct if d.nonqs_pct is not None else float("inf"),
    d.name,
))


# ── Main header / status ───────────────────────────────────────────────
st.title("Stellarator Device Browser")
notes = []
if max_rerr_thr is not None:
    notes.append(f"max rel. err ≤ {max_rerr_thr*100:g}%")
if fav_only:
    notes.append("favorites only")
suffix = "  ·  " + ", ".join(notes) if notes else ""
st.caption(
    f"Root: `{root_path}`  ·  "
    f"Showing **{len(filtered)}** of **{len(records)}** devices{suffix}"
)

if not filtered:
    st.warning("No devices match the current filter.")
    st.stop()


# ── Device list (selectable dataframe) ─────────────────────────────────
def df_for_devices(records: list[Device]) -> pd.DataFrame:
    rows = []
    for r in records:
        is_fav = device_key(r, root_path) in favorites
        rows.append({
            "⭐":              "★" if is_fav else "",
            "ID":             device_id(r.name),
            "Device":         r.name,
            "QS error (%)":   r.nonqs_pct,
            "Max rel err (%)": (r.max_rel_error * 100.0
                                if r.max_rel_error is not None else None),
            **{k: r.params.get(k, "") for k in keys},
        })
    return pd.DataFrame(rows)

df = df_for_devices(filtered)

left, right = st.columns([0.42, 0.58], gap="large")

with left:
    st.subheader("Devices")
    # st.dataframe supports row selection in recent Streamlit (≥ 1.35).
    # The Device column is forced narrow ("small") because device names
    # are long; users can still see the full name by hovering or by
    # expanding the column manually.
    event = st.dataframe(
        df.style.format({
            "QS error (%)":     "{:.4g}",
            "Max rel err (%)":  "{:.4g}",
        }, na_rep="n/a").background_gradient(
            subset=["QS error (%)"], cmap="RdYlGn_r", vmin=0, vmax=30,
        ).background_gradient(
            subset=["Max rel err (%)"], cmap="RdYlGn_r", vmin=0, vmax=20,
        ),
        use_container_width=True,
        height=420,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="device_table",
        column_config={
            "⭐": st.column_config.TextColumn(
                "⭐",
                width="small",
                help="Favorited devices. Toggle in the right pane.",
            ),
            "ID": st.column_config.NumberColumn(
                "ID",
                format="%d",
                width="small",
                help="Deterministic crc32 of the folder name; also the "
                     "seed used for that device's base-coil perturbation.",
            ),
            "Device": st.column_config.TextColumn(
                "Device",
                width="small",
                help="Full folder name (truncated; hover to see all).",
            ),
        },
    )

    sel_rows = event.selection.rows if hasattr(event, "selection") else []
    sel_idx = sel_rows[0] if sel_rows else 0
    selected_device = filtered[sel_idx]

    # Stats table — directly below the devices table
    st.markdown("##### summary.txt")
    if selected_device.summary_path is None:
        st.info("No `summary.txt` in this folder.")
    elif not selected_device.metrics:
        st.info("`summary.txt` could not be parsed.")
    else:
        st.caption(f"`{selected_device.summary_path}`")

        # Monodromy (tangent) matrix per X-point, reconstructed from the
        # monodromy_Mab_idx* rows written by boozer_all.py / compute_summary.py.
        mono_entries: dict[int, dict[tuple[int, int], float]] = {}
        for mm in selected_device.metrics:
            mo = re.match(r"monodromy_M(\d)(\d)_idx(\d+)$", mm.name)
            if mo and mm.value is not None:
                a, b, idx = int(mo.group(1)), int(mo.group(2)), int(mo.group(3))
                mono_entries.setdefault(idx, {})[(a, b)] = mm.value
        for idx in sorted(mono_entries):
            e = mono_entries[idx]
            if all((a, b) in e for a in (0, 1) for b in (0, 1)):
                tr = e[(0, 0)] + e[(1, 1)]
                st.markdown(f"**Monodromy matrix** — X-point {idx}  ·  tr(M) = {tr:.6f}")
                M = pd.DataFrame([[e[(0, 0)], e[(0, 1)]],
                                  [e[(1, 0)], e[(1, 1)]]])
                st.dataframe(M.style.format("{:.6e}"),
                             hide_index=True, use_container_width=True)

        stats_df = pd.DataFrame([
            {
                "Metric":         mm.name,
                "Value":          mm.value,
                "Threshold":      mm.threshold,
                "Relative error": mm.rel_error,
            }
            for mm in selected_device.metrics
        ])
        st.dataframe(
            stats_df.style.format({
                "Value":          "{:.6e}",
                "Threshold":      "{:.6e}",
                "Relative error": "{:.4e}",
            }, na_rep="n/a").background_gradient(
                subset=["Relative error"], cmap="RdYlGn_r",
                vmin=0, vmax=0.2,
            ),
            use_container_width=True,
            hide_index=True,
            height=460,
        )


# ── Right column: images ───────────────────────────────────────────────
with right:
    st.subheader(selected_device.name)
    st.caption(f"ID: `{device_id(selected_device.name)}`")

    # Favorite toggle for the currently selected device.  Using a per-device
    # widget key so the toggle's visible state always matches the file-backed
    # set when switching between devices.
    sel_key = device_key(selected_device, root_path)
    is_fav = sel_key in favorites
    new_fav = st.toggle(
        "⭐ Favorite this device",
        value=is_fav,
        key=f"fav_toggle::{sel_key}",
        help=f"Saved to `{root_path / FAVORITES_FILENAME}`",
    )
    if new_fav != is_fav:
        if new_fav:
            favorites.add(sel_key)
        else:
            favorites.discard(sel_key)
        save_favorites(root_path, favorites)
        st.rerun()

    m = selected_device.params.get("mono")
    if m in MONO_LEGEND:
        st.caption(f"`mono={m}`  ·  {MONO_LEGEND[m]}")
    cid = selected_device.params.get("configID")
    if cid in CONFIG_LEGEND:
        st.caption(f"`configID={cid}`  ·  {CONFIG_LEGEND[cid]}")
    vid = selected_device.params.get("vesselID")
    if vid in VESSEL_LEGEND:
        st.caption(f"`vesselID={vid}`  ·  {VESSEL_LEGEND[vid]}")

    # Images: xs taller on the left, scene_top / scene_left stacked on the right
    img_left, img_right = st.columns([1.6, 1.0], gap="small")
    with img_left:
        if selected_device.xs_png and selected_device.xs_png.exists():
            st.image(str(selected_device.xs_png),
                     caption=selected_device.xs_png.name,
                     use_container_width=True)
        else:
            st.info("xs_*.png missing")
    with img_right:
        if selected_device.scene_top and selected_device.scene_top.exists():
            st.image(str(selected_device.scene_top),
                     caption="scene_top.png",
                     use_container_width=True)
        else:
            st.info("scene_top.png missing")
        if selected_device.scene_left and selected_device.scene_left.exists():
            st.image(str(selected_device.scene_left),
                     caption="scene_left.png",
                     use_container_width=True)
        else:
            st.info("scene_left.png missing")
