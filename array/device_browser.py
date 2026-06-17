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

import html
import json
import os
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


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
    "distance", "configID", "vesselID", "mono", "null", "attempt",
]

# Parameters parsed from folder names but deliberately NOT shown as a filter
# selectbox in the sidebar.
HIDDEN_PARAMS = {"num_aux"}

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

NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

FAVORITES_FILENAME = ".device_favorites.json"
HIDDEN_FILENAME = ".device_hidden.json"


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


# summary.txt vessel-intersection flags (1 = intersects the vacuum vessel, 0 = not):
#   - inward_manifold_hits_vessel: any inward (plasma-pointing) X-point manifold leg
#     reaches the vessel (written by mk_manifolds.py).
#   - LCFS_hits_vessel: the last closed flux surface pokes outside the vessel
#     (written by mk_LCFS.py).
MANIFOLD_INTERSECT_METRIC = "inward_manifold_hits_vessel"
LCFS_INTERSECT_METRIC = "LCFS_hits_vessel"


def metric_value(device: "Device", name: str) -> Optional[float]:
    """Look up a parsed summary.txt metric value by name (None if absent)."""
    for m in device.metrics:
        if m.name == name:
            return m.value
    return None


def vessel_hit_state(device: "Device", source: str) -> tuple[bool, bool]:
    """(hit, known) for the chosen vessel-intersection source.
    hit   = the selected condition(s) intersect the vessel.
    known = enough metrics are present to decide 'no intersection' (vs 'unknown').
    source is 'manifolds', 'LCFS', or 'either' (intersects if EITHER does)."""
    vm = metric_value(device, MANIFOLD_INTERSECT_METRIC)
    vl = metric_value(device, LCFS_INTERSECT_METRIC)
    hm = vm is not None and vm >= 0.5
    hl = vl is not None and vl >= 0.5
    if source == "manifolds":
        return hm, (vm is not None)
    if source == "LCFS":
        return hl, (vl is not None)
    # "either": intersects if either hits; only fully known when BOTH are present.
    return (hm or hl), (vm is not None and vl is not None)


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


def load_hidden(root: Path) -> set[str]:
    """Read hidden device keys from ``<root>/.device_hidden.json``."""
    p = root / HIDDEN_FILENAME
    if not p.exists():
        return set()
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return {str(x) for x in data}
    except (OSError, json.JSONDecodeError):
        pass
    return set()


def save_hidden(root: Path, keys: set[str]) -> None:
    """Persist hidden device keys to ``<root>/.device_hidden.json``."""
    p = root / HIDDEN_FILENAME
    try:
        p.write_text(json.dumps(sorted(keys), indent=2))
    except OSError as e:
        st.warning(f"Could not save hidden list to `{p}`: {e}")


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

# Global font-size bump. Streamlit sizes most text in rem, so raising the root
# html font-size scales the whole UI proportionally (markdown, captions, headers,
# buttons, widgets, sidebar — and the rem-based metadata table). Bump BASE_FONT_PX
# to taste. NOTE: st.dataframe renders on a canvas and is NOT affected by CSS.
BASE_FONT_PX = 19
st.markdown(
    f"""
    <style>
    html {{ font-size: {BASE_FONT_PX}px; }}
    /* Trim Streamlit's large default top padding and tighten vertical gaps so the
       device name + Prev/Next + the two panes fit on screen without scrolling down
       to the device list. */
    .block-container, [data-testid="stMainBlockContainer"] {{
        padding-top: 1.5rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
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


# ── Load favorites + hidden list for this root (cached in session state) ─
fav_state_key = f"favorites::{root_path}"
if fav_state_key not in st.session_state:
    st.session_state[fav_state_key] = load_favorites(root_path)
favorites: set[str] = st.session_state[fav_state_key]

hidden_state_key = f"hidden::{root_path}"
if hidden_state_key not in st.session_state:
    st.session_state[hidden_state_key] = load_hidden(root_path)
hidden: set[str] = st.session_state[hidden_state_key]


# ── Sidebar: thresholds and parameter filters ───────────────────────────
with st.sidebar:
    # Filled once the filters are applied (the device count is only known then).
    # Sits at the top of the filters so the "Showing X of Y devices" status no
    # longer needs a title row in the main pane.
    status_slot = st.empty()
    st.markdown("---")
    st.subheader("Thresholds")

    fav_only = st.checkbox(
        f"⭐ Favorites only ({len(favorites)})",
        value=False,
        help=f"Stored in `{FAVORITES_FILENAME}` under the scan root.",
    )

    # Hidden devices are excluded by default; this shows ONLY the hidden ones (the
    # "hidden section"), where Ctrl-D / the button unhides them again.
    hidden_only = st.checkbox(
        f"🚫 Hidden only ({len(hidden)})",
        value=False,
        help=f"Hidden devices are excluded normally; tick to review/unhide them. "
             f"Stored in `{HIDDEN_FILENAME}` under the scan root.",
    )

    id_query = st.text_input(
        "🔎 Find by device ID",
        key="id_search",
        placeholder="e.g. 2880890725",
        help="Jump to the device with this ID (the crc32 in the ID column).",
    )

    intersect_source = st.selectbox(
        "Vessel intersection — source",
        ["either", "manifolds", "LCFS"],
        help="Which condition counts as 'intersecting the vessel': the inward "
             "(plasma-pointing) X-point manifold legs (`inward_manifold_hits_vessel`, "
             "mk_manifolds.py), the LCFS surface (`LCFS_hits_vessel`, mk_LCFS.py), or "
             "EITHER of them.",
    )
    intersect_choice = st.selectbox(
        "Vessel intersection",
        ["Any", "Intersects", "No intersection", "Unknown"],
        help="Filter on the chosen source. 'Intersects' = the source hits the vessel; "
             "'No intersection' = it does not (and is known — for 'either', both flags "
             "present); 'Unknown' = not computed (no LCFS/manifold render yet).",
    )

    st.markdown("---")
    st.subheader("Sort")
    # Server-side sort: this is what the device table AND the ← / → navigation follow. (The
    # table's header-click sorting only reorders the view client-side -- Streamlit doesn't report
    # it to Python, so navigation can't follow it; this control makes the order explicit so the
    # arrows always step through devices in the order you chose.)
    SORT_QS = "QS error (%)"
    _metric_names, _seen = [], set()
    for _r in records:
        for _m in _r.metrics:
            if _m.name not in _seen:
                _seen.add(_m.name)
                _metric_names.append(_m.name)
    sort_options = [SORT_QS] + sorted(_metric_names)
    sort_by = st.selectbox(
        "Sort devices by", sort_options, index=0,
        help="Orders the table and the ← / → navigation (e.g. pick 'aspect_ratio' to step "
             "through devices by aspect ratio).",
    )
    sort_desc = st.checkbox("Descending", value=False)

    st.markdown("---")
    st.subheader("Parameter filters")

    available_keys = [k for k in PARAM_ORDER
                      if k not in HIDDEN_PARAMS and any(k in r.params for r in records)]
    extras = sorted({k for r in records for k in r.params
                     if k not in available_keys and k not in HIDDEN_PARAMS})
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
def keep(r: Device) -> bool:
    rk = device_key(r, root_path)
    # Hidden handling: "Hidden only" shows just the hidden devices; otherwise hidden
    # devices are excluded entirely.
    if hidden_only:
        if rk not in hidden:
            return False
    elif rk in hidden:
        return False
    for k, choice in selections.items():
        if choice != "Any" and r.params.get(k) != choice:
            return False
    if fav_only and rk not in favorites:
        return False
    if intersect_choice != "Any":
        hit, known = vessel_hit_state(r, intersect_source)
        if intersect_choice == "Intersects" and not hit:
            return False
        if intersect_choice == "No intersection" and not ((not hit) and known):
            return False
        if intersect_choice == "Unknown" and not ((not hit) and (not known)):
            return False
    return True

# Device-ID search over ALL devices: a fresh query reveals + selects the matching
# device, even if the current filters/hidden would exclude it. The match persists
# (revealed_key) so it stays in the list and selectable until the query changes or
# is cleared.
if id_query != st.session_state.get("last_search"):
    st.session_state["last_search"] = id_query
    _q = id_query.strip()
    _match = None
    if _q:
        try:
            _tid = int(_q)
        except ValueError:
            _tid = None
        if _tid is not None:
            _match = next((r for r in records if device_id(r.name) == _tid), None)
    if _match is not None:
        st.session_state["revealed_key"] = device_key(_match, root_path)
        st.session_state["sel_key_nav"] = st.session_state["revealed_key"]
    else:
        st.session_state["revealed_key"] = None
        if _q:
            st.sidebar.caption(f"⚠️ No device with ID `{_q}`")
revealed_key = st.session_state.get("revealed_key")

filtered = [r for r in records if keep(r)]
# Reveal the searched device even if the filters/hidden would have excluded it.
if revealed_key is not None and all(device_key(r, root_path) != revealed_key for r in filtered):
    _rev = next((r for r in records if device_key(r, root_path) == revealed_key), None)
    if _rev is not None:
        filtered.append(_rev)
def _sort_value(d: Device) -> Optional[float]:
    if sort_by == SORT_QS:
        return d.nonqs_pct
    return next((m.value for m in d.metrics if m.name == sort_by), None)

# Order the device list by the chosen column; this is the order BOTH the table and the ← / →
# navigation follow. Missing values always sort to the END (regardless of direction); ties are
# broken by name (ascending) for a stable order.
filtered.sort(key=lambda d: (
    _sort_value(d) is None,
    -(_sort_value(d) or 0.0) if sort_desc else (_sort_value(d) or 0.0),
    d.name,
))


# ── Status ──────────────────────────────────────────────────────────────
# No main-area title (saves vertical space); the root folder is already shown in
# the sidebar. The "Showing X of Y devices" status is rendered into the sidebar
# slot above the filters.
notes = []
if fav_only:
    notes.append("favorites only")
if hidden_only:
    notes.append("hidden only")
suffix = "  ·  " + ", ".join(notes) if notes else ""
status_slot.caption(
    f"Showing **{len(filtered)}** of **{len(records)}** devices{suffix}"
)

if not filtered:
    st.warning("No devices match the current filter.")
    st.stop()


# ── Device list (selectable dataframe) ─────────────────────────────────
def df_for_devices(records: list[Device]) -> pd.DataFrame:
    # Every summary.txt metric becomes its own column (value only) so the table can be
    # sorted by it via header-click. Union the metric names across these devices in
    # first-seen order; a device missing a metric gets NaN ("n/a"). Namespace any name
    # that collides with a fixed/parameter column (e.g. the achieved 'well' metric vs
    # the 'well' folder-parameter) so neither is clobbered.
    fixed_cols = {"⭐", "ID", "Device", "QS error (%)", "Max rel err (%)"} | set(keys)
    metric_col: dict[str, str] = {}   # summary metric name -> (collision-free) column header
    for r in records:
        for m in r.metrics:
            if m.name not in metric_col:
                metric_col[m.name] = (m.name if m.name not in fixed_cols
                                      else f"{m.name} (summary)")
    rows = []
    for r in records:
        mvals = {m.name: m.value for m in r.metrics}
        is_fav = device_key(r, root_path) in favorites
        rows.append({
            "⭐":              "★" if is_fav else "",
            "ID":             device_id(r.name),
            "Device":         r.name,
            "QS error (%)":   r.nonqs_pct,
            "Max rel err (%)": (r.max_rel_error * 100.0
                                if r.max_rel_error is not None else None),
            **{k: r.params.get(k, "") for k in keys},
            **{col: mvals.get(name) for name, col in metric_col.items()},
        })
    return pd.DataFrame(rows, columns=(["⭐", "ID", "Device", "QS error (%)", "Max rel err (%)"]
                                       + list(keys) + list(metric_col.values())))

# ── Keyboard / button navigation + current-device resolution ───────────
# The table is shown in the navigation (sorted) order and the selection is tracked by device
# KEY (identity), not row index, so it survives filtering/sorting. It is moved by the ◀/▶
# buttons (and the ← / → arrows bound to them), which step through this same order, and by a
# row click. Because the displayed order IS the navigation order, → / ← load the adjacent row.
keys_list = [device_key(d, root_path) for d in filtered]
n_dev = len(filtered)
st.session_state["nav_keys"] = keys_list   # read by the button callbacks (run pre-script)
if st.session_state.get("sel_key_nav") not in keys_list:
    st.session_state["sel_key_nav"] = keys_list[0]

def _nav_step(delta):
    ks = st.session_state.get("nav_keys", [])
    cur = st.session_state.get("sel_key_nav")
    if cur in ks:
        st.session_state["sel_key_nav"] = ks[max(0, min(len(ks) - 1, ks.index(cur) + delta))]

def _nav_prev():
    _nav_step(-1)

def _nav_next():
    _nav_step(1)

def _table_selected_row():
    """First selected row index from the device table's PRIOR widget state, or
    None. Read before the table is (re)drawn so a click is reflected the same run;
    handles both the attribute- and dict-shaped selection state."""
    s = st.session_state.get("device_table")
    sel = getattr(s, "selection", None)
    if sel is None and isinstance(s, dict):
        sel = s.get("selection")
    rows = getattr(sel, "rows", None)
    if rows is None and isinstance(sel, dict):
        rows = sel.get("rows")
    return rows[0] if rows else None

# A FRESH click (selection row changed since last render) selects the device at that row. A
# stale, unchanged selection does NOT move us, so arrow/button navigation isn't snapped back to
# the last-clicked row. The table is shown in the SAME (sorted) order the ← / → navigation uses
# (no pin-to-top), so a click maps straight to keys_list[row] and pressing → / ← loads the very
# next / previous row. A FIXED widget key is used so Streamlit re-sends the data and the list
# updates in place each step (a per-step key would remount and not repaint reliably).
_prev_display_keys = st.session_state.get("display_keys", [])
_clicked = _table_selected_row()
if (_clicked is not None and _clicked != st.session_state.get("last_clicked")
        and 0 <= _clicked < len(_prev_display_keys)):
    ck = _prev_display_keys[_clicked]
    if ck in keys_list:
        st.session_state["sel_key_nav"] = ck
st.session_state["last_clicked"] = _clicked

# (The device-ID search is handled before filtering — it reveals + selects the
# match, so sel_key_nav already points at it and the device is in keys_list.)

# Resolve the plotted device. The table is drawn in the navigation (sorted) order, with the
# selected row highlighted in place (see _highlight_plotted) -- NOT pinned to the top -- so the
# displayed row order matches the ← / → order exactly.
sel_idx = keys_list.index(st.session_state["sel_key_nav"])
selected_device = filtered[sel_idx]
df = df_for_devices(filtered)
st.session_state["display_keys"] = list(keys_list)

# Currently-plotted device name, spanning the full width (both columns) in a small
# font (replaces the old per-column subheader).
st.markdown(
    f"<div style='font-size:0.95rem; font-weight:600; overflow-wrap:anywhere; "
    f"margin-bottom:0.25rem;'>{html.escape(selected_device.name)}</div>",
    unsafe_allow_html=True,
)

# ◀ Prev / Next ▶ navigation across the TOP, spanning the full width (both columns).
_prev_col, _next_col = st.columns(2)
_prev_col.button("◀ Prev", key="dev_prev", on_click=_nav_prev,
                 use_container_width=True, help="Previous device (← arrow)")
_next_col.button("Next ▶", key="dev_next", on_click=_nav_next,
                 use_container_width=True, help="Next device (→ arrow)")

# Shared height for the two top panes (metrics table on the left, images on the
# right). Sized so the device name + Prev/Next + the panes fit on screen without
# scrolling to reach the Devices row below; lower/raise it to suit your display.
PANE_HEIGHT = 440
left, right = st.columns([0.42, 0.58], gap="large")

with left:
    # Device metrics (parsed from summary.txt). The metrics table is given the SAME
    # fixed height as the images on the right (PANE_HEIGHT) so the two panes line up.
    if selected_device.summary_path is None:
        st.info("No `summary.txt` in this folder.")
    elif not selected_device.metrics:
        st.info("`summary.txt` could not be parsed.")
    else:
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
            height=PANE_HEIGHT,
        )


# ── Right column: images ───────────────────────────────────────────────
with right:
    # Favourite the currently selected device. A button (not a toggle) so the
    # Ctrl-S shortcut can drive it without the toggle/value desync a keyed
    # st.toggle suffers when its set is changed from outside the widget. The
    # on_click callback flips the file-backed favorites set; Streamlit reruns
    # automatically afterwards (no explicit st.rerun needed).
    sel_key = device_key(selected_device, root_path)
    is_fav = sel_key in favorites

    def _toggle_fav(k):
        favs = st.session_state[fav_state_key]
        if k in favs:
            favs.discard(k)
        else:
            favs.add(k)
        save_favorites(root_path, favs)

    st.button(
        "★ Favourited  (Ctrl-S)" if is_fav else "☆ Favourite  (Ctrl-S)",
        key="fav_btn",
        on_click=_toggle_fav,
        args=(sel_key,),
        help=f"Toggle favourite (Ctrl-S) · saved to `{root_path / FAVORITES_FILENAME}`",
    )
    # Red highlight for the favourited button (targets it via its st-key-<key>
    # class so no other button is affected). The <style> element is ALWAYS
    # rendered — empty when not favourited — so it occupies the same slot in
    # Streamlit's vertical layout either way and the spacing below the button stays
    # consistent; only the CSS rules inside differ.
    _fav_css = (
        ".st-key-fav_btn button{background-color:#d62728!important;"
        "border-color:#d62728!important;color:#ffffff!important;}"
        ".st-key-fav_btn button:hover{background-color:#b01f1f!important;"
        "border-color:#b01f1f!important;color:#ffffff!important;}"
    ) if is_fav else ""
    st.markdown(f"<style>{_fav_css}</style>", unsafe_allow_html=True)

    # Hide / unhide the current device (toggle). Ctrl-D drives this button (see the
    # keydown JS); pressing it again on a hidden device unhides it. A hidden device
    # is normally filtered out, so to unhide you tick "🚫 Hidden only" in the sidebar
    # to view it, then Ctrl-D.
    is_hidden = sel_key in hidden

    def _toggle_hidden(k):
        h = st.session_state[hidden_state_key]
        if k in h:
            h.discard(k)
        else:
            h.add(k)
        save_hidden(root_path, h)

    st.button(
        "🚫 Unhide  (Ctrl-D)" if is_hidden else "🚫 Hide  (Ctrl-D)",
        key="hide_btn",
        on_click=_toggle_hidden,
        args=(sel_key,),
        help=f"Hide/unhide this device (Ctrl-D) · stored in `{root_path / HIDDEN_FILENAME}`",
    )

    # Device metadata as a 2-row × 2-column table (bigger font) above the images.
    # st.table/st.dataframe have a small fixed font, so render an HTML table to
    # control the size. Cell text is escaped (e.g. the '>' in "trace(M) > 2").
    def _meta_cell(label, val, legend):
        if val is None:
            return f"{label}: —"
        desc = legend.get(val)
        return f"{label}={val} · {desc}" if desc else f"{label}={val}"

    _meta = [
        f"ID: {device_id(selected_device.name)}",
        _meta_cell("mono", selected_device.params.get("mono"), MONO_LEGEND),
        _meta_cell("configID", selected_device.params.get("configID"), CONFIG_LEGEND),
        _meta_cell("vesselID", selected_device.params.get("vesselID"), VESSEL_LEGEND),
    ]
    _cell = ("padding:7px 12px; border:1px solid rgba(128,128,128,0.35); "
             "font-size:1.2rem;")
    _td = lambda s: f'<td style="{_cell}">{html.escape(str(s))}</td>'
    st.markdown(
        '<table style="width:100%; border-collapse:collapse; margin-bottom:0.6rem;">'
        f"<tr>{_td(_meta[0])}{_td(_meta[1])}</tr>"
        f"<tr>{_td(_meta[2])}{_td(_meta[3])}</tr>"
        "</table>",
        unsafe_allow_html=True,
    )

    # Images in a fixed-height container so this pane matches the list height
    # (it scrolls if the images are taller). xs taller on the left, scene_top /
    # scene_left stacked on the right.
    with st.container(height=PANE_HEIGHT, border=False):
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


# ── Device list (full window width, bottom row) ─────────────────────────
st.subheader("Devices")

# st.dataframe row selection (Streamlit ≥ 1.35) is the click INPUT; the click is read back
# above before this is drawn. The table is in navigation (sorted) order; the plotted device's
# row (sel_idx) is highlighted IN PLACE via the Styler, applied AFTER the gradients so the
# highlight wins on that row.
def _highlight_plotted(row):
    if row.name == sel_idx:
        return ["background-color: rgba(31, 119, 180, 0.45); font-weight: 600"] * len(row)
    return [""] * len(row)

# Compact numeric formatting for QS error, max rel err, AND every summary.txt metric
# column (everything that is not the ⭐/ID/Device cells or a folder-parameter column).
# Header-click sorting in st.dataframe acts on the underlying numeric values regardless.
_num_cols = [c for c in df.columns if c not in ({"⭐", "ID", "Device"} | set(keys))]
_fmt = {c: "{:.4g}" for c in _num_cols}

st.dataframe(
    df.style.format(_fmt, na_rep="n/a").background_gradient(
        subset=["QS error (%)"], cmap="RdYlGn_r", vmin=0, vmax=30,
    ).background_gradient(
        subset=["Max rel err (%)"], cmap="RdYlGn_r", vmin=0, vmax=20,
    ).apply(_highlight_plotted, axis=1),
    use_container_width=True,
    height=PANE_HEIGHT,
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

st.caption(f"Device {sel_idx + 1} of {n_dev}  ·  use ← / → to navigate")

# Bind Left/Right arrow keys to the ◀/▶ buttons. This component runs in an iframe;
# it reaches into the parent Streamlit document and, on an arrow key (unless the
# user is typing in a field or a dropdown is focused), clicks the matching button —
# which fires its on_click callback and reruns. Bound once per document (guard
# flag) so reruns don't stack listeners; buttons are looked up at click time so
# they always resolve to the freshly-rendered nodes.
components.html(
    """
    <script>
    const doc = window.parent.document;
    // Re-bind every run (remove the previously-stored handler, add a fresh one) instead of a
    // persistent guard: a one-shot guard survives reruns AND code edits on the already-open
    // page, leaving a stale listener bound. CAPTURE phase + stopPropagation so the keys win
    // even when the data-grid (after a row click) or a button has focus.
    if (doc._devKeyNav) {
      doc.removeEventListener('keydown', doc._devKeyNav, true);
      doc.removeEventListener('keydown', doc._devKeyNav, false);
    }
    doc._devKeyNav = function (e) {
      const t = e.target || {};
      const tag = (t.tagName || '').toLowerCase();
      // Ctrl-S / Cmd-S: (un)favourite the currently plotted device. Handled before the input
      // guard (and preventDefault'd) so it works anywhere and suppresses the Save dialog.
      if ((e.ctrlKey || e.metaKey) && (e.key === 's' || e.key === 'S')) {
        const fb = Array.from(doc.querySelectorAll('button'))
                        .find(b => (b.innerText || '').includes('Favourite'));
        if (fb) { e.preventDefault(); e.stopPropagation(); fb.click(); }
        return;
      }
      // Ctrl-D / Cmd-D: hide (or unhide) the current device. preventDefault'd to suppress the
      // browser's bookmark dialog. Matched by the "Ctrl-D" label.
      if ((e.ctrlKey || e.metaKey) && (e.key === 'd' || e.key === 'D')) {
        const hb = Array.from(doc.querySelectorAll('button'))
                        .find(b => (b.innerText || '').includes('Ctrl-D'));
        if (hb) { e.preventDefault(); e.stopPropagation(); hb.click(); }
        return;
      }
      if (tag === 'input' || tag === 'textarea' || t.isContentEditable) return;
      if (t.closest && t.closest('[data-baseweb="select"],[data-baseweb="popover"],[role="listbox"],[role="combobox"]')) return;
      let mark = null;
      if (e.key === 'ArrowLeft')       mark = '◀';  /* ◀ */
      else if (e.key === 'ArrowRight') mark = '▶';  /* ▶ */
      if (!mark) return;
      const btn = Array.from(doc.querySelectorAll('button'))
                       .find(b => (b.innerText || '').includes(mark));
      if (btn) { e.preventDefault(); e.stopPropagation(); btn.click(); }
    };
    doc.addEventListener('keydown', doc._devKeyNav, true);
    </script>
    """,
    height=0,
)
