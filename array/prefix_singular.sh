#!/bin/bash
# Simple driver for the UNPOLISHED (with-aux, no optimization) workflow. Unlike
# prefix.sh -- which runs boozer_all, gates, polishes (boozer_singular_opt.py) and
# renders -- this one ONLY computes the auxiliary coils for an EXISTING boozer_all
# device and renders it. No boozer_all run, no coil optimization.
#
#   bash ./prefix_singular.sh <design_opt_final.json> [num_aux]
#
# Steps:
#   1) boozer_singular.py (via run_singular.sh, BOOZER venv): build the auxiliary
#      coils for the input device with the trace/identity monodromy enforced
#      exactly, and save the device to a sibling '<folder>_unpolished' directory.
#   2) LCFS (via run_LCFS.sh, BOOZER venv): grow the last-closed-flux-surface so the
#      render can overlay it on the xs figure. Best-effort, non-fatal.
#   3) render the unpolished device (via run_render.sh, MATPLOTLIB venv + paraview):
#      trace + the xs cross-section plots (mk_manifolds.py + plot_manifolds.py) and
#      the paraview device views (mk_paraview.py).
#   4) elongation (via run_elongation.sh, BOOZER venv): append elongation_min/max/mean
#      to summary.txt. Best-effort, non-fatal.
#
# Each phase runs as its own subprocess so the boozer and matplotlib venvs never
# leak into each other. <design_opt_final.json> is a boozer_all device json; its
# sibling .yaml supplies the monodromy constraint.
set -uo pipefail

design_json="$1"
# Number of auxiliary planar coils (must match the constraint: >= 1 for trace,
# >= 3 for identity). Same default as the polish.
num_aux="${2:-10}"

# Per-device run log, sharded like prefix.sh's logs/ (first two hex chars of md5 of the
# folder name, so any one logs/ subdir stays under ~500 entries). prefix_singular.sh runs
# in place (no scratch), so tee straight to the final logs/<shard>/<name>_unpolished.out
# -- duplicating output to the log while keeping the disBatch stdout/stderr pipe alive
# (same trick as prefix.sh). Derived from the folder name in the path so logging starts
# before we touch the filesystem (the not-found error below is captured too).
shard() { printf '%s' "$1" | md5sum | cut -c1-2; }
unpolished_name="$(basename "$(dirname "$design_json")")_unpolished"
mkdir -p "logs/$(shard "$unpolished_name")"
LOG="logs/$(shard "$unpolished_name")/${unpolished_name}.out"
exec > >(tee "$LOG") 2>&1

if [ ! -f "$design_json" ]; then
  echo "ERROR: design json not found: $design_json"
  exit 1
fi

# boozer_singular.py derives its own output dir: the input device folder name with
# '_unpolished' appended, as a sibling of the input.
in_dir="$(cd "$(dirname "$design_json")" && pwd)"
unpolished_dir="${in_dir}_unpolished"

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Input device json: $design_json"
echo "Unpolished output dir: $unpolished_dir"
echo "Log: $LOG"
echo "num_aux=$num_aux"

# ── (1) compute the aux coils + save the unpolished device (BOOZER venv) ──
bash run_singular.sh "$design_json" "$num_aux"

# boozer_singular.py writes design_unpolished_final_<DEVICE_ID>.json (the device ID
# is in the name); locate it by glob (one per device dir).
UNPOLISHED_JSON="$(ls "$unpolished_dir"/design_unpolished_final_*.json 2>/dev/null | head -1)"
if [ -z "$UNPOLISHED_JSON" ] || [ ! -f "$UNPOLISHED_JSON" ]; then
  echo "ERROR: design_unpolished_final_*.json not produced in $unpolished_dir — boozer_singular.py failed"
  exit 1
fi

# ── (2) LCFS (BOOZER venv): grow the last-closed-flux-surface BEFORE the render so
# mk_manifolds.py can slice it into LCFS_cross_*.txt for the xs overlay. Best-effort,
# non-fatal (matches prefix.sh).
bash run_LCFS.sh "$UNPOLISHED_JSON" || echo "WARNING: LCFS computation failed for $unpolished_dir"

# ── (3) render: xs plots (with LCFS overlay) + device views (MATPLOTLIB venv + paraview) ──
echo "=== rendering unpolished device $unpolished_dir ==="
if ! bash run_render.sh "$UNPOLISHED_JSON" "$unpolished_dir"; then
  echo "ERROR: render failed for $unpolished_dir"
  exit 1
fi

# ── (4) elongation (BOOZER venv): append elongation_min/max/mean to summary.txt
# AFTER the render. Best-effort, non-fatal (matches prefix.sh).
bash run_elongation.sh "$UNPOLISHED_JSON" || echo "WARNING: elongation computation failed for $unpolished_dir"

echo "Finished: $(date)"
