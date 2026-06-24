#!/bin/bash
# Simple driver for the UNPOLISHED (with-aux, no optimization) workflow. Unlike
# prefix.sh -- which runs boozer_all, gates, polishes (boozer_singular_opt.py) and
# renders -- this one ONLY computes the auxiliary coils for an EXISTING boozer_all
# device and renders it. No boozer_all run, no coil optimization.
#
#   bash ./prefix_singular.sh <design_opt_final.json> [num_aux]
#
# Like prefix.sh, all the I/O-heavy work runs on NODE-LOCAL scratch ($TMPDIR), and only
# the finished device dir (+ the log) is copied back to ceph at the end -- so the many
# small reads/writes (json load/save, tracing intermediates, VTK, PNGs) hit fast local
# disk instead of ceph. The <design_opt_final.json> source on ceph is only READ (its
# json + sibling .yaml are staged into scratch); it is NEVER written to.
#
# Steps (each its own subprocess so the boozer and matplotlib venvs never leak):
#   1) boozer_singular.py (via run_singular.sh, BOOZER venv): build the auxiliary coils
#      with the trace/identity monodromy enforced exactly, saving the device to a sibling
#      '<folder, num_aux rewritten>_unpolished' dir IN SCRATCH.
#   2) LCFS (via run_LCFS.sh, BOOZER venv): grow the last-closed-flux-surface so the
#      render can overlay it on the xs figure. Best-effort, non-fatal.
#   3) render (via run_render.sh, MATPLOTLIB venv + paraview): trace + xs cross-section
#      plots (mk_manifolds.py + plot_manifolds.py) + paraview device views (mk_paraview.py).
#   4) elongation (via run_elongation.sh, BOOZER venv): append elongation to summary.txt.
#      Best-effort, non-fatal.
# Then the finished scratch device dir is rsync'd to ceph output/<shard>/<name>/.
set -uo pipefail

design_json="$1"
# Number of auxiliary planar coils (must match the constraint: >= 1 for trace,
# >= 3 for identity). Same default as the polish.
num_aux="${2:-10}"

# HOME_DIR = the (ceph) submit dir this is launched from (array/), holding the code, the
# source devices (output/) and the kept logs (logs/). design_json is given RELATIVE to it.
HOME_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Source device dir (relative) + the unpolished device's name. The unpolished name carries
# the ACTUAL aux-coil count (num_aux token rewritten, the input is num_aux=0) + '_unpolished'
# -- byte-identical to the OUT_DIR boozer_singular.py derives (re.sub r'num_aux=\d+').
shard() { printf '%s' "$1" | md5sum | cut -c1-2; }
src_dir="$(dirname "$design_json")"
unpolished_name="$(basename "$src_dir" | sed -E "s/num_aux=[0-9]+/num_aux=${num_aux}/")_unpolished"

# Node-local scratch: do ALL the work here, then copy the finished device + log to ceph.
SCRATCH="${TMPDIR:-/tmp/$USER}/disbatch_singular_${SLURM_JOB_ID:-local}_${unpolished_name}"
RUN="$SCRATCH/run"
LOG="$SCRATCH/log.out"
mkdir -p "$RUN" "$HOME_DIR/output" "$HOME_DIR/logs"

# Tee everything to the scratch log while keeping the disBatch stdout/stderr pipe alive;
# sync_log copies it to ceph on exit (so even a failed run leaves a log behind).
exec > >(tee "$LOG") 2>&1

# Keeper filter for what gets copied back from the finished device dir (the unpolished
# subset of prefix.sh's SYNC_INCLUDES; patterns that don't apply simply match nothing).
SYNC_INCLUDES=(
  --include='*/'
  --include='design_unpolished_final_*.json'
  --include='design_unpolished_final_*.yaml'
  --include='LCFS_*.json'
  --include='LCFS[0-9][0-9]_*.json'
  --include='summary.txt'
  --include='scene_*.png'
  --include='xs_*.png'
  --include='poincare*.txt'
  --include='xpoint.txt'
  --include='phis.txt'
  --include='xpoint_type.txt'
  --include='legs.txt'
  --include='snowflake_discriminant.txt'
  --include='vessel_cross_*.txt'
  --include='surface_cross_*.txt'
  --include='LCFS_cross_*.txt'
  --include='LCFS[0-9][0-9]_cross_*.txt'
  --include='fixed_points_*.txt'
  --include='aux_coils_*.vtu'
  --include='surf_opt_*_final.vts'
  --include='curves_opt_final.vtu'
  --include='ma_opt_final.vtu'
  --include='xpoint_curves_opt_final.vtu'
  --include='xpoint_singular_curves_opt_final.vtu'
  --include='vessel_opt_final.vtr'
  --exclude='*'
)

# Copy ONE finished local device dir into its ceph shard: output/<shard>/<name>/. Sharded
# by the unpolished folder name (md5 prefix) like prefix.sh, so no output/ subdir grows
# unbounded. rsync (no --delete) only adds/updates files inside this <name>_unpolished dir;
# nothing outside it on ceph -- in particular no SOURCE device dir -- is ever touched.
sync_dir() {
  local d="${1%/}" name dest
  [ -d "$d" ] || return 0
  name="$(basename "$d")"
  dest="$HOME_DIR/output/$(shard "$name")"
  mkdir -p "$dest"
  rsync -a "${SYNC_INCLUDES[@]}" "$d/" "$dest/$name/"
}

# Always preserve the log on ceph, regardless of success/failure.
sync_log() {
  [ -f "$LOG" ] || return 0
  local dest="$HOME_DIR/logs/$(shard "$unpolished_name")"
  mkdir -p "$dest"
  rsync -a "$LOG" "$dest/${unpolished_name}.out" || true
}

cleanup() {
  status=$?
  sync_log
  rm -rf "$SCRATCH"
  exit "$status"
}
trap cleanup EXIT

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "HOME_DIR (ceph): $HOME_DIR"
echo "Scratch: $SCRATCH"
echo "Input device json: $design_json"
echo "num_aux=$num_aux"

# Validate the source ON CEPH (read-only) before staging it into scratch.
if [ ! -f "$HOME_DIR/$design_json" ]; then
  echo "ERROR: design json not found: $HOME_DIR/$design_json"
  exit 1
fi
if [ ! -f "$HOME_DIR/${design_json%.json}.yaml" ]; then
  echo "ERROR: sibling yaml not found: $HOME_DIR/${design_json%.json}.yaml"
  exit 1
fi

# Stage the CODE (everything but output*/logs*) and the source device's json + yaml into
# scratch at the same relative path, so the now-local boozer_singular.py reads them
# locally and writes its outputs locally (OUT_DIR is a sibling of the input). Only the
# json + yaml are staged -- not the source's own renders -- so the copy stays small.
rsync -a --exclude 'output*' --exclude 'logs*' --exclude '*_disBatch_*' "$HOME_DIR/" "$RUN/"
mkdir -p "$RUN/$src_dir"
cp "$HOME_DIR/$design_json" "$RUN/$design_json"
cp "$HOME_DIR/${design_json%.json}.yaml" "$RUN/${design_json%.json}.yaml"

cd "$RUN"

# Output dir boozer_singular.py will create, as a sibling of the (now-local) input dir.
# Relative path, resolved under $RUN.
unpolished_dir="$(dirname "$src_dir")/${unpolished_name}"

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
# mk_manifolds.py can slice it into LCFS_cross_*.txt for the xs overlay. Best-effort.
bash run_LCFS.sh "$UNPOLISHED_JSON" || echo "WARNING: LCFS computation failed for $unpolished_dir"

# ── (3) render: xs plots (with LCFS overlay) + device views (MATPLOTLIB venv + paraview) ──
echo "=== rendering unpolished device $unpolished_dir ==="
if ! bash run_render.sh "$UNPOLISHED_JSON" "$unpolished_dir"; then
  echo "ERROR: render failed for $unpolished_dir"
  exit 1
fi

# ── (4) elongation (BOOZER venv): append elongation_min/max/mean to summary.txt AFTER
# the render. Best-effort, non-fatal.
bash run_elongation.sh "$UNPOLISHED_JSON" || echo "WARNING: elongation computation failed for $unpolished_dir"

# ── copy the finished device from scratch onto ceph (output/<shard>/<name>_unpolished/) ──
echo "=== syncing $unpolished_name to ceph ==="
sync_dir "$unpolished_dir"

echo "Finished: $(date)"
