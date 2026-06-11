#!/bin/bash
set -uo pipefail

# RESTART master driver. Like prefix.sh, but it tries to REUSE the num_aux=0 device a
# previous prefix.sh run already wrote to ceph instead of recomputing it. Two checks
# decide how much of the init phase is skipped:
#
#   * design_opt_final.{json,yaml} present on ceph?  -> copy them into scratch and
#     SKIP boozer_all.py.  Absent -> FALL BACK to prefix.sh's init: recompute the
#     device with run_boozer_all.sh.
#   * the init device's *.png renders present on ceph?  -> the device is already
#     traced + rendered, so SKIP its poincare+render.  Absent (or the device was just
#     recomputed) -> run run_render.sh (poincare + render) for it and copy it to ceph.
#
# Either way, the polish (run_polish.sh) + the polished poincare+render (run_render.sh)
# are then run. So the restart is self-healing: a missing init device, or a present
# device that was never rendered, is regenerated and pushed to ceph for next time.
# Each lower-level step loads its own venv; the master does all ceph copying /
# keep-discard decisions. Use this to re-run a rewritten boozer_singular_opt.py
# without recomputing the (expensive) num_aux=0 device when it already exists.
#
#   bash ./prefix_restart.sh <margin> <well> <Z> <distance> <on_vessel> <config> \
#                            <vessel_id> <mono> <attempt> <null(DN|SN)> [num_aux]
#
# mono=0 has no polish step, so the restart aborts immediately. A polished device that
# fails or exceeds 0.1% is discarded (nothing to ceph but the log).

margin="$1"
well="$2"
Z="$3"
distance="$4"
on_vessel="$5"
config="$6"
vessel_id="$7"
mono="$8"
attempt="$9"
null="${10}"

NUM_AUX_POLISH="${11:-10}"
# Aspect-ratio knob forwarded to boozer_all.py (--AR); part of the device identity,
# so it is in the folder name (task_name) too. Must match prefix.sh.
AR="${12:-0}"

if [ "$well" = "OFF" ]; then
  well_str="OFF"
else
  well_str=$(python -c 'import sys; print(float(sys.argv[1]))' "$well")
fi
margin_str=$(printf "%.2f" "$margin" | sed 's/\./p/')

HOME_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

task_name() {
  echo "margin=${margin_str}_well=${well_str}_Z=${Z}_onvessel=${on_vessel}_distance=${distance}_configID=${config}_vesselID=${vessel_id}_mono=${mono}_null=${null}_num_aux=${1}_AR=${AR}_attempt=${attempt}"
}

# Shard helper -- identical to prefix.sh so the reused device is read from exactly
# where prefix.sh wrote it: output/<shard>/<device>.
shard() { printf '%s' "$1" | md5sum | cut -c1-2; }

INIT_NAME="$(task_name 0)"

SCRATCH="${TMPDIR:-/tmp/$USER}/disbatch_${SLURM_JOB_ID:-local}_${INIT_NAME}"
RUN="$SCRATCH/run"
LOG="$SCRATCH/log.out"

mkdir -p "$RUN" "$HOME_DIR/output"  "$HOME_DIR/logs"

exec > >(tee "$LOG") 2>&1

SYNC_INCLUDES=(
  --include='*/'
  --include='design_opt_final_*.json'
  --include='design_opt_final_*.yaml'
  --include='design_polished_final_*.json'
  --include='design_polished_final_*.yaml'
  --include='singular.json'
  --include='singular.yaml'
  --include='summary.txt'
  --include='max_rel_error.txt'
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
  --include='fixed_points_*.txt'
  --include='sc*.vts'
  --include='aux_coils_*.vtu'
  --include='surf_opt_*_final.vts'
  --include='curves_opt_final.vtu'
  --include='ma_opt_final.vtu'
  --include='xpoint_curves_opt_final.vtu'
  --include='xpoint_singular_curves_opt_final.vtu'
  --include='vessel_opt_final.vtr'
  --include='*xpoint_deletion*'
  --exclude='*'
)

sync_dir() {
  local d="${1%/}" name dest
  [ -d "$d" ] || return 0
  name="$(basename "$d")"
  dest="$HOME_DIR/output/$(shard "$name")"
  mkdir -p "$dest"
  rsync -a "${SYNC_INCLUDES[@]}" "$d/" "$dest/$name/"
}

sync_log() {
  [ -f "$LOG" ] || return 0
  local dest="$HOME_DIR/logs/$(shard "$INIT_NAME")"
  mkdir -p "$dest"
  rsync -a "$LOG" "$dest/${INIT_NAME}.out" || true
}

cleanup() {
  status=$?
  sync_log
  rm -rf "$SCRATCH"
  exit "$status"
}
trap cleanup EXIT

# A mono=0 device has no monodromy polish, so there is nothing for the restart to
# do. Fail fast; the EXIT trap cleans up scratch and preserves the log.
if [ "$mono" -eq 0 ]; then
  echo "ERROR: mono=0 has no polish step — nothing to restart from boozer_singular_opt.py; aborting task"
  exit 1
fi

rsync -a --exclude 'output*' --exclude 'logs*' --exclude '*_disBatch_*' "$HOME_DIR/" "$RUN/"
mkdir -p "$SCRATCH/convert"
rsync -a "$HOME_DIR/../convert/" "$SCRATCH/convert/"

cd "$RUN"

echo "Host: $(hostname)"
echo "Scratch: $SCRATCH"
echo "Started: $(date)"
echo "Init task: $INIT_NAME"
echo "mono=$mono attempt=$attempt (RESTART from boozer_singular_opt.py)"

INIT_DIR="./output/$INIT_NAME"

# ──── (1) num_aux=0 device: REUSE from ceph if present, else RECOMPUTE, then gate ─
# If a previous prefix.sh run left design_opt_final_<ID>.{json,yaml} in the persistent
# ceph shard dir, copy them into local scratch (where run_polish.sh reads the json,
# located by glob since the device ID is in the name) and skip boozer_all.py.
# Otherwise fall back to prefix.sh's init path and recompute the device.
SRC_INIT_DIR="$HOME_DIR/output/$(shard "$INIT_NAME")/$INIT_NAME"
SRC_INIT_JSON="$(ls "$SRC_INIT_DIR"/design_opt_final_*.json 2>/dev/null | head -1)"
if [ -n "$SRC_INIT_JSON" ] && [ -f "${SRC_INIT_JSON%.json}.yaml" ]; then
  reused_init=1
  mkdir -p "$INIT_DIR"
  cp "$SRC_INIT_JSON" "$INIT_DIR/$(basename "$SRC_INIT_JSON")"
  cp "${SRC_INIT_JSON%.json}.yaml" "$INIT_DIR/$(basename "${SRC_INIT_JSON%.json}.yaml")"
  [ -f "$SRC_INIT_DIR/max_rel_error.txt" ] && cp "$SRC_INIT_DIR/max_rel_error.txt" "$INIT_DIR/max_rel_error.txt"
  INIT_JSON="$INIT_DIR/$(basename "$SRC_INIT_JSON")"
  echo "restart: reused $(basename "$SRC_INIT_JSON") (+yaml,+max_rel_error.txt) from $SRC_INIT_DIR (skipping boozer_all)"
else
  reused_init=0
  echo "restart: design_opt_final_*.json not on ceph in $SRC_INIT_DIR — recomputing the num_aux=0 device with boozer_all.py"
  bash run_boozer_all.sh \
    "$margin" "$well" "$Z" "$distance" "$on_vessel" \
    "$config" "$vessel_id" "$mono" "$attempt" "$null" "$AR"
  INIT_JSON="$(ls "$INIT_DIR"/design_opt_final_*.json 2>/dev/null | head -1)"
  if [ -z "$INIT_JSON" ] || [ ! -f "$INIT_JSON" ]; then
    echo "ERROR: design_opt_final_*.json not produced in $INIT_DIR — boozer_all.py failed"
    exit 1
  fi
fi

# Same max-relative-error gate as prefix.sh, applied whether the init device was
# reused or freshly recomputed; >0.1% (or a missing max_rel_error.txt) aborts.
maxerr_file="$INIT_DIR/max_rel_error.txt"
if [ ! -f "$maxerr_file" ]; then
  echo "ERROR: $maxerr_file not found"
  exit 1
fi
if awk '{ exit !($1 > 0.001) }' "$maxerr_file"; then
  echo "ERROR: max relative error = $(cat "$maxerr_file") > 0.1%, aborting"
  exit 1
fi
echo "max relative error = $(cat "$maxerr_file") (<= 0.1%, OK)"

# ──── (2) ensure the init device is traced + rendered, then copy it to ceph ───────
# Run the init poincare+render (run_render.sh) when EITHER the device was just
# recomputed (fresh, never rendered) OR it was reused but has no *.png renders on ceph
# (its tracing never ran / never finished). If a reused device already has its *.png
# renders, it is left untouched on ceph. Any (re)render is copied back to ceph with
# sync_dir so a later restart finds a complete init device.
if [ "$reused_init" -eq 1 ] && ls "$SRC_INIT_DIR"/*.png >/dev/null 2>&1; then
  echo "init device on ceph already has *.png renders — skipping init poincare+render"
else
  if [ "$reused_init" -eq 1 ]; then
    echo "=== reused init device has no *.png renders — running poincare+render for $INIT_DIR ==="
  else
    echo "=== rendering recomputed init device $INIT_DIR ==="
  fi
  if ! bash run_render.sh "$INIT_JSON" "$INIT_DIR"; then
    echo "ERROR: init poincare+render failed"
    exit 1
  fi
  sync_dir "$INIT_DIR"
fi

# ─────────── (3) polish (mono 1/2) + poincare+render/copy ONLY the polished ───
if [ "$mono" -eq 1 ]; then
  MONO_CONSTRAINT="identity"
else
  MONO_CONSTRAINT="trace"   # mono==2 (mono==0 already aborted above)
fi

num_aux="$NUM_AUX_POLISH"
POLISH_NAME="$(task_name "$num_aux")"
POLISH_DIR="./output/$POLISH_NAME"
mkdir -p "$POLISH_DIR"

echo "--- polishing+optimizing num_aux=$num_aux ($MONO_CONSTRAINT) ---"
# Copy the init json + paired yaml in, preserving their (ID-suffixed) basenames;
# boozer_singular_opt.py reads the json + its sibling yaml and writes
# design_polished_final_<ID>.json (+ .yaml) in place.
INIT_JSON_BASE="$(basename "$INIT_JSON")"
INIT_YAML="${INIT_JSON%.json}.yaml"
cp "$INIT_JSON" "$POLISH_DIR/$INIT_JSON_BASE"
cp "$INIT_YAML" "$POLISH_DIR/$(basename "$INIT_YAML")"

bash run_polish.sh "$POLISH_DIR/$INIT_JSON_BASE" "$num_aux" || true

# The copied init design was only the polish INPUT; remove it (and its yaml) now the
# polish has finished, leaving only the polished design in the dir.
rm -f "$POLISH_DIR/$INIT_JSON_BASE" "$POLISH_DIR/$(basename "$INIT_YAML")"

POLISHED_JSON="$(ls "$POLISH_DIR"/design_polished_final_*.json 2>/dev/null | head -1)"
if [ -z "$POLISHED_JSON" ]; then
  echo "singular optimization num_aux=$num_aux failed (no design_polished_final_*.json) — discarding (nothing to ceph)"
  rm -rf "$POLISH_DIR"
else
  polish_err="$POLISH_DIR/max_rel_error.txt"
  if [ ! -f "$polish_err" ] || awk '{ exit !($1 > 0.001) }' "$polish_err"; then
    echo "polished num_aux=$num_aux out of spec (max rel err = $(cat "$polish_err" 2>/dev/null)) — discarding (nothing to ceph except log)"
    rm -rf "$POLISH_DIR"
  else
    echo "singular optimization num_aux=$num_aux succeeded (max rel err = $(cat "$polish_err") <= 0.1%)"
    echo "=== rendering polished device $POLISH_DIR ==="
    if ! bash run_render.sh "$POLISHED_JSON" "$POLISH_DIR"; then
      echo "ERROR: polished render failed"
      exit 1
    fi
    sync_dir "$POLISH_DIR"
  fi
fi

echo "Finished: $(date)"
