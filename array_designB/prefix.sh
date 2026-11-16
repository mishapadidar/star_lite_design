#!/bin/bash
set -uo pipefail

# Master driver. Orchestrates the per-device workflow by calling lower-level steps,
# each of which loads its OWN venv (so there is no env/venv leakage between phases):
#   run_boozer_all.sh -> initial num_aux=0 optimization   (boozer venv)
#   run_polish.sh     -> singular polish + coil opt        (boozer venv)
#   run_render.sh     -> trace + plot + paraview render    (matplotlib venv + paraview)
# The master sets up local scratch, applies the max-relative-error gates, decides
# what to keep, and COPIES (shards) kept results to ceph. Nothing is written to ceph
# for a failed / out-of-spec device except the run log.
#
#   bash ./prefix.sh <margin> <well> <Z> <distance> <on_vessel> <config> \
#                    <vessel_id> <mono> <null(DN|SN)> [num_aux] [AR] [attempt] [force] [mono_thr] [step] \
#                    [elongation] [trace_min]
#
# elongation (16th): upper bound on the near-axis elongation, or OFF (default); a field of the
# device name. trace_min (17th, step=double only): lower bound on tr M of the config 1 X-point;
# when given, the double is named <num_aux=N name>_trmin=<bound>_double.
#
# Flow: boozer_all -> gate. If it passes, render the num_aux=0 device and copy it to
# ceph BEFORE polishing. Then (mono 1/2) polish at num_aux=NUM_AUX_POLISH; if the
# polished device passes its own 0.1% gate, render + copy it; otherwise copy only its
# polished design, summary.txt and max_rel_error.txt to output_oos/ (so that other
# tolerances can be applied afterwards) and nothing to output/.
#
# step selects the part of the flow this task runs (default: all):
#   all     boozer_all, then the polish at num_aux, in this one task (the flow above);
#   boozer  boozer_all + gate + render + copy of the num_aux=0 device, then stop: exit 0
#           iff the device passed and was copied to ceph (designB_driver.py then submits
#           one polish task per num_aux);
#   polish  the polish at num_aux of a device already on ceph: its design_opt_final json
#           and yaml are read from $INIT_ROOT/<shard>/<num_aux=0 name>/, where INIT_ROOT
#           defaults to $HOME_DIR/output and may be relative to $HOME_DIR (e.g.
#           INIT_ROOT=archive/campaign_7463849/output to polish an archived device).
#           The task (scratch dir, log) is named after the polished device (num_aux=N).
#   double  the two-configuration optimization (boozer_double_opt.py via run_double.sh:
#           config 1 = the device itself, aux coils off, hyperbolic X-point; config 2 = a
#           snowflake with num_aux aux coils) of a device already on ceph, read as for
#           polish. Same 0.1% gate on its max_rel_error.txt: within spec -> output/, out of
#           spec -> output_oos/. Its dir, scratch dir and log are named <num_aux=N name>_double,
#           so that it runs alongside the polish task of the same device and num_aux. No
#           render (the render scripts expect a single-configuration design).

margin="$1"
well="$2"
Z="$3"
distance="$4"
on_vessel="$5"
config="$6"
vessel_id="$7"
mono="$8"
null="$9"

# Number of auxiliary planar coils for the polish (mono=1,2): devices have either
# 0 (unpolished) or exactly this many aux coils.
NUM_AUX_POLISH="${10:-10}"

# Aspect-ratio / on-axis-iota knob forwarded to boozer_all.py (--AR): 0 = leave AR as-is and
# do NOT pin the on-axis iota; 1 = leave AR as-is but pin the on-axis iota; 2 = reduce the
# plasma aspect ratio toward ~5 AND pin the on-axis iota. It is part of the device identity,
# so it also goes into the folder name (task_name) below.
AR="${11:-0}"

# Attempt index (perturbation seed).
attempt="${12}"

# Maximum coil force per unit length (N/m) forwarded to boozer_all.py (--force), or OFF for
# no force constraint. Part of the device identity.
force="${13:-OFF}"

# Monodromy threshold of the boozer_all constraint (--mono-thr). Part of the device
# identity: LAST positional arg so the prefix args mirror the folder-name tail
# (..._num_aux_AR_attempt_force_monothr).
mono_thr="${14:-0.1}"

# Part of the flow to run (all | boozer | polish | double), see the header. Last positional
# arg, after the device identity, so it changes no folder name.
STEP="${15:-all}"
case "$STEP" in
  all|boozer|polish|double) ;;
  *) echo "ERROR: step must be all, boozer, polish or double (got '$STEP')"; exit 1 ;;
esac
# Upper bound on the near-axis elongation forwarded to boozer_all.py (--elongation), or OFF
# for none. It is the last field of the folder name (task_name) but not part of the
# perturbation seed: an attempt starts from the same coils for every bound.
elong="${16:-OFF}"

# step=double only: lower bound on tr M of the config 1 X-point (boozer_double_opt.py
# --trace-min). Empty (default): the script's own 2 + mono_thr, and the task keeps the plain
# <num_aux=N name>_double name. Otherwise the task is named <num_aux=N name>_trmin=<bound>_double,
# so that the doubles of one device at several bounds run side by side.
trace_min="${17:-}"
DOUBLE_SUFFIX="_double"
[ -n "$trace_min" ] && DOUBLE_SUFFIX="_trmin=${trace_min}_double"

# polish and double start from a device already on ceph instead of running boozer_all
FROM_CEPH=0
case "$STEP" in polish|double) FROM_CEPH=1 ;; esac

if [ "$well" = "OFF" ]; then
  well_str="OFF"
else
  # python3 (system /usr/bin/python3): no venv is active yet here, and a plain `python`
  # may not exist in the submission environment (it silently gave "well=_" folders).
  well_str=$(python3 -c 'import sys; print(float(sys.argv[1]))' "$well")
  if [ -z "$well_str" ]; then
    echo "ERROR: could not format well='$well' (python3 missing?)"
    exit 1
  fi
fi
margin_str=$(printf "%.2f" "$margin" | sed 's/\./p/')

HOME_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"

# Folder-name builder; must match boozer_all.py's TASK_NAME exactly so the device
# IDs (and thus the perturbation seed) line up. Argument: num_aux.
task_name() {
  echo "margin=${margin_str}_well=${well_str}_Z=${Z}_onvessel=${on_vessel}_distance=${distance}_configID=${config}_vesselID=${vessel_id}_mono=${mono}_null=${null}_num_aux=${1}_AR=${AR}_attempt=${attempt}_force=${force}_monothr=${mono_thr}_elong=${elong}"
}

# Shard folder name -> a <=256-bucket subdir (md5 prefix) so that no directory on
# ceph (output/, logs/) ever holds more than ~500 entries: output/<shard>/<device>
# and logs/<shard>/<name>.out. Pure function of the (deterministic) folder name, so
# it matches prefix_restart.sh and readers that discover devices by content (e.g.
# device_browser's os.walk) need no change.
shard() { printf '%s' "$1" | md5sum | cut -c1-2; }

INIT_NAME="$(task_name 0)"
# A polish task is named after its polished device (num_aux=N), so that the polish tasks
# of one device at different num_aux get their own scratch dir and log; the other steps
# keep the num_aux=0 name.
if [ "$STEP" = "polish" ]; then
  TASK_LOG_NAME="$(task_name "$NUM_AUX_POLISH")"
elif [ "$STEP" = "double" ]; then
  # the two-configuration optimization at num_aux=N: its own name, next to the polish's
  TASK_LOG_NAME="$(task_name "$NUM_AUX_POLISH")${DOUBLE_SUFFIX}"
else
  TASK_LOG_NAME="$INIT_NAME"
fi

SCRATCH="${TMPDIR:-/tmp/$USER}/disbatch_${SLURM_JOB_ID:-local}_${TASK_LOG_NAME}"
RUN="$SCRATCH/run"
LOG="$SCRATCH/log.out"

mkdir -p "$RUN" "$HOME_DIR/output"  "$HOME_DIR/logs"

# Duplicate stdout/stderr to $LOG while keeping the engine pipe alive.
exec > >(tee "$LOG") 2>&1

# Filter for what gets copied back from a kept device dir.
SYNC_INCLUDES=(
  --include='*/'
  --include='design_opt_final_*.json'
  --include='design_opt_final_*.yaml'
  --include='design_polished_final_*.json'
  --include='design_polished_final_*.yaml'
  --include='design_unpolished_final_*.json'
  --include='design_unpolished_final_*.yaml'
  --include='design_double_final_*.json'
  --include='design_double_final_*.yaml'
  --include='design_double_unpolished_*.json'
  --include='design_double_unpolished_*.yaml'
  --include='LCFS_*.json'
  --include='LCFS[0-9][0-9]_*.json'
  --include='singular.json'
  --include='singular.yaml'
  --include='summary.txt'
  --include='max_rel_error.txt'
  --include='scene_*.png'
  --include='elongation_profile.txt'
  --include='elongation_profile.png'
  --include='unfolding_C*.txt'
  --include='unfolding_C_*.png'
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

# Copy ONE local device dir into its ceph shard: output/<shard>/<device>/...
# Called explicitly by the master only for devices that should be KEPT.
sync_dir() {
  local d="${1%/}" name dest
  [ -d "$d" ] || return 0
  name="$(basename "$d")"
  dest="$HOME_DIR/output/$(shard "$name")"
  mkdir -p "$dest"
  rsync -a "${SYNC_INCLUDES[@]}" "$d/" "$dest/$name/"
}

# Copy ONLY the unpolished (with-aux, pre-polish) design files of a local device
# dir to its ceph shard. The unpolished device carries the aux coils but has not
# been optimized, so it may not meet all design constraints; we keep it as a
# reference EVEN WHEN the polished device is later discarded for being out of
# spec, so it must be copied off scratch before any rm -rf of the polish dir.
sync_unpolished() {
  local d="${1%/}" name dest
  [ -d "$d" ] || return 0
  ls "$d"/design_unpolished_final_*.json >/dev/null 2>&1 || return 0
  name="$(basename "$d")"
  dest="$HOME_DIR/output/$(shard "$name")/$name"
  mkdir -p "$dest"
  rsync -a --include='design_unpolished_final_*.json' \
           --include='design_unpolished_final_*.yaml' \
           --exclude='*' "$d/" "$dest/"
}

# Copy the polished design of a device that FAILED its 0.1% gate to
# output_oos/<shard>/<device>/, with summary.txt (value, threshold and relative error of
# every constraint) and max_rel_error.txt, so that other tolerances can be applied
# afterwards. output_oos/ stays out of output/, which holds only devices within spec.
sync_oos() {
  local d="${1%/}" name dest
  [ -d "$d" ] || return 0
  name="$(basename "$d")"
  dest="$HOME_DIR/output_oos/$(shard "$name")/$name"
  mkdir -p "$dest"
  rsync -a --include='design_polished_final_*.json' \
           --include='design_polished_final_*.yaml' \
           --include='design_double_final_*.json' \
           --include='design_double_final_*.yaml' \
           --include='summary.txt' \
           --include='max_rel_error.txt' \
           --exclude='*' "$d/" "$dest/"
}

# Always preserved so the run is traceable, regardless of success/failure.
sync_log() {
  [ -f "$LOG" ] || return 0
  local dest="$HOME_DIR/logs/$(shard "$TASK_LOG_NAME")"
  mkdir -p "$dest"
  rsync -a "$LOG" "$dest/${TASK_LOG_NAME}.out" || true
}

cleanup() {
  status=$?
  # Device dirs are copied to ceph EXPLICITLY (sync_dir) at the points where they
  # are decided to be kept, so cleanup only preserves the log and wipes scratch.
  sync_log
  rm -rf "$SCRATCH"
  exit "$status"
}
trap cleanup EXIT

# Thinning a campaign while it runs: $HOME_DIR/skip_tasks.txt, when present, holds one
# extended regular expression per line; a task whose name matches one of them stops here
# with status 1 (not accepted, so the driver submits no later step for it).
if [ -s "$HOME_DIR/skip_tasks.txt" ] && printf '%s\n' "$TASK_LOG_NAME" | grep -q -E -f "$HOME_DIR/skip_tasks.txt"; then
  echo "SKIPPED: the task name matches a pattern of skip_tasks.txt"
  echo "Finished: $(date)"
  exit 1
fi

# archive*: past campaigns (outputs, logs) kept in array_designB/archive/, never staged
rsync -a --exclude 'output*' --exclude 'logs*' --exclude '*_disBatch_*' --exclude 'archive*' "$HOME_DIR/" "$RUN/"
mkdir -p "$SCRATCH/convert"
rsync -a "$HOME_DIR/../convert/" "$SCRATCH/convert/"

cd "$RUN"

echo "Host: $(hostname)"
echo "Scratch: $SCRATCH"
echo "Started: $(date)"
echo "Init task: $INIT_NAME"
echo "Step: $STEP  task: $TASK_LOG_NAME"
echo "mono=$mono attempt=$attempt"

INIT_DIR="./output/$INIT_NAME"

# step=polish or double: the num_aux=0 device was accepted and copied to ceph by its boozer
# task; take its design json (+ paired yaml) from there instead of running steps (1)-(2).
if [ "$FROM_CEPH" -eq 1 ]; then
  INIT_ROOT="${INIT_ROOT:-$HOME_DIR/output}"
  case "$INIT_ROOT" in /*) ;; *) INIT_ROOT="$HOME_DIR/$INIT_ROOT" ;; esac  # we are in $RUN now
  INIT_SRC="$INIT_ROOT/$(shard "$INIT_NAME")/$INIT_NAME"
  INIT_JSON="$(ls "$INIT_SRC"/design_opt_final_*.json 2>/dev/null | head -1)"
  if [ -z "$INIT_JSON" ] || [ ! -f "${INIT_JSON%.json}.yaml" ]; then
    echo "ERROR: no design_opt_final_*.json + .yaml in $INIT_SRC (device not accepted?)"
    exit 1
  fi
  echo "$STEP step: init device $INIT_JSON"
fi

# Steps (1)-(2), the boozer part, run for step=all and step=boozer. Not indented, so that
# they stay identical to the original flow.
if [ "$FROM_CEPH" -eq 0 ]; then
# ───────────────────── (1) initial optimization -> num_aux=0 ──────────────────
bash run_boozer_all.sh \
  "$margin" "$well" "$Z" "$distance" "$on_vessel" \
  "$config" "$vessel_id" "$mono" "$attempt" "$null" "$AR" "$force" "$mono_thr" "$elong"

# boozer_all.py writes design_opt_final_<DEVICE_ID>.json (the device ID is in the
# name); locate it by glob (one per device dir).
INIT_JSON="$(ls "$INIT_DIR"/design_opt_final_*.json 2>/dev/null | head -1)"
if [ -z "$INIT_JSON" ] || [ ! -f "$INIT_JSON" ]; then
  echo "ERROR: design_opt_final_*.json not produced in $INIT_DIR — boozer_all.py failed"
  exit 1
fi

# Max-relative-error gate on the initial device: boozer_all.py writes the largest
# relative constraint error (a fraction) to max_rel_error.txt. If it exceeds 0.1%
# the device did not meet its constraints -> abort (nothing to ceph but the log).
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

# ─────────── (2) boozer_all passed: render the num_aux=0 device, copy to ceph ──
# Render BEFORE polishing so the init device is secured on ceph independently of
# whatever happens to the polished device.
# Last closed flux surface for the init device (best-effort, non-fatal). Runs BEFORE the
# render so mk_manifolds can slice it into LCFS_cross_*.txt for the xs figure; writes
# LCFS_<ID>.json and appends LCFS_aspect_ratio to summary.txt.
bash run_LCFS.sh "$INIT_JSON" || echo "WARNING: LCFS computation failed for init device"
echo "=== rendering init device $INIT_DIR ==="
if ! bash run_render.sh "$INIT_JSON" "$INIT_DIR"; then
  echo "ERROR: init render failed"
  exit 1
fi
# Append min/max/mean axis elongation to summary.txt (best-effort, non-fatal).
bash run_elongation.sh "$INIT_JSON" || echo "WARNING: elongation computation failed for init device"
sync_dir "$INIT_DIR"
fi  # end of steps (1)-(2), the boozer part

# step=boozer stops here: the device passed its gate and is on ceph (exit 0), and
# designB_driver.py submits its polish tasks.
if [ "$STEP" = "boozer" ]; then
  echo "Finished: $(date)"
  exit 0
fi

# ───────────────────────────── (3) polish (mono 1/2) ──────────────────────────
if [ "$mono" -eq 1 ]; then
  MONO_CONSTRAINT="identity"
elif [ "$mono" -eq 2 ]; then
  MONO_CONSTRAINT="trace"
else
  MONO_CONSTRAINT=""
fi

if [ -n "$MONO_CONSTRAINT" ] && [ "$STEP" != "double" ]; then
  num_aux="$NUM_AUX_POLISH"
  POLISH_NAME="$(task_name "$num_aux")"
  POLISH_DIR="./output/$POLISH_NAME"
  mkdir -p "$POLISH_DIR"

  echo "--- polishing+optimizing num_aux=$num_aux ($MONO_CONSTRAINT) ---"
  # boozer_singular_opt.py reads the input json + its sibling yaml and writes
  # design_polished_final_<DEVICE_ID>.json (+ .yaml) IN PLACE in the polish dir.
  # Copy the init json + paired yaml in, preserving their (ID-suffixed) basenames.
  INIT_JSON_BASE="$(basename "$INIT_JSON")"
  INIT_YAML="${INIT_JSON%.json}.yaml"
  cp "$INIT_JSON" "$POLISH_DIR/$INIT_JSON_BASE"
  cp "$INIT_YAML" "$POLISH_DIR/$(basename "$INIT_YAML")"

  # snowflake-discriminant constraint (Delta > DISCRIMINANT_THRESHOLD) for the identity
  # (M = I) polish only
  DISC=0; [ "$MONO_CONSTRAINT" = "identity" ] && DISC=1
  bash run_polish.sh "$POLISH_DIR/$INIT_JSON_BASE" "$num_aux" "$DISC" || true

  # The copied init design was only the polish INPUT; remove it (and its yaml) now
  # the polish has finished, leaving only the polished design in the dir.
  rm -f "$POLISH_DIR/$INIT_JSON_BASE" "$POLISH_DIR/$(basename "$INIT_YAML")"

  # Preserve the unpolished (with-aux) device on ceph BEFORE the gate, so it
  # survives even if the polished device is discarded below for being out of spec.
  sync_unpolished "$POLISH_DIR"

  POLISHED_JSON="$(ls "$POLISH_DIR"/design_polished_final_*.json 2>/dev/null | head -1)"
  if [ -z "$POLISHED_JSON" ]; then
    echo "singular optimization num_aux=$num_aux failed (no design_polished_final_*.json) — discarding (nothing to ceph)"
    rm -rf "$POLISH_DIR"
  else
    # Same gate as boozer_all: a polished device that exceeds 0.1% is discarded
    # entirely (nothing to ceph but the log) -- symmetric with the init device.
    polish_err="$POLISH_DIR/max_rel_error.txt"
    if [ ! -f "$polish_err" ] || awk '{ exit !($1 > 0.001) }' "$polish_err"; then
      echo "polished num_aux=$num_aux out of spec (max rel err = $(cat "$polish_err" 2>/dev/null)): polished design + summary.txt kept in output_oos/, not in output/"
      sync_oos "$POLISH_DIR"
      rm -rf "$POLISH_DIR"
    else
      echo "singular optimization num_aux=$num_aux succeeded (max rel err = $(cat "$polish_err") <= 0.1%)"
      # LCFS for the polished device (best-effort, non-fatal). Runs BEFORE the render so
      # mk_manifolds can slice it into LCFS_cross_*.txt for the xs figure; writes
      # LCFS_<ID>.json + appends LCFS_aspect_ratio to summary.txt.
      bash run_LCFS.sh "$POLISHED_JSON" || echo "WARNING: LCFS computation failed for polished device"
      echo "=== rendering polished device $POLISH_DIR ==="
      if ! bash run_render.sh "$POLISHED_JSON" "$POLISH_DIR"; then
        echo "ERROR: polished render failed"
        exit 1
      fi
      # Append min/max/mean axis elongation to summary.txt (best-effort, non-fatal).
      bash run_elongation.sh "$POLISHED_JSON" || echo "WARNING: elongation computation failed for polished device"
      sync_dir "$POLISH_DIR"
    fi
  fi
fi

# ──────────── (4) two-configuration optimization (step=double, mono 1/2) ───────
if [ -n "$MONO_CONSTRAINT" ] && [ "$STEP" = "double" ]; then
  num_aux="$NUM_AUX_POLISH"
  DOUBLE_NAME="$(task_name "$num_aux")${DOUBLE_SUFFIX}"
  DOUBLE_DIR="./output/$DOUBLE_NAME"
  mkdir -p "$DOUBLE_DIR"

  echo "--- double optimization num_aux=$num_aux ($MONO_CONSTRAINT): X-point (aux off) + snowflake ---"
  # boozer_double_opt.py reads the input json + its sibling yaml and writes
  # design_double_final_<DEVICE_ID>.json (+ .yaml), summary.txt and max_rel_error.txt IN
  # PLACE in the double dir. Copy the init json + paired yaml in, as for the polish.
  INIT_JSON_BASE="$(basename "$INIT_JSON")"
  INIT_YAML="${INIT_JSON%.json}.yaml"
  cp "$INIT_JSON" "$DOUBLE_DIR/$INIT_JSON_BASE"
  cp "$INIT_YAML" "$DOUBLE_DIR/$(basename "$INIT_YAML")"

  DISC=0; [ "$MONO_CONSTRAINT" = "identity" ] && DISC=1
  # trace_min (17th argument), when given, becomes boozer_double_opt.py's --trace-min, added
  # to whatever DOUBLE_OPT_ARGS the environment already carries.
  DOUBLE_OPT_ARGS="${DOUBLE_OPT_ARGS:-}${trace_min:+ --trace-min $trace_min}" \
    bash run_double.sh "$DOUBLE_DIR/$INIT_JSON_BASE" "$num_aux" "$DISC" || true
  rm -f "$DOUBLE_DIR/$INIT_JSON_BASE" "$DOUBLE_DIR/$(basename "$INIT_YAML")"

  DOUBLE_JSON="$(ls "$DOUBLE_DIR"/design_double_final_*.json 2>/dev/null | head -1)"
  if [ -z "$DOUBLE_JSON" ]; then
    echo "double optimization num_aux=$num_aux failed (no design_double_final_*.json) — discarding (nothing to ceph)"
    rm -rf "$DOUBLE_DIR"
  else
    # Same 0.1% gate; its max_rel_error.txt is the worst constraint error over BOTH
    # configurations (including the trace bound of the X-point).
    double_err="$DOUBLE_DIR/max_rel_error.txt"
    if [ ! -f "$double_err" ] || awk '{ exit !($1 > 0.001) }' "$double_err"; then
      echo "double num_aux=$num_aux out of spec (max rel err = $(cat "$double_err" 2>/dev/null)): design + summary.txt kept in output_oos/, not in output/"
      sync_oos "$DOUBLE_DIR"
      rm -rf "$DOUBLE_DIR"
    else
      echo "double optimization num_aux=$num_aux succeeded (max rel err = $(cat "$double_err") <= 0.1%)"
      # Render BOTH configurations: boozer_double_opt.py wrote one single-configuration json
      # per configuration in config1/ (X-point, aux off) and config2/ (snowflake), and the
      # render scripts write next to their input. A failed render is a warning only: the
      # device is still copied to ceph.
      for cfg in config1 config2; do
        CFG_JSON="$(ls "$DOUBLE_DIR/$cfg"/design_double_final_*.json 2>/dev/null | head -1)"
        echo "=== rendering double device, $cfg: $DOUBLE_DIR/$cfg ==="
        if [ -z "$CFG_JSON" ] || ! bash run_render.sh "$CFG_JSON" "$DOUBLE_DIR/$cfg" noviews; then
          echo "WARNING: render of $cfg failed"
        fi
        # Append min/max/mean axis elongation to this configuration's summary.txt (best-effort, non-fatal).
        [ -z "$CFG_JSON" ] || bash run_elongation.sh "$CFG_JSON" || echo "WARNING: elongation computation failed for $cfg"
      done
      # Views of the whole double (scene_top.png, scene_left.png): mk_paraview.py on the double
      # dir itself, with the VTK of the matplotlib venv. Best-effort, non-fatal.
      echo "=== device views of the double: $DOUBLE_DIR ==="
      ( source /mnt/home/agiuliani/ceph/STAR_LITE/venv_matplotlib/bin/activate \
        && env -u DISPLAY python mk_paraview.py "$DOUBLE_DIR" ) \
        || echo "WARNING: device views (scene_*.png) of the double failed"
      # Post-processing of the in-spec double (run_double_analysis.sh): (1) toroidal profile of
      # the near-axis elongation, (3) unfolding constant C over 128 GP coil perturbations, with
      # their figures. Best-effort, non-fatal (about 30 minutes). Step (2), the repair with the
      # aux currents, is not run here.
      echo "=== analysis of the double (elongation profile, unfolding constant C): $DOUBLE_DIR ==="
      bash run_double_analysis.sh "$DOUBLE_DIR" \
        || echo "WARNING: $? analysis step(s) of the double failed"
      sync_dir "$DOUBLE_DIR"
    fi
  fi
fi

echo "Finished: $(date)"
