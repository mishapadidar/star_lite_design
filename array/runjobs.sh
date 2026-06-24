#!/bin/bash
#SBATCH --job-name=singopt_test
#SBATCH --partition=ccm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=singopt_test.%j.out
#SBATCH --error=singopt_test.%j.out

# Single test run of boozer_singular_opt.py (+ full postprocessing) on a genoa node.
#
# Copies design_opt_final_<ID>.json (+ its sibling .yaml, from which the polish reads the
# weights AND the monodromy constraint) out of the boozer_all.py device folder below into a
# SEPARATE $OUT_DIR, then runs the singular polish + coil optimization THERE -- boozer_singular_opt.py
# writes its outputs next to its input, so running on the copy leaves the SOURCE device
# folder untouched. Writes design_polished_final_<ID>.json (combined modular+aux coil set)
# into $OUT_DIR.
#
# Then it postprocesses the polished device exactly as prefix.sh does (each step its own
# run_*.sh subprocess + venv): LCFS (run_LCFS.sh -> mk_LCFS.py), render (run_render.sh ->
# mk_manifolds.py trace + Poincaré, plot_manifolds.py xs plots, mk_paraview.py device views),
# and elongation (run_elongation.sh -> mk_elongation.py). Postprocessing is best-effort: a
# failed step prints a WARNING but the job's exit status still reflects the polish.
#
#   sbatch runjobs.sh          # or:  bash runjobs.sh   (interactive test)

set -uo pipefail

# ── the boozer_all.py device to polish ──────────────────────────────────────
# folder: margin=0p10  well=0.0  Z=0  onvessel=1  distance=0  configID=1
#         vesselID=1   mono=1 (-> identity)  null=DN  AR=0  attempt=1
NAME="margin=0p10_well=0.0_Z=0_onvessel=1_distance=0_configID=1_vesselID=1_mono=1_null=DN_num_aux=0_AR=0_attempt=1"

# ── singular-polish settings ───────────────────────────────────────────────
NUM_AUX=5                  # planar circular aux coils (identity needs >= 3)
CONSTRAINT="identity"      # mono=1 -> identity, mono=2 -> trace (read from the yaml; informational here)
OUT_DIR="./output_noPFcurrentconstraint/${NAME}_singopt_numaux=${NUM_AUX}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs "$OUT_DIR"

# The device folder is sharded on ceph: output/<shard>/<NAME>/, where shard = first two hex
# chars of md5 of the folder name (matching prefix.sh).
shard() { printf '%s' "$1" | md5sum | cut -c1-2; }
SRC_DIR="./output/$(shard "$NAME")/${NAME}"

INPUT="$(ls "$SRC_DIR"/design_opt_final_*.json 2>/dev/null | head -1)"
if [ -z "$INPUT" ] || [ ! -f "$INPUT" ]; then
  echo "ERROR: design_opt_final_*.json not found in $SRC_DIR"
  exit 1
fi
if [ ! -f "${INPUT%.json}.yaml" ]; then
  echo "ERROR: ${INPUT%.json}.yaml not found (boozer_singular_opt.py reads the weights from it)"
  exit 1
fi

# Stage the polish INPUT (json + sibling yaml) into $OUT_DIR so the optimization runs and
# writes THERE, leaving the source device folder untouched (mirrors prefix.sh's polish dir).
cp "$INPUT" "$OUT_DIR/$(basename "$INPUT")"
cp "${INPUT%.json}.yaml" "$OUT_DIR/$(basename "${INPUT%.json}.yaml")"
POLISH_INPUT="$OUT_DIR/$(basename "$INPUT")"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPATH="/mnt/home/agiuliani/ceph/STAR_LITE:${PYTHONPATH:-}"

source /mnt/home/agiuliani/ceph/STAR_LITE/venv/bin/activate

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Source device: $SRC_DIR"
echo "Polish input:  $POLISH_INPUT"
echo "Output dir:    $OUT_DIR"
echo "num_aux=$NUM_AUX  constraint=$CONSTRAINT (mono=1)"

./boozer_singular_opt.py "$POLISH_INPUT" --num-aux "$NUM_AUX"
status=$?

# Done with the inline boozer venv; the postprocessing steps below each run as their own
# subprocess (run_*.sh) and source their own venv, so leave this one before calling them.
deactivate

if [ "$status" -ne 0 ]; then
  echo "polish failed (exit $status) — skipping postprocessing"
  echo "Finished: $(date)  (polish exit $status)"
  exit "$status"
fi

# boozer_singular_opt.py wrote design_polished_final_<ID>.json into $OUT_DIR (next to its
# input); locate it by glob so the postprocessing runs on the polished device.
POLISHED_JSON="$(ls "$OUT_DIR"/design_polished_final_*.json 2>/dev/null | head -1)"
if [ -z "$POLISHED_JSON" ] || [ ! -f "$POLISHED_JSON" ]; then
  echo "ERROR: design_polished_final_*.json not produced in $OUT_DIR"
  echo "Finished: $(date)  (polish exit $status, no polished json)"
  exit 1
fi

# ── postprocessing (each its own subprocess + venv, exactly as prefix.sh drives them) ──
echo "=== postprocessing $POLISHED_JSON ==="
# (1) LCFS (BOOZER venv): grow the last-closed-flux-surface BEFORE the render so
#     mk_manifolds.py can slice it into LCFS_cross_*.txt for the xs overlay.
bash run_LCFS.sh "$POLISHED_JSON" || echo "WARNING: LCFS (mk_LCFS.py) failed"
# (2) render (MATPLOTLIB venv + paraview): field-line trace + Poincaré data (mk_manifolds.py),
#     xs cross-section plots (plot_manifolds.py), and device views (mk_paraview.py).
bash run_render.sh "$POLISHED_JSON" "$OUT_DIR" || echo "WARNING: render (run_render.sh) failed"
# (3) elongation (BOOZER venv): append elongation_min/max/mean to summary.txt, AFTER the render.
bash run_elongation.sh "$POLISHED_JSON" || echo "WARNING: elongation (mk_elongation.py) failed"

echo "Finished: $(date)  (polish exit $status)"
exit "$status"
