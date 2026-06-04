#!/bin/bash
#SBATCH --job-name=singopt_test
#SBATCH --partition=ccm
#SBATCH --constraint=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/singopt_test.%j.out
#SBATCH --error=logs/singopt_test.%j.out

# Single test run of boozer_singular_opt.py on a genoa node.
#
# Loads design_opt_final.json (and, automatically, its sibling
# design_opt_final.yaml for the weights) from the boozer_all.py output folder
# below, runs the singular polish + coil optimization, and writes
# design_opt_final.json (combined modular+aux coil set) into $OUT_DIR.
#
#   sbatch runjobs.sh          # or:  bash runjobs.sh   (interactive test)

set -uo pipefail

# ── the boozer_all.py run to load ──────────────────────────────────────────
# folder: margin=0p12  well=0.0  Z=0  onvessel=1  distance=0  configID=1
#         vesselID=0   mono=1 (-> identity)  null=SN
NAME="margin=0p12_well=0.0_Z=0_onvessel=1_distance=0_configID=1_vesselID=0_mono=1_null=SN"
INPUT="./output/${NAME}/design_opt_final.json"

# ── singular-polish settings ───────────────────────────────────────────────
NUM_AUX=5                 # planar circular aux coils (identity needs >= 3)
CONSTRAINT="identity"     # mono=1 -> identity, mono=2 -> trace
OUT_DIR="./output/${NAME}_singopt_numaux=${NUM_AUX}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs "$OUT_DIR"

if [ ! -f "$INPUT" ]; then
  echo "ERROR: $INPUT not found"
  exit 1
fi
if [ ! -f "${INPUT%.json}.yaml" ]; then
  echo "ERROR: ${INPUT%.json}.yaml not found (boozer_singular_opt.py reads the weights from it)"
  exit 1
fi

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
echo "Input: $INPUT"
echo "Output dir: $OUT_DIR"

# args decoded from the folder name above
./boozer_singular_opt.py \
  --input "$INPUT" \
  --num-aux "$NUM_AUX" \
  --constraint "$CONSTRAINT" \
  --margin 0.12 \
  --well 0.0 \
  --Z 0 \
  --distance 0 \
  --on-vessel 1 \
  --config 1 \
  --outdir "$OUT_DIR"
status=$?

deactivate
echo "Finished: $(date)  (exit $status)"
exit "$status"
