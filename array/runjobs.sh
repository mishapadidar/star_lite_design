#!/bin/bash
#SBATCH --job-name=singopt_test
#SBATCH --partition=ccm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=singopt_test.%j.out
#SBATCH --error=singopt_test.%j.out

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
NAME="margin=0p06_well=0.0_Z=0_onvessel=0_distance=0_configID=0_vesselID=2_mono=1_null=SN_num_aux=10_attempt=0"
# INPUT (design_opt_final_<DEVICE_ID>.json) is located by glob after the cd below.

# ── singular-polish settings ───────────────────────────────────────────────
NUM_AUX=10                 # planar circular aux coils (identity needs >= 3)
CONSTRAINT="identity"     # mono=1 -> identity, mono=2 -> trace
OUT_DIR="./output/${NAME}_singopt_numaux=${NUM_AUX}"

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs "$OUT_DIR"

INPUT="$(ls "./output/${NAME}"/design_opt_final_*.json 2>/dev/null | head -1)"
if [ -z "$INPUT" ] || [ ! -f "$INPUT" ]; then
  echo "ERROR: design_opt_final_*.json not found in ./output/${NAME}"
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
./boozer_singular_opt.py  "$INPUT" --num-aux 10
status=$?

deactivate
echo "Finished: $(date)  (exit $status)"
exit "$status"
