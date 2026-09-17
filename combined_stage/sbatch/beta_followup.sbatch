#!/bin/bash
#SBATCH --job-name=cs_beta_followup
#SBATCH --partition=compute
#SBATCH --exclude=cn12,cn07
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#
# After a finished vacuum double-null run: seed and submit its two finite-beta (1 %) runs, 24 coils (3 types) and the
# run's own 32 coils, with the configs of the repaired-boundary runs (configs/hsx_nfp4_dn_{3types,trialE32}_L5_beta1.yaml).
#   cd star_lite_design/combined_stage
#   RUN=<vacuum run dir> TAG=<name> sbatch --dependency=afterok:<run job> --output=<run dir>/logs/beta_followup_%j.out \
#     sbatch/beta_followup.sbatch
# Writes seeds/coils.<TAG>_3types_24coils.json and seeds/input.<TAG>_final_beta1 and submits
# runs/combined_stage/<TAG>_{24,32}coils_L5_beta1. If the run's final boundary does not solve cold, the beta seed is
# made from the run's (cold-solvable) stage seed and the plasma walks to the final boundary (--plasma-target).
set -euo pipefail
: "${RUN:?set RUN}" "${TAG:?set TAG}"
cd "${CS_DIR:-$SLURM_SUBMIT_DIR}"
[ -f combined_stage_vmex.py ] || { echo "run from star_lite_design/combined_stage or set CS_DIR" >&2; exit 2; }
CS=$(pwd)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/data/common/raffael.wendlinger/envs/simflare/bin/python
R=/data/common/raffael.wendlinger/simflare/runs/combined_stage
FINAL=$RUN/inputs/input.combined_stage_final
COILS32=$RUN/coils/coils_final.json
XPT=$RUN/xpoints/xpoint_final.json
COILS24=$R/seeds/coils.${TAG}_3types_24coils.json
BETA=$R/seeds/input.${TAG}_final_beta1
for f in "$FINAL" "$COILS32" "$XPT"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 3; }; done
mkdir -p $R/seeds/logs

echo "=== 24-coil seed from $COILS32"
"$PY" -u tools/reduce_coil_types.py "$COILS32" 3 "$COILS24" --vmec-input "$FINAL" --xpoint "$XPT"

echo "=== beta 1 % seed from the final boundary"
TARGET=()
if "$PY" -u tools/make_beta_seed.py configs/hsx_nfp4_dn_3types_L5_beta1.yaml "$FINAL" "$BETA" 0.01 2>&1 \
    | grep -v "compilation cache\|warnings.warn" | tee $R/seeds/logs/${TAG}_beta_seed.log | grep -q "SEED READY"; then
  echo "final boundary solves cold: no walk"
else
  echo "final boundary does NOT solve cold: beta seed from the stage seed, walk to the final boundary"
  "$PY" -u tools/make_beta_seed.py configs/hsx_nfp4_dn_3types_L5_beta1.yaml "$RUN/inputs/input.combined_stage_seed" "$BETA" 0.01 2>&1 \
    | grep -v "compilation cache\|warnings.warn" | tee -a $R/seeds/logs/${TAG}_beta_seed.log | grep -q "SEED READY" \
    || { echo "stage seed failed too" >&2; exit 4; }
  TARGET=(PLASMA_TARGET="$FINAL")
fi

submit() {  # <config> <coils> <outdir name> <job name>
  local out=$R/$3
  [ -e "$out" ] && { echo "EXISTS $out, not submitted" >&2; return 1; }
  mkdir -p "$out/logs"
  env CS_DIR="$CS" CONFIG="$1" SEED="$BETA" COILS="$2" XPOINT="$XPT" OUTDIR="$out" "${TARGET[@]}" \
    sbatch --job-name="$4" --cpus-per-task=12 --mem=48G --time=7-00:00:00 --output="$out/logs/slurm_%j.out" \
    sbatch/run_stage.sbatch
}
submit configs/hsx_nfp4_dn_3types_L5_beta1.yaml "$COILS24" ${TAG}_24coils_L5_beta1 cs_b1_24c_${TAG}
submit configs/hsx_nfp4_dn_trialE32_L5_beta1.yaml "$COILS32" ${TAG}_32coils_L5_beta1 cs_b1_32c_${TAG}
echo "done $(date)"
