#!/bin/bash
# Copy of array/run_boozer_all.sh for the solid-body optimization of design A
# (boozer_solid_body.py, designA_solid_body/): run it from designA_solid_body/ (the script reads
# ../convert/designA_after_scaled.*), under the BOOZER venv; writes ./output/<task>/..., e.g.
#   sbatch -p ccm -C icelake -n 1 -c 1 -t 3-00:00:00 -o logs/solid_body_%j.out \
#          run_boozer_solid_body.sh 0.06 0.0 0 0 0 0 0 1 0 DN 0
#
#   run_boozer_solid_body.sh <margin> <well> <Z> <distance> <on_vessel> <config> \
#                     <vessel_id> <mono> <attempt> <null> [AR]
#
# Exit status is boozer_solid_body.py's.
set -uo pipefail

margin="$1"; well="$2"; Z="$3"; distance="$4"; on_vessel="$5"
config="$6"; vessel_id="$7"; mono="$8"; attempt="$9"; null="${10}"; AR="${11:-0}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPATH="/mnt/home/agiuliani/ceph/STAR_LITE:${PYTHONPATH:-}"

source /mnt/home/agiuliani/ceph/STAR_LITE/venv/bin/activate

python ./boozer_solid_body.py \
  --margin "$margin" \
  --well "$well" \
  --Z "$Z" \
  --distance "$distance" \
  --on-vessel "$on_vessel" \
  --config "$config" \
  --vessel-id "$vessel_id" \
  --mono "$mono" \
  --num-aux 0 \
  --attempt "$attempt" \
  --null "$null" \
  --AR "$AR"
