#!/bin/bash
# Lower-level step (called by prefix.sh): run the initial num_aux=0 optimization
# with boozer_all.py under the BOOZER venv. Operates purely in the current
# directory (the master's local scratch run dir); writes ./output/<task>/...
# Does NOT touch ceph -- the master decides what to copy back.
#
#   run_boozer_all.sh <margin> <well> <Z> <distance> <on_vessel> <config> \
#                     <vessel_id> <mono> <attempt> <null> [AR]
#
# Exit status is boozer_all.py's; the master gates on the produced design json.
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

./boozer_all.py \
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
