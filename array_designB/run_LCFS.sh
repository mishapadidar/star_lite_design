#!/bin/bash
# Lower-level step (called by prefix.sh): compute the last closed flux surface (LCFS)
# for ONE device by pushing its Boozer-surface volume to the largest value that still
# converges without self-intersecting, via adaptive continuation (see mk_LCFS.py).
# Runs under the BOOZER venv -- it does a Boozer solve, like run_boozer_all.sh /
# run_polish.sh, NOT a render. Operates purely in the current (scratch) dir and writes
# LCFS_<device ID>.json into the device's output dir. Does NOT touch ceph.
#
#   run_LCFS.sh <design_json>
#
# <design_json> is design_opt_final_<ID>.json (init device) or
# design_polished_final_<ID>.json (polished device). Exit status is mk_LCFS.py's.
set -uo pipefail

design_json="$1"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPATH="/mnt/home/agiuliani/ceph/STAR_LITE:${PYTHONPATH:-}"

source /mnt/home/agiuliani/ceph/STAR_LITE/venv/bin/activate

./mk_LCFS.py "$design_json"
