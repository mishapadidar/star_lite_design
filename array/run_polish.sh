#!/bin/bash
# Lower-level step (called by prefix.sh / prefix_restart.sh): run the singular
# polish + coil optimization (boozer_singular_opt.py) on a design json, under the
# BOOZER venv. Operates in the current (scratch) dir and writes its outputs in
# place next to the input json. Does NOT touch ceph.
#
#   run_polish.sh <design_opt_final.json> <num_aux>
#
# Exit status is boozer_singular_opt.py's; the master gates on the produced
# design_polished_final.json and its max_rel_error.txt.
set -uo pipefail

design_json="$1"
num_aux="$2"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPATH="/mnt/home/agiuliani/ceph/STAR_LITE:${PYTHONPATH:-}"

source /mnt/home/agiuliani/ceph/STAR_LITE/venv/bin/activate

./boozer_singular_opt.py "$design_json" --num-aux "$num_aux"
