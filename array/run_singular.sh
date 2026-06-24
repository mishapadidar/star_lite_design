#!/bin/bash
# Lower-level step (called by prefix_singular.sh): compute the auxiliary coils for a
# design json (trace/identity monodromy enforced EXACTLY) WITHOUT any coil
# optimization, via boozer_singular.py, under the BOOZER venv. Operates in the
# current (scratch) dir; boozer_singular.py writes its outputs to a sibling
# '<folder>_unpolished' directory next to the input json. Does NOT touch ceph.
#
#   run_singular.sh <design_opt_final.json> <num_aux>
#
# Exit status is boozer_singular.py's.
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

./boozer_singular.py "$design_json" --num-aux "$num_aux"
