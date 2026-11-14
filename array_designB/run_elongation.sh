#!/bin/bash
# Lower-level step (called by prefix.sh): compute the min/max/mean flux-surface
# elongation along the magnetic axis (via the tangent map, utils/tangent_map.py) for
# ONE device and APPEND elongation_min/max/mean to its summary.txt. Runs under the
# BOOZER venv -- it does tangent-map/Boozer solves, like run_boozer_all.sh /
# run_LCFS.sh, NOT a render. Operates purely in the current (scratch) dir; does NOT
# touch ceph.
#
#   run_elongation.sh <design_json>
#
# <design_json> is design_opt_final_<ID>.json (init device) or
# design_polished_final_<ID>.json (polished device). Exit status is mk_elongation.py's.
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

./mk_elongation.py "$design_json"
