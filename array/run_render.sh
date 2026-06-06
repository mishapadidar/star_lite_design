#!/bin/bash
# Lower-level step (called by prefix.sh / prefix_restart.sh): trace + plot + render
# ONE device, under the MATPLOTLIB venv + paraview module. Operates in the current
# (scratch) dir and writes its plots/renders into the device's output dir. Does NOT
# touch ceph.
#
#   run_render.sh <design_json> <output_dir>
#
# <design_json> is design_opt_final.json for the init device or
# design_polished_final.json for a polished device; <output_dir> is the device
# folder the renders are written into. Exit status is non-zero if any step fails.
set -uo pipefail

design_json="$1"
out_dir="$2"

module load modules/2.3-20240529
module load paraview/5.10.1
source /mnt/home/agiuliani/ceph/STAR_LITE/venv_matplotlib/bin/activate
export PYTHONPATH="${PYTHONPATH:-}:/mnt/home/agiuliani/ceph/STAR_LITE/"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

./mk_manifolds.py "$design_json"
./plot_manifolds.py "$design_json"
xvfb-run -a pvbatch --force-offscreen-rendering mk_paraview.py "$out_dir"
