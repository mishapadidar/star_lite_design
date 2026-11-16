#!/bin/bash
# Lower-level step (called by prefix.sh / prefix_restart.sh): trace + plot + render
# ONE device, under the MATPLOTLIB venv (which also has VTK, for the device views made
# by mk_paraview.py). Operates in the current (scratch) dir and
# writes its plots/renders into the device's output dir. Does NOT touch ceph.
#
#   run_render.sh <design_json> <output_dir> [noviews]
#
# <design_json> is design_opt_final.json for the init device or
# design_polished_final.json for a polished device; <output_dir> is the device
# folder the renders are written into. Exit status is non-zero if any step fails.
set -uo pipefail

design_json="$1"
out_dir="$2"
# Optional third argument "noviews": skip the device views. Used for the two configurations of
# a double: the geometry files that mk_paraview.py reads are in the double's own directory,
# where prefix.sh makes the views itself.
views="${3:-views}"

# No module load: the module set this used (modules/2.3-20240529, with paraview/5.10.1)
# was retired and no ParaView module is left on the cluster; the venv's python does not
# need it (the device views use the venv's own VTK).
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
# Run with DISPLAY removed for THIS command only (env -u), so plot_manifolds.py's
# guard reliably pins the non-interactive Agg backend and nothing is shown. Scoped
# to the one command, so the xvfb-run line below (which sets its own DISPLAY) is
# unaffected.
env -u DISPLAY ./plot_manifolds.py "$design_json"
# Device views (scene_*.png): optional. mk_paraview.py only needs VTK, which is installed
# in the matplotlib venv (no ParaView on the cluster any more). A failure is reported with
# a warning instead of failing the render, which would make prefix.sh exit before copying
# the device to ceph.
if [ "$views" = "noviews" ]; then
  echo "device views (scene_*.png) not made here (noviews)"
elif python -c "import vtk" 2> /dev/null; then
  xvfb-run -a python mk_paraview.py "$out_dir" \
    || echo "WARNING: mk_paraview.py failed on $(hostname): device views (scene_*.png) skipped"
else
  echo "WARNING: no VTK in the python of $(hostname): device views (scene_*.png) skipped"
fi
