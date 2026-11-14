#!/bin/bash
# Lower-level step (called by prefix.sh / prefix_restart.sh): trace + plot + render
# ONE device, under the MATPLOTLIB venv (+ pvbatch for the paraview views, when one is
# on the PATH: they are skipped otherwise). Operates in the current (scratch) dir and
# writes its plots/renders into the device's output dir. Does NOT touch ceph.
#
#   run_render.sh <design_json> <output_dir>
#
# <design_json> is design_opt_final.json for the init device or
# design_polished_final.json for a polished device; <output_dir> is the device
# folder the renders are written into. Exit status is non-zero if any step fails.
set -uo pipefail

design_json="$1"
out_dir="$2"

# No module load: the module set this used (modules/2.3-20240529, with paraview/5.10.1)
# was retired and no ParaView module is left on the cluster; the venv's python does not
# need it.
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
# Paraview device views (scene_*.png): optional. Without pvbatch (no ParaView on the
# cluster any more) they are skipped with a warning instead of failing the render, which
# made prefix.sh exit before copying the device to ceph. mk_paraview.py only needs VTK, so
# the views can be made later from the device's *_final VTK files with any VTK python.
if command -v pvbatch > /dev/null; then
  xvfb-run -a pvbatch --force-offscreen-rendering mk_paraview.py "$out_dir"
else
  echo "WARNING: pvbatch not found (no ParaView on $(hostname)): paraview views (scene_*.png) skipped"
fi
