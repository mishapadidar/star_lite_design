#!/bin/bash
# Post-processing of ONE in-spec double device, in separate steps (each skipped when its
# output is already there, so the script can be re-run on a device dir):
#   (1) mk_elongation_profile.py  toroidal profile of the near-axis elongation, 128 angles
#                                 -> elongation_profile.txt / .png          (matplotlib venv)
#   (3) mk_unfolding_exact.py     mean unfolding constant E[C] of the snowflake under GP coil
#                                 perturbations, computed exactly (no realizations): at
#                                 phi = 0, in every section and averaged over the sections
#                                 -> unfolding_EC_exact_profile.txt and the unfolding_EC_exact*
#                                    lines of summary.txt                   (gradgradB venv)
#       plot_unfolding_exact.py   -> unfolding_EC_exact_profile.png         (matplotlib venv)
# Step (2), the repair of the snowflake with the aux currents, is a separate script. The
# statistics over realizations (mk_unfolding.py, plot_unfolding.py: about 30 minutes per
# device) are no longer run here.
#
#   run_double_analysis.sh <double_device_dir>
#
# Works IN PLACE in <double_device_dir> (it holds design_double_final_<TAG>.json and its
# yaml); run from the directory of the scripts. A failed step is reported and the others
# still run; the exit status is the number of failed steps.
set -uo pipefail

dir="${1%/}"
json="$(ls "$dir"/design_double_final_*.json 2>/dev/null | head -1)"
[ -n "$json" ] || { echo "ERROR: no design_double_final_*.json in $dir"; exit 99; }

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export PYTHONPATH="/mnt/home/agiuliani/ceph/STAR_LITE:${PYTHONPATH:-}"
VENVS=/mnt/home/agiuliani/ceph/STAR_LITE
failed=0

if [ -s "$dir/elongation_profile.txt" ] && [ -s "$dir/elongation_profile.png" ]; then
  echo "(1) elongation profile: already there"
else
  ( source "$VENVS/venv_matplotlib/bin/activate" && ./mk_elongation_profile.py "$json" ) \
    || { echo "WARNING: elongation profile failed for $dir"; failed=$((failed + 1)); }
fi

# (3) exact E[C] (mk_unfolding_exact.py): at phi = 0, its profile over the sections and their mean
if grep -q 'unfolding_EC_exact_phi_mean' "$dir/summary.txt" 2>/dev/null \
   && [ -s "$dir/unfolding_EC_exact_profile.txt" ]; then
  echo "(3) exact E[C]: already there"
else
  ( source "$VENVS/venv_gradgradB/bin/activate" && ./mk_unfolding_exact.py "$json" --force ) \
    || { echo "WARNING: exact E[C] failed for $dir"; failed=$((failed + 1)); }
fi
# figure: the profile of the exact E[C] over the sections
if [ -s "$dir/unfolding_EC_exact_profile.txt" ] && [ ! -s "$dir/unfolding_EC_exact_profile.png" ]; then
  ( source "$VENVS/venv_matplotlib/bin/activate" && ./plot_unfolding_exact.py "$dir" ) \
    || { echo "WARNING: exact E[C] figure failed for $dir"; failed=$((failed + 1)); }
fi
exit "$failed"
