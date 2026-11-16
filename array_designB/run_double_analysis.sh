#!/bin/bash
# Post-processing of ONE in-spec double device, in separate steps (each skipped when its
# output is already there, so the script can be re-run on a device dir):
#   (1) mk_elongation_profile.py  toroidal profile of the near-axis elongation, 128 angles
#                                 -> elongation_profile.txt / .png          (matplotlib venv)
#   (3) mk_unfolding.py           unfolding constant C of the snowflake over 128 GP coil
#                                 perturbations, at 128 sections over a field period
#                                 -> unfolding_C.txt (phi = 0), unfolding_C_phi.txt,
#                                    unfolding_C_profile.txt                (gradgradB venv)
#       plot_unfolding.py         -> unfolding_C_hist.png (phi = 0) and
#                                    unfolding_C_profile.png (mean, std)    (matplotlib venv)
# Step (2), the repair of the snowflake with the aux currents, is a separate script.
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

# (done when the profile has its 128 sections: an older profile with fewer is redone)
if [ -s "$dir/unfolding_C.txt" ] && [ -s "$dir/unfolding_C_profile.txt" ] \
   && [ "$(grep -vc '^#' "$dir/unfolding_C_profile.txt")" -ge 128 ]; then
  echo "(3) unfolding constant: already there"
else
  ( source "$VENVS/venv_gradgradB/bin/activate" && ./mk_unfolding.py "$json" ) \
    || { echo "WARNING: unfolding constant failed for $dir"; failed=$((failed + 1)); }
fi
# figures: the histogram at phi = 0 and the toroidal profiles of the mean and std of C
if [ -s "$dir/unfolding_C.txt" ] && { [ ! -s "$dir/unfolding_C_hist.png" ] \
     || { [ -s "$dir/unfolding_C_profile.txt" ] && [ ! -s "$dir/unfolding_C_profile.png" ]; }; }; then
  ( source "$VENVS/venv_matplotlib/bin/activate" && ./plot_unfolding.py "$dir" ) \
    || { echo "WARNING: unfolding figures failed for $dir"; failed=$((failed + 1)); }
fi
exit "$failed"
