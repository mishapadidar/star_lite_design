#!/bin/bash
# Lower-level step (called by prefix.sh, step=double): run the two-configuration
# optimization (boozer_risk_opt.py, which replaced boozer_double_opt.py: config 1 = the
# boozer_all device with the aux coils off and a hyperbolic X-point, config 2 = a snowflake
# with <num_aux> aux coils, optionally with a bound on E[C]) on a boozer_all design json,
# under the BOOZER venv. Operates in the current (scratch) dir and writes its outputs in
# place next to the input json. Does NOT touch ceph.
#
#   run_double.sh <design_opt_final.json> <num_aux> [discriminant]
#
# Exit status is boozer_risk_opt.py's; the master gates on the produced
# design_double_final json and its max_rel_error.txt.
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

# venv_gradgradB, as run_polish.sh: its simsopt provides BiotSavart.d3B_by_dXdXdX, needed by
# the snowflake-discriminant constraint of config 2 (--discriminant 1).
source /mnt/home/agiuliani/ceph/STAR_LITE/venv_gradgradB/bin/activate

# AVX-512 build of simsoptpp on the Intel nodes that have AVX-512 (icelake), started through
# avx512_loader.py as in run_boozer_all.sh: only once the file ENABLED is next to the build,
# and only if it loads; otherwise the build of the venv is used.
AVX512_SO="$(ls /mnt/ceph/users/agiuliani/STAR_LITE/simsoptpp_avx512/simsopt_gradgradB/simsoptpp*.so 2> /dev/null | head -1)"
RUN_PY=(python)
if grep -q GenuineIntel /proc/cpuinfo && grep -q avx512f /proc/cpuinfo \
   && [ -n "$AVX512_SO" ] && [ -f "$(dirname "$AVX512_SO")/ENABLED" ] \
   && python ./avx512_loader.py "$AVX512_SO" --check > /dev/null 2>&1; then
  RUN_PY=(python ./avx512_loader.py "$AVX512_SO")
  echo "simsoptpp: AVX-512 build ($AVX512_SO)"
else
  echo "simsoptpp: build of the venv (no AVX-512 build used on $(hostname))"
fi

# optional 3rd argument: 1 adds the snowflake-discriminant constraint (identity polish only).
# DOUBLE_OPT_ARGS (optional, environment): extra options, e.g. "--stages 1 --maxiter 3" for
# a quick end-to-end test, or the bounds that prefix.sh adds (--trace-min, --ec-max with
# --ec-section). Without --ec-max, E[C] is only reported.
# run_double.sh --argv-file FILE: the next level of a relative E[C] continuation, whose command (the
# script and its arguments, one per line) the previous level left in FILE (prefix.sh runs it)
if [ "$design_json" = "--argv-file" ]; then
  mapfile -t LEVEL_ARGS < "$num_aux"
  "${RUN_PY[@]}" "${LEVEL_ARGS[@]}"
  exit $?
fi
"${RUN_PY[@]}" ./boozer_risk_opt.py "$design_json" --num-aux "$num_aux" --discriminant "${3:-0}" ${DOUBLE_OPT_ARGS:-}
