#!/bin/bash
# Lower-level step (called by prefix.sh): run the initial num_aux=0 optimization
# with boozer_all.py under the BOOZER venv. Operates purely in the current
# directory (the master's local scratch run dir); writes ./output/<task>/...
# Does NOT touch ceph -- the master decides what to copy back.
#
#   run_boozer_all.sh <margin> <well> <Z> <distance> <on_vessel> <config> \
#                     <vessel_id> <mono> <attempt> <null> [AR] [force] [mono_thr] [elongation] [extra]
#
# force: maximum coil force per unit length in N/m, or OFF (default) for none.
# mono_thr: monodromy threshold of the boozer_all constraint (default 0.1).
# elongation: upper bound on the near-axis elongation, or OFF (default) for none.
# Exit status is boozer_all.py's; the master gates on the produced design json.
set -uo pipefail

margin="$1"; well="$2"; Z="$3"; distance="$4"; on_vessel="$5"
config="$6"; vessel_id="$7"; mono="$8"; attempt="$9"; null="${10}"; AR="${11:-0}"
force="${12:-OFF}"
mono_thr="${13:-0.1}"
elongation="${14:-OFF}"
# extra coils (boozer_all.py --extra-coils): none (default), circ, T or L
extra="${15:-none}"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export PYTHONPATH="/mnt/home/agiuliani/ceph/STAR_LITE:${PYTHONPATH:-}"

source /mnt/home/agiuliani/ceph/STAR_LITE/venv/bin/activate

# AVX-512 build of simsoptpp (same source, compiled on an icelake node with -march=native) on
# the Intel nodes that have AVX-512: the script is then started through avx512_loader.py,
# which loads that build first (the simsoptpp of the venv, built without AVX-512, also runs on
# the AMD nodes; PYTHONPATH cannot select another build, the editable install comes first).
# Used only once the file ENABLED is next to the build (created by hand after it was checked on
# an icelake node; remove it to go back to the build of the venv), and only if it loads.
AVX512_SO="$(ls /mnt/ceph/users/agiuliani/STAR_LITE/simsoptpp_avx512/simsopt/simsoptpp*.so 2> /dev/null | head -1)"
RUN_PY=(python)
if grep -q GenuineIntel /proc/cpuinfo && grep -q avx512f /proc/cpuinfo \
   && [ -n "$AVX512_SO" ] && [ -f "$(dirname "$AVX512_SO")/ENABLED" ] \
   && python ./avx512_loader.py "$AVX512_SO" --check > /dev/null 2>&1; then
  RUN_PY=(python ./avx512_loader.py "$AVX512_SO")
  echo "simsoptpp: AVX-512 build ($AVX512_SO)"
else
  echo "simsoptpp: build of the venv (no AVX-512 build used on $(hostname))"
fi

"${RUN_PY[@]}" ./boozer_all.py \
  --margin "$margin" \
  --well "$well" \
  --Z "$Z" \
  --distance "$distance" \
  --on-vessel "$on_vessel" \
  --config "$config" \
  --vessel-id "$vessel_id" \
  --mono "$mono" \
  --num-aux 0 \
  --attempt "$attempt" \
  --null "$null" \
  --AR "$AR" \
  --force "$force" \
  --mono-thr "$mono_thr" \
  --elongation "$elongation" \
  --extra-coils "$extra"
