#!/bin/bash
#SBATCH --job-name=cs_polish
#SBATCH --partition=compute
#SBATCH --exclude=cn12,cn07
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#
# Mercier polish: one more rung after a finished ladder, at higher VMEX radial resolution.
#
# The ladders optimise at ns = 16, where DMerc[2:-1] is unconverged: hsx_scan_ext/c24_i0.25_A10_w3/07 reads Mercier-
# stable at ns 16 but its core (s < 0.45) is unstable at ns 31 / 41 / 51 / 101 (runs/vmex/c24_ns_test, 2026-10-26).
# This reruns the ladder's last rung -- its own config_used.yaml, so every objective, constraint and weight carries
# over -- from that rung's final state, with
#   vmex.ns = NS (default 51)                       the equilibrium every term is evaluated on
#   plasma_objective.mercier.s_min = S_MIN (0.1)    the near-axis DMerc diverges with ns; s >= 0.1 converges
#   augmented_lagrangian.outer_maxiter = OUTER (10)
# ~2x the seconds per evaluation of the ns = 16 rung (28.5 vs 14.6 s at the start of c24's beta 1 % rung).
# Output <device>/08_mercier_polish (the opt viewer shows it as the ladder's next rung).
#
#   DEVICE=<ladder dir> [FROM=07_beta1_G_band] [NS=51] [S_MIN=0.1] [OUTER=10] \
#     [MERC_EXTRA="normalize: phi_edge, smoothing: 1.0e-4, hard_scale: 1.0"] [HARD_ADD=mercier] [OUT_NAME=09_mercier_hard] \
#     sbatch --output=<DEVICE>/logs/polish_%j.out combined_stage/tools/mercier_polish.sbatch
# MERC_EXTRA: extra / overriding keys of the Mercier block (YAML flow items); HARD_ADD: keys appended to the rung's
# coils.hard list (2026-11-07: normalised Mercier as a hard AL row, see plasma_objective.mercier in combined_stage_vmex.py).
set -euo pipefail
: "${DEVICE:?set DEVICE}"
FROM=${FROM:-$(ls -d "$DEVICE"/0*_beta* | grep -v '_v[0-9]*[a-z]*$' | sort | tail -1 | xargs basename)}
S="$DEVICE/$FROM"
NS=${NS:-51}; S_MIN=${S_MIN:-0.1}; OUTER=${OUTER:-10}
OUT="$DEVICE/${OUT_NAME:-08_mercier_polish}"
[ -f "$S/coils/coils_final.json" ] || { echo "$S has not finished (no coils_final.json)"; exit 1; }
PY=/data/common/raffael.wendlinger/envs/simflare/bin/python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg PYTHONUNBUFFERED=1
cd /data/common/raffael.wendlinger/simflare/ext/star_lite_design/combined_stage
# the rung's own Mercier block, with s_min added (weight / margin / smoothing unchanged)
MERC=$("$PY" -c "
import yaml, sys
m = (yaml.safe_load(open('$S/config_used.yaml'))['plasma_objective'].get('mercier') or {})
m = dict({'enabled': True, 'margin': 0.0, 'smoothing': 1.0e-3, 'weight': 1.0e-4}, **m, s_min=$S_MIN)
m.update(yaml.safe_load('{${MERC_EXTRA:-}}') or {})
print('{' + ', '.join(f'{k}: {v}' for k, v in m.items()) + '}')")
echo "host $(hostname)  start $(date)  polish $S -> $OUT  ns=$NS  mercier=$MERC  outer=$OUTER"
EXTRA=()
if [ -n "${HARD_ADD:-}" ]; then
  HARD=$("$PY" -c "
import yaml
h = list(yaml.safe_load(open('$S/config_used.yaml'))['coils'].get('hard') or [])
h += [k for k in '${HARD_ADD}'.split(',') if k not in h]
print('[' + ', '.join(h) + ']')")
  EXTRA+=(--set "coils.hard=$HARD")
  echo "coils.hard=$HARD"
fi
[ -f "$S/xpoints/xpoint_final.json" ] && EXTRA+=(--xpoint "$S/xpoints/xpoint_final.json")
[ -f "$S/vessel/vessel_final.json" ] && EXTRA+=(--set "vessel.seed_file=$S/vessel/vessel_final.json")   # vessel runs
[ -f "$OUT/state/checkpoint.npz" ] && [ -f "$OUT/state/al_state.yaml" ] && EXTRA+=(--resume)
mkdir -p "$OUT/logs"
"$PY" -u combined_stage_vmex.py --config "$S/config_used.yaml" --vmec-input "$S/inputs/input.combined_stage_final" \
    --coils "$S/coils/coils_final.json" --outdir "$OUT" --set vmex.ns=$NS \
    --set vmex.restart_wout="$S/wout/wout_combined_stage_final.nc" --set "plasma_objective.mercier=$MERC" \
    --set augmented_lagrangian.outer_maxiter=$OUTER ${EXTRA[@]+"${EXTRA[@]}"} \
    >> "$OUT/logs/run.log" 2>&1
echo "end $(date)"
