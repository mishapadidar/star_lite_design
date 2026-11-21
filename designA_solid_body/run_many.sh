#!/bin/bash
# Several solid-body runs (3 configurations, DN) in ONE job, one core each, side by side on the
# same node (a rusty job gets whole nodes: one job per run would take one node per run):
#   sbatch -p ccm -C rome -N 1 -n <number of runs> -c 1 -t 3-00:00:00 -J dA_sb_many \
#          -o logs/solid_body_many_%j.out run_many.sh <vessel id> [<vessel id> ...]
# Each run writes its own log, logs/solid_body_vessel<id>_<job>.out.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
for v in "$@"; do
  echo "$(date '+%F %T') starting vessel $v"
  bash run_boozer_solid_body.sh 0.06 0.0 0 0 0 -1 "$v" 1 0 DN 0 \
       > "logs/solid_body_vessel${v}_${SLURM_JOB_ID:-local}.out" 2>&1 &
done
wait
echo "$(date '+%F %T') all runs finished"
