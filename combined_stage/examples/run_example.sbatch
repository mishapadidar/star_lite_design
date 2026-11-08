#!/bin/bash
#SBATCH --job-name=cs_example
#SBATCH --partition=compute
#SBATCH --exclude=cn12,cn07
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#
# Run one combined-stage example script on a compute node; outputs land in the submit dir:
#   cd <run dir>
#   sbatch --output=slurm_%j.out <this dir>/run_example.sbatch <this dir>/combined_stage_simple.py
# Extra arguments are passed on to the script, e.g. ... combined_stage_ladder_simple.py --smoke
set -euo pipefail
: "${1:?usage: sbatch run_example.sbatch <example.py>}"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
echo "host $(hostname)  start $(date)  script $1"
/data/common/raffael.wendlinger/envs/simflare/bin/python -u "$@"     # extra arguments go to the script
echo "end $(date)"
