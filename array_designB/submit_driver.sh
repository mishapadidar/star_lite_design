#!/bin/bash
# Launch a designB campaign: designB_driver.py (embedded disBatch) on tasks.jobs, written by
# mk_tasks.sh. Run with a login shell, so that `module` exists:
#     bash -l submit_driver.sh [driver options]     e.g. --resume after a cancelled run
# The driver output goes to logs/designB_driver_<job>.out; its disBatch files and CSV are
# designB_driver_disBatch_<job>* in this directory.
cd "$(dirname "$0")"
[ -f tasks.jobs ] || { echo "ERROR: no tasks.jobs here (run mk_tasks.sh first)"; exit 1; }
mkdir -p logs
module load disBatch/beta
# 30 icelake nodes x 64 cores = 1920 cores. More cores join the same run by submitting the
# designB_driver_disBatch_<job>_dbUtil.sh that disBatch writes at start, e.g. 15 genoa nodes
# x 96 cores: 45 nodes and 3360 cores, under the per-user cap of the ccm QOS (45 nodes, 3840 cpus).
sbatch -p ccm -N 30 -C icelake --ntasks-per-node=64 -c 1 -t 7-00:00:00 -J designB \
       -o logs/designB_driver_%j.out --wrap "python3 designB_driver.py tasks.jobs $*"
