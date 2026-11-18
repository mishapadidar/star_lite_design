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
# After the boozer step: only doubles, for num_aux 3, 4, 5, the trace bound 2.05, two tasks on
# E[C] (mean over the sections, initial weight 10): the CONTINUATION started at 1.0 (prefix.sh
# goes on down to 0.5 by 0.1, each level restarting from the previous one, until the first level
# out of spec; one device per level, named _ec=<bound>_from=<previous bound> after the first)
# and the reference without bound (OFF); each for the two targets (double = X-point + snowflake,
# snowflake = the snowflake alone) and the bounds 5, 10, 15 and 20 kA on the aux (PF) coil
# currents: 3 x 2 x 2 x 4 = 48 tasks per accepted device; options given on the command line
# come after and win.
# Started on FOUR icelake nodes in the gen partition (2026-11-17: the cluster is being drained
# and 30 nodes at once do not come; gen is served before ccm, PriorityTier 20 against 10, but
# allows only about 4 nodes per user); the other nodes are added one by one in ccm with the
# dbUtil.sh of the run.
sbatch -p gen -N 4 -C icelake --ntasks-per-node=64 -c 1 -t 7-00:00:00 -J designB \
       -o logs/designB_driver_%j.out \
       --wrap "python3 designB_driver.py tasks.jobs --steps double --num-aux 3 4 5 --trace-min 2.05 --ec-max 1.0 OFF --target double snowflake --pf-max 5000 10000 15000 20000 $*"
