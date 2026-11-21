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
# and the reference without bound (OFF); only the target double (config 1 X-point with the aux
# coils off + config 2 snowflake with the aux coils) and only the bound 5 kA on the aux (PF) coil
# currents (decided on 2026-11-20; 10 kA later, only for devices that fail at 5 kA):
# 3 x 2 = 6 tasks per accepted device; options given on the command line come after and win.
# --ec-max takes, per value: a number (fixed bound, continued down by prefix.sh's EC_CONTINUATION
# list), OFF (no bound) or f<factor>, e.g. f0.9 (RELATIVE continuation since 2026-11-21: each level
# bounded by factor x the E[C] reached by the previous one, down to EC_FLOOR = 0.5; levels named
# ..._ec=f0.9_level=<k>_..., each one post-processed and copied to ceph as soon as it ends; initial
# E[C] weight 0.1, EC_WEIGHT0). E.g. --ec-max f0.9 OFF, or 1.0 f0.9 OFF for both modes.
# Started on FOUR icelake nodes in the gen partition (gen is served before ccm, PriorityTier 20
# against 10, but allows only about 4 nodes per user); the other nodes join one by one, as
# single-node icelake jobs in ccm running the dbUtil.sh of the run. (A single 30-node job in ccm
# was tried on 2026-11-18: with the cluster being drained, Slurm only expected it to start the
# next morning.)
sbatch -p gen -N 4 -C icelake --ntasks-per-node=64 -c 1 -t 7-00:00:00 -J designB \
       -o logs/designB_driver_%j.out \
       --wrap "python3 designB_driver.py tasks.jobs --steps double --num-aux 3 4 5 --trace-min 2.05 --ec-max 1.0 OFF --target double --pf-max 5000 $*"
