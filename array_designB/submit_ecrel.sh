#!/bin/bash
# Relative E[C] continuation doubles (2026-11-21): disBatch on tasks_ecrel.jobs, in array_designB.
cd "$(dirname "$0")"
module load disBatch/beta
sbatch --parsable -p ccm -N 4 -C icelake --ntasks-per-node=64 -c 1 -t 7-00:00:00 -J designB_ecrel \
       -o logs/designB_ecrel_%j.out --wrap "disBatch tasks_ecrel.jobs"
