#!/bin/bash
# designB campaign decided on 2026-11-15, after campaign 7474635 (archived in
# archive/campaign_7474635/): config 0, SN, margin 0.06, well 0.0 only, AR 2, monothr 0.05,
# with a bound on the near-axis elongation (boozer_all.py --elongation: elongation in the
# plane normal to the magnetic axis <= bound at 16 toroidal angles), and ONLY double devices
# after the boozer step (no polish).
#
# tasks.jobs holds only the boozer step of each device (prefix.sh step=boozer, num_aux=0):
#   elongation bound in {OFF, 5, 4, 3} x attempts 0-959 -> 4 x 960 = 3840 tasks, on 30 icelake
#   nodes x 64 cores (submit_driver.sh) + 15 genoa nodes x 96 cores added with the run's
#   dbUtil.sh = 3360 cores. Attempt 0 is the unperturbed reference device; attempts >= 1 are
#   perturbed with a seed derived from the device name WITHOUT its elongation field, so attempt
#   k starts from the same coils for the four bounds (and, for OFF, from those of attempt k at
#   well 0.0 of campaign 7474635).
# It is run by designB_driver.py (embedded disBatch), which submits, for each device accepted
# by its boozer task, the two-configuration optimization (prefix.sh step=double,
# boozer_double_opt.py) for each num_aux in {3, 4, 5} and each lower bound on tr M of the
# config 1 X-point in {2.05, 2.2}: 6 double tasks per accepted device, into the same disBatch
# run, on whatever cores are free (submit_driver.sh: --steps double --num-aux 3 4 5
# --trace-min 2.05 2.2). The elongation bound applies to both configurations of a double.
# Doubles out of spec are kept in output_oos/ with their summary.txt (prefix.sh).
# Writes tasks.jobs, shuffled with a new random seed at every run (see below).
NUM_AUX=0  # boozer step: the num_aux=0 device (designB_driver.py sets num_aux per double task)
margin=0.06
well=0
elongs="OFF 5 4 3"  # one task per (elongation bound, attempt); emit() reads $elong from the loop below
Z=0
distance=0
on_vessel=0
config=0
vessel_id=0
mono=1
null=SN
AR=2
force=OFF

mkdir -p logs
: > tasks.jobs
emit() {  # emit <attempt> <mono_thr>
  echo "bash ./prefix.sh ${margin} ${well} ${Z} ${distance} ${on_vessel} ${config} ${vessel_id} ${mono} ${null} ${NUM_AUX} ${AR} $1 ${force} $2 boozer ${elong}" >> tasks.jobs
}
for elong in $elongs; do
  for attempt in $(seq 0 959); do emit "$attempt" 0.05; done
done

# Shuffle the task list, so that the order in which the boozer tasks start (hence which ones
# make the first wave) is random: a NEW seed at every run (the time, or SEED=<n> to
# reproduce an order), printed and recorded in tasks.jobs.seed. It only orders the tasks: the
# devices and their perturbations (seeded by the device name) do not depend on it.
SEED="${SEED:-$(date +%s)}"
echo "$SEED" > tasks.jobs.seed
python3 -c "import random, sys; L = open('tasks.jobs').read().splitlines(); random.Random(int(sys.argv[1])).shuffle(L); open('tasks.jobs', 'w').write('\n'.join(L) + '\n')" "$SEED"
echo "task order shuffled with seed $SEED (recorded in tasks.jobs.seed)"
echo "$(wc -l < tasks.jobs) tasks written to tasks.jobs"
