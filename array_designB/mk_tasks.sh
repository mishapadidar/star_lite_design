#!/bin/bash
# designB campaign decided on 2026-11-17, after campaign 7477835 (archived in
# archive/campaign_7477835/): config 0, SN, well 0.0 only, AR in {0, 2} (0: optimization
# surface left as it is, no on-axis iota constraint; 2: aspect ratio lowered, on-axis iota
# pinned, as in the earlier campaigns), monothr 0.05, plasma-to-vessel margin in {0.06, 0.08} m,
# NO bound on the near-axis elongation (boozer_all.py --elongation OFF), the bound 3600 N/m
# on the maximum coil force per unit length or none (--force 3600 | OFF; the starting design
# is at about 3300 N/m; the bound 3600 is run for AR 0 only), and ONLY double devices after
# the boozer step (no polish).
#
# tasks.jobs holds only the boozer step of each device (prefix.sh step=boozer, num_aux=0):
#   margin in {0.06, 0.08} x (AR 0 with force 3600 and OFF, AR 2 with force OFF) x attempts
#   0-63 -> 6 x 64 = 384 tasks (64 attempts only: each accepted device gets 144 tasks after
#   the boozer step, see submit_driver.sh). Attempt 0 is the unperturbed reference device; attempts
#   >= 1 are perturbed with a seed derived from the device name WITHOUT its elongation field
#   but WITH its AR, margin and force fields: attempt k starts from DIFFERENT coils in the
#   six series.
# It is run by designB_driver.py (embedded disBatch), which submits, for each device accepted
# by its boozer task, the two-configuration optimization (prefix.sh step=double) for each
# num_aux in {3, 4, 5}, with the lower bound 2.05 on tr M of the config 1 X-point and each
# bound on E[C] of submit_driver.sh (--ec-max), into the same disBatch run, on whatever
# cores are free. The force bound of a device also applies to its doubles (yaml).
# Doubles out of spec are kept in output_oos/ with their summary.txt (prefix.sh).
# Writes tasks.jobs, shuffled with a new random seed at every run (see below).
NUM_AUX=0  # boozer step: the num_aux=0 device (designB_driver.py sets num_aux per double task)
margins="0.06 0.08"  # plasma-boundary-to-vessel margin [m]; emit() reads $margin from the loop below
well=0
elongs="OFF"  # no elongation bound; one task per (elongation bound, attempt); emit() reads $elong from the loop below
Z=0
distance=0
on_vessel=0
config=0
vessel_id=0
mono=1
null=SN
ARs="0 2"  # 0: optimization surface left as it is, no on-axis iota constraint; 2: aspect ratio lowered, on-axis iota pinned; emit() reads $AR from the loop below
forces="3600 OFF"  # bound on the maximum coil force per unit length [N/m], or OFF; emit() reads $force from the loop below

mkdir -p logs
: > tasks.jobs
emit() {  # emit <attempt> <mono_thr>
  echo "bash ./prefix.sh ${margin} ${well} ${Z} ${distance} ${on_vessel} ${config} ${vessel_id} ${mono} ${null} ${NUM_AUX} ${AR} $1 ${force} $2 boozer ${elong}" >> tasks.jobs
}
for AR in $ARs; do
  for margin in $margins; do
    for force in $forces; do
      # the force bound is scanned for AR 0 only: AR 2 runs without a bound
      if [ "$force" != "OFF" ] && [ "$AR" != "0" ]; then continue; fi
      for elong in $elongs; do
        for attempt in $(seq 0 63); do emit "$attempt" 0.05; done
      done
    done
  done
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
