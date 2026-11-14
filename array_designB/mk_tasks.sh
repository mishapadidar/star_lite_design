#!/bin/bash
# Next designB campaign (decided 2026-11-14, after campaign 7463849): config 0, SN,
# margin 0.06, well 0.0 and well OFF, AR 2, monothr 0.05 only, on-axis |B| ripple
# threshold 0.5% (MODB_RIPPLE_THRESHOLD; 0.1% up to 7463849).
#
# tasks.jobs holds only the boozer step of each device (prefix.sh step=boozer, num_aux=0).
# It is run by designB_driver.py (embedded disBatch), which submits, for each device
# accepted by its boozer task, one polish task (prefix.sh step=polish) per num_aux in
# {3, 4, 5, 6} into the same disBatch run, on whatever cores are free.
#
# Campaign 7463849 (2400 tasks; cancelled on 2026-11-14 after 1920 had finished; archived in
# archive/campaign_7463849/ with its task list, its generator mk_tasks.sh, its output/ and
# logs/) kept accepted devices for 791 of the 2000 tasks at monothr 0.05 (39.6 %) and for 99
# of the 400 at monothr 0.02 (24.8 %). No polish succeeded at 0.02 (2 of the 423 finished
# polishes succeeded overall), so 0.02 is dropped.
#   monothr 0.05, for each of well 0.0 and well OFF: attempts 0-1919 -> 2 x 1920 = 3840 tasks,
#   on 30 icelake nodes x 64 cores (submit_driver.sh) + 15 genoa nodes x 96 cores added with the
#   run's dbUtil.sh = 3360 cores (attempt 0 is the unperturbed reference device; attempts
#   >= 1 are perturbed with a seed derived from the device name). At well 0.0 these names,
#   hence seeds and starting points, are those of archived devices of 7456743 (attempts 0-5)
#   and 7463849 (6-1919), but the runs differ: the ripple threshold, not in the name, changed.
# Polished devices out of spec are kept in output_oos/ with their summary.txt (prefix.sh).
# The polish now uses AUX_RADIUS_MAX = 1.2, N_POLISH_STAGES = 10 and DISCRIMINANT_WEIGHT = 1e-6
# (boozer_singular_opt.py); campaign 7463849 ran with 1.5, 5 and 1e-8.
# Writes tasks.jobs (shuffled with a fixed seed, as mk_tasks.sh does).
NUM_AUX=0  # boozer step: the num_aux=0 device (designB_driver.py sets num_aux per polish task)
margin=0.06
wells="0 OFF"  # one task per (well, attempt); emit() reads $well from the loop below
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
  echo "bash ./prefix.sh ${margin} ${well} ${Z} ${distance} ${on_vessel} ${config} ${vessel_id} ${mono} ${null} ${NUM_AUX} ${AR} $1 ${force} $2 boozer" >> tasks.jobs
}
for well in $wells; do
  for attempt in $(seq 0 1919); do emit "$attempt" 0.05; done
done

# Shuffle the task list (fixed seed, so the order is reproducible) so that every disBatch
# wave mixes all attempts.
python3 -c "import random; L = open('tasks.jobs').read().splitlines(); random.Random(0).shuffle(L); open('tasks.jobs', 'w').write('\n'.join(L) + '\n')"
echo "$(wc -l < tasks.jobs) tasks written to tasks.jobs"
