#!/bin/bash
# designB campaign decided on 2026-11-20 (aim: max coil force below 3600 N/m at fixed |B| on
# axis and major radius): config 0, SN, well 0.0, AR in {0, 1, 2} (0: optimization surface left
# as it is, no on-axis iota constraint; 1: on-axis iota pinned; 2: aspect ratio lowered, on-axis
# iota pinned), monothr 0.05, plasma-to-vessel margin 0.06 m, NO bound on the near-axis
# elongation (--elongation OFF), the bound 3600 or 4000 N/m on the maximum coil force per unit
# length (--force), modular coils only (--extra-coils none), the keep-out cylinder (always on),
# and relaxed coil geometry (prefix.sh 18th argument geom, boozer_all.py --length-max,
# --curvature-max, --msc-max, --clearance): coil length <= 4.25, 4.5, 4.75 or 5 m; (max
# curvature [1/m], max mean squared curvature [1/m^2]) = (10, 20), (20, 30) or (30, 40);
# coil-to-vessel clearance >= 0.07, 0.08 or 0.09 m (0.12 before). ONLY double devices after the
# boozer step (no polish); the doubles inherit the geometry bounds of their device (yaml).
#
# tasks.jobs holds only the boozer step of each device (prefix.sh step=boozer, num_aux=0):
#   AR {0, 1, 2} x force {3600, 4000} x length {4.25, 4.5, 4.75, 5} x curvature {3 pairs} x
#   clearance {0.07, 0.08, 0.09} x attempts ($attempts below) -> 216 series. Attempt 0 is the
#   unperturbed reference device; attempts >= 1 are perturbed with a seed derived from the
#   device name WITHOUT its elongation, extra and geometry fields but WITH its AR, margin and
#   force fields: attempt k starts from the same coils for every geometry, and from DIFFERENT
#   coils in the six AR x force series.
# It is run by designB_driver.py (embedded disBatch), which submits, for each device accepted
# by its boozer task, the two-configuration optimization (prefix.sh step=double) for each
# num_aux in {3, 4, 5}, with the lower bound 2.05 on tr M of the config 1 X-point and each
# bound on E[C] of submit_driver.sh (--ec-max), into the same disBatch run, on whatever
# cores are free. The force bound of a device also applies to its doubles (yaml).
# Doubles out of spec are kept in output_oos/ with their summary.txt (prefix.sh).
# Writes tasks.jobs, shuffled with a new random seed at every run (see below).
NUM_AUX=0  # boozer step: the num_aux=0 device (designB_driver.py sets num_aux per double task)
margins="0.06"  # plasma-boundary-to-vessel margin [m]; emit() reads $margin from the loop below
well=0
elongs="OFF"  # no elongation bound; one task per (elongation bound, attempt); emit() reads $elong from the loop below
Z=0
distance=0
on_vessel=0
config=0
vessel_id=0
mono=1
null=SN
ARs="0 1 2"  # 0: optimization surface left as it is, no on-axis iota constraint; 1: on-axis iota pinned; 2: aspect ratio lowered, on-axis iota pinned; emit() reads $AR from the loop below
forces="3600 4000"  # bound on the maximum coil force per unit length [N/m], for every AR; emit() reads $force from the loop below
# extra coils (boozer_all.py --extra-coils, utils/extra_coils.py), currents starting at 0: none,
# circ (4 vertical circular coils), T (design A's 2 T coils, turned by 45 deg about z), LL (design
# A's 4 L coils), LT and TL (per half period one L and one T, in that toroidal order; 2 L + 2 T
# coils); emit() reads $extra from the loop below
extras="none"  # modular coils only (2026-11-20; circ only as a fallback if this gives no device)
# relaxed coil geometry (prefix.sh 18th argument geom); emit() reads $len, $kap, $msc, $clr from
# the loops below
lengths="4.25 4.5 4.75 5"  # coil length bound [m] (boozer_all.py --length-max; 4.0 before)
curvs="10:20 20:30 30:40"  # max curvature [1/m] : max mean squared curvature [1/m^2] (8.67 : 20 before)
clearances="0.07 0.08 0.09"  # coil-to-vessel clearance [m] (--clearance; 0.12 before)
attempts="$(seq 0 15)"  # 16 attempts per series: 216 series -> 3456 stage-1 tasks

mkdir -p logs
: > tasks.jobs
emit() {  # emit <attempt> <mono_thr>
  echo "bash ./prefix.sh ${margin} ${well} ${Z} ${distance} ${on_vessel} ${config} ${vessel_id} ${mono} ${null} ${NUM_AUX} ${AR} $1 ${force} $2 boozer ${elong} ${extra} len=${len},kap=${kap},msc=${msc},clr=${clr}" >> tasks.jobs
}
for AR in $ARs; do
  for margin in $margins; do
    for force in $forces; do
      for extra in $extras; do
        for elong in $elongs; do
          for len in $lengths; do
            for curv in $curvs; do
              kap="${curv%%:*}"; msc="${curv##*:}"
              for clr in $clearances; do
                for attempt in $attempts; do emit "$attempt" 0.05; done
              done
            done
          done
        done
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
