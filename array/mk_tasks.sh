#!/bin/bash
# Emit the disBatch task list for the merged (init + polish) workflow.
#
# One line per (margin, well, Z, distance, on_vessel, config, vessel_id, mono,
# attempt). prefix.sh always produces the num_aux=0 (unpolished) device, and
# then for mono=1,2 it polishes+re-optimizes the freshly-computed num_aux=0
# design at EXACTLY num_aux=NUM_AUX (passed as the 11th argument; no scan).
# Devices therefore carry either 0 or NUM_AUX auxiliary coils, and num_aux>0
# devices are created only for mono=1,2, as required.
#
# Each parameter combination is attempted ATTEMPTS times; each attempt perturbs
# the base modular coils with a different (device-ID-seeded) jitter so the
# optimizer starts from a different point.
NUM_AUX=6
margins=(0.06 0.08 0.10 0.12)
wells=(OFF 100 0 -100)
binary_values=(0 1)
mono_values=(0 1 2)
configs=(0 1 2 3)
vessel_values=(0 1 2)
# Null type: DN = double-null (stellsym, current behavior); SN = single-null
# (drop stellsym, push the bottom X-point to the lower wall).
null_values=(DN SN)
# Aspect-ratio knob (boozer_all.py --AR, prefix.sh's 12th arg): 0 = leave AR as-is,
# 1 = reduce the plasma aspect ratio toward ~5. Each value is a DISTINCT device (AR
# is in the folder name), so scanning {0,1} doubles the task count.
AR_values=(0 1)
ATTEMPTS=5

mkdir -p logs
: > tasks.jobs

for margin in "${margins[@]}"; do
  for well in "${wells[@]}"; do
    for Z in "${binary_values[@]}"; do
      for distance in "${binary_values[@]}"; do
        for on_vessel in "${binary_values[@]}"; do
          for vessel_id in "${vessel_values[@]}"; do
            for config in "${configs[@]}"; do
              for mono in "${mono_values[@]}"; do
                for null in "${null_values[@]}"; do
                  for ((attempt=0; attempt<ATTEMPTS; attempt++)); do
                    for AR in "${AR_values[@]}"; do
                      echo "bash ./prefix.sh ${margin} ${well} ${Z} ${distance} ${on_vessel} ${config} ${vessel_id} ${mono} ${attempt} ${null} ${NUM_AUX} ${AR}" >> tasks.jobs
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
