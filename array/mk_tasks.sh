#!/bin/bash
# Emit the disBatch task list for the merged (init + polish) workflow.
#
# One line per (margin, well, Z, distance, on_vessel, config, vessel_id, mono,
# attempt). num_aux is NOT a task dimension: prefix.sh always produces the
# num_aux=0 (unpolished) device, and then for mono=1,2 it loops internally over
# num_aux = 1..NUM_AUX_MAX, polishing the freshly-computed num_aux=0 design.
# So num_aux>0 devices are created only for mono=1,2, as required.
#
# Each parameter combination is attempted ATTEMPTS times; each attempt perturbs
# the base modular coils with a different (device-ID-seeded) jitter so the
# optimizer starts from a different point.
margins=(0.06 0.08 0.10 0.12)
wells=(OFF 100 0 -100)
binary_values=(0 1)
mono_values=(0 1 2)
configs=(0 1 2 3)
vessel_values=(0 1 2)
ATTEMPTS=4

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
                for ((attempt=0; attempt<ATTEMPTS; attempt++)); do
                  echo "bash ./prefix.sh ${margin} ${well} ${Z} ${distance} ${on_vessel} ${config} ${vessel_id} ${mono} ${attempt}" >> tasks.jobs
                done
              done
            done
          done
        done
      done
    done
  done
done
