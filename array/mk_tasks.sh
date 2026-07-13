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
NUM_AUX=5
margins=(0.06 0.08 0.10 0.12 0.14 0.16)
wells=(OFF 0 -100)
binary_values=(0 1)
mono_values=(0 1 2)
configs=(0 1 2 3)
# Vessel geometry: 0/1/2 = pill-pipe / renaissance / torus; 3 = constant-radius
# helical (centerline starts as the magnetic axis); 4 = variable-radius helical
# (radius R(t) is a Fourier series of the same order as the centerline).
# vessel_id >= 5000 = welded-pipe helical vessel whose design variables are the
# discrete xyz centreline nodes; the id ENCODES the segment count, nseg =
# vessel_id - 5000 (so each nseg is a distinct device via the vesselID folder
# field, no folder-name change). nfp=2, so nseg must be a multiple of 2*nfp=4 to
# be valid for the DN (stellsym) devices: 5004/5008/5012 = 4/8/12 segments.
vessel_values=(0 1 2 3 4 5004 5008 5012)
# Null type: DN = double-null (stellsym, current behavior); SN = single-null
# (drop stellsym, push the bottom X-point to the lower wall).
null_values=(DN SN)
# Aspect-ratio / on-axis-iota knob (boozer_all.py --AR, prefix.sh's 11th arg):
#   0 = leave AR as-is, NO on-axis iota constraint (original behaviour),
#   1 = leave AR as-is but PIN the on-axis iota,
#   2 = for the first 5 outer BFGS iterations lower the AR toward the 80% LCFS toroidal-flux
#       surface AND pin the on-axis iota.
# Each value is a DISTINCT device (AR is in the folder name), so scanning {0,1,2} triples
# the task count.
AR_values=(0 1 2)
ATTEMPTS=2

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
                      echo "bash ./prefix.sh ${margin} ${well} ${Z} ${distance} ${on_vessel} ${config} ${vessel_id} ${mono} ${null} ${NUM_AUX} ${AR} ${attempt}" >> tasks.jobs
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
