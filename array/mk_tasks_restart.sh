#!/bin/bash
# Emit the disBatch task list for RESTARTING the polish from just after the boozer_all
# stage, for mono=1 (identity) devices only.
#
# Same workflow as prefix.sh but starting at boozer_singular_opt.py: each line runs
# prefix_restart.sh, which REUSES the existing num_aux=0 boozer_all device from ceph
# (skipping the expensive boozer_all recompute), then runs run_polish.sh ->
# boozer_singular_opt.py + gate + render + sync. Use it to re-run a rewritten
# boozer_singular_opt.py against every mono=1 device already on ceph.
#
# Only combinations whose num_aux=0 device json actually exists on ceph are emitted (same
# inferred-path existence check as mk_tasks_singular.sh), so no task falls back to a full
# boozer_all recompute. The path is INFERRED from the parameter combination -- no globbing --
# mirroring boozer_all.py's folder name and prefix.sh's shard:
#   output/<shard>/<TASK_NAME(num_aux=0)>/design_opt_final_<DEVICE_ID>.json
# with DEVICE_ID = zlib.crc32(TASK_NAME), shard = md5(TASK_NAME)[:2].

NUM_AUX=5
margins=(0.06 0.08 0.10 0.12 0.14 0.16)
wells=(OFF 0 -100)
binary_values=(0 1)
# Restart the identity devices only.
mono_values=(1)
configs=(0 1 2 3)
# Vessel geometry: 0/1/2 = pill-pipe / renaissance / torus; 3 = constant-radius
# helical; 4 = variable-radius helical; >=5000 = welded-pipe helical (discrete
# centreline nodes), id ENCODES the segment count nseg = vessel_id - 5000. Must
# match mk_tasks.sh so restarts cover the same device set.
vessel_values=(0 1 2 3 4 5004 5008 5012)
# Null type: DN = double-null (stellsym); SN = single-null.
null_values=(DN SN)
# Aspect-ratio / on-axis-iota knob (part of the device identity / folder name).
AR_values=(0 1 2)
ATTEMPTS=2

mkdir -p logs
: > tasks_restart.jobs

# Emit raw parameter tuples (one per line); a single python pass turns each tuple into the
# inferred sharded json path (existence-checked) and the prefix_restart.sh command.
{
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
                      echo "${margin} ${well} ${Z} ${distance} ${on_vessel} ${config} ${vessel_id} ${mono} ${null} ${AR} ${attempt}"
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
} | NUM_AUX="$NUM_AUX" python3 -c '
import sys, os, hashlib, zlib

num_aux = os.environ["NUM_AUX"]
n_found = n_missing = 0
for line in sys.stdin:
    f = line.split()
    if not f:
        continue
    margin, well, Z, distance, on_vessel, config, vessel_id, mono, null, AR, attempt = f
    # Replicate boozer_all.py byte-for-byte (the init device folder is num_aux=0).
    margin_str = f"{float(margin):.2f}".replace(".", "p")
    well_str = "OFF" if well == "OFF" else str(float(well))
    name = (f"margin={margin_str}_well={well_str}_Z={Z}_onvessel={on_vessel}"
            f"_distance={distance}_configID={config}_vesselID={vessel_id}"
            f"_mono={mono}_null={null}_num_aux=0_AR={AR}_attempt={attempt}")
    shard = hashlib.md5(name.encode()).hexdigest()[:2]
    dev_id = zlib.crc32(name.encode())
    json = f"output/{shard}/{name}/design_opt_final_{dev_id}.json"
    # Only relaunch devices that actually exist on ceph (boozer_all keeps just the ones
    # passing its gate); a missing json would otherwise trigger a full boozer_all recompute.
    if not os.path.exists(json):
        n_missing += 1
        continue
    n_found += 1
    # prefix_restart.sh signature: margin well Z distance on_vessel config vessel_id mono
    #                              null NUM_AUX AR attempt
    print(f"bash ./prefix_restart.sh {margin} {well} {Z} {distance} {on_vessel} "
          f"{config} {vessel_id} {mono} {null} {num_aux} {AR} {attempt}")
sys.stderr.write(f"mk_tasks_restart: {n_found} mono=1 devices found, {n_missing} combinations skipped (no json)\n")
' >> tasks_restart.jobs

echo "wrote $(wc -l < tasks_restart.jobs) tasks to tasks_restart.jobs (mono=1, num_aux=${NUM_AUX})"
