#!/bin/bash
# Emit the disBatch task list for the UNPOLISHED (with-aux, no optimization) workflow.
#
# Unlike mk_tasks.sh (which runs the full boozer_all -> polish -> render pipeline via
# prefix.sh), this emits one prefix_singular.sh line per EXISTING boozer_all init
# device (num_aux=0): each line computes NUM_AUX auxiliary coils for that device with
# the trace/identity monodromy enforced exactly, then renders xs plots (+ LCFS overlay),
# device views, and appends elongation. No boozer_all run, no coil optimization.
#
# The device json path is INFERRED from the parameter combination -- no globbing --
# mirroring how boozer_all.py named the folder and how prefix.sh sharded it on ceph:
#   output/<shard>/<TASK_NAME>/design_opt_final_<DEVICE_ID>.json
# where TASK_NAME is boozer_all.py's folder name at num_aux=0,
#       DEVICE_ID = zlib.crc32(TASK_NAME)            (matches boozer_all.py), and
#       shard     = first two hex chars of md5(TASK_NAME)  (matches prefix.sh's
#                   `printf '%s' "$name" | md5sum | cut -c1-2`).
# The name/shard/id are resolved in a SINGLE python pass (byte-for-byte identical to
# boozer_all.py / prefix.sh) so the ~10^4 combinations don't each spawn python+md5sum.
#
# Only mono=1 (identity) and mono=2 (trace) devices are emitted: mono=0 has no monodromy
# constraint, so boozer_singular.py has no auxiliary-coil system to solve. The matching
# design_opt_final_*.yaml is ASSUMED present next to each json (boozer_singular.py reads
# the monodromy constraint from it); a combination whose inferred json is absent is
# skipped, so the list only contains devices that were actually produced.

NUM_AUX=5
margins=(0.06 0.08 0.10 0.12 0.14 0.16)
wells=(OFF 0 -100)
binary_values=(0 1)
# Only the monodromy devices: mono=1 -> identity, mono=2 -> trace. mono=0 is excluded.
mono_values=(1 2)
configs=(0 1 2 3)
# Vessel geometry: 0/1/2 = pill-pipe / renaissance / torus; 3 = constant-radius
# helical; 4 = variable-radius helical; >=5000 = welded-pipe helical (discrete
# centreline nodes), id ENCODES the segment count nseg = vessel_id - 5000. Must
# match mk_tasks.sh so the singular-polish sweep covers the same device set.
vessel_values=(0 1 2 3 4 5004 5008 5012)
# Null type: DN = double-null (stellsym); SN = single-null.
null_values=(DN SN)
# Aspect-ratio / on-axis-iota knob (part of the device identity / folder name).
AR_values=(0 1 2)
ATTEMPTS=2

mkdir -p logs
: > tasks_singular.jobs

# Emit raw parameter tuples (one per line); a single python pass turns each tuple into
# the inferred sharded json path and the prefix_singular.sh command.
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
    # The device must actually exist (boozer_all only keeps devices passing its gate);
    # skip combinations whose json is absent so the list has no guaranteed-fail tasks.
    if not os.path.exists(json):
        n_missing += 1
        continue
    n_found += 1
    print(f"bash ./prefix_singular.sh {json} {num_aux}")
sys.stderr.write(f"mk_tasks_singular: {n_found} devices found, {n_missing} combinations skipped (no json)\n")
' >> tasks_singular.jobs

echo "wrote $(wc -l < tasks_singular.jobs) tasks to tasks_singular.jobs (num_aux=${NUM_AUX})"
