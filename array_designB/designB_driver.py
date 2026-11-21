#!/usr/bin/env python3
"""designB campaign driver (embedded disBatch): the boozer step of every device, then one
polish task per num_aux for each device that passes, all in ONE disBatch run.

Run inside a Slurm allocation, with the disBatch/beta module loaded before sbatch (its
PYTHONPATH provides `disbatchc`, and sbatch exports the environment):

    module load disBatch/beta
    sbatch -p ccm -N 30 --ntasks-per-node=64 -c 1 -t 7-00:00:00 -J designB \
           -o logs/designB_driver_%j.out --wrap "python3 designB_driver.py tasks.jobs"

tasks.jobs (written by mk_tasks.sh) holds one 'bash ./prefix.sh <15 args>' line per device,
with step=boozer (15th argument). disBatch runs these lines on the allocated cores. Each
time a boozer task returns 0 (device accepted and copied to ceph), the driver submits the
same line with num_aux (10th argument) set to N and step set to polish and to double
(--steps), once per N in --num-aux: for each N, the polish (boozer_singular_opt.py) and the
two-configuration optimization (boozer_double_opt.py) of the device run side by side, as
separate tasks that start as soon as cores are free, on any node. Once every boozer
task has returned, the driver tells disBatch that no more tasks are coming: nodes are then
retired as they fall idle, and disBatch exits when the last polish returns.

Every submitted and every returned task is appended to designB_driver_disBatch_<jobid>.csv;
the disBatch files (status, logs, dbUtil) are prefixed with designB_driver_disBatch_<jobid>.

--resume (e.g. after a cancelled run): a device whose num_aux=0 device is in output/
(accepted) only gets the polishes whose log does not show they ran to their end; a device
whose boozer log shows it ran to its end (rejected, crashed) is skipped; any other device
(no log, or a task killed mid-way) gets its boozer task again.
--dry-run: print what would be submitted, without starting disBatch.
Any other option is passed on to disBatch.
"""
import argparse
import csv
import glob
import hashlib
import os
import random
import time

# prefix.sh positional arguments, 0-based here:
#   0 margin  1 well  2 Z  3 distance  4 on_vessel  5 config  6 vessel_id  7 mono  8 null
#   9 num_aux  10 AR  11 attempt  12 force  13 mono_thr  14 step  15 elongation  16 extra coils
#   17 geom (overridden coil-geometry / clearance bounds, e.g. len=4.5,kap=10,msc=20,clr=0.07, or -)
# (a 19th, trace_min, then ec_max, target and pf_max, are appended by make_cmd() to the
# double tasks)
NUM_AUX, ATTEMPT, STEP, ELONG, EXTRA, GEOM = 9, 11, 14, 15, 16, 17
N_ARGS = 18

# Messages that prefix.sh prints when a step ends (any outcome), searched in the tail of
# the task log. A task killed mid-way (job cancelled) has none of them.
BOOZER_END = (b"> 0.1%, aborting", b"boozer_all.py failed", b"max_rel_error.txt not found",
              b"init render failed", b"Finished: ")
POLISH_END = (b"succeeded (max rel err", b"out of spec (max rel err",
              b"(no design_polished_final", b"polished render failed", b"device not accepted?",
              b"Finished: ")
# a double task (prefix.sh step=double) prints the same succeeded / out of spec messages
DOUBLE_END = (b"succeeded (max rel err", b"out of spec (max rel err",
              b"(no design_double_final", b"device not accepted?", b"Finished: ")
# Steps submitted for each accepted device and each num_aux, with the suffix prefix.sh adds
# to the num_aux=N name of the task (scratch dir, log) and the end markers of its log.
POST_STEPS = {"polish": ("", POLISH_END), "double": ("_double", DOUBLE_END)}
TAIL_BYTES = 1 << 16


# A double task run with its own bound on tr M of the config 1 X-point (--trace-min) is the
# step "double@<bound>", e.g. "double@2.2": prefix.sh gets step=double and the bound as its
# 19th argument, and names the task <num_aux=N name>_trmin=<bound>_double.
# With a bound on E[C] as well (--ec-max, a number or OFF) the step is "double@<bound>@<ec>",
# e.g. "double@2.05@0.8": the E[C] bound is prefix.sh's 20th argument and the task is named
# <num_aux=N name>_trmin=<bound>_ec=<ec>_double.
def step_base(step):
    return step.partition("@")[0]


# With a target as well (--target double | snowflake) the step is "double@<bound>@<ec>@<target>":
# the target is prefix.sh's 21st argument, and a snowflake task (the snowflake configuration
# alone) gets the field _target=snowflake before _double; the name of a double is unchanged.
# With a bound on the aux (PF) coil currents as well (--pf-max, amperes) the step is
# "double@<bound>@<ec>@<target>@<pf>": the bound is prefix.sh's 22nd argument and the name gets
# the field _pf=<amperes> after the E[C] field.
def step_parts(step):
    """(base, trace bound or '', E[C] bound or '', target or '', PF current bound or '') of a step."""
    p = step.split("@") + ["", "", "", ""]
    return p[0], p[1], p[2], p[3], p[4]


def step_suffix(step):
    """What prefix.sh appends to the num_aux=N name of the task (scratch dir, log)."""
    base, tmin, ec, target, pf = step_parts(step)
    if not tmin:
        return POST_STEPS[base][0]
    # a relative E[C] task (ec = f<factor>) is named ..._ec=f<factor>_level=0_... (prefix.sh's EC_FIELD)
    ec_field = ("_ec=%s" % ec + ("_level=0" if ec.startswith("f") else "")) if ec else ""
    return "_trmin=%s%s%s%s_double" % (tmin, ec_field, "_pf=%s" % pf if pf else "",
                                       "_target=snowflake" if target == "snowflake" else "")


def log(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S ") + msg, flush=True)


def split_cmd(cmd):
    """(head, args) of a task line: head = the tokens up to and including prefix.sh,
    args = its 18 positional arguments (a line without the 16th, the elongation bound, gets
    OFF; a line without the 17th, the extra coils, gets none; a line without the 18th, the
    geometry bounds, gets -)."""
    toks = cmd.split()
    for k, t in enumerate(toks):
        if t.endswith("prefix.sh"):
            args = toks[k + 1:]
            if len(args) == N_ARGS - 3:
                args.append("OFF")
            if len(args) == N_ARGS - 2:
                args.append("none")
            if len(args) == N_ARGS - 1:
                args.append("-")
            if len(args) != N_ARGS:
                raise SystemExit("expected %d prefix.sh arguments, got %d in: %s" % (N_ARGS, len(args), cmd))
            return toks[:k + 1], args
    raise SystemExit("no prefix.sh in task line: %s" % cmd)


def make_cmd(head, args, num_aux, step):
    base, tmin, ec, target, pf = step_parts(step)
    a = list(args)
    a[NUM_AUX], a[STEP] = str(num_aux), base
    return " ".join(head + a + ([tmin] if tmin else []) + ([ec] if ec else []) + ([target] if target else [])
                    + ([pf] if pf else []))


def well_str(w):
    return "OFF" if w == "OFF" else str(float(w))


def geom_field(g):
    """prefix.sh's GEOM_FIELD: _len=.._kap=.._msc=.._clr=.. (in that order, %g) of a geom
    argument, '' for -."""
    if g == "-":
        return ""
    d = dict(kv.split("=", 1) for kv in g.split(","))
    return "".join("_%s=%g" % (k, float(d[k])) for k in ("len", "kap", "msc", "clr") if k in d)


def task_name(args, num_aux):
    """Device folder name, as built by prefix.sh's task_name() (must match it exactly)."""
    a = args
    return ("margin=%s_well=%s_Z=%s_onvessel=%s_distance=%s_configID=%s_vesselID=%s_mono=%s"
            "_null=%s_num_aux=%s_AR=%s_attempt=%s_force=%s_monothr=%s_elong=%s%s%s"
            % (("%.2f" % float(a[0])).replace(".", "p"), well_str(a[1]), a[2], a[4], a[3], a[5],
               a[6], a[7], a[8], num_aux, a[10], a[11], a[12], a[13], a[ELONG],
               "" if a[EXTRA] == "none" else "_extra=%s" % a[EXTRA], geom_field(a[GEOM])))


def shard(name):
    """prefix.sh's shard(): first 2 hex digits of the md5 of the folder name."""
    return hashlib.md5(name.encode()).hexdigest()[:2]


def log_path(name):
    return os.path.join("logs", shard(name), name + ".out")


def ran_to_end(path, markers):
    """True if the log at path exists and its tail holds one of the end markers."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - TAIL_BYTES))
            tail = f.read()
    except OSError:
        return False
    return any(m in tail for m in markers)


def accepted_on_ceph(args):
    """True if the num_aux=0 device was accepted and copied to output/ by its boozer task
    (its design_opt_final json and paired yaml, the polish inputs, are there)."""
    n0 = task_name(args, 0)
    jsons = glob.glob(os.path.join("output", shard(n0), n0, "design_opt_final_*.json"))
    return any(os.path.isfile(j[:-len(".json")] + ".yaml") for j in jsons)


def post_tasks(head, args, num_aux_list, steps):
    """(step, head, args, num_aux) of the tasks that follow an accepted boozer task: for each
    num_aux, one task per step (polish, double), so that they run side by side."""
    tasks = [(step, head, args, n) for n in num_aux_list for step in steps]
    # random order, so that the cores free at a given time are shared between every N, bound
    # and target instead of going to the first combinations of the list
    random.shuffle(tasks)
    return tasks


def initial_tasks(devices, num_aux_list, steps, resume):
    """(step, head, args, num_aux) of the tasks to submit at start, and the number of
    devices skipped by --resume."""
    tasks, skipped = [], 0
    for head, args in devices:
        if not resume:
            tasks.append(("boozer", head, args, 0))
        elif accepted_on_ceph(args):
            todo = [t for t in post_tasks(head, args, num_aux_list, steps)
                    if not ran_to_end(log_path(task_name(args, t[3]) + step_suffix(t[0])),
                                      POST_STEPS[step_base(t[0])][1])]
            tasks += todo
            skipped += not todo
        elif ran_to_end(log_path(task_name(args, 0)), BOOZER_END):
            skipped += 1
        else:
            tasks.append(("boozer", head, args, 0))
    return tasks, skipped


def main():
    ap = argparse.ArgumentParser(
        description="Embedded-disBatch driver of a designB campaign: boozer step of every "
                    "device, then one polish task per num_aux for each accepted device.")
    ap.add_argument("tasks", help="task file written by mk_tasks.sh (step=boozer lines)")
    ap.add_argument("--num-aux", type=int, nargs="+", default=[3, 4, 5, 6],
                    help="numbers of aux coils to polish each accepted device at (default: 3 4 5 6)")
    ap.add_argument("--steps", nargs="+", default=["polish", "double"], choices=sorted(POST_STEPS),
                    help="steps run for each accepted device and each num_aux: polish "
                         "(boozer_singular_opt.py) and/or double (boozer_double_opt.py), as "
                         "separate tasks side by side (default: polish double)")
    ap.add_argument("--trace-min", nargs="+", default=[], metavar="BOUND",
                    help="lower bounds on tr M of the config 1 X-point for the double step: one "
                         "double task per bound (boozer_double_opt.py --trace-min), named "
                         "..._trmin=<bound>_double (default: one double with the script's own "
                         "bound, named ..._double)")
    ap.add_argument("--ec-max", nargs="+", default=[], metavar="BOUND",
                    help="upper bounds on E[C], the mean unfolding constant of the snowflake, for "
                         "the double step: one double task per trace bound and per E[C] bound "
                         "(boozer_risk_opt.py --ec-max; OFF = no bound), named "
                         "..._trmin=<bound>_ec=<bound>_double; f<factor> (e.g. f0.9) = the RELATIVE "
                         "continuation (boozer_risk_opt.py --ec-factor: each level bounded by factor x "
                         "the E[C] reached by the previous one), named ..._ec=f<factor>_level=0_... "
                         "(its next levels _level=1, 2, ...). Needs --trace-min")
    ap.add_argument("--target", nargs="+", default=[], choices=["double", "snowflake"],
                    help="what the double step optimizes (boozer_risk_opt.py --target): double = both "
                         "configurations, snowflake = the snowflake configuration alone, named "
                         "..._target=snowflake_double; one task per target given. Needs --ec-max")
    ap.add_argument("--pf-max", nargs="+", default=[], metavar="AMPERES",
                    help="bounds on the current of each auxiliary (PF) coil, in amperes, for the double "
                         "step (boozer_risk_opt.py --aux-current-max): one task per bound, named with "
                         "the field _pf=<amperes>. Needs --target")
    ap.add_argument("--resume", action="store_true",
                    help="skip the work whose logs/outputs show it is done (see the module doc)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be submitted, without starting disBatch")
    opts, db_args = ap.parse_known_args()
    num_aux_list = opts.num_aux
    if min(num_aux_list) < 1 or len(set(num_aux_list)) != len(num_aux_list):
        raise SystemExit("--num-aux needs distinct positive integers, got %s" % num_aux_list)

    devices = []
    with open(opts.tasks) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            head, args = split_cmd(line.strip())
            if args[STEP] != "boozer":
                raise SystemExit("expected step=boozer task lines, got: %s" % line.strip())
            devices.append((head, args))

    steps = list(dict.fromkeys(opts.steps))   # distinct, in the order given
    # with --trace-min, the double step becomes one "double@<bound>" step per bound
    for b in opts.trace_min:
        float(b)   # a number, kept as typed: it goes into the task name
    # with --ec-max as well, one "double@<bound>@<ec>" step per (trace bound, E[C] bound)
    for e in opts.ec_max:
        if e != "OFF":
            float(e[1:] if e.startswith("f") else e)   # a number, f<factor> or OFF, kept as typed: it goes into the task name
    if opts.ec_max and not opts.trace_min:
        raise SystemExit("--ec-max needs --trace-min (the E[C] bound follows the trace bound "
                         "in prefix.sh's arguments)")
    # with --target as well, one "double@<bound>@<ec>@<target>" step per target
    if opts.target and not opts.ec_max:
        raise SystemExit("--target needs --ec-max (the target follows the E[C] bound in "
                         "prefix.sh's arguments; use --ec-max OFF for no bound)")
    # with --pf-max as well, one "double@<bound>@<ec>@<target>@<pf>" step per PF current bound
    for a in opts.pf_max:
        float(a)   # amperes, kept as typed: it goes into the task name
    if opts.pf_max and not opts.target:
        raise SystemExit("--pf-max needs --target (the PF bound follows the target in prefix.sh's "
                         "arguments; use --target double for the two-configuration optimization)")
    if opts.trace_min and "double" in steps:
        k = steps.index("double")
        steps[k:k + 1] = ["double@" + b + ("@" + e if e else "") + ("@" + t if t else "") + ("@" + a if a else "")
                          for b in dict.fromkeys(opts.trace_min)
                          for e in (dict.fromkeys(opts.ec_max) or [""])
                          for t in (dict.fromkeys(opts.target) or [""])
                          for a in (dict.fromkeys(opts.pf_max) or [""])]
    start, skipped = initial_tasks(devices, num_aux_list, steps, opts.resume)
    n_boozer = sum(1 for t in start if t[0] == "boozer")
    log("%d devices in %s: %d boozer tasks and %d %s tasks to submit now, %d devices "
        "skipped (resume=%s, num_aux=%s, steps after boozer=%s)"
        % (len(devices), opts.tasks, n_boozer, len(start) - n_boozer, "/".join(steps), skipped,
           opts.resume, num_aux_list, steps))
    if opts.dry_run:
        for step in ["boozer"] + steps:
            cmds = [make_cmd(h, a, n, s) for s, h, a, n in start if s == step]
            log("%s: %d tasks%s" % (step, len(cmds), ", first ones:" if cmds else ""))
            for c in cmds[:3]:
                print("    " + c)
        return

    from disbatchc import disBatch  # provided by the disBatch/beta module

    jobid = os.environ.get("SLURM_JOB_ID", "local")
    # The name holds "_disBatch_" so that the staging rsync of prefix.sh (--exclude '*_disBatch_*')
    # skips these files, which grow during the run, as it does for a plain disBatch run.
    prefix = "designB_driver_disBatch_%s" % jobid
    db = disBatch.DisBatcher(tasksname="designB", args=["-p", prefix] + db_args)

    csv_file = open(prefix + ".csv", "a", newline="")
    out = csv.writer(csv_file)
    out.writerow(["time", "event", "task_id", "step", "attempt", "num_aux", "return_code", "cmd"])
    tasks = {}  # disBatch task id (0, 1, 2, ... in submission order) -> (step, head, args, num_aux)

    def record(event, tid, rc=""):
        # cmd makes the CSV self-contained: the monitor reads the tasks of the run from it.
        step, head, args, num_aux = tasks[tid]
        out.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event, tid, step, args[ATTEMPT], num_aux, rc,
                      make_cmd(head, args, num_aux, step)])
        csv_file.flush()

    def submit(step, head, args, num_aux):
        tid = len(tasks)
        db.submit(make_cmd(head, args, num_aux, step))
        tasks[tid] = (step, head, args, num_aux)
        record("submit", tid)

    done_sent = False

    def no_more_tasks():
        # Same message as DisBatcher.done(), but without joining the disBatch thread, so
        # that the results of the remaining polish tasks can still be read and recorded.
        nonlocal done_sent
        if not done_sent:
            db.kvs.put(db.taskkey, db.donetask, False)
            done_sent = True
            log("all boozer tasks returned: told disBatch that no more tasks are coming")

    try:
        for t in start:
            submit(*t)
        boozer_left, accepted, pending = n_boozer, 0, len(start)
        if boozer_left == 0:
            no_more_tasks()
        while pending:
            try:
                s = db.wait_one_task()
            except Exception as e:  # e.g. disBatch already shut its KVS server down
                if not done_sent:
                    raise
                log("stopped reading task results after the end of the run (%r)" % e)
                break
            pending -= 1
            tid, rc = s["TaskId"], s["ReturnCode"]
            step, head, args, num_aux = tasks[tid]
            record("return", tid, rc)
            if step == "boozer":
                boozer_left -= 1
                if rc == 0:
                    accepted += 1
                    for t in post_tasks(head, args, num_aux_list, steps):
                        submit(*t)
                        pending += 1
            log("task %d (%s, attempt %s, num_aux %s) returned %s | boozer left %d, accepted %d, "
                "tasks pending %d" % (tid, step, args[ATTEMPT], num_aux, rc, boozer_left, accepted, pending))
            if boozer_left == 0:
                no_more_tasks()
    finally:
        # Also on an error in the driver: let disBatch finish the running tasks and exit.
        no_more_tasks()
        if db.db_thread is not None:
            db.db_thread.join()
        csv_file.close()
    log("disBatch run finished")


if __name__ == "__main__":
    main()
