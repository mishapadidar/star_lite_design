"""Folder layout of one combined-stage run (one optimization stage).

    <outdir>/config_used.yaml, history.yaml     what was run and what it reached (top level)
    <outdir>/state/     checkpoint.npz, al_state.yaml, scaling_D.npy, coil_weights*.yaml   (--resume state)
    <outdir>/coils/     coils_<tag>.json
    <outdir>/xpoints/   xpoint_<tag>.json
    <outdir>/inputs/    input.combined_stage_<tag>   (VMEX boundary namelists)
    <outdir>/wout/      wout_combined_stage_final.nc
    <outdir>/logs/      run.log, SLURM logs
    <outdir>/snapshots/ snapshot_<n>.json + snapshots.yaml: coils, VMEX boundary, magnetic axis and X-point line at the
                        stage start, every `history_every` L-BFGS-B iterations, each outer iteration's end and the end
    <outdir>/figures/   every figure of the run (including the device figures)
    <outdir>/device/    array-campaign device data: design archive, summary.txt, LCFS surfaces, cross-section data

Runs written before 2026-09-14 kept every file at the top level; :func:`find` and :func:`matches` read both layouts,
so resuming or post-processing an old run works unchanged.
"""
import glob
import os

STATE_FILES = ("checkpoint.npz", "al_state.yaml", "scaling_D.npy")


def kind_of(name):
    """Subfolder of a run file (None: top level)."""
    if name.startswith("snapshot_") or name == "snapshots.yaml":
        return "snapshots"
    if name in STATE_FILES or name.startswith("coil_weights"):
        return "state"
    if name.startswith("coils_") and name.endswith(".json"):
        return "coils"
    if name.startswith("xpoint_") and name.endswith(".json"):
        return "xpoints"
    if name.startswith("input."):
        return "inputs"
    if name.startswith("wout_") and name.endswith(".nc"):
        return "wout"
    if name == "run.log" or name.endswith((".out", ".err", ".log")):
        return "logs"
    if name.endswith(".png"):
        return "figures"
    return None


def path(outdir, name, create=False):
    """Where ``name`` (a file name or file-name glob) belongs in the current layout."""
    kind = kind_of(name)
    folder = outdir if kind is None else os.path.join(outdir, kind)
    if create:
        os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name)


def find(outdir, name):
    """The existing ``name`` in either layout (current first); the current-layout path if it exists in neither."""
    current = path(outdir, name)
    if os.path.exists(current):
        return current
    legacy = os.path.join(outdir, name)
    return legacy if os.path.exists(legacy) else current


def matches(outdir, pattern):
    """Sorted files matching the file-name glob ``pattern``, from the current layout if any exist there."""
    current = sorted(glob.glob(path(outdir, pattern)))
    return current if current else sorted(glob.glob(os.path.join(outdir, pattern)))
