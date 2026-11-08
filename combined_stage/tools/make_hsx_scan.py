#!/usr/bin/env python3
"""HSX-class device scan (2026-10-18): one continuation ladder per device, every objective on.

Grid (64 devices): coils {16, 24, 32, 40} x final iota floor {0.30, 0.37, 0.42, 0.50} x aspect {6, 8} x well floor
{1 %, 3 %}. Each ladder is ladders/hsx_nfp4_far.yaml (circular coils at 3.5 a, conformal vessel room: coils >= 0.22 m
from the plasma and both X-lines, double null, beta 1 % with virtual casing + G + B.n band 3e-3) with the full objective
set of array/boozer_all.py where it maps onto the combined stage:
    QS, iota floor, aspect, magnetic well (depth >= floor, V'' <= 0), |B| ripple on the axis (1e-3, weight 1e2),
    coil ArclengthVariation (1e-3: boozer_all's 100 reshaped converged coils, run *_arclen100), X-line flatness
    (boozer_all --Z, user requirement: |z - mean z| <= 10 mm escalated + a pull toward flat, from dn_converge on; off
    while the X-line is first made a field line in dn_introduce), coil room for the vessel,
    and at finite beta the Mercier criterion (weight 1e-4, escalated).
Well and ripple come in at max_mode 3 (stage 2); flatness after the X-line exists (dn_converge); Mercier at beta 0.3 %
then 1 %.
v2 (2026-10-19, after v1 in runs/combined_stage/archive_scan_v1): v1 never met the iota floor in the vacuum stages
(weight 10, not escalated) and paid for it in QS (1-2.7e-2 -> 8-10e-2) once dn_introduce forced iota; X-lines left
dn_converge 3-9 cm from flat (flatness 10 / pull 10, 25 outers); devices entered beta 1 % Mercier-unstable (cost ~1e4)
and four beta seeds / solves failed on a sign-changed Jacobian. v2: iota escalated from stage 0, flatness 1e3 / pull
1e2 over 40 outers, boundary regularity from dn_converge on, Mercier 1e-4 and a beta 0.3 % stage before 1 %.
v2.1 (2026-10-19 evening): boundary regularity also in dn_introduce. Without it the X-line squeezed the boundary to the
boundary_guard floor (speed ratio 0.05) in half the W7-X and some HSX devices, and dn_converge could not cold-solve that
boundary (INITIAL JACOBIAN CHANGED SIGN; not fixed by the axis guess, multigrid or a smaller DELT). Failed devices are
resubmitted from dn_introduce (old stage dirs kept as <stage>_v2bad).
v3 (2026-10-20): the beta stages and dn_converge froze (B.n 2-14 % at beta 1 %, unchanged over 40 outers) under the
x10/outer escalation of unmeetable limits (weights 1e7-1e8). Escalation now capped at 100x and the B.n block starts at
penalty 1e5 (band 1e4), max 1e8, in dn_converge and the beta stages. Devices that had reached dn_converge rerun from it
(old stage dirs kept as <stage>_v2stall).
v3.1 (2026-10-21): VMEX accepts an unconverged solve only up to fsq/ftol 1e3 (default 1e6). At 1e6 the X-line stages
walked onto boundaries VMEX cannot converge (27 of 1585 stage finals at fsq/ftol 1e4-9e5, ier_flag 2) and finished
there; the next stage's first solve of that boundary then failed (status 2, NaN boundary gradient, implausible virtual
casing): 20 of the 27 failed v3 jobs. Converged solves end at fsq/ftol ~1-2. Stages that produced an unconverged final
are redone (old dirs <stage>_v3unconv).
v3.2 (2026-10-27): the beta stages solve VMEX at ns = 51 (was 16) and the Mercier term counts s >= 0.1 only
(plasma_objective.mercier.s_min). At ns = 16 DMerc is unconverged: re-solving all 646 finished beta rungs at ns = 51
flipped the verdict for 107 and found 78 % unstable, typically over most of the radius (runs/vmex/mercier_survey).
~2x the seconds per evaluation in those stages. Finished ladders get the same via tools/mercier_polish.sbatch
(rung 08_mercier_polish).

Families (--family, 2026-10-19):
    hsx      the grid above (v2), runs/combined_stage/hsx_scan
    hsx_ext  128 more HSX-class devices: the same coils x iota {0.25, 0.33, 0.40, 0.46} x aspect {5, 7, 9, 10} x well,
             runs/combined_stage/hsx_scan_ext (seeds input.qa_nfp4_R1p2_A<A>, a = 1.2 m / A)
    w7x      64 W7-X-size QA double-null devices (user, 2026-10-19: HSX-like and W7-X-like reactors, both QA with a DN
             non-resonant divertor; the size class, not the QI concept): nfp 5, R0 5.5 m, B0 2.5 T, coils {30, 40, 50,
             60} x iota {0.45, 0.60, 0.75, 0.90} x aspect {8, 10} x well, from the HSX-style rotating-ellipse seeds
             seeds/input.qa_nfp5_R5p5_A{8,10} and the same ladder. Every length-valued engineering setting x 3.5 (a 0.52
             vs 0.15 m): coils >= 0.77 m from plasma and X-lines, coil-coil >= 0.28 m, curvature <= 2.86 /m, MSC <= 1.63,
             X-line 14-42 cm outside the plasma (initial offset 21 cm); flatness <= 10 mm kept absolute. Earlier W7-X
             runs (QI, R0 1.5 and 2.0 m) are archived in runs/combined_stage/archive_scan_v1/w7x_scan_R1p5 / _R2p0.

usage: make_hsx_scan.py [--family hsx|hsx_ext|w7x] [--submit] [--only NAME ...] [--stagger-min 2]
Writes ladders/<family dir>/<name>.yaml and, with --submit, one sbatch/ladder.sbatch job per device into
runs/combined_stage/<family dir>/<name>/ (submissions staggered: every Python start reads JAX's ~1000 .pyc files over NFS,
and a burst of them stalled cn01 / cn08 on 2026-10-17/18).
"""
import argparse
import itertools
import math
import re
import os
import subprocess

import yaml

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = "/data/common/raffael.wendlinger/simflare/runs/combined_stage"
# base coils per half period, length limit (m), current limit (A): the circle at 3.5 a is 3.3 m (A 8) / 4.4 m (A 6)
# long, and B0 = 1 T needs 2 pi R0 B0 / (mu0 N) = 7.5 MA / N, limited at 4/3 of that
COILS = {16: (2, 7.0, 500000.0), 24: (3, 6.5, 333334.0), 32: (4, 6.0, 250000.0), 40: (5, 5.5, 200000.0)}
WELLS = (0.01, 0.03)
# W7-X size, nfp 5, R0 5.5 m, B0 2.5 T, a 0.55 (A 10) / 0.69 m (A 8): 2 pi R0 B0 / mu0 = 68.75 MA in total, limited at
# 4/3 of the share per coil (50 coils: 1.38 MA, about the real W7-X amp-turns); length limits = the HSX ones x 3.5
COILS_W7X = {30: (3, 26.0, 3056000.0), 40: (4, 24.0, 2292000.0), 50: (5, 22.5, 1834000.0), 60: (6, 21.0, 1528000.0)}
# every length-valued engineering setting of the HSX family (a ~ 0.15 m) x 3.5 for the W7-X size (a 0.52 m); flatness
# (<= 10 mm) stays absolute (user, 2026-10-19)
W7X_SCALE = 3.5
FAMILIES = {
    "hsx": dict(dir="hsx_scan", job="hs", coils=COILS, iotas=(0.30, 0.37, 0.42, 0.50), aspects=(6, 8), qi=False,
                seed=lambda aspect: f"{RUNS}/seeds/input.qa_nfp4_R1p2_A{aspect}"),
    "hsx_ext": dict(dir="hsx_scan_ext", job="hx", coils=COILS, iotas=(0.25, 0.33, 0.40, 0.46), aspects=(5, 7, 9, 10),
                    qi=False, seed=lambda aspect: f"{RUNS}/seeds/input.qa_nfp4_R1p2_A{aspect}"),
    "w7x": dict(dir="w7x_scan", job="w7", coils=COILS_W7X, iotas=(0.45, 0.60, 0.75, 0.90), aspects=(8, 10), qi=False,
                scale=W7X_SCALE, seed=lambda aspect: f"{RUNS}/seeds/input.qa_nfp5_R5p5_A{aspect}"),
    # v4 test (2026-10-22, user): blanket/divertor room -- coils >= 0.4 m (HSX) / 0.6 m (W7-X) from the plasma AND the
    # X-lines; length, curvature, coil-coil and both clearances as hard augmented-Lagrangian limits (coils.hard)
    # instead of escalated weights; coil length <= 2 x the circumference 2 pi (a + clearance) of the smallest loop at
    # that clearance (all v3 coils sat at their 2.5-3x cap, spent on wiggles); circular seed coils at a + 1.25 x
    # clearance. Six of the best v3 devices.
    "hsx_v4": dict(dir="hsx_v4_test", job="h4", coils=COILS, qi=False, R0=1.2, clearance=0.4, length_factor=2.0,
                   seed=lambda aspect: f"{RUNS}/seeds/input.qa_nfp4_R1p2_A{aspect}",
                   devices=[(24, 0.25, 10, 0.03), (32, 0.25, 10, 0.01), (24, 0.25, 10, 0.01), (24, 0.37, 8, 0.01)]),
    "w7x_v4": dict(dir="w7x_v4_test", job="w4", coils=COILS_W7X, qi=False, scale=W7X_SCALE, R0=5.5, clearance=0.6,
                   length_factor=2.0, seed=lambda aspect: f"{RUNS}/seeds/input.qa_nfp5_R5p5_A{aspect}",
                   devices=[(40, 0.60, 8, 0.03), (50, 0.45, 10, 0.03)]),
    # vessel scan (2026-10-22, user: "like Andrew did it for the star_lite dataset", focus on the welded-pipe vessel):
    # array/boozer_all.py's vessel inside the optimization from dn_introduce on -- plasma and X-line >= margin inside,
    # coils >= clearance outside, X-line at a constant gap to it (--distance) -- for every vessel family that fits an
    # nfp 4/5 device (PillPipe / Rennaissance are 2-fold Star_Lite shapes): helical tube with constant (vessel_id 3)
    # and variable radius (4), torus (2), and welded pipes (5000 + nseg) with 2-8 legs per field period. v4 coil
    # settings (hard limits, 0.4 / 0.6 m coil clearance, length 2x the smallest loop), fresh seeds.
    "hsx_vessel": dict(dir="vessel_scan", job="vh", coils={**COILS, 48: (6, 5.0, 166667.0)}, qi=False, R0=1.2,
                       clearance=0.4, length_factor=2.0, vessel_margin=0.10, vessel_clearance=0.15,
                       seed=lambda aspect: f"{RUNS}/seeds/input.qa_nfp4_R1p2_A{aspect}",
                       devices=[(24, 0.25, 10, 0.03), (32, 0.25, 10, 0.01), (24, 0.25, 10, 0.01), (24, 0.37, 8, 0.01)],
                       vessels=["helical_r0", "helical_r2", "torus"] + [f"pipe{n}" for n in (8, 12, 16, 20, 24, 28, 32)],
                       extra=[((48, 0.25, 10, 0.03), [f"pipe{n}" for n in (12, 16, 20, 24)])]),
    "w7x_vessel": dict(dir="vessel_scan", job="vw", coils=COILS_W7X, qi=False, scale=W7X_SCALE, R0=5.5, clearance=0.6,
                       length_factor=2.0, vessel_margin=0.15, vessel_clearance=0.20,
                       seed=lambda aspect: f"{RUNS}/seeds/input.qa_nfp5_R5p5_A{aspect}",
                       devices=[(40, 0.60, 8, 0.03), (50, 0.45, 10, 0.03)],
                       vessels=["helical_r0", "helical_r2", "torus"] + [f"pipe{n}" for n in (10, 15, 20, 25, 30, 35, 40)]),
    # vessel scan 2 (2026-10-24, user): coil length <= 1.6 x the smallest loop (was 2x); coil-plasma / coil-X-line as
    # closest-approach rows (ClosestApproachConstraint; scan 1's averaged rows let coils sit 30 % inside 0.4 m); hard
    # blocks get their own tolerance (ctol/eta0 key "hard"); vessels = variable-radius helical + welded pipes (torus and
    # constant-radius helical dropped); one node per class: 8 devices x 8 vessels. Fresh seeds.
    "hsx_vessel2": dict(dir="vessel_scan2", job="xh", coils={**COILS, 48: (6, 5.0, 166667.0)}, qi=False, R0=1.2,
                        clearance=0.4, length_factor=1.6, vessel_margin=0.10, vessel_clearance=0.15,
                        seed=lambda aspect: f"{RUNS}/seeds/input.qa_nfp4_R1p2_A{aspect}",
                        devices=[(24, 0.25, 10, 0.03), (32, 0.25, 10, 0.01), (40, 0.25, 10, 0.01), (48, 0.25, 10, 0.03),
                                 (24, 0.33, 10, 0.01), (32, 0.33, 10, 0.03), (48, 0.33, 10, 0.01), (24, 0.25, 9, 0.03)],
                        vessels=["helical_r2"] + [f"pipe{n}" for n in (8, 12, 16, 20, 24, 28, 32)]),
    "w7x_vessel2": dict(dir="vessel_scan2", job="xw", coils=COILS_W7X, qi=False, scale=W7X_SCALE, R0=5.5, clearance=0.6,
                        length_factor=1.6, vessel_margin=0.15, vessel_clearance=0.20,
                        seed=lambda aspect: f"{RUNS}/seeds/input.qa_nfp5_R5p5_A{aspect}",
                        devices=[(40, 0.60, 8, 0.03), (50, 0.45, 10, 0.03), (50, 0.60, 10, 0.03), (60, 0.45, 10, 0.01),
                                 (40, 0.45, 10, 0.03), (60, 0.60, 8, 0.03), (50, 0.75, 10, 0.01), (50, 0.90, 10, 0.03)],
                        vessels=["helical_r2"] + [f"pipe{n}" for n in (10, 15, 20, 25, 30, 35, 40)]),
}
HARD_LIMITS = ["length", "curvature", "coil_coil", "coil_plasma", "coil_xline"]
# vessel scan 3 (2026-10-26, user: higher beta for the W7-X class (2 %), continue the devices that came closest): the
# best 32 + 32 vessel-scan-2 devices (passed the X-line gate after dn_converge; ranked by distance to the B.n and flatness
# goals + QS, list files from_list) continue from their dn_introduce result (stages 0-4 hard-linked, not recomputed).
# Changes: X-line flatness a hard per-point limit (scan 2: HSX X-lines 24-50 mm); the beta stages' B.n band 3e-3 -> 6e-4
# per harmonic (3e-3 let all 161 harmonics sit at the edge: rms(B.n/B) 1.0-1.3 % at beta 1 % although 0.18-0.3 % at the
# end of dn_converge; pointwise rms ~ 4x the largest harmonic); the W7-X class ramps on to beta 2 %. With flatness hard
# every HSX X-line went elliptic (|tr M| 1.7-1.95 by outer 10): hyperbolicity and the X-line-plasma distance window are
# hard too (HSX restarted at dn_converge, old stage kept as 05_dn_converge_v3elliptic).
FAMILIES["hsx_vessel3"] = dict(FAMILIES["hsx_vessel2"], dir="vessel_scan3", job="yh", from_dir="vessel_scan2",
                               from_list=f"{HERE}/ladders/vessel_scan3_hsx.txt", flat_hard=True, band=6.0e-4,
                               betas=(0.003, 0.01))
FAMILIES["w7x_vessel3"] = dict(FAMILIES["w7x_vessel2"], dir="vessel_scan3", job="yw", from_dir="vessel_scan2",
                               from_list=f"{HERE}/ladders/vessel_scan3_w7x.txt", flat_hard=True, band=6.0e-4,
                               betas=(0.003, 0.01, 0.02))
HARD_CTOL, HARD_ETA0 = 1.0e-3, 1.0e-2     # relative violation: 0.4 mm on a 0.4 m clearance


def with_hard_tolerances(lad):
    """Give the coils.hard blocks their own tolerance (key "hard"): with the stages' B.n-sized ctol (1e-4 / 1e-5) their
    penalties never stopped growing."""
    def patch(item, key, value):
        if not item.startswith(f"augmented_lagrangian.{key}="):
            return item
        rhs = item.split("=", 1)[1]
        body = rhs[1:-1] + ", " if rhs.startswith("{") else f"default: {rhs}, "
        return f"augmented_lagrangian.{key}={{{body}hard: {value}}}"
    lad["common"] = [patch(patch(x, "ctol", HARD_CTOL), "eta0", HARD_ETA0) for x in lad["common"]]
    if not any(x.startswith("augmented_lagrangian.ctol=") for x in lad["common"]):   # stages without their own ctol
        lad["common"].append(f"augmented_lagrangian.ctol={{default: 1.0e-5, hard: {HARD_CTOL}}}")
    for st in lad["stages"]:
        st["set"] = [patch(patch(x, "ctol", HARD_CTOL), "eta0", HARD_ETA0) for x in st["set"]]
    return lad
VESSEL_HARD = ["vessel_margin", "vessel_clearance"]


def vessel_sets(tag, margin, clearance):
    """--set items for a vessel tag: helical_r0 / helical_r2 / torus / pipe<nseg>."""
    shape = ({"helical_r0": ["vessel.family=helical", "vessel.radius_modes=0"],
              "helical_r2": ["vessel.family=helical", "vessel.radius_modes=2"],
              "torus": ["vessel.family=torus"]}.get(tag)
             or ["vessel.family=pipe", f"vessel.nseg={int(tag[len('pipe'):])}"])
    return ["vessel.enabled=true", f"vessel.margin={margin}", f"vessel.coil_clearance={clearance}",
            "vessel.xline_distance_weight=10.0"] + shape


def ladder(iota, aspect, well, qi=False, scale=1.0, clearance=None, R0=None, vessel=(), band=3.0e-3,
           betas=(0.003, 0.01), flat_hard=False):
    L = lambda x: round(x * scale, 6)              # length-valued engineering settings (m), x scale for bigger devices
    well_set = f"plasma_objective.magnetic_well={{enabled: true, floor: {well}, weight: 10.0}}"
    ripple_set = ("plasma_objective.axis_ripple={enabled: false}" if qi else
                  "plasma_objective.axis_ripple={enabled: true, threshold: 1.0e-3, weight: 100.0}")
    vac = ["double_null.enabled=false", "vmex.plasma_field=vacuum", "consistency.field_strength=toroidal_flux",
           "consistency.mpol=6", "consistency.ntor=9", "plasma_objective.iota_weight=10.0",
           "plasma_objective.iota_escalate=true"]
    dn = ["double_null.enabled=true", "double_null.mode=free", "double_null.free_order=10",
          "double_null.hyperbolicity_margin=0.2", "double_null.hyperbolicity_weight=1.0e-3",
          f"double_null.distance_min={L(0.04)}", f"double_null.distance_max={L(0.12)}", "double_null.distance_weight=1.0e+2",
          f"double_null.coil_distance_min={clearance if clearance else L(0.22)}", "double_null.coil_distance_weight=1.0e+4",
          "double_null.fieldline_penalty=1000.0", f"plasma_objective.iota_floor={iota}",
          "plasma_objective.iota_weight=1000.0", "plasma_objective.iota_escalate=true", well_set, ripple_set] + list(vessel)
    flat = ["double_null.flatness_weight=1000.0", "double_null.flatness_threshold=0.01",
            "double_null.flatness_pull_weight=100.0"]
    regular = ["boundary_regularity.enabled=true", "boundary_regularity.min_speed_ratio=0.15"]
    # v3 (2026-10-20): escalation capped at 100x (was 1e4x: unmeetable limits drove weights to 1e7-1e8, L ~ 1e21, inner
    # solves of 1-4 iterations, B.n and Mercier frozen for 40 outers) and the B.n block given priority: starting penalty
    # 1e5 (band 1e4) instead of 1e3, growing to 1e8 instead of 1e6
    stiff = ["coils.escalation_max_factor=100.0", "augmented_lagrangian.penalty_max=1.0e+8"]

    def beta_stage(name, beta, outer):
        return {"name": name, "beta_seed": beta, "set": [
            "vmex.plasma_field=virtual_casing", "consistency.field_strength=net_poloidal_current",
            "consistency.mpol=8", "consistency.ntor=9", f"consistency.bnormal_band={band}", "vmex.max_mode=2",
            "scaling.plasma_factor=0.1", "plasma_objective.aspect_target=null",
            # Mercier needs a converged radial resolution (2026-10-27, user: >= 51 surfaces): at ns = 16 the DMerc verdict
            # was wrong for 107 of 646 finished beta rungs, and 78 % are unstable at ns = 51 (runs/vmex/mercier_survey).
            # The beta rungs therefore solve VMEX at ns = 51 (~2x the seconds per evaluation) and penalise s >= 0.1 only
            # (the near-axis DMerc diverges with ns)
            "vmex.ns=51",
            "plasma_objective.mercier={enabled: true, margin: 0.0, smoothing: 1.0e-3, weight: 1.0e-4, s_min: 0.1}",
            "augmented_lagrangian.eta0={bnormal_band: 1.0e-4, net_poloidal_current: 1.0e-4, fieldline: 3.0e-4, "
            "boundary_regularity: 5.0e-3}",
            "augmented_lagrangian.ctol={bnormal_band: 1.0e-4, net_poloidal_current: 1.0e-5, fieldline: 3.0e-4, "
            "boundary_regularity: 5.0e-3}",
            f"augmented_lagrangian.outer_maxiter={outer}", "augmented_lagrangian.penalty_blocks={bnormal_band: 1.0e+4}"]
            + stiff + regular + dn + flat}

    if qi:
        climb = [
            {"name": "m2_qi", "set": vac + ["vmex.max_mode=2", f"plasma_objective.iota_floor={iota}", well_set, ripple_set,
                                            "augmented_lagrangian.outer_maxiter=10", "augmented_lagrangian.ctol=1.0e-4"]},
            {"name": "m3_qi", "set": vac + ["vmex.max_mode=3", f"plasma_objective.iota_floor={iota}", well_set, ripple_set,
                                            "augmented_lagrangian.outer_maxiter=10"]}]
    else:
        climb = [
            {"name": "m1_iota0.15", "set": vac + ["vmex.max_mode=1", "plasma_objective.iota_floor=0.15",
                                                   "augmented_lagrangian.outer_maxiter=6", "augmented_lagrangian.ctol=1.0e-4"]},
            {"name": "m2_iota0.25", "set": vac + ["vmex.max_mode=2", "plasma_objective.iota_floor=0.25",
                                                   "augmented_lagrangian.outer_maxiter=10", "augmented_lagrangian.ctol=1.0e-4"]},
            {"name": f"m3_iota{min(0.35, iota):.2f}", "set": vac + [
                "vmex.max_mode=3", f"plasma_objective.iota_floor={min(0.35, iota)}", well_set, ripple_set,
                "augmented_lagrangian.outer_maxiter=10", "augmented_lagrangian.ctol=1.0e-4"]},
            {"name": f"m3_iota{iota:.2f}", "set": vac + [
                "vmex.max_mode=3", f"plasma_objective.iota_floor={iota}", well_set, ripple_set,
                "augmented_lagrangian.outer_maxiter=9"]}]
    return {
        "config": "../../config.yaml",
        "common": (["plasma_objective.symmetry=qi"] if qi else [])
        + ([f"coils.hard=[{', '.join(HARD_LIMITS + (VESSEL_HARD if vessel else []) + (['xline_flatness', 'hyperbolicity', 'xpoint_distance'] if flat_hard else []))}]"]
           if clearance else []) + [
            "coil_seed.type=circular", "coil_seed.order=6",
            f"coil_seed.minor_radius_factor={round((R0 / aspect + 1.25 * clearance) / (R0 / aspect), 3) if clearance else 3.5}",
            "scaling.mode=jacobi",
            "vmex.mpol=6", "vmex.ntor=6", "vmex.nphi=32", "vmex.ntheta=32", f"plasma_objective.aspect_target={aspect}.0",
            f"coils.curvature_threshold={round(10.0 / scale, 6)}", f"coils.msc_threshold={round(20.0 / scale ** 2, 6)}",
            f"coils.coil_coil_threshold={L(0.08)}", f"coils.coil_plasma_threshold={clearance if clearance else L(0.22)}",
            "coils.arclength_weight=1.0e-3",
            "augmented_lagrangian.penalty=1000.0", "augmented_lagrangian.eta0=1.0e-3",
            "augmented_lagrangian.omega0=1.0e-2", "augmented_lagrangian.penalty_growth=2.0",
            "augmented_lagrangian.penalty_max=1.0e+6", "augmented_lagrangian.inner_maxiter=100",
            "boundary_guard.min_speed_ratio=0.05", "vmex.problem_kwargs={max_fsq_ratio: 1.0e+3}"],
        "stages": climb + [
            {"name": "dn_introduce", "set": [
                "vmex.plasma_field=vacuum", "consistency.field_strength=toroidal_flux", "consistency.mpol=6",
                "consistency.ntor=9", "vmex.max_mode=3", f"double_null.free_offset={L(0.06)}",
                "double_null.free_prefit_maxiter=0", "double_null.flatness_weight=0.0", "augmented_lagrangian.outer_maxiter=7",
                "augmented_lagrangian.eta0={default: 1.0e-3, boundary_regularity: 5.0e-3}",
                "augmented_lagrangian.ctol={default: 1.0e-4, boundary_regularity: 5.0e-3}"] + regular + dn},
            {"name": "dn_converge", "set": [
                "vmex.plasma_field=vacuum", "consistency.field_strength=toroidal_flux", "consistency.mpol=6",
                "consistency.ntor=9", "vmex.max_mode=2", "scaling.plasma_factor=0.1",
                "augmented_lagrangian.outer_maxiter=40",
                "augmented_lagrangian.eta0={default: 1.0e-3, boundary_regularity: 5.0e-3}",
                "augmented_lagrangian.ctol={default: 1.0e-5, boundary_regularity: 5.0e-3}",
                "augmented_lagrangian.penalty_blocks={bnormal: 1.0e+5}"] + stiff + regular + dn + flat},
        ] + [beta_stage(f"beta{100 * b:g}_G_band", b, 15 if b < 0.005 else 25) for b in betas] + [
        ],
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=sorted(FAMILIES), default="hsx")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--stagger-min", type=float, default=2.0)
    p.add_argument("--exclude", default="cn12,cn07,cn08,cn01")
    p.add_argument("--mem", default="16G", help="per job; ladder jobs peak at 4-11 GB RSS (v1 sacct)")
    p.add_argument("--startup-jitter", type=int, default=1800,
                   help="max random sleep (s) before Python starts (sbatch/ladder.sbatch STARTUP_JITTER)")
    p.add_argument("--nice", type=int, default=0, help="sbatch --nice (lower priority than other families)")
    p.add_argument("--resubmit", action="store_true", help="submit into existing run dirs too (the ladder resumes)")
    args = p.parse_args()
    fam = FAMILIES[args.family]
    os.makedirs(os.path.join(HERE, "ladders", fam["dir"]), exist_ok=True)
    k = 0
    grid = fam.get("devices") or itertools.product(fam["coils"], fam["iotas"], fam["aspects"], WELLS)
    if fam.get("from_list"):
        def parse(n):
            m = re.match(r"c(\d+)_i([\d.]+)_A(\d+)_w(\d+)_(.+)$", n)
            return (int(m[1]), float(m[2]), int(m[3]), int(m[4]) / 100.0), m[5]
        grid = [parse(n) for n in open(fam["from_list"]).read().split()]
    elif fam.get("vessels"):
        grid = [(dev, v) for dev in grid for v in fam["vessels"]] + [(dev, v) for dev, vs in fam.get("extra", []) for v in vs]
    else:
        grid = [(dev, None) for dev in grid]
    for (ncoils, iota, aspect, well), vtag in grid:
        name = f"c{ncoils}_i{iota:.2f}_A{aspect}_w{int(round(well * 100))}" + (f"_{vtag}" if vtag else "")
        vessel = vessel_sets(vtag, fam["vessel_margin"], fam["vessel_clearance"]) if vtag else ()
        if args.only and name not in args.only:
            continue
        path = os.path.join(HERE, "ladders", fam["dir"], f"{name}.yaml")
        with open(path, "w") as fh:
            fh.write(f"# {args.family} scan device {name}: written by tools/make_hsx_scan.py (edit the generator, not this "
                     "file)\n")
            lad = ladder(iota, aspect, well, fam["qi"], fam.get("scale", 1.0), fam.get("clearance"), fam.get("R0"), vessel,
                         band=fam.get("band", 3.0e-3), betas=fam.get("betas", (0.003, 0.01)),
                         flat_hard=fam.get("flat_hard", False))
            yaml.safe_dump(with_hard_tolerances(lad) if fam.get("length_factor") == 1.6 else lad,
                           fh, sort_keys=False, width=200)
        if not args.submit:
            continue
        nb, length, current = fam["coils"][ncoils]
        if fam.get("length_factor"):
            length = round(fam["length_factor"] * 2.0 * math.pi * (fam["R0"] / aspect + fam["clearance"]), 2)
        out = os.path.join(RUNS, fam["dir"], name)
        if os.path.exists(os.path.join(out, "logs")) and not args.resubmit:
            print(f"EXISTS {out}, not submitted (resubmit by hand to resume)")
            continue
        os.makedirs(os.path.join(out, "logs"), exist_ok=True)
        if fam.get("from_dir"):
            # continue from the earlier scan's dn_introduce: stages 0-4 hard-linked (run_ladder skips finished stages)
            src = os.path.join(RUNS, fam["from_dir"], name)
            for stage in sorted(os.listdir(src)):
                if re.match(r"0[0-4]_", stage) and not re.search(r"_v\d", stage) and not os.path.exists(os.path.join(out, stage)):
                    subprocess.check_call(["cp", "-al", os.path.join(src, stage), os.path.join(out, stage)])
        env = dict(os.environ, LADDER=os.path.relpath(path, HERE), SEED=fam["seed"](aspect), OUTDIR=out,
                   STARTUP_JITTER=str(args.startup_jitter),
                   SETS=f"coil_seed.ncoils={nb} coils.length_threshold={length} coils.current_threshold={current}")
        begin = f"now+{int(round(k * args.stagger_min * 60))}"
        jid = subprocess.check_output(
            ["sbatch", "--parsable", f"--job-name={fam['job']}_{name}", f"--exclude={args.exclude}", "--cpus-per-task=4",
             f"--mem={args.mem}", f"--nice={args.nice}", f"--begin={begin}", f"--output={out}/logs/ladder_%j.out", "sbatch/ladder.sbatch"],
            cwd=HERE, env=env, text=True).strip()
        print(f"{name} {jid} begin {begin}")
        k += 1


if __name__ == "__main__":
    main()
