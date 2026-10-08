#!/usr/bin/env python
r"""
How much would designA's coils have to change at finite beta? A stage-II beta scan.

For each volume-averaged beta (0, the design point 1e-4, 1e-3, 3e-3, 1e-2) this

1. solves designA's fixed-boundary equilibrium with VMEC2000 (p(s) = PRES_SCALE (1 - s),
   zero net toroidal current; PRES_SCALE scaled from the beta = 1 % calibration of
   inputs/input.designA_beta1),
2. runs ONE virtual-casing evaluation (simsopt VirtualCasing, VC_DIGITS digits) for the
   plasma's own normal field B_plasma·n on the boundary -- the target the coils must
   cancel, B_coil·n = -B_plasma·n,
3. scores designA's own (vacuum-optimised) coils against that target, and
4. re-fits designA's coils to it (stage II, SquaredFlux "normalized"), starting FROM
   designA, shapes only (currents fixed), stellarator symmetry pinned, every
   engineering limit just outside designA's own coils (inactive at the start),
   and reports how far the coils had to move.

beta = 0 is the reference: there B_plasma = 0 exactly, so whatever virtual casing
returns is its numerical noise, and designA's miss is its plain vacuum fit error. A
beta whose target sits below these two floors needs no coil change.

Results (job 46) and two caveats -- read beta_response_designA.py for the answer:

* the beta = 0 target is NOT zero but 1.4e-3 of |B| (rms), independent of the virtual-casing grid and digits: a
  systematic error of the input boundary field, identical at every beta (same fixed boundary). Use differences
  B_ext·n(beta) - B_ext·n(0) for the plasma's own field (1.22e-4 at the design point, linear in beta);
* the re-fit's coil displacement (~20 mm even at beta = 0) is the optimizer drifting along a flat objective, not a
  required change. beta_response_designA.py computes the smallest change instead (Tikhonov linear response):
  0.4 mm rms at the design point, ~4 mm at 1e-3, ~5 cm at 1e-2.

Run on a compute node (5 VMEC solves + 5 virtual-casing evaluations, ~15 min):

    python stage_two_beta_scan_designA.py
"""

import os
import re
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from simsopt import load, save
from simsopt.field import BiotSavart
from simsopt.geo import CurveCurveDistance, CurveLength, CurveSurfaceDistance, LpCurveCurvature, SurfaceRZFourier
from simsopt.mhd import VirtualCasing, Vmec
from simsopt.objectives import QuadraticPenalty, SquaredFlux
from star_lite_design.utils.vmex_combined_stage import coil_stellsym_error, fix_self_symmetric_coil_parity

HERE = Path(__file__).resolve().parent

# Betas to scan (volume averaged); 1e-4 is Star_Lite's design point (n 5e17 m^-3, Te 10 eV, Ti 1 eV, p0 0.88 Pa):
BETAS = [0.0, 1e-4, 1e-3, 3e-3, 1e-2]

# designA at beta = 1 % (PRES_SCALE 56.6 Pa -> VMEC2000 beta 0.998 %); other betas scale PRES_SCALE linearly
template_input = HERE / "inputs" / "input.designA_beta1"
PRES_SCALE_BETA1 = 56.5965115789313273
design_archive = HERE.parents[1] / "convert" / "designA_after_scaled.json"

# Boundary grid (half period) and virtual-casing resolution:
nphi = 32
ntheta = 32
vc_src_nphi = 80
VC_DIGITS = 6

# Stage-II re-fit of designA's coils. Limits sit just outside designA's own coils (lengths 3.03 / 3.09 m, max
# curvature 7.95 / 10.47 /m, coil-coil 0.149 m, coil-plasma 0.133 m), so none is active at the start:
LENGTH_MAX = 3.3
CURVATURE_MAX = 11.0
CC_THRESHOLD = 0.12
CS_THRESHOLD = 0.10
LENGTH_WEIGHT = 1e-2
CURVATURE_WEIGHT = 1e-4
CC_WEIGHT = 1e1
CS_WEIGHT = 1e1
MAXITER = 400

#######################################################
# End of input parameters.
#######################################################

out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)
template = template_input.read_text()


def namelist_for(beta):
    """designA namelist at `beta`: PRES_SCALE scaled from the beta = 1 % calibration."""
    scale = PRES_SCALE_BETA1 * beta / 0.01
    text = re.sub(r"(?m)^(\s*PRES_SCALE\s*=\s*).*$", lambda m: f"{m.group(1)}{scale:.17E}", template)
    path = out_dir / f"input.designA_beta{beta:.0e}"
    path.write_text(text)
    return path


def bn_stats(bs, s, target):
    """<|B·n - target|>/|B| and max, on the boundary grid."""
    bs.set_points(s.gamma().reshape((-1, 3)))
    B = bs.B().reshape((nphi, ntheta, 3))
    r = np.abs(np.sum(B * s.unitnormal(), axis=2) - target) / np.linalg.norm(B, axis=2)
    return float(np.mean(r)), float(np.max(r))


rows = []
for beta in BETAS:
    t0 = time.time()
    print(f"\n######## beta = {beta:.0e} ########", flush=True)
    tag = f"designA_beta{beta:.0e}"
    vmec_file = out_dir / f"wout_{tag}.nc"
    if not vmec_file.is_file():
        equil = Vmec(str(namelist_for(beta)))
        equil.run()
        os.replace(equil.output_file, vmec_file)
    wout = Vmec(str(vmec_file)).wout
    beta_vmec = float(wout.betatotal)
    vc_file = out_dir / f"vcasing_{tag}.nc"
    if vc_file.is_file():
        vc = VirtualCasing.load(str(vc_file))
    else:
        vc = VirtualCasing.from_vmec(str(vmec_file), src_nphi=vc_src_nphi, trgt_nphi=nphi, trgt_ntheta=ntheta,
                                     digits=VC_DIGITS, filename=str(vc_file))
    s = SurfaceRZFourier.from_wout(str(vmec_file), range="half period", nphi=nphi, ntheta=ntheta)

    # designA's own coils, untouched
    coils = load(str(design_archive))[0][0].biotsavart.coils
    bs = BiotSavart(coils)
    bs.set_points(s.gamma().reshape((-1, 3)))
    Bmag = np.linalg.norm(bs.B().reshape((nphi, ntheta, 3)), axis=2)
    target = vc.B_external_normal
    target_mean, target_max = float(np.mean(np.abs(target) / Bmag)), float(np.max(np.abs(target) / Bmag))
    miss0_mean, miss0_max = bn_stats(bs, s, target)

    # stage-II re-fit from designA: shapes only, symmetric, limits inactive at the start
    curves = [c.curve for c in coils]
    base, seen = [], set()
    for c in coils:
        for o in c.curve.unique_dof_lineage:
            if o.local_dof_size > 0 and id(o) not in seen and "Curve" in type(o).__name__:
                seen.add(id(o))
                base.append(o)
    for c in coils:                       # the Currents behind the ScaledCurrent wrappers
        for o in c.current.unique_dof_lineage:
            o.fix_all()
    fix_self_symmetric_coil_parity(base)
    gamma0 = [c.gamma().copy() for c in base]
    Jf = SquaredFlux(s, bs, target=target, definition="normalized")
    Jls = [CurveLength(c) for c in base]
    JF = (Jf + LENGTH_WEIGHT * sum(QuadraticPenalty(J, LENGTH_MAX, "max") for J in Jls)
          + CURVATURE_WEIGHT * sum(LpCurveCurvature(c, 2, CURVATURE_MAX) for c in base)
          + CC_WEIGHT * CurveCurveDistance(curves, CC_THRESHOLD)
          + CS_WEIGHT * CurveSurfaceDistance(curves, s, CS_THRESHOLD))
    J_start = float(JF.J())

    def fun(dofs):
        JF.x = dofs
        return JF.J(), JF.dJ()

    res = minimize(fun, JF.x, jac=True, method="L-BFGS-B",
                   options={"maxiter": MAXITER, "maxcor": 300, "ftol": 1e-20, "gtol": 1e-20}, tol=1e-20)
    JF.x = res.x
    miss1_mean, miss1_max = bn_stats(bs, s, target)
    disp = np.concatenate([np.linalg.norm(c.gamma() - g0, axis=1) for c, g0 in zip(base, gamma0)])
    save(coils, out_dir / f"coils_refit_{tag}.json")
    row = dict(beta=beta, beta_vmec=beta_vmec, target_mean=target_mean, target_max=target_max,
               miss0_mean=miss0_mean, miss0_max=miss0_max, miss1_mean=miss1_mean, miss1_max=miss1_max,
               disp_rms_mm=1e3 * float(np.sqrt(np.mean(disp ** 2))), disp_max_mm=1e3 * float(np.max(disp)),
               lengths=[round(float(J.J()), 4) for J in Jls], kappa=max(float(np.max(c.kappa())) for c in base),
               sym=coil_stellsym_error(coils), J_start=J_start, J_end=float(res.fun), nit=int(res.nit))
    rows.append(row)
    print(f"beta {beta:.0e} (VMEC {beta_vmec:.3e}): target <|B_plasma·n|>/|B| = {target_mean:.2e} (max {target_max:.2e}); "
          f"designA's coils miss it by {miss0_mean:.2e} (max {miss0_max:.2e}); re-fit {miss1_mean:.2e} "
          f"(max {miss1_max:.2e}) after {res.nit} it; coil displacement rms {row['disp_rms_mm']:.3f} mm, "
          f"max {row['disp_max_mm']:.3f} mm; lengths {row['lengths']}, max kappa {row['kappa']:.2f}, "
          f"symmetry {row['sym']:.1e} m ({time.time() - t0:.0f}s)", flush=True)

print("\n######## summary ########")
print("  beta     | target <|Bp.n|>/|B| | designA miss | re-fit miss | coil move rms / max [mm]")
for r in rows:
    print(f"  {r['beta']:8.0e} | {r['target_mean']:19.2e} | {r['miss0_mean']:12.2e} | {r['miss1_mean']:11.2e} | "
          f"{r['disp_rms_mm']:8.3f} / {r['disp_max_mm']:.3f}")
import json  # noqa: E402
(out_dir / "beta_scan_summary.json").write_text(json.dumps(rows, indent=1))
