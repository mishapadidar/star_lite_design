#!/usr/bin/env python
r"""
Smallest change of designA's coils that cancels its finite-beta plasma field (linear response).

Companion of stage_two_beta_scan_designA.py; run it in the same directory, after the scan
(it reads the scan's cached ``output/wout_designA_beta*.nc`` and ``output/vcasing_*.nc``).

Why differences: on designA the virtual-casing target B_ext·n at beta = 0 is 1.4e-3 of
|B| although the plasma field is zero there, independent of the virtual-casing grid and
digits (src_nphi 80-320, digits 6-8) -- a systematic error of the input boundary field,
identical at every beta because the fixed boundary is identical. The plasma's own
contribution is therefore

    dt(beta) = B_ext·n(beta) - B_ext·n(0)      (1.22e-4 rms of |B| at beta 1e-4, linear in beta),

and the coils must change so that their B·n changes by dt. For the 148 free coil
coefficients x (stellarator symmetry pinned, currents fixed) this is the Tikhonov problem

    min_dx  |W (J dx - dt)|^2 + lam^2 |dx|^2,    J = d(B_coil·n)/dx,  W = 1/|B| / sqrt(npoints),

solved for a range of lam (an L-curve): the residual says how much of dt the coils can
absorb, the coil displacement how far they must move for it. The yardstick is designA's
own vacuum fit error rms(B_coil·n/|B|) on this boundary.
"""

import json
from pathlib import Path

import numpy as np
from simsopt import load
from simsopt.field import BiotSavart
from simsopt.geo import SurfaceRZFourier
from simsopt.mhd import VirtualCasing
from star_lite_design.utils.vmex_combined_stage import fix_self_symmetric_coil_parity

HERE = Path(__file__).resolve().parent
design_archive = HERE.parents[1] / "convert" / "designA_after_scaled.json"
BETAS = [1e-4, 1e-3, 3e-3, 1e-2]
LAMBDAS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
FD_STEP = 1e-6          # m, finite-difference step for the Jacobian (B·n is linear-ish in the coil coefficients)
nphi = ntheta = 32

out = Path("output")
s = SurfaceRZFourier.from_wout(str(out / "wout_designA_beta0e+00.nc"), range="half period", nphi=nphi, ntheta=ntheta)
normal = s.unitnormal()
coils = load(str(design_archive))[0][0].biotsavart.coils
for c in coils:
    for o in c.current.unique_dof_lineage:
        o.fix_all()
base, seen = [], set()
for c in coils:
    for o in c.curve.unique_dof_lineage:
        if o.local_dof_size > 0 and id(o) not in seen and "Curve" in type(o).__name__:
            seen.add(id(o))
            base.append(o)
fix_self_symmetric_coil_parity(base)
bs = BiotSavart(coils)
bs.set_points(s.gamma().reshape(-1, 3))


def bn():
    return np.sum(bs.B().reshape(nphi, ntheta, 3) * normal, axis=2)


Bmag = np.linalg.norm(bs.B().reshape(nphi, ntheta, 3), axis=2)
W = (1.0 / Bmag).ravel() / np.sqrt(Bmag.size)
rms = lambda v: float(np.linalg.norm(W * np.ravel(v)))          # rms of v/|B|
bn0 = bn()
print(f"designA's own vacuum fit error on this boundary: rms(B_coil·n/|B|) = {rms(bn0):.2e}")

# Jacobian of B_coil·n w.r.t. the free coil coefficients (central differences), and of the coil points
x0 = bs.x.copy()
J = np.zeros((bn0.size, x0.size))
G = []                                                  # d(points of all base curves)/dx
g0 = np.concatenate([b.gamma() for b in base])
for j in range(x0.size):
    xp, xm = x0.copy(), x0.copy()
    xp[j] += FD_STEP
    xm[j] -= FD_STEP
    bs.x = xp
    bp, gp = bn().ravel(), np.concatenate([b.gamma() for b in base])
    bs.x = xm
    bm, gm = bn().ravel(), np.concatenate([b.gamma() for b in base])
    J[:, j] = (bp - bm) / (2 * FD_STEP)
    G.append((gp - gm) / (2 * FD_STEP))
bs.x = x0
G = np.stack(G, axis=-1)                                # (npoints_curves, 3, ndofs)
U, S, Vt = np.linalg.svd(W[:, None] * J, full_matrices=False)
print(f"{x0.size} free coil coefficients; singular values of W J: max {S[0]:.2e}, min {S[-1]:.2e}")

dt0 = VirtualCasing.load(str(out / "vcasing_designA_beta0e+00.nc")).B_external_normal
rows = []
for beta in BETAS:
    dt = VirtualCasing.load(str(out / f"vcasing_designA_beta{beta:.0e}.nc")).B_external_normal - dt0
    b = W * dt.ravel()
    print(f"\nbeta {beta:.0e}: plasma signal rms(dt/|B|) = {rms(dt):.2e} "
          f"({rms(dt) / rms(bn0):.2f} x designA's vacuum fit error)")
    for lam in LAMBDAS:
        f = S / (S ** 2 + lam ** 2)
        dx = Vt.T @ (f * (U.T @ b))
        resid = float(np.linalg.norm(W * (J @ dx) - b))
        disp = np.linalg.norm(np.einsum("pkj,j->pk", G, dx), axis=1)
        rows.append(dict(beta=beta, signal=rms(dt), lam=lam, residual=resid, absorbed=1 - resid / rms(dt),
                         disp_rms_mm=1e3 * float(np.sqrt(np.mean(disp ** 2))), disp_max_mm=1e3 * float(disp.max())))
        r = rows[-1]
        print(f"  lam {lam:7.0e}: residual {resid:.2e} ({100 * r['absorbed']:5.1f} % absorbed), "
              f"coil move rms {r['disp_rms_mm']:.3f} mm, max {r['disp_max_mm']:.3f} mm")
(out / "beta_response_summary.json").write_text(json.dumps(dict(vacuum_fit_error=rms(bn0), rows=rows), indent=1))
