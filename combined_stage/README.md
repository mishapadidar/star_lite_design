# Combined stage-1/2 optimization with VMEX and an augmented Lagrangian

Optimizes the plasma boundary and the coils **together**, following the combined
("single-stage") method of Jorge, Goodman, Landreman, Rodrigues & Wechsung,
*Plasma Phys. Control. Fusion* **65**, 074003 (2023), doi:10.1088/1361-6587/acd957,
with two changes proposed by A. Giuliani:

1. **Exact plasma gradients.** The fixed-boundary equilibrium is solved with
   [VMEX](https://github.com/uwplasma/vmex) (JAX VMEC), whose implicit adjoint
   gives exact derivatives of plasma quantities w.r.t. the boundary — no finite
   differences of VMEC. Coil derivatives are simsopt's analytic ones.
2. **Consistency as a constraint.** Instead of `f_QS + λ f_BdotN` with a large
   fixed λ (ill-conditioned as λ → ∞), plasma/coil consistency is an equality
   constraint solved with an augmented Lagrangian, which reaches feasibility at
   a finite penalty.

## Formulation

```
min_{s, c}   f_QS(s) + Σ_k w_k P_k(c, s)                      (objective)
s.t.         B·n harmonics of (B_plasma(s) + B_coil(c)) / B_ref = 0   (bnormal)
             one field-strength condition:
               (∮A_coil·dl − Φ_target)/|Φ_target| = 0, |Φ_target| = PHIEDGE   (toroidal_flux, default)
               or ⟨|B_out|² − |B_in|² − 2μ0 p_edge⟩ / B_ref² = 0            (pressure_balance)
```

* `s` — VMEX boundary Fourier modes up to `max_mode` (optionally the current profile);
  `c` — coil shapes and currents.
* `f_QS` — VMEX least-squares cost of the plasma terms (quasisymmetry ratio
  residual, aspect ratio, optional iota floor and beta target).
* `P_k` — coil engineering penalties reused from `array/boozer_all.py` (length,
  curvature, mean-squared curvature, coil–coil distance, current bound) plus a
  coil–plasma distance to the moving boundary. They stay penalties for now.
* **Why Fourier harmonics of B·n, not point values:** one constraint per
  quadrature point gives more equality constraints than dofs; they can't all
  hold at once (and constraint qualification fails). Harmonics `m ≤ mpol`,
  `|n| ≤ ntor` keep the count below the dof count. For a stellarator-symmetric
  boundary B·n is odd, so only `sin(mθ − n·nfp·φ)` modes are constrained
  (40 constraints for `mpol = ntor = 4`). `mode: points` is available too.
* **Why a field-strength condition:** B·n = 0 cannot see the harmonic field that
  is tangential everywhere on the torus — the toroidal field set by the total
  poloidal coil current. Without it a vacuum problem is "solved" by turning the
  coils off. One scalar, independent of the B·n rows. `toroidal_flux` (default)
  requires the coil flux through the φ = 0 boundary cross-section, `∮A·dl`
  evaluated on the cross-section computed straight from the boundary Fourier
  coefficients, to equal PHIEDGE: every gradient is analytic. `pressure_balance`
  is the finite-beta alternative but needs `|B_in|²` at grid points on the boundary,
  whose gradient is gauge-sensitive (see validation).
* **Plasma field model** (`vmex.plasma_field`): `vacuum` sets `B_plasma = 0`, the
  vacuum single-stage formulation, right for Star_Lite's β ≈ 3e-4 (where the
  virtual-casing plasma field is at the level of its own discretization error);
  `virtual_casing` includes it for finite β.
* `B_ref` is `sqrt(⟨|B|²⟩)` on the seed boundary, frozen.

## Reactor-size QA double-null variants (in progress)

Goal (2026-09-10): QA, nfp = 2, double-null variants of designA at reference-device sizes —
**HSX-size** (R = 1.2 m, A ≈ 8, B0 = 1.0 T, β = 1 %) and **W7-X-size** (R = 5.5 m, A ≈ 10.5,
B0 = 2.5 T, β = 3 %). The boundary and X-points may move; the double null must survive.

* All shaping is done at designA scale (R = 0.5 m): the field-line geometry is size-invariant,
  and ideal MHD is size-invariant at fixed β. Scale at the end (`workflows/scaling/scale_design.py`),
  with engineering ranges given at target scale ÷ L.
* **ι must rise.** A rough equilibrium-limit estimate Δ/a ≈ 1.4 · βA/(2ι²) (factor calibrated to
  designA's free-boundary scan) needs ι ≈ 0.33 (limit) → 0.43–0.53 (comfortable) for the HSX
  variant and ≈ 0.66 → 0.86–1.05 for the W7-X variant; designA has ι ≈ 0.19–0.24.
* **Double null** = the upper X-point periodic field line (lower = its stellarator-symmetric image)
  keeps solving, stays hyperbolic and stays near the boundary. designA: tr M = 2.99 (eigenvalues
  0.38, 2.60), 4.8–9.7 cm from the optimization-surface boundary, 6 coils (41 / 28 kA).
* Terms (`utils/vmex_double_null.py`, section `double_null`): `XpointHyperbolicity`
  = max(2 + margin − |tr M|, 0)² through the tangent-map adjoint; `XpointPlasmaDistance` keeps every
  X-point point within [distance_min, distance_max] of the boundary sampled over the full torus
  (`VmexPlasma` output `boundary_points`, exact in the boundary dofs). Failure of the X-point solve
  is a rejected trial (barrier).
* Continuation: `run_ladder.py ladders/<variant>.yaml --vmec-input <seed> --outdir <dir>` runs the
  stages in sequence, each seeded from the previous stage's `input.combined_stage_final`,
  `coils_final.json`, `xpoint_final.json`; finished stages are skipped on resubmission.
  `ladders/hsx_variant_vacuum.yaml`: A 7.0 → 7.5 → 8.0, ι floor 0.24 → 0.30 → 0.36 → 0.42, max_mode 3.
* Coils: shapes, positions and currents are free with range-type limits (length ≤, curvature ≤,
  current ≤, coil–coil ≥, coil–plasma ≥). The NUMBER of coils is discrete — designA has 6; the
  thinner, longer A = 8–10.5 plasmas may need more per period (not yet implemented).

Validated so far: double-null unit tests 4/4 (Taylor 3e-12 distance, 1e-8 hyperbolicity, one
shared pullback); `boundary_points` pullback exact (1e-12) and identical to the solved VMEX boundary;
driver smoke with the double null on designA (2 × 3 iterations): X-point kept (tr M 2.987 → 2.988,
4.95–9.46 cm), X-point/coil checkpoints written and reseeded successfully.

**Seed-namelist truncation (fixed 2026-09-10).** `workflows/hint/run_vmec_from_json.py` wrote
`MPOL = surface.mpol` together with RBC/ZBS entries at m = MPOL, which VMEC/VMEX ignore (they use
m < MPOL): designA's solved boundary was up to 1.48 mm off its design surface. It now writes
`MPOL = surface.mpol + 1`; a regenerated designA namelist reproduces the design surface exactly.
Namelists generated before the fix keep the truncated boundary; the ladders use an `MPOL = 11` copy
(`simflare/runs/combined_stage/seeds/input.designA_opt_beta0p03_mpol11.fixed`).

With the MPOL = 11 seed the designA coil flux matches PHIEDGE to 2.5e-5 (1.1e-3 with the truncated
seed). Keep the interface grid at or above the boundary's Nyquist resolution (`vmex.nphi`,
`vmex.ntheta` ≥ 2·max(mpol, ntor)): at 16 × 16 an mpol 11 / ntor 10 boundary aliases high-order B·n
into the constrained low harmonics; the ladders use 32 × 32.

**HSX stage 0 attempts (2026-09-10) — what the combined method did on designA.**

| run | scaling / penalty | outcome |
|---|---|---|
| 456 | fixed (coil_step 1e-2), ρ0 10 | B·n drifted to 2e-2; coils moved 2 mm; stopped |
| 482/487 | fixed, ρ0 1e3 ×10 growth, inner 40, mpol 6 | penalty ran to 1e7; converged back to the seed (coils 0.9 mm) |
| 490/491 | Jacobi (constraint columns), ρ0 1e3 ×2, inner 100 | coils moved 93 mm, constraints ~3e-4, X-point trM 3.28, but aspect/ι penalties flat over 3 outer it.; died on a driver bug (fixed) |

Constraint-Jacobian analysis at the seed (unit feasible steepest-descent step in the optimizer variables):
Jacobi scaling moves the plasma 2.0 mm per 53 mm of coil motion (objective rate −0.036); uniform 1e-2 gives
6.3 mm / 7.7 mm (−0.144); plasma 1e-2 with coils Jacobi and balanced blocks 5.5 mm / 318 mm (−0.263). Two
conclusions: equilibrating constraint columns starves the plasma dofs the objective depends on; and every
direction that keeps B·n = 0 needs 30–60× more coil than plasma motion — designA's 6-coil set is a weak lever for
aspect-ratio/ι reshaping. Tooling: `simflare/runs/combined_stage/_proto/proto_jacobian.py`.

**nfp = 4 from circular coils (2026-09-11).** Decision: keep the single-stage method and give it a
stronger coil lever instead of alternating stage-1/stage-2 runs. The new variant starts from scratch at
target size rather than from designA:

* seed `simflare/runs/combined_stage/seeds/input.qa_nfp4_R1p2_A8`: nfp = 4, R = 1.2 m, a = 0.15 m,
  B0 = 1.0 T (PHIEDGE = πa²B0), vacuum, rotating ellipse RBC(1,1) = −ZBS(1,1) = 0.2 a
  (A = 8.17, ι 0.144 → 0.159). Equal signs give a breathing circle with ι = 0 and a φ = 0 cross-section
  1.44× too large (toroidal-flux residual −0.44 at the first evaluation) — don't.
  VMEX's INDATA parser rejects `RAXIS_C`/`ZAXIS_S`; leave the axis guess out.
* coils: `coil_seed.type: circular` builds `ncoils` planar circles per half period
  (`create_equally_spaced_curves` + `coils_via_symmetries`), radius `minor_radius_factor`·a, each
  carrying I0 = 2πR0·B0/(μ0·2·nfp·ncoils). Ladder `ladders/hsx_nfp4_circular_vacuum.yaml`: 32 coils
  (160 coil + 8 plasma dofs at max_mode 1), 187.5 kA each, A = 8, ι floor 0.15 → 0.42, max_mode 1 → 3,
  ranges length ≤ 3.5 m, curvature ≤ 10 /m, coil–coil ≥ 8 cm, coil–plasma ≥ 15 cm, |I| ≤ 250 kA.
* No double null in this ladder: circular coils have no X-point to track, and none of the star_lite_design
  terms (`boozer_all.py` DN/SN/mono, `boozer_singular_opt.py`) create one — they all need a solved
  `PeriodicFieldLine` seed. Next: search the resulting coil field outside the boundary for hyperbolic
  period-1 fixed points (`utils/fixed_point_analysis.find_fixed_points`); seed a double-null ladder from an
  up/down pair, or, if there is none, add a target X-line constraint (a free curve required to be a
  field line with |tr M| > 2).
* Risk: QA quality degrades with more field periods at fixed aspect ratio; nfp = 4 at A = 8 is untested.
* X-point search (`simflare/runs/combined_stage/_proto/xpoint_search.py <stage dir> [tag] [offsets] [ntheta]`):
  seeds follow the boundary (fixed VMEC θ over one field period, pushed outward along the normal), fitted as a
  non-stellsym one-period curve and passed to `utils/fixed_point_analysis.find_fixed_points`; hyperbolic points
  are saved in the `--xpoint` format. On the stage-1 coils (A 8.01, ι 0.25 → 0.28) seeds 3 and 8 cm outside found
  only the magnetic axis (tr M 1.846, i.e. ι 0.25) — no X-point that close. Solutions where Newton ran off to
  |B| ≈ 0 (R ~ 100 m) are discarded (tr M not finite or > 0.3 m from the boundary). The rotation per field period
  targeted here (0.42 / 4 ≈ 0.105) matches designA's (≈ 0.1), so a period-1 double null is not excluded; a wider
  search (3–20 cm) decides between seeding the double-null terms and a target X-line constraint. It found only
  the axis again, so the double null has to be created.
* **Free X-line** (`double_null.mode: free`, `utils/vmex_double_null.py`). The X-line is a design variable: a
  one-period `CurveXYZFourierSymmetries` (not stellarator symmetric, one toroidal turn) plus its length
  (`FreeXLine`). `XLineFieldLineConstraint` makes it a field line as an augmented-Lagrangian block — the
  `periodicfieldline.py` residual γ′/L − B/|B| and the y(0) = 0 label, as many equations as the line adds dofs —
  `FreeXLineHyperbolicity` keeps |tr M| ≥ 2 + margin with gradients straight through the field and its first two
  derivatives (no field-line adjoint), and `XpointPlasmaDistance(signed=True)` keeps it 4–12 cm OUTSIDE the
  boundary. It starts at the top of every cross-section, `free_offset` out; its stellarator-symmetric image is
  the lower null. At feasibility it is exactly the line the tracked terms see: on designA's solved X-point the
  residual is 9e-16 and tr M agrees to all printed digits (+2.987121).
  Introduce it gently. With the X-line block at ρ = 1000 (starting residual 0.42) and designA's hyperbolicity
  weight 1e2 the first 3 iterations bent the coils to 39 /m and raised B·n 100×. A least-squares pre-fit of the
  curve on the frozen coils (`free_prefit_maxiter`) is off: with no X-point in the field it slid onto the magnetic
  axis, 10 cm inside the plasma, which the unsigned distance accepted. The X-line block therefore has its own
  starting penalty (`fieldline_penalty`, 10) and the hyperbolicity and distance weights (1 and 1e2) escalate ×10
  per outer iteration while violated, like the coil ranges (`ladders/hsx_nfp4_dn_free.yaml`).
  That alone is not enough: in the next smoke tr M rose 1.48 → 2.04 while the X-line residual stayed at 0.42 — the
  coils were raising the return-map trace of a curve that is not a field line. Part of the reason was the
  augmented-Lagrangian schedule: after a penalty increase the block tolerance was reset to the textbook 1/ρ^0.1,
  which assumes O(1) constraints (0.74 at ρ = 20), so the X-line block's 0.4 residual counted as met and its penalty
  grew only every other outer iteration. The reset now keeps the scale of `eta0` (η = η₀·ρ₀^0.1 / ρ^0.1; unchanged
  without `eta0`). Whether the field line or the hyperbolicity should lead is being tested with two 10-outer trials
  from the stage-1 coils (`simflare/runs/combined_stage/dn_free_trial_{A,B}`: X-line penalty 10 with hyperbolicity
  weight 1, and penalty 1000 with weight 1e-3).
  First outer iterations: making the curve a field line first (B) works — residual 0.42 → 1.0e-3 with tr M 2.68,
  5.7–8.5 cm outside the boundary, after one outer iteration (100 inner, 19 min); the price is B·n (rms 1.3e-2) and
  f_QS (0.0057 → 0.029), which the following outer iterations have to recover. Leading with the hyperbolicity (A)
  reaches tr M 2.27–2.62 but only residual 0.11. `simflare/runs/combined_stage/_proto/verify_xline.py` checks
  whether such a line is a genuine X-point: Newton-solve it as a `PeriodicFieldLine` on the checkpoint coils,
  classify it and report its signed distance to the boundary. Trial B's line after that first outer iteration is
  one: Newton converges in 3 iterations (residual 6e-16, 0.43 mm from the free line), hyperbolic with tr M +2.684,
  5.5–8.4 cm outside the boundary, above the plasma at φ = 0 (R = 1.194 m, Z = +0.182 m); its stellarator-symmetric
  image is the lower null. Trial A's line does not solve (Newton diverges). The double-null ladder uses trial B's
  settings.
  A Poincaré section of those coils (`_proto/poincare_xline.py`) shows the separatrix: nested diamond-shaped
  surfaces whose tips run into the X-points at Z = ±0.18 m. At that point they do not match the VMEX boundary
  (R 1.05–1.29 m, |Z| ≤ 0.17 m against R 1.01–1.37 m, |Z| ≤ 0.125 m), which is the rms B·n/B of 1.3e-2. Over the
  next three outer iterations B·n recovers (rms 3.2e-3) with the X-point kept (tr M 3.2, 7.4–10.6 cm out), but the
  rotational transform falls: |ι| 0.25–0.28 → 0.18–0.19, uniformly, with the minimum at the axis. This is not the
  separatrix pulling edge ι to zero (a fixed-boundary equilibrium does not see the coil field outside it); the
  B·n constraint drags the boundary toward the coil field's non-rotating surfaces, which carry less transform. With
  `plasma_objective.iota_weight` 10 the floor term (0.5·w·Δι² = 2.4e-2) is too weak, so the double-null ladder uses
  1e3; trial C (`dn_free_trial_C`, trial B's settings plus that weight) checks that the double null still forms.
  It does (tr M 4.2, 7–10 cm out) with ι held at 0.24–0.25, while trial B (weight 10) lets the X-point drift past
  the 12 cm band and ι fall to ~0.18.
* Circular-coil ladder, final (stage 3, cut at 9 outer iterations when B·n fell ~10 % per iteration while f_QS rose
  ~10 %): A 8.03, |ι| 0.379 (axis) / 0.400 (s = 0.5) / 0.385 (edge) — 0.04 short of its 0.42 floor at weight 10 —
  f_QS 0.012, largest B·n harmonic 3.4e-4, rms B·n/B 6.0e-4. The double-null ladder (`ladders/hsx_nfp4_dn_free.yaml`,
  `simflare/runs/combined_stage/hsx_nfp4_dn_free`) starts from it with the free X-line, field line first, ι floor
  0.37 at weight 1e3 (hold the seed's ι instead of also pushing to 0.42). After one outer iteration the X-line is
  hyperbolic (tr M 3.74, residual 1.8e-2, 5.2–6.8 cm out) and ι is held (0.369), but the QS term jumped 16×
  (6.1e-3 → 9.6e-2), against ~5× in trial C at ι 0.25: at this ι the double null and quasi-axisymmetry compete much
  harder.
* Escalated ι floor (`plasma_objective.iota_escalate`). Even at weight 1e3 the double null keeps pulling ι below its
  floor (trial C: min |ι| 0.249 → 0.239 over three outer iterations): a fixed quadratic weight only balances a
  persistent pull at some shortfall. With the option on, the ι-floor term enters VMEX at unit weight and the driver
  weights it separately — `VmexTermCost(plasma, "iota_floor")`, the cost of that term's residual rows, pulled back
  through the new `rows` output of `VmexPlasma` — and escalates that weight like the coil ranges (relative shortfall
  Δι/floor). `VmexQuasisymmetry(plasma, exclude=("iota_floor",))` then carries the rest of the VMEX objective.
  Check: `simflare/runs/combined_stage/_proto/verify_iota_term.py` (the split must equal the full cost in value and
  gradient). The value split is exact (1e-15). A pullback of the `rows` output equals that of `fqs` exactly; pulling
  the two parts back SEPARATELY differs by 1.8e-4 relative at VMEX's default adjoint tolerance and 6e-10 with
  `adjoint_tol=1e-12` — the implicit adjoint is an iterative solve, linear only to its tolerance. In the driver
  `WeightedSum` merges both cotangents into one pullback, so they share one solve. Smoke with the option on: the
  ι shortfall against a 0.30 floor went 0.16 → 0.096 in one outer iteration while its weight escalated 10 → 1e3.
* How the two 10-outer trials ended, both with a fixed ι-floor weight:

  | | trial B (weight 10) | trial C (weight 1e3) |
  |---|---|---|
  | rms B·n/B | 7.9e-4 | 2.2e-3 |
  | X-point tr M | 2.44 (margin 2.2) | 4.17 |
  | X-point distance | 9.0–12.0 cm (band edge 12) | 8.0–11.7 cm |
  | min ι (floor 0.25) | far below the floor | 0.212 |
  | QS term | ~1.7e-2 | 2.0e-2 |

  Both reach B·n consistency by giving up ι, fast at weight 10 and slowly at 1e3; trial C's aspect ratio also starts
  to drift. Trial B's final Poincaré section (`dn_free_trial_B/poincare_final.png`) shows the payoff and the price:
  the VMEX boundary lies on a coil-field flux surface inside a double-null separatrix, but the X-points sit 9–12 cm
  out with a weak tr M. The double-null ladder therefore runs with `iota_escalate`.
* Double-null ladder, stage 0 (`dn_introduce`), switched to `iota_escalate` after outer 2. Outer 3–4 kept
  improving (f_QS 0.048 → 0.039, rms B·n/B 7.3e-3 → 5.5e-3, tr M ≈ 5, X-point 6–8 cm out, ι held at 0.37), then
  outer 4–6 stalled at exactly that state while the augmented Lagrangian doubled its penalties every outer iteration
  (B·n 16000 → 128000). No coil or ι range was violated by more than 1 %; instead 26 of 27 rejected trials since the
  switch were VMEX status 1. VMEX's trial statuses (`vmex/core/implicit.py`, `_host_solve_and_mask_status`): 0 =
  derivative-certified equilibrium, 1 = failed solve (a typed `VmecError`, e.g. a Jacobian sign change: an unsolvable
  trial boundary), 2 = iteration budget exhausted above `max_fsq_ratio`. Loosening `ftol`/`niter` would not help. The
  barrier message now includes the `VmecError` text (`implicit._LAST_STATUS_ERROR`). Stage 0 was finalized at 7
  outer iterations and `dn_converge` restarts from its state with fresh penalties and a new Jacobi scaling.
  The logged failure is `VmecJacobianError: INITIAL JACOBIAN CHANGED SIGN!`. Each VMEX trial already walks a seed
  ladder — perturbation prediction, hot restart from the last converged state, then a cold start
  (`implicit._host_solve`) — and fails only when the cold start fails too, so the failure belongs to the trial
  boundary itself: its interpolated interior has a bad Jacobian from any seed (the same class as the bad-Jacobian
  Star_Lite boundaries in the VMEC boundary test). Better seeds or solver tolerances will not remove it; if it keeps
  limiting the steps, the boundary shape has to be kept regular (fewer boundary modes or a shape-regularity
  penalty). With fresh penalties the first `dn_converge` outer iteration moved again (f_QS 0.039 → 0.032, tr M 4.92,
  X-line residual 2.4e-3).
* Fallback bug (fixed). After an inner solve that ends on a rejected trial the driver continued from the most recent
  successfully solved evaluation — which can be a step the line search had already refused for insufficient
  decrease. `dn_converge` outer 1 was good (f_QS 0.027, rms B·n/B 5.7e-3, X-line residual 1.7e-4, tr M 4.84); in outer
  2 the first, oversized L-BFGS step solved but raised L from 0.08 to 5.7, the next trial failed VMEX, and the run
  continued from that bad step. The driver now keeps the lowest-L successful evaluation of each inner solve and
  continues from it whenever the solver's own end point failed or is worse. That run is archived in
  `simflare/runs/combined_stage/hsx_nfp4_dn_free/_superseded_fallback_bug/`.
* A boundary the optimizer reached is not necessarily a valid cold VMEX seed. Restarting from the outer-1 namelist
  failed in the seed solve with `INITIAL JACOBIAN CHANGED SIGN!`, also with the magnetic axis copied in from a nearby
  solved equilibrium (the driver's checkpoint namelists carry all-zero `RAXIS_CC`/`ZAXIS_CS`): that boundary was only
  ever solved hot-started. `dn_converge` was therefore rerun from the stage-0 final state, which solves cold.
* VMEX warm-start path dependence. The rerun reproduced outer 0–1 exactly, then crashed at outer 2: the inner solve
  ended on a rejected trial, the driver fell back to the best evaluation, and re-evaluating that point — solved
  minutes earlier — raised `INITIAL JACOBIAN CHANGED SIGN!`. VMEX seeds every trial from its latest converged state,
  then cold (`implicit._host_solve`); by then the latest state belonged to a different point, and the best point is
  not cold-solvable. (The old "most recent evaluation" fallback never hit this because that point's state was still
  the warm one.) Two fixes in the driver:
  - the best evaluation also keeps VMEX's solve of it (`implicit._LAST_SOLVE[config]`, a (parameter key, result)
    pair) and the warm-start state (`_HOT_CACHE`); `restore_vmex` puts both back before the fallback re-evaluation,
    so VMEX returns that stored result for exactly those parameters instead of solving again;
  - a resumed stage whose checkpoint does not solve from the stage seed walks from the seed to the checkpoint in
    fractions (steps of up to 0.25, halved on failure), each solve warm-started from the previous one.

  Both work: resuming `dn_converge` from outer 1 walked t = 0 → 1 in steps of 0.0625 and reproduced outer 1 exactly
  (f_QS 0.02689, tr M 4.843, X-line residual 1.72e-4); at outer 2 the fallback to the best evaluation ran without a
  jump or crash. But the stage stalled again at max_mode 3: the first L-BFGS step of each inner solve landed on a
  boundary VMEX cannot start from, and the solver stopped after one iteration while the penalties kept doubling — the
  same stall as `dn_introduce` outers 4–6. `dn_converge` therefore runs with `vmex.max_mode=2` (the stalled attempts
  are in `simflare/runs/combined_stage/hsx_nfp4_dn_free/_superseded_jacobian_stall_m3/`); trials B and C created and
  held their X-points at max_mode 2 with only occasional Jacobian failures. It restarts from the `dn_introduce` final
  state, since the plasma dof set changes and the outer-1 boundary cannot be solved cold.

  At max_mode 2 the Jacobian failures are rare (0, 4 and 6 in the first three outer iterations) and the X-line gets
  back to residual 2.8e-4, but B·n stays flat at a largest harmonic of 2.85e-3 (rms B·n/B 5.6e-3). The coils are out
  of room, with every engineering range binding at once:

  | range | limit | outer 2 |
  |---|---|---|
  | coil length | 3.5 m | 3.453–3.487 m, all four base coils |
  | max curvature | 10 /m | 9.85–10.24 /m (one peak per coil; mean ~4.2 /m) |
  | min coil–coil distance | 8 cm | 8.01 cm |
  | ι floor | 0.37 | within 0.13 % |

  Without the double null the same 32 coils under the same ranges reached a largest harmonic of 3.4e-4, so holding
  the X-line and ι leaves no coil freedom inside the HSX-size ranges. Trial D
  (`simflare/runs/combined_stage/dn_trial_D_relaxed_coils`) repeats `dn_converge` with length 4.0 m, curvature
  12 /m and coil–coil 7 cm to measure what the double null gains from that freedom.

  The reference run at the HSX-size ranges was stopped after outer 4: outer 3 still ran 28 inner iterations (f_QS
  0.034, X-line residual 1.8e-4, tr M 4.90, X-point 6.0–8.1 cm out), outer 4 ended after one on a rejected trial with
  nothing changed, and the largest B·n harmonic had stayed at 2.85e-3 from outer 0 on while the penalty grew to 32000.
  Its outer-4 checkpoint is kept in `hsx_nfp4_dn_free/01_dn_converge`.

  Trial D refuted the "no room inside the ranges" reading. With length 4.0 m, curvature 12 /m and coil–coil 7 cm
  the coils did not move at all in its first full outer iteration (lengths 3.452–3.486 m, peak curvature 10.2 /m) and
  B·n stayed at 2.88e-3; the one-sided range penalties were not what held them. The working hypothesis is the step
  itself: each L-BFGS step moves plasma and coil dofs together, and when its boundary part is a shape VMEX cannot
  start from, the whole step — coil motion included — is rejected; the Jacobi scaling makes those boundary steps
  large. `scaling.plasma_factor` (default 1) now shrinks the plasma dofs' scale, and trial E
  (`simflare/runs/combined_stage/dn_trial_E_plasma_step`) repeats `dn_converge` at the HSX-size ranges with factor
  0.1.

  That was it. With the plasma scale ×0.1 VMEX rejected no trial in outers 1–2 and 2 in outer 3 (against 37 in the
  seven outer iterations of `dn_introduce`), and every outer iteration ran its full 100 inner iterations and improved
  both the field match and quasi-symmetry with the X-point kept:

  | trial E | outer 1 | outer 2 | outer 3 |
  |---|---|---|---|
  | largest B·n harmonic | 2.85e-3 | 2.67e-3 | 2.41e-3 |
  | rms B·n/B | 5.39e-3 | 5.13e-3 | 4.72e-3 |
  | f_QS | 0.035 | 0.032 | 0.029 |
  | X-line residual | 1.9e-4 | 1.7e-4 | 1.7e-4 |
  | X-point tr M | 4.92 | 4.89 | 4.88 |
  | X-point distance | 6.1–8.2 cm | 6.3–8.4 cm | 6.8–9.0 cm |
  | max curvature | 11.0 /m | 10.7 /m | 10.1 /m |

  The HSX-size coil ranges are compatible with the double null after all; the stall was the step size. The
  double-null ladder's `dn_converge` stage now sets `scaling.plasma_factor=0.1`.

  The rest of trial E's 10 outer iterations continued the trend, now well below the old plateau:

  | trial E | outer 4 | outer 5 | outer 6 | outer 7 | outer 8 |
  |---|---|---|---|---|---|
  | largest B·n harmonic | 2.14e-3 | 1.46e-3 | 1.29e-3 | 9.6e-4 | 8.6e-4 |
  | rms B·n/B | 4.43e-3 | 3.99e-3 | 3.71e-3 | 3.49e-3 | 3.32e-3 |
  | f_QS | 0.0284 | 0.0281 | 0.0280 | 0.0287 | 0.0291 |
  | X-point tr M | 4.91 | 4.89 | 4.95 | 4.98 | 4.99 |
  | X-point distance | 6.9–9.3 cm | 7.7–10.3 cm | 8.0–10.6 cm | 8.4–11.2 cm | 8.7–11.6 cm |

  In outer 8 the escalated ι floor fired for the first time in a real run (ι 1.04 % below 0.37: weight 1e3 → 1e4),
  together with a 2.7 % coil–plasma violation. Outer 9 then stalled after one inner iteration while both weights kept
  escalating (to 1e5 and 1e7) on unchanged violations — escalation turns a small persistent violation into
  stiffness. The run is extended to 25 outer iterations (ctol 1e-5) from its outer-9 state with the weights reset
  to their configured values (the escalated ones are kept in `coil_weights.escalated_outer009.yaml`); a resume loads
  saved weights only if `coil_weights.yaml` exists.

  The extension (outers 10–24, no VMEX trial rejected; the resume walk reached the checkpoint in one step) settled
  into a plateau with every limit held:

  | trial E | outer 9 | outer 13 | outer 20 | outer 24 |
  |---|---|---|---|---|
  | largest B·n harmonic | 8.6e-4 | 7.2e-4 | 6.8e-4 | 6.6e-4 |
  | rms B·n/B | 3.3e-3 | 3.0e-3 | 3.0e-3 | 2.9e-3 |
  | QS term | 2.91e-2 | 3.42e-2 | 3.60e-2 | 3.70e-2 |
  | min ι (floor 0.37) | 0.366 | 0.3695 | 0.3694 | 0.3695 |
  | X-point tr M | 4.99 | 5.14 | 5.23 | 5.27 |
  | X-point distance | 8.7–11.6 cm | 9.5–12.3 cm | 9.6–12.0 cm | 9.7–12.0 cm |
  | X-line residual | 3.8e-4 | 6e-5 | 3e-5 | 1e-5 |

  The rise in f_QS is quasi-symmetry itself (the ι floor holds to 0.14 %, the aspect term stays ~5e-5): the
  optimizer now trades quasi-axisymmetry against the field match and the X-point rather than improving them
  together, with one coil at its length limit, curvature at 10 /m, the X-point at the 12 cm edge of its band and the
  coil–plasma weight at its cap. The constraint tolerance (1e-5) was not reached in 25 outer iterations.

  On the final coils the X-line Newton-solves in 2 iterations (residual 1e-15, 0.01 mm from the free line) as a
  hyperbolic X-point with tr M +5.278, 9.5–12.0 cm outside the boundary (R = 1.201 m, Z = ±0.234 m at φ = 0). The
  coil-field ι runs from 0.43 in the core through an island band at 10–7.5 cm inside the boundary (one traced line
  locks at exactly 2/5) to ≈ 0.25 at the boundary, and drops to ~0 three centimetres outside it. The Poincaré section
  (`dn_trial_E_plasma_step/poincare_final.png`) is much cleaner than at outer 3: nested surfaces with the VMEX
  boundary on an outer one, island chains 2–6 cm outside the boundary, then the X-points — the boundary island
  chain of outer 3 now lies outside the plasma.

  At outer 3 the X-line Newton-solves as a genuine X-point (2 iterations, residual 4e-14, 0.2 mm from the free line;
  hyperbolic, tr M +4.887; 6.7–8.4 cm outside the boundary, at R = 1.205 m, Z = ±0.204 m at φ = 0). The Poincaré
  section (`dn_trial_E_plasma_step/poincare_outer003.png`) is not yet the clean picture of trial B: nested surfaces
  in the core, a chain of about ten small islands on and just inside the VMEX boundary, a thin chaotic layer, then the
  X-points. I first read the chain as ι = 4/10 (edge ι of the fixed-boundary equilibrium ≈ 0.38–0.40). The coil
  field says otherwise. Its rotational transform, traced from the magnetic axis outward
  (`simflare/runs/combined_stage/_proto/coil_iota_profile.py`), is strongly sheared:

  | R − R_boundary (outboard midplane) | −19 cm | −11.3 cm | −10.4 cm | −9.4 cm | −4.6 cm | −0.8 cm | +0.1 cm | +3 cm |
  |---|---|---|---|---|---|---|---|---|
  | coil-field ι | 0.437 | 0.421 | 0.370 | 0.340 | 0.283 | 0.249 | 0.260 | 0.208 |

  ι = 0.4 lies 11 cm inside the boundary, in a steep drop (0.42 → 0.34 over 2 cm, through 2/5, 4/11 and 1/3), and the
  edge ι is ≈ 0.25: the chain at the boundary is most likely ι = 1/4 = 4/16 (m = 16; the island count from the plot
  was rough). The fixed-boundary VMEX profile was nearly flat at 0.38–0.40; while B·n still disagrees (rms 4.7e-3) the
  coil field near the double-null separatrix carries far less edge transform than the equilibrium the ι floor
  constrains. (A fixed-boundary VMEX solve of this checkpoint fails cold, like the other optimizer checkpoints, so
  the coil-field trace is the measurement.)
* Engineering weights escalate (`coils.escalate`, as in `array/boozer_all.py`): after each outer iteration
  every range violated by more than `escalation_tol` gets its weight × `escalation_factor`; weights are logged in
  `history.yaml` and kept across `--resume` (`coil_weights.yaml`). Without it the first outer iteration from
  circular coils (100 inner it.) took B·n from 9.7e-2 to 1.3e-2 but let the maximum curvature run from 2.7 to
  19.1 /m (limit 10): a fixed curvature weight of 1e-3 is invisible next to the augmented-Lagrangian terms.
  boozer_all's tolerance (0.1 %) without a cap overshoots the other way: in stage 2 a 0.14 % coil–coil violation
  (0.1 mm) drove that weight to 1e9 and a 1.4 % curvature excess drove curvature to 100, L jumped from 0.02 to 0.88
  and the inner solves shrank to 9–20 iterations with rejected VMEX steps. Defaults are now
  `escalation_tol` 1e-2 and `escalation_max_factor` 1e4 (weight ≤ 1e4 × its configured value).
* **The constrained B·n harmonics must cover the coil ripple.** After 4 outer iterations with escalation the
  constrained block (m ≤ 4, |n| ≤ 4) was at 3e-4 and every coil range held, yet rms B·n/B was 5.7e-3 and max
  2.1e-2: only 0.6 % of the B·n energy lies in that block (`simflare/runs/combined_stage/_proto/bn_spectrum.py`).
  |n| = 8 per field period — the ripple of 8 coils per period — carries 55 %, n = 5–7 29 %, n = 9 5 %; m ≤ 6
  holds essentially all of it. `consistency.ntor` has to reach at least the number of coils per field period
  (+1), and `vmex.nphi` > 2·ntor. Whether 32 coils can cancel that ripple at 15 cm clearance at all is checked
  by `_proto/ripple_feasibility.py` (fixed boundary, coils only) before widening the constraint set.
  It can: on the fixed outer-3 boundary the same 32 coils reach rms B·n/B 7.0e-5 (max 2.1e-4) within every
  range (curvature 8.7 /m, length 3.5 m, coil–coil 10.8 cm, coil–plasma 24 cm); 40 fresh circles reach 5.8e-5
  with curvature 5.1 /m. The ladder therefore constrains m ≤ 6, |n| ≤ 9 (123 harmonics + flux) and was
  restarted from the outer-3 coils and boundary (`run_ladder.py --coils`, sbatch `COILS=`; the old stage 0 is in
  `simflare/runs/combined_stage/hsx_nfp4_circular_vacuum/_superseded_bn_m4n4/`). A changed constraint set cannot
  `--resume` — the augmented-Lagrangian multipliers are per constraint.
* Stage budgets: `augmented_lagrangian.outer_maxiter` counts over the whole stage — `--resume` continues after
  the newest `coils_outerNNN.json` — so re-submitting a ladder past a stage's budget writes that stage's final
  outputs and moves on. The nfp = 4 ladder gives stage 0 six outer iterations and stages 1–2 ten with
  `ctol` 1e-4; only the last stage runs to 25 / 1e-5. (Stage 0 at max_mode 1 flattened after outer 3–5:
  f_QS 0.0079 → 0.0076, rms B·n/B 7.8e-4 → 6.2e-4, ~20 min per outer iteration.)

Planned stages: (1) vacuum reshaping HSX variant (paused — see above; `simflare/runs/combined_stage/hsx_variant_vacuum`); (2) continue to A = 10.5, ι → 0.7–1, more coils if
needed; (3) β ladder (1 % / 3 %) with self-consistent bootstrap current and virtual casing, double
null checked in the coil + plasma field; (4) free-boundary re-solve, Mercier/ballooning, scaling.

### Self-intersecting boundaries, virtual casing, and the boundary guard (2026-09-15)

**What went wrong.** The double-null boundaries from `dn_introduce` outer 4 on — trial E, `dn_3types_24coils_L5p25`
(job 792) and the first finite-beta attempts (jobs 794/795, 804/805) — carry a tiny self-intersecting loop at the
phi = 0 outboard tip: trial E final 0.21 mm (R) x 0.84 mm (Z) over 24 deg of theta, self-intersecting on the cross-sections
phi nfp / 2 pi = 0 ... 0.078 (simsopt `is_self_intersecting`). The boundary's poloidal speed |dx/dtheta| drops to 3e-3 of
its median there (curvature ~8e4 /m). History of that speed ratio / max curvature near (phi 0, theta 0): circular-coil
ladder 0.63-0.79 / <= 12 /m in every stage; `dn_introduce` outer 0-3: 0.45, 0.35, 0.19, 0.084; **outer 4: 5e-3 / 8e4 /m**
(the loop forms, exactly where that stage began rejecting 26 of 27 VMEX trials with INITIAL JACOBIAN CHANGED SIGN).
The driver had no check, and VMEX's hot restarts keep solving such boundaries.

**Consequences.** VMEC's Jacobian is degenerate at that point (sqrt_g 500x below its median; `ru12` = 0 by symmetry and
`zu12` -> 0), so B^u = B . grad theta blows up (162 at the solver grid 18 x 16, 288 at 32 x 32; B^v stays finite).
The truncated B^u spectrum turns this into a spike of the boundary field on any resampled grid (2.4-3.8 T at phi = 0
where |B| from `bmnc` is 0.9-1.2 T; `virtual_casing.surface_field_data_from_state` and `..._from_wout` alike). Virtual
casing integrates that field: with ZERO pressure it returned |B_plasma| = 2 % of |B| with 3.5 T outliers, and it needed
229 GB and 126 s per value + gradient at a 24 x 20 grid (32 x 32 at 3 digits: 1067 GB; jobs 794/795 were OOM-killed).
`pressure_balance` uses the same field. Cold VMEX solves of these boundaries fail (INITIAL JACOBIAN). The vacuum-coupled
constraints (B.n harmonics, toroidal flux) read only the boundary geometry and are not affected; the loop's effect on
f_QS / iota / B.n of those runs is unmeasured. Probes: `simflare/runs/combined_stage/_proto/vc_memory_probe.py`,
`vc_sanity_probe.py`, `boundary_field_outliers.py`, `boundary_field_rows.py`, `boundary_field_spectra.py`,
`boundary_field_solver_grid.py`, `boundary_field_realspace.py`, `boundary_field_terms.py`.

**Re-tested on clean boundaries (2026-09-15, jobs 827 / 828; logs `_proto/logs/vcfix_*`).** Same probes, config and grids,
on repair A, the clean-lineage final (`dn_clean_guard`) and designA x 2: boundary field vs |B| from bmnc max error
2.49 T → 0.002 T (0 of 1024 points above 0.1 T); B^u at (theta 0, zeta 0) 444x → 2-9x its surface median; sqrt_g ratio
0.002 → 0.3. Virtual casing (24 x 20, 3 digits): |B_plasma| at beta 0 median 0.12 % / max 0.88 % of |B| (was 2 % / 3.5 T);
at beta 1 % median 0.94 % / max 2.2 % (8x the beta-0 noise); **peak memory 6.8 GB (was 228 GB), 7.6 s + 27.5 s per value +
gradient** (was 56 s + 70 s). So virtual casing is usable on regular boundaries; before relying on it re-check its ~2 %
gradient inconsistency (designA tests, independent of the fold) and a finer grid (beta-0 noise max vs beta-1 % signal).

**Boundary guard.** `utils/vmex_combined_stage.boundary_regularity(plasma)` evaluates, from the plasma dofs alone (no
solve), the minimum |dx/dtheta| relative to its cross-section median (exact Fourier derivatives, 8 cross-sections over
the symmetry period) and whether any cross-section self-intersects. `combined_stage_vmex.py` rejects such trials with the
barrier before VMEX runs (`boundary_guard: {min_speed_ratio: 0.1, nphi: 8, ntheta: 512}` in `config.yaml`, 0 switches it
off), prints the start boundary's value and logs `boundary_min_speed_ratio` every outer iteration.

**Repair.** `tools/repair_boundary.py` removes the degeneracy with the smallest shape change (displacement along the
original normal, speed ratio >= 0.15 on every cross-section, varying either the `vmex.max_mode` modes or all modes) and
reports the geometric change and self-intersections on 32 cross-sections. For trial E's final boundary the max_mode-2
repair (`seeds/input.trialE_final_repaired_m2_normal`) has no self-intersection and speed ratio 0.148 at a change of
4.2 mm max (at the tip), 1.3 mm rms, +0.5 % volume, and solves cold in VMEX (aspect 7.999, iota axis/edge 0.399/0.361;
its beta = 1 % copy too) where trial E's original boundary does not. All-mode repairs still self-intersected (optimized
on 16 cross-sections: 12.5 mm max change; on 48 cross-sections x 1440 theta: 18.9 mm, and it fails cold), so the
max_mode-2 repair is the only usable one — a 4 mm change, not sub-mm.

**Runs from here (2026-09-15).** Both routes, each for 24 and 32 coils, coil length 5.0 m, beta 1 % with vacuum coupling:
(a) from the repaired boundary: `simflare/runs/combined_stage/dn_3types_24coils_L5_beta1` (job 816) and
`dn_trialE_32coils_L5_beta1` (job 817), seed `seeds/input.trialE_final_repaired_m2_normal_beta1`, trial E's X-line;
(b) a clean lineage that never had the loop: `dn_clean_guard` (job 821, `configs/hsx_nfp4_dn_clean_guard.yaml` = trial E's
settings + guard, vacuum) restarts from `dn_introduce` outer 1 (speed ratio 0.35; outers 0-2 solve cold), and
`sbatch/beta_followup.sbatch` (job 822, afterok) then seeds and submits `dn_clean_24coils_L5_beta1` and
`dn_clean_32coils_L5_beta1` from its final state.

### designA x 2 at beta 1 % (2026-09-15)

`configs/designA_L2_beta1.yaml` optimizes designA scaled uniformly by 2 (R 1.0 m, a 0.15 m, A 6.67, nfp 2, its 6 coils)
with the coil currents doubled (same |B|, 0.087 T on axis) at beta 1 %, keeping the double null as a tracked X-point
(designA's solved periodic field line, scaled with the device). `tools/scale_device.py` scales the design archive (through
`simflare/workflows/scaling/scale_design.py`, which now also scales the X-point lines, all configs' surfaces and coils,
the Boozer label targets and G) and the seed namelist (boundary x 2, PHIEDGE x 4), and checks the copy: every quantity
lands on its expected ratio exactly (X-point tr M 2.987, 9.7-19.4 cm outside the boundary; rms B.n/B 5.7e-4; currents
55 / 82 kA, so the 60 kA cap becomes 120 kA). Coil ranges are designA's x 2 in length, except curvature (5.3 /m),
mean-squared curvature (5.1) and coil-coil distance (0.29 m), which sit just outside the scaled designA coils because
designA itself does not satisfy the default curvature range. Beta 1 % at this shape with zero net current lowers |iota|
(converged solve, mpol = ntor = 14, ns 101) from 0.236 / 0.190 (axis / edge) to 0.123 / 0.131; the run's 0.13 floor was
set from the ns 16, mpol 6 value (min |iota| 0.137) and lies above the converged 0.120. The run (job 826) failed at outer 7
(plasma variables effectively frozen by the scaling, X-point lost, driver fallback bug) and ran on a TRUNCATED boundary.

**Resolution audit (2026-09-15).** `vmex.mpol` / `vmex.ntor` truncate the seed boundary (`VmecInput.change_resolution`
keeps m < mpol, |n| <= ntor); every config sets 6 / 6 for speed. For designA x 2 (native MPOL 11 / NTOR 10) that moves the
boundary 3.1 mm max / 1.1 mm rms from designA's Boozer surface (the full namelist: 0.38 / 0.08 mm), and a least-squares
refit at 6 / 6 or 8 / 8 does not recover it (2.8 / 2.5 mm max): the dropped m = 9-10 content is shape, not an
angle-fit tail as this README used to claim (`simflare/runs/combined_stage/_proto/boundary_vs_boozer.py`,
`tools/refit_boundary_resolution.py`). Its effect on the equilibrium is small against a converged solve (f_QS 0.1 %, edge
iota 0.7 % vs mpol = ntor = 14; the native 11 / 10 solve is itself not converged at beta 1 %). The HSX-lineage boundaries
are native 6 / 6, so nothing was truncated, and 6 / 6 agrees with 12 / 12 to 0.2 % (f_QS) and 0.5 % (iota). The single
radial grid ns = 16 is the larger error: iota on axis moves 5-7 % (HSX) and up to 20 % (designA x 2, beta 1 %) toward
ns = 101, which biases the iota floor wherever min |iota| sits on axis
(`simflare/runs/combined_stage/_proto/resolution_convergence.py`, logs `_proto/logs/rc_*`).
Seeds: `simflare/runs/combined_stage/seeds/prep_designA_L2.sh`; run: `simflare/runs/combined_stage/designA_L2_beta1`.

**Seeding tools added alongside.** `tools/reduce_coil_types.py` (fewer coil types by blending neighbouring base coils,
currents x n_old/n_new), `tools/make_beta_seed.py` (pressure profile p = PRES_SCALE (1 - s) calibrated to a volume-
averaged beta), `combined_stage_vmex.py --plasma-target <namelist>` (walk the plasma dofs from a cold-solvable seed to a
boundary that only solves warm), `sbatch/run_stage.sbatch` (one resumable run), `configs/` (per-run configs).

## Implementation

| file | contents |
|---|---|
| `utils/augmented_lagrangian.py` | `AugmentedLagrangian` (a simsopt Optimizable: `J`, `dJ`), `EqualityConstraint`, Nocedal & Wright Alg. 17.4 outer update, checkpoint state |
| `utils/vmex_combined_stage.py` | `VmexPlasma` (dofs `s` + cached VMEX evaluation), `VmexQuasisymmetry`, `PlasmaCoilInterface`, `CoilPlasmaDistance`, `WeightedSum` |
| `combined_stage/combined_stage_vmex.py` | driver: builds everything from `config.yaml`, scaled variables, failed-solve barrier, outer ALM loop, checkpoints |
| `tests/utils/test_augmented_lagrangian.py` | fast ALM unit tests (KKT point, multipliers, Taylor test, shared pullback, small ρ₀, infeasible problem) |
| `tests/utils/test_vmex_combined_stage.py` | slow designA integration tests, opt-in via `VMEX_COMBINED_STAGE_TESTS=1` |
| `utils/vmex_double_null.py` | `XpointHyperbolicity`, `XpointPlasmaDistance` (double-null terms) |
| `tests/utils/test_vmex_double_null.py` | Taylor tests of the double-null terms (coils + a fake plasma owner), shared pullback |
| `combined_stage/run_ladder.py`, `combined_stage/ladders/` | continuation ladders seeded stage to stage |

**One VMEX adjoint per gradient.** `VmexPlasma` runs ONE jitted forward
evaluation per dof vector returning `f_QS` and the boundary data (points,
normals, area weights, `|B_in|²`, and the plasma's own field). It keeps the JAX
pullback. Consumers (`VmexQuasisymmetry`, `PlasmaCoilInterface`,
`CoilPlasmaDistance`) return cotangents for those outputs; `WeightedSum` and
`AugmentedLagrangian` add them up and call `VmexPlasma.pullback` once. The coil
field's dependence on where the boundary points are enters through simsopt's
`dB_by_dX`.

**Scaled variables.** The optimizer works in `u` with `x = x_seed + D u`:
VMEX's ESS scales for boundary modes (`scaling.plasma_step`), a uniform
`scaling.coil_step` for coil dofs.

**Failed VMEX solves** become the same smooth quadratic barrier toward the last
good point that `array/boozer_all.py` uses, so line searches retreat.

## Running

The seed needs a VMEC namelist of the starting boundary, e.g. from
`simflare/workflows/hint/run_vmec_from_json.py <design json> --beta-percent 0.03`,
and an interior boundary (the true LCFS of designA does not solve in VMEX).

```bash
cd combined_stage
srun --partition=compute --cpus-per-task=32 --mem=64G --time=24:00:00 \
  python combined_stage_vmex.py --config config.yaml \
    --vmec-input /path/to/input.<seed>.fixed --outdir output/<tag>
# add --smoke for a 2 x 3 iteration end-to-end check, --resume to continue
```

Outputs in `--outdir` (`run_layout.py`; runs from before 2026-09-14 are flat and still read):
`config_used.yaml` and `history.yaml` (per outer iteration: f_QS terms, constraint norms, rms B·n/B, pressure
balance, coil metrics, penalties, VMEX solve counts) at the top; `state/` (`checkpoint.npz`, `al_state.yaml`
with multipliers, penalties and tolerances, `scaling_D.npy`, `coil_weights.yaml`), `coils/coils_<tag>.json`,
`xpoints/xpoint_<tag>.json`, `inputs/input.combined_stage_<tag>`, `wout/wout_combined_stage_final.nc`,
`logs/`, `figures/`. The stage's input state is kept as `coils/coils_seed.json`, `xpoints/xpoint_seed.json` and
`inputs/input.combined_stage_seed`. `snapshots/` rebuilds the optimization history: `snapshot_<n>.json` (a simsopt
archive of a dict: `coils`, `boundary` SurfaceRZFourier + `vmec_input` namelist text, `magnetic_axis` CurveRZFourier of
the VMEX equilibrium, `xpoint` curve + length, `label`, `outer`, `inner_iteration`, `L`) at the stage start, every
`history_every` (default 5) L-BFGS-B iterations, at each outer iteration's end and at the end, indexed in
`snapshots/snapshots.yaml`; load one with `simsopt._core.load`.

Device figures like the array campaign's (`postprocess/`): `RUN=<run dir> TAG=final sbatch
--output=<run dir>/logs/device_%j.out postprocess/render_device.sbatch` builds `<run dir>/device/` — a design
archive (Boozer surface, axis and X-point solved on the run's coils), `summary.txt`, LCFS and 90/80/70 % flux
surfaces and the cross-section data — and writes every figure into `<run dir>/figures/`: `xs_manifolds.png`
(`array/mk_manifolds.py`, `plot_manifolds.py`), the nine-angle `poincare_grid.png` (`poincare_grid.py`) and
`scene_iso/top/left/right.png` (`render_views.py`, envs/vtkrender).
Older runs and smokes were moved to `simflare/runs/combined_stage/_archive/` on 2026-09-14; the paths quoted
above for them now carry that prefix.

## Validation status (2026-09-10)

Seed: designA coils + its optimization surface as a VMEC namelist (β = 3e-4), ns = 16,
max_mode 2, 16×16 interface grid.

| check | result |
|---|---|
| ALM unit tests (`test_augmented_lagrangian.py`) | 6/6 pass: analytic KKT point and multipliers at finite ρ, nonlinear case, Taylor test, one shared pullback per gradient, tiny ρ₀, infeasibility → penalty growth |
| designA coils vs VMEX boundary, vacuum | rms (B·n)/|B| = 2.8e-4, max B·n harmonic 1.8e-5, toroidal flux vs PHIEDGE 1.1e-3 |
| field direction (VMEX B vs coil B) | cos ≥ 0.999995, |B_coil|/|B_vmex| = 1.0009 |
| coil gradient vs central FD | 5e-9 relative (h = 1e-6), converges as h² |
| plasma gradient vs central FD, **vacuum** + `toroidal_flux` | 2.7e-4 relative — the FD floor of VMEX's own certified f_QS gradient (unchanged at ftol 1e-14) |
| VMEX adjoints per gradient | exactly 1 |
| `vacuum` integration suite | 5/5 pass |
| plasma gradient, **virtual_casing** + `pressure_balance` | 2.3 % vs re-solve FD — expected failure, explained below |
| driver `--smoke` (2 outer × 3 inner) | runs end to end; f_QS 0.060 → 0.045, rejected VMEX trials handled by the barrier, all outputs written; ~20 s per evaluation |

**Boundary-field gradients (virtual_casing / pressure_balance) — not a VMEX bug.**
Adjoint gradients of B read at VMEC grid points (edge field, |B|², field direction,
interior surfaces) differ from central *re-solve* finite differences by 0.5–4 %
(up to ~15 % in field direction). Diagnosis (probes in `simflare/runs/combined_stage/_proto/`):

| test | result |
|---|---|
| adjoint vs VMEX's `frozen_path_directional_fd` (its recommended reference) | ≤ 1.1e-6 for all B outputs, 5.8e-8 for f_QS (Newton residual 3e-12) |
| adjoint vs independent forward tangent solve / multi-RHS pullback (gcrot) | agree to the same level |
| field construction at fixed state (jvp vs FD) | exact (3e-9 in λ, 5e-6 in geometry) |
| virtual casing w.r.t. its inputs | exact (1e-8 / 1e-12) |
| hot vs cold host solve of the SAME boundary | different converged states (λ, interior R/Z m=1 content) with |F| ≈ 1e-10; B values differ ~1e-5 relative, grid-point derivatives by percent |
| cold-solve FD vs hot-solve FD at h = 1e-5 | up to 2× apart |
| ns 16 → 51, adjoint_tol 1e-10, ftol 1e-14 | no change |

VMEX's implicit adjoint differentiates the fixed point of the **frozen** residual
(preconditioner, dof mask, released m=1 combination, λ axis row captured at the base
point), as its documentation states. The discrete equilibrium is not unique along a
near-null direction, and a re-solve drifts along it in a history-dependent way, so for
quantities that read the solver state at grid points the re-solve map has no unique
derivative. Gauge-free quantities (f_QS, boundary geometry) agree to the FD floor.

A local fix was attempted and **failed validation**: evaluating the equilibrium along the
frozen path (Newton on the anchored frozen residual, from the anchor state or from a
host solve) diverges for boundary steps of 1e-3 × ESS scale (the anchored residual of a
host state is O(1); pinning the released m=1 direction while the boundary moves is too
stiff), and VMEX's direct `solve_implicit_with_aux` fails cold on designA
(`INITIAL JACOBIAN CHANGED SIGN`) where the hot-restarted path succeeds. It was removed.

Consequences: `vacuum` + `toroidal_flux` touch no grid-point field and are fully
consistent — use them for Star_Lite (β ≈ 3e-4). With `virtual_casing` / `pressure_balance`
the augmented-Lagrangian gradient along boundary dofs is ~2 % inconsistent with the
evaluated function (its test is an expected failure); quasi-Newton methods usually tolerate
that, but it is untested in a full run. A proper finite-beta fix belongs in VMEX (a
gauge-canonical state after each solve, or a boundary-field evaluation that does not
extrapolate interior half-mesh values) — worth raising with R. Jorge with these probes.

**QS metric normalization.** f_QS is VMEX's two-term QS residual summed over the sampled
surfaces; star_lite_design optimized `NonQuasiSymmetricRatio` on Boozer surfaces. Both vanish
at exact QS but are normalized differently: designA's optimization surface is 1.26e-3
(√ = 0.036) in star_lite_design's metric and 0.119 in VMEX's (s = 0.1…0.9); LCFS70 1.44e-3 vs
0.139. Same ranking, ~100× different scale — do not compare the numbers across codes, and
set augmented-Lagrangian penalties relative to the f_QS scale actually used.

## Known limitations / TODO

* **No free-boundary acceptance test yet.** The real test of consistency is a
  free-boundary VMEX solve from the optimized coils reproducing the optimized
  boundary; not implemented in the driver.
* **Stellarator-symmetric only.** VMEX's live-state boundary-field path
  (`_state_field_spectra`) raises `NotImplementedError` for `lasym = True`, so
  single-null (SN) devices are not supported yet.
* **Engineering and edge terms are penalties, not constraints**; no vessel,
  X-point or divertor terms yet (`utils/` has differentiable coil-field versions
  that could become constraints).
* Single radial grid during optimization; resolution continuation is manual.
* Finite-beta (`virtual_casing` / `pressure_balance`) gradients are ~2 % inconsistent
  with the evaluated (re-solve) map; untested in a full optimization.
* Not yet run: a production-length optimization, and the stage-1-only
  reproduction (frozen coils, no constraints) of the VMEX QA results.
* VMEX rejects some trial boundaries (`INITIAL JACOBIAN`/non-convergence); they
  are handled by the barrier but cost evaluations.

## Open questions (for A. Giuliani)

1. Per-harmonic B·n constraints (implemented) vs per-point vs the scalar
   `f_BdotN = 0` (degenerate: zero gradient at feasibility).
2. How to pin the field strength: pressure balance (implemented) vs fixing the
   total coil current.
3. Scale-up targets (size, field, beta) and whether the coil engineering terms
   should also become augmented-Lagrangian constraints.
