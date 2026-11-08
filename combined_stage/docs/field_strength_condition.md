# The field-strength condition: Boozer G, pressure balance, toroidal flux

This file covers how the combined stage-1/2 optimization ties the coils to the
strength of the VMEX equilibrium field:

* why B·n = 0 needs one more condition;
* the Boozer-G condition;
* pointwise pressure balance, implemented the way VMEX's finite-β example does it;
* the end-of-run check that applies to every run.

Code: `utils/vmex_combined_stage.py` (`PlasmaCoilInterface`,
`net_poloidal_current_output`, `PressureBalancePenalty`, `interface_check`),
driver `combined_stage/combined_stage_vmex.py` (`consistency.field_strength`,
`final_check`), examples
`combined_stage/examples/`.

## 1. Why B·n = 0 is not enough

The plasma and the coils each set their own field strength:

* **VMEX** solves a *fixed-boundary* equilibrium. Its field strength comes from
  its inputs: PHIEDGE (edge toroidal flux), the pressure profile in pascals,
  and the current profile.
* **The coils** produce `B_coil` by Biot–Savart. Its strength comes from the
  coil currents.

The consistency constraint `(B_plasma + B_coil)·n = 0` on the boundary only
requires the vacuum-side field to be *tangent* to the surface. If every coil
current is multiplied by the same factor λ, B·n stays zero. So B·n leaves one
direction in coil space free: the overall current scale. In a vacuum problem
the cheapest way to satisfy B·n is then to switch the coils off.

**Vacuum (β = 0, no net current).** Scaling |B| does not change the VMEX
geometry, and PHIEDGE only relabels the device. VMEX's own vacuum example
(`vmex/examples/optimization/single_stage_optimization.py`) relies on this. It
keeps the coil currents fixed, normalizes B·n by |B_coil|, and has no scale
condition at all.

**Finite β.** The pressure is fixed in pascals, so β = 2μ0⟨p⟩/B² depends on
the field strength. A coil set that is too strong or too weak describes a
different equilibrium from the one VMEX solved. The scale condition is then a
physics constraint. It is also needed whenever coil currents are free under a
current cap, because the scale is what gives "60 kA" a meaning.

So every run needs **exactly one** mechanism that fixes the scale. In the
driver that is the scalar `field_strength` equality on `PlasmaCoilInterface`;
it refuses to start without one unless `coils.fix_currents` is set. The
pointwise pressure-balance penalty (section 4) also fixes the scale, but it is
used only in `examples/combined_stage_simple.py`.

## 2. The plasma's own field in B·n (virtual casing)

At finite β the plasma currents (diamagnetic, Pfirsch–Schlüter, bootstrap)
produce a field `B_plasma` with a normal component on the boundary. The coils
must cancel it, so the constraint is

    B_coil·n = −B_plasma·n,   not   B_coil·n = 0.

VMEX gives only the total field `B_in` on a prescribed boundary. The
virtual-casing principle recovers `B_plasma` from it: outside the boundary,
the plasma's volume currents produce the same field as the surface current
`K = n × B_in / μ0`. That needs only the boundary field, and for a fixed trial
boundary it does not depend on the coils. Set
`vmex.plasma_field: virtual_casing` for this. `vacuum` sets `B_plasma = 0`,
which is right only at Star_Lite's β ≈ 3e-4. At β = 0.93 %,
rms `B_plasma·n/|B|` = 5.8e-3 (job 2119), so vacuum-coupled β ≈ 1 % runs carried
an error of that size.

This is the same B·n term VMEX's finite-β example uses
(`interface.bnormal_residual = Bn_plasma + external_Bn`). We constrain its
low-order Fourier harmonics as AL equalities; VMEX penalizes every grid point.

## 3. The options

| mechanism | form | valid at finite β | gradient | notes |
|---|---|---|---|---|
| `field_strength: toroidal_flux` (default) | AL equality `(∮A_coil·dl − Φ_target)/|Φ_target|`, φ = 0 cross-section, `|Φ_target|` = PHIEDGE | **no** | analytic | the coil flux misses the plasma's diamagnetic flux; exact in vacuum |
| `field_strength: pressure_balance` | AL equality, area-weighted **mean** `⟨|B_out|² − |B_in|² − 2μ0 p_edge⟩ / B_ref²` | yes | ~2 % virtual-casing error | legacy (the hsx_*_vc_* runs); one average, says nothing about individual points |
| `field_strength: net_poloidal_current` | AL equality `(I_coil − 2πG/μ0) / |I_seed|` | **yes, exactly** | adjoint vs frozen-path FD 2.8e-5 (job 2119); coil part 2.4e-9 (job 2312) | one exact scalar; fixes the scale only |
| `PressureBalancePenalty` (**example only**) | objective term `w · ½ Σᵢ [√aᵢ (|B_out|² − |B_in|² − 2μ0 p_edge)ᵢ / B_ref²]²`, **every grid point** | yes | ~2 % virtual-casing error (plasma dofs); coils exact | VMEX's finite-β formulation; fixes the scale **and** the local |B| pattern |

### The Boozer-G condition

In Boozer coordinates `B = G(ψ)∇φ + I(ψ)∇θ + K∇ψ`:

* `G(ψ)` = μ0/2π × the **poloidal** current linked by a toroidal loop on the
  surface ψ. That is the current through the torus hole *outside* ψ. VMEX's
  name is `bvco`; at the boundary, `rbtor = 1.5 bvco(ns) − 0.5 bvco(ns−1)`.
* `I(ψ)` = μ0/2π × the **toroidal** current *inside* ψ (net current,
  bootstrap, CURTOR). VMEX's name is `buco`.

Take Ampère's law on a loop C that runs once toroidally *on the boundary*. The
disk bounded by C is crossed once by every modular coil, through the hole. The
plasma's poloidal currents flow inside the boundary and never cross it. So

    ∮_C B·dl = 2π G_edge = μ0 I_coil,

**exact at any β**. On an inner surface ψ, the poloidal plasma currents between
ψ and the edge do cross the disk, which is why G(ψ) varies inside a finite-β
plasma. At the edge only the coils are left. In vacuum G is constant, and the
condition is equivalent to `toroidal_flux`. The bootstrap current changes
**I**, not G (bootstrap consistency is a separate block, `BootstrapConsistency`).

Implementation:

* **Plasma side.** `VmexPlasma(..., extra_outputs={"rbtor": net_poloidal_current_output()})`
  returns `G` from the same jitted VMEX forward as everything else, so its
  gradient is one more scalar cotangent on the shared adjoint. It is converged
  at ns = 16 (1.7e-6 vs ns = 101). The driver adds it automatically.
* **Coil side.** `I_coil = ∮B_coil·dl / μ0` on a fixed z = 0 circle at the
  seed's R00, through the plasma, which every modular coil links once. **Do not
  sum the coil currents instead.** simsopt's stellarator-symmetric copies carry
  flipped signs, so the plain sum is exactly 0. On the band-run final it gave 0
  against 6.29 MA from the loop integral.
* **Sign.** The relative sign of `I_coil` and `G` (loop vs VMEX angle
  orientation) is fixed once at the seed.

## 4. Pointwise pressure balance, as in VMEX's finite-β example

The ideal-MHD jump condition at the plasma–vacuum interface is total-pressure
continuity, **at every point** of the boundary:

    p_edge + |B_in|²/2μ0 = |B_plasma + B_coil|²/2μ0.

`vmex/examples/optimization/single_stage_optimization_finite_beta.py` imposes
it like this:

```python
pressure_rows = jnp.sqrt(interface.weights).ravel() * (
    interface.pressure_balance_residual(external_field) / B_scale**2).ravel()
cost = 0.5 * PRESSURE_BALANCE_WEIGHT * jnp.vdot(pressure_rows, pressure_rows)   # weight 1e2
```

The example uses one row per virtual-casing grid point, area weights summing to
1, and a quadratic penalty in the objective next to the B·n penalty (weight
1e3). The coil currents are free, and there is no G or flux term: pressure
balance is what fixes the scale, because scaling the coils changes |B_out|.

`PressureBalancePenalty(interface)` is the same term. It uses the same rows
(`PlasmaCoilInterface.pressure_balance_rows()`), the same normalization and the
same penalty form, and it goes into the `WeightedSum` objective with its weight.
It is used **only in `examples/combined_stage_simple.py`** (weight 1e2,
`field_strength=None`), as the reference implementation of VMEX's formulation;
the driver does not offer it, and production runs use the G condition. The only
difference is that `B_ref` is the seed's `sqrt(⟨|B|²⟩)`, frozen. VMEX
recomputes `B_scale` at every iterate; the two differ only by the scale change
the rows themselves control.

**Why a penalty and not an AL equality block.** There is one row per grid point
(480 at 24×20, 1024 at 32×32). That is more conditions than the boundary and
coil dofs can meet exactly; this is also why B·n is constrained only through
harmonics. As an equality block the AL would keep growing its penalty without
converging. As a penalty it is minimized in least squares, the way VMEX does it.

Gradient: the coil part is exact (simsopt `B_vjp`). The plasma part goes
through the virtual casing and carries its ~2 % gradient inconsistency (README,
Validation status). It shares the interface's cache and the single VMEX adjoint.

**G vs pressure balance.** Ideal MHD only requires *total pressure* to be
continuous across the boundary. The tangential field may change direction
there, which is a surface current sheet. So:

* pointwise pressure balance checks |B| point by point, but on its own it
  doesn't fix the net current the coils link;
* G checks one exact integral of the tangential field (its toroidal
  circulation), but says nothing about the local |B| pattern;
* neither replaces B·n.

VMEX's fixed-boundary solution has no sheet current, so a truly consistent coil
set satisfies all three at once. In the driver, G holds the scale exactly and
the end-of-run check (section 5) reports how well the local |B| pattern
followed.

## 5. End-of-run check (every run)

The constraints and penalties act on the optimization grid only. So every run
also reports:

* **During the run.** Each outer iteration logs `pressure jump rms=…` and
  writes `pressure_jump: {mean, rms, max}` to `history.yaml` (`PlasmaCoilInterface.pressure_balance_stats()`).
* **At the end.** `interface_check()` recomputes pointwise B·n and the pressure
  jump with virtual casing on an **independent, finer grid** (`final_check:`,
  default 32×32 per field period, digits 4), as VMEX's example does, so an
  optimum aliased to the optimization grid shows up. Results go to the log and
  to `<outdir>/final_check.yaml`. The check always uses virtual casing, so a
  `plasma_field: vacuum` run is tested against the plasma field it ignored.

Normalization: `e = (|B_out|² − |B_in|² − 2μ0 p_edge) / |B_in|²`, area-weighted
mean / rms / max |e|. Since `e ≈ 2 δ|B|/|B|`, an rms of 1e-3 is a 0.05 %
field-strength mismatch.

Validation (job 1267, Landreman–Paul QA β 2.5 %): on the optimization grid,
`interface_check` reproduces the interface's mean, rms and B·n to ~1e-6. With
seed coils 1.71× too strong (G residual +0.71), the measured mean jump is 2.08,
close to 1.71² − 1 = 1.92. The 32×32 check takes 13 s.

Virtual casing returned nondeterministic garbage on many cores (jobs
2323–2326), and pinning to one core deadlocks VMEX (jobs 2328/2406). 2 or 4
cores are deterministic (job 2418), so run with `--cpus-per-task=4` and
`PIN_CORES` unset (`sbatch/run_stage.sbatch`). The final check runs in the same
process.

## 6. Which to use

| situation | `plasma_field` | scale mechanism |
|---|---|---|
| Star_Lite, β ≈ 3e-4 | `vacuum` | `field_strength: toroidal_flux` (or `net_poloidal_current`, equivalent here) |
| finite β (driver, production) | `virtual_casing` | `field_strength: net_poloidal_current` |
| VMEX's pointwise formulation, for comparison | `virtual_casing` | `examples/combined_stage_simple.py` only (`PressureBalancePenalty`, `field_strength=None`) |

In every case, read `final_check.yaml` before trusting a finite-β result.

## 7. Examples

* `examples/combined_stage_simple.py`: **pointwise pressure balance**, VMEX's
  formulation. It uses the Landreman–Paul QA at β = 2.5 % (VMEX's finite-β
  seed), `field_strength=None` with `PressureBalancePenalty` weight 1e2, B·n
  harmonics as AL equalities, and `interface_check` at 32×32 at the end.
* `examples/combined_stage_designA_beta1.py`: **Boozer G**. designA at β = 1 %,
  `field_strength="net_poloidal_current"` with `extra_outputs={"rbtor": …}`,
  and the pointwise pressure balance reported and checked at the end.
* Production configs with G: `configs/hsx_dn_32coils_vc_eqG32.yaml` (run 2319)
  and `configs/hsx_dn_32coils_vc_eqG32_boot.yaml` (+ bootstrap, run 2327).
