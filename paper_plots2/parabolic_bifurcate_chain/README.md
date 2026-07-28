# Continue an X-point across the tr = 2 parabolic bifurcation into an island chain

Starting from a fixed design, drive one hyperbolic X-point's monodromy trace downward across the
parabolic point (tr = 2) so it turns into an elliptic O-point, while PINNING the trace of every other
fixed point. Then trace and plot the separatrix manifolds before and after.

Trace `tr` classifies each fixed point (residue `R = (2 - tr) / 4`): `|tr| < 2` elliptic (O-point),
`|tr| > 2` hyperbolic (X-point), `tr = 2` the parabolic bifurcation.

## Input / output

- Input (fixed): `hyperbolic_SN_x2o_coupled_trX1.500_fixedpoints.json` in this directory, the grouped
  fixed-point file `[BiotSavart, {type: [curve, ...]}]` written by `mk_manifolds.py`.
- All generated files go into `output/`.

## Pipeline

1. `xpoint_to_opoint_coupled.py` — coupled continuation. Loads the fixed input json, reuses its circular
   PF (aux) coils (plus any extra ones) as the shared free currents, and drives the X-point CLOSEST to
   the magnetic axis across tr = 2 while pinning all other traces. Saves one design json per grid target
   (`output/*_x2o_coupled2_trX<|trace|>.json`), plus VTK steps and a `*_cond.txt` conditioning log.
   Usage: `./xpoint_to_opoint_coupled.py [num_extra_PF=0] [target_|trace|=1.0] [STEP0=0.1]`.
2. `mk_manifolds.py` — for a design json, trace the separatrix manifolds of each X-point and the nested
   surfaces of each O-point using the field exactly as loaded. Writes `<stem>_fixedpoints.json` (curves
   grouped by type) and `<stem>_allmanifolds.txt` (phi = PHI Poincare hits: `seed_id, R, Z`, with header
   lines for the fixed points and per-leg kinds). Usage: `./mk_manifolds.py <design_json> [search_R search_Z]`.
3. `plot_manifolds_row.py` — plot two `_allmanifolds.txt` datasets side by side (panels A and B) into
   `<left_stem>_row.png`. A per-panel radius density filter (`DENS_A`, `DENS_B` near the top of the file)
   drops the isolated scattered dots left by escaped orbits, keeping only the dense separatrix legs and
   O-surfaces. Usage: `./plot_manifolds_row.py <left.txt> <right.txt>`.

## Other files

- `plot_manifolds.py` — single-panel version of the plotter (one `_allmanifolds.txt` or design json);
  also importable as `plot_manifolds(path)`. `plot_manifolds_row.py` reuses its formatting.
- `mk_chain_fig.sh` — end-to-end driver: runs the continuation, then `mk_manifolds.py` on two of the
  resulting design jsons, then `plot_manifolds_row.py` to build the final figure.

## Quick start

```sh
./mk_chain_fig.sh
```
