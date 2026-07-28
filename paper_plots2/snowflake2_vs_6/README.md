# Snowflake-2 vs snowflake-6: side-by-side Poincare / manifold sections

A paper figure comparing the separatrix-manifold structure of two designs in one 1x2 row:

- `design_polished_final_59354326`   -- the snowflake-2 design.
- `design_unpolished_final_1625253525` -- the snowflake-6 design.

Each design json is the rich layout `[boozer_surfaces, iota_Gs, axes, xpoints, sdf]`; `dat[2]` is the
magnetic axis (an elliptic O-point) and `dat[3]` is the tracked snowflake fixed point.

## Pipeline (`mk_plot.sh`)

1. `mk_manifolds.py <design_json>` -- trace, using the field exactly as loaded, the separatrix manifolds
   of every X-point / snowflake and the nested surfaces of every elliptic O-point (the magnetic axis
   `dat[2]` is included, plus any extra fixed points found by a multistart-Newton search). Writes
   `<stem>_fixedpoints.json` (curves grouped by type) and `<stem>_allmanifolds.txt` (phi = PHI Poincare
   hits: `seed_id, R, Z`, with header lines for the fixed points and per-leg kinds), and auto-plots a
   single-panel `<stem>_allmanifolds.png`.
2. `plot_manifolds_row.py <left.txt> <right.txt>` -- draw the two `_allmanifolds.txt` datasets side by
   side (panels A and B, shared axes, no gap) into `<left_stem>_row.png`. A per-panel radius density
   filter (`DENS_A`, `DENS_B`) drops the isolated scattered dots left by escaped orbits; the shared R/Z
   window is `WIN_XLIM` / `WIN_YLIM`. Both near the top of the file.

Run the whole thing with:

```sh
./mk_plot.sh
```

## Files

- `mk_manifolds.py` -- manifold / nested-surface tracer (per-class seed / transit / time-cap tunables at
  the top).
- `plot_manifolds.py` -- single-panel plotter; also importable as `plot_manifolds(path)`.
  `plot_manifolds_row.py` reuses its formatting.
- `plot_manifolds_row.py` -- the 1x2 comparison plotter (this figure). No zoom inset.
- `in_jsons/` -- pristine input designs; `jsons/` holds the working copies and all generated output.
