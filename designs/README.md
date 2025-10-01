

`designA_after_scaled.json` and `designB_after_scaled.json` were the designs that Andrew created by doing the optimization with multiple iotas.  

We found that the coil forces
in `designB_after_scaled.json` could be reduced, so Misha's first attempt to reduce the coil forces is in `designB_after_forces_opt.json`. It succeeds `designB_after_scaled.json`.

`sheetmetal_chamber.obj` is the vacuum vessel as of July 8, 2025.

Andrew's first attempt at coil force minimization is in `designB_after_forces_9.json`, and the final attempt after scanning a bit more is in `designB_after_forces_19.json`.
`designB_after_forces_19_23072025.json` is the coil design sent to Georg, contains only the coils. `designB_after_forces_opt_19.json` is the same, but contains the boozer surface information.

`designB_after_currents_opt_9.json` was just the current optimization to 58kA before the force optimization was done.
