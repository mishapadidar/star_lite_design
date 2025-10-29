# Free Boundary Analysis
Analyze starlite configurations with non-zero pressure/current.

The analysis is only done for `design A`.
- First run `vmec_free_boundary.py` with various pressure and current profiles. Do this on the cluster using `submit_vmec_free_boundary.sub` submits free boundary runs to SLURM.
- Move the `wout` files to the `output` directory.
- Now run `generate_plot_data.py` to generate data for plotting.
- Finally run `plot.py`.
