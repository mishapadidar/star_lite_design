from star_lite_design.continuous_winding_model.winding import continuous_winding
from simsopt._core import load
from multiprocessing import Pool
import numpy as np
import os

"""
Wind all 6 coils of design A with the continuous winding model, in parallel, and save
the paths. Each coil is wound separately with the same parameters, in its own
centroid frame (see README.md: on the stellarator-symmetry copies the frame is
mirrored). The leads are 1 m long; the circuit closes between their far ends.

Writes output/designA/coil{i}.npz: the path and the info dict of continuous_winding.

Run with
    module load gcc openmpi
    source ../../env_star_lite_design/bin/activate
    export PYTHONPATH="${PYTHONPATH}:/mnt/home/mpadidar/code/star_lite"
    python wind_coils.py
"""

infile = "../designs/designA/designA_after_scaled.json"
outdir = "./output/designA/"
params = dict(n_winds=18, radius=0.0108, lead_length=1.0)


def wind(i):
    coil = load(infile)[0][0].biotsavart.coils[i]
    path, info = continuous_winding(coil.curve, **params)
    np.savez(outdir + f"coil{i}.npz", path=path, current=coil.current.get_value(),
             **{k: v for k, v in info.items() if k != "turns"}, turns=np.array(info["turns"]))
    return i, info["sigma_lead_out"]


if __name__ == "__main__":
    os.makedirs(outdir, exist_ok=True)
    with Pool(6) as pool:
        for i, sigma_out in pool.imap_unordered(wind, range(6)):
            print(f"coil {i} wound, finishing lead at sigma = {100*sigma_out:.1f} cm", flush=True)
