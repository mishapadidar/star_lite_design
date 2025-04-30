import numpy as np
from simsopt._core import load
from simsopt.field.selffield import regularization_circ
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.geo import curves_to_vtk
import os


"""
Compute the coil forces following the example,
https://github.com/hiddenSymmetries/simsopt/blob/master/examples/3_Advanced/coil_forces.py
"""

design = "A"


# load the boozer surfaces (1 per Current configuration, so 3 total.)
data = load(f"../designs/design{design}_after_scaled.json")
bsurfs = data[0] # BoozerSurfaces

print("Design", design)
# 3 different iota optimizations
for iota_group_idx in range(3):
    print("")
    print("iota group:", iota_group_idx)
    biotsavart = bsurfs[iota_group_idx].biotsavart

    coils = biotsavart.coils
    curves = [coil.curve for coil in coils]
    for ii, c in enumerate(coils):
        force = np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)
        print(f"max force on coil {ii}: %.2f"%(np.max(np.abs(force))))

    def pointData_forces(coils):
        forces = []
        for c in coils:
            force = np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)
            force = np.append(force, force[0])
            forces = np.concatenate([forces, force])
        point_data = {"F": forces}
        return point_data

    outdir = "./output"
    os.makedirs(outdir, exist_ok=True)
    curves_to_vtk(curves, outdir + f"/design_{design}_group_{iota_group_idx}_curves_with_forces", close=True, extra_data=pointData_forces(coils))

