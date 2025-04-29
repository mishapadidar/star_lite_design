import numpy as np
from simsopt._core import load
from simsopt.field.selffield import regularization_circ
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.geo import curves_to_vtk


"""
Compute the coil forces following the example,
https://github.com/hiddenSymmetries/simsopt/blob/master/examples/3_Advanced/coil_forces.py
"""

design = "B"
current_group_idx = 0 # 3 current groups


# load the boozer surfaces (1 per Current configuration, so 3 total.)
data = load(f"../designs/design{design}_after_scaled.json")
bsurfs = data[0] # BoozerSurfaces
biotsavart = bsurfs[current_group_idx].biotsavart

coils = biotsavart.coils
curves = [coil.curve for coil in coils]
for ii, c in enumerate(coils):
    force = np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)
    print(f"max force on coil {ii}", np.max(np.abs(force)))

def pointData_forces(coils):
    forces = []
    for c in coils:
        force = np.linalg.norm(coil_force(c, coils, regularization_circ(0.05)), axis=1)
        force = np.append(force, force[0])
        forces = np.concatenate([forces, force])
    point_data = {"F": forces}
    return point_data

curves_to_vtk(curves, "./curves_with_forces", close=True, extra_data=pointData_forces(coils))

