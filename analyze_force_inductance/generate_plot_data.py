import numpy as np
import pickle
import os
from simsopt._core import load
from simsopt.field.force import coil_force
from simsopt.field.selffield import regularization_circ
from simsopt.geo import curves_to_vtk
from star_lite_design.analyze_force_inductance.inductance import Inductance


design_file = "../designs/designA_after_scaled.json"

# load the biotsavart (1 per Current configuration, so 3 total.)
data = load(design_file)
current_group = 0
bsurf = data[0][current_group] # BoozerSurfaces
axis_curve = data[2][current_group] # magnetic axis CurveRZFouriers

biotsavart = bsurf.biotsavart
coils = biotsavart.coils

# minor radius
minor_radius = 0.054 # 54mm

# coil resistance of 4/0 AWG copper wire at 20C
resistance_4_0_awg = 0.04901/1000 # Ohms / ft
meter_per_ft = 0.3048 
resistance_per_meter = resistance_4_0_awg / meter_per_ft  # Ohms / meter
print('Resistance per meter (Ohms/m):', resistance_per_meter)

# compute the inductance matrix
ind = Inductance(coils, minor_radius)
L = ind.calculate()

print("self inductances", np.diag(L))

# check mean |B| on axis
xyz = axis_curve.gamma()
biotsavart.set_points(xyz)
modB = biotsavart.AbsB()
print("mean |B| on axis: %.4f"%(np.mean(modB)))

coils = biotsavart.coils
curves = [coil.curve for coil in coils]
forces = []
for ii, c in enumerate(coils):
    f = np.linalg.norm(coil_force(c, coils, regularization_circ(minor_radius)), axis=1) # (n,)
    forces.append(f)
    print(f"max force on coil {ii}: %.2f"%(np.max(np.abs(f))))

forces = np.array(forces)

# generate vtk data
f_plot = []
for f in forces:
    f = np.append(f, f[0])
    f_plot = np.concatenate([f_plot, f])
point_data = {"F": f_plot}


# curve lengths (for resistance calculation)
N = len(curves[0].quadpoints)
arc_lengths = np.array([np.cumsum(curve.incremental_arclength())/N for curve in curves])  # (n_coils, N)
length_per_turn = np.array([np.mean(curve.incremental_arclength()) for curve in curves])
n_turns = 18
lengths = n_turns * length_per_turn
print("Coil lengths (m):", lengths)
resistances = np.array([resistance_per_meter * L for L in lengths])
print("Coil resistances (Ohms):", resistances)
currents = np.array([coil.current.get_value() for coil in coils])

outdir = "./plot_data/"
os.makedirs(outdir, exist_ok=True)
curves_to_vtk(curves, outdir + f"/curves_with_forces", close=True, extra_data=point_data)

print("")
# arclength of flange points
flange_arclength = []
for curve in curves:
    xyz = curve.gamma()
    # flange is at max z and min z
    idx1 = np.argmax(xyz[:,2])  # max z
    idx2 = np.argmin(xyz[:,2])  # min z
    # gp = curve.gammadash()
    # quadpoints = curve.quadpoints
    # dphi = np.diff(quadpoints)[0]
    # arclengths = np.cumsum(np.linalg.norm(gp, axis=-1) * dphi)
    # arcl1 = arclengths[idx1]
    # arcl2 = arclengths[idx2]
    flange_arclength.append([idx1, idx2])


# save the data
outfile = outdir + "plot_data.pkl"
outdata = {}
outdata['design_file'] = design_file
outdata['current_group'] = current_group
outdata['minor_radius'] = minor_radius
outdata['resistance_per_meter'] = resistance_per_meter
outdata['L'] = L
outdata['forces'] = forces
outdata['resistances'] = resistances
outdata['arc_lengths'] = arc_lengths
outdata['currents'] = currents
outdata['flange_arclength'] = flange_arclength
pickle.dump(outdata, open(outfile, "wb"))


# analytic formula for self-inductance of circular loop
R = length_per_turn[0] / (2 * np.pi)
L_circular = 4e-7 * np.pi * R * (np.log(8 * R / minor_radius) - 7/4)
print(f"Analytic self-inductance of circular loop: {L_circular}")

