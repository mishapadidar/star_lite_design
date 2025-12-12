from star_lite_design.finite_build.hex_finite_build_coil import _hex_grid_of_circles, create_hexagonal_filament_grid
from star_lite_design.utils.find_x_point import find_x_point
from simsopt.geo import CurveXYZFourier, curves_to_vtk
from simsopt.field import Coil, Current, BiotSavart
import numpy as np
import os, sys
from simsopt._core import load
import matplotlib.pyplot as plt


plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 16})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
# colors = ['lightcoral', 'goldenrod', 'mediumseagreen','orange', "lightskyblue", "plum"]
colors = ['goldenrod', 'mediumseagreen',"lightskyblue", "plum", 'orange', 'lightcoral', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.']
markers= ['o', 's', 'o', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h']

"""
Analyze the effect of finite build coils compared to filamentary coils

Run with 
    python analyze_finite_build.py design

Arguments:
design : str
    Design to plot, either "A" or "B"
"""


design = str(sys.argv[1])

print("")
print("Running with parameters", flush=True)
print("design", design, flush=True)

if design == "A":
    infile = "../designs/designA_after_scaled.json"
else:
    infile = "../designs/designB_after_forces_opt_19.json"

# load the boozer surfaces (1 per Current configuration, so 3 total.)
data = load(f"../designs/design{design}_after_scaled.json")
bsurf = data[0][0] # BoozerSurfaces
surf = bsurf.surface
biotsavart = bsurf.biotsavart
ma = data[2][0] # magnetic axis CurveRZFourier
xpoint = data[3][0] # X-point CurveRZFouriers

n_layers = 2
winding_pack_radius = 0.054 # m
cable_radius = winding_pack_radius / (2*n_layers + 1)
n_winds_per_coil = 18

# build the finite build coils
coils = biotsavart.coils
coils_fb = []
for c in coils:
    center_curve = c.curve

    # get each filament curve for this coil
    curve_turns = create_hexagonal_filament_grid(center_curve, n_layers=n_layers, radius=cable_radius)

    # divide the current evenly among the filaments
    current_turns = [Current((c.current.get_value() / n_winds_per_coil)) for _ in range(n_winds_per_coil)]
    
    # build the coil
    for ii, ct in enumerate(curve_turns):
        coils_fb.append(Coil(curve_turns[ii], current_turns[ii]))

biotsavart_fb = BiotSavart(coils_fb)

# compare the fields on axis
xyz = ma.gamma()
biotsavart.set_points(xyz)
modB = biotsavart.AbsB()
print('original: mean |B| on axis:', np.mean(modB))
biotsavart_fb.set_points(xyz)
modB = biotsavart_fb.AbsB()
print('finite build: mean |B| on axis:', np.mean(modB))

# compare the x-point positions
xyz_xpoint = xpoint.gamma()
r0 = np.sqrt(xyz_xpoint[:, 0]**2 + xyz_xpoint[:, 1]**2)
z0 = xyz_xpoint[:, 2]
nfp = xpoint.nfp
order = xpoint.order
xpoint_fb, _, _ = find_x_point(biotsavart_fb, r0, z0, nfp, order)

xyz_xpoint_fb = xpoint_fb.gamma()
diff = xyz_xpoint_fb - xyz_xpoint
print('Max difference in X-point:', np.max(np.linalg.norm(diff, axis=1)))



