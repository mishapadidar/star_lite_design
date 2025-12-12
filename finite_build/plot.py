from star_lite_design.finite_build.hex_finite_build_coil import _hex_grid_of_circles, create_hexagonal_filament_grid
from simsopt.geo import CurveXYZFourier, curves_to_vtk
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
Plot finite build coils

Run with 
    python plot.py design

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

outdir = "./plot_data/"
if not os.path.exists(outdir):
    os.makedirs(outdir)

n_layers = 2
winding_pack_radius = 0.054 # m
cable_radius = winding_pack_radius / (2*n_layers + 1)

# write paraview files of the finite build coils
coils = biotsavart.coils
center_curves = [c.curve for c in coils]
curves_fb = []
for c in center_curves:
    curves = create_hexagonal_filament_grid(c, n_layers=n_layers, radius=cable_radius)
    curves_fb.extend(curves)
curves_to_vtk(curves_fb, outdir + "finite_build", close=True)
curves_to_vtk(center_curves, outdir + "finite_build_center", close=True)


# make a plot of the hexagonal grid
shifts_normal, shifts_binormal = _hex_grid_of_circles(n_layers, cable_radius)
plt.figure(figsize=(6,6))
plt.scatter(shifts_normal, shifts_binormal, color='k', s=5)
for dn, db in zip(shifts_normal, shifts_binormal):
    if np.sqrt(dn**2 + db**2) < 2.5*cable_radius:
        color=colors[0]
    else:
        color=colors[1]
    circle = plt.Circle((dn, db), cable_radius, fill=True, facecolor=color, edgecolor=color, linewidth=1, alpha=0.5)
    plt.gca().add_patch(circle)
# center circle
circle = plt.Circle((0, 0), cable_radius, fill=False, edgecolor='k', linewidth=1, label='spine')
plt.gca().add_patch(circle)
# outer circle
circle = plt.Circle((0, 0), winding_pack_radius, fill=False, edgecolor='k', linewidth=1, label='spine')
plt.gca().add_patch(circle)
# line showing radius
plt.plot((0,winding_pack_radius), (0,0), color='k', lw=2, ls='--')
plt.text(winding_pack_radius*0.6, 0.002, f'$r=${winding_pack_radius:.3f}m', ha='center', fontsize=12)


# plt.title(f'Hexagonal filament grid with {n_layers} layers')
plt.xlabel('Normal Coordinate [m]')
plt.ylabel('Binormal Coordinate [m]')
plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.xlim(-6*cable_radius, 6*cable_radius)
plt.ylim(-6*cable_radius, 6*cable_radius)
plt.gca().set_aspect('equal', adjustable='box')
plt.xticks([-0.04,0.0,0.04])
plt.yticks([-0.04,0.0,0.04])
plt.tight_layout()
plt.savefig(outdir + "coil_cross_section.pdf", format='pdf')
plt.show()