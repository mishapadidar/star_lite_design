import matplotlib.pyplot as plt
import numpy as np
from simsopt._core import load
from simsopt.geo import (SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual,
                         Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas, CurveCurveDistance)
from simsopt.field.selffield import regularization_circ
from simsopt.field.force import coil_force, LpCurveForce
from simsopt.field import BiotSavart
import os

plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 11})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]


# load original configurations
[boozer_surfaces_orig, iota_Gs_orig, axis_curves_orig, x_point_curves_orig] = load("../designs/designB_after_scaled.json")
# boozer_surfaces_orig = data[0] # BoozerSurfaces
# iota_Gs_orig = data[1] # (iota, G) pairs
# axis_curves = data[2] # magnetic axis CurveRZFouriers
# # x_point_curves = data[3] # X-point CurveRZFouriers

# load the NEW optimized design
force_weight=1e-11
[boozer_surfaces_new, iota_Gs_new, axis_curves_new, xpoint_curves_new] = load(f"./output/designB/force_weight_{force_weight}/designB_after_forces_opt.json")

coil_minor_radius = 0.054 # 54mm

# design B base curves
base_curve_idx = [0, 2]
n_curves = len(base_curve_idx)

n_surfs = len(boozer_surfaces_new)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
width = 0.1  # the width of the bars
gap = width/2 # gap between groups of bars
x = np.array([0, width])  # left edge of bars

xticks = []
for ii in range(n_surfs):
    print("")
    print(f"Group {ii}")

    # check forces on new coils
    coils = boozer_surfaces_new[ii].biotsavart.coils
    forces_new = [np.linalg.norm(coil_force(coils[idx], coils, regularization_circ(coil_minor_radius)), axis=1) for idx in base_curve_idx]
    mean_forces_new = np.array([np.mean(np.abs(f)) for f in forces_new])
    max_forces_new = np.array([np.max(np.abs(f)) for f in forces_new])

    # check forces on original coils
    coils = boozer_surfaces_orig[ii].biotsavart.coils
    forces_orig = [np.linalg.norm(coil_force(coils[idx], coils, regularization_circ(coil_minor_radius)), axis=1) for idx in base_curve_idx]
    mean_forces_orig = np.array([np.mean(np.abs(f)) for f in forces_orig])
    max_forces_orig = np.array([np.max(np.abs(f)) for f in forces_orig])

    # take ratios
    y_mean = np.abs((mean_forces_new - mean_forces_orig) / mean_forces_orig)
    y_max = np.abs((max_forces_new - max_forces_orig) / max_forces_orig)
    y = np.vstack((y_mean, y_max)) # (2, n_curves)
    y = 100 * y # convert to percentage

    if ii == 0:
        label = ['coil 0', 'coil 2']
    else:
        label = None

    # location of left edge
    x_left = x + ii*(n_curves*width + gap)

    # mean forces
    rects = ax1.bar(x_left, y[0], width=width,  align='edge', color=[colors[0], colors[1]], alpha=0.6, label=label)

    # max forces
    ax2.bar(x_left, y[1], width=width, align='edge', color=[colors[0], colors[1]], alpha=0.6, label=label)

    xticks.append(x_left[1])

# darken the border
ax1.patch.set_edgecolor('black')
ax1.patch.set_linewidth(1)
ax2.patch.set_edgecolor('black')
ax2.patch.set_linewidth(1)

# line
# ax.axhline(y=100.0, color='k', linestyle='--', lw=2)

# labels
ax1.set_ylabel('Reduction in Mean Coil Forces [%]')
ax2.set_ylabel('Reduction in Max Coil Forces [%]')

ax1.set_xticks(xticks, labels=[f'Group {ii}' for ii in range(n_surfs)])
ax2.set_xticks(xticks, labels=[f'Group {ii}' for ii in range(n_surfs)])

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')

plt.tight_layout()

# save the figure as a PDF
outdir='./viz/'
os.makedirs(outdir, exist_ok=True)
fig.savefig(outdir+"force_optimization_plot.pdf", format="pdf", bbox_inches="tight")
plt.show()
