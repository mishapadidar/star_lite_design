import numpy as np
import matplotlib.pyplot as plt
from simsopt._core import load
from simsopt.mhd import Vmec, vmec_compute_geometry, Quasisymmetry, QuasisymmetryRatioResidual,Boozer
import glob
import pickle
import pandas as pd
plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 16})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
# colors = ['lightcoral', 'goldenrod', 'mediumseagreen','orange', "lightskyblue", "plum"]
# colors = ['goldenrod', 'mediumseagreen',"lightskyblue", "plum", 'orange', 'lightcoral', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']
colors = ['#1F77B4', '#D62728']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.']
markers= ['o', 's', 'o', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h']

"""
Plot the result of the free boundary computation e.g.
    python plot.py
First you need to run vmec_free_boundary.py and save the wout files in output/.
Then run generate_plot_data.py to generate the data files in ./plot_data/.
"""

design = 'A'
filelist = glob.glob(f"./plot_data/plot_data_design_{design}*.pkl")


data = pickle.load(open(f'./plot_data/plot_data_design_{design}_beta_0.0_Imin_0.0.pkl', "rb"))
iota_vac = data['iota_free'][-1] # boundary iota
qs_err_vac = 100 * np.mean(data['qs_err_free'])
x_point_vac = data['xp_fp']

df_data = {'qs_err':[], 'iota':[], 'xp_shift':[], 'beta':[], 'Imin':[], 'I0':[]}

for ii, ff in enumerate(filelist):
    data = pickle.load(open(ff, "rb"))
    # compute metrics
    iota = data['iota_free'][-1]
    qs_err = 100* np.mean(data['qs_err_free'])
    xp = data['xp_fp']
    xp_shift = 1000 * np.max(np.linalg.norm(xp - x_point_vac, axis=-1)) # in mm
    I0 = 4*data['Imin']

    # save it
    df_data['iota'].append(iota)
    df_data['qs_err'].append(qs_err)
    df_data['xp_shift'].append(xp_shift)
    df_data['beta'].append(data['beta'])
    df_data['Imin'].append(data['Imin'])
    df_data['I0'].append(I0)


df = pd.DataFrame(df_data)
df.sort_values(by=['beta', 'Imin'], inplace=True)
print(df)

fig, (ax1,ax2,ax3) = plt.subplots(figsize=(13,4), ncols=3)

for ii, beta in enumerate(df['beta'].unique()):
    if beta == 0.0:
        continue
    df_sub = df[df['beta']==beta]
    ax1.plot(df_sub['I0'], df_sub['qs_err'],   lw=3, marker=markers[ii], color=colors[ii-1], alpha=0.8, markersize=8, label=r'$\beta$' + f'={beta}%')
    ax2.plot(df_sub['I0'], df_sub['iota'], lw=3, marker=markers[ii], color=colors[ii-1], alpha=0.8, markersize=8, label=r'$\beta$' + f'={beta}%')
    ax3.plot(df_sub['I0'], df_sub['xp_shift'],  lw=3, marker=markers[ii], color=colors[ii-1], alpha=0.8, markersize=8, label=r'$\beta$' + f'={beta}%')

# # plot the expected configuration as a star
# df_sub = df[(df['beta']==0.01) & (df['Imin']==0.01)]
# ax1.scatter(df_sub['I0'], df_sub['qs_err'],   marker='*', s=100, color='red', label=f'TJ-K scale',zorder=100)
# ax2.scatter(df_sub['I0'], df_sub['iota'], marker='*', s=100, color='red', label=f'TJ-K scale',zorder=100)
# ax3.scatter(df_sub['I0'], df_sub['xp_shift'],   marker='*', s=100, color='red', label=f'TJ-K scale',zorder=100)


axis_fontsize = 18
ax1.set_xlabel(r'$I_0$ [A]', fontsize=axis_fontsize)
ax1.set_ylabel('$J_{QS}$ [%]', fontsize=axis_fontsize)
ax1.grid(color='lightgrey', linestyle='--', linewidth=0.5)
ax1.axhline(qs_err_vac, color='black', linestyle=':', lw=2, label='vacuum')

gap_scale = 3
qs_gap = np.abs(df[(df['beta']==0.1) & (df['Imin']==0.0)].qs_err.item() - qs_err_vac)
ax1.set_ylim(qs_err_vac-gap_scale*qs_gap, qs_err_vac+gap_scale*qs_gap)


ax2.set_xlabel(r'$I_0$ [A]', fontsize=axis_fontsize)
ax2.set_ylabel(r'$\iota_{edge}$', fontsize=axis_fontsize)
ax2.axhline(iota_vac, color='black', linestyle=':', lw=2,label='vacuum')
ax2.grid(color='lightgrey', linestyle='--', linewidth=0.5)
# ax2.set_ylim(0.18, 0.20)
iota_gap = np.abs(df[(df['beta']==0.1) & (df['Imin']==0.0)].iota.item() - iota_vac)
ax2.set_ylim(iota_vac-gap_scale*iota_gap, iota_vac+gap_scale*iota_gap)
ax2.legend(loc='upper left', fontsize=14)

ax3.set_xlabel(r'$I_0$ [A]', fontsize=axis_fontsize)
ax3.set_ylabel('X-point displacement [mm]', fontsize=16)
ax3.axhline(0.0, color='black', linestyle=':', lw=2,label='vacuum')
ax3.grid(color='lightgrey', linestyle='--', linewidth=0.5)
# ax3.set_ylim(-0.3, 5.0)
xp_shift_gap = np.abs(df[(df['beta']==0.1) & (df['Imin']==0.0)].xp_shift.item())
ax3.set_ylim(-0.3, 2*gap_scale*xp_shift_gap)

ax1.set_xlim(np.min(df['I0'])-2e-3, np.max(df['I0'])+20)
ax2.set_xlim(np.min(df['I0'])-2e-3, np.max(df['I0'])+20)
ax3.set_xlim(np.min(df['I0'])-2e-3, np.max(df['I0'])+20)

ax1.set_xscale('symlog',linthresh=1e-2)
ax2.set_xscale('symlog',linthresh=1e-2)
ax3.set_xscale('symlog',linthresh=1e-2)
plt.tight_layout()
plt.savefig(f"./plot_data/design_{design}_free_boundary_plot.pdf", bbox_inches='tight', format='pdf')
plt.show()
