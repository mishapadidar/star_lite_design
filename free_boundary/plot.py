import numpy as np
import matplotlib.pyplot as plt
from simsopt._core import load
from simsopt.mhd import Vmec, vmec_compute_geometry, Quasisymmetry, QuasisymmetryRatioResidual,Boozer
import glob
import pickle
plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 12})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
# colors = ['lightcoral', 'goldenrod', 'mediumseagreen','orange', "lightskyblue", "plum"]
colors = ['goldenrod', 'mediumseagreen',"lightskyblue", "plum", 'orange', 'lightcoral', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.']

"""
Plot the result of the free boundary computation e.g.
    python plot.py

"""


filelist = glob.glob("./plot_data/plot_data*.pkl")


fig, (ax1,ax2) = plt.subplots(figsize=(12,4), ncols=2)


# (beta, Imin)
keep_list = [(0.0,0.0), (0.01, 0.0), (0.01, -1.0), (0.1, 0.0), (0.1, -10.0)]

for ii, ff in enumerate(filelist):
    data = pickle.load(open(ff, "rb"))
    iota_free = data['iota_free']
    s_iota_free = data['s_iota_free']
    qs_err_free = data['qs_err_free']
    s_qs_err_free = data['s_qs_err_free']
    design = data['design']
    beta = data['beta']
    Imin = data['Imin']

    if (beta, Imin) not in keep_list:
        continue

    label = rf'beta={beta :.3f} % , I={Imin :.2f} A'
    ax1.plot(s_qs_err_free, 100 * qs_err_free, lw=2, ls =linestyles[ii], color=colors[ii], label=label)
    
    ax2.plot(s_iota_free, iota_free, lw=2, ls =linestyles[ii], color=colors[ii])


# ax1.set_yscale('log')
ax1.set_xscale('log')
ax1.grid(color='lightgrey', linestyle='--', linewidth=0.5)
ax1.set_xlabel(r'$s$')
ax1.set_ylabel('QS-error [%]')
# ax1.legend(loc='upper left')

fig.legend(loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.1), fontsize=10)
ax2.set_xlabel(r'$s$')
ax2.set_ylabel(r'$\iota$')
ax2.grid(color='lightgrey', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig("./plot_data/qs_iota_plot.pdf", bbox_inches='tight', format='pdf')
# plt.show()
