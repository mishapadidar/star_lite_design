import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 20})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
# colors = ['lightcoral', 'goldenrod', 'mediumseagreen','orange', "lightskyblue", "plum"]
# colors = ['goldenrod', 'mediumseagreen',"lightskyblue", "plum", 'orange', 'lightcoral', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']
colors = ['#1F77B4', '#D62728']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.']
markers= ['o', 's', 'o', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h']

outdir = "./plot_data/"
data = pickle.load(open(outdir + "plot_data.pkl", "rb"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

idx_distinct = [0, -1]  # indices of two distinct coils
labels = ['L-coil', 'T-coil']

# plot forces
arc_lengths = data['arc_lengths']
forces = data['forces']
flange_points = data['flange_arclength']
ax1.plot(arc_lengths[idx_distinct[0]], forces[idx_distinct[0]], label=labels[0], color=colors[0], lw=3, alpha=0.8)
ax1.plot(arc_lengths[idx_distinct[1]],forces[idx_distinct[1]], label=labels[1], color=colors[1], lw=3, alpha=0.8)
# # plot flange points
# idx_flange = flange_points[idx_distinct[0]]
# ax1.scatter(arc_lengths[idx_distinct[0]][idx_flange], forces[idx_distinct[0]][idx_flange], color='k', ls='--', lw=2)
# idx_flange = flange_points[idx_distinct[1]]
# ax1.scatter(arc_lengths[idx_distinct[1]][idx_flange], forces[idx_distinct[1]][idx_flange], color='k', ls='--', lw=2)

ax1.set_xlabel('Arc Length [m]')
ax1.set_ylabel('Force Magnitude [N/m]')
ax1.set_title('Force Distribution Along Coils')
ax1.legend(loc='upper right')
ax1.grid(color='lightgray', linestyle='-', linewidth=0.5)


# plot current I(t)
L = np.diag(data['L'])
R = data['resistances']
currents = np.abs(data['currents'])
tmax = 6*np.max(L/R)
print(L)
print(R)
for ii, idx in enumerate(idx_distinct):
    L_i = L[idx]
    R_i = R[idx]
    I_i = currents[idx]
    def I_t(t):
        return I_i * (1 - np.exp(-R_i/L_i * t))
    ts = np.linspace(0, tmax, 100)
    Is = I_t(ts)
    ax2.plot(ts*1000, Is/1000, label=labels[ii], color=colors[ii], lw=3, alpha=0.8)  # in kA and ms
    ax2.axhline(I_i/1000, color=colors[ii], lw=2, ls='--', alpha=0.8)
ax2.set_xlabel('Time [ms]')
ax2.set_ylabel('Current [kA]')
ax2.set_title('Current Ramp-up in Coils')
ax2.grid(color='lightgray', linestyle='-', linewidth=0.5)


plt.tight_layout()
plt.savefig(outdir + "force_inductance_analysis.pdf", format='pdf')
plt.show()