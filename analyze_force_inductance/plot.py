import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif')
# plt.rc('text.latex', preamble=r'\\usepackage{amsmath,bm}')
plt.rcParams.update({'font.size': 12})
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", 
          "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
# colors = ['lightcoral', 'goldenrod', 'mediumseagreen','orange', "lightskyblue", "plum"]
colors = ['goldenrod', 'mediumseagreen',"lightskyblue", "plum", 'orange', 'lightcoral', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:purple']
linestyles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.']
markers= ['o', 's', 'o', '^', 'D', 'v', 'P', '*', 'X', '<', '>', 'h']

outdir = "./plot_data/"
data = pickle.load(open(outdir + "plot_data.pkl", "rb"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

idx_distinct = [0, -1]  # indices of two distinct coils
labels = ['L-coil', 'T-coil']

# plot forces
arc_lengths = data['arc_lengths']
forces = data['forces']
ax1.plot(arc_lengths[idx_distinct[0]], forces[idx_distinct[0]], label=labels[0], color=colors[0], lw=2)
ax1.plot(arc_lengths[idx_distinct[1]],forces[idx_distinct[1]], label=labels[1], color=colors[1], lw=2)
ax1.set_xlabel('Arc Length [m]')
ax1.set_ylabel('Force Magnitude [N/m]')
ax1.set_title('Force Distribution Along Coils')
ax1.legend(loc='upper right')
ax1.grid(color='lightgray', linestyle='--', linewidth=0.5)


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
    ax2.plot(ts*1000, Is/1000, label=labels[ii], color=colors[ii], lw=2)  # in kA and ms
    ax2.axhline(I_i/1000, color=colors[ii], lw=2, ls='--')
ax2.set_xlabel('Time [ms]')
ax2.set_ylabel('Current [kA]')
ax2.set_title('Current Ramp-up in Coils')

plt.tight_layout()
plt.savefig(outdir + "force_inductance_analysis.pdf", format='pdf')
plt.show()