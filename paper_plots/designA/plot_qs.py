#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D

class HandlerVertLines(HandlerBase):
    def __init__(self, pad=0.15, **kwargs):
        super().__init__(**kwargs)
        self.pad = pad

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        l1, l2 = orig_handle

        x0, x1 = xdescent, xdescent + width

        y_top = ydescent + height * (1 - self.pad)
        y_bot = ydescent + height * (0 + self.pad)

        a1 = Line2D([x0, x1], [y_top, y_top])
        a2 = Line2D([x0, x1], [y_bot, y_bot])

        for a, l in [(a1, l1), (a2, l2)]:
            a.set_linestyle(l.get_linestyle())
            a.set_linewidth(l.get_linewidth())
            a.set_color(l.get_color())
            a.set_transform(trans)

        return [a1, a2]


matplotlib.use("Qt5Agg") 

fs = 15
_new_black = '#373737'
sns.set_theme(style='ticks', font_scale=0.75, rc={
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'svg.fonttype': 'none',
    'text.usetex': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.size': fs,
    'axes.labelsize': fs,
    'axes.titlesize': fs,
    'axes.labelpad': 2,
    'axes.linewidth': 0.5,
    'axes.titlepad': 4,
    'lines.linewidth': 0.5,
    'legend.fontsize': fs,
    'legend.title_fontsize': fs,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'xtick.major.size': 2,
    'xtick.major.pad': 1,
    'xtick.major.width': 0.5,
    'ytick.major.size': 2,
    'ytick.major.pad': 1,
    'ytick.major.width': 0.5,
    'xtick.minor.size': 2,
    'xtick.minor.pad': 1,
    'xtick.minor.width': 0.5,
    'ytick.minor.size': 2,
    'ytick.minor.pad': 1,
    'ytick.minor.width': 0.5,

    # Avoid black unless necessary
    'text.color': _new_black,
    'patch.edgecolor': _new_black,
    'patch.force_edgecolor': False, # Seaborn turns on edgecolors for histograms by default and I don't like it
    'hatch.color': _new_black,
    'axes.edgecolor': _new_black,
    # 'axes.titlecolor': _new_black # should fallback to text.color
    'axes.labelcolor': _new_black,
    'xtick.color': _new_black,
    'ytick.color': _new_black,
    "legend.fontsize": 12,     # Legend font
    # Default colormap - personal preference
    # 'image.cmap': 'inferno'
})

before_color = '#1f77b4'
after_color  = '#d62728'
coil1_color  = '#9467bd'
coil2_color  = '#ff7f0e'

refline_color = '#6baed6'  
refline_style = (0, (5, 3))

data_010483 = pd.read_csv('serial0104183.txt', sep=',')
dataA_before = pd.read_csv('0104183_symmetrized.txt', sep=',')
dataA_after = pd.read_csv("designA_after_scaled.txt", sep=",")

fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

axes[0].plot(
    dataA_before.edge_iota, 100 * dataA_before.nonqs**0.5,
    color=before_color, lw=2, linestyle='--', 
) 
axes[0].plot(
    data_010483.edge_iota, 100 * data_010483.nonqs**0.5,
    color=before_color, lw=2, linestyle='-'
)

axes[0].plot(
    dataA_after.edge_iota, 100 * dataA_after.nonqs**0.5,
    color=after_color, lw=2, linestyle='-'
)

axes[0].set_xlabel('edge ι')
axes[0].set_ylabel('QA error [%]')
axes[0].set_xlim([0.1, 0.3])
axes[0].grid()

from matplotlib.lines import Line2D
combo = (Line2D([0], [0], color=before_color, lw=2, linestyle='--'),Line2D([0], [0], color=before_color, lw=2, linestyle='-'))
star_lite= Line2D([0], [0], color=after_color,  lw=2, linestyle='-')
legend_labels = [
    'design 0104183',
    'STAR_Lite (design A)'
]
axes[0].legend(
    handles=[combo, star_lite],
    labels=legend_labels,
    handler_map={combo: HandlerVertLines(pad=0.25)},  # <-- increase pad
    )


axes[1].plot(
    dataA_after.edge_iota, np.abs(dataA_after.current1)/1e3,
    label='L coil', color= _new_black, lw=2, linestyle='--'
) 
axes[1].plot(
    dataA_after.edge_iota, np.abs(dataA_after.current2)/1e3,
    label='T coil', color= _new_black, lw=2, linestyle='-',
)   
axes[1].grid(True)  
axes[1].set_xlabel('edge ι')
axes[1].set_ylabel('current (kA)')
axes[1].set_xlim([0.1, 0.3])
axes[1].set_ylim([0, 60])
axes[1].set_yticks([10, 20, 30, 40, 50, 60])
axes[1].legend()

axes[0].text(
    0.02, 0.98, 'a)',
    transform=axes[0].transAxes,
    fontsize=16,
    va='top',
    ha='left'
)

axes[1].text(
    0.02, 0.98, 'b)',
    transform=axes[1].transAxes,
    fontsize=16,
    va='top',
    ha='left'
)

xvals = (0.1364426675770125, 0.1883043138864307,0.2631113300760912)
labels = ['$\iota_1=0.13$', '$\iota_2=0.18$', '$\iota_3=0.26$']

for ax in axes:
    for x, lab in zip(xvals, labels):
        ax.axvline(x, ls='--', color='0.7', lw=2)
        ax.text(
            x+0.0025, 0.07, lab,
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            fontsize=12,
            color='0.4'
        )

plt.tight_layout()
plt.savefig("results_qs.png", dpi=300, bbox_inches='tight')
plt.show()
