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
dataA_before = pd.read_csv('designA.txt', sep=',')
dataA_after = pd.read_csv("designA_after_scaled.txt", sep=",")  # \s+ = one or more spaces/tabs

# Create a 1x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(12.0, 7), sharey=True, sharex=True,
        gridspec_kw={'wspace': 0.25, 'hspace':0})

axes[:, [1, 0, 2]] = axes

plt.subplots_adjust(
    bottom=0.25,   # <- increases space for the legend
)

energy_sets = [
    ('20eV/output_designA_init_scaled',   '20eV/output_designA_after_scaled',   '20eV/output_serial0104183_iota'),
    ('2.86keV/output_designA_init_scaled','2.86keV/output_designA_after_scaled',   '2.86keV/output_serial0104183_iota'),
]

tmax = 2e-2

    
for idx, (init_dir, after_dir, orig_dir) in enumerate(energy_sets):
    for OUT_DIR, state in [(init_dir, 'init'), (after_dir, 'after'), (orig_dir, 'orig')]:
        for config in range(3):
            try:
                cidx = [1,0,2][config] if "orig" in state else config
                loss_times=np.loadtxt(OUT_DIR + f'/losses_{cidx}.txt') 
                nparticles = loss_times.size
                loss_times = loss_times[loss_times<tmax]
                sort_loss_times = np.sort(loss_times)
                cumulative_fraction = np.arange(1, len(sort_loss_times) + 1) / nparticles
                sort_loss_times = np.concatenate(([0], sort_loss_times, [tmax]))
                cumulative_fraction = np.concatenate(([0], cumulative_fraction, [cumulative_fraction[-1] if cumulative_fraction.size > 0 else 0.]))

                color = before_color if 'init' in OUT_DIR or 'iota' in OUT_DIR else after_color
                ls = '--' if 'init' in OUT_DIR else '-'
                axes[idx, config].semilogx(sort_loss_times, 100*cumulative_fraction, c=color,  linestyle=ls, lw=2)
                axes[idx, config].set_xlabel('time [s]')
                axes[idx, config].set_yticks([10, 20, 30, 40, 50])
                
                x_end = sort_loss_times[-1]
                y_end = 100 * cumulative_fraction[-1]
                axes[idx, config].text(
                    x_end * 1.8,          # move slightly right in log space
                    y_end,
                    f"{y_end:.1f}%",
                    color=color,
                    fontsize=fs - 4,
                    va='center',
                    ha='left',
                    clip_on=False
                )
            except:
                print('not found')
            axes[idx, config].set_xticks([1e-6, 1e-5, 1e-4, 1e-3, 1e-2])
            axes[idx, config].grid()

panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

axes[0, 1].set_ylabel('loss fraction [%]')
axes[1, 1].set_ylabel('loss fraction [%]')

for idx in range(3):
    axes[0, idx].text(
        0.02, 0.98, panel_labels[idx+0],
        transform=axes[0, idx].transAxes,
        fontsize=fs-2,
        va='top',
        ha='left')
    axes[1, idx].text(
        0.02, 0.98, panel_labels[idx+3],
        transform=axes[1,idx].transAxes,
        fontsize=fs-2,
        va='top',
        ha='left'

    )

from matplotlib.lines import Line2D

combo = (Line2D([0], [0], color=before_color, lw=2, linestyle='--'),Line2D([0], [0], color=before_color, lw=2, linestyle='-'))
star_lite= Line2D([0], [0], color=after_color,  lw=2, linestyle='-')
legend_labels = [
    'device 0104183',
    'STAR_Lite (design A)'
]
fig.legend(
    handles=[combo, star_lite],
    labels=['design 0104183', 'STAR_Lite (design A)'],
    handler_map={combo: HandlerVertLines(pad=0.25)},  # <-- increase pad
    handlelength=3,
    bbox_to_anchor=(0.775, 0.15),
    fontsize=fs,
        ncol=2
    )

col_labels = ['$\iota_2=0.18$', '$\iota_1=0.13$', '$\iota_3=0.26$']
for col, label in enumerate(col_labels):
    axes[0, col].set_title(
        label,
        fontsize=fs,
        pad=12   
    )

row_labels = ['20 eV', '2.86 keV']

for row, label in enumerate(row_labels):
    ax = axes[row, -1] 
    bbox = ax.get_position()

    fig.text(
        bbox.x1 + 0.04,             
        0.5 * (bbox.y0 + bbox.y1),  
        label,
        rotation=-90,
        va='center',
        ha='left',
        fontsize=fs    )

plt.savefig("results_losses.png", dpi=300, bbox_inches='tight')
plt.show()
