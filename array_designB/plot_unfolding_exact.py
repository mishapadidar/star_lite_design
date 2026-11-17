#!/usr/bin/env python3
"""Figure of the exact mean unfolding constant E[C] of a snowflake over the toroidal sections.

Usage:
    plot_unfolding_exact.py <device_dir>

Reads <device_dir>/unfolding_EC_exact_profile.txt (written by mk_unfolding_exact.py: t, the
fraction of the full turn of the snowflake field line, and E[C] in the section through that
point) and writes <device_dir>/unfolding_EC_exact_profile.png: E[C] against t, with its mean
over the sections and its value at phi = 0. The X-points born from the snowflake under a
perturbation of amplitude sigma of the modular coils separate by C * sqrt(sigma).
Only needs numpy and matplotlib (venv_matplotlib).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1].rstrip("/")
t, ec = np.loadtxt(os.path.join(d, "unfolding_EC_exact_profile.txt"), unpack=True)
# mean over t with the trapezoidal rule on the (non-uniform) nodes
mean = float(np.sum(0.5 * (ec[1:] + ec[:-1]) * np.diff(t)) / (t[-1] - t[0]))
ec0 = float(ec[np.argmin(t)])

fig, ax = plt.subplots(figsize=(7.5, 4.4))
ax.plot(t, ec, lw=1.8, color="#2b6a99", label="E[C] in the section")
ax.axhline(mean, color="#d08a1c", ls="--", lw=1.2, label=f"mean over the sections: {mean:.3f}")
ax.plot([t.min()], [ec0], "o", color="#b23a3a", ms=5, label=f"section at $\\phi$ = 0: {ec0:.3f}")
ax.set_xlabel("fraction of the full turn of the snowflake field line")
ax.set_ylabel("E[C]  [m$^{1/2}$]")
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 2.0)      # the same range on every device, so that the figures compare by eye
ax.grid(alpha=0.3)
ax.legend(frameon=False, fontsize=9, loc="lower right")
ax.set_title(os.path.basename(d)[-70:] + "\nexact mean unfolding constant of the snowflake "
             f"(min {ec.min():.3f}, max {ec.max():.3f})", fontsize=8)
fig.tight_layout()
out = os.path.join(d, "unfolding_EC_exact_profile.png")
fig.savefig(out, dpi=150)
print(f"wrote {out}")
