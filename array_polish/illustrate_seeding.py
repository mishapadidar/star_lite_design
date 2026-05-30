#!/usr/bin/env python3
"""
Seed spacing for manifold tracing: which distribution tiles the manifold
seamlessly under ONE iteration of the return map.

Two fixed-point types:
  hyperbolic:  r -> lambda * r          (multiplicative; tile = zoom level)
  parabolic :  r -> r + 0.5*c*r^2       (quadratic drift; tile = 1/r segment)

The map-invariant seeding is the one that becomes EVENLY SPACED in the
coordinate where the map is a pure shift:
  hyperbolic -> shift in log r -> uniform in log r  (geomspace)
  parabolic  -> shift in 1/r   -> uniform in 1/r

Usage:
  python illustrate_seeding.py --map hyperbolic [--lam 2] [--r0 0.025] [--n 8]
  python illustrate_seeding.py --map parabolic  [--c 40]  [--r0 0.02]  [--n 8]
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="manifold seed-spacing illustration")
parser.add_argument("--map", choices=["hyperbolic", "parabolic"], default="hyperbolic")
parser.add_argument("--lam", type=float, default=2.0, help="hyperbolic eigenvalue (>1)")
parser.add_argument("--c", type=float, default=40.0, help="parabolic 2nd-order coeff c (drift = c/2 r^2)")
parser.add_argument("--r0", type=float, default=None, help="inner edge of the fundamental domain")
parser.add_argument("--n", type=int, default=8, help="number of seeds")
args = parser.parse_args()

KIND = args.map
LAM = args.lam
ALPHA = 0.5 * args.c          # drift coefficient: r -> r + alpha r^2
N = args.n
R0 = args.r0 if args.r0 is not None else (0.025 if KIND == "hyperbolic" else 0.02)


def apply_map(r):
    if KIND == "hyperbolic":
        return LAM * r
    return r + ALPHA * r**2


def fundamental_domain(r0):
    if KIND == "hyperbolic":
        return r0, LAM * r0
    return r0, apply_map(r0)        # [r0, T(r0))


def make_seeds(method, dom):
    a, b = dom
    if method == "linspace":
        return np.linspace(a, b, N)
    if method == "geomspace":
        return np.geomspace(a, b, N)
    if method == "inv":            # uniform in 1/r
        return 1.0 / np.linspace(1.0 / a, 1.0 / b, N)
    raise ValueError(method)


# Which methods to show, and which is the invariant one for this map.
if KIND == "hyperbolic":
    methods = [("linspace", "linspace"),
               ("geomspace (invariant)", "geomspace")]
else:
    methods = [("linspace", "linspace"),
               ("geomspace (approx)", "geomspace"),
               ("uniform in 1/r (invariant)", "inv")]

dom = fundamental_domain(R0)
b0, b1, b2 = dom[0], dom[1], apply_map(dom[1])   # domain boundaries r0, T(r0), T^2(r0)

fig, axes = plt.subplots(len(methods), 1, figsize=(9, 2.2 * len(methods) + 0.6),
                         sharex=True, squeeze=False)
axes = axes[:, 0]
for ax, (title, method) in zip(axes, methods):
    seeds = make_seeds(method, dom)
    img = apply_map(seeds)
    ax.scatter(seeds, np.ones_like(seeds), s=45, color="tab:blue",
               zorder=3, label="seeds (tile k)")
    ax.scatter(img, 0.5 * np.ones_like(img), s=45, facecolors="none",
               edgecolors="tab:red", zorder=3, label="images (tile k+1)")
    for s, i in zip(seeds, img):
        ax.plot([s, i], [1.0, 0.5], color="0.78", lw=0.8, zorder=1)
    for x in (b0, b1, b2):
        ax.axvline(x, color="k", ls=":", lw=0.8)
    sd = np.diff(seeds)
    ax.set_title(f"{title}    seed-gap ratio max/min = {sd.max()/sd.min():.2f}")
    ax.set_yticks([0.5, 1.0]); ax.set_yticklabels(["images", "seeds"])
    ax.set_ylim(0.2, 1.35)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

axes[-1].set_xlabel("r   (linear scale)")
if KIND == "hyperbolic":
    fig.suptitle(f"hyperbolic  r -> {LAM:g} r    domain [{b0:g}, {b1:g}]")
else:
    fig.suptitle(f"parabolic  r -> r + {ALPHA:g} r^2    domain [{b0:g}, {b1:.4g}]")
fig.tight_layout()
out = f"seeding_{KIND}.png"
fig.savefig(out, dpi=150)
print(f"wrote {out}")

# Numbers: consecutive differences in r, and in 1/r (constant for the
# translation-invariant method of each map).
for title, method in methods:
    seeds = make_seeds(method, dom)
    img = apply_map(seeds)
    print(f"\n[{KIND}] {title}")
    print("  seeds       :", np.array2string(seeds, precision=4))
    print("  seed r-diffs:", np.array2string(np.diff(seeds), precision=4))
    print("  1/r-diffs   :", np.array2string(np.diff(1.0 / seeds), precision=3))
    print("  image r-diffs:", np.array2string(np.diff(img), precision=4))

plt.show()
