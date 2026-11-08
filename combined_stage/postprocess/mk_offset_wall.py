#!/usr/bin/env python3
"""Conformal FLARE wall for a combined-stage device (these runs have no vacuum vessel).

The wall is the run's VMEX boundary offset outward by --offset in every cross-section
(src/preprocessing/kisslinger.expand_kisslinger), written as a half-period Kisslinger file in cm
(stellarator-symmetric, like the json vessels).  Before writing it reports the clearances that
decide whether the wall is usable for strike lines: the X-line must lie INSIDE it (so the divertor
legs reach it) and the coils must stay OUTSIDE it.

--offset auto scans --scan offsets (smallest first) and keeps the first wall that has the whole
X-line at least --xline-margin inside it and every coil point outside; runs without an X-line
(no xpoints/xpoint_<tag>.json) only need the coils outside.  The chosen offset is printed as
``offset <m>`` on the last line.

usage: mk_offset_wall.py <run dir> <tag: final | outerNNN> <out.kisslinger> [--offset 0.15 | auto] [--nphi 91]
"""
import argparse
import os
import sys

import netCDF4
import numpy as np
from matplotlib.path import Path as MplPath
from simsopt._core import load

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 4))
sys.path.insert(0, os.path.join(REPO, 'src'))
from preprocessing.kisslinger import expand_kisslinger, read_kisslinger_raw, write_kisslinger_raw  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('run')
p.add_argument('tag')
p.add_argument('out')
p.add_argument('--offset', default='0.15', help='outward offset of the VMEX boundary [m], or "auto"')
p.add_argument('--scan', default='0.08:0.30:0.01', help='auto offsets lo:hi:step [m]')
p.add_argument('--xline-margin', type=float, default=0.03,
               help='auto: minimum distance of the X-line inside the wall [m]')
p.add_argument('--nphi', type=int, default=91, help='planes over the half field period')
p.add_argument('--ntheta', type=int, default=360)
a = p.parse_args()

w = netCDF4.Dataset(os.path.join(a.run, 'wout', f'wout_combined_stage_{a.tag}.nc'))
xm, xn, nfp = w['xm'][:], w['xn'][:], int(w['nfp'][:])
rmnc, zmns = w['rmnc'][:][-1], w['zmns'][:][-1]
phi = np.linspace(0.0, np.pi / nfp, a.nphi)
th = np.linspace(0.0, 2 * np.pi, a.ntheta, endpoint=False)
ang = xm[None, None, :] * th[None, :, None] - xn[None, None, :] * phi[:, None, None]
R, Z = (rmnc * np.cos(ang)).sum(-1), (zmns * np.sin(ang)).sum(-1)
R, Z = np.concatenate([R, R[:, :1]], 1), np.concatenate([Z, Z[:, :1]], 1)   # closed contours

tmp = a.out + '.boundary'
# symmetry code -nfp: stellarator-symmetric half period (see preprocessing.vessel)
write_kisslinger_raw(tmp, 'vmex_boundary', -nfp, [0.0, 0.0], np.degrees(phi), 100 * np.stack([R, Z], -1))

xfile = os.path.join(a.run, 'xpoints', f'xpoint_{a.tag}.json')
xline = load(xfile)['curve'].gamma() if os.path.isfile(xfile) else None
coils = np.concatenate([c.curve.gamma() for c in load(os.path.join(a.run, 'coils', f'coils_{a.tag}.json'))])


def check(offset):
    """Write the wall at this offset; return (X-line inside fraction, its min distance, coils inside fraction,
    coil clearance) -- distances in m, None when there is nothing to measure."""
    try:
        expand_kisslinger(tmp, a.out, 100 * offset)
    except ValueError as exc:
        # a node-normal offset folds where the boundary is concave on a scale below the offset (designA's bean
        # side at x2 already at 8 cm): use the distance envelope, which rounds the pocket over
        print(f'  {100 * offset:.0f} cm: normal offset folds ({exc}); using the distance envelope')
        expand_kisslinger(tmp, a.out, 100 * offset, method='envelope')
    _, _, _, wphi, wrz = read_kisslinger_raw(a.out)
    wrz = wrz / 100

    def inside(points):
        """Per point: is it inside the wall, and its distance to the wall contour in its plane."""
        flags, dists = [], []
        for x, y, z in points:
            f = np.mod(np.arctan2(y, x), 2 * np.pi / nfp)   # fold into the half period by stellarator symmetry
            flip = f > np.pi / nfp
            f = 2 * np.pi / nfp - f if flip else f
            rz = wrz[int(np.argmin(np.abs(np.radians(wphi) - f)))].copy()
            if flip:
                rz[:, 1] *= -1
            q = np.array([np.hypot(x, y), z])
            flags.append(MplPath(rz).contains_point(q))
            dists.append(np.min(np.linalg.norm(rz - q, axis=1)))
        return np.array(flags), np.array(dists)

    xin, xd = inside(xline) if xline is not None else (np.ones(1, bool), np.full(1, np.inf))
    cin, cd = inside(coils)
    return (xin.mean(), xd[xin].min() if xin.any() else None,
            cin.mean(), cd[~cin].min() if (~cin).any() else None)


def report(offset, r):
    xf, xd, cf, cd = r
    print(f'wall: VMEX boundary + {100 * offset:.1f} cm  ->  {a.out}')
    if xline is None:
        print('X-line: none in this run')
    else:
        print(f'X-line: {xf:.1%} inside the wall' + (f', min distance to the wall {100 * xd:.1f} cm' if xd else ''))
    print(f'coils : {cf:.1%} inside the wall' + (f', min clearance {100 * cd:.1f} cm' if cd else ''))


try:
    if a.offset == 'auto':
        lo, hi, step = (float(x) for x in a.scan.split(':'))
        for offset in np.arange(lo, hi + step / 2, step):
            r = check(offset)
            if r[0] == 1.0 and (xline is None or r[1] >= a.xline_margin) and r[2] == 0.0:
                break
        else:
            sys.exit(f'ERROR: no offset in {a.scan} m keeps the X-line {100 * a.xline_margin:g} cm inside the wall '
                     'and the coils outside it')
    else:
        offset = float(a.offset)
        r = check(offset)
finally:
    os.remove(tmp)
report(offset, r)
if r[0] < 1.0 or r[2] > 0.0:
    sys.exit('ERROR: wall does not separate the X-line from the coils; change --offset')
print(f'offset {offset:.3f}')
