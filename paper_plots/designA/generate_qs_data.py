#!/usr/bin/env python3
# this script makes the data for the QS vs rotational transform plot

import os
import numpy as np
import sys
from tqdm import tqdm
import pandas as pd
from simsopt._core import load
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, NonQuasiSymmetricRatio, CurveXYZFourierSymmetries, CurveLength, Volume
from simsopt.field import BiotSavart
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine

def compute_data(current0, current1, boozer_surface, axis):
    coils = boozer_surface.biotsavart.coils
    
    # rescale the currents in the L and T coils
    coils[0].current.x = current0
    coils[-1].current.x = current1
    
    # recompute the magnetic axis
    res_fl = axis.run_code(axis.res['length'])
    assert res_fl['success']
    
    # compute the mean modB on the new magnetic axis
    bs = BiotSavart(coils)
    bs.set_points(axis.curve.gamma())
    mean_modB = np.mean(bs.AbsB().flatten())
    target_modB = 87.5*1e-3
    scale = target_modB/mean_modB
    
    # scale the L and T coil currents to the target_modB
    coils[0].current.x*=scale
    coils[-1].current.x*=scale
    new_mean_modB = np.mean(bs.AbsB().flatten())
    assert np.abs(new_mean_modB-target_modB) < 1e-16

    iota, G = boozer_surface.res['iota'], boozer_surface.res['G']

    # compute surface first using LBFGS, this will just be a rough initial guess
    res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-13, maxiter=50, constraint_weight=100., iota=iota, G=G)
    boozer_surface.need_to_run_code = True
    
    # polish 
    res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-13, maxiter=50, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')
    boozer_surface.need_to_run_code = True
    
    # Newton
    res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=1e-12, maxiter=20, iota=res['iota'], G=res['G'], verbose=False, weight_inv_modB=True)
    assert res['success'] and not boozer_surface.surface.is_self_intersecting()
    
    nonQS = NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils))
    dat = {'nonqs': nonQS.J(), 'current1':coils[0].current.get_value(), 'current2':coils[-1].current.get_value(), 'edge_iota': res['iota']}
    return dat

name = "../../designs/designA_after_scaled.json"
dat = load(name)

if 'designA_after_scaled.json' in name:
    [boozer_surfaces, iota_Gs, in_axes, in_xpoints] = dat
    axes_new = []
    for axis_RZ, boozer_surface in zip(in_axes, boozer_surfaces):
        stellsym=True
        nfp = 2
        order=16
        tmp = CurveXYZFourierSymmetries(axis_RZ.quadpoints, order, nfp, stellsym)
        tmp.least_squares_fit(axis_RZ.gamma())
        quadpoints = np.linspace(0, 1/nfp, 2*order+1, endpoint=False)
        axis = CurveXYZFourierSymmetries(quadpoints, order, nfp, stellsym)
        axis.x = tmp.x
        axis_fl = PeriodicFieldLine(BiotSavart(boozer_surface.biotsavart.coils), axis)
        res = axis_fl.run_code(CurveLength(axis_fl.curve).J())
        assert res['success']
        axes_new.append(axis_fl)
    axes = axes_new

elif '0104183_symmetrized.json' in name:
    [in_axis, in_surface, in_coils] = dat
    in_vol = Volume(in_surface)
    in_vol_target = -0.0555166952946639

    in_boozer_surface = BoozerSurface(BiotSavart(coils), in_surface, in_vol, in_vol_target, options={'newton_tol':1e-12, 'newton_maxiter':10})
    boozer_surfaces = [in_boozer_surface]
    axes = [in_axis]
    xpoints = [None]

# unfix all the currents
for boozer_surface in boozer_surfaces:
    boozer_surface.biotsavart.unfix_all()

for axis in axes:
    if isinstance(axis, PeriodicFieldLine):
        axis.biotsavart.unfix_all()

# CREATE A BOOZERLS SURFACE WHICH IS MORE ROBUST
s_orig = boozer_surfaces[0].surface
phis = np.linspace(0, 1/s_orig.nfp, 2*s_orig.ntor+10, endpoint=False)
thetas = np.linspace(0, 1, 2*s_orig.mpol+10, endpoint=False)
stemp = SurfaceXYZTensorFourier(mpol=s_orig.mpol, ntor=s_orig.ntor, stellsym=s_orig.stellsym, nfp=s_orig.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
stemp.x = s_orig.x

phis = np.linspace(0, 1/s_orig.nfp, 2*s_orig.ntor+10, endpoint=False)
thetas = np.linspace(0, 1, 2*s_orig.mpol+10, endpoint=False)
surface_ls = SurfaceXYZTensorFourier(mpol=6, ntor=6, stellsym=s_orig.stellsym, nfp=s_orig.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
surface_ls.least_squares_fit(stemp.gamma())

vol = Volume(surface_ls)
vol_target = vol.J()
iota0 = 0.2
current_sum = sum(abs(c.current.get_value()) for c in boozer_surfaces[0].biotsavart.coils)
G0 = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

## compute the surface
boozer_surface = BoozerSurface(BiotSavart(boozer_surfaces[0].biotsavart.coils), surface_ls, vol, vol_target, options={'newton_tol':1e-12, 'newton_maxiter':10})

boozer_surface.need_to_run_code = True
# compute surface first using LBFGS, this will just be a rough initial guess
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=1e-13, \
        maxiter=50, constraint_weight=100., iota=iota0, G=G0, weight_inv_modB=True)
boozer_surface.need_to_run_code = True

# now drive the residual down using a specialised least squares algorithm
res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=1e-13, \
        maxiter=50, constraint_weight=100., iota=res['iota'], G=res['G'], method='manual')

boozer_surface.need_to_run_code = True
res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=1e-12, \
        maxiter=20, iota=res['iota'], G=res['G'], verbose=True, weight_inv_modB=True)

if res['success'] and not surface_ls.is_self_intersecting():
    xorig = surface_ls.x.copy()
    boozer_surface_orig = boozer_surface
else:
    print('failed')

# use the original axis
axis = axes[0]

# save the original currents, axis
c0 = boozer_surface.biotsavart.coils[0].current.x 
c1 = boozer_surface.biotsavart.coils[-1].current.x

adofs_orig = axis.curve.x.copy()
dofs_orig = boozer_surface.biotsavart.x.copy()
sdofs_orig = boozer_surface.surface.x.copy()
iotaG = [res['iota'], res['G']]

# SCALE UP THE L-COIL CURRENT
dat_list = {'current1':[], 'current2':[], 'nonqs':[], 'edge_iota':[]}
for r in tqdm(1. + 0.01*np.arange(50)):
    try:
        dat = compute_data(c0*r, c1, boozer_surface, axis)
    except:
        break

    for key in dat.keys():
        dat_list[key].append(dat[key])

# reset back to the original surface, axis
axis.curve.x = adofs_orig
boozer_surface.biotsavart.x = dofs_orig
boozer_surface.surface.x = sdofs_orig
boozer_surface.need_to_run_code = True
res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=1e-12, \
        maxiter=20, iota=iotaG[0], G=iotaG[1], verbose=True, weight_inv_modB=True)

# SCALE DOWN THE L-COIL CURRENT
for r in tqdm(1. - 0.01*np.arange(50)):
    try:
        dat = compute_data(c0*r, c1, boozer_surface, axis)
    except:
        break

    for key in dat.keys():
        dat_list[key].insert(0, dat[key])

base = os.path.splitext(os.path.basename(name))[0]
outfile = base + ".txt"  # or ".csv" if you want CSV
df = pd.DataFrame(dat_list)
df.to_csv(outfile, index=False, sep=",")
