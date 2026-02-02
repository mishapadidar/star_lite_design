#!/usr/bin/env python3
# this script makes the data for the QS vs rotational transform plot
# for the perturbed stellarators

import sys
import os
from pathlib import Path
import numpy as np
from numpy.random import Generator, PCG64DXSM, SeedSequence
from tqdm import tqdm
import pandas as pd
from simsopt._core import load
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, NonQuasiSymmetricRatio, CurveXYZFourierSymmetries, CurveLength, Volume, GaussianSampler, CurvePerturbed, PerturbationSample
from simsopt.field import BiotSavart, Coil
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine

TARGET_MODB = 87.5e-3
LBFGS_TOL = 1e-13
NEWTON_TOL = 1e-12
NEWTON_MAXITER = 30
LBFGS_MAXITER = 50
CONSTRAINT_WEIGHT = 100.

def get_nonstellsym_surface(in_surface, mpol_new=5, ntor_new=5):
    phis = np.linspace(0, 1., in_surface.nfp*(2*in_surface.ntor+10), endpoint=False)
    thetas = np.linspace(0, 1, 2*in_surface.mpol+10, endpoint=False)
    stemp = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    stemp.x = in_surface.x
    
    surface_ls = SurfaceXYZTensorFourier(mpol=mpol_new, ntor=in_surface.nfp*ntor_new, stellsym=False, nfp=1, quadpoints_phi=phis, quadpoints_theta=thetas)
    surface_ls.least_squares_fit(stemp.gamma())
    return surface_ls

def get_nonstellsym_curve(curve, order_new=None):
    stellsym=curve.stellsym
    nfp = curve.nfp

    if order_new is None:
        order_new=curve.order
        
    order = curve.nfp * order_new
    quadpoints = np.linspace(0, 1, 2*order+1, endpoint=False)
    tmp = CurveXYZFourierSymmetries(quadpoints, curve.order, nfp, stellsym)
    tmp.x = curve.x

    curve_new = CurveXYZFourierSymmetries(quadpoints, order, 1, False)
    curve_new.least_squares_fit(tmp.gamma())
    return curve_new

def compute_data(current0, current1, current2, boozer_surface, axis):
    coils = boozer_surface.biotsavart.coils
    
    # rescale the currents in the L and T coils
    coils[0].current.x = current0
    coils[1].current.x = current1
    coils[2].current.x = current2
    
    # recompute the magnetic axis in the configuration with a new current ratio
    res_fl = axis.run_code(axis.res['length'])
    assert res_fl['success']
    
    # compute the mean modB on the new magnetic axis
    bs = BiotSavart(coils)
    bs.set_points(axis.curve.gamma())
    mean_modB = np.mean(bs.AbsB().flatten())
    scale = TARGET_MODB/mean_modB
    
    # scale the L and T coil currents to the TARGET_MODB
    coils[0].current.x*=scale
    coils[1].current.x*=scale
    coils[2].current.x*=scale
    new_mean_modB = np.mean(bs.AbsB().flatten())
    assert np.abs(new_mean_modB-TARGET_MODB) < 1e-16

    iota, G = boozer_surface.res['iota'], boozer_surface.res['G']

    # compute surface first using LBFGS, this will just be a rough initial guess
    res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=LBFGS_TOL, maxiter=LBFGS_MAXITER, constraint_weight=CONSTRAINT_WEIGHT, iota=iota, G=G)
    boozer_surface.need_to_run_code = True
    
    # polish 
    res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=LBFGS_TOL, maxiter=LBFGS_MAXITER, constraint_weight=CONSTRAINT_WEIGHT, iota=res['iota'], G=res['G'], method='manual')
    boozer_surface.need_to_run_code = True
    
    # Newton
    res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=NEWTON_TOL, maxiter=NEWTON_MAXITER, iota=res['iota'], G=res['G'], verbose=False, weight_inv_modB=True)
    assert res['success'] and not boozer_surface.surface.is_self_intersecting()
    
    nonQS = NonQuasiSymmetricRatio(boozer_surface, BiotSavart(coils))
    dat = {'nonqs': nonQS.J(), 'current1':coils[0].current.get_value(), 'current2':coils[1].current.get_value(), 'current3':coils[2].current.get_value(), 'edge_iota': res['iota']}
    return dat

fname = sys.argv[1]
dat = load(fname)


#in_vol_target = -0.0555166952946639
if 'serial0104183_iota.json' in fname:
    [boozer_surfaces, iota_Gs, in_axes] = load(fname)
    axes = []
    for axis, boozer_surface in zip(in_axes, boozer_surfaces):
        axis_fl = PeriodicFieldLine(BiotSavart(boozer_surface.biotsavart.coils), axis)
        res = axis_fl.run_code(CurveLength(axis_fl.curve).J())
        assert res['success']
        axes.append(axis_fl)
else:
    raise Exception("This script only works for serial0104183_iota.json")


# unfix all the currents
for boozer_surface in boozer_surfaces:
    boozer_surface.biotsavart.unfix_all()

for axis in axes:
    if isinstance(axis, PeriodicFieldLine):
        axis.biotsavart.unfix_all()

surface_orig = boozer_surfaces[0].surface
coils_orig = boozer_surfaces[0].biotsavart.coils
curves_orig = [c.curve for c in coils_orig]


# GENERATE THE PERTURBED COILS
seed = int(sys.argv[2])
# for seed == -1, it doesn't matter what the seed is since sigma=0
seed = seed if seed > -1 else 0
L = float(sys.argv[3])
SIGMA = float(sys.argv[4])
sampler = GaussianSampler(curves_orig[0].quadpoints, SIGMA, L, n_derivs=2)
perturbed_curves = []
rg = Generator(PCG64DXSM(seed))
for c in curves_orig:
    pert = PerturbationSample(sampler, randomgen=rg)
    perturbed_curves.append(CurvePerturbed(c, pert))

coils_pert = [Coil(perturbed_curve, coil.current) for (perturbed_curve, coil) in zip(perturbed_curves, coils_orig)]

# CREATE A NON-STELLSYM BOOZERLS SURFACE WHICH IS MORE ROBUST
bs_pert = BiotSavart(coils_pert)
bs_pert.unfix_all()

surface_ls = get_nonstellsym_surface(surface_orig)
vol = Volume(surface_ls)
vol_target = -0.0555166952946639
iota0 = iota_Gs[0][0]
G0 = iota_Gs[0][1]

## compute the surface
boozer_surface = BoozerSurface(BiotSavart(coils_pert), surface_ls, vol, vol_target, options={'newton_tol':NEWTON_TOL, 'newton_maxiter':NEWTON_MAXITER})

boozer_surface.need_to_run_code = True
# compute surface first using LBFGS, this will just be a rough initial guess
res = boozer_surface.minimize_boozer_penalty_constraints_LBFGS(tol=LBFGS_TOL, \
        maxiter=LBFGS_MAXITER, constraint_weight=CONSTRAINT_WEIGHT, iota=iota0, G=G0, weight_inv_modB=True)
boozer_surface.need_to_run_code = True

# now drive the residual down using a specialised least squares algorithm
res = boozer_surface.minimize_boozer_penalty_constraints_ls(tol=LBFGS_TOL, \
        maxiter=LBFGS_MAXITER, constraint_weight=CONSTRAINT_WEIGHT, iota=res['iota'], G=res['G'], method='manual')

boozer_surface.need_to_run_code = True
res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=NEWTON_TOL, \
        maxiter=NEWTON_MAXITER, iota=res['iota'], G=res['G'], verbose=True, weight_inv_modB=True)

if res['success'] and not surface_ls.is_self_intersecting():
    xorig = surface_ls.x.copy()
    boozer_surface_orig = boozer_surface
else:
    print('failed')
    exit(-1)

# CREATE A NONSTELLSYM MAGNETIC AXIS
axis_orig = axes[0].curve
axis = get_nonstellsym_curve(axis_orig, order_new=16)
axis_fl = PeriodicFieldLine(BiotSavart(coils_pert), axis)
res_axis = axis_fl.run_code(CurveLength(axis_fl.curve).J())
assert res_axis['success']
axis_dofs_orig = axis.x.copy()

# save the original currents, axis
c0 = coils_pert[0].current.x 
c1 = coils_pert[1].current.x
c2 = coils_pert[2].current.x

adofs_orig = axis_fl.curve.x.copy()
dofs_orig = boozer_surface.biotsavart.x.copy()
sdofs_orig = boozer_surface.surface.x.copy()
iotaG = [res['iota'], res['G']]

# SCALE UP THE L-COIL CURRENT
dat_list = {'current1':[], 'current2':[], 'current3':[], 'nonqs':[], 'edge_iota':[]}
#for r in tqdm(1. + 0.01*np.arange(50)):
#    try:
#        dat = compute_data(c0*r, c1, c2*r, boozer_surface, axis_fl)
#    except Exception as e:
#        tqdm.write(f"Failed at r={r}: {e}")
#        break
#
#    for key in dat.keys():
#        dat_list[key].append(dat[key])

import ipdb;ipdb.set_trace()

# reset back to the original surface, axis
axis_fl.curve.x = adofs_orig
boozer_surface.biotsavart.x = dofs_orig
boozer_surface.surface.x = sdofs_orig
boozer_surface.need_to_run_code = True
res = boozer_surface.minimize_boozer_penalty_constraints_newton(tol=NEWTON_TOL, \
        maxiter=NEWTON_MAXITER, iota=iotaG[0], G=iotaG[1], verbose=True, weight_inv_modB=True)

# SCALE DOWN THE L-COIL CURRENT
for r in tqdm(1. - 0.01*np.arange(50)[1:]):
    try:
        dat = compute_data(c0*r, c1, c2*r, boozer_surface, axis_fl)
    except Exception as e:
        tqdm.write(f"Failed at r={r}: {e}")
        break

    for key in dat.keys():
        dat_list[key].insert(0, dat[key])

OUT_DIR = Path("output") / Path(fname).stem / f"L={L}/output_{SIGMA}/"
OUT_DIR.mkdir(parents=True, exist_ok=True)
outfile = OUT_DIR / f"{seed}.txt"  # or ".csv" if you want CSV
df = pd.DataFrame(dat_list)
df.to_csv(str(outfile), index=False, sep=",")
