#!/usr/bin/env python3

import os
import numpy as np
from scipy.optimize import minimize

from simsopt.field.coil import ScaledCurrent 
from simsopt.field import BiotSavart, Current, Coil, coils_via_symmetries
from simsopt.configs import get_ncsx_data
from simsopt.field import BiotSavart, coils_via_symmetries
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, curves_to_vtk, boozer_surface_residual, \
    Volume, MajorRadius, CurveLength, NonQuasiSymmetricRatio, Iotas
from simsopt.objectives import QuadraticPenalty
from simsopt.util import in_github_actions
from simsopt._core import load, save
from simsopt.geo import (SurfaceRZFourier, curves_to_vtk, create_equally_spaced_curves,
                         CurveLength, CurveCurveDistance, MeanSquaredCurvature,
                         LpCurveCurvature, ArclengthVariation, 
                        #  CurveBoozerSurfaceDistance,
                         BoozerResidual, CurveRZFourier)

def find_magnetic_axis(biotsavart, r0, z0, nfp, order, stellsym):
    from scipy.spatial.distance import cdist
    from scipy.optimize import fsolve
    
    n = r0.size
    if n % 2 == 0:
        n+=1
    
    length = 2*np.pi/nfp
    points = np.linspace(0, length, n, endpoint=False).reshape((n, 1))
    oneton = np.asarray(range(0, n)).reshape((n, 1))
    fak = 2*np.pi / length
    dists = fak * cdist(points, points, lambda a, b: a-b)
    np.fill_diagonal(dists, 1e-10)  # to shut up the warning
    if n % 2 == 0:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.tan(0.5 * dists)
    else:
        D = 0.5 \
            * np.power(-1, cdist(oneton, -oneton)) \
            / np.sin(0.5 * dists)

    np.fill_diagonal(D, 0)
    D *= fak
    phi = points

    def build_residual(rz):
        inshape = rz.shape
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        B = biotsavart.B()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        Br = np.cos(phi)*Bx + np.sin(phi)*By
        Bphi = np.cos(phi)*By - np.sin(phi)*Bx
        residual_r = D @ r - r * Br / Bphi
        residual_z = D @ z - r * Bz / Bphi
        return np.vstack((residual_r, residual_z)).reshape(inshape)

    def build_jacobian(rz):
        rz = rz.reshape((2*n, 1))
        r = rz[:n ]
        z = rz[n:]
        xyz = np.hstack((r*np.cos(phi), r*np.sin(phi), z))
        biotsavart.set_points(xyz)
        GradB = biotsavart.dB_by_dX()
        B = biotsavart.B()
        Bx = B[:, 0].reshape((n, 1))
        By = B[:, 1].reshape((n, 1))
        Bz = B[:, 2].reshape((n, 1))
        dxBx = GradB[:, 0, 0].reshape((n, 1))
        dyBx = GradB[:, 1, 0].reshape((n, 1))
        dzBx = GradB[:, 2, 0].reshape((n, 1))
        dxBy = GradB[:, 0, 1].reshape((n, 1))
        dyBy = GradB[:, 1, 1].reshape((n, 1))
        dzBy = GradB[:, 2, 1].reshape((n, 1))
        dxBz = GradB[:, 0, 2].reshape((n, 1))
        dyBz = GradB[:, 1, 2].reshape((n, 1))
        dzBz = GradB[:, 2, 2].reshape((n, 1))
        cosphi = np.cos(phi)
        sinphi = np.sin(phi)
        Br = cosphi*Bx + sinphi*By
        Bphi = cosphi*By - sinphi*Bx
        drBr = cosphi*cosphi * dxBx + cosphi*sinphi*dyBx + sinphi*cosphi*dxBy + sinphi*sinphi*dyBy
        dzBr = cosphi*dzBx + sinphi*dzBy
        drBphi = cosphi*cosphi*dxBy + cosphi*sinphi*dyBy - sinphi*cosphi*dxBx - sinphi*sinphi*dyBx
        dzBphi = cosphi*dzBy - sinphi*dzBx
        drBz = cosphi * dxBz + sinphi*dyBz
        # residual_r = D @ r - r * Br / Bphi
        # residual_z = D @ z - r * Bz / Bphi
        dr_resr = D + np.diag((-Br/Bphi - r*drBr/Bphi + r*Br*drBphi/Bphi**2).reshape((n,)))
        dz_resr = np.diag((-r*dzBr/Bphi + r*Br*dzBphi/Bphi**2).reshape((n,)))
        dr_resz = np.diag((-Bz/Bphi - r*drBz/Bphi + r*Bz*drBphi/Bphi**2).reshape((n,)))
        dz_resz = D + np.diag((-r*dzBz/Bphi + r*Bz*dzBphi/Bphi**2).reshape((n,)))
        return np.block([[dr_resr, dz_resr], [dr_resz, dz_resz]])
    
    x0 = np.vstack((r0, z0))
    
    soln = fsolve(build_residual, x0, fprime=build_jacobian, xtol=1e-13)
    
    res = build_residual(soln)
    norm_res = np.sqrt(np.sum(res**2))
    ma_success = norm_res < 1e-10

    xyz = np.hstack((soln[:n, None]*np.cos(phi), soln[:n, None]*np.sin(phi), soln[n:, None]))
    quadpoints = np.linspace(0, 1/nfp, n, endpoint=False)
    ma_fp = CurveRZFourier(quadpoints, order, nfp, stellsym)
    ma_fp.least_squares_fit(xyz)

    quadpoints = np.linspace(0, nfp, nfp*n, endpoint=False)
    ma_ft = CurveRZFourier(quadpoints, order, nfp, stellsym)
    ma_ft.x = ma_fp.x

    return ma_fp, ma_ft, ma_success, soln[:n], soln[n:]





"""
This example optimizes the NCSX coils and currents for QA on a single surface.  The objective is

    J = ( \int_S B_nonQA**2 dS )/(\int_S B_QA dS)
        + 0.5*(iota - iota_0)**2
        + 0.5*(major_radius - target_major_radius)**2
        + 0.5*max(\sum_{coils} CurveLength - CurveLengthTarget, 0)**2

We first compute a surface close to the magnetic axis, then optimize for QA on that surface.  
The objective also includes penalty terms on the rotational transform, major radius,
and total coil length.  The rotational transform and major radius penalty ensures that the surface's
rotational transform and aspect ratio do not stray too far from the value in the initial configuration.
There is also a penalty on the total coil length as a regularizer to prevent the coils from becoming
too complex.  The BFGS optimizer is used, and quasisymmetry is improved substantially on the surface.

More details on this work can be found at doi:10.1017/S0022377822000563 or arxiv:2203.03753.
"""

# Directory for output
OUT_DIR = "./output/"
os.makedirs(OUT_DIR, exist_ok=True)

print("Running 2_Intermediate/boozerQA.py")
print("================================")

[magnetic_axis,  surface, coils] = load('../designs/designB_after_scaled.json')

curves = [c.curve for c in coils]
base_curves = [curves[0], curves[ -2]]
print(BiotSavart(coils).x.size)

## COMPUTE THE INITIAL SURFACE ON WHICH WE WANT TO OPTIMIZE FOR QA##
# Resolution details of surface on which we optimize for qa
mpol = 10
ntor = 10
stellsym = True
nfp = 2

phis = np.linspace(0, 1/nfp, 2*ntor+1, endpoint=False)
thetas = np.linspace(0, 1, 2*mpol+1, endpoint=False)
stemp = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=surface.quadpoints_phi, quadpoints_theta=surface.quadpoints_theta)
stemp.least_squares_fit(surface.gamma())


bs_list = []
xpoints = []
coils_list = []
ratios=[1, 1.6, 0.6]
boozer_surfaces = []
for idx, r in enumerate(ratios):
    curve0 = coils[0].curve
    curve1 = coils[-2].curve
    
    scale = coils[0].current.scale
    curr_x = coils[0].current.full_x * r
    current0 = ScaledCurrent(Current(curr_x), scale)
    scale = coils[-2].current.scale
    curr_x = coils[-2].current.full_x
    current1 = ScaledCurrent(Current(curr_x), scale)
    
    current0.fix_all()
    coils_local = coils_via_symmetries([curve0], [current0], 2, False) \
            + coils_via_symmetries([curve1], [current1], 2, False)
    
    s = SurfaceXYZTensorFourier(
        mpol=mpol, ntor=ntor, stellsym=stellsym, nfp=nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
    s.x = stemp.x

    # Use a volume surface label
    vol = Volume(s)
    vol_target = vol.J()

    iota = 0.188
    current_sum = sum(abs(c.current.get_value()) for c in coils_local)
    G = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))

    bs = BiotSavart(coils_local)
    boozer_surface = BoozerSurface(bs, s, vol, vol_target, options={'newton_tol':1e-13, 'newton_maxiter':20})
    res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G)
    out_res = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)[0]
    print(f"NEWTON {res['success']}: iter={res['iter']}, iota={res['iota']:.3f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(out_res):.3e}, AR={s.aspect_ratio():.3f}")
    boozer_surfaces.append(boozer_surface)
    coils_list.append(coils_local)
    bs_list.append(BiotSavart(coils_local))

    N_axis=121
    
    curve_stellsym=False
    nfp=2
    order=16
    r0 = 0.55967771 * np.ones(N_axis)
    z0 = 0.15932784 * np.ones(N_axis)
    xp0_fp, xp0_ft, x_success, rx0, zx0= find_magnetic_axis(bs, r0, z0, nfp, order, curve_stellsym)
    xpoints.append([rx0, zx0])
    assert x_success
    print("found x-point manifold")

save([boozer_surfaces, coils_list], OUT_DIR + 'init.json')

#import ipdb;ipdb.set_trace()
#[_, in_coils_list] = load('output_init/opt.json')
#for cl, co in zip(coils_list, in_coils_list):
#    BiotSavart(cl).x = BiotSavart(co).x
#
#for boozer_surface in boozer_surfaces:
#    iota = 0.188
#    current_sum = sum(abs(c.current.get_value()) for c in coils_local)
#    G = 2. * np.pi * current_sum * (4 * np.pi * 10**(-7) / (2 * np.pi))
#
#    res = boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=iota, G=G)
#    out_res = boozer_surface_residual(s, res['iota'], res['G'], bs, derivatives=0)[0]
#    print(f"NEWTON {res['success']}: iter={res['iter']}, iota={res['iota']:.3f}, vol={s.volume():.3f}, ||residual||={np.linalg.norm(out_res):.3e}, AR={s.aspect_ratio():.3f}")
#    coils_list.append(coils_local)

## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
mr = MajorRadius(boozer_surfaces[0])
ls = [CurveLength(c) for c in base_curves]
brs = [BoozerResidual(boozer_surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]



# Threshold and weight for the coil-to-coil distance penalty in the objective function:
CC_THRESHOLD = 0.15
CC_WEIGHT = 1000

# Threshold and weight for the coil-to-surface distance penalty in the objective function:
CS_THRESHOLD = 0.15
CS_WEIGHT = 1e2

# Threshold and weight for the curvature penalty in the objective function:
CURVATURE_THRESHOLD = 4.3352595043*2
CURVATURE_WEIGHT = 1e-6

# Threshold and weight for the mean squared curvature penalty in the objective function:
MSC_THRESHOLD = 20.
MSC_WEIGHT = 1e-3

AL_WEIGHT = 1.
LENGTH_WEIGHT = 1e-1

IOTAS_WEIGHT=1e4
MR_WEIGHT=1e3

BR_WEIGHT=1e7

LENGTH_THRESHOLD = 4.0
J_major_radius = QuadraticPenalty(mr, 0.5, 'identity')  # target major radius is that computed on the initial surface
#J_iotas = sum([QuadraticPenalty(Iotas(boozer_surface), iota, 'identity') for boozer_surface, iota in zip(boozer_surfaces, [0.1882775, 0.10, 0.30])]) # target rotational transform is that computed on the initial surface
J_iotas = sum([QuadraticPenalty(Iotas(boozer_surface), boozer_surface.res['iota'], 'identity') for boozer_surface in boozer_surfaces]) # target rotational transform is that computed on the initial surface
nonQS_list = [NonQuasiSymmetricRatio(boozer_surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]
print([J.J()**0.5 for J in nonQS_list])
J_nonQSRatio = (1./len(boozer_surfaces)) * sum(nonQS_list)

Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, CC_THRESHOLD)
# Jcsdist = CurveBoozerSurfaceDistance(curves, boozer_surfaces[0], CS_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jal = sum(ArclengthVariation(curve) for curve in base_curves)

length_penalty = LENGTH_WEIGHT * sum([QuadraticPenalty(Jl, LENGTH_THRESHOLD, "max") for Jl in Jls])
cc_penalty = CC_WEIGHT * Jccdist
# cs_penalty =  CS_WEIGHT * Jcsdist
curvature_penalty = CURVATURE_WEIGHT * sum(Jcs)
msc_penalty = MSC_WEIGHT * sum(QuadraticPenalty(J, MSC_THRESHOLD, "max") for J in Jmscs)
Jbrs = sum(brs)



# sum the objectives together
JF = (J_nonQSRatio + IOTAS_WEIGHT * J_iotas + MR_WEIGHT * J_major_radius
    + length_penalty
    + cc_penalty
    # TODO: uncomment
    # + cs_penalty
    + curvature_penalty
    + msc_penalty
    + AL_WEIGHT * Jal
    + BR_WEIGHT * Jbrs
    )
print(JF.x.size)





curves_to_vtk(curves, OUT_DIR + "curves_init")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_init_{idx}")

# save these as a backup in case the boozer surface Newton solve fails
res_list = [{'sdofs': boozer_surface.surface.x.copy() , 'iota': boozer_surface.res['iota'], 'G': boozer_surface.res['G'], 'rx0':rz[0].copy(), 'zx0': rz[1].copy()} for rz, boozer_surface in zip(xpoints, boozer_surfaces)]
dat_dict = {'J': JF.J(), 'dJ': JF.dJ().copy()}

def callback(dofs):
    for res, boozer_surface, rz in zip(res_list, boozer_surfaces, xpoints):
        res['sdofs'] = boozer_surface.surface.x.copy()
        res['rx0'] = rz[0].copy()
        res['zx0'] = rz[1].copy()
        res['iota'] =  boozer_surface.res['iota']
        res['G'] = boozer_surface.res['G']
    
    dat_dict['J'] = JF.J()
    dat_dict['dJ'] = JF.dJ().copy()

    outstr = f"J={dat_dict['J']:.1e}, J_nonQSRatio={J_nonQSRatio.J():.2e}, mr={mr.J():.2e} ({J_major_radius.J()*MR_WEIGHT:.1e})"
    iota_string = ", ".join([f"{res['iota']:.3f}" for res in res_list])
    cl_string = ", ".join([f"{J.J():.1f}" for J in Jls])
    kap_string = ", ".join(f"{np.max(c.kappa()):.1f}" for c in base_curves)
    msc_string = ", ".join(f"{J.J():.1f}" for J in Jmscs)
    brs_string = ", ".join(f"{J.J():.1e}" for J in brs)
    xp_string = ", ".join(f"{rz[0][0]:.2e} {rz[1][0]:.2e} " for rz in xpoints)
    outstr += f", iotas=[{iota_string}] ({IOTAS_WEIGHT*J_iotas.J():.1e}), Len=sum([{cl_string}])={sum(J.J() for J in Jls):.1f} ({length_penalty.J():.1e}), ϰ=[{kap_string}] ({curvature_penalty.J():.1e}), ∫ϰ²/L=[{msc_string}] ({msc_penalty.J():.1e})"
    outstr += f", C-C-Sep={Jccdist.shortest_distance():.2f} ({cc_penalty.J():.1e}), C-S-Sep={Jcsdist.shortest_distance():.2f} ({cs_penalty.J():.1e})"
    outstr += f", al var={Jal.J():.2e}"
    outstr += f", brs={brs_string} {BR_WEIGHT*Jbrs.J():.2e}"
    outstr += f", xps={xp_string}"
    outstr += f", ║∇J║={np.linalg.norm(dat_dict['dJ']):.1e}"
    print(outstr)

def fun(dofs):
    JF.x = dofs
    J = JF.J()
    grad = JF.dJ()
    
    axis_success = []
    for res, rz, coils_local, bs_local in zip(res_list, xpoints, coils_list, bs_list):
        curve_stellsym=False
        nfp=2
        order=16
        _, _, success, rx0, zx0 = find_magnetic_axis(bs_local, res['rx0'], res['zx0'], nfp, order, curve_stellsym)
        print(f"axis solve: {success}")
        rz[0] = rx0.copy()
        rz[1] = zx0.copy()
        axis_success.append(success)

    if not np.all([boozer_surface.res['success'] and (not boozer_surface.surface.is_self_intersecting()) for boozer_surface in boozer_surfaces]) or not np.all(axis_success):
        print('failed')
        # failed, so reset back to previous surface and return a large value
        # of the objective.  The purpose is to trigger the line search to reduce
        # the step size.
        J = 1e3
        grad = -dat_dict['dJ']
        for res, boozer_surface in zip(res_list,  boozer_surfaces): 
            boozer_surface.surface.x = res['sdofs']
            boozer_surface.res['iota'] = res['iota']
            boozer_surface.res['G'] = res['G']
    return J, grad


#print("""
#################################################################################
#### Perform a Taylor test ######################################################
#################################################################################
#""")
#f = fun
#dofs = JF.x.copy()
#np.random.seed(1)
#h = np.loadtxt(OUT_DIR+'h.txt')
#J0, dJ0 = f(dofs)
#dJh = sum(dJ0 * h)
#for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
#    J1, _ = f(dofs + 2*eps*h)
#    J2, _ = f(dofs + eps*h)
#    J3, _ = f(dofs - eps*h)
#    J4, _ = f(dofs - 2*eps*h)
#    print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/np.linalg.norm(dJh))
#quit()



print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")
# Number of iterations to perform:
MAXITER = 1000

for restart in range(5):
    dofs = JF.x
    callback(dofs)
    res = minimize(fun, dofs, jac=True, method='L-BFGS-B', options={'maxiter': MAXITER, 'maxcor':20}, tol=1e-15, callback=callback)
    #res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)
    print(res.message)

    curves_to_vtk(curves, OUT_DIR + "curves_opt")
    for idx, boozer_surface in enumerate(boozer_surfaces):
        boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}")
    save([boozer_surfaces, coils_list], OUT_DIR + 'opt.json')

print("End of 2_Intermediate/boozerQA.py")
print("================================")

print("""
################################################################################
### Perform a Taylor test ######################################################
################################################################################
""")
f = fun
dofs = res.x.copy()
np.random.seed(1)
h = np.random.uniform(size=dofs.shape)
J0, dJ0 = f(dofs)
dJh = sum(dJ0 * h)
for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
    J1, _ = f(dofs + 2*eps*h)
    J2, _ = f(dofs + eps*h)
    J3, _ = f(dofs - eps*h)
    J4, _ = f(dofs - 2*eps*h)
    print("err", ((J1*(-1/12) + J2*(8/12) + J3*(-8/12) + J4*(1/12))/eps - dJh)/np.linalg.norm(dJh))
np.savetxt(OUT_DIR+'h.txt', h)

