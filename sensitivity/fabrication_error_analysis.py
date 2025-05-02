import numpy as np
from simsopt._core import load
from simsopt.geo import plot, BoozerSurface, SurfaceXYZTensorFourier, Volume, boozer_surface_residual, curves_to_vtk
from simsopt.field import BiotSavart, Coil, Current
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from star_lite_design.utils.curvecorrected import CurveCorrected
from star_lite_design.utils.rotate_nfp import rotate_nfp
from star_lite_design.utils.find_x_point import find_x_point
from star_lite_design.utils.nonQS import nonQS
import pandas as pd
import os
from randomgen import PCG64
from datetime import datetime
from simsopt.geo import (GaussianSampler, CurvePerturbed, PerturbationSample)



"""
This script performs a sensitivity analysis of the boozer surface to fabrication errors.
It perturbs the coils by a gaussian process, computes the boozer surface, performance metrics,
and then save the results.
"""

design = "B"
iota_group_idx = 0 # 3 current groups


# load the boozer surfaces (1 per Current configuration, so 3 total.)
data = load(f"../designs/design{design}_after_scaled.json")
bsurfs = data[0] # BoozerSurfaces
iota_Gs = data[1] # (iota, G) pairs
axis_curves = data[2] # magnetic axis CurveRZFouriers
x_point_curves = data[3] # X-point CurveRZFouriers

# get the boozer surface
bsurf = bsurfs[iota_group_idx]
mpol = bsurf.surface.mpol
ntor = bsurf.surface.ntor
nfp = bsurf.surface.nfp
iota0 = iota_Gs[iota_group_idx][0] # iota
G0 = iota_Gs[iota_group_idx][1] # G
x_point_curve = x_point_curves[iota_group_idx] # X-point CurveRZFourier

# guess of the x point location
x_point_xyz = x_point_curve.gamma()
x_point_r0 = np.sqrt(x_point_xyz[:, 0]**2 + x_point_xyz[:, 1]**2)
x_point_z0 = x_point_xyz[:, 2]

# get the coils
biotsavart = bsurfs[iota_group_idx].biotsavart
coils = biotsavart.coils
currents = [c.current for c in coils]
curves = [c.curve for c in coils]

 # distinct curves (others are symmetric to these)
if design == "A":
    base_curves = [curves[0], curves[1], curves[4]]
    base_currents = [currents[0], currents[1], currents[4]]
elif design == "B":
    base_curves = [curves[0], curves[2]]
    base_currents = [currents[0], currents[2]]


# fix all current
bsurf.unfix_all()
for c in currents:
    c.fix_all()

""" 
Build a non stellsym boozer surface with one field period.
"""

# create a non-stellarator symmetric, nfp = 1, surface
nphi = nfp * len(bsurf.surface.quadpoints_phi)
ntheta = len(bsurf.surface.quadpoints_theta)
phis = np.linspace(0, 1, nphi, endpoint=False)
thetas = np.linspace(0, 1, ntheta, endpoint=False)

mpol = ntor = 8
temp_surf = SurfaceXYZTensorFourier(
    mpol=mpol, ntor=ntor, stellsym=False, nfp=1,
    quadpoints_phi=phis, quadpoints_theta=thetas)
g = bsurf.surface.gamma()
nphi, ntheta, _ = g.shape
gamma = np.zeros((nfp * nphi, ntheta, 3))
for ii in range(nfp):
    g = rotate_nfp(g, 1, nfp)
    gamma[ii * nphi : (ii+1) * nphi] = g
temp_surf.least_squares_fit(gamma)
temp_surf.unfix_all()

# get a new surface with the right number of quadpoints
phis = np.linspace(0, 1, 2*temp_surf.mpol + 1, endpoint=False)
thetas = np.linspace(0, 1, 2*temp_surf.ntor + 1, endpoint=False)
surf_ft = SurfaceXYZTensorFourier(
    mpol=temp_surf.mpol, ntor=temp_surf.ntor, stellsym=False, nfp=1,
    quadpoints_phi=phis, quadpoints_theta=thetas)
surf_ft.unfix_all()
surf_ft.x = temp_surf.x

# create a boozer surface with the corrected curves
targetlabel = bsurf.surface.volume()
label = Volume(surf_ft)
options = {'verbose':True, 'newton_maxiter': 20, 'bfgs_maxiter': 1200, 'newton_tol': 1e-11, 'bfgs_tol': 1e-8}
bsurf_ft = BoozerSurface(biotsavart, surf_ft, label=label, targetlabel=targetlabel, constraint_weight=1,
                                options=options)
# fit the surface
bsurf_ft.run_code(iota=iota0, G=G0)
# save the dofs
x0_surf_ft = surf_ft.x

# # sanity plot
# ax = plt.figure().add_subplot(projection='3d')
# plot([bsurf_ft.surface], alpha=0.3, show=False, ax=ax)
# plot([surf_ft], alpha=0.3, show=False, ax=ax)
# plt.show()

# visualize
vizdir = f"./viz/fabrication_error_analysis/design_{design}/group_{iota_group_idx}"
os.makedirs(vizdir, exist_ok=True)
curves_to_vtk(curves,vizdir + f"/original_curves_design_{design}_group_{iota_group_idx}")

""" 
Sensitivity analysis of the boozer surface to fabrication errors.
    1. Perturb the curves by GPs.
    2. Compute the boozer surface. 
    3. Compute metrics.
    4. save data.
"""

# sampler parameters
n_samples = 16
sigmas = [0.001, 0.005, 0.01] # amplitude [meters]
lengthscale = 0.03 # [0.03,0.1] is reasonable

# random seed
seed = datetime.now().second + 60 * datetime.now().minute

# storage
columns = ['sample_idx', 'iota', 'G',
           'x_point', 'x_point_deviation',
           'solve_status', 'qs_err',
           'residual_mse', 'residual_max',
           'is_self_intersecting', 'solver', 'sigma', 'lengthscale']
data = {key: [] for key in columns}

for sigma in sigmas:
    print("")
    print(f"perturbation sigma: {sigma}")
    print(f"perturbation lengthscale: {lengthscale}")
    # sampler
    rg = np.random.Generator(PCG64(seed, inc=0))
    sampler = GaussianSampler(curves[0].quadpoints, sigma, lengthscale, n_derivs=1)

    for i_sample in range(n_samples):
        print("")
        print(f"perturbation {i_sample+1}/{n_samples}")

        data['sample_idx'].append(i_sample)
        data['sigma'].append(sigma)
        data['lengthscale'].append(lengthscale)

        # reset coil and surface dofs
        surf_ft.x = x0_surf_ft

        # TODO: do systematic perturbations
        # # first add the 'systematic' error
        # base_curves_perturbed = [CurvePerturbed(c, PerturbationSample(sampler, randomgen=rg)) for c in base_curves]
        # coils = coils_via_symmetries(base_curves_perturbed, base_currents, surf_ft.nfp, True)

        # now add the 'statistical' error.
        coils_pert = [Coil(CurvePerturbed(c.curve, PerturbationSample(sampler, randomgen=rg)), c.current) for c in coils]
        curves_pert = [c.curve for c in coils_pert]
        biotsavart_pert = BiotSavart(coils_pert)

        bsurf_pert = BoozerSurface(biotsavart_pert, surf_ft, label=label, targetlabel=targetlabel, constraint_weight=1,
                                    options=options)

        # compute boozer surface
        res = bsurf_pert.run_code(iota=iota0, G=G0)
        if res['success']:
            data['solver'].append('bfgs+newton')
        else:
            # newton failed, try LBFGS
            print("newton failed, trying LBFGS")
            surf_ft.x = x0_surf_ft
            bsurf_pert.need_to_run_code = True
            data['solver'].append('bfgs')
            res = bsurf_pert.minimize_boozer_penalty_constraints_LBFGS(tol=options['newton_tol'], maxiter=1500, iota=iota0, G=G0, verbose=True)


        # boozer surface data
        data['solve_status'].append(res['success'])
        data['iota'].append(res['iota'])
        data['G'].append(res['G'])

        # get boozer residual
        residuals = boozer_surface_residual(bsurf_pert.surface, res['iota'], res['G'], biotsavart_pert)[0]
        data['residual_mse'] = np.mean(residuals**2)
        data['residual_max'] = np.max(np.abs(residuals))

        # compute QS metric
        data['qs_err'].append(np.sqrt(nonQS(surf_ft, biotsavart_pert)))

        # get x-point position
        _, ma, ma_success = find_x_point(biotsavart_pert, x_point_r0, x_point_z0, nfp, 10)
        x_point_new = ma.gamma()[0] # phi = 0 X-point
        data['x_point'].append(x_point_new)
        data['x_point_deviation'].append(np.linalg.norm(x_point_new - x_point_xyz[0]))

        is_self_intersecting = np.any([surf_ft.is_self_intersecting(angle) for angle in np.linspace(0, 2*np.pi, 10)])
        data['is_self_intersecting'].append(is_self_intersecting)

    # visualize last perturbation
    curves_to_vtk(curves_pert,vizdir + f"/perturbed_curves_sample_{i_sample}_sigma_{sigma}_lengthscale_{lengthscale}")
    # surf_ft.to_vtk(vizdir + f"/surf_design_{design}_group_{iota_group_idx}_curve_{i_curve}_dof_{dof}_val_{dof_val}")
    # curves_to_vtk([ma], vizdir + f"/X_point_design_{design}_group_{iota_group_idx}_curve_{i_curve}_dof_{dof}_val_{dof_val}")

    df = pd.DataFrame(data, columns = columns)
    df = df.reset_index(drop=True)

    print(df)

    # Create output directory if it doesn't exist
    output_dir = './output/fabrication_error_analysis'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = output_dir + f'/fabrication_error_analysis_{design}_group_{iota_group_idx}.csv'
    if os.path.exists(csv_path):
        # load existing data and append new data
        existing_df = pd.read_csv(csv_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    # save data to csv
    df.to_csv(csv_path, index=False)


