import numpy as np
from simsopt._core import load
from simsopt.geo import plot, BoozerSurface, SurfaceXYZTensorFourier, Volume, boozer_surface_residual, curves_to_vtk
from simsopt.objectives import SquaredFlux
from simsopt.field import BiotSavart, Coil, Current
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from star_lite_design.utils.curvecorrected import CurveCorrected
from star_lite_design.utils.rotate_nfp import rotate_nfp
from star_lite_design.utils.find_x_point import find_x_point
from star_lite_design.utils.nonQS import nonQS
from star_lite_design.utils.stage_2 import stage_2_currents_only
import pandas as pd
import os



"""
This script performs a sensitivity analysis of the boozer surface to the solid body
translations and rotations of the coils. It perturbs the curves along the 5 free directions
and computes the boozer surface for each perturbation. It then computes the metrics
and saves the data.
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
# mpol = bsurf.surface.mpol
# ntor = bsurf.surface.ntor
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

# fix all current, curve and surface dofs
bsurf.fix_all()

""" 
Build the corrected curves: curves with additional degrees of freedom for solid body translation
and rotation. Eventually they will be used in the sensitivity analysis.
"""
# make a biotsavart object for the corrected curves
corrected_curves = [CurveCorrected(c) for c in curves]
corrected_currents = currents
corrected_coils = [Coil(ccurve, current) for ccurve, current in zip(corrected_curves, corrected_currents)]
corrected_biotsavart = BiotSavart(corrected_coils)

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
corrected_surf = SurfaceXYZTensorFourier(
    mpol=temp_surf.mpol, ntor=temp_surf.ntor, stellsym=False, nfp=1,
    quadpoints_phi=phis, quadpoints_theta=thetas)
corrected_surf.unfix_all()
corrected_surf.x = temp_surf.x

# create a boozer surface with the corrected curves
targetlabel = bsurf.surface.volume()
label = Volume(corrected_surf)
options = {'verbose':True, 'newton_maxiter': 20, 'bfgs_maxiter': 1000, 'newton_tol': 1e-11, 'bfgs_tol': 1e-8}
corrected_bsurf = BoozerSurface(corrected_biotsavart, corrected_surf, label=label, targetlabel=targetlabel, constraint_weight=1,
                                options=options)

# # sanity plot
# ax = plt.figure().add_subplot(projection='3d')
# plot([bsurf.surface], alpha=0.3, show=False, ax=ax)
# plot([corrected_surf], alpha=0.3, show=False, ax=ax)
# plt.show()

# set the origins to be the curve centroids
curve_radii = np.zeros(len(corrected_curves))
for ii, c_curve in enumerate(corrected_curves):
    base_curve = c_curve.curve

    # compute the centroid
    g = base_curve.gamma() # (n, 3)
    gp = base_curve.gammadash() # (n, 3)
    dl_dt = np.linalg.norm(gp, axis=-1)
    dt = np.diff(base_curve.quadpoints)[0]
    dl = dl_dt * dt # (n,)
    centroid = np.sum(g * dl[:, None], axis=0) / np.sum(dl)

    # set the origin of rotation
    c_curve.set('origin(x)', centroid[0])
    c_curve.set('origin(y)', centroid[1])
    c_curve.set('origin(z)', centroid[2])

    # get the minor radius
    dist = np.linalg.norm(g - centroid[None, :], axis=-1)
    curve_radii[ii] = np.sum(dist * dl) / np.sum(dl)


""" Compute the original (unperturbed) surface """


# comute boozer surface
res = corrected_bsurf.run_code(iota=iota0, G=G0)

# keep surface dofs;
x0_surf = corrected_surf.x

# keep biotsavart dofs
corrected_biotsavart.unfix_all()
x0_bs = corrected_biotsavart.x
corrected_biotsavart.fix_all()

""" 
For each corrected curve. Perturb the curve along the 5 free directions.
Compute the boozer surface. 
Compute metrics.
save data.
"""


# admissible directions
free_dofs = ['translation(x)', 'translation(y)','alpha', 'beta', 'gamma']


# bounds on the perturbations
max_perturbation = 0.01 # [meters]

# number of perturbations
n_perturbations = 3 # use odd number to sample at 0.0

# storage
suffix = "_stage2"
columns = ['curve_idx', 'dof_name', 'dof_value',
           'iota', 'G',
           'x_point', 'x_point_deviation',
           'solve_status', 'qs_err', 'squared_flux',
           'residual_mse', 'residual_max',
           'is_self_intersecting', 'solver',
           'iota'+suffix, 'G'+suffix,
           'x_point'+suffix, 'x_point_deviation'+suffix,
           'solve_status'+suffix, 'qs_err'+suffix, 'squared_flux'+suffix,
           'residual_mse'+suffix, 'residual_max'+suffix,
           'is_self_intersecting'+suffix, 'solver'+suffix,
           'squared_flux_with_original_surf'+suffix]
df = pd.DataFrame(columns = columns)


# indexes of distinct curves (others are symmetric to these)
if design == "A":
    distinct_curves_idx = [0, 1, 4]
elif design == "B":
    distinct_curves_idx = [0, 2]



for i_curve in distinct_curves_idx:
    c_curve = corrected_curves[i_curve]

    for i_dof, dof in enumerate(free_dofs):
        
        # dof bounds
        if dof in ['translation(x)', 'translation(y)']:
            dof_lb = - max_perturbation
            dof_ub = max_perturbation
        else:
            # set |r * theta| < max_perturbation
            dof_lb = - max_perturbation / curve_radii[i_curve]
            dof_ub = max_perturbation / curve_radii[i_curve]

        # values of the dof to perturb
        dof_values = np.linspace(dof_lb, dof_ub, n_perturbations) # [meters]

        data = {key: [] for key in columns}
        data['curve_idx'] = [i_curve] * n_perturbations
        data['dof_name'] = [dof] * n_perturbations
        data['dof_value'] = dof_values

        for i_val, dof_val in enumerate(dof_values):
            print("")
            print(f"curve {i_curve}, dof {dof}, value {dof_val}")

            # reset coil dofs
            corrected_biotsavart.unfix_all()
            corrected_biotsavart.x = x0_bs
            corrected_biotsavart.fix_all()

            # perturb curve
            c_curve.set(dof, dof_val)

            # reset surface dofs before solve
            corrected_surf.x = x0_surf
            
            # comute boozer surface
            res = corrected_bsurf.run_code(iota=iota0, G=G0)
            if res['success']:
                data['solver'].append('newton')
            else:
                # newton failed, try LBFGS
                print("newton failed, trying LBFGS")
                corrected_surf.x = x0_surf
                corrected_bsurf.need_to_run_code = True
                data['solver'].append('bfgs')
                res = corrected_bsurf.minimize_boozer_penalty_constraints_LBFGS(tol=options['newton_tol'], maxiter=1500, iota=iota0, G=G0, verbose=True)

            # boozer surface data
            data['solve_status'].append(res['success'])
            data['iota'].append(res['iota'])
            data['G'].append(res['G'])

            # get boozer residual
            residuals = boozer_surface_residual(corrected_bsurf.surface, res['iota'], res['G'], corrected_biotsavart)[0]
            data['residual_mse'] = np.mean(residuals**2)
            data['residual_max'] = np.max(np.abs(residuals))

            # compute QS metric
            data['qs_err'].append(np.sqrt(nonQS(corrected_surf, corrected_biotsavart)))
            print("qs_err", data['qs_err'][-1])

            # get x-point position
            _, ma, ma_success = find_x_point(corrected_biotsavart, x_point_r0, x_point_z0, nfp, 10)
            x_point_new = ma.gamma()[0] # phi = 0 X-point
            data['x_point'].append(x_point_new)
            data['x_point_deviation'].append(np.linalg.norm(x_point_new - x_point_xyz[0]))
            print("x_point_deviation", data['x_point_deviation'][-1])

            is_self_intersecting = np.any([corrected_surf.is_self_intersecting(angle) for angle in np.linspace(0, 2*np.pi, 10)])
            data['is_self_intersecting'].append(is_self_intersecting)

            # squared flux
            Jf = SquaredFlux(corrected_surf, corrected_biotsavart)
            data['squared_flux'].append(Jf.J())
            print("squared_flux", data['squared_flux'][-1])

            # visualize
            vizdir = f"./viz/solid_body_analysis/design_{design}/group_{iota_group_idx}"
            os.makedirs(vizdir, exist_ok=True)
            corrected_surf.to_vtk(vizdir + f"/surf_design_{design}_group_{iota_group_idx}_curve_{i_curve}_dof_{dof}_val_{dof_val}")
            curves_to_vtk(corrected_curves,vizdir + f"/curves_design_{design}_group_{iota_group_idx}_curve_{i_curve}_dof_{dof}_val_{dof_val}")
            curves_to_vtk([ma], vizdir + f"/X_point_design_{design}_group_{iota_group_idx}_curve_{i_curve}_dof_{dof}_val_{dof_val}")

            """ Run a stage-2 optimization to correct errorsby varying the coil currents """
            print('stage-2 optimization')

            # fix dofs besides the some currents
            corrected_biotsavart.fix_all()
            for ii in distinct_curves_idx[1:]:
                corrected_currents[ii].unfix_all()

            # set surface to be original surface
            corrected_surf.x = x0_surf

            # run stage-2
            corrected_biotsavart, _ = stage_2_currents_only(corrected_surf, corrected_biotsavart)
            corrected_biotsavart.fix_all()

            # squared flux with original surface
            Jf = SquaredFlux(corrected_surf, corrected_biotsavart)
            data['squared_flux_with_original_surf' + suffix].append(Jf.J())
            print("squared_flux_with_original_surf", data['squared_flux_with_original_surf' + suffix][-1])
            
            # comute boozer surface
            res = corrected_bsurf.run_code(iota=iota0, G=G0)
            if res['success']:
                data['solver' + suffix].append('newton')
            else:
                # newton failed, try LBFGS
                print("newton failed, trying LBFGS")
                corrected_surf.x = x0_surf
                corrected_bsurf.need_to_run_code = True
                data['solver' + suffix].append('bfgs')
                res = corrected_bsurf.minimize_boozer_penalty_constraints_LBFGS(tol=options['newton_tol'], maxiter=1500, iota=iota0, G=G0, verbose=True)

            # boozer surface data
            data['solve_status' + suffix].append(res['success'])
            data['iota' + suffix].append(res['iota'])
            data['G' + suffix].append(res['G'])

            # get boozer residual
            residuals = boozer_surface_residual(corrected_bsurf.surface, res['iota'], res['G'], corrected_biotsavart)[0]
            data['residual_mse' + suffix] = np.mean(residuals**2)
            data['residual_max' + suffix] = np.max(np.abs(residuals))

            # compute QS metric
            data['qs_err' + suffix].append(np.sqrt(nonQS(corrected_surf, corrected_biotsavart)))
            print("qs_err", data['qs_err' + suffix][-1])

            # get x-point position
            _, ma, ma_success = find_x_point(corrected_biotsavart, x_point_r0, x_point_z0, nfp, 10)
            x_point_new = ma.gamma()[0] # phi = 0 X-point
            data['x_point' + suffix].append(x_point_new)
            data['x_point_deviation' + suffix].append(np.linalg.norm(x_point_new - x_point_xyz[0]))
            print("x_point_deviation", data['x_point_deviation' + suffix][-1])

            is_self_intersecting = np.any([corrected_surf.is_self_intersecting(angle) for angle in np.linspace(0, 2*np.pi, 10)])
            data['is_self_intersecting' + suffix].append(is_self_intersecting)
            
            # squared flux
            Jf = SquaredFlux(corrected_surf, corrected_biotsavart)
            data['squared_flux' + suffix].append(Jf.J())


        df1 = pd.DataFrame(data)
        df = pd.concat([df, df1], ignore_index=True)

        # reset dofs
        c_curve.set(dof, 0.0)

    df = df.reset_index(drop=True)

    print(df)

    # save data
    # Create output directory if it doesn't exist
    output_dir = './output/solid_body_analysis'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_dir + f'/solid_body_error_analysis_design_{design}_group_{iota_group_idx}.csv', index=False)




