# From Andrew Guiliani email 2025-04-16
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import fsolve
from simsopt.geo import CurveRZFourier

def find_x_point(biotsavart, r0, z0, nfp, order):
    """Find the X-point curve. This function tries to find a closed field line
    of the biotsavart field.

    Args:
        biotsavart (BiotSavart): BiotSavart magnetic field.
        r0 (array): (n, 3) array guessing the major radius coordinate, R, of the X-point.
        z0 (array): (n, 3) array guessing the vertical position, Z, of the X-point.
        nfp (int): number of field periods.
        order (int): order of the Fourier expansion.

    Returns:
        ma_fp (CurveRZFourier): closed curve on one field period.
        ma_ft (CurveRZFourier): closed curve on the full torus.
        ma_success (bool): True if the X-point was found, False otherwise.
    """
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
    #print(norm_res)

    xyz = np.hstack((soln[:n, None]*np.cos(phi), soln[:n, None]*np.sin(phi), soln[n:, None]))
    quadpoints = np.linspace(0, 1/nfp, n, endpoint=False)
    ma_fp = CurveRZFourier(quadpoints, order, nfp, False)
    ma_fp.least_squares_fit(xyz)

    quadpoints = np.linspace(0, nfp, nfp*n, endpoint=False)
    ma_ft = CurveRZFourier(quadpoints, order, nfp, False)
    ma_ft.x = ma_fp.x

    return ma_fp, ma_ft, ma_success

if __name__ == "__main__":
    from simsopt._core import load
    # from star_lite_design.utils.rotate_nfp import rotate_nfp
    design = "A"
    iota_group_idx = 0 # 3 current groups

    # load the boozer surfaces (1 per Current configuration, so 3 total.)
    data = load(f"../designs/design{design}_after_scaled.json")
    bsurfs = data[0] # BoozerSurfaces
    x_point_curves = data[3] # X-point CurveRZFouriers

    # get the boozer surface
    bsurf = bsurfs[iota_group_idx]
    biotsavart = bsurf.biotsavart
    nfp = bsurf.surface.nfp
    x_point_curve = x_point_curves[iota_group_idx] # X-point CurveRZFourier

    # compute the magnetic axis
    xyz = x_point_curve.gamma()
    r0 = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    z0 = xyz[:, 2]
    _, ma, succes = find_x_point(biotsavart, r0, z0, nfp, 16)

    # plot it
    import matplotlib.pyplot as plt
    ax = plt.subplot(111, projection='3d')
    ax.plot(*xyz.T, color='black', linestyle='-.', alpha=0.3, label='actual')
    xyz = ma.gamma()
    ax.plot(*xyz.T, color='red', linestyle='-', alpha=0.6,label='found')
    plt.legend()
    plt.show()

