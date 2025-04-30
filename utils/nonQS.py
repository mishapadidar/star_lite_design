import numpy as np

def nonQS(surface, biotsavart):
    """Compute the NonQuasiSymmetricRatio metric from Simsopt.

    Args:
        surface (Surface): Surface object from Simsopt.
        biotsavart (BiotSavart): BiotSavart object from Simsopt.

    Returns:
        J (float): value of the NonQuasiSymmetricRatio metric.
    """

    biotsavart.set_points(surface.gamma().reshape((-1, 3)))
    axis = 0

    # compute J
    nphi = surface.quadpoints_phi.size
    ntheta = surface.quadpoints_theta.size

    B = biotsavart.B()
    B = B.reshape((nphi, ntheta, 3))
    modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)

    nor = surface.normal()
    dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

    B_QS = np.mean(modB * dS, axis=axis) / np.mean(dS, axis=axis)

    if axis == 0:
        B_QS = B_QS[None, :]
    else:
        B_QS = B_QS[:, None]

    B_nonQS = modB - B_QS
    return np.mean(dS * B_nonQS**2) / np.mean(dS * B_QS**2)