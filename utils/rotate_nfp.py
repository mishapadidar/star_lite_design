import numpy as np
def rotate_nfp(X, jj, nfp):
    """Rotate a scalar or vector field X on a flux surface by jj field periods. Given say the normal 
    vectors X to a flux surface on one field period, rotate_nfp(X, jj, nfp) will yield the normal vectors
    on the other field periods.

    Args:
        X (array): 3D array, shape (nphi, ntheta, m) of values on a flux surface. m = 3 for vector
            quantites such as normal vectors, and m = 1 for scalar fields such as sqrt{g}.
        jj (int): number of field period rotations. 0 for no rotation, 1 for rotation by 2pi/nfp, etc.
        nfp (int): number of field periods

    Returns:
        X: 3D array, shape (nphi, ntheta, m) of rotated values.
    """
    if jj == 0:
        # just to not mess with signs etc
        return X
    
    angle = np.array(2 * np.pi * jj / nfp)
    Q = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]]
                    )
    return np.einsum('ij,klj->kli', Q, X)