import numpy as np

def finite_difference(f, x, eps=1e-6, *args, **kwargs):
    """Approximate jacobian with central difference.

    Args:
        f (function): function to differentiate, can be scalar valued or 1d-array
            valued.
        x (1d-array): input to f(x) at which to take the gradient.
        eps (float, optional): finite difference step size. Defaults to 1e-6.

    Returns:
        array: Jacobian of f at x. If f returns a scalar, the output will be a 1d-array. Otherwise,
            the output will be a 2d-array with shape (len(f(x)), len(x)).
    """
    jac_est = []
    for i in range(len(x)):
        x[i] += eps
        fx = f(x, *args, **kwargs)
        x[i] -= 2*eps
        fy = f(x, *args, **kwargs)
        x[i] += eps
        jac_est.append((fx-fy)/(2*eps))
    return np.array(jac_est).T