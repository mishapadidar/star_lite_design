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

def taylor_test(f, dofs, order=6, *args, **kwargs):
    """ Taylor test of f in a random direction of parameter space.

    Args:
        f (function): function handle returning f(x), gradf(x).  f(x) is a scalar.
        x (1d-array): x value at which to do the Taylor test
        order (int, optional): order of the Taylor test approximation, defaults to 6.

    Returns:
        min_rel_error (float): the minimum relative error attained during the Taylor test.  A good
                               rule of thumb is 10 digits of accuracy is acceptable, i.e. 
                               min_rel_error < 1e-10
    """

    np.random.seed(1)
    h = np.random.rand(dofs.size)-0.5
    J0, dJ0 = f(dofs)
    dJ0h = sum(dJ0 * h)
    
    if order == 1:
        shifts = [0, 1]
        weights = [-1, 1]
    elif order == 2:
        shifts = [-1, 1]
        weights = [-0.5, 0.5]
    elif order == 4:
        shifts = [-2, -1, 1, 2]
        weights = [1/12, -2/3, 2/3, -1/12]
    elif order == 6:
        shifts = [-3, -2, -1, 1, 2, 3]
        weights = [-1/60, 3/20, -3/4, 3/4, -3/20, 1/60]
    
    min_rel_err = np.inf
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        xp = dofs + shifts[0]*eps*h
        Jp, dJp = f(xp)
        fd = weights[0] * Jp
        for i in range(1, len(shifts)):
            xp = dofs + shifts[i]*eps*h
            Jp, dJp = f(xp)
            fd += weights[i] * Jp
        err = abs(fd/eps - dJ0h)
        min_rel_err = np.min([err/np.abs(dJ0h), min_rel_err])
        print(f"eps: {eps:.6e}, adjoint deriv: {dJ0h:.6e}, fd deriv: {fd/eps:.6e}, err: {err:.6e}, rel. err:{np.abs(err/dJ0h):.6e}")
    return min_rel_err
