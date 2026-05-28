import numpy as np
from scipy.linalg import lu

from simsopt._core import Optimizable
from simsopt.geo import CurveLength, CurveXYZFourierSymmetries

__all__ = ['SingularPeriodicFieldLine']

import jax
import jax.numpy as jnp
from jax import jacfwd
from functools import partial

# mu_0 / (4 pi)  in SI units (T·m/A)
_MU0_4PI = 1.0e-7
# Rescale auxiliary currents so I=1 here matches simsopt's
# ScaledCurrent(Current(1.0), 1e7/(4*pi)) convention.
_CURRENT_SCALE = 1.0e7 / (4.0 * np.pi)


def _B_single_circle(p, r, z, I, n_quad):
    """
    Magnetic field at a single point from one planar circular coil.

    p       : (3,)  field point
    r, z, I : scalars  radius, height, current
    n_quad  : int (static)  number of quadrature nodes around the loop

    Coil parameterization (lies in plane Z = z, centered on z-axis, radius r):
        r'(phi) = (r cos phi,  r sin phi,  z),     phi in [0, 2 pi)
        dl/dphi = (-r sin phi, r cos phi,  0  )

    Trapezoidal rule on a uniform phi-grid -- spectrally accurate for
    this smooth periodic integrand.
    """
    I = I * _CURRENT_SCALE   # match simsopt's ScaledCurrent(Current(1.0), 1e7/(4*pi)) convention.

    phi = jnp.linspace(0.0, 2.0 * jnp.pi, n_quad, endpoint=False)
    dphi = 2.0 * jnp.pi / n_quad

    cos_p = jnp.cos(phi)
    sin_p = jnp.sin(phi)

    Rx = p[0] - r * cos_p
    Ry = p[1] - r * sin_p
    Rz = p[2] - z

    inv_R3 = (Rx * Rx + Ry * Ry + Rz * Rz) ** (-1.5)

    dlx = -r * sin_p
    dly =  r * cos_p

    cx =  dly * Rz
    cy = -dlx * Rz
    cz =  dlx * Ry - dly * Rx

    pref = _MU0_4PI * I * dphi
    Bx = pref * jnp.sum(cx * inv_R3)
    By = pref * jnp.sum(cy * inv_R3)
    Bz = pref * jnp.sum(cz * inv_R3)

    return jnp.stack([Bx, By, Bz])


def _B_single_circlesN(p, mu, n_quad):
    """
    Field from N planar circular coils in the XY plane at height Z.

    mu : (2N+1,) = (I1, ..., IN, r1, ..., rN, Z)
    """
    N = (mu.shape[0] - 1) // 2
    Z = mu[-1]
    B = jnp.zeros(3)
    for k in range(N):
        B = B + _B_single_circle(p, mu[N + k], Z, mu[k], n_quad)
    return B


def _B_double_circlesN(p, mu, n_quad):
    """
    N circular coils plus their stellarator-symmetric partners.

    A CCW circle at (r, +Z) carrying current +I is partnered with a CCW
    circle at (r, -Z) carrying the *same* current +I. The total field then
    obeys the standard stellarator-symmetry condition
        B_x(x, -y, -z) = -B_x(x, y, z)   (odd)
        B_y(x, -y, -z) = +B_y(x, y, z)   (even)
        B_z(x, -y, -z) = +B_z(x, y, z)   (even)
    i.e. in cylindrical (R, phi, Z) coordinates, B_R is odd and B_phi, B_Z
    are even under (R, phi, Z) -> (R, -phi, -Z).

    mu_image = (I_1..I_N, r_1..r_N, Z) -> (I_1..I_N, r_1..r_N, -Z).
    """
    N = (mu.shape[0] - 1) // 2
    mu_image = jnp.concatenate([mu[:N], mu[N:2 * N], -mu[-1:]])
    return _B_single_circlesN(p, mu, n_quad) + _B_single_circlesN(p, mu_image, n_quad)


# -----------------------------------------------------------------------------
# Multi-point field and its derivatives. Each is just vmap of the single-point
# primitive, optionally composed with jacfwd.
# -----------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("n_quad", "stellsym"))
def _B_aux(pts, mu, n_quad=128, stellsym=True):
    """
    pts : (Npts, 3)
    mu  : (2N+1,) = (I1..IN, r1..rN, Z)
    -> B : (Npts, 3)
    """
    kernel = _B_double_circlesN if stellsym else _B_single_circlesN
    return jax.vmap(kernel, in_axes=(0, None, None))(pts, mu, n_quad)


@partial(jax.jit, static_argnames=("n_quad", "stellsym"))
def _dB_aux_by_dmu(pts, mu, n_quad=128, stellsym=True):
    """
    Jacobian of B wrt the parameter vector mu.

    Returns
    -------
    J : (Npts, 3, len(mu))
        J[i, j, k] = dB_j / dmu_k  at point i.
    """
    kernel = _B_double_circlesN if stellsym else _B_single_circlesN
    grad_single = jax.jacfwd(kernel, argnums=1)
    return jax.vmap(grad_single, in_axes=(0, None, None))(pts, mu, n_quad)


@partial(jax.jit, static_argnames=("n_quad", "stellsym"))
def _dB_aux_by_dX(pts, mu, n_quad=128, stellsym=True):
    """
    Spatial gradient of B in simsopt convention.

    Returns
    -------
    G : (Npts, 3, 3)
        G[i, k, j] = dB_j / dx_k   at point i.
        (First free index = spatial direction, second = B component.)

    In any current-free region this satisfies:
        trace(G) = div B          = 0
        G - G^T  = -[curl B]_x    = 0    (since curl B = mu_0 J = 0 off the wire)
    so G should be symmetric and traceless -- handy as a numerical check.
    """
    kernel = _B_double_circlesN if stellsym else _B_single_circlesN
    grad_single = jax.jacfwd(kernel, argnums=0)   # (3,) -> (3, 3)  [j_Bcomp, k_spatial]
    G = jax.vmap(grad_single, in_axes=(0, None, None))(pts, mu, n_quad)
    return G.transpose(0, 2, 1)  # -> [i, k_spatial, j_Bcomp]

@partial(jax.jit, static_argnames=("n_quad", "stellsym"))
def _d2B_aux_by_dXdX(pts, mu, n_quad=128, stellsym=True):
    """
    Spatial Hessian of B in simsopt convention.

    Returns
    -------
    H : (Npts, 3, 3, 3)
        H[i, k1, k2, j] = d^2 B_j / (dx_k1 dx_k2)  at point i.
        (First two free indices = spatial directions, last = B component.)

    Symmetries:
      * H[..., k1, k2, j] = H[..., k2, k1, j]   (Clairaut, holds everywhere)
      * Off the wire B = -grad phi  =>  H is fully symmetric in (k1, k2, j).
      * div B = 0  =>  sum_j H[..., j, k, j] = 0  for every k.
    """
    kernel = _B_double_circlesN if stellsym else _B_single_circlesN
    hess_single = jax.jacfwd(jax.jacfwd(kernel, argnums=0), argnums=0)
    H = jax.vmap(hess_single, in_axes=(0, None, None))(pts, mu, n_quad)
    return H.transpose(0, 2, 3, 1)  # [i, j_Bcomp, k1, k2] -> [i, k1, k2, j_Bcomp]


@partial(jax.jit, static_argnames=("n_quad", "stellsym"))
def _dgradB_aux_by_dmu(pts, mu, n_quad=128, stellsym=True):
    """
    Mixed second derivative wrt spatial position and parameters, in simsopt convention.

    Returns
    -------
    H : (Npts, 3, 3, len(mu))
        H[i, k, j, l] = d^2 B_j / (dx_k dmu_l)   at point i.
        (First free index = spatial direction, second = B component, third = mu index.)

    Off-coil identities (current-free), useful as numerical checks:
        sum_j H[i, j, j, l]      = 0         (d/dmu of div B  = 0)
        H[i, k, j, l] - H[i, j, k, l] = 0    (d/dmu of curl B = 0)
    """
    kernel = _B_double_circlesN if stellsym else _B_single_circlesN
    inner = jax.jacfwd(kernel, argnums=0)         # (3,) -> (3, 3)  [j_Bcomp, k_spatial]
    outer = jax.jacfwd(inner,  argnums=1)         # -> (3, 3, len(mu))  [j_Bcomp, k_spatial, l_mu]
    H = jax.vmap(outer, in_axes=(0, None, None))(pts, mu, n_quad)
    return H.transpose(0, 2, 1, 3)  # -> [i, k_spatial, j_Bcomp, l_mu]

def cheb(Npts, a, b):
    N = Npts - 1
    assert N >= 1
    alt = (-np.ones(N + 1)) ** np.arange(N + 1)
    x = np.cos(np.pi * np.linspace(0, 1, N + 1))
    c = np.array([2] + [1] * (N - 1) + [2]) * alt
    X = np.outer(x, np.ones(N + 1))
    dX = X - X.T
    D = np.outer(c, np.array([1] * (N + 1)) / c) / (dX + np.identity(N + 1))
    D = D - np.diag(np.sum(D, axis=1))

    n = np.arange(0, N // 2 + 1)[:, None]
    k = np.arange(0, N // 2 + 1)[None, :]
    DD = 2 * np.cos(2 * np.pi * n * k / N) / N
    DD[0, :] *= 0.5
    d = np.concatenate(([1.0], 2.0 / (1.0 - np.square(np.arange(2, N + 1, 2)))))
    w = DD @ d
    w = np.concatenate((w, np.flip(w[:-1])))

    x = 0.5 * (x + 1) * (b - a) + a
    D = D / (0.5 * (b - a))
    w = 0.5 * w * (b - a)

    x = x[::-1]
    w = w[::-1]
    D = D[::-1, :]
    D = D[:, ::-1]
    return D, x, w


def A_pure(B, gradB):
    # gradB[i, k, j] = dB_j/dx_k  (simsopt convention: k=spatial, j=B-comp)
    modB = jnp.linalg.norm(B, axis=-1)
    dmodB = jnp.sum(B[:, None, :] * gradB, axis=-1) / modB[:, None]   # dmodB[i,k] = d|B|/dx_k
    A = (gradB.transpose(0, 2, 1) * modB[:, None, None] - B[:, :, None] * dmodB[:, None, :]) / modB[:, None, None] ** 2
    return A


def tangent_map_residual_pure(T, B, gradB, L, ic, D):
    A = A_pure(B, gradB)
    AT = jnp.einsum('ijk,ik->ij', A, T)
    Tprime = jnp.matmul(D, T)
    residual = Tprime / L - AT
    ic0 = T[0] - ic
    return jnp.concatenate((ic0[None, :], residual[1:]), axis=0)


def tangent_map_pure(B, gradB, L, D):
    N = B.shape[0]
    T = jnp.zeros_like(B)
    b1 = -tangent_map_residual_pure(T, B, gradB, L, jnp.array([1.0, 0.0, 0.0]), D).ravel()
    b2 = -tangent_map_residual_pure(T, B, gradB, L, jnp.array([0.0, 1.0, 0.0]), D).ravel()
    b3 = -tangent_map_residual_pure(T, B, gradB, L, jnp.array([0.0, 0.0, 1.0]), D).ravel()
    A = jacfwd(tangent_map_residual_pure, argnums=0)(T, B, gradB, L, jnp.array([0.0, 0.0, 0.0]), D).reshape((3 * N, 3 * N))
    T1 = jnp.linalg.solve(A, b1).reshape((N, 3))
    T2 = jnp.linalg.solve(A, b2).reshape((N, 3))
    T3 = jnp.linalg.solve(A, b3).reshape((N, 3))
    return jnp.concatenate((T1[:, :, None], T2[:, :, None], T3[:, :, None]), axis=-1)


def monodromy_pure(B, gradB, L, gamma, D):
    tangent = jnp.matmul(D, gamma)
    fT = tangent / jnp.linalg.norm(tangent, axis=-1)[:, None]
    normal = jnp.matmul(D, fT)
    fN = normal / jnp.linalg.norm(normal, axis=-1)[:, None]
    binormal = jnp.cross(fT, fN)
    fB = binormal / jnp.linalg.norm(binormal, axis=-1)[:, None]
    NB = jnp.concatenate((fN[:, :, None], fB[:, :, None]), axis=-1)
    NB_t = jnp.concatenate((fN[:, None, :], fB[:, None, :]), axis=-2)
    M = tangent_map_pure(B, gradB, L, D)
    return jnp.matmul(NB_t, jnp.matmul(M, NB[0]))


def monodromy_matrix_pure(B, gradB, L, gamma, D):
    return monodromy_pure(B, gradB, L, gamma, D)[-1]

def singular_field_line_residual(curve, curve_tm, length, field, mu, monodromy_fns,
                                 stellsym=True, monodromy_constraint='identity'):
    
    pts = curve.gamma()
    dpts_dcurve = curve.dgamma_by_dcoeff()
    
    field.set_points(pts.reshape((-1, 3)))
    
    Bcoils = field.B()
    dBcoils_by_dX = field.dB_by_dX()
    d2Bcoils_by_dXdX = field.d2B_by_dXdX()
    
    B_aux = _B_aux(pts, mu, stellsym=stellsym)
    dB_aux_by_dX = _dB_aux_by_dX(pts, mu, stellsym=stellsym)
    d2B_aux_by_dXdX = _d2B_aux_by_dXdX(pts, mu, stellsym=stellsym)

    dB_aux_by_dmu = _dB_aux_by_dmu(pts, mu, stellsym=stellsym)
    dgradB_aux_by_dmu = _dgradB_aux_by_dmu(pts, mu, stellsym=stellsym)

    B = Bcoils + B_aux
    dB_by_dX = dBcoils_by_dX + dB_aux_by_dX
    d2B_by_dXdX = d2Bcoils_by_dXdX + d2B_aux_by_dXdX

    modB = np.linalg.norm(B, axis=1) 
    res = curve.gammadash() / length - B / modB[:, None]
    res = res.flatten()
    if not curve.stellsym:
        res_y = curve.gamma()[0, 1]
        dres_y = np.concatenate([curve.dgamma_by_dcoeff()[0, 1, :], [0], np.zeros(len(mu))])  # ADDED: extend the label equation Jacobian by the new mu columns.

    dres1_dcoeff = curve.dgammadash_by_dcoeff() / length

    idx = np.arange(3)
    diag = np.zeros((pts.shape[0], 3, 3))
    diag[:, idx, idx] = 1 / modB[:, None]
    dres2_dB = -B[:, None, :] * B[:, :, None] / modB[:, None, None] ** 3 + diag  # ADDED: derivative of B/|B| for the total field.

    dB_dcurve  = np.einsum('ikl,ikm->ilm', dB_by_dX,  dpts_dcurve, optimize=True)
    dgradB_dcurve  = np.einsum('ijkl,ikm->ijlm', d2B_by_dXdX, dpts_dcurve, optimize=True)
    
    dres2_dcoeff = np.einsum('ikl,ikm->ilm', dres2_dB, dB_dcurve, optimize=True)

    dres_del = -curve.gammadash().reshape((-1, 1)) / length ** 2
    dres_dmu = -np.einsum('ikl,ikm->ilm', dres2_dB, dB_aux_by_dmu, optimize=True)
    dres_dcoeff = dres1_dcoeff - dres2_dcoeff
    dres = np.concatenate((dres_dcoeff.reshape((res.size, -1)),\
                           dres_del, 
                           dres_dmu.reshape((res.size, -1))), axis=-1)

    if not curve.stellsym:
        res = np.concatenate((res, [res_y]))
        dres = np.concatenate((dres, dres_y[None, :]), axis=0)


    pts = curve_tm.gamma()
    dpts_dcurve = curve_tm.dgamma_by_dcoeff()
    
    field.set_points(pts.reshape((-1, 3)))
    
    Bcoils = field.B()
    dBcoils_by_dX = field.dB_by_dX()
    d2Bcoils_by_dXdX = field.d2B_by_dXdX()
    
    B_aux = _B_aux(pts, mu, stellsym=stellsym)
    dB_aux_by_dX = _dB_aux_by_dX(pts, mu, stellsym=stellsym)
    d2B_aux_by_dXdX = _d2B_aux_by_dXdX(pts, mu, stellsym=stellsym)

    dB_aux_by_dmu = _dB_aux_by_dmu(pts, mu, stellsym=stellsym)
    dgradB_aux_by_dmu = _dgradB_aux_by_dmu(pts, mu, stellsym=stellsym)

    B = Bcoils + B_aux
    dB_by_dX = dBcoils_by_dX + dB_aux_by_dX
    d2B_by_dXdX = d2Bcoils_by_dXdX + d2B_aux_by_dXdX
    
    dB_dcurve  = np.einsum('ikl,ikm->ilm', dB_by_dX,  dpts_dcurve, optimize=True)
    dgradB_dcurve  = np.einsum('ijkl,ikm->ijlm', d2B_by_dXdX, dpts_dcurve, optimize=True)


    M = monodromy_fns['jax'](B, dB_by_dX, length, pts)

    dM_dB = monodromy_fns['dB'](B, dB_by_dX, length, pts).reshape((4,) + B.shape)
    dM_dgradB = monodromy_fns['dgradB'](B, dB_by_dX, length, pts).reshape((4,) + dB_by_dX.shape)
    dM_dL = monodromy_fns['dL'](B, dB_by_dX, length, pts).reshape(4, 1)
    dM_dgamma_partial = monodromy_fns['dgamma'](B, dB_by_dX, length, pts).reshape((4,) + pts.shape)

    dM_dcurve = np.einsum('aij,ijm->am', dM_dB, dB_dcurve, optimize=True)
    dM_dcurve += np.einsum('aijk,ijkm->am', dM_dgradB, dgradB_dcurve, optimize=True)
    dM_dcurve += np.einsum('aik,ikm->am', dM_dgamma_partial, dpts_dcurve, optimize=True)

    dM_dmu = np.einsum('aij,ijm->am', dM_dB, dB_aux_by_dmu, optimize=True)
    dM_dmu += np.einsum('aijk,ijkm->am', dM_dgradB, dgradB_aux_by_dmu, optimize=True)

    # Rows of J_mon correspond to M[0,0], M[0,1], M[1,0], M[1,1] (C-order flatten of M).
    J_mon = np.concatenate((dM_dcurve, dM_dL, dM_dmu), axis=1)

    if monodromy_constraint == 'trace':
        # Single equation: tr(M) - 2 = M[0,0] + M[1,1] - 2.
        r_mon = np.array([M[0, 0] + M[1, 1] - 2.0])
        J_mon = J_mon[0:1] + J_mon[3:4]
    elif monodromy_constraint == 'identity':
        # Four equations: M - I = 0 (one is redundant since det(M)=1; caller drops it).
        r_mon = (M - np.eye(2)).reshape(4)
    else:
        raise ValueError(f"Unknown monodromy_constraint {monodromy_constraint!r}; must be 'identity' or 'trace'.")

    r = np.concatenate((res, r_mon))
    J = np.vstack((dres, J_mon))
    return r, J, M


class SingularPeriodicFieldLine(Optimizable):
    def __init__(self, biotsavart, curve, options=None, stellsym_aux=True):
        super().__init__(depends_on=[biotsavart])
        self.biotsavart = biotsavart
        self.curve = curve
        self.need_to_run_code = True

        if options is None:
            options = {}
        if 'verbose' not in options:
            options['verbose'] = False
        if 'newton_tol' not in options:
            options['newton_tol'] = 1e-13
        if 'newton_maxiter' not in options:
            options['newton_maxiter'] = 40
        if 'use_lstsq' not in options:
            options['use_lstsq'] = False
        if 'monodromy_constraint' not in options:
            options['monodromy_constraint'] = 'identity'
        self.options = options
        self.stellsym_aux = stellsym_aux

        nfp = curve.nfp  # ADDED: match the tangent-map toroidal period.
        N = 6 * curve.order + 1  # ADDED: use the same Chebyshev resolution as TangentMap.
        D, xh, wh = cheb(N, 0, 1.0 / nfp)  # ADDED: tangent map is evaluated on the Chebyshev grid.
        self.D = D  # ADDED: store the Chebyshev differentiation matrix.
        self.xh = xh  # ADDED: store the Chebyshev nodes.
        self.wh = wh  # ADDED: keep the quadrature weights for consistency with TangentMap.
        self.curve_tm = CurveXYZFourierSymmetries(self.xh, curve.order, curve.nfp, curve.stellsym, ntor=curve.ntor, dofs=curve.dofs)  # ADDED: monodromy is evaluated on a Chebyshev-grid copy of the curve.

        self.monodromy_matrix_jax = lambda B, gradB, L, gamma: monodromy_matrix_pure(B, gradB, L, gamma, self.D)
        self.monodromy_matrix_dB = lambda B, gradB, L, gamma: jacfwd(self.monodromy_matrix_jax, argnums=0)(B, gradB, L, gamma)
        self.monodromy_matrix_dgradB = lambda B, gradB, L, gamma: jacfwd(self.monodromy_matrix_jax, argnums=1)(B, gradB, L, gamma)
        self.monodromy_matrix_dL = lambda B, gradB, L, gamma: jacfwd(self.monodromy_matrix_jax, argnums=2)(B, gradB, L, gamma)
        self.monodromy_matrix_dgamma = lambda B, gradB, L, gamma: jacfwd(self.monodromy_matrix_jax, argnums=3)(B, gradB, L, gamma)
        self.monodromy_fns = {
            'jax': self.monodromy_matrix_jax,
            'dB': self.monodromy_matrix_dB,
            'dgradB': self.monodromy_matrix_dgradB,
            'dL': self.monodromy_matrix_dL,
            'dgamma': self.monodromy_matrix_dgamma,
        }

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def run_code(self, length, mu=None, fixed_mu=None):
        if not self.need_to_run_code:
            return self.res
        if mu is None:
            mu = np.zeros(4)
        return self.solve_residual_equation_exactly_newton(length=length, mu=mu, fixed_mu=fixed_mu, tol=self.options['newton_tol'], maxiter=self.options['newton_maxiter'], verbose=self.options['verbose'])

    def get_stellsym_mask(self, tail=4):
        # tail = 4 for the row mask (4 monodromy equations); tail = len(mu) for the col mask.
        order = self.curve.order
        stellsym = self.curve.stellsym
        if not stellsym:
            mask = np.ones((2 * order + 1) * 3 + 1, dtype=bool)
            return np.concatenate((mask, np.ones(tail, dtype=bool)))

        mask = np.ones((2 * order + 1, 3), dtype=bool)
        mask[0, 0] = False
        mask[order + 1:, :] = False
        mask = mask.flatten()
        return np.concatenate((mask, np.ones(tail, dtype=bool)))

    def _format_mu(self, mu, col_mask):
        """Pretty-print mu: bold names, '(fix)' suffix when masked out.
        Currents are scaled by _CURRENT_SCALE so values shown are in physical Amperes."""
        BOLD, RST = '\033[1m', '\033[0m'
        nmu = len(mu)
        N = (nmu - 1) // 2
        names = [f'I{k+1}' for k in range(N)] + [f'r{k+1}' for k in range(N)] + ['z']
        parts = []
        for k, (name, val) in enumerate(zip(names, mu)):
            is_fixed = not bool(col_mask[-nmu + k])
            if k < N:
                token = f'{BOLD}{name}{RST}={val * _CURRENT_SCALE:+.4e}'
            else:
                token = f'{BOLD}{name}{RST}={val:+.4f}'
            if is_fixed:
                token += ' (fix)'
            parts.append(token)
        return '  '.join(parts)

    def _print_iter(self, i, b, Jm, M, mu, col_mask):
        M_str = f'[{M[0,0]:+.4f} {M[0,1]:+.4f} | {M[1,0]:+.4f} {M[1,1]:+.4f}]'
        extra = ''
        if self.options['monodromy_constraint'] == 'trace':
            extra = f'  tr(M)={float(M[0,0] + M[1,1]):+.6f}'
        print(f'iter {i:3d}  ||r||={np.linalg.norm(b):.3e}  cond(J)={np.linalg.cond(Jm):.3e}  M={M_str}{extra}')
        print(f'  mu: {self._format_mu(mu, col_mask)}')

    def residual_norm_no_aux(self, biotsavart, length=None):
        """Evaluate the periodic field-line + monodromy residual at self.curve
        and self.curve_tm using `biotsavart` directly. No auxiliary coils, no
        Newton iteration, no Jacobian. Returns (r, M)."""
        if length is None:
            length = CurveLength(self.curve).J()
        curve = self.curve
        curve_tm = self.curve_tm

        # periodic-field-line residual
        pts = curve.gamma()
        biotsavart.set_points(pts.reshape((-1, 3)))
        B = biotsavart.B()
        modB = np.linalg.norm(B, axis=1)
        res_fl = (curve.gammadash() / length - B / modB[:, None]).flatten()
        if not curve.stellsym:
            res_fl = np.concatenate((res_fl, [curve.gamma()[0, 1]]))
        
        # monodromy residual on the Chebyshev-grid curve
        pts_tm = curve_tm.gamma()
        biotsavart.set_points(pts_tm.reshape((-1, 3)))
        B_tm = biotsavart.B()
        dB_tm = biotsavart.dB_by_dX()
        M = np.asarray(self.monodromy_matrix_jax(B_tm, dB_tm, length, pts_tm))

        mon_constraint = self.options.get('monodromy_constraint', 'identity')
        if mon_constraint == 'trace':
            r_mon = np.array([float(M[0, 0] + M[1, 1] - 2.0)])
        elif mon_constraint == 'identity':
            r_mon = (M - np.eye(2)).reshape(4)
        else:
            raise ValueError(f"Unknown monodromy_constraint {mon_constraint!r}.")

        r = np.concatenate((res_fl, r_mon))
        return r, M

    def solve_residual_equation_exactly_newton(self, tol=1e-9, maxiter=10, length=None, mu=None, fixed_mu=None, verbose=False):
        if not self.need_to_run_code:
            return self.res
        
        curve = self.curve
        curve_tm = self.curve_tm
        if length is None:
            length = CurveLength(self.curve).J()
        if mu is None:
            mu = np.zeros((4,))
        
        # make the column and row masks
        mon_constraint = self.options['monodromy_constraint']
        n_mon = 4 if mon_constraint == 'identity' else 1
        nmu = len(mu)
        row_mask = self.get_stellsym_mask(tail=n_mon)
        col_mask = self.get_stellsym_mask(tail=nmu)
        x = np.concatenate((curve.get_dofs(), [length], mu))
        i = 0

        if mon_constraint == 'identity':
            row_mask[-4] = False  # det(M)=1 makes one of the four equations redundant.
        N = (nmu - 1) // 2
        _mu_mask_idx = {'z': -1}
        for k in range(N):
            _mu_mask_idx[f'I{k+1}'] = -nmu + k
            _mu_mask_idx[f'r{k+1}'] = -nmu + N + k
        if fixed_mu is not None:
            for k in ([fixed_mu] if isinstance(fixed_mu, str) else fixed_mu):
                col_mask[_mu_mask_idx[k]] = False
        r, J, M = singular_field_line_residual(curve, curve_tm, length, self.biotsavart, mu, self.monodromy_fns, stellsym=self.stellsym_aux, monodromy_constraint=mon_constraint)
        
        b = r[row_mask]
        Jm = J[row_mask][:, col_mask]
        if verbose:
            self._print_iter(i, b, Jm, M, mu, col_mask)

        norm = 1e6
        while i < maxiter:
            norm = np.linalg.norm(b)
            if norm <= tol:
                break
            if self.options['use_lstsq']:
                dx, _, _, _ = np.linalg.lstsq(Jm, b, rcond=None)
            else:
                dx = np.linalg.solve(Jm, b)
                dx += np.linalg.solve(Jm, b - Jm @ dx)
            x[col_mask] -= dx
            curve.set_dofs(x[:-(nmu + 1)])
            length = x[-(nmu + 1)]
            mu = x[-nmu:]
            i += 1
            r, J, M = singular_field_line_residual(curve, curve_tm, length, self.biotsavart, mu, self.monodromy_fns, stellsym=self.stellsym_aux, monodromy_constraint=mon_constraint)
            b = r[row_mask]
            Jm = J[row_mask][:, col_mask]
            self._print_iter(i, b, Jm, M, mu, col_mask)

        #P, L, U = lu(Jm)
        res = {
            'residual': r,
            'jacobian': Jm,
            'iter': i,
            'success': norm <= tol,
            'length': length,
            'mu': mu,  
            #'PLU': (P, L, U),
            'mask': col_mask,
            'monodromy_matrix': M,  
        }

        if verbose:
            extra = ''
            if mon_constraint == 'trace':
                extra = f"  tr(M)={float(M[0,0] + M[1,1]):+.6f}"
            print(f"NEWTON solve  success={res['success']}  iter={res['iter']}  length={res['length']:.8f}  "
                  f"||r||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e}  cond(J)={np.linalg.cond(Jm):.3e}{extra}",
                  flush=True)
            print(f"  mu: {self._format_mu(res['mu'], col_mask)}", flush=True)

        self.res = res
        self.need_to_run_code = False
        return res

        #P, L, U = lu(Jm)
        res = {
            'residual': r,
            'jacobian': Jm,
            'iter': i,
            'success': norm <= tol,
            'length': length,
            'mu': mu,  
            #'PLU': (P, L, U),
            'mask': col_mask,
            'monodromy_matrix': M,  
        }

        if verbose:
            print(f"NEWTON solve - {res['success']}  iter={res['iter']}, length={res['length']:.8f}, mu={res['mu']}, ||residual||_inf = {np.linalg.norm(res['residual'], ord=np.inf):.3e}, cond(J) = {np.linalg.cond(Jm):.3e}", flush=True)
        

        self.res = res
        self.need_to_run_code = False
        return res






        #rng = np.random.default_rng(0)
        #curve = self.curve
        #ndofs = curve.num_dofs()
        #nmu = len(self.fixed_field)
        #x0 = np.concatenate([curve.get_dofs(), [CurveLength(curve).J()], np.zeros(nmu)])
        #v = rng.standard_normal(x0.size)
        #v /= np.linalg.norm(v)
        #jv = J@v
        #jv = jv[:-4]

        #v*=0
        #v[0] = 1.
        #def fun(x):
        #    curve.set_dofs(x[:ndofs])
        #    length = x[ndofs]
        #    mu = x[ndofs+1:]
        #    r, _, _ = self.residual_and_jacobian(length, mu)
        #    return r[:-4]
        #
        #def fd6_directional(fun, x, v, h):
        #    return (
        #        -fun(x + 3*h*v)
        #        + 9*fun(x + 2*h*v)
        #        - 45*fun(x + h*v)
        #        + 45*fun(x - h*v)
        #        - 9*fun(x - 2*h*v)
        #        + fun(x - 3*h*v)
        #    ) / (60*h)

        #hs = 0.5 ** np.arange(5, 10)
        #print('extra-row Taylor test')
        #for h in hs:
        #    fd1 = fd6_directional(fun, x0, v, h)
        #    err_jac = np.linalg.norm(fd1 - jv)
        #    print(f'h={h:.3e}  |FD-Jv|={err_jac:.3e} ')
        #import ipdb;ipdb.set_trace()


