"""
SingularPeriodicFieldline_diff
==============================

A SELF-CONTAINED differentiable singular periodic field line.  This module has
no dependency on ``singularperiodicfieldline.py``: all auxiliary-field,
tangent-map/monodromy and residual machinery is included here.

The class solves, by Newton's method, for a *polished* singular periodic field
line: it finds the field-line curve, its length, and a set of auxiliary planar
circular-coil parameters

    mu = (I_1, ..., I_N, r_1, ..., r_N, Z)

such that the total field B = B_modular + B_aux(mu) admits a periodic field
line whose monodromy matrix equals the identity (or has trace 2).

Partitioned formulation (simsopt dofs)
--------------------------------------
The class is an :class:`Optimizable` whose LOCAL DOFS ARE mu, named
``'I1'..'IN', 'r1'..'rN', 'z'``.  The field-line variables (curve geometry,
length, mu) are partitioned into *dependent* and *independent* variables:

  * the curve geometry dofs and the length are ALWAYS dependent;
  * a mu dof FIXED in the simsopt fashion (``fl.fix('z')``) is DEPENDENT: the
    Newton solver solves for it.  Being an output of the solve, it is removed
    from the optimization design space -- exactly what simsopt fixing does;
  * a mu dof left FREE is INDEPENDENT: a design variable, held at its dof
    value during the Newton solve.

The constraint g = 0 (periodic-field-line equations + label row + monodromy
constraint) implicitly defines the dependent variables as functions of the
independent variables: the free mu and the modular-coil dofs c.

:meth:`solve_residual_equation_exactly_newton` (or :meth:`run_code`) adapts to
the partition: when dg/d(dependent) is SQUARE the Newton step uses the standard
LU solve; otherwise (e.g. UNDER-DETERMINED when all mu are fixed/dependent) the
step is the pseudo-inverse step.

Gradients (implicit function theorem, adjoint)
----------------------------------------------
At the converged polish g(y_dep; y_indep, c) = 0.  Differentiating,

    dy_dep/d(.) = -(dg/dy_dep)^{-1} dg/d(.),     (.) in {independent mu, c}.

For each dependent-mu output q, :meth:`dmu_by_dindependent` solves the SQUARE
adjoint system

    (dg/dy_dep)^T lambda_q = e_q      (forward_backward on lu(Jm))

where e_q selects mu_q among the dependent variables, and assembles

    d(mu_q)/d(mu_indep) = -lambda_q^T (dg/dmu_indep)
        -- the independent-mu columns of the residual Jacobian, and

    d(mu_q)/dc          = -lambda_q^T (dg/dc)
        -- via SIMSOPT's Derivative framework.  g depends on c only through
        Bcoils at the field-line points and (Bcoils, grad Bcoils) at the
        tangent-map points, so lambda_q^T dg/dc is delivered by two VJPs:

        field-line block, at curve.gamma():   B_vjp(seed_B_fl)
            seed_B_fl[i,j] = - sum_l lm_fl[i,l] dres2_dB[i,l,j],
            dres2_dB[i,l,j] = d(B_l/|B|)/dB_j   (so dres/dB = -dres2_dB);

        monodromy block, at curve_tm.gamma(): B_and_dB_vjp(seed_B_mon, seed_gradB_mon)
            seed_B_mon[i,j]       = sum_a lm_mon[a] dM_dB[a,i,j],
            seed_gradB_mon[i,k,j] = sum_a lm_mon[a] dM_dgradB[a,i,k,j],
            in the simsopt dB_by_dX layout [i,k,j] = dB_j/dx_k.

        lm_full is lambda_q scattered back onto the full residual rows
        (lm_full[row_mask] = lambda_q), split into field-line part lm_fl and
        monodromy part lm_mon.  For the 'trace' constraint the single monodromy
        row is M00+M11-2, so dM_dB/dM_dgradB are combined as rows 0 and 3.

:meth:`dmu_by_dindependent` returns one simsopt ``Derivative`` per dependent
(fixed) mu dof:

    Derivative({self: dmu_q/dmu})  +  dmu_q/dc     (coil-dof VJP Derivative)

where the ``self`` array is the full nmu-vector holding -lambda_q^T dg/dmu_indep
at the free (independent) slots and zeros at the fixed (dependent) slots.
Because the independent mu are the FREE dofs, calling the summed Derivative on
an Optimizable graph collects exactly the independent-mu and coil-dof
sensitivities.  These Derivatives are summable/composable with any other
simsopt Derivative.  If the converged partition is not square (e.g. the
under-determined pinv solve), requesting the derivatives raises a RuntimeError.
"""

import sys
from functools import partial

import numpy as np
from scipy.linalg import lu

import jax
jax.config.update("jax_enable_x64", True)   # the residual/monodromy chains need float64
import jax.numpy as jnp
from jax import jit

from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.geo import CurveLength, CurveXYZFourierSymmetries
from simsopt.objectives import forward_backward

__all__ = ['SingularPeriodicFieldline_diff', 'DependentMu', 'AuxCoilDistance',
           'singularperiodicfieldline_dcoils_dcurrents_vjp']

# mu_0 / (4 pi)  in SI units (T·m/A)
_MU0_4PI = 1.0e-7
# Rescale auxiliary currents so I=1 here matches simsopt's
# ScaledCurrent(Current(1.0), 1e7/(4*pi)) convention.
_CURRENT_SCALE = 1.0e7 / (4.0 * np.pi)


# -----------------------------------------------------------------------------
# Auxiliary circular-coil field (jax)
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Chebyshev collocation, tangent map and monodromy
# -----------------------------------------------------------------------------

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


def tangent_map_pure(B, gradB, L, D):
    """Solve the tangent-map collocation system on the Chebyshev grid.

    The ODE  T'/L = A(x) T,  T(0) = ic  is LINEAR in T, so the collocation
    matrix is assembled directly instead of via jacfwd of a residual:

        M[3i+l, 3m+l'] = D[i,m]/L * delta_{l,l'} - delta_{i,m} A[i,l,l'],

    with the first block-row replaced by the initial condition T[0] = ic.
    All three unit initial conditions are solved with ONE factorization.
    """
    N = B.shape[0]
    A = A_pure(B, gradB)                                   # (N,3,3): A[i,l,k] = d(B_l/|B|)/dx_k
    I3 = jnp.eye(3)
    M = (jnp.kron(jnp.asarray(D), I3) / L).reshape(N, 3, N, 3)
    idx = jnp.arange(N)
    M = M.at[idx, :, idx, :].add(-A)                       # subtract the block diagonal
    M = M.reshape(3 * N, 3 * N)
    M = M.at[0:3, :].set(0.0).at[0:3, 0:3].set(I3)         # initial-condition rows
    rhs = jnp.zeros((3 * N, 3)).at[0:3, 0:3].set(I3)       # the three unit ICs side by side
    T = jnp.linalg.solve(M, rhs)                           # one factorization, 3 RHS
    return T.reshape(N, 3, 3)                              # [node, component, which-IC]


def monodromy_pure(B, gradB, L, gammadash, gammadashdash, D):
    """Project the tangent map onto the (normal, binormal) frame.

        fT = g'/|g'|,   fT' = (g'' - fT (fT.g''))/|g'|,   fN = fT'/|fT'|.

    WHY g'/g'' INSTEAD OF D@gamma (don't "simplify" this back!):
    -----------------------------------------------------------
    The frame here is built from the curve's EXACT analytic Fourier
    derivatives (curve_tm.gammadash()/gammadashdash()), NOT by applying the
    Chebyshev differentiation matrix D to the sampled curve points. The old
    code did `tangent = D@gamma; normal = D@fT` -- TWO numerical
    differentiations -- and that floored the Newton residual at ~5e-10 (M was
    only reproducible to ~1e-8). The fix took it to ~1e-14.

    The mechanism: differentiating SAMPLED values amplifies round-off, because
    the per-node float64 error (~1e-16, ABSOLUTE and the same size at every
    node) is broadband -- it has energy at all modes up to the grid frequency
    ~N, including the highest modes, where the (band-limited, smooth) true
    curve has ~zero content. D multiplies mode k by ~k and, being a Chebyshev
    matrix on N points, amplifies the top modes by ~||D|| ~ N^2. So a 1e-16
    noise mode sitting where the signal is ~0 gets blown up by ~N^2 with no
    signal to keep the relative error small; doing it TWICE (tangent, then
    normal) squares that to ~N^4. With N = 6*order+1 = 97, N^4 ~ 1e8, hence
    1e-16 -> ~1e-8 garbage in fN/fB and therefore in M.

    The analytic Fourier derivative does NOT have this problem even though it
    also multiplies high modes by k: the curve is stored as only ~order Fourier
    coefficients c_k (no spurious high modes exist), and each c_k carries a
    RELATIVE ~1e-16 error. Differentiation scales the coefficient AND its error
    by the same (2*pi*k), so the relative accuracy is preserved (the factor
    cancels in the ratio) -- it is a diagonal scaling with condition number
    ~order(~16), not the ~N^2-norm, noise-injecting D-on-samples operator.
    In short: the danger is not "multiply by k", it is differentiating
    resampled band-limited data, where noise is a fixed absolute floor sitting
    in high modes whose true value is 0. Staying in coefficient form avoids it.

    (The tangent-map SOLVE below still uses D once, inside a linear solve; that
    contributes only ~1e-15 to M -- a single, well-conditioned use, not a raw
    double differentiation of the curve. Only the frame construction was the
    culprit, and only it was changed.)
    """
    ngd = jnp.linalg.norm(gammadash, axis=-1)[:, None]
    fT = gammadash / ngd
    tp = (gammadashdash - fT * jnp.sum(fT * gammadashdash, axis=-1)[:, None]) / ngd
    fN = tp / jnp.linalg.norm(tp, axis=-1)[:, None]
    binormal = jnp.cross(fT, fN)
    fB = binormal / jnp.linalg.norm(binormal, axis=-1)[:, None]
    NB = jnp.concatenate((fN[:, :, None], fB[:, :, None]), axis=-1)
    NB_t = jnp.concatenate((fN[:, None, :], fB[:, None, :]), axis=-2)
    M = tangent_map_pure(B, gradB, L, D)
    return jnp.matmul(NB_t, jnp.matmul(M, NB[0]))


def monodromy_matrix_pure(B, gradB, L, gammadash, gammadashdash, D):
    return monodromy_pure(B, gradB, L, gammadash, gammadashdash, D)[-1]


# -----------------------------------------------------------------------------
# Residual and Jacobian
# -----------------------------------------------------------------------------

def singular_field_line_residual(curve, curve_tm, length, field, mu, monodromy_fns,
                                 stellsym=True, monodromy_constraint='identity'):

    pts = curve.gamma()
    dpts_dcurve = curve.dgamma_by_dcoeff()

    field.set_points(pts.reshape((-1, 3)))

    Bcoils = field.B()
    dBcoils_by_dX = field.dB_by_dX()
    #d2Bcoils_by_dXdX = field.d2B_by_dXdX()  # unused in the field-line block; skip the expensive Hessian

    B_aux = _B_aux(pts, mu, stellsym=stellsym)
    dB_aux_by_dX = _dB_aux_by_dX(pts, mu, stellsym=stellsym)
    #d2B_aux_by_dXdX = _d2B_aux_by_dXdX(pts, mu, stellsym=stellsym)

    dB_aux_by_dmu = _dB_aux_by_dmu(pts, mu, stellsym=stellsym)
    #dgradB_aux_by_dmu = _dgradB_aux_by_dmu(pts, mu, stellsym=stellsym)

    B = Bcoils + B_aux
    dB_by_dX = dBcoils_by_dX + dB_aux_by_dX
    #d2B_by_dXdX = d2Bcoils_by_dXdX + d2B_aux_by_dXdX

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
    #dgradB_dcurve  = np.einsum('ijkl,ikm->ijlm', d2B_by_dXdX, dpts_dcurve, optimize=True)

    dres2_dcoeff = np.einsum('ikl,ikm->ilm', dres2_dB, dB_dcurve, optimize=True)

    dres_del = -curve.gammadash().reshape((-1, 1)) / length ** 2
    dres_dmu = -np.einsum('ikl,ikm->ilm', dres2_dB, dB_aux_by_dmu, optimize=True)
    dres_dcoeff = dres1_dcoeff - dres2_dcoeff
    dres = np.concatenate((dres_dcoeff.reshape((res.size, -1)),
                           dres_del,
                           dres_dmu.reshape((res.size, -1))), axis=-1)

    if not curve.stellsym:
        res = np.concatenate((res, [res_y]))
        dres = np.concatenate((dres, dres_y[None, :]), axis=0)

    pts = curve_tm.gamma()
    dpts_dcurve = curve_tm.dgamma_by_dcoeff()
    # exact Fourier derivatives of the curve for the monodromy frame
    gd = curve_tm.gammadash()
    gdd = curve_tm.gammadashdash()
    dgd_dcurve = curve_tm.dgammadash_by_dcoeff()
    dgdd_dcurve = curve_tm.dgammadashdash_by_dcoeff()

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

    # Fused path: one primal tangent-map solve + one (vmapped) vjp pullback
    # yields M and all five derivative tensors together.
    M, dM_dB, dM_dgradB, dM_dL, dM_dgd, dM_dgdd = monodromy_fns['all'](B, dB_by_dX, length, gd, gdd)
    M = np.asarray(M)
    dM_dB = np.asarray(dM_dB).reshape((4,) + B.shape)
    dM_dgradB = np.asarray(dM_dgradB).reshape((4,) + dB_by_dX.shape)
    dM_dL = np.asarray(dM_dL).reshape(4, 1)
    dM_dgd = np.asarray(dM_dgd).reshape((4,) + gd.shape)
    dM_dgdd = np.asarray(dM_dgdd).reshape((4,) + gdd.shape)

    dM_dcurve = np.einsum('aij,ijm->am', dM_dB, dB_dcurve, optimize=True)
    dM_dcurve += np.einsum('aijk,ijkm->am', dM_dgradB, dgradB_dcurve, optimize=True)
    dM_dcurve += np.einsum('aik,ikm->am', dM_dgd, dgd_dcurve, optimize=True)
    dM_dcurve += np.einsum('aik,ikm->am', dM_dgdd, dgdd_dcurve, optimize=True)

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


def _mu_names(nmu):
    """Names of the mu components: (I_1..I_N, r_1..r_N, z)."""
    N = (nmu - 1) // 2
    return [f'I{k+1}' for k in range(N)] + [f'r{k+1}' for k in range(N)] + ['z']


def singularperiodicfieldline_dcoils_dcurrents_vjp(lm, biotsavart, fieldline):
    """lm^T dg/d(external inputs) at the converged singular polish, with the
    same call signature as periodicfieldline_dcoils_dcurrents_vjp so the class
    can be used by VesselDistance / FieldLineMeanZ / etc. via res['vjp'].

    ``lm`` lives on the ACTIVE residual rows (it is the forward_backward
    solution of Jm^T adj = dJ/dx built from res['PLU']).  Returns a simsopt
    Derivative with two parts:
      * the modular-coil part, assembled with B_vjp / B_and_dB_vjp;
      * the INDEPENDENT-mu part, Derivative({fieldline: lm^T dg/dmu_indep}),
        so downstream objectives also get correct gradients with respect to
        the free mu design variables (the polished curve depends on them).
    """
    row_mask, n_mon = fieldline._row_mask()
    dres2_dB, dM_dB, dM_dgradB = fieldline._field_partials()
    deriv = fieldline._lm_to_vjp(lm, row_mask, n_mon, dres2_dB, dM_dB, dM_dgradB)
    J_indep = fieldline.res.get('J_indep')
    if J_indep is not None and J_indep.shape[1] > 0:
        g = np.zeros(fieldline.local_full_dof_size)
        g[fieldline.res['indep_idx']] = lm @ J_indep
        deriv = deriv + Derivative({fieldline: g})
    return deriv


class SingularPeriodicFieldline_diff(Optimizable):
    """Self-contained singular periodic field line whose LOCAL DOFS ARE mu,
    with an adaptive Newton polish over the dependent/independent partition
    implied by the simsopt fixed/free dof state, and adjoint gradients of the
    dependent (fixed) mu returned as simsopt ``Derivative`` objects.

    Typical use::

        fl = SingularPeriodicFieldline_diff(biotsavart, curve, mu0, options=..., stellsym_aux=...)
        for name in ('I1', 'I2', 'I3'):
            fl.fix(name)              # fixed mu = DEPENDENT (solved by Newton)
        res = fl.run_code(length)     # square -> LU; non-square -> pinv step
        grads = fl.dmu_by_dindependent()   # {'I1': Derivative, 'I2': ..., 'I3': ...}
        grads['I1'](opt)              # d I1 / d(free dofs of opt's graph):
                                      # independent mu + modular-coil dofs
    """

    def __init__(self, biotsavart, curve, mu=None, options=None, stellsym_aux=True,
                 monodromy_matrix=None, length=None, dofs=None):
        # mu ARE the local dofs of this Optimizable.  Fixing a mu dof
        # (self.fix('z')) marks it DEPENDENT: the Newton solver solves for it
        # and it is removed from the optimization design space.  Free mu dofs
        # are INDEPENDENT design variables, held during the Newton solve.
        #
        # ``dofs`` is supplied by simsopt's from_dict on reconstruction: it is a
        # DOFs object carrying the mu values, names AND the fixed/free mask, so
        # it preserves the dependent/independent partition across save/load.
        # It takes precedence over ``mu`` (also serialized, but maskless).
        if dofs is not None:
            super().__init__(dofs=dofs, depends_on=[biotsavart])
        else:
            mu = np.asarray(mu, dtype=float)
            super().__init__(x0=mu, names=_mu_names(len(mu)), depends_on=[biotsavart])
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
        if 'monodromy_constraint' not in options:
            options['monodromy_constraint'] = 'identity'
        self.options = options
        self.stellsym_aux = stellsym_aux
        # Solve results we want to survive save/load. They appear as __init__
        # kwargs so simsopt's as_dict/from_dict round-trips them automatically
        # (mu round-trips through the dofs).
        self.monodromy_matrix = np.asarray(monodromy_matrix) if monodromy_matrix is not None else None
        self.length = length

        nfp = curve.nfp  # match the tangent-map toroidal period.
        N = 6 * curve.order + 1  # use the same Chebyshev resolution as TangentMap.
        D, xh, wh = cheb(N, 0, 1.0 / nfp)  # tangent map is evaluated on the Chebyshev grid.
        self.D = D  # store the Chebyshev differentiation matrix.
        self.xh = xh  # store the Chebyshev nodes.
        self.wh = wh  # keep the quadrature weights for consistency with TangentMap.
        self.curve_tm = CurveXYZFourierSymmetries(self.xh, curve.order, curve.nfp, curve.stellsym, ntor=curve.ntor, dofs=curve.dofs)  # monodromy is evaluated on a Chebyshev-grid copy of the curve.

        # Build the AD transforms ONCE and jit them: the lambda closes over the
        # (constant) Chebyshev matrix D and shapes are fixed per object, so each
        # function compiles on its first call and runs as XLA afterwards.
        _mono = lambda B, gradB, L, gd, gdd: monodromy_matrix_pure(B, gradB, L, gd, gdd, self.D)
        self.monodromy_matrix_jax = jit(_mono)   # primal only (used by residual_norm_no_aux)

        # Fused evaluation: M is 2x2 (4 outputs) while B/gradB/gd/gdd have
        # ~3N/9N/3N/3N inputs, so reverse mode is the right AD direction.  ONE
        # vjp pullback, vmapped over the 4 unit cotangents of M, returns the
        # cotangents of ALL five inputs at once (dM/dB, dM/dgradB, dM/dL,
        # dM/dgammadash, dM/dgammadashdash), sharing the primal tangent-map solve.
        def _mono_all(B, gradB, L, gd, gdd):
            M, pullback = jax.vjp(_mono, B, gradB, L, gd, gdd)
            dB, dgradB, dL, dgd, dgdd = jax.vmap(pullback)(jnp.eye(4).reshape((4, 2, 2)))
            return M, dB, dgradB, dL, dgd, dgdd
        self.monodromy_matrix_all = jit(_mono_all)

        self.monodromy_fns = {
            'jax': self.monodromy_matrix_jax,
            'all': self.monodromy_matrix_all,
        }

        # Reconstructed from a saved file with results present: expose them via
        # self.res and don't re-solve. We deliberately do NOT set 'success' —
        # the saved fieldline's convergence status is not recorded, so callers
        # must not assume it solved correctly.
        if self.monodromy_matrix is not None:
            self.res = {'monodromy_matrix': self.monodromy_matrix, 'mu': self.mu,
                        'length': self.length}
            self.need_to_run_code = False

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True
        self._dmu_cache = None       # invalidate the cached dmu_by_dindependent() dict
        self._partials_cache = None  # invalidate the cached _field_partials() tensors

    @property
    def mu(self):
        """The auxiliary-coil parameter vector (I_1..I_N, r_1..r_N, Z) == the
        local dofs of this Optimizable (both free and fixed entries)."""
        return np.asarray(self.local_full_x)

    @mu.setter
    def mu(self, value):
        self.local_full_x = np.asarray(value, dtype=float)

    # ------------------------------------------------------------ solvers
    def run_code(self, length, mu=None):
        if not self.need_to_run_code:
            return self.res
        return self.solve_residual_equation_exactly_newton(length=length, mu=mu, tol=self.options['newton_tol'], maxiter=self.options['newton_maxiter'], verbose=self.options['verbose'])

    def num_independent_mu(self, monodromy_constraint):
        """Number of mu dofs that must remain FREE (independent) so that
        dg/d(dependent variables) is square; equivalently, the user must FIX
        ``nmu - num_independent_mu(...)`` mu dofs (the dependent ones).

        Parameters
        ----------
        monodromy_constraint : str
            'trace' (1 monodromy row) or 'identity' (4 rows, one redundant).

        Returns
        -------
        int : required number of independent (free) mu
            = #(state columns: curve dofs + length + all nmu)
            - #(constraint rows: field-line + label + monodromy, with the
                redundant identity row already dropped).

        For example, with a non-stellsym order-16 curve (99 curve dofs + 1
        length) and num_aux=3 (nmu=7): 'trace' has 101 constraint rows
        -> 6 free mu (fix 1); 'identity' has 103 -> 4 free mu (fix 3).
        """
        return self._num_independent_mu(self.local_full_dof_size, monodromy_constraint)

    def _num_independent_mu(self, nmu, monodromy_constraint):
        """num_independent_mu for an explicit nmu."""
        if monodromy_constraint not in ('identity', 'trace'):
            raise ValueError(f"Unknown monodromy_constraint {monodromy_constraint!r}; "
                             "must be 'identity' or 'trace'.")
        row_mask, _ = self._row_mask(monodromy_constraint)
        n_rows = int(np.sum(row_mask))
        n_cols_full = int(np.sum(self.get_stellsym_mask(tail=int(nmu))))
        n_indep = n_cols_full - n_rows
        if n_indep < 0 or n_indep > nmu:
            raise ValueError(
                f"No valid partition exists: {n_cols_full} state columns vs "
                f"{n_rows} constraint rows requires {n_indep} independent mu, "
                f"but mu has only {nmu} components.")
        return n_indep

    def solve_residual_equation_exactly_newton(self, tol=1e-9, maxiter=10, length=None, mu=None, verbose=False):
        """Newton solve for the DEPENDENT variables: the curve dofs, the
        length, and the FIXED mu dofs.  FREE mu dofs are independent design
        variables and are held at their current dof values.

        The linear step adapts to the partition implied by the dof state:
        a square dg/d(dependent) uses the standard LU solve; a non-square
        system (e.g. under-determined when all mu are fixed/dependent) uses
        the pseudo-inverse (minimum-norm least-squares) step.

        ``mu`` optionally overrides the initial mu; it is written into the
        local dofs.
        """
        if not self.need_to_run_code:
            return self.res

        curve = self.curve
        curve_tm = self.curve_tm
        if length is None:
            length = CurveLength(self.curve).J()
        if mu is not None:
            self.mu = mu        # write the supplied initial mu into the dofs
        mu = self.mu            # full local dofs (free and fixed entries)

        # make the column and row masks; the mu tail of the column mask is the
        # NEGATED simsopt free-dof status: fixed = dependent (solved for),
        # free = independent (held).
        mon_constraint = self.options['monodromy_constraint']
        n_mon = 4 if mon_constraint == 'identity' else 1
        nmu = len(mu)
        row_mask = self.get_stellsym_mask(tail=n_mon)
        col_mask = self.get_stellsym_mask(tail=nmu)
        col_mask[-nmu:] = ~np.asarray(self.local_dofs_free_status, dtype=bool)
        x = np.concatenate((curve.get_dofs(), [length], mu))
        i = 0

        if mon_constraint == 'identity':
            row_mask[-4] = False  # det(M)=1 makes one of the four equations redundant.
        r, J, M = singular_field_line_residual(curve, curve_tm, length, self.biotsavart, mu, self.monodromy_fns, stellsym=self.stellsym_aux, monodromy_constraint=mon_constraint)

        b = r[row_mask]
        Jm = J[row_mask][:, col_mask]
        square = Jm.shape[0] == Jm.shape[1]
        #if verbose:
        #    self._print_iter(i, b, Jm, M, mu, col_mask)

        norm = 1e6
        while i < maxiter:
            norm = np.linalg.norm(b, ord=np.inf)
            if norm <= tol:
                break
            if square:
                dx = np.linalg.solve(Jm, b)             # LU
                dx += np.linalg.solve(Jm, b - Jm @ dx)  # one step of iterative refinement
            else:
                # non-square: pseudo-inverse step, dx = pinv(Jm) @ b
                # (minimum-norm least-squares solution).
                dx, _, _, _ = np.linalg.lstsq(Jm, b, rcond=None)
            x[col_mask] -= dx
            curve.set_dofs(x[:-(nmu + 1)])
            length = x[-(nmu + 1)]
            mu = x[-nmu:]
            i += 1
            r, J, M = singular_field_line_residual(curve, curve_tm, length, self.biotsavart, mu, self.monodromy_fns, stellsym=self.stellsym_aux, monodromy_constraint=mon_constraint)
            b = r[row_mask]
            Jm = J[row_mask][:, col_mask]
            #if verbose:
            #    self._print_iter(i, b, Jm, M, mu, col_mask)

        res = {
            'residual': b,
            'jacobian': Jm,
            'iter': i,
            'success': norm <= tol,
            'length': length,
            'mu': mu,
            'mask': col_mask,
            'square': square,
            'monodromy_matrix': M,
        }
        # Adjoint plumbing for downstream objectives (VesselDistance,
        # FieldLineMeanZ, ...): the independent-mu columns of dg on the active
        # rows, and -- for square partitions -- the PLU factorization and the
        # dg/d(coils, independent mu) vjp.
        indep_idx = np.where(~col_mask[-nmu:])[0]
        res['indep_idx'] = indep_idx
        res['J_indep'] = J[row_mask][:, J.shape[1] - nmu + indep_idx]
        if square:
            res['PLU'] = lu(Jm)
            res['vjp'] = singularperiodicfieldline_dcoils_dcurrents_vjp

        if verbose:
            print(f"NEWTON solve - {res['success']}  iter={res['iter']}, length={res['length']:.8f}, "
                  f"||residual||_inf = {np.linalg.norm(res['residual'], ord=np.inf):.3e}, "
                  f"cond(J) = {np.linalg.cond(Jm):.3e} singular", flush=True)
            #print(f"  mu: {self._format_mu(res['mu'], col_mask)}", flush=True)

        self.res = res
        # Mirror the converged results into the serialized attributes; writing
        # self.mu sets the local dofs (and rings recompute_bell), so clear the
        # need_to_run_code flag afterwards.
        self.monodromy_matrix = np.asarray(M)
        self.mu = np.asarray(mu)
        self.length = length
        self.need_to_run_code = False
        return res

    # ------------------------------------------------------------------ utils
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

    def _row_mask(self, monodromy_constraint=None):
        """Reconstruct the residual-row mask used by the Newton solve.
        Defaults to the constraint in self.options."""
        mon_constraint = monodromy_constraint if monodromy_constraint is not None \
            else self.options['monodromy_constraint']
        n_mon = 4 if mon_constraint == 'identity' else 1
        row_mask = self.get_stellsym_mask(tail=n_mon)
        if mon_constraint == 'identity':
            row_mask[-4] = False  # det(M)=1 makes one of the four equations redundant.
        return row_mask, n_mon

    def _format_mu(self, mu, col_mask):
        """Pretty-print mu: bold names, '(indep)' suffix for entries not solved
        for (free dofs, held during Newton).  Currents are scaled by
        _CURRENT_SCALE so values shown are in physical Amperes."""
        # Only emit ANSI bold codes for an interactive terminal; otherwise (e.g.
        # piped through tee into a log file) they show up as raw ^[[1m bytes.
        if sys.stdout.isatty():
            BOLD, RST = '\033[1m', '\033[0m'
        else:
            BOLD, RST = '', ''
        nmu = len(mu)
        N = (nmu - 1) // 2
        names = _mu_names(nmu)
        parts = []
        for k, (name, val) in enumerate(zip(names, mu)):
            is_indep = not bool(col_mask[-nmu + k])   # not solved for = free dof
            if k < N:
                token = f'{BOLD}{name}{RST}={val * _CURRENT_SCALE:+.4e}'
            else:
                token = f'{BOLD}{name}{RST}={val:+.4f}'
            if is_indep:
                token += ' (indep)'
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
        M = np.asarray(self.monodromy_matrix_jax(B_tm, dB_tm, length,
                                                 curve_tm.gammadash(), curve_tm.gammadashdash()))

        mon_constraint = self.options.get('monodromy_constraint', 'identity')
        if mon_constraint == 'trace':
            r_mon = np.array([float(M[0, 0] + M[1, 1] - 2.0)])
        elif mon_constraint == 'identity':
            r_mon = (M - np.eye(2)).reshape(4)
        else:
            raise ValueError(f"Unknown monodromy_constraint {mon_constraint!r}.")

        r = np.concatenate((res_fl, r_mon))
        return r, M

    def _field_partials(self):
        """Recompute the field-dependence partials of the residual at the
        converged state.

        Returns
        -------
        dres2_dB : (npts, 3, 3)   d(B_l/|B|)/dB_j on the field-line points.
        dM_dB    : (n_mon, npts_tm, 3)      d r_mon / dB on the monodromy points.
        dM_dgradB: (n_mon, npts_tm, 3, 3)   d r_mon / d(gradB) ([i,k,j] layout).

        Cached per converged state (several downstream penalties call the vjp
        per gradient evaluation); cleared by recompute_bell.
        """
        if getattr(self, '_partials_cache', None) is not None:
            return self._partials_cache
        res = self.res
        length = res['length']
        mu = np.asarray(res['mu'])
        stellsym = self.stellsym_aux
        field = self.biotsavart

        # --- field-line points -------------------------------------------------
        pts = self.curve.gamma()
        field.set_points(pts.reshape((-1, 3)))
        B = field.B() + np.asarray(_B_aux(pts, mu, stellsym=stellsym))
        modB = np.linalg.norm(B, axis=1)
        idx = np.arange(3)
        diag = np.zeros((pts.shape[0], 3, 3))
        diag[:, idx, idx] = 1.0 / modB[:, None]
        # dres2_dB[i,l,j] = delta_lj/|B| - B_l B_j/|B|^3  (matches the residual).
        dres2_dB = -B[:, None, :] * B[:, :, None] / modB[:, None, None] ** 3 + diag

        # --- monodromy (Chebyshev) points -------------------------------------
        pts_tm = self.curve_tm.gamma()
        field.set_points(pts_tm.reshape((-1, 3)))
        B_tm = field.B() + np.asarray(_B_aux(pts_tm, mu, stellsym=stellsym))
        dB_tm = field.dB_by_dX() + np.asarray(_dB_aux_by_dX(pts_tm, mu, stellsym=stellsym))

        _, dM_dB, dM_dgradB, _, _, _ = self.monodromy_fns['all'](
            B_tm, dB_tm, length, self.curve_tm.gammadash(), self.curve_tm.gammadashdash())
        dM_dB = np.asarray(dM_dB).reshape((4,) + B_tm.shape)
        dM_dgradB = np.asarray(dM_dgradB).reshape((4,) + dB_tm.shape)

        mon_constraint = self.options['monodromy_constraint']
        if mon_constraint == 'trace':
            # single equation M00 + M11 - 2  ->  combine rows 0 and 3.
            dM_dB = (dM_dB[0] + dM_dB[3])[None, ...]
            dM_dgradB = (dM_dgradB[0] + dM_dgradB[3])[None, ...]
        self._partials_cache = (dres2_dB, dM_dB, dM_dgradB)
        return self._partials_cache

    def _lm_to_vjp(self, lm_active, row_mask, n_mon, dres2_dB, dM_dB, dM_dgradB):
        """Given an adjoint vector on the active (masked) residual rows, return
        the Derivative  lm^T dg/dc  over the modular-coil dofs."""
        field = self.biotsavart
        stellsym = self.curve.stellsym

        # scatter back to the full residual rows
        lm_full = np.zeros(row_mask.shape[0])
        lm_full[row_mask] = lm_active

        base = row_mask.shape[0] - n_mon          # size of the field-line block
        lm_fl_flat = lm_full[:base]
        lm_mon = lm_full[base:]                    # length n_mon (a masked entry may be 0)

        # --- field-line block:  B_vjp at curve.gamma() ------------------------
        if not stellsym:
            lm_fl_flat = lm_fl_flat[:-1]           # drop the y=0 label row (no B dependence)
        npts = lm_fl_flat.size // 3
        lm_fl = lm_fl_flat.reshape((npts, 3))
        # dres/dB = -dres2_dB  ->  seed[i,j] = -sum_l lm_fl[i,l] dres2_dB[i,l,j]
        seed_B_fl = -np.einsum('il,ilj->ij', lm_fl, dres2_dB, optimize=True)
        pts = self.curve.gamma()
        field.set_points(pts.reshape((-1, 3)))
        deriv = field.B_vjp(seed_B_fl)

        # --- monodromy block: B_and_dB_vjp at curve_tm.gamma() ----------------
        seed_B_mon = np.einsum('a,aij->ij', lm_mon, dM_dB, optimize=True)
        seed_gradB_mon = np.einsum('a,aikj->ikj', lm_mon, dM_dgradB, optimize=True)
        pts_tm = self.curve_tm.gamma()
        field.set_points(pts_tm.reshape((-1, 3)))
        dB_deriv, dgradB_deriv = field.B_and_dB_vjp(seed_B_mon, seed_gradB_mon)
        deriv = deriv + dB_deriv + dgradB_deriv
        return deriv

    # --------------------------------------------------------------- gradients
    def dmu_by_dindependent(self):
        r"""Adjoint gradients of the DEPENDENT (fixed) mu dofs with respect to
        the independent variables: the FREE mu dofs and the modular-coil dofs.

        Requires a converged solve whose partition makes dg/d(dependent)
        square; for a non-square solve (e.g. the under-determined pinv Newton)
        this raises a RuntimeError -- fix exactly the right number of mu dofs
        first (see :meth:`num_independent_mu`).

        Returns
        -------
        dict mapping each dependent-mu name (e.g. 'I1') to a simsopt
        ``Derivative``:

            Derivative({self: d mu_q / d mu})  +  d mu_q / d coils

        where the ``self`` array is the full nmu-vector with
        -lambda_q^T dg/dmu_indep at the FREE (independent) slots and zeros at
        the FIXED (dependent) slots, and the coil part is the VJP Derivative
        over the BiotSavart dofs.  Because the independent mu are free dofs,
        calling the Derivative on an Optimizable graph collects exactly the
        independent-mu and coil-dof sensitivities.  These Derivatives are
        summable with any other simsopt Derivative.
        """
        if self.need_to_run_code:
            raise RuntimeError("Polish the field line (run_code) before "
                               "requesting dmu_by_dindependent.")
        # cached result from a previous call at this converged state (cleared
        # by recompute_bell whenever anything upstream or the dofs change)
        if getattr(self, '_dmu_cache', None) is not None:
            return dict(self._dmu_cache)
        res = self.res
        col_mask = np.asarray(res['mask'], dtype=bool)
        mu = np.asarray(res['mu'])
        length = res['length']
        nmu = len(mu)
        names = _mu_names(nmu)
        mon_constraint = self.options['monodromy_constraint']
        row_mask, n_mon = self._row_mask()

        # Recompute the FULL residual Jacobian at the converged state: we need
        # both the dependent block (square Jm) and the independent-mu columns.
        _, J, _ = singular_field_line_residual(
            self.curve, self.curve_tm, length, self.biotsavart, mu,
            self.monodromy_fns, stellsym=self.stellsym_aux,
            monodromy_constraint=mon_constraint)
        J_act = J[row_mask]

        Jm = J_act[:, col_mask]                     # dg/d(dependent)
        n_rows, n_dep = Jm.shape
        if n_rows != n_dep:
            n_dep_needed = n_rows - (n_dep - int(np.sum(col_mask[-nmu:])))
            raise RuntimeError(
                f"dg/d(dependent) is not square ({n_rows} rows x {n_dep} dependent "
                f"variables): the solve was not square, so dmu/d(independent) is "
                f"not defined. Fix exactly {n_dep_needed} mu dofs "
                f"(e.g. self.fix('I1')) and re-solve.")

        # partition of the mu block (mu is the last nmu state columns):
        # col_mask True = solved for = FIXED dof = dependent.
        mu_dep_mask = col_mask[-nmu:]
        dep_idx = np.where(mu_dep_mask)[0]
        indep_idx = np.where(~mu_dep_mask)[0]
        n_dep_mu = len(dep_idx)
        if n_dep_mu == 0:
            return {}

        # independent-mu columns of the active residual rows: dg/d(mu_indep)
        ncol_full = J.shape[1]
        J_indep = J_act[:, ncol_full - nmu + indep_idx]      # (n_rows, n_indep)

        # Adjoint right-hand sides: the dependent mu are the LAST n_dep_mu
        # entries of the dependent state vector (mu is the last block of
        # x = [curve_dofs, length, mu] and masking preserves order).
        S = np.zeros((n_dep, n_dep_mu))
        for q in range(n_dep_mu):
            S[n_dep - n_dep_mu + q, q] = 1.0

        P, L, U = lu(Jm)
        Lambda = forward_backward(P, L, U, S)       # Jm^T Lambda = S, columns lambda_q

        # d(mu_dep)/d(mu_indep) = -Lambda^T (dg/dmu_indep)
        dmu_dindep = -(Lambda.T @ J_indep)          # (n_dep_mu, n_indep)

        # d(mu_dep)/dc = -lambda_q^T (dg/dc), assembled with the field VJPs
        dres2_dB, dM_dB, dM_dgradB = self._field_partials()
        out = {}
        for q in range(n_dep_mu):
            # coil part: a Derivative over the BiotSavart dofs
            coil_deriv = -1.0 * self._lm_to_vjp(Lambda[:, q], row_mask, n_mon,
                                                dres2_dB, dM_dB, dM_dgradB)
            # independent-mu part: scatter into the FULL local mu vector
            # (zeros at the dependent slots; data arrays in a Derivative span
            # all local dofs, free and fixed).
            self_arr = np.zeros(nmu)
            self_arr[indep_idx] = dmu_dindep[q]
            out[names[dep_idx[q]]] = Derivative({self: self_arr}) + coil_deriv
        self._dmu_cache = out
        return dict(out)


class DependentMu(Optimizable):
    """Objective wrapper exposing one DEPENDENT (fixed) mu dof of a
    :class:`SingularPeriodicFieldline_diff`.

    ``J()`` returns the value of the named dependent mu at the converged
    polish (re-running the Newton solve first if anything upstream changed).
    ``dJ()`` returns its gradient with respect to the independent variables
    (the free mu dofs and the modular-coil dofs): the simsopt ``Derivative``
    produced by ``fl.dmu_by_dindependent()[name]``.  Following the simsopt
    convention, ``dJ(partials=True)`` returns the ``Derivative`` object itself
    and plain ``dJ()`` the gradient array over the free dofs.

    Typical use::

        fl = SingularPeriodicFieldline_diff(bs, curve, mu0, ...)
        fl.fix('I1')                  # I1 is dependent (solved by Newton)
        I1 = DependentMu(fl, 'I1')
        I1.J()                        # solved value of I1
        I1.dJ()                       # d I1 / d(independent mu, coil dofs)
    """

    def __init__(self, fl, name):
        Optimizable.__init__(self, depends_on=[fl])
        names = _mu_names(fl.local_full_dof_size)
        if name not in names:
            raise ValueError(f"Unknown mu name {name!r}; valid names are {names}.")
        self.fl = fl
        self.name = name
        self._idx = names.index(name)
        self.recompute_bell()

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        """Value of the named dependent mu at the converged polish."""
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        """Gradient of the named dependent mu with respect to the independent
        variables (free mu dofs and modular-coil dofs)."""
        if self._dJ is None:
            self.compute()
        return self._dJ

    def compute(self):
        fl = self.fl
        if fl.need_to_run_code:
            length = None
            if getattr(fl, 'res', None) is not None:
                length = fl.res.get('length')
            if length is None:
                length = CurveLength(fl.curve).J()
            fl.run_code(length)
        if bool(np.asarray(fl.local_dofs_free_status)[self._idx]):
            raise RuntimeError(
                f"mu dof {self.name!r} is FREE (independent); DependentMu requires "
                f"a dependent (fixed) dof. Call fl.fix({self.name!r}) and re-solve.")
        self._J = float(fl.mu[self._idx])
        self._dJ = fl.dmu_by_dindependent()[self.name]

    return_fn_map = {'J': J, 'dJ': dJ}


class AuxCoilDistance(Optimizable):
    """One-sided clearance penalty keeping the planar circular AUXILIARY coils
    of one :class:`SingularPeriodicFieldline_diff` at least ``threshold`` away
    from a set of MODULAR coil curves.

    The aux coils are NOT Curve objects -- they live as the
    mu = (I_1..I_N, r_1..r_N, Z) dofs of ``fl`` -- so, like :class:`DependentMu`,
    this reads the aux radii r_k and height Z straight from ``fl.mu`` and returns
    a simsopt ``Derivative`` landing on BOTH fl's (free) mu dofs AND the
    modular-curve dofs (via each curve's dgamma_by_dcoeff vjp).

    Aux coil k is a circle of radius r_k centered on the z-axis at height Z,
    plus its stellarator-symmetric partner at -Z when ``fl.stellsym_aux``.  The
    distance from a modular-curve point p=(x, y, z) to such a circle is closed
    form,

        d = sqrt((rho - r_k)^2 + (z - Z_s)^2),    rho = sqrt(x^2 + y^2),

    so the penalty is the usual squared one-sided violation, summed over every
    modular-curve quadrature point and every aux circle,

        J = sum max(threshold - d, 0)^2.

    Only the geometry (radii r_k, height Z) enters, all of which are independent
    (free) mu design variables, so J/dJ need no Newton solve; the aux currents
    do not affect the distance and receive zero gradient.

    Typical use::

        Jdist = AuxCoilDistance(fl, modular_curves, threshold=0.12)
        Jdist.J()                  # sum of squared clearance violations
        Jdist.shortest_distance()  # min center-line distance (for callbacks)
    """

    def __init__(self, fl, curves, threshold):
        Optimizable.__init__(self, depends_on=[fl, *curves])
        self.fl = fl
        self.curves = list(curves)
        self.threshold = float(threshold)

    def _geom(self):
        """Return the aux radii (array), N, and the aux-circle heights as
        (Z_s, sZ) pairs with Z_s = sZ * Z (base coil + stellsym partner)."""
        mu = np.asarray(self.fl.mu)
        N = (mu.shape[0] - 1) // 2
        radii = mu[N:2 * N]
        Z = float(mu[-1])
        heights = [(Z, +1.0), (-Z, -1.0)] if self.fl.stellsym_aux else [(Z, +1.0)]
        return radii, N, heights

    def shortest_distance(self):
        """Minimum center-line distance between any modular-curve point and any
        aux circle (a single scalar; useful for callback display / error checks)."""
        radii, _, heights = self._geom()
        dmin = np.inf
        for c in self.curves:
            g = c.gamma()
            rho = np.hypot(g[:, 0], g[:, 1])
            z = g[:, 2]
            for rk in radii:
                for Zs, _ in heights:
                    d = np.sqrt((rho - rk) ** 2 + (z - Zs) ** 2)
                    dmin = min(dmin, float(d.min()))
        return dmin

    def J(self):
        radii, _, heights = self._geom()
        thr = self.threshold
        total = 0.0
        for c in self.curves:
            g = c.gamma()
            rho = np.hypot(g[:, 0], g[:, 1])
            z = g[:, 2]
            for rk in radii:
                for Zs, _ in heights:
                    d = np.sqrt((rho - rk) ** 2 + (z - Zs) ** 2)
                    total += float(np.sum(np.maximum(thr - d, 0.0) ** 2))
        return total

    @derivative_dec
    def dJ(self):
        radii, N, heights = self._geom()
        thr = self.threshold
        eps = 1e-30
        g_mu = np.zeros(self.fl.local_full_dof_size)
        curve_deriv = None
        for c in self.curves:
            g = c.gamma()
            x, y, z = g[:, 0], g[:, 1], g[:, 2]
            rho = np.hypot(x, y)
            inv_rho = 1.0 / np.maximum(rho, eps)
            seed = np.zeros_like(g)            # dJ/d(curve point), for the curve vjp
            for k, rk in enumerate(radii):
                for Zs, sZ in heights:
                    dr = rho - rk
                    dz = z - Zs
                    d = np.sqrt(dr ** 2 + dz ** 2)
                    viol = np.maximum(thr - d, 0.0)
                    # coef = (dJ/dd)/d = -2 viol / d   (zero where not violated)
                    coef = np.where(viol > 0.0, -2.0 * viol / np.maximum(d, eps), 0.0)
                    seed[:, 0] += coef * dr * x * inv_rho
                    seed[:, 1] += coef * dr * y * inv_rho
                    seed[:, 2] += coef * dz
                    # mu gradient: r_k (slot N+k) and Z (last slot). Z_s = sZ*Z,
                    # so d(dz)/dZ = -sZ; currents (slots 0..N-1) do not move d.
                    g_mu[N + k] += float(np.sum(coef * (-dr)))
                    g_mu[-1] += float(np.sum(coef * dz * (-sZ)))
            cd = c.dgamma_by_dcoeff_vjp(seed)
            curve_deriv = cd if curve_deriv is None else curve_deriv + cd
        return Derivative({self.fl: g_mu}) + curve_deriv

    return_fn_map = {'J': J, 'dJ': dJ}
