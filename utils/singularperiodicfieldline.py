import numpy as np
from scipy.linalg import lu
import jax.numpy as jnp
from jax import jacfwd

from simsopt._core import Optimizable
from simsopt.geo import CurveLength, CurveXYZFourierSymmetries

__all__ = ['SingularPeriodicFieldLine']


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
    modB = jnp.linalg.norm(B, axis=-1)
    dmodB = jnp.sum(B[:, :, None] * gradB, axis=1) / modB[:, None]
    A = (gradB * modB[:, None, None] - B[:, :, None] * dmodB[:, None, :]) / modB[:, None, None] ** 2
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


def _set_points_and_collect(field, pts):
    field.set_points(pts.reshape((-1, 3)))  # ADDED: reuse the same evaluation points for the base and fixed fields.
    return field.B().reshape((-1, 3)), field.dB_by_dX(), field.d2B_by_dXdX()  # ADDED: collect B, gradB, and gradgradB with no finite differences.


def _combined_field_data(field, fixed_fields, pts, mu):
    B0, G0, H0 = _set_points_and_collect(field, pts)
    B = B0.copy()  # ADDED: start from the original magnetic field.
    G = G0.copy()  # ADDED: start from the original field gradient.
    H = H0.copy()  # ADDED: start from the original field Hessian.
    fixed_B = []  # ADDED: cache each fixed field contribution for Jacobians.
    fixed_G = []  # ADDED: cache each fixed field gradient for Jacobians.
    fixed_H = []  # ADDED: cache each fixed field Hessian for curve-derivative chains.
    for muk, fixed in zip(mu, fixed_fields):  # ADDED: add the four mu-weighted fixed fields.
        Bk, Gk, Hk = _set_points_and_collect(fixed, pts)
        fixed_B.append(Bk)
        fixed_G.append(Gk)
        fixed_H.append(Hk)
        B = B + muk * Bk
        G = G + muk * Gk
        H = H + muk * Hk
    return B, G, H, fixed_B, fixed_G, fixed_H


def singular_field_line_residual(curve, length, field, fixed_fields, mu):
    pts = curve.gamma()
    B, _, _, fixed_B, _, _ = _combined_field_data(field, fixed_fields, pts, mu)  # ADDED: use the mu-augmented field in the periodic field-line equations.
    modB = np.linalg.norm(B, axis=1)  # ADDED: total |B| for the mu-augmented field.
    res = curve.gammadash() / length - B / modB[:, None]
    res = res.flatten()
    if not curve.stellsym:
        res_y = curve.gamma()[0, 1]
        dres_y = np.concatenate([curve.dgamma_by_dcoeff()[0, 1, :], [0], np.zeros(len(mu))])  # ADDED: extend the label equation Jacobian by the new mu columns.

    dres1_dcoeff = curve.dgammadash_by_dcoeff() / length

    idx = np.arange(3)
    diag = np.zeros((pts.shape[0], 3, 3))
    diag[:, idx, idx] = 1 / modB[:, None]
    dres2_dBmat = -B[:, None, :] * B[:, :, None] / modB[:, None, None] ** 3 + diag  # ADDED: derivative of B/|B| for the total field.

    dB_by_dX = field.dB_by_dX()
    dB_dc = np.einsum('ikl,ikm->ilm', dB_by_dX, curve.dgamma_by_dcoeff(), optimize=True)
    for muk, fixed in zip(mu, fixed_fields):  # ADDED: include the mu-weighted fixed-field geometry dependence.
        dB_dc += muk * np.einsum('ikl,ikm->ilm', fixed.dB_by_dX(), curve.dgamma_by_dcoeff(), optimize=True)
    dres2_dcoeff = np.einsum('ikl,ikm->ilm', dres2_dBmat, dB_dc, optimize=True)
    dres_dcoeff = dres1_dcoeff - dres2_dcoeff

    ncurve = dres1_dcoeff.shape[-1]
    dres_dcoeff = dres_dcoeff.reshape((-1, ncurve))
    dres_del = -curve.gammadash().reshape((-1, 1)) / length ** 2
    dres_dmu = np.column_stack([  # ADDED: exact Jacobian of the periodic-field-line equations with respect to the four mu unknowns.
        -np.einsum('ijk,ik->ij', dres2_dBmat, Bk).reshape((-1,)) for Bk in fixed_B
    ])
    dres = np.concatenate((dres_dcoeff, dres_del, dres_dmu), axis=-1)  # ADDED: append the length and mu columns.

    dres_dB = -dres2_dBmat.reshape((-1, 3))
    if not curve.stellsym:
        res = np.concatenate((res, [res_y]))
        dres = np.concatenate((dres, dres_y[None, :]), axis=0)
        dres_dB = np.concatenate([dres_dB, np.zeros((1, 3))], axis=0)
    return res, dres, dres_dB


class SingularPeriodicFieldLine(Optimizable):

    def __init__(self, biotsavart, curve, fixed_field, options=None):
        super().__init__(depends_on=[biotsavart])
        self.biotsavart = biotsavart
        self.curve = curve
        self.fixed_field = fixed_field  # ADDED: store the four fixed fields that are scaled by mu.
        self.need_to_run_code = True

        if options is None:
            options = {}
        if 'verbose' not in options:
            options['verbose'] = False
        if 'newton_tol' not in options:
            options['newton_tol'] = 1e-13
        if 'newton_maxiter' not in options:
            options['newton_maxiter'] = 40
        self.options = options

        nfp = curve.nfp  # ADDED: match the tangent-map toroidal period.
        N = 6 * curve.order + 1  # ADDED: use the same Chebyshev resolution as TangentMap.
        D, xh, wh = cheb(N, 0, 1.0 / nfp)  # ADDED: tangent map is evaluated on the Chebyshev grid.
        self.D = D  # ADDED: store the Chebyshev differentiation matrix.
        self.xh = xh  # ADDED: store the Chebyshev nodes.
        self.wh = wh  # ADDED: keep the quadrature weights for consistency with TangentMap.
        self.tm_curve = CurveXYZFourierSymmetries(self.xh, curve.order, curve.nfp, curve.stellsym, ntor=curve.ntor, dofs=curve.dofs)  # ADDED: monodromy is evaluated on a Chebyshev-grid copy of the curve.

        self.monodromy_matrix_jax = lambda B, gradB, L, gamma: monodromy_matrix_pure(B, gradB, L, gamma, self.D)  # ADDED: exact monodromy on the Chebyshev grid.
        self.monodromy_matrix_dB = lambda B, gradB, L, gamma: jacfwd(self.monodromy_matrix_jax, argnums=0)(B, gradB, L, gamma)  # ADDED: JAX derivative w.r.t. B.
        self.monodromy_matrix_dgradB = lambda B, gradB, L, gamma: jacfwd(self.monodromy_matrix_jax, argnums=1)(B, gradB, L, gamma)  # ADDED: JAX derivative w.r.t. gradB.
        self.monodromy_matrix_dL = lambda B, gradB, L, gamma: jacfwd(self.monodromy_matrix_jax, argnums=2)(B, gradB, L, gamma)  # ADDED: JAX derivative w.r.t. length.
        self.monodromy_matrix_dgamma = lambda B, gradB, L, gamma: jacfwd(self.monodromy_matrix_jax, argnums=3)(B, gradB, L, gamma)  # ADDED: JAX derivative w.r.t. the Chebyshev-grid curve points.

    def recompute_bell(self, parent=None):
        self.need_to_run_code = True

    def run_code(self, length, mu=None):
        if not self.need_to_run_code:
            return self.res
        if mu is None:
            mu = np.zeros(len(self.fixed_field))  # ADDED: default to zero fixed-field amplitudes.
        return self.solve_residual_equation_exactly_newton(length=length, mu=mu, tol=self.options['newton_tol'], maxiter=self.options['newton_maxiter'], verbose=self.options['verbose'])

    def get_stellsym_mask(self):
        order = self.curve.order
        stellsym = self.curve.stellsym
        if not stellsym:
            mask = np.ones((2 * order + 1) * 3 + 1, dtype=bool)
            return np.concatenate((mask, np.ones(4, dtype=bool)))  # ADDED: append the four monodromy equations.

        mask = np.ones((2 * order + 1, 3), dtype=bool)
        mask[0, 0] = False
        mask[order + 1:, :] = False
        mask = mask.flatten()
        return np.concatenate((mask, np.ones(4, dtype=bool)))  # ADDED: append the four monodromy equations.

    def residual_and_jacobian(self, length, mu):
        curve = self.curve
        tm_curve = self.tm_curve
        tm_curve.set_dofs(curve.get_dofs())  # ADDED: keep the Chebyshev-grid curve synchronized with the Newton iterate.

        r_fl, J_fl, _ = singular_field_line_residual(curve, length, self.biotsavart, self.fixed_field, mu)

        gamma_tm = tm_curve.gamma()  # ADDED: Chebyshev-grid curve used for the tangent map.
        B_tm, gradB_tm, gradgradB_tm, fixed_B_tm, fixed_gradB_tm, fixed_gradgradB_tm = _combined_field_data(self.biotsavart, self.fixed_field, gamma_tm, mu)  # ADDED: total field data on the Chebyshev grid.
        M = np.asarray(self.monodromy_matrix_jax(jnp.asarray(B_tm), jnp.asarray(gradB_tm), length, jnp.asarray(gamma_tm)))  # ADDED: monodromy matrix on the Chebyshev grid.
        r_mon = (M - np.eye(2)).reshape(4)  # ADDED: enforce M - I = 0 as four scalar equations.
        
        dM_dB = np.asarray(self.monodromy_matrix_dB(jnp.asarray(B_tm), jnp.asarray(gradB_tm), length, jnp.asarray(gamma_tm))).reshape((4,) + B_tm.shape)  # ADDED: exact derivative of monodromy w.r.t. B values.
        dM_dgradB = np.asarray(self.monodromy_matrix_dgradB(jnp.asarray(B_tm), jnp.asarray(gradB_tm), length, jnp.asarray(gamma_tm))).reshape((4,) + gradB_tm.shape)  # ADDED: exact derivative of monodromy w.r.t. gradB values.
        dM_dL = np.asarray(self.monodromy_matrix_dL(jnp.asarray(B_tm), jnp.asarray(gradB_tm), length, jnp.asarray(gamma_tm))).reshape(4, 1)  # ADDED: exact derivative of monodromy w.r.t. length.
        dM_dgamma_partial = np.asarray(self.monodromy_matrix_dgamma(jnp.asarray(B_tm), jnp.asarray(gradB_tm), length, jnp.asarray(gamma_tm))).reshape((4,) + gamma_tm.shape)  # ADDED: exact derivative of monodromy w.r.t. curve points.

        dgamma_da_tm = tm_curve.dgamma_by_dcoeff()  # ADDED: derivative of the Chebyshev-grid curve points w.r.t. curve coefficients.
        dB_da_tm = np.einsum('ikl,ikm->ilm', gradB_tm, dgamma_da_tm, optimize=True)  # ADDED: chain rule for B through the moving curve.
        dgradB_da_tm = np.einsum('ijkl,ikm->ijlm', gradgradB_tm, dgamma_da_tm, optimize=True)  # ADDED: chain rule for gradB through the moving curve.

        dM_dcoeff = np.einsum('aij,ijm->am', dM_dB, dB_da_tm, optimize=True)  # ADDED: monodromy derivative through B.
        dM_dcoeff += np.einsum('aijk,ijkm->am', dM_dgradB, dgradB_da_tm, optimize=True)  # ADDED: monodromy derivative through gradB.
        dM_dcoeff += np.einsum('aik,ikm->am', dM_dgamma_partial, dgamma_da_tm, optimize=True)  # ADDED: monodromy derivative through explicit gamma dependence.

        dM_dmu = np.zeros((4, len(mu)))  # ADDED: monodromy derivative with respect to the four mu unknowns.
        for k, (Bk, Gk) in enumerate(zip(fixed_B_tm, fixed_gradB_tm)):
            dM_dmu[:, k] = np.einsum('aij,ij->a', dM_dB, Bk, optimize=True) + np.einsum('aijk,ijk->a', dM_dgradB, Gk, optimize=True)  # ADDED: exact chain rule for each mu_k.

        J_mon = np.concatenate((dM_dcoeff, dM_dL, dM_dmu), axis=1)  # ADDED: append the new monodromy Jacobian block.
        r = np.concatenate((r_fl, r_mon))  # ADDED: full residual includes periodic-field-line and monodromy equations.
        J = np.vstack((J_fl, J_mon))  # ADDED: full Jacobian includes the four added equations.
        return r, J, M

    def solve_residual_equation_exactly_newton(self, tol=1e-9, maxiter=10, length=None, mu=None, verbose=False):
        tol = 1e-8
        verbose = True
        if not self.need_to_run_code:
            return self.res

        curve = self.curve
        mask = self.get_stellsym_mask()
        if length is None:
            length = CurveLength(self.curve).J()
        if mu is None:
            mu = np.zeros(len(self.fixed_field))  # ADDED: initialize the four extra unknowns if they are not supplied.
        mu = np.asarray(mu, dtype=float)  # ADDED: keep mu as a NumPy vector for the Newton solve.

        x = np.concatenate((curve.get_dofs(), [length], mu))  # ADDED: Newton unknowns now include length and the four mu values.
        i = 0
        
        #mask[[-2, -1]] = False
        mask[[-4]] = False
        r, J, M = self.residual_and_jacobian(length, mu)
        
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


        b = r[mask]
        Jm = J[mask]
        Jm = Jm[:, mask]
        print(np.linalg.norm(b), np.linalg.cond(Jm),M)
        
        norm = 1e6
        while i < maxiter:
            norm = np.linalg.norm(b)
            if norm <= tol:
                break
            dx = np.linalg.solve(Jm, b)
            dx += np.linalg.solve(Jm, b - Jm @ dx)
            x[mask] -= dx
            curve.set_dofs(x[:-5])  # ADDED: update only the curve coefficients.
            length = x[-5]  # ADDED: recover the updated length.
            mu = x[-4:]  # ADDED: recover the updated four mu values.
            i += 1
            r, J, M = self.residual_and_jacobian(length, mu)
            b = r[mask]
            Jm = J[mask]
            Jm = Jm[:, mask]
            print(np.linalg.norm(b), np.linalg.cond(Jm),M)

        P, L, U = lu(Jm)
        res = {
            'residual': r,
            'jacobian': Jm,
            'iter': i,
            'success': norm <= tol,
            'length': length,
            'mu': mu,  # ADDED: store the solved fixed-field amplitudes.
            'PLU': (P, L, U),
            'mask': mask,
            'monodromy_matrix': M,  # ADDED: expose the final 2x2 monodromy matrix.
        }
        self.monodromy_matrix = M  # ADDED: make the monodromy matrix directly available on the object.
        if verbose:
            print(f"NEWTON solve - {res['success']}  iter={res['iter']}, length={res['length']:.8f}, mu={res['mu']}, ||residual||_inf = {np.linalg.norm(res['residual'], ord=np.inf):.3e}, cond(J) = {np.linalg.cond(Jm):.3e}", flush=True)
        

        self.res = res
        self.need_to_run_code = False
        return res
