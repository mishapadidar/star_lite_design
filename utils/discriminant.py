"""
Reduced return-map quadratic jet, its snowflake diagnostics, and the
snowflake-discriminant objective for a SingularPeriodicFieldline.

This module owns everything SECOND-order about the reduced return map (the
quadratic jet K, its snowflake leg directions and discriminant Delta), plus the
Optimizable that controls Delta. tangent_map.py keeps only the LINEAR objects
(tangent map, monodromy, iota, elongation) and the shared frame builder
reduced_frame_pure.

Pure functions
--------------
hess_b_pure, second_var_residual_pure, quadratic_jet_pure,
quadratic_jet_matrix_pure : the return-map quadratic jet K (with the return-time
    correction always applied; see tangent_map_jet.pdf).
snowflake_discriminant_pure : Delta of the binary cubic f(v)=v x K[v,v]; its SIGN
    decides the number of snowflake legs (Delta>0 -> 6, Delta<0 -> 2, Delta=0 ->
    bifurcation).
snowflake_angles_from_jet2 : the leg-direction angles (roots of f).
tangent_map_jet2 : evaluate K from a TangentMap instance (any axis).

Optimizable
-----------
SnowflakeDiscriminant : J()=Delta, dJ()=dDelta/d(coil dofs) through a
    SingularPeriodicFieldline (analogue of Monodromy / AxisIota, but wired through
    the singular polish's PLU/vjp adjoint, carrying the extra gradgradB pieces).
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad, jacfwd

from numpy.random import Generator, PCG64DXSM
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.objectives import forward_backward
from simsopt.geo import GaussianSampler, PerturbationSample, CurveXYZFourierSymmetries

from .tangent_map import reduced_frame_pure, tangent_map_pure, A_pure, cheb
from .singularperiodicfieldline import (
    _MU0_4PI,
    _B_aux, _dB_aux_by_dX, _d2B_aux_by_dXdX,
    _dB_aux_by_dmu, _dgradB_aux_by_dmu, _dgradgradB_aux_by_dmu,
    _d3B_aux_by_dXdXdX,
)


# =============================================================================
# Second variation and the quadratic jet
# =============================================================================
def hess_b_pure(B, gradB, gradgradB):
    """
    Hessian of b = B / |B|.

    Inputs
    ------
    B         : (N,3)
    gradB     : (N,3,3)      gradB[:,i,j] = d B_i / d x_j
    gradgradB : (N,3,3,3)    gradgradB[:,i,j,k] = d^2 B_i / d x_j d x_k

    Returns
    -------
    Hb : (N,3,3,3)           Hb[:,i,j,k] = d^2 b_i / d x_j d x_k
    """
    modB = jnp.linalg.norm(B, axis=-1)                              # (N,)
    g = jnp.einsum('ni,nij->nj', B, gradB) / modB[:, None]          # d|B|/dx_j

    # Hessian of |B|
    hm = (
        jnp.einsum('nik,nij->nkj', gradB, gradB)
        + jnp.einsum('ni,nijk->njk', B, gradgradB)
        - g[:, :, None] * g[:, None, :]
    ) / modB[:, None, None]

    m1 = modB[:, None, None, None]
    Hb = gradgradB / m1
    Hb -= gradB[:, :, :, None] * g[:, None, None, :] / (m1**2)      # - dB_i/dx_j * g_k / m^2
    Hb -= gradB[:, :, None, :] * g[:, None, :, None] / (m1**2)      # - dB_i/dx_k * g_j / m^2
    Hb -= B[:, :, None, None] * hm[:, None, :, :] / (m1**2)         # - B_i * h_{jk} / m^2
    Hb += 2.0 * B[:, :, None, None] * g[:, None, :, None] * g[:, None, None, :] / (m1**3)
    return Hb


def second_var_residual_pure(Q, B, gradB, gradgradB, L, T1, T2, D):
    """
    Residual for the second variational equation:
        Q' / L - A Q - Hb[T1,T2] = 0,
    with Q(0)=0.
    """
    A = A_pure(B, gradB)
    Hb = hess_b_pure(B, gradB, gradgradB)

    AQ = jnp.einsum('nij,nj->ni', A, Q)
    src = jnp.einsum('nijk,nj,nk->ni', Hb, T1, T2)
    Qprime = jnp.matmul(D, Q)

    residual = Qprime / L - AQ - src
    ic0 = Q[0]  # Q(0)=0
    return jnp.concatenate((ic0[None, :], residual[1:]), axis=0)


def quadratic_jet_pure(B, gradB, gradgradB, L, gamma, D, frame='NB'):
    """
    Quadratic jet of the reduced 2D Poincare RETURN map, in the chosen frame.

    `frame` selects the 2x2 frame the jet is projected into: 'NB' (default) for the
    moving normal-binormal frame of the axis, or 'RZ' for the cylindrical (R,Z)
    Poincare-section frame (same embedding/projection as monodromy_pure).

    The jet is the true return-map jet: to the projected second variation of the
    flow, Pt . w_ab (w_ab = d^2 F / de_a de_b, F = flow over one field period), we add
    the RETURN-TIME correction, since each field line hits the section at its own
    parameter time tau, not the fixed t_p. The linear part (monodromy) is unaffected;
    the quadratic part gets
        Delta_K_ab = c_a Pt(Dv' g_b) + c_b Pt(Dv' g_a) + c_a c_b Pt v'dot,
        g_a = Phi(t) P0[:,a],  v' = L b,  Dv' = L A,  v'dot = L^2 A b,
        c_a = -<n, g_a>/<n, v'>,   n the section normal (fT for NB, e_phi for RZ).
    The correction is applied per collocation point, so K[s] is the return-map jet at
    the section through node s. See tangent_map_jet.pdf for the derivation.

    Returns
    -------
    K : (N,2,2,2)
        K[s, :, a, b] is the return-map quadratic jet at collocation point s,
        projected into the local `frame`, with initial coordinates taken in the
        `frame` at s=0. The final (full-period) return-map jet is K[-1].
    """
    P0, Pt, n = reduced_frame_pure(B, gamma, D, frame)   # (3,2), (N,2,3), (N,3)

    # Full 3x3 tangent map in Cartesian coordinates
    Tfull = tangent_map_pure(B, gradB, L, D)                             # (N,3,3)

    # per-point pieces of the return-time correction (all at collocation point t)
    A = A_pure(B, gradB)                                            # (N,3,3) = grad b
    bhat = B / jnp.linalg.norm(B, axis=-1)[:, None]                 # (N,3)
    vprime = L * bhat                                               # v' = L b
    vprime_dot = L * jnp.einsum('nij,nj->ni', A, vprime)           # Dv' . v' = L^2 A b
    nv = jnp.sum(n * vprime, axis=-1)                              # <n, v'>  (N,)

    # Linear operator for Q is the same as for T
    Npts = B.shape[0]
    Q0 = jnp.zeros_like(B)
    Lop = jacfwd(second_var_residual_pure, argnums=0)(
        Q0, B, gradB, gradgradB, L,
        jnp.zeros_like(B), jnp.zeros_like(B), D
    ).reshape((3*Npts, 3*Npts))

    K = jnp.zeros((Npts, 2, 2, 2))

    for a in range(2):
        for b in range(2):
            ia = P0[:, a]   # initial Cartesian direction for reduced coord a
            ib = P0[:, b]   # initial Cartesian direction for reduced coord b

            Ta = jnp.einsum('nij,j->ni', Tfull, ia)   # first variation g_a = Phi P0[:,a]
            Tb = jnp.einsum('nij,j->ni', Tfull, ib)   # first variation g_b = Phi P0[:,b]

            rhs = -second_var_residual_pure(Q0, B, gradB, gradgradB, L, Ta, Tb, D).ravel()
            Qab = jnp.linalg.solve(Lop, rhs).reshape((Npts, 3))   # w_ab (N,3)

            # return-time correction Delta_K (Cartesian), before projection
            ca = -jnp.sum(n * Ta, axis=-1) / nv                       # (N,)
            cb = -jnp.sum(n * Tb, axis=-1) / nv                       # (N,)
            DvpTa = L * jnp.einsum('nij,nj->ni', A, Ta)               # Dv' g_a
            DvpTb = L * jnp.einsum('nij,nj->ni', A, Tb)               # Dv' g_b
            Qab = Qab + (ca[:, None]*DvpTb + cb[:, None]*DvpTa
                         + (ca*cb)[:, None]*vprime_dot)

            # project displacement into local frame
            Kab = jnp.einsum('nij,nj->ni', Pt, Qab)   # (N,2)
            K = K.at[:, :, a, b].set(Kab)

    return K


def quadratic_jet_matrix_pure(B, gradB, gradgradB, L, gamma, D, frame='NB'):
    return quadratic_jet_pure(B, gradB, gradgradB, L, gamma, D, frame)[-1]


def snowflake_discriminant_pure(K):
    """Discriminant Delta of the binary cubic  f(v) = v x K[v,v]  whose real roots are
    the snowflake leg directions. K is the (2,2,2) return-map quadratic jet (K[i,a,b],
    symmetric in a,b). Writing f = A v0^3 + B v0^2 v1 + C v0 v1^2 + D v1^3,

        A = K[1,0,0],  B = 2 K[1,0,1] - K[0,0,0],
        C = K[1,1,1] - 2 K[0,0,1],  D = -K[0,1,1],
        Delta = 18 A B C D - 4 B^3 D + B^2 C^2 - 4 A C^3 - 27 A^2 D^2.

    The SIGN of Delta decides the number of legs (f is pi-antiperiodic, so roots come
    in antipodal pairs):
        Delta > 0 : three real leg-lines  -> 6 legs,
        Delta < 0 : one real leg-line     -> 2 legs,
        Delta = 0 : a repeated root       -> bifurcation (legs merging).
    The sign is invariant under real linear changes of the 2D frame (Delta scales by
    (det S)^6 > 0), so it is a property of the map, not of the NB/RZ choice. Only
    meaningful in the degenerate (M ~ identity) snowflake regime, where the quadratic
    jet governs the local structure."""
    A = K[1, 0, 0]
    B = 2 * K[1, 0, 1] - K[0, 0, 0]
    C = K[1, 1, 1] - 2 * K[0, 0, 1]
    D = -K[0, 1, 1]
    return 18*A*B*C*D - 4*B**3*D + B**2*C**2 - 4*A*C**3 - 27*A**2*D**2


def snowflake_angles_from_jet2(K, ntheta=512, xtol=1e-12, tangent_tol=1e-6):
    """
    Roots of f(theta) = v x K(v,v) with v = (cos theta, sin theta) -- snowflake leg
    directions -- for a (2,2,2) return-map quadratic jet K (e.g. tangent_map_jet2(tm)).

    Refines each bracketed sign change with brentq (machine-precision roots), and also
    detects tangent (double) zeros that lie at sign-preserving local minima of |f|,
    which the sign-change method misses near bifurcations (Delta = 0).

    Returns
    -------
    np.ndarray of root angles in [0, 2pi), sorted ascending.
    """
    from scipy.optimize import brentq, minimize_scalar

    K = np.asarray(K)

    def f(t):
        v = np.array([np.cos(t), np.sin(t)])
        q = np.einsum('iab,a,b->i', K, v, v)
        return v[0]*q[1] - v[1]*q[0]

    th = np.linspace(0.0, 2*np.pi, ntheta, endpoint=False)
    vals = np.array([f(t) for t in th])
    scale = float(np.max(np.abs(vals))) + 1e-300

    roots = []

    def _add(r):
        r = r % (2*np.pi)
        for rr in roots:
            d = abs(r - rr)
            if min(d, 2*np.pi - d) < 1e-6:
                return
        roots.append(r)

    # --- sign-change roots: refine with brentq on each bracket ---
    for i in range(ntheta):
        j = (i + 1) % ntheta
        a = th[i]
        b = th[j] if j != 0 else 2*np.pi
        if vals[i] == 0.0:
            _add(a)
            continue
        if vals[i] * vals[j] < 0.0:
            _add(brentq(f, a, b, xtol=xtol, rtol=1e-14))

    # --- tangent roots: local minima of |f| that are essentially zero ---
    av = np.abs(vals)
    for i in range(ntheta):
        im, ip = (i - 1) % ntheta, (i + 1) % ntheta
        if av[i] < av[im] and av[i] < av[ip] and av[i] / scale < 1e-2:
            a = th[im] if im < i else th[im] - 2*np.pi
            c = th[ip] if ip > i else th[ip] + 2*np.pi
            res = minimize_scalar(lambda t: abs(f(t)), bounds=(a, c),
                                  method='bounded', options={'xatol': xtol})
            if abs(f(res.x)) / scale < tangent_tol:
                _add(res.x)

    return np.sort(np.asarray(roots))


def tangent_map_jet2(tm, frame='RZ'):
    """Return-map quadratic jet matrix K[-1] for a TangentMap instance ``tm`` (any
    axis), in the requested frame ('RZ' default, or 'NB'). Pairs with
    snowflake_angles_from_jet2 / snowflake_discriminant_pure for the leg directions /
    the leg count. Evaluation only (no coil derivative -- use SnowflakeDiscriminant
    for that)."""
    axis = tm.axis
    curve = tm.curve
    biotsavart = tm.biotsavart
    if axis.need_to_run_code:
        axis.run_code(axis.res['length'])
    biotsavart.set_points(curve.gamma())
    B = biotsavart.B()
    gradB = biotsavart.dB_by_dX()
    gradgradB = biotsavart.d2B_by_dXdX()
    L = axis.res['length']
    gamma = curve.gamma()
    return np.asarray(quadratic_jet_matrix_pure(B, gradB, gradgradB, L, gamma, tm.D, frame))


# =============================================================================
# Optimizable: control the snowflake discriminant w.r.t. the coil dofs
# =============================================================================
class SnowflakeDiscriminant(Optimizable):
    r"""Discriminant Delta of the reduced return-map quadratic jet, controllable
    w.r.t. the modular-coil dofs through a :class:`SingularPeriodicFieldline`.

    Args:
        fieldline (SingularPeriodicFieldline): converged (square) singular polish;
            supplies the tangent-map grid curve ``curve_tm``, the Chebyshev matrix
            ``D``, the length and the aux-coil ``mu`` state, plus the adjoint
            plumbing ``res['PLU']`` / ``res['vjp']``.
        biotsavart (BiotSavart): the modular field (same object driving the solve).
        frame (str): 2D frame the jet/discriminant are computed in, 'RZ' (default,
            the physical (R, Z) cross-section) or 'NB'. The SIGN of Delta -- hence
            the leg count -- is frame-independent, but the value/gradient are not.
        phi0 (float): toroidal fraction in [0,1) of the Poincare section where the
            return-map jet/discriminant are measured (default 0 = the field-line
            curve's origin). The sign of Delta is phi0-invariant; the value is not.

    ``J()`` returns Delta; ``dJ()`` its gradient w.r.t. the modular-coil dofs and
    the free (independent) mu dofs.
    """

    def __init__(self, fieldline, biotsavart, frame='RZ', phi0=0.0):
        self.fieldline = fieldline
        self.biotsavart = biotsavart
        self.frame = frame
        self.phi0 = float(phi0)
        self.stellsym_aux = fieldline.stellsym_aux
        D = np.asarray(fieldline.D)
        # Poincare section: by default (phi0=0) the return-map jet is measured on the field
        # line's own per-period grid (fl.curve_tm). For phi0 in [0,1) the SAME per-period
        # Chebyshev grid is started at toroidal fraction phi0 along the closed field line: the
        # curve is periodic and the Chebyshev matrix D is translation-invariant, so only the
        # node POSITIONS shift (mod 1, array order preserved) while D and the length are
        # unchanged. The jet/discriminant then correspond to the section at phi0.
        if self.phi0 == 0.0:
            self._sec_curve = fieldline.curve_tm
        else:
            c = fieldline.curve
            self._sec_curve = CurveXYZFourierSymmetries(
                np.mod(np.asarray(fieldline.curve_tm.quadpoints) + self.phi0, 1.0),
                c.order, c.nfp, c.stellsym, ntor=c.ntor, dofs=c.dofs)

        # Objective as a jax function of the field data in SIMSOPT gradient
        # convention (gradB_ss[i,k,j] = dB_j/dx_k, gradgradB_ss[i,k1,k2,j] =
        # d^2 B_j/dx_k1 dx_k2). We transpose to the component-first convention
        # quadratic_jet_pure uses inside, so jax hands back cotangents already in
        # simsopt convention -- ready for B_and_dB_and_d2B_vjp and the field-motion
        # einsums. (On-axis the field is curl-free so gradB/gradgradB are symmetric
        # and the transpose is a no-op numerically; it is kept for clarity.)
        def _jetmat(B, gradB_ss, gradgradB_ss, L, gamma):
            gradB_tm = jnp.transpose(gradB_ss, (0, 2, 1))            # -> [i,j,k]=dB_i/dx_k
            gradgradB_tm = jnp.transpose(gradgradB_ss, (0, 3, 1, 2))  # -> [i,j,k1,k2]
            return quadratic_jet_matrix_pure(B, gradB_tm, gradgradB_tm, L, gamma, D, frame)

        def _disc(B, gradB_ss, gradgradB_ss, L, gamma):
            return snowflake_discriminant_pure(_jetmat(B, gradB_ss, gradgradB_ss, L, gamma))

        self._jetmat = jit(_jetmat)
        self._disc = jit(_disc)
        self._grads = jit(grad(_disc, argnums=(0, 1, 2, 3, 4)))
        # partials of  sum(cK * K)  w.r.t. the field data, for a general jet cotangent cK
        # (used by jet_cotangent_vjp; e.g. the unfolding-C gradient).
        self._jet_grads = jit(jax.grad(
            lambda B, gB, ggB, L, g, cK: jnp.sum(cK * _jetmat(B, gB, ggB, L, g)),
            argnums=(0, 1, 2, 3, 4)))

        self.recompute_bell()
        super().__init__(depends_on=[fieldline])

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def value(self):
        """Discriminant value Delta only (no coil gradient) -- cheap enough for a
        per-iteration diagnostic / an escalation violation check, and it does not
        require a square solve (unlike dJ)."""
        fl = self.fieldline
        if fl.need_to_run_code:
            fl.run_code(fl.res['length'])
        res = fl.res
        pts = self._sec_curve.gamma()
        B, gradB, gradgradB = self._field_data(pts, np.asarray(res['mu']))
        Bj, gBj, ggBj, gj = map(jnp.asarray, (B, gradB, gradgradB, pts))
        return float(self._disc(Bj, gBj, ggBj, float(res['length']), gj))

    def _field_data(self, pts, mu):
        """Total field B, gradB (simsopt conv), gradgradB (simsopt conv) at pts,
        with BiotSavart's points left set to pts."""
        stell = self.stellsym_aux
        bs = self.biotsavart
        bs.set_points(pts.reshape((-1, 3)))
        B = bs.B() + np.asarray(_B_aux(pts, mu, stellsym=stell))
        gradB = bs.dB_by_dX() + np.asarray(_dB_aux_by_dX(pts, mu, stellsym=stell))
        gradgradB = bs.d2B_by_dXdX() + np.asarray(_d2B_aux_by_dXdX(pts, mu, stellsym=stell))
        return B, gradB, gradgradB

    def _partials_to_coils(self, dJ_dB, dJ_dgB, dJ_dggB, dJ_dL, dJ_dgamma):
        """Push jax field-partials (dJ/dB, dJ/dgradB, dJ/dgradgradB, dJ/dL, dJ/dgamma,
        all SIMSOPT convention) to a coil-dof Derivative: direct field vjp
        (B_and_dB_and_d2B_vjp) + curve/length/mu state adjoint (res['PLU']/vjp) + free-mu.
        Shared by dJ (discriminant) and jet_cotangent_vjp. Requires a SQUARE solve."""
        fl = self.fieldline
        res = fl.res
        if 'PLU' not in res or 'vjp' not in res:
            raise RuntimeError(
                "needs a SQUARE converged solve (res['PLU']/res['vjp']); fix the right "
                "number of mu dofs and re-solve (see SingularPeriodicFieldline.num_independent_mu).")
        bs = self.biotsavart
        mu = np.asarray(res['mu']); nmu = mu.size; stell = self.stellsym_aux
        pts = self._sec_curve.gamma()
        _, gradB, gradgradB = self._field_data(pts, mu)             # sets bs points to pts
        d3B = bs.d3B_by_dXdXdX() + np.asarray(_d3B_aux_by_dXdXdX(pts, mu, stellsym=stell))
        coil_direct = sum(bs.B_and_dB_and_d2B_vjp(dJ_dB, dJ_dgB, dJ_dggB))

        dgamma_dc = self._sec_curve.dgamma_by_dcoeff()             # (N,3,n_c)
        res_curve = (np.einsum('ij,ikj,ikm->m', dJ_dB, gradB, dgamma_dc, optimize=True)
                     + np.einsum('ikj,ikpj,ipm->m', dJ_dgB, gradgradB, dgamma_dc, optimize=True)
                     + np.einsum('iklj,iklpj,ipm->m', dJ_dggB, d3B, dgamma_dc, optimize=True)
                     + np.einsum('ij,ijm->m', dJ_dgamma, dgamma_dc, optimize=True))
        dB_aux_dmu = np.asarray(_dB_aux_by_dmu(pts, mu, stellsym=stell))
        dgB_aux_dmu = np.asarray(_dgradB_aux_by_dmu(pts, mu, stellsym=stell))
        dggB_aux_dmu = np.asarray(_dgradgradB_aux_by_dmu(pts, mu, stellsym=stell))
        dJ_dmu = (np.einsum('ij,ijl->l', dJ_dB, dB_aux_dmu, optimize=True)
                  + np.einsum('ikj,ikjl->l', dJ_dgB, dgB_aux_dmu, optimize=True)
                  + np.einsum('iklj,ikljq->q', dJ_dggB, dggB_aux_dmu, optimize=True))

        col_mask = np.asarray(res['mask'], dtype=bool)             # over [curve, length, mu]
        full_dJ_dx = np.concatenate([res_curve, [float(dJ_dL)], dJ_dmu])
        P, L, U = res['PLU']
        adj = forward_backward(P, L, U, full_dJ_dx[col_mask])
        adj_term = res['vjp'](adj, fl.biotsavart, fl)
        mu_free = ~col_mask[-nmu:]
        mu_direct = np.zeros(nmu)
        mu_direct[mu_free] = dJ_dmu[mu_free]
        return coil_direct + Derivative({fl: mu_direct}) - adj_term

    def compute(self):
        fl = self.fieldline
        if fl.need_to_run_code:
            fl.run_code(fl.res['length'])
        res = fl.res
        pts = self._sec_curve.gamma(); mu = np.asarray(res['mu']); length = float(res['length'])
        B, gradB, gradgradB = self._field_data(pts, mu)
        Bj, gBj, ggBj, gj = map(jnp.asarray, (B, gradB, gradgradB, pts))
        self._J = float(self._disc(Bj, gBj, ggBj, length, gj))
        parts = [np.asarray(a) for a in self._grads(Bj, gBj, ggBj, length, gj)]
        self._dJ = self._partials_to_coils(*parts)

    def jet(self):
        """Return-map quadratic jet matrix K = K[-1] (2,2,2) in `frame` (value only, no
        gradient, no square-solve requirement)."""
        fl = self.fieldline
        if fl.need_to_run_code:
            fl.run_code(fl.res['length'])
        res = fl.res
        pts = self._sec_curve.gamma()
        B, gradB, gradgradB = self._field_data(pts, np.asarray(res['mu']))
        return np.asarray(self._jetmat(jnp.asarray(B), jnp.asarray(gradB),
                                       jnp.asarray(gradgradB), float(res['length']), jnp.asarray(pts)))

    def jet_cotangent_vjp(self, cK):
        """Coil-dof gradient of  sum(cK * K[-1])  for a (2,2,2) cotangent cK on the jet, as a
        simsopt Derivative (same state adjoint as dJ, with cK-weighted field partials).
        Requires a SQUARE solve. Building block of the unfolding-C gradient."""
        fl = self.fieldline
        if fl.need_to_run_code:
            fl.run_code(fl.res['length'])
        res = fl.res
        pts = self._sec_curve.gamma(); mu = np.asarray(res['mu']); length = float(res['length'])
        B, gradB, gradgradB = self._field_data(pts, mu)
        parts = [np.asarray(a) for a in self._jet_grads(
            jnp.asarray(B), jnp.asarray(gradB), jnp.asarray(gradgradB), length,
            jnp.asarray(pts), jnp.asarray(cK))]
        return self._partials_to_coils(*parts)

    return_fn_map = {'J': J, 'dJ': dJ}


# =============================================================================
# Coil Gaussian-process perturbation field dB/dsigma (JAX Biot-Savart)
# =============================================================================
class CoilPerturbationField(Optimizable):
    r"""First-order response dB/dsigma of the MODULAR coil field to one fixed
    Gaussian-process fabrication-error realization, scaled by an amplitude sigma
    (exactly the perturbation of array/mk_perturb_manifold.py / mk_xpoint_distance.py:
    each coil curve is displaced by sigma * (unit GP sample)).

    A JAX filament Biot-Savart is built directly from the coils' quadrature points
    gamma, gamma' (read live from the SIMSOPT coils, so it tracks the coil dofs).
    The perturbed coil is  gamma + sigma * d,  gamma' + sigma * d'  with d = sample[0],
    d' = sample[1] the fixed unit displacement and its theta-derivative. Then
        delta_B := dB/dsigma|_{sigma=0}
    is obtained by JAX autodiff in sigma -- ANALYTIC (no finite difference in sigma).
    Because gamma, gamma', I enter the JAX field, the coil-dof gradient of delta_B is
    also analytic: `dB_by_dsigma_vjp` autodiffs (v . delta_B) w.r.t. gamma/gamma'/I and
    chains to the coil dofs through the SIMSOPT curve/current vjps.

    The unperturbed JAX field matches SIMSOPT's modular BiotSavart(coils).B() to
    machine precision, and delta_B matches a central finite difference of
    perturbed_field(sigma) to the FD floor (both verified).

    Args:
        biotsavart (BiotSavart): the MODULAR field; its `.coils` are perturbed. (The
            aux field, if any, is NOT perturbed, so dB/dsigma is the modular response.)
        length_scale (float): GP correlation length L (GaussianSampler arg).
        seed (int): RNG seed selecting the fixed unit realization.
        n_derivs (int): derivative order of the GP sample (>=1; needs sample[0], [1]).
    """

    def __init__(self, biotsavart, length_scale=0.5, seed=0, n_derivs=2):
        self.biotsavart = biotsavart
        self.coils = list(biotsavart.coils)
        # one fixed unit-amplitude GP realization per coil (never resampled); every
        # sigma scales it -- matches mk_perturb_manifold / mk_xpoint_distance.
        sampler = GaussianSampler(self.coils[0].curve.quadpoints, 1.0, length_scale,
                                  n_derivs=n_derivs)
        rg = Generator(PCG64DXSM(seed))
        self._unit_samples = [PerturbationSample(sampler, randomgen=rg) for _ in self.coils]
        # displacement d = sample[0] and its theta-derivative d' = sample[1] (fixed).
        d = jnp.array([np.asarray(us._sample[0]) for us in self._unit_samples])   # (ncoil,nq,3)
        dprime = jnp.array([np.asarray(us._sample[1]) for us in self._unit_samples])
        nq = d.shape[1]

        # JAX filament Biot-Savart of the (perturbed) modular coils, matching SIMSOPT:
        #   B(x) = (mu0/4pi) sum_k I_k (1/nq) sum_j gd_kj x (x - g_kj) / |x - g_kj|^3.
        def _Bfield(points, gammas, gammadashs, currents, sigma):
            g = gammas + sigma * d
            gd = gammadashs + sigma * dprime
            R = points[:, None, None, :] - g[None, :, :, :]            # (Npts,ncoil,nq,3)
            invR3 = jnp.sum(R * R, axis=-1) ** (-1.5)                  # (Npts,ncoil,nq)
            cross = jnp.cross(gd[None, :, :, :], R, axis=-1)           # (Npts,ncoil,nq,3)
            return _MU0_4PI * jnp.sum(currents[None, :, None, None] * cross * invR3[..., None],
                                     axis=(1, 2)) / nq                 # (Npts,3)
        self._Bfield = jit(_Bfield)
        self._dBdsigma = jit(lambda p, g, gd, I: jax.jacfwd(lambda s: _Bfield(p, g, gd, I, s))(0.0))
        # gradient of (v . dB/dsigma) w.r.t. (gamma, gamma', current) -- for the coil-dof vjp.
        self._dBdsigma_grad = jit(jax.grad(
            lambda p, g, gd, I, v: jnp.sum(v * jax.jacfwd(lambda s: _Bfield(p, g, gd, I, s))(0.0)),
            argnums=(1, 2, 3)))

        # spatial gradient of delta_B (= d(gradB)/dsigma), SIMSOPT convention [i,k,j]=dB_j/dx_k.
        def _gradB(points, gammas, gammadashs, currents, sigma):
            Jac = jax.vmap(lambda p: jax.jacfwd(
                lambda q: _Bfield(q[None, :], gammas, gammadashs, currents, sigma)[0])(p))(points)
            return jnp.transpose(Jac, (0, 2, 1))          # [j,k]=dB_j/dx_k -> [k,j]
        self._dgradBdsigma = jit(lambda p, g, gd, I: jax.jacfwd(lambda s: _gradB(p, g, gd, I, s))(0.0))

        super().__init__(depends_on=[biotsavart])

    def _coil_data(self):
        """Live coil quadrature data (moves with the coil dofs), SIMSOPT convention."""
        gammas = jnp.array([c.curve.gamma() for c in self.coils])
        gammadashs = jnp.array([c.curve.gammadash() for c in self.coils])
        currents = jnp.array([c.current.get_value() for c in self.coils])
        return gammas, gammadashs, currents

    def dB_by_dsigma(self, points):
        """delta_B = dB/dsigma at `points` (shape (Npts,3)) -> (Npts,3), per unit sigma."""
        g, gd, I = self._coil_data()
        return np.asarray(self._dBdsigma(jnp.asarray(points), g, gd, I))

    def dgradB_by_dsigma(self, points):
        """Spatial gradient of delta_B: d(gradB)/dsigma at `points` -> (Npts,3,3) in SIMSOPT
        convention [i,k,j]=dB_j/dx_k. Needed for the grid-motion term of a D_sigma adjoint."""
        g, gd, I = self._coil_data()
        return np.asarray(self._dgradBdsigma(jnp.asarray(points), g, gd, I))

    def dB_by_dsigma_vjp(self, points, v):
        """Coil-dof gradient  v . d(delta_B)/d(coil dofs)  as a simsopt Derivative, for a
        cotangent `v` (shape (Npts,3)) on delta_B. Analytic: JAX for d/d(gamma,gamma',I),
        SIMSOPT curve/current vjps for the chain to the coil dofs."""
        g, gd, I = self._coil_data()
        vg, vgd, vI = self._dBdsigma_grad(jnp.asarray(points), g, gd, I, jnp.asarray(v))
        vg, vgd, vI = np.asarray(vg), np.asarray(vgd), np.asarray(vI)
        deriv = None
        for k, c in enumerate(self.coils):
            dk = (c.curve.dgamma_by_dcoeff_vjp(vg[k])
                  + c.curve.dgammadash_by_dcoeff_vjp(vgd[k])
                  + c.current.vjp(np.array([vI[k]])))
            deriv = dk if deriv is None else deriv + dk
        return deriv


# =============================================================================
# D_sigma: first-order response of the reduced return-map residual to the
# perturbation, by Chebyshev collocation on the FULL torus.
# =============================================================================
def dsigma_pure(B, gradB_ss, dB, L, gamma, D, frame='RZ'):
    """First-order response D_sigma (2-vector) of the reduced return-map residual to a
    field perturbation delta_B, by Chebyshev collocation -- the SAME linear operator as
    the tangent map (tangent_map_pure), with an inhomogeneous source and eta(0)=0.

    Inputs (all on the FULL-torus grid): B (N,3); gradB_ss (N,3,3) in SIMSOPT convention
    [i,k,j]=dB_j/dx_k; dB = delta_B (N,3); L the (full) length; gamma the grid points;
    D the Chebyshev matrix on [0,1); frame 'RZ' or 'NB'. Solves

        eta'/L - A eta = (I - b b^T) delta_B / |B|,   eta(0) = 0,   A = grad(B/|B|),

    (delta of the flow's unit-field forcing), then projects D_sigma = Pt[-1] eta[-1].
    First order in sigma, so Pt kills the along-flow part -- no return-time correction
    (as for the linear monodromy)."""
    gradB = jnp.transpose(gradB_ss, (0, 2, 1))       # -> component-first for A_pure
    N = B.shape[0]
    A = A_pure(B, gradB)
    modB = jnp.linalg.norm(B, axis=-1)
    bhat = B / modB[:, None]
    source = (dB - bhat * jnp.sum(bhat * dB, axis=-1)[:, None]) / modB[:, None]

    def resid(eta):
        r = jnp.matmul(D, eta) / L - jnp.einsum('nij,nj->ni', A, eta) - source
        return jnp.concatenate([eta[0][None, :], r[1:]], axis=0)   # first row = IC eta(0)=0

    z = jnp.zeros_like(B)
    Lop = jax.jacfwd(resid)(z).reshape((3 * N, 3 * N))   # = (1/L) D-kron-I - blockdiag(A), IC row
    rhs = -resid(z).ravel()                              # = source on interior rows, 0 on IC row
    eta = jnp.linalg.solve(Lop, rhs).reshape((N, 3))
    _, Pt, _ = reduced_frame_pure(B, gamma, D, frame)
    return Pt[-1] @ eta[-1]


class DSigma(Optimizable):
    r"""First-order response D_sigma (2x1) of the reduced snowflake return-map residual
    to a coil GP perturbation (from a :class:`CoilPerturbationField`), computed by
    Chebyshev collocation on the FULL torus [0,1) -- the perturbed field is nfp=1 /
    stellsym=False, so the genuine return map is over the whole torus.

    It is the D_sigma of the unfolding shape equation  0.5 K[xi,xi] = -D_sigma  (K the
    return-map quadratic jet); the split X-point pair emerges along its solution.

    Args:
        fieldline (SingularPeriodicFieldline): the snowflake (nfp-symmetric). Supplies
            the closed field line (fl.curve), length and aux mu; the full-torus grid is
            built internally by resampling fl.curve on a Chebyshev grid over [0,1).
        perturbation (CoilPerturbationField): supplies delta_B = dB/dsigma (modular).
        frame (str): 'RZ' (default) or 'NB' -- the 2D frame D_sigma is projected into.
        biotsavart (BiotSavart): the MODULAR field for the unperturbed B/gradB (defaults
            to perturbation.biotsavart); the aux field (fl.mu) is added on top.
        N_full (int): number of Chebyshev nodes on [0,1) (default 6*order*nfp + 1).
        phi0 (float): toroidal fraction in [0,1) of the section where D_sigma is measured
            (default 0). Must match the SnowflakeDiscriminant phi0 in the unfolding.
    """

    def __init__(self, fieldline, perturbation, frame='RZ', biotsavart=None, N_full=None, phi0=0.0):
        self.fieldline = fieldline
        self.perturbation = perturbation
        self.biotsavart = biotsavart if biotsavart is not None else perturbation.biotsavart
        self.frame = frame
        self.phi0 = float(phi0)
        self.stellsym_aux = fieldline.stellsym_aux
        c = fieldline.curve
        Nf = int(N_full) if N_full is not None else (6 * c.order * c.nfp + 1)
        D_full, xh, wh = cheb(Nf, 0.0, 1.0)
        self.D_full = np.asarray(D_full)
        # fl.curve is nfp-symmetric, so evaluating it over t in [0,1) traces the whole closed
        # field line -> a full-torus (nfp=1) grid that shares the curve dofs. For a section at
        # toroidal fraction phi0 the SAME Chebyshev grid is started at phi0 (nodes shifted mod 1,
        # array order preserved; D_full is translation-invariant and the length is unchanged),
        # so D_sigma is measured at that section.
        xh_sec = xh if self.phi0 == 0.0 else np.mod(xh + self.phi0, 1.0)
        self.curve_full = CurveXYZFourierSymmetries(xh_sec, c.order, c.nfp, c.stellsym,
                                                    ntor=c.ntor, dofs=c.dofs)
        D = self.D_full
        self._dsig = jit(lambda B, gB, dB, L, g: dsigma_pure(B, gB, dB, L, g, D, frame))
        # gradient of the scalar  v . D_sigma  w.r.t. (B, gradB_ss, delta_B, L, gamma).
        self._dsig_grad = jit(jax.grad(
            lambda B, gB, dB, L, g, v: jnp.dot(v, dsigma_pure(B, gB, dB, L, g, D, frame)),
            argnums=(0, 1, 2, 3, 4)))
        self.recompute_bell()
        super().__init__(depends_on=[fieldline])

    def recompute_bell(self, parent=None):
        self._D_sigma = None

    def _field_data(self):
        """Full-torus grid points, total unperturbed field B, gradB (SIMSOPT conv),
        perturbation delta_B, and length L."""
        fl = self.fieldline
        if fl.need_to_run_code:
            fl.run_code(fl.res['length'])
        mu = np.asarray(fl.res['mu'])
        L = float(fl.res['length'])
        stell = self.stellsym_aux
        pts = self.curve_full.gamma()
        bs = self.biotsavart
        bs.set_points(pts.reshape((-1, 3)))
        B = bs.B() + np.asarray(_B_aux(pts, mu, stellsym=stell))
        gradB = bs.dB_by_dX() + np.asarray(_dB_aux_by_dX(pts, mu, stellsym=stell))
        dB = self.perturbation.dB_by_dsigma(pts)
        return pts, B, gradB, dB, L

    def d_sigma(self):
        """D_sigma (2-vector) in the constructor `frame`."""
        if self._D_sigma is None:
            pts, B, gradB, dB, L = self._field_data()
            self._D_sigma = np.asarray(self._dsig(
                jnp.asarray(B), jnp.asarray(gradB), jnp.asarray(dB), L, jnp.asarray(pts)))
        return self._D_sigma

    def J(self):
        return self.d_sigma()

    def d_sigma_vjp(self, v):
        """Coil-dof gradient of the scalar  v . D_sigma  (cotangent v, 2-vector) as a
        simsopt Derivative. Analytic: jax partials of D_sigma w.r.t. (B, gradB, delta_B,
        L, gamma), pushed to the coils via the modular B_and_dB_vjp, the perturbation's
        dB_by_dsigma_vjp, and the field-line state adjoint (res['PLU']/vjp). Requires a
        SQUARE converged solve. Mirrors SnowflakeDiscriminant.compute plus the delta_B terms."""
        fl = self.fieldline
        if fl.need_to_run_code:
            fl.run_code(fl.res['length'])
        res = fl.res
        if 'PLU' not in res or 'vjp' not in res:
            raise RuntimeError(
                "DSigma.d_sigma_vjp needs a SQUARE converged solve (res['PLU']/res['vjp']); "
                "fix the right number of mu dofs and re-solve.")
        mu = np.asarray(res['mu']); nmu = mu.size; L = float(res['length'])
        stell = self.stellsym_aux
        pts = self.curve_full.gamma()
        bs = self.biotsavart
        bs.set_points(pts.reshape((-1, 3)))
        B = bs.B() + np.asarray(_B_aux(pts, mu, stellsym=stell))
        gradB = bs.dB_by_dX() + np.asarray(_dB_aux_by_dX(pts, mu, stellsym=stell))
        dB = self.perturbation.dB_by_dsigma(pts)

        v = np.asarray(v, dtype=float)
        dsB, dsgB, dsdB, dsL, dsg = self._dsig_grad(
            jnp.asarray(B), jnp.asarray(gradB), jnp.asarray(dB), L, jnp.asarray(pts), jnp.asarray(v))
        dsB = np.asarray(dsB); dsgB = np.asarray(dsgB); dsdB = np.asarray(dsdB)
        dsL = float(dsL); dsg = np.asarray(dsg)

        # --- direct parts (grid fixed): modular field (B, gradB) and perturbation (delta_B) ---
        bs.set_points(pts.reshape((-1, 3)))
        coil_direct = sum(bs.B_and_dB_vjp(dsB, dsgB))
        dB_direct = self.perturbation.dB_by_dsigma_vjp(pts, dsdB)

        # --- state adjoint over x = [curve dofs, length, mu] ---
        gradgradB = bs.d2B_by_dXdX() + np.asarray(_d2B_aux_by_dXdX(pts, mu, stellsym=stell))
        dgradB_pert = self.perturbation.dgradB_by_dsigma(pts)          # delta(gradB) (N,3,3)
        dgamma_dc = self.curve_full.dgamma_by_dcoeff()                 # (N,3,n_c)
        res_curve = (np.einsum('ij,ikj,ikm->m', dsB, gradB, dgamma_dc, optimize=True)
                     + np.einsum('ikj,ikpj,ipm->m', dsgB, gradgradB, dgamma_dc, optimize=True)
                     + np.einsum('ij,ikj,ikm->m', dsdB, dgradB_pert, dgamma_dc, optimize=True)
                     + np.einsum('ij,ijm->m', dsg, dgamma_dc, optimize=True))
        dB_aux_dmu = np.asarray(_dB_aux_by_dmu(pts, mu, stellsym=stell))
        dgradB_aux_dmu = np.asarray(_dgradB_aux_by_dmu(pts, mu, stellsym=stell))
        d_mu = (np.einsum('ij,ijl->l', dsB, dB_aux_dmu, optimize=True)
                + np.einsum('ikj,ikjl->l', dsgB, dgradB_aux_dmu, optimize=True))

        col_mask = np.asarray(res['mask'], dtype=bool)
        full_dJ_dx = np.concatenate([res_curve, [dsL], d_mu])
        dJ_ds = full_dJ_dx[col_mask]
        P, Lm, U = res['PLU']
        adj = forward_backward(P, Lm, U, dJ_ds)
        adj_term = res['vjp'](adj, fl.biotsavart, fl)

        mu_free = ~col_mask[-nmu:]
        mu_direct = np.zeros(nmu)
        mu_direct[mu_free] = d_mu[mu_free]
        free_mu = Derivative({fl: mu_direct})

        return coil_direct + dB_direct + free_mu - adj_term


# =============================================================================
# Unfolding constant C: solve the shape equation 0.5 K[xi,xi] = -D_sigma
# =============================================================================
def unfolding_solutions(K, Dsigma):
    """Real split directions xi solving  0.5 K[xi,xi] = -D_sigma  in CLOSED FORM
    (array/closedform.png), for a return-map jet K (2,2,2) and D_sigma (2,).

    Step 1 (direction): g2*eq1 - g1*eq2 kills the RHS -> homogeneous quadratic
    alpha xi1^2 + 2 beta xi1 xi2 + gamma xi2^2 = 0, with alpha=g2 a1-g1 a2, beta=g2 b1-g1 b2,
    gamma=g2 c1-g1 c2 (a_k=K[k,0,0], b_k=K[k,0,1], c_k=K[k,1,1], g=D_sigma); slopes
    t=(-beta +- sqrt(beta^2-alpha*gamma))/gamma (real & distinct iff beta^2-alpha*gamma>0).
    Step 2 (magnitude): xi1^2 = -2 g_k/(a_k+2 b_k t+c_k t^2), kept iff >0 (the pair that
    unfolds at this sign of dsigma). Returns dicts {t, xi, C=2||xi||} sorted by C ascending
    (nearest split pair first); [] if there is no real split."""
    K = np.asarray(K)
    g = np.asarray(Dsigma)
    a = np.array([K[0, 0, 0], K[1, 0, 0]])
    b = np.array([K[0, 0, 1], K[1, 0, 1]])
    c = np.array([K[0, 1, 1], K[1, 1, 1]])
    g1, g2 = float(g[0]), float(g[1])
    alpha = g2 * a[0] - g1 * a[1]
    beta = g2 * b[0] - g1 * b[1]
    gamma = g2 * c[0] - g1 * c[1]

    cands = []                                  # candidate slopes t (np.inf = vertical xi1=0)
    if gamma != 0.0:
        disc = beta * beta - alpha * gamma
        if disc >= 0.0:
            sq = np.sqrt(disc)
            cands = [(-beta + sq) / gamma, (-beta - sq) / gamma]
    else:                                       # gamma=0: a root at infinity + one finite slope
        cands = [np.inf] + ([-alpha / (2.0 * beta)] if beta != 0.0 else [])

    sols = []
    for t in cands:
        if np.isinf(t):                         # vertical: c_k xi2^2 = -2 g_k
            k = 0 if abs(c[0]) >= abs(c[1]) else 1
            if c[k] == 0.0:
                continue
            xisq = -2.0 * g[k] / c[k]
            if xisq <= 0.0:
                continue
            xi = np.array([0.0, np.sqrt(xisq)])
            C = 2.0 * np.sqrt(xisq)
        else:
            den = a + 2.0 * b * t + c * t * t
            k = 0 if abs(den[0]) >= abs(den[1]) else 1
            if den[k] == 0.0:
                continue
            xi1sq = -2.0 * g[k] / den[k]
            if xi1sq <= 0.0:
                continue
            xi = np.sqrt(xi1sq) * np.array([1.0, t])
            C = 2.0 * np.sqrt(xi1sq * (1.0 + t * t))
        sols.append({'t': float(t), 'xi': xi, 'C': float(C)})
    sols.sort(key=lambda s: s['C'])
    return sols


class SnowflakeUnfolding(Optimizable):
    r"""Unfolding of the snowflake under a coil perturbation. Solves the parameter-free
    shape equation (see array/unfolding.png)

        0.5 * K[xi, xi] = -D_sigma,

    with K the return-map quadratic jet on the FULL torus (K = nfp * the 1/nfp jet from
    SnowflakeDiscriminant) and D_sigma from :class:`DSigma`, both in the same frame. Its
    real solutions xi* are the split-pair directions: the snowflake splits as
    x_{1,2} = x0 +- sqrt(dsigma) xi*, separating by  C * sqrt(dsigma)  with

        C = 2 ||xi*||  = 2 sqrt( 2 ||D_sigma|| / ||K[uhat, uhat]|| ),

    uhat the direction with K[uhat,uhat] anti-parallel to D_sigma. J() = C for the nearest
    split pair (smallest C); dJ() = dC/d(coil dofs) via the implicit function theorem,
    chaining DSigma.d_sigma_vjp and SnowflakeDiscriminant.jet_cotangent_vjp.

    Args:
        fieldline (SingularPeriodicFieldline): the snowflake.
        perturbation (CoilPerturbationField): the coil GP perturbation.
        frame (str): 'RZ' (default) or 'NB'.
        biotsavart (BiotSavart): modular field (defaults to perturbation.biotsavart).
        ntheta (int): angular samples for the direction scan.
        phi0 (float): toroidal fraction in [0,1) of the Poincare section where C is measured
            (default 0). C depends on phi0 (a physical separation in that section); it is
            threaded to BOTH DSigma and SnowflakeDiscriminant so K and D_sigma share the section.
    """

    def __init__(self, fieldline, perturbation, frame='RZ', biotsavart=None, ntheta=2048, phi0=0.0):
        self.fieldline = fieldline
        bs = biotsavart if biotsavart is not None else perturbation.biotsavart
        self.frame = frame
        self.phi0 = float(phi0)
        self.nfp = int(fieldline.curve.nfp)
        self.ntheta = int(ntheta)
        self.dsigma = DSigma(fieldline, perturbation, frame=frame, biotsavart=bs, phi0=self.phi0)
        self.disc = SnowflakeDiscriminant(fieldline, bs, frame=frame, phi0=self.phi0)
        self.recompute_bell()
        super().__init__(depends_on=[self.dsigma, self.disc])

    def recompute_bell(self, parent=None):
        self._sols = None

    def _K_and_Dsigma(self):
        K = self.nfp * np.asarray(self.disc.jet())    # (2,2,2) full-torus jet
        Dsig = np.asarray(self.dsigma.d_sigma())      # (2,)
        return K, Dsig

    def solutions(self):
        """Real split directions solving 0.5 K[xi,xi] = -D_sigma, IN CLOSED FORM
        (array/closedform.png). Step 1 (direction): g2*eq1 - g1*eq2 kills the RHS, leaving
        the homogeneous quadratic alpha xi1^2 + 2 beta xi1 xi2 + gamma xi2^2 = 0 with
        alpha = g2 a1 - g1 a2, beta = g2 b1 - g1 b2, gamma = g2 c1 - g1 c2 (a_k=K[k,0,0],
        b_k=K[k,0,1], c_k=K[k,1,1], g=D_sigma); slopes t = (-beta +- sqrt(beta^2-alpha*gamma))/gamma
        (real & distinct iff beta^2-alpha*gamma > 0). Step 2 (magnitude): xi1^2 = -2 g_k /
        (a_k + 2 b_k t + c_k t^2), kept iff > 0 (the pair that unfolds at this sign of dsigma;
        the other slope unfolds at the opposite sign). Returns dicts {t, xi, C} sorted by C
        ascending (nearest pair first); [] if no real split. Thin wrapper over the
        module-level closed form unfolding_solutions(K, D_sigma)."""
        if self._sols is None:
            K, g = self._K_and_Dsigma()
            self._sols = unfolding_solutions(K, g)
        return self._sols

    def C(self):
        """Unfolding constant C = 2||xi*|| of the NEAREST split pair (smallest C); nan if the
        shape equation has no real solution (no real split for this D_sigma / K)."""
        sols = self.solutions()
        return float(sols[0]['C']) if sols else float('nan')

    def leg_directions(self, tol=1e-9):
        r"""Separatrix (leg) directions of the two X-points born from the snowflake, expressed
        in the unfolding's OWN frame (RZ or NB, as chosen at construction via `frame=`).

        A born X-point sits at x0 +- sqrt(sigma) xi* (xi* = nearest split direction). Its
        monodromy is M = I + sqrt(sigma) G + O(sigma) with G_ij = K_ijk xi*_k -- exactly the
        shape Jacobian `Jmat` of dC(). Area preservation makes G trace-free, so its eigenvalues
        are +-lambda with lambda = sqrt(-det G); its eigenvectors ARE the separatrix lines.
        M and G share eigenvectors, and the unstable/stable multipliers are 1 +- sqrt(sigma)*lambda
        to leading order. The two born points (at +xi* and -xi*) have the SAME two eigenvector
        lines but SWAPPED stable/unstable roles (the point at -xi* uses G -> -G).

        Directions are returned as UNIT 2-vectors in `self.frame`:
            'RZ' -> components (dR, dZ) in the meridional Poincare plane;
            'NB' -> components along (normal, binormal) of the axis frame.
        The 4 rays of each X-point are +-unstable and +-stable.

        Returns None if there is no real split. Otherwise a dict:
            {'frame', 'xi', 'C', 'lambda', 'hyperbolic', 'det_G', 'trace_G',
             'unstable', 'stable', 'line_angles',     # for the X-point at +xi*
             'xpoints': [ {'at': +1, 'unstable', 'stable'},
                          {'at': -1, 'unstable', 'stable'} ]}   # roles swapped at -xi*
        """
        sols = self.solutions()
        if not sols:
            return None
        K, _ = self._K_and_Dsigma()
        xi = sols[0]['xi']
        G = np.einsum('icb,b->ic', K, xi)             # = dC()'s Jmat; trace ~ 0
        detG = float(np.linalg.det(G))
        traceG = float(np.trace(G))
        evals, evecs = np.linalg.eig(G)
        hyperbolic = (detG < 0.0) and bool(np.all(np.abs(np.imag(evals)) <= tol * (1.0 + np.abs(evals))))
        if hyperbolic:
            evals = np.real(evals)
            evecs = np.real(evecs)
            iu = int(np.argmax(evals))                # +lambda -> unstable manifold
            is_ = int(np.argmin(evals))               # -lambda -> stable manifold
            vu = evecs[:, iu] / np.linalg.norm(evecs[:, iu])
            vs = evecs[:, is_] / np.linalg.norm(evecs[:, is_])
            lam = float(evals[iu])
        else:                                         # elliptic (O-point) -- excluded for Delta>0
            vu = vs = None
            lam = float(np.sqrt(abs(detG)))
        _ang = lambda v: None if v is None else float(np.arctan2(v[1], v[0]))
        return {
            'frame': self.frame, 'xi': xi, 'C': float(sols[0]['C']),
            'lambda': lam, 'hyperbolic': bool(hyperbolic), 'det_G': detG, 'trace_G': traceG,
            'unstable': vu, 'stable': vs, 'line_angles': (_ang(vu), _ang(vs)),
            'xpoints': [{'at': +1.0, 'unstable': vu, 'stable': vs},
                        {'at': -1.0, 'unstable': vs, 'stable': vu}],
        }

    def born_xpoints(self, sigma=None):
        r"""The two X-points the snowflake unfolds into, with their separatrix leg directions, in
        the unfolding's OWN frame (RZ or NB). The X-points sit at x0 +- sqrt(sigma) xi* (x0 = the
        snowflake, at the ORIGIN of the reduced frame); their separation is C*sqrt(sigma),
        C=2||xi*||. The legs are the eigenvectors of G = K[.,.,xi*] (unit 2-vectors in self.frame,
        from leg_directions()); the two X-points share the same leg LINES with stable/unstable roles
        swapped.

        `sigma` is OPTIONAL. If sigma is None (default), the X-point POSITIONS are NOT formed: the
        result carries only sigma-INDEPENDENT data -- the split direction xi*, the snowflake anchor
        x0_RZ, each X-point's displacement DIRECTION (+-xi*) and legs -- and the position fields are
        None. Pass a numeric sigma to also get the positions x0 +- sqrt(sigma) xi*.

        Returns None if there is no real split, else a dict:
            {'frame', 'sigma',                 # sigma is None or the value passed
             'hyperbolic', 'lambda',
             'xi':    xi* (2-vector, split direction in self.frame),
             'x0_RZ': (R, Z) of the snowflake (frame='RZ' only, else None),
             'reduced': (2,2) [[+], [-]] 2D positions in self.frame relative to x0, or None (sigma=None),
             'RZ':      (2,2) physical (R, Z) positions (frame='RZ' only), or None,
             'xpoints': [ {'at': +1, 'xi': +xi*, 'reduced', 'RZ', 'unstable', 'stable'},
                          {'at': -1, 'xi': -xi*, 'reduced', 'RZ', 'unstable', 'stable'} ]}
        where each 'xi' is that X-point's displacement DIRECTION from x0 (+xi* for at=+1, -xi* for
        at=-1), 'reduced'/'RZ' its position (None when sigma is None), and 'unstable'/'stable' its
        leg unit vectors in self.frame (None if not hyperbolic, impossible for a Delta>0 snowflake)."""
        sols = self.solutions()
        if not sols:
            return None
        xi = sols[0]['xi']                                       # split direction in self.frame
        ld = self.leg_directions()                              # legs of the two born X-points
        x0_RZ = None
        if self.frame == 'RZ':
            g0 = np.asarray(self.disc._sec_curve.gamma())[0]    # section anchor (xyz)
            x0_RZ = np.array([float(np.hypot(g0[0], g0[1])), float(g0[2])])
        signs = [1.0 if lp['at'] > 0 else -1.0 for lp in ld['xpoints']]   # +xi* for at=+1, -xi* for at=-1
        if sigma is None:
            reduced = RZ = None
            red_i = [None, None]
            rz_i = [None, None]
        else:
            d = float(np.sqrt(sigma)) * xi
            reduced = np.array([d, -d])                         # row 0 (+xi*), row 1 (-xi*)
            RZ = None if x0_RZ is None else (reduced + x0_RZ)
            red_i = [reduced[0], reduced[1]]
            rz_i = [None, None] if RZ is None else [RZ[0], RZ[1]]
        xpoints = [{'at': lp['at'], 'xi': signs[i] * xi,
                    'reduced': red_i[i], 'RZ': rz_i[i],
                    'unstable': lp['unstable'], 'stable': lp['stable']}
                   for i, lp in enumerate(ld['xpoints'])]
        return {'frame': self.frame, 'sigma': sigma, 'xi': xi, 'x0_RZ': x0_RZ,
                'hyperbolic': ld['hyperbolic'], 'lambda': ld['lambda'],
                'reduced': reduced, 'RZ': RZ, 'xpoints': xpoints}

    def plot(self, axis_RZ=None, sigma=1e-1, ax=None, show=False):
        r"""Sketch the snowflake unfolding (deploiement) in the (R,Z) plane at perturbation amplitude
        sigma: the original snowflake x0, the two born X-points x0 +- sqrt(sigma) xi*, and each
        X-point's separatrix legs (unstable solid, stable dashed). If a magnetic axis position is
        given it is drawn too (with a guide line to x0) and the SF+/SF- classification goes in the
        title. Requires the RZ frame.

        Args:
            axis_RZ: (R, Z) of the magnetic axis in this section; if None (default) the axis is not
                     drawn and no SF+/SF- label is computed.
            sigma:   perturbation amplitude that sets the (visual) split size (default 1e-1).
            ax:      an optional matplotlib Axes (a new figure is made if None).
        Returns the matplotlib Axes.
        """
        import matplotlib.pyplot as plt
        assert self.frame == 'RZ', "plot works in the RZ frame only"
        born = self.born_xpoints(sigma)
        if born is None:
            raise RuntimeError("no real split at this section (nothing to plot)")
        x0 = np.asarray(born['x0_RZ'])
        axis = None if axis_RZ is None else np.asarray(axis_RZ, dtype=float)
        P = [np.asarray(xp['RZ']) for xp in born['xpoints']]
        leglen = 0.6 * float(np.linalg.norm(P[0] - P[1]))            # legs scaled to the split size
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.plot([P[0][0], P[1][0]], [P[0][1], P[1][1]], '-', color='0.6', lw=1, zorder=1)   # split segment
        ax.plot(x0[0], x0[1], 'k*', ms=9, label=r'snowflake $x_0$', zorder=4)
        if axis is not None:                                          # magnetic axis + guide line to x0
            ax.plot([x0[0], axis[0]], [x0[1], axis[1]], ':', color='tab:blue', lw=1, zorder=1)
            ax.plot(axis[0], axis[1], 'o', color='tab:blue', ms=5, label='magnetic axis', zorder=4)
        for i, (xp, col) in enumerate(zip(born['xpoints'], ('tab:red', 'tab:orange'))):
            p = P[i]
            ax.plot(p[0], p[1], 'X', color=col, ms=7, zorder=5,
                    label=f"X-point ({'+' if xp['at'] > 0 else '-'})")
            for v, ls, nm in ((xp['unstable'], '-', 'unstable'), (xp['stable'], '--', 'stable')):
                if v is None:
                    continue
                v = np.asarray(v)
                ax.plot([p[0] - leglen * v[0], p[0] + leglen * v[0]],
                        [p[1] - leglen * v[1], p[1] + leglen * v[1]], ls, color=col, lw=1.5,
                        zorder=3, label=(nm if i == 0 else None))
        label = classify(axis, self.born_xpoints(), self.phi0) if axis is not None else None
        title = "snowflake unfolding" + (f": {label}" if label else "") + \
                f"   (sigma={sigma:g}, C={self.C():.3f})"
        ax.set_aspect('equal')
        ax.set_xlabel('R'); ax.set_ylabel('Z')
        ax.set_title(title)
        ax.legend(loc='best', fontsize=8)

        if show:
            plt.show()

        return ax

    def J(self):
        return self.C()

    def dC(self):
        """dC/d(coil dofs) for the nearest split pair, as a simsopt Derivative. IFT on
        F(xi)=0.5 K[xi,xi]+D_sigma=0: dC = w . dD_sigma + 0.5 w_i xi_a xi_b dK_iab, with
        w = -2 J^{-T} xihat, J_ic = K_icb xi_b. Uses DSigma.d_sigma_vjp and
        SnowflakeDiscriminant.jet_cotangent_vjp (K = nfp * K_p -> cotangent nfp*0.5 w xi xi)."""
        sols = self.solutions()
        if not sols:
            raise RuntimeError("no real split direction (C is nan); dC is undefined.")
        K, _ = self._K_and_Dsigma()
        xi = sols[0]['xi']
        xihat = xi / np.linalg.norm(xi)
        Jmat = np.einsum('icb,b->ic', K, xi)          # Jacobian of the shape map at xi*
        w = -2.0 * np.linalg.solve(Jmat.T, xihat)     # C-residual adjoint (2,)
        cK = self.nfp * 0.5 * np.einsum('i,a,b->iab', w, xi, xi)   # cotangent on the 1/nfp jet
        return self.dsigma.d_sigma_vjp(w) + self.disc.jet_cotangent_vjp(cK)

    @derivative_dec
    def dJ(self):
        return self.dC()

    return_fn_map = {'J': J, 'dJ': dJ}


def classify(axis_RZ, born, in_phi):
    r"""Classify a snowflake unfolding (deploiement) as "SF+" or "SF-" (snowflake-plus /
    snowflake-minus) in the sigma -> 0+ limit, working in the (R,Z) frame ONLY.

    The result is INDEPENDENT of the amplitude sigma: only DIRECTIONS enter (leg directions and the
    xi displacement direction), never the sqrt(sigma) magnitude -- so `born` can be built with
    sigma=None. `born` is a SnowflakeUnfolding.born_xpoints(...) dict at the cylindrical section
    phi=in_phi, and `axis_RZ` = (R, Z) of the magnetic axis in that same section.

    The two X-points are at x0 +- sqrt(sigma) xi (x0 = the snowflake, born['x0_RZ']). Pick the one
    that WILL be nearest the magnetic axis as sigma -> 0+, i.e. the one displaced TOWARD the axis
    (sign of xi . (axis - x0)). Its two separatrix legs are two lines through x0 that cut the (R,Z)
    plane into four quadrants. Place, by DIRECTION only, the magnetic axis (direction x0 -> axis)
    and the OTHER X-point (direction of its xi displacement) among them:
        adjacent quadrants  -> "SF-",     opposite (diagonal) quadrants -> "SF+".

    Returns "SF+", "SF-", or None (no real split, born not hyperbolic, a direction on a leg line, or
    both in the same quadrant).

    Args:
        axis_RZ: (R, Z) of the magnetic axis at the section phi=in_phi.
        born:    SnowflakeUnfolding.born_xpoints(sigma=None) dict; MUST have frame 'RZ'.
        in_phi:  cylindrical angle fraction in [0, 1) of the section (axis and born share it).
    """
    if born is None:
        return None
    assert born.get('frame') == 'RZ', \
        "classify must be done in the RZ frame (born['frame'] must be 'RZ')"
    assert 0.0 <= float(in_phi) < 1.0, "in_phi must be a cylindrical fraction in [0, 1)"
    if not born.get('hyperbolic', False):
        return None

    axis = np.asarray(axis_RZ, dtype=float)
    x0 = np.asarray(born['x0_RZ'], dtype=float)            # snowflake (R,Z) (present in the RZ frame)
    e = axis - x0                                          # snowflake -> magnetic axis (sigma-independent)
    xps = born['xpoints']
    # the X-point displaced TOWARD the axis is the primary/near one as sigma -> 0+ (sign of xi . e)
    i_near = 0 if float(np.dot(np.asarray(xps[0]['xi'], dtype=float), e)) > 0.0 else 1
    i_far = 1 - i_near
    u = np.asarray(xps[i_near]['unstable'], dtype=float)   # legs of the near X-point (sigma-independent)
    s = np.asarray(xps[i_near]['stable'], dtype=float)
    v_far = np.asarray(xps[i_far]['xi'], dtype=float)      # DIRECTION to the far X-point (-+ xi)

    def _quadrant(w):
        """(side of the unstable-leg line, side of the stable-leg line) for the DIRECTION w, or None
        if w lies on a leg line. The two leg lines through the snowflake cut the plane into four."""
        nw = float(np.linalg.norm(w))
        cu = float(u[0] * w[1] - u[1] * w[0])              # cross(u, w): side of the unstable line
        cs = float(s[0] * w[1] - s[1] * w[0])              # cross(s, w): side of the stable line
        if nw == 0.0 or abs(cu) < 1e-9 * nw or abs(cs) < 1e-9 * nw:
            return None
        return (cu > 0.0, cs > 0.0)

    qa = _quadrant(e)          # magnetic axis, by the direction snowflake -> axis
    qf = _quadrant(v_far)      # far X-point, by the direction of its xi displacement
    if qa is None or qf is None:
        return None
    ndiff = int(qa[0] != qf[0]) + int(qa[1] != qf[1])
    if ndiff == 1:
        return "SF-"          # adjacent quadrants
    if ndiff == 2:
        return "SF+"          # opposite (diagonal) quadrants
    return None               # same quadrant -> undetermined
