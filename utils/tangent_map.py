import simsoptpp as sopp
from simsopt.field import BiotSavart
from simsopt.geo import ToroidalFlux, Volume, CurveXYZFourierSymmetries
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.objectives.utilities import forward_backward
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacfwd
import jax

#from jax import config
#config.update("jax_disable_jit", True)

# adapted from https://hackmd.io/@NCTUIAM5804/Sk1JhoXoI
def cheb(Npts, a, b):
    N = Npts-1
    assert N >= 1
    alt = (-np.ones(N+1))**np.arange(N+1)
    x = np.cos(np.pi*np.linspace(0,1,N+1))
    c = np.array([2] + [1]*(N-1)  + [2]) * alt
    X = np.outer(x, np.ones(N+1))
    dX = X-X.T
    D = np.outer(c, np.array([1]*(N+1))/c) / (dX + np.identity(N+1))
    D = D - np.diag(np.sum(D,axis=1))
   
    # Clenshaw-Curtis quadrature weights on the Chebyshev points x = cos(pi*j/N),
    # j=0..N (Trefethen, "Spectral Methods in MATLAB", clencurt.m). The previous
    # half-and-mirror construction was only correct for EVEN N: for odd N the
    # reflection np.flip(w[:-1]) yielded N weights instead of N+1 (wrong quadrature),
    # so the even/odd cases are handled explicitly here.
    theta = np.pi * np.arange(N + 1) / N
    w = np.zeros(N + 1)
    ii = np.arange(1, N)            # interior nodes 1..N-1
    v = np.ones(N - 1)
    if N % 2 == 0:
        w[0] = w[N] = 1.0 / (N**2 - 1)
        for kk in range(1, N // 2):
            v -= 2.0 * np.cos(2.0 * kk * theta[ii]) / (4.0 * kk**2 - 1)
        v -= np.cos(N * theta[ii]) / (N**2 - 1)
    else:
        w[0] = w[N] = 1.0 / N**2
        for kk in range(1, (N - 1) // 2 + 1):
            v -= 2.0 * np.cos(2.0 * kk * theta[ii]) / (4.0 * kk**2 - 1)
    w[ii] = 2.0 * v / N
    
    x = 0.5*(x+1)*(b-a) + a
    D = D/(0.5*(b-a))
    w = 0.5 * w * (b-a)

    x = x[::-1]
    w = w[::-1]
    D = D[::-1, :]
    D = D[:, ::-1]
    
    return D, x, w

#I = np.zeros((B.shape[0], 3, 3))
#I[:, 0, 0] = 1.0
#I[:, 1, 1] = 1.0
#I[:, 2, 2] = 1.0

#b = B/modB[:, None]
#temp = (1.0/modB[:, None, None]) * np.matmul((I - b[:, :, None] * b[:, None, :]), gradB) 
#import ipdb;ipdb.set_trace()



def reduced_frame_pure(B, gamma, D, frame='NB'):
    """Embedding P0 and projection Pt for the reduced 2x2 map, in the chosen frame.

    Shared by monodromy_pure and discriminant.quadratic_jet_pure so both use
    identical embeddings.

    Returns
    -------
    P0 : (3,2)     embedding of R^2 into the plane at t=0 (columns are the two
                   in-plane basis vectors).
    Pt : (N,2,3)   projection back into the 2D frame at each collocation point.
    n  : (N,3)     unit normal of the Poincare section at each collocation point
                   (fT for 'NB', e_phi for 'RZ'). Needed by the return-time
                   correction in quadratic_jet_pure; unused by monodromy_pure.

    frame='NB' (default): moving normal-binormal frame of the axis. Pt is the
        ORTHOGONAL projection onto the {N,B} plane (i.e. along the tangent fT).
    frame='RZ': cylindrical (R,Z) Poincare-section frame. The embedding at t=0 uses
        the primal poloidal basis [e_R(0), e_Z(0)]; the projection at t uses the
        reciprocal (dual) covectors of {e_R, e_Z, b}(t), which project the field
        direction b out (b_phi != 0 required). See monodromy_pure for the formula.
    """
    if frame == 'NB':
        tangent = jnp.matmul(D, gamma)
        fT = tangent / jnp.linalg.norm(tangent, axis=-1)[:, None]

        normal = jnp.matmul(D, fT)
        fN = normal / jnp.linalg.norm(normal, axis=-1)[:, None]

        binormal = jnp.cross(fT, fN)
        fB = binormal / jnp.linalg.norm(binormal, axis=-1)[:, None]

        P0 = jnp.concatenate((fN[:, :, None], fB[:, :, None]), axis=-1)[0]  # (3,2)
        Pt = jnp.concatenate((fN[:, None, :], fB[:, None, :]), axis=-2)     # (N,2,3)
        n  = fT                                                             # (N,3)
    elif frame == 'RZ':
        phi = jnp.arctan2(gamma[:, 1], gamma[:, 0])
        cphi, sphi = jnp.cos(phi), jnp.sin(phi)
        zero, one = jnp.zeros_like(cphi), jnp.ones_like(cphi)
        eR   = jnp.stack([cphi, sphi, zero], axis=-1)    # (N,3)
        ephi = jnp.stack([-sphi, cphi, zero], axis=-1)   # (N,3)
        eZ   = jnp.stack([zero, zero, one], axis=-1)     # (N,3)

        b = B / jnp.linalg.norm(B, axis=-1)[:, None]
        bR   = jnp.sum(b*eR,   axis=-1)   # (N,)
        bphi = jnp.sum(b*ephi, axis=-1)   # (N,)
        bZ   = jnp.sum(b*eZ,   axis=-1)   # (N,)

        # reciprocal covectors of {e_R, e_Z, b}: annihilate b, reproduce (R,Z) components
        eR_star = eR - (bR/bphi)[:, None]*ephi   # (N,3)
        eZ_star = eZ - (bZ/bphi)[:, None]*ephi   # (N,3)

        P0 = jnp.stack([eR[0], eZ[0]], axis=-1)      # (3,2) primal embed at t=0
        Pt = jnp.stack([eR_star, eZ_star], axis=-2)  # (N,2,3) dual project at t
        n  = ephi                                    # (N,3) section {phi=const} normal
    else:
        raise ValueError(f"frame must be 'NB' or 'RZ', got {frame!r}")
    return P0, Pt, n














def A_pure(B, gradB):
    modB = jnp.linalg.norm(B, axis=-1)
    dmodB = jnp.sum(B[:, :, None] * gradB, axis=1) / modB[:, None]
    A = (gradB*modB[:, None, None]-B[:, :, None]*dmodB[:, None, :])/modB[:, None, None]**2
    return A

def tangent_map_residual_pure(T, B, gradB, L, ic, D):
    A = A_pure(B, gradB)
    AT = jnp.einsum('ijk,ik->ij', A, T)
    Tprime = jnp.matmul(D, T)
    residual = Tprime/L - AT
    ic0 = T[0] - ic
    res = jnp.concatenate((ic0[None, :], residual[1:]), axis=0)
    return res

def tangent_map_pure(B, gradB, L, D):
    N = B.shape[0]
    T = jnp.zeros_like(B)
    
    b1 = -tangent_map_residual_pure(T, B, gradB, L, jnp.array([1, 0, 0]), D).ravel()
    b2 = -tangent_map_residual_pure(T, B, gradB, L, jnp.array([0, 1, 0]), D).ravel()
    b3 = -tangent_map_residual_pure(T, B, gradB, L, jnp.array([0, 0, 1]), D).ravel()
    A = jacfwd(tangent_map_residual_pure, argnums=0)(T, B, gradB, L, jnp.array([0, 0, 0]), D).reshape((3*N, 3*N))
    
    T1 = jnp.linalg.solve(A, b1).reshape((N, 3))
    T2 = jnp.linalg.solve(A, b2).reshape((N, 3))
    T3 = jnp.linalg.solve(A, b3).reshape((N, 3))
    
    T = jnp.concatenate((T1[:, :, None], T2[:, :, None], T3[:, :, None]), axis=-1)
    return T

def monodromy_pure(B, gradB, L, gamma, D, frame='NB'):
    """Return map / monodromy along the field line, projected into a 2x2 frame.

    frame='NB' (default): the moving normal-binormal frame of the axis,
        M_NB = R^T(t_p) Phi(t_p) R(0)  with the columns of R being (N, B).
    frame='RZ': the cylindrical (R, Z) Poincare-section frame. The embedding at
        t=0 uses the primal poloidal basis [e_R(0), e_Z(0)], while the projection
        at t_p uses the reciprocal (dual) covectors of the frame {e_R, e_Z, b}(t_p)
        so that the flow's field-direction component is projected out (see
        ~/Downloads/formula.png):
            M_RZ = [e_R*(t_p)^T; e_Z*(t_p)^T] Phi(t_p) [e_R(0), e_Z(0)],
            e_R* = e_R - (b_R/b_phi) e_phi,   e_Z* = e_Z - (b_Z/b_phi) e_phi,
        with b = B/|B| the unit field and {e_R, e_phi, e_Z} the orthonormal
        cylindrical basis (b_phi != 0 is required, true on a field-period axis).

    P0 is the (3,2) embedding of R^2 into the plane at t=0; Pt is the (N,2,3)
    projection back into the 2D frame at each collocation point.
    """
    M = tangent_map_pure(B, gradB, L, D)
    P0, Pt, _ = reduced_frame_pure(B, gamma, D, frame)
    R = jnp.matmul(Pt, jnp.matmul(M, P0))
    return R


def monodromy_eps_pure(B, gradB, L, gamma, D, eps, mtype, frame='NB'):
    R = monodromy_pure(B, gradB, L, gamma, D, frame)
    Rf=R[-1]

    if mtype == 'identity':
        diff = jnp.abs(Rf-jnp.eye(2))
    elif mtype == 'jordan':
        diff = jnp.abs(Rf[0, 0] + Rf[1, 1] - 2.)
    else:
        raise Exception('mtype not implemented')
    return jnp.mean(jnp.maximum(diff-eps, 0)**2)

def monodromy_identity_pure(B, gradB, L, gamma, D, frame='NB'):
    R = monodromy_pure(B, gradB, L, gamma, D, frame)
    Rf=R[-1]
    return jnp.mean((Rf-jnp.eye(2))**2)
def monodromy_matrix_pure(B, gradB, L, gamma, D, frame='NB'):
    R = monodromy_pure(B, gradB, L, gamma, D, frame)
    Rf=R[-1]
    return Rf


def iota_pure(B, gradB, L, gamma, D, nfp, frame='NB'):
    """On-axis rotational transform from the one-field-period return map R[-1]. For an
    area-preserving elliptic 2x2 map tr(R[-1]) = 2*cos(theta), so
    theta = arctan2(sqrt(4 - tr^2), tr) is the rotation over one field period and
    iota = nfp*theta/(2*pi). Smooth in R[-1] (hence in B, gradB, L, gamma), so jax can
    differentiate it for the on-axis-iota penalty. Frame-independent to machine
    precision (NB and RZ return maps are similar, so tr(R[-1]) is invariant)."""
    Rf = monodromy_matrix_pure(B, gradB, L, gamma, D, frame)
    tr = Rf[0, 0] + Rf[1, 1]
    theta = jnp.arctan2(jnp.sqrt(jnp.maximum(4.0 - tr * tr, 0.0)), tr)
    return nfp * theta / (2.0 * jnp.pi)



def eigenvalues_pure(B, gradB, L, gamma, D, frame='NB'):
    R = monodromy_pure(B, gradB, L, gamma, D, frame).astype(complex)
    #tr = jnp.trace(R, axis1=1, axis2=2).astype(complex)
    # skip the nondifferentiable tr=2 in the IC
    a = R[1:, 0, 0]
    b = R[1:, 0, 1]
    c = R[1:, 1, 0]
    d = R[1:, 1, 1]

    eigs1 = ((a+d) + jnp.sqrt((a-d)**2 + 4*b*c)) / 2.
    eigs2 = ((a+d) - jnp.sqrt((a-d)**2 + 4*b*c)) / 2.
    return eigs1, eigs2

def elongation_pure(B, gradB, L, gamma, D, frame='NB'):
    R = monodromy_pure(B, gradB, L, gamma, D, frame)
    # get an eigenvector, then make S
    #v = jnp.linalg.eig(R[-1])[1][:, 0]
    #S = jnp.concatenate([v[:, None].real, v[:, None].imag], axis=1)
    #S = sqrtm_2x2_pure(R[-1].T @ R[-1])
    Rf = R[-1].astype(complex)
    a = Rf[0, 0]
    b = Rf[0, 1]
    c = Rf[1, 0]
    d = Rf[1, 1]

    eig1 = ((a+d) + jnp.sqrt((a-d)**2 + 4*b*c)) / 2.
    eig2 = ((a+d) - jnp.sqrt((a-d)**2 + 4*b*c)) / 2.
    v1 = jnp.array([-b, a-eig1])
    S = jnp.concatenate([v1[:, None].real, v1[:, None].imag], axis=1)

    # Elongation = aspect ratio of the invariant ellipse S(unit circle) = ratio of
    # the SINGULAR values of S, NOT its eigenvalues (S is generally non-symmetric, so
    # its eigenvalues are not the ellipse semi-axes). sigma^2 are the eigenvalues of
    # the SPD matrix S^T S.
    a, b = S[0]
    c, d = S[1]
    p = a*a + c*c          # (S^T S)_00
    q = b*b + d*d          # (S^T S)_11
    r = a*b + c*d          # (S^T S)_01 = (S^T S)_10
    disc = jnp.sqrt(jnp.maximum((p - q)**2 + 4.0*r*r, 0.0))
    s1 = jnp.sqrt((p + q + disc) / 2.)   # largest singular value
    s2 = jnp.sqrt((p + q - disc) / 2.)   # smallest singular value
    return s1 / s2

class TangentMap(Optimizable):
    def __init__(self, axis, biotsavart, threshold, mtype='identity', phi=0.0, frame='NB'):
        """
        Evaluate the tangent map on a fieldline over ONE field period starting at the
        toroidal angle `phi` (in units of phi/2pi). The return map / monodromy /
        elongation are therefore anchored at `phi`, and sweeping `phi` traces out their
        toroidal profile; phi=0 reproduces the original [0, 1/nfp] window.

        `frame` selects the 2x2 frame the return map is projected into: 'NB' (default)
        for the moving normal-binormal frame of the axis, or 'RZ' for the cylindrical
        Poincare-section frame (see monodromy_pure). The RZ frame gives the physical
        (R, Z) cross-section elongation; iota is frame-independent.

        The second-order objects (the quadratic jet, snowflake leg directions and
        discriminant) now live in utils/discriminant.py: pass this TangentMap to
        discriminant.tangent_map_jet2 / snowflake_angles_from_jet2 /
        snowflake_discriminant_pure.

        Args:
        """
        super().__init__(depends_on=[axis])
        self.biotsavart = biotsavart
        self.frame = frame
        
        nfp = axis.curve.nfp
        # Integrate over one field period [phi, phi+1/nfp]; phi shifts the window so the
        # return map (and hence the elongation) is anchored at that toroidal angle.
        N = 5*axis.curve.order+1  # Number of intervals (N+1 grid points)
        D, xh, wh = cheb(N, phi, phi + 1./nfp)
        self.D = D
        self.xh = xh
        self.wh = wh

        self.monodromy_matrix      = lambda B, gradB, L, gamma: monodromy_matrix_pure(B, gradB, L, gamma, self.D, self.frame)
        self.monodromy_jax       = lambda B, gradB, L, gamma: monodromy_eps_pure(B, gradB, L, gamma, self.D, threshold, mtype, self.frame)
        self.monodromy_dB        = lambda B, gradB, L, gamma: grad(self.monodromy_jax, argnums=0)(B, gradB, L, gamma)
        self.monodromy_dgradB    = lambda B, gradB, L, gamma: grad(self.monodromy_jax, argnums=1)(B, gradB, L, gamma)
        self.monodromy_dL        = lambda B, gradB, L, gamma: grad(self.monodromy_jax, argnums=2)(B, gradB, L, gamma)
        self.monodromy_dgamma    = lambda B, gradB, L, gamma: grad(self.monodromy_jax, argnums=3)(B, gradB, L, gamma)

        # on-axis iota (and its partials) from the same field-period return map.
        self.iota_jax    = lambda B, gradB, L, gamma: iota_pure(B, gradB, L, gamma, self.D, nfp, self.frame)
        self.iota_dB     = lambda B, gradB, L, gamma: grad(self.iota_jax, argnums=0)(B, gradB, L, gamma)
        self.iota_dgradB = lambda B, gradB, L, gamma: grad(self.iota_jax, argnums=1)(B, gradB, L, gamma)
        self.iota_dL     = lambda B, gradB, L, gamma: grad(self.iota_jax, argnums=2)(B, gradB, L, gamma)
        self.iota_dgamma = lambda B, gradB, L, gamma: grad(self.iota_jax, argnums=3)(B, gradB, L, gamma)


        self.axis = axis
        curve = axis.curve
        self.curve = CurveXYZFourierSymmetries(self.xh, curve.order, curve.nfp, curve.stellsym, ntor=curve.ntor, dofs=curve.dofs)
    
    def recompute_bell(self, parent=None):
        self._monodromy = None
        self._dmonodromy_dcoils = None
        self._matrix = None
        self._iota = None
        self._diota_dcoils = None
    
    @property
    def matrix(self):
        if self._matrix is None:
            axis = self.axis
            curve = self.curve
            biotsavart = self.biotsavart
            
            if axis.need_to_run_code:
                res = axis.res
                axis.run_code(res['length'])

            biotsavart.set_points(curve.gamma())
            B = biotsavart.B()
            gradB = biotsavart.dB_by_dX()
            L = axis.res['length']
            gamma = curve.gamma()
            self._matrix = self.monodromy_matrix(B, gradB, L, gamma)
        return self._matrix
    
    @property
    def monodromy(self):
        if self._monodromy is None:
            self.compute()
        return self._monodromy
    
    @property
    def dmonodromy_dcoils(self):
        if self._dmonodromy_dcoils is None:
            self.compute()
        return self._dmonodromy_dcoils

    def compute(self):
        axis = self.axis
        curve = self.curve
        biotsavart = self.biotsavart
        
        if axis.need_to_run_code:
            res = axis.res
            axis.run_code(res['length'])

        biotsavart.set_points(curve.gamma())
        B = biotsavart.B()
        gradB = biotsavart.dB_by_dX()
        gradgradB = biotsavart.d2B_by_dXdX()
        L = axis.res['length']
        gamma = curve.gamma()
        self._monodromy = self.monodromy_jax(B, gradB, L, gamma)

        dmonodromy_dB = self.monodromy_dB(B, gradB, L, gamma)
        dmonodromy_dgradB = self.monodromy_dgradB(B, gradB, L, gamma)
        dmonodromy_dL = self.monodromy_dL(B, gradB, L, gamma)
        dmonodromy_dgamma_partial = self.monodromy_dgamma(B, gradB, L, gamma)

        Pc, Lc, Uc = axis.res['PLU']
        dmonodromy_dcoils = sum(biotsavart.B_and_dB_vjp(dmonodromy_dB, dmonodromy_dgradB))
        
        dgamma_da = curve.dgamma_by_dcoeff()
        dB_da = np.einsum('ikl,ikm->ilm', gradB, dgamma_da, optimize=True)
        dgradB_da = np.einsum('ijkl,ikm->ijlm', gradgradB, dgamma_da, optimize=True)
        dmonodromy_dgamma = np.einsum('ij,ijm->m', dmonodromy_dB, dB_da, optimize=True) + \
                       np.einsum('ijk,ijkm->m', dmonodromy_dgradB, dgradB_da, optimize=True) + \
                       np.einsum('ik,ikm->m', dmonodromy_dgamma_partial, dgamma_da)
        dJ_ds = np.concatenate([dmonodromy_dgamma, [dmonodromy_dL]])
        
        adj = forward_backward(Pc, Lc, Uc, dJ_ds)
        dmonodromy_dcoils -= axis.res['vjp'](adj, axis.biotsavart, axis)
        self._dmonodromy_dcoils = dmonodromy_dcoils

    @property
    def iota(self):
        if self._iota is None:
            self.compute_iota()
        return self._iota

    @property
    def diota_dcoils(self):
        if self._diota_dcoils is None:
            self.compute_iota()
        return self._diota_dcoils

    def compute_iota(self):
        """On-axis iota and its derivative w.r.t. the coil dofs. Mirrors compute() (the
        monodromy adjoint): partials of iota_pure w.r.t. B/gradB/L/gamma, chained through
        the field's B_and_dB_vjp and the field-line solve adjoint (PLU + vjp)."""
        axis = self.axis
        curve = self.curve
        biotsavart = self.biotsavart

        if axis.need_to_run_code:
            res = axis.res
            axis.run_code(res['length'])

        biotsavart.set_points(curve.gamma())
        B = biotsavart.B()
        gradB = biotsavart.dB_by_dX()
        gradgradB = biotsavart.d2B_by_dXdX()
        L = axis.res['length']
        gamma = curve.gamma()
        self._iota = float(self.iota_jax(B, gradB, L, gamma))

        diota_dB = self.iota_dB(B, gradB, L, gamma)
        diota_dgradB = self.iota_dgradB(B, gradB, L, gamma)
        diota_dL = self.iota_dL(B, gradB, L, gamma)
        diota_dgamma_partial = self.iota_dgamma(B, gradB, L, gamma)

        Pc, Lc, Uc = axis.res['PLU']
        diota_dcoils = sum(biotsavart.B_and_dB_vjp(diota_dB, diota_dgradB))

        dgamma_da = curve.dgamma_by_dcoeff()
        dB_da = np.einsum('ikl,ikm->ilm', gradB, dgamma_da, optimize=True)
        dgradB_da = np.einsum('ijkl,ikm->ijlm', gradgradB, dgamma_da, optimize=True)
        diota_dgamma = np.einsum('ij,ijm->m', diota_dB, dB_da, optimize=True) + \
                       np.einsum('ijk,ijkm->m', diota_dgradB, dgradB_da, optimize=True) + \
                       np.einsum('ik,ikm->m', diota_dgamma_partial, dgamma_da)
        dJ_ds = np.concatenate([diota_dgamma, [diota_dL]])

        adj = forward_backward(Pc, Lc, Uc, dJ_ds)
        diota_dcoils -= axis.res['vjp'](adj, axis.biotsavart, axis)
        self._diota_dcoils = diota_dcoils


    @property
    def elongation(self):
        """Return-map elongation at this tangent map's starting phi: the aspect ratio of
        the invariant ellipse of the one-field-period monodromy R[-1] (i.e. the
        flux-surface cross-section elongation at phi). Built only from the return map
        R[-1] -- see elongation_pure. Non-differentiable (evaluation only). Sweep `phi`
        (see __init__) to obtain the toroidal elongation profile."""
        axis = self.axis
        curve = self.curve
        biotsavart = self.biotsavart
        if axis.need_to_run_code:
            axis.run_code(axis.res['length'])
        biotsavart.set_points(curve.gamma())
        B = biotsavart.B()
        gradB = biotsavart.dB_by_dX()
        L = axis.res['length']
        gamma = curve.gamma()
        return float(elongation_pure(B, gradB, L, gamma, self.D, self.frame))

class Monodromy(Optimizable):
    def __init__(self, tangent_map):
        super().__init__(depends_on=[tangent_map])
        self.tangent_map = tangent_map

    def J(self):
        return self.tangent_map.monodromy
    
    @derivative_dec
    def dJ(self):
        return self.tangent_map.dmonodromy_dcoils

class AxisElongation(Optimizable):
    def __init__(self, tangent_map):
        super().__init__(depends_on=[tangent_map])
        self.tangent_map = tangent_map

    def J(self):
        return self.tangent_map.elongation
    
    @derivative_dec
    def dJ(self):
        return self.tangent_map.delongation_dcoils

class AxisIota(Optimizable):
    def __init__(self, tangent_map):
        super().__init__(depends_on=[tangent_map])
        self.tangent_map = tangent_map

    def J(self):
        return self.tangent_map.iota
    
    @derivative_dec
    def dJ(self):
        return self.tangent_map.diota_dcoils
