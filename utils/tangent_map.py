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
   
    n = np.arange(0, N//2 + 1)[:, None]  # column vector
    k = np.arange(0, N//2 + 1)[None, :]  # row vector

    DD = 2 * np.cos(2 * np.pi * n * k / N) / N
    DD[0, :] *= 0.5

    d = np.concatenate(([1.0], 2.0 / (1.0 - np.square(np.arange(2, N + 1, 2)))))
    w = DD @ d
    w = np.concatenate((w, np.flip(w[:-1])))
    
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



# FROM CHATGPT
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


def quadratic_jet_pure(B, gradB, gradgradB, L, gamma, D):
    """
    Quadratic jet of the reduced 2D map in the normal-binormal plane.

    Returns
    -------
    K : (N,2,2,2)
        K[s, :, a, b] is the quadratic jet at collocation point s,
        projected into the local (N,B) frame, with initial coordinates
        taken in the (N,B) frame at s=0.

        The final return-map quadratic jet is K[-1].
    """
    tangent = jnp.matmul(D, gamma)
    fT = tangent / jnp.linalg.norm(tangent, axis=-1)[:, None]

    normal = jnp.matmul(D, fT)
    fN = normal / jnp.linalg.norm(normal, axis=-1)[:, None]

    binormal = jnp.cross(fT, fN)
    fB = binormal / jnp.linalg.norm(binormal, axis=-1)[:, None]

    NB = jnp.concatenate((fN[:, :, None], fB[:, :, None]), axis=-1)     # (N,3,2)
    NB_t = jnp.concatenate((fN[:, None, :], fB[:, None, :]), axis=-2)   # (N,2,3)

    # Full 3x3 tangent map in Cartesian coordinates
    Tfull = tangent_map_pure(B, gradB, L, D)                             # (N,3,3)

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
            ia = NB[0, :, a]   # initial Cartesian direction for reduced coord a
            ib = NB[0, :, b]   # initial Cartesian direction for reduced coord b

            Ta = jnp.einsum('nij,j->ni', Tfull, ia)   # first variation along ia
            Tb = jnp.einsum('nij,j->ni', Tfull, ib)   # first variation along ib

            rhs = -second_var_residual_pure(Q0, B, gradB, gradgradB, L, Ta, Tb, D).ravel()
            Qab = jnp.linalg.solve(Lop, rhs).reshape((Npts, 3))

            # project final displacement into local (N,B) frame
            Kab = jnp.einsum('nij,nj->ni', NB_t, Qab)   # (N,2)
            K = K.at[:, :, a, b].set(Kab)

    return K


def quadratic_jet_matrix_pure(B, gradB, gradgradB, L, gamma, D):
    return quadratic_jet_pure(B, gradB, gradgradB, L, gamma, D)[-1]

# FROM CHATGPT^














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

def monodromy_pure(B, gradB, L, gamma, D):
    tangent = jnp.matmul(D, gamma)
    fT = tangent/jnp.linalg.norm(tangent, axis=-1)[:, None]
    
    normal =  jnp.matmul(D, fT)
    fN = normal/jnp.linalg.norm(normal, axis=-1)[:, None]
    
    binormal = jnp.cross(fT, fN)
    fB = binormal/jnp.linalg.norm(binormal, axis=-1)[:, None]
    

    NB = jnp.concatenate((fN[:, :, None], fB[:, :, None]), axis=-1)
    NB_t = jnp.concatenate((fN[:, None, :], fB[:, None, :]), axis=-2)

    M = tangent_map_pure(B, gradB, L, D)
    R = jnp.matmul(NB_t, jnp.matmul(M, NB[0]))
    return R

def monodromy_identity_pure(B, gradB, L, gamma, D):
    R = monodromy_pure(B, gradB, L, gamma, D)
    Rf=R[-1]
    return jnp.mean((Rf-jnp.eye(2))**2)
def monodromy_matrix_pure(B, gradB, L, gamma, D):
    R = monodromy_pure(B, gradB, L, gamma, D)
    Rf=R[-1]
    return Rf



def eigenvalues_pure(B, gradB, L, gamma, D):
    R = monodromy_pure(B, gradB, L, gamma, D).astype(complex)
    #tr = jnp.trace(R, axis1=1, axis2=2).astype(complex)
    # skip the nondifferentiable tr=2 in the IC
    a = R[1:, 0, 0]
    b = R[1:, 0, 1]
    c = R[1:, 1, 0]
    d = R[1:, 1, 1]

    eigs1 = ((a+d) + jnp.sqrt((a-d)**2 + 4*b*c)) / 2.
    eigs2 = ((a+d) - jnp.sqrt((a-d)**2 + 4*b*c)) / 2.
    return eigs1, eigs2

def elongation_pure(B, gradB, L, gamma, D):
    R = monodromy_pure(B, gradB, L, gamma, D)
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

    a, b = S[0]
    c, d = S[1]
    eig1 = ((a+d) + jnp.sqrt((a-d)**2 + 4*b*c)) / 2.
    eig2 = ((a+d) - jnp.sqrt((a-d)**2 + 4*b*c)) / 2.
    elong = jnp.abs(eig1/eig2)
    return jnp.max(jnp.array([elong, 1/elong]))

class TangentMap(Optimizable):
    def __init__(self, axis, biotsavart):
        """
        Evaluate the the tangent map on a fieldline

        Args:
        """
        super().__init__(depends_on=[axis])
        self.biotsavart = biotsavart
        
        nfp = axis.curve.nfp
        #print("when integrating from 0 to 0.5, check the eigenvectors are correct")
        # Example usage:
        N = 5*axis.curve.order+1  # Number of intervals (N+1 grid points)
        D, xh, wh = cheb(N, 0, 1./nfp)
        self.D = D
        self.xh = xh
        self.wh = wh

        self.monodromy_matrix      = lambda B, gradB, L, gamma: monodromy_matrix_pure(B, gradB, L, gamma, self.D)
        self.monodromy_jax       = lambda B, gradB, L, gamma: monodromy_identity_pure(B, gradB, L, gamma, self.D)
        self.monodromy_dB        = lambda B, gradB, L, gamma: grad(self.monodromy_jax, argnums=0)(B, gradB, L, gamma)
        self.monodromy_dgradB    = lambda B, gradB, L, gamma: grad(self.monodromy_jax, argnums=1)(B, gradB, L, gamma)
        self.monodromy_dL        = lambda B, gradB, L, gamma: grad(self.monodromy_jax, argnums=2)(B, gradB, L, gamma)
        self.monodromy_dgamma    = lambda B, gradB, L, gamma: grad(self.monodromy_jax, argnums=3)(B, gradB, L, gamma)

        self.quadratic_jet_matrix = lambda B, gradB, gradgradB, L, gamma: quadratic_jet_matrix_pure(B, gradB, gradgradB, L, gamma, self.D)

        self.axis = axis
        curve = axis.curve
        self.curve = CurveXYZFourierSymmetries(self.xh, curve.order, curve.nfp, curve.stellsym, ntor=curve.ntor, dofs=curve.dofs)
    
    def recompute_bell(self, parent=None):
        self._monodromy = None
        self._dmonodromy_dcoils = None
        self._matrix = None
        self._jet2 = None 
    
    @property
    def jet2(self):
        if self._jet2 is None:
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
            self._jet2 = self.quadratic_jet_matrix(B, gradB, gradgradB, L, gamma)
            self._matrix = self.monodromy_matrix(B, gradB, L, gamma)
        return self._jet2
    
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
            gradgradB = biotsavart.d2B_by_dXdX()
            L = axis.res['length']
            gamma = curve.gamma()
            self._jet2 = self.quadratic_jet_matrix(B, gradB, gradgradB, L, gamma)
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
        

    def snowflake_angles_from_jet2(self, ntheta=512, xtol=1e-12, tangent_tol=1e-6):
        """
        Roots of f(θ) = v × K(v,v) with v = (cos θ, sin θ) — snowflake leg directions.
    
        Refines each bracketed sign change with brentq (machine-precision roots),
        and also detects tangent (double) zeros that lie at sign-preserving local
        minima of |f|, which the sign-change method misses near bifurcations.
    
        Returns
        -------
        np.ndarray of root angles in [0, 2π), sorted ascending.
        """
        from scipy.optimize import brentq, minimize_scalar
    
        K = np.asarray(self.jet2)
    
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
                                      method='bounded',
                                      options={'xatol': xtol})
                if abs(f(res.x)) / scale < tangent_tol:
                    _add(res.x)
    
        return np.sort(np.asarray(roots))

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
