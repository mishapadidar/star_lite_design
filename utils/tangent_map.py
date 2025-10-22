import simsoptpp as sopp
from simsopt.field import BiotSavart
from simsopt.geo import ToroidalFlux, Volume, CurveXYZFourierSymmetries
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.objectives.utilities import forward_backward
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacfwd

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

def angle_pure(B, gradB, L, gamma, D):
    R = monodromy_pure(B, gradB, L, gamma, D)
    Rf = R[-1]
    a = Rf[0, 0]
    b = Rf[0, 1]
    c = Rf[1, 0]
    d = Rf[1, 1]
    
    eig1 = ((a+d) + jnp.sqrt((a-d)**2 + 4*b*c)) / 2.
    eig2 = ((a+d) - jnp.sqrt((a-d)**2 + 4*b*c)) / 2.
    v1 = jnp.array([-b, a-eig1])
    v2 = jnp.array([-b, a-eig2])
    v1 /= jnp.linalg.norm(v1)
    v2 /= jnp.linalg.norm(v2)
    theta1 = jnp.arccos(jnp.dot(v1, v2))
    theta2 = jnp.pi - theta1
    return jnp.min(jnp.array([theta1, theta2]))
    
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

#print("theta is", theta[-1])
#theta = jnp.atan2(eigs.imag, eigs.real)/np.pi
def winding_pure(B, gradB, L, gamma, D, wh, nfp):
    eigs1, eigs2 = eigenvalues_pure(B, gradB, L, gamma, D)
    #eigs_normalized = eigs/jnp.abs(eigs)
    #winding = nfp*jnp.sum(wh*(D@eigs_normalized)/eigs_normalized)/(2*jnp.pi*1j)
    #return winding.real
    winding = nfp*jnp.atan2(eigs1[-1].imag, eigs1[-1].real)/(2*np.pi)
    return winding



class XTangentMap(Optimizable):
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
        N = 2*axis.curve.order+1  # Number of intervals (N+1 grid points)
        D, xh, wh = cheb(N, 0, 1./nfp)
        self.D = D
        self.xh = xh
        self.wh = wh
        
        self.angle_jax       = jit(lambda B, gradB, L, gamma: angle_pure(B, gradB, L, gamma, self.D))
        self.angle_dB        = jit(lambda B, gradB, L, gamma: grad(self.angle_jax, argnums=0)(B, gradB, L, gamma))
        self.angle_dgradB    = jit(lambda B, gradB, L, gamma: grad(self.angle_jax, argnums=1)(B, gradB, L, gamma))
        self.angle_dL        = jit(lambda B, gradB, L, gamma: grad(self.angle_jax, argnums=2)(B, gradB, L, gamma))
        self.angle_dgamma    = jit(lambda B, gradB, L, gamma: grad(self.angle_jax, argnums=3)(B, gradB, L, gamma))
        
        self.axis = axis
        curve = axis.curve
        self.curve = CurveXYZFourierSymmetries(self.xh, curve.order, curve.nfp, curve.stellsym, ntor=curve.ntor, dofs=curve.dofs)
    
    def recompute_bell(self, parent=None):
        self._angle = None
        self._dangle_dcoils = None
    
    @property
    def angle(self):
        if self._angle is None:
            self.compute()
        return self._angle
    
    @property
    def dangle_dcoils(self):
        if self._dangle_dcoils is None:
            self.compute()
        return self._dangle_dcoils

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
        self._angle = self.angle_jax(B, gradB, L, gamma)

        dangle_dB = self.angle_dB(B, gradB, L, gamma)
        dangle_dgradB = self.angle_dgradB(B, gradB, L, gamma)
        dangle_dL = self.angle_dL(B, gradB, L, gamma)
        dangle_dgamma_partial = self.angle_dgamma(B, gradB, L, gamma)

        Pc, Lc, Uc = axis.res['PLU']
        dangle_dcoils = sum(biotsavart.B_and_dB_vjp(dangle_dB, dangle_dgradB))
        
        dgamma_da = curve.dgamma_by_dcoeff()
        dB_da = np.einsum('ikl,ikm->ilm', gradB, dgamma_da, optimize=True)
        dgradB_da = np.einsum('ijkl,ikm->ijlm', gradgradB, dgamma_da, optimize=True)
        dangle_dgamma = np.einsum('ij,ijm->m', dangle_dB, dB_da, optimize=True) + \
                       np.einsum('ijk,ijkm->m', dangle_dgradB, dgradB_da, optimize=True) + \
                       np.einsum('ik,ikm->m', dangle_dgamma_partial, dgamma_da)
        dJ_ds = np.concatenate([dangle_dgamma, [dangle_dL]])
        
        adj = forward_backward(Pc, Lc, Uc, dJ_ds)
        dangle_dcoils -= axis.res['vjp'](adj, axis.biotsavart, axis)
        self._dangle_dcoils = dangle_dcoils

class OTangentMap(Optimizable):
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
        N = 2*axis.curve.order+1  # Number of intervals (N+1 grid points)
        D, xh, wh = cheb(N, 0, 1./nfp)
        self.D = D
        self.xh = xh
        self.wh = wh
        
        self.winding_jax       = jit(lambda B, gradB, L, gamma: winding_pure(B, gradB, L, gamma, self.D, self.wh, nfp))
        self.winding_dB        = jit(lambda B, gradB, L, gamma: grad(self.winding_jax, argnums=0)(B, gradB, L, gamma))
        self.winding_dgradB    = jit(lambda B, gradB, L, gamma: grad(self.winding_jax, argnums=1)(B, gradB, L, gamma))
        self.winding_dL        = jit(lambda B, gradB, L, gamma: grad(self.winding_jax, argnums=2)(B, gradB, L, gamma))
        self.winding_dgamma    = jit(lambda B, gradB, L, gamma: grad(self.winding_jax, argnums=3)(B, gradB, L, gamma))
        self.elongation_jax    = jit(lambda B, gradB, L, gamma: elongation_pure(B, gradB, L, gamma, self.D))
        self.elongation_dB     = jit(lambda B, gradB, L, gamma: grad(self.elongation_jax, argnums=0)(B, gradB, L, gamma))
        self.elongation_dgradB = jit(lambda B, gradB, L, gamma: grad(self.elongation_jax, argnums=1)(B, gradB, L, gamma))
        self.elongation_dL     = jit(lambda B, gradB, L, gamma: grad(self.elongation_jax, argnums=2)(B, gradB, L, gamma))
        self.elongation_dgamma = jit(lambda B, gradB, L, gamma: grad(self.elongation_jax, argnums=3)(B, gradB, L, gamma))

        self.axis = axis
        curve = axis.curve
        self.curve = CurveXYZFourierSymmetries(self.xh, curve.order, curve.nfp, curve.stellsym, ntor=curve.ntor, dofs=curve.dofs)
    
    def recompute_bell(self, parent=None):
        self._elongation = None
        self._iota = None
        self._diota_dcoils = None
        self._delongation_dcoils = None
    
    @property
    def elongation(self):
        if self._elongation is None:
            self.compute()
        return self._elongation

    @property
    def iota(self):
        if self._iota is None:
            self.compute()
        return self._iota
    
    @property
    def delongation_dcoils(self):
        if self._delongation_dcoils is None:
            self.compute()
        return self._delongation_dcoils
    
    @property
    def diota_dcoils(self):
        if self._diota_dcoils is None:
            self.compute()
        return self._diota_dcoils

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
        self._iota = self.winding_jax(B, gradB, L, gamma)
        self._elongation = self.elongation_jax(B, gradB, L, gamma)

        diota_dB = self.winding_dB(B, gradB, L, gamma)
        diota_dgradB = self.winding_dgradB(B, gradB, L, gamma)
        diota_dL = self.winding_dL(B, gradB, L, gamma)
        diota_dgamma_partial = self.winding_dgamma(B, gradB, L, gamma)

        delongation_dB = self.elongation_dB(B, gradB, L, gamma)
        delongation_dgradB = self.elongation_dgradB(B, gradB, L, gamma)
        delongation_dL = self.elongation_dL(B, gradB, L, gamma)
        delongation_dgamma_partial = self.elongation_dgamma(B, gradB, L, gamma)

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

        delongation_dcoils = sum(biotsavart.B_and_dB_vjp(delongation_dB, delongation_dgradB))
        
        delongation_dgamma = np.einsum('ij,ijm->m', delongation_dB, dB_da, optimize=True) + \
                             np.einsum('ijk,ijkm->m', delongation_dgradB, dgradB_da, optimize=True) + \
                             np.einsum('ik,ikm->m', delongation_dgamma_partial, dgamma_da)
        dJ_ds = np.concatenate([delongation_dgamma, [delongation_dL]])
        adj = forward_backward(Pc, Lc, Uc, dJ_ds)
        delongation_dcoils -= axis.res['vjp'](adj, axis.biotsavart, axis)
        self._delongation_dcoils = delongation_dcoils

class Xangle(Optimizable):
    def __init__(self, tangent_map):
        super().__init__(depends_on=[tangent_map])
        self.tangent_map = tangent_map

    def J(self):
        return self.tangent_map.angle
    
    @derivative_dec
    def dJ(self):
        return self.tangent_map.dangle_dcoils

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
