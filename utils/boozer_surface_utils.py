import jax.numpy as jnp
from jax import grad
import numpy as np
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.geo.jit import jit
from simsopt.geo.surfaceobjectives import boozer_surface_residual, boozer_surface_residual_dB
import simsoptpp as sopp
from simsopt.objectives import forward_backward

class BoozerResidual(Optimizable):
    r"""
    This term returns the Boozer residual penalty term
    
    .. math::
       J = \int_0^{1/n_{\text{fp}}} \int_0^1 \| \mathbf r \|^2 ~d\theta ~d\varphi + w (\text{label.J()-boozer_surface.constraint_weight})^2.
    
    where
    
    .. math::
        \mathbf r = \frac{1}{\|\mathbf B\|}[G\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)]
    
    """

    def __init__(self, boozer_surface, bs):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface
        
        # same number of points as on the solved surface
        phis = np.linspace(0, 1/in_surface.nfp, in_surface.quadpoints_phi.size*4, endpoint=False)
        thetas =  np.linspace(0, 1., in_surface.quadpoints_theta.size*4, endpoint=False)

        s = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas)
        s.set_dofs(in_surface.get_dofs())

        print('set constraint weight to 0')
        self.constraint_weight = 0.
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        """
        Return the value of the penalty function.
        """
        
        if self._J is None:
            self.compute()
        return self._J
    
    @derivative_dec
    def dJ(self):
        """
        Return the derivative of the penalty function with respect to the coil degrees of freedom.
        """

        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['iota'], G=res['G'])

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))
 
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        # compute J
        surface = self.surface
        iota = self.boozer_surface.res['iota']
        G = self.boozer_surface.res['G']
        r, J = boozer_surface_residual(surface, iota, G, self.biotsavart, derivatives=1, weight_inv_modB=True)
        rtil = np.concatenate((r/np.sqrt(num_points), [np.sqrt(self.constraint_weight)*(self.boozer_surface.label.J()-self.boozer_surface.targetlabel)]))
        self._J = 0.5*np.sum(rtil**2)
        
        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = booz_surf.res['vjp']

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[:dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate((J/np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]), axis=0)
        dJ_ds = Jtil.T@rtil
        
        adj = forward_backward(P, L, U, dJ_ds)
        
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil
        
    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        
        surface = self.surface
        res = self.boozer_surface.res
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        r, r_dB = boozer_surface_residual_dB(surface, self.boozer_surface.res['iota'], self.boozer_surface.res['G'], self.biotsavart, derivatives=0, weight_inv_modB=True)

        r /= np.sqrt(num_points)
        r_dB /= np.sqrt(num_points)
        
        dJ_by_dB = r[:, None]*r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB

def cbs_distance_pure(gammac, lc, gammas, ls1, ls2, minimum_distance, p):
    """
    This function is used in a Python+Jax implementation of the curve-surface distance
    formula.
    """
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None] \
        * jnp.linalg.norm(jnp.cross(ls1, ls2, axis=-1), axis=1)[None, :]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance-dists, 0)**p)

class CurveBoozerSurfaceDistance(Optimizable):
    r"""
    CurveSurfaceDistance is a class that computes

    .. math::
        J = \sum_{i = 1}^{\text{num_coils}} d_{i}

    where

    .. math::
        d_{i} = \int_{\text{curve}_i} \int_{surface} \max(0, d_{\min} - \| \mathbf{r}_i - \mathbf{s} \|_2)^2 ~dl_i ~ds\\

    and :math:`\mathbf{r}_i`, :math:`\mathbf{s}` are points on coil :math:`i`
    and the surface, respectively. :math:`d_\min` is a desired threshold
    minimum coil-to-surface distance.  This penalty term is zero when the
    points on all coils :math:`i` and on the surface lie more than
    :math:`d_\min` away from one another.

    """

    def __init__(self, curves, boozer_surface, minimum_distance, p=2):
        self.curves = curves
        self.boozer_surface = boozer_surface
        self.surface = boozer_surface.surface
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gammac, lc, gammas, ls1, ls2: cbs_distance_pure(gammac, lc, gammas, ls1, ls2, minimum_distance, p))
        self.thisgrad0 = jit(lambda gammac, lc, gammas, ls1, ls2: grad(self.J_jax, argnums=0)(gammac, lc, gammas, ls1, ls2))
        self.thisgrad1 = jit(lambda gammac, lc, gammas, ls1, ls2: grad(self.J_jax, argnums=1)(gammac, lc, gammas, ls1, ls2))
        self.thisgrad2 = jit(lambda gammac, lc, gammas, ls1, ls2: grad(self.J_jax, argnums=2)(gammac, lc, gammas, ls1, ls2))
        self.thisgrad3 = jit(lambda gammac, lc, gammas, ls1, ls2: grad(self.J_jax, argnums=3)(gammac, lc, gammas, ls1, ls2))
        self.thisgrad4 = jit(lambda gammac, lc, gammas, ls1, ls2: grad(self.J_jax, argnums=4)(gammac, lc, gammas, ls1, ls2))
        self.candidates = None
        super().__init__(depends_on=curves+[boozer_surface])  # Bharat's comment: Shouldn't we add surface here

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
                [c.gamma() for c in self.curves], [self.surface.gamma().reshape((-1, 3))], self.minimum_distance)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i, _ in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.surface.gamma().reshape((-1, 3))
        return min([np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i in range(len(self.curves))])

    def J(self):
        """
        This returns the value of the quantity.
        """
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['type'], res['iota'], G=res['G'])


        self.compute_candidates()
        res = 0
        gammas = self.surface.gamma().reshape((-1, 3))
        ls1 = self.surface.gammadash1().reshape((-1, 3))
        ls2 = self.surface.gammadash2().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            res += self.J_jax(gammac, lc, gammas, ls1, ls2)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res['type'], res['iota'], G=res['G'])
        
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size

        # partial wrt to coils
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]
        
        dgammas_by_dcoeff_vjp_vecs     = [np.zeros_like(self.surface.gamma()) for c in self.curves]
        dgammadash1_by_dcoeff_vjp_vecs = [np.zeros_like(self.surface.gammadash1()) for c in self.curves]
        dgammadash2_by_dcoeff_vjp_vecs = [np.zeros_like(self.surface.gammadash2()) for c in self.curves]

        gammas = self.surface.gamma().reshape((-1, 3))
        ls1 = self.surface.gammadash1().reshape((-1, 3))
        ls2 = self.surface.gammadash2().reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gammac, lc, gammas, ls1, ls2)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gammac, lc, gammas, ls1, ls2)

            dgammas_by_dcoeff_vjp_vecs[i] += self.thisgrad2(gammac, lc, gammas, ls1, ls2).reshape((nphi, ntheta, -1))
            dgammadash1_by_dcoeff_vjp_vecs[i] += self.thisgrad3(gammac, lc, gammas, ls1, ls2).reshape((nphi, ntheta, -1))
            dgammadash2_by_dcoeff_vjp_vecs[i] += self.thisgrad4(gammac, lc, gammas, ls1, ls2).reshape((nphi, ntheta, -1))

        res_curve = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        res_surface = [self.surface.dgamma_by_dcoeff_vjp(dgammas_by_dcoeff_vjp_vecs[i]) + self.surface.dgammadash1_by_dcoeff_vjp(dgammadash1_by_dcoeff_vjp_vecs[i]) + self.surface.dgammadash2_by_dcoeff_vjp(dgammadash2_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        res_curve = sum(res_curve)
        res_surface = sum(res_surface)

        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = self.boozer_surface.res['vjp']

        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.zeros(L.shape[0])
        dj_ds = res_surface.copy()
        dJ_ds[:dj_ds.size] = dj_ds
        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        res = res_curve -1 * adj_times_dg_dcoil

        return res

    return_fn_map = {'J': J, 'dJ': dJ}