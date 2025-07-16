import simsoptpp as sopp
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from simsopt.objectives import forward_backward


def flv_distance_pure(gammac, lc, gammas, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-surface distance
    formula.
    """
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance-dists, 0)**2)


class FieldLineVesselDistance(Optimizable):
    def __init__(self, fieldline, vessel, minimum_distance):
        self.fieldline = fieldline
        self.vessel = vessel
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gammac, lc, gammas: flv_distance_pure(gammac, lc, gammas, minimum_distance))
        self.thisgrad0 = jit(lambda gammac, lc, gammas: grad(self.J_jax, argnums=0)(gammac, lc, gammas))
        self.thisgrad1 = jit(lambda gammac, lc, gammas: grad(self.J_jax, argnums=1)(gammac, lc, gammas))
        self.candidates = None
        super().__init__(depends_on=[fieldline])  # Bharat's comment: Shouldn't we add surface here

    def recompute_bell(self, parent=None):
        self.candidates = None

    def shortest_distance(self):
        from scipy.spatial.distance import cdist
        xyz_surf = self.vessel.reshape((-1, 3))
        return np.min(cdist(self.fieldline.curve.gamma(), xyz_surf))
        # why is self.min_dist included in the original penalty?

    def J(self):
        """
        This returns the value of the quantity.
        """
        if self.fieldline.need_to_run_code:
            res = self.fieldline.res
            res = self.fieldline.run_code(res['length'])

        gammas = self.vessel.reshape((-1, 3))
        gammac = self.fieldline.curve.gamma()
        lc = self.fieldline.curve.gammadash()
        res = self.J_jax(gammac, lc, gammas)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        if self.fieldline.need_to_run_code:
            res = self.fieldline.res
            res = self.fieldline.run_code(res['length'])

        curve = self.fieldline.curve
        dgamma_by_dcoeff_vjp_vecs = np.zeros_like(curve.gamma())
        dgammadash_by_dcoeff_vjp_vecs = np.zeros_like(curve.gammadash())
        gammas = self.vessel.reshape((-1, 3))
        gammac = self.fieldline.curve.gamma()
        lc = self.fieldline.curve.gammadash()
        dgamma_by_dcoeff_vjp_vecs = self.thisgrad0(gammac, lc, gammas)
        dgammadash_by_dcoeff_vjp_vecs = self.thisgrad1(gammac, lc, gammas)
        res_curve = curve.dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs) + curve.dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs)
        res_curve = res_curve(self.fieldline.curve)
        
        fieldline = self.fieldline
        P, L, U = fieldline.res['PLU']
        dconstraint_dcoils_vjp = fieldline.res['vjp']

        # tack on dJ_dlength = 0 to the end of dJ_ds
        dJ_dc = np.zeros(L.shape[0])
        dj_dc = res_curve.copy()
        dJ_dc[:dj_dc.size] = dj_dc
        adj = forward_backward(P, L, U, dJ_dc)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, fieldline.biotsavart, fieldline)
        res = -1 * adj_times_dg_dcoil
        return res

    return_fn_map = {'J': J, 'dJ': dJ}



