import simsoptpp as sopp
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
import numpy as np
import jax.numpy as jnp
from jax import jit, grad


def cv_distance_pure(gammac, lc, gammas, minimum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-surface distance
    formula.
    """
    dists = jnp.sqrt(jnp.sum(
        (gammac[:, None, :] - gammas[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance-dists, 0)**2)


class CurveVesselDistance(Optimizable):
    def __init__(self, curves, vessel, minimum_distance):
        """A curve-vessel distance function

        Args:
            curves (list): list of curves
            vessel (array): (n, 3) array representing the vessel surface
            minimum_distance (float): minimum distance threshold for the distance calculation.
        """
        self.curves = curves
        self.vessel = vessel
        self.minimum_distance = minimum_distance

        self.J_jax = jit(lambda gammac, lc, gammas: cv_distance_pure(gammac, lc, gammas, minimum_distance))
        self.thisgrad0 = jit(lambda gammac, lc, gammas: grad(self.J_jax, argnums=0)(gammac, lc, gammas))
        self.thisgrad1 = jit(lambda gammac, lc, gammas: grad(self.J_jax, argnums=1)(gammac, lc, gammas))
        self.candidates = None
        super().__init__(depends_on=curves)  # Bharat's comment: Shouldn't we add surface here

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
                [c.gamma() for c in self.curves], [self.vessel.reshape((-1, 3))], self.minimum_distance)
            self.candidates = candidates

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.vessel.reshape((-1, 3))
        return min([self.minimum_distance] + [np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i, _ in self.candidates])

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()
        from scipy.spatial.distance import cdist
        xyz_surf = self.vessel.reshape((-1, 3))
        return min([np.min(cdist(self.curves[i].gamma(), xyz_surf)) for i in range(len(self.curves))])

    def J(self):
        """
        This returns the value of the quantity.
        """
        self.compute_candidates()
        res = 0
        gammas = self.vessel.reshape((-1, 3))
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            res += self.J_jax(gammac, lc, gammas)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        self.compute_candidates()
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]
        gammas = self.vessel.reshape((-1, 3))

        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gammac, lc, gammas)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gammac, lc, gammas)
        res = [self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) + self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i]) for i in range(len(self.curves))]
        return sum(res)

    return_fn_map = {'J': J, 'dJ': dJ}