import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # float64 (JAX defaults to float32); also set in utils/__init__.py
import jax.numpy as jnp
from jax import jit, grad

import simsoptpp as sopp
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.objectives import forward_backward


def cpf_distance_pure(gammac, lc, gammafl, minimum_distance, p=2):
    """
    Pure JAX kernel for curve-to-periodic-fieldline distance penalty.
    """
    dists = jnp.sqrt(jnp.sum((gammac[:, None, :] - gammafl[None, :, :])**2, axis=2))
    integralweight = jnp.linalg.norm(lc, axis=1)[:, None]
    return jnp.mean(integralweight * jnp.maximum(minimum_distance - dists, 0) ** p)


class CurvesPeriodicFieldlineDistance(Optimizable):
    r"""
    Penalizes the distance between many curves and one periodic fieldline:

    .. math::
        J = \sum_{i=1}^{\mathrm{num\_curves}} d_i

    where

    .. math::
        d_i = \int_{\mathrm{curve}_i} \int_{\mathrm{pfl}}
              \max(0, d_{\min} - \|\mathbf{r}_i - \mathbf{r}_{fl}\|_2)^p
              ~dl_i~d\ell_{fl}

    In the discrete implementation, the fieldline integral is represented by
    an average over sampled fieldline points, while the curve gets the usual
    |gamma'| weight, matching the style of the existing classes.
    """

    def __init__(self, curves, periodic_fieldline, minimum_distance, p=2):
        self.curves = curves
        self.periodic_fieldline = periodic_fieldline
        self.minimum_distance = minimum_distance
        self.p = p

        self.J_jax = jit(
            lambda gammac, lc, gammafl:
                cpf_distance_pure(gammac, lc, gammafl, minimum_distance, p)
        )
        self.thisgrad0 = jit(lambda gammac, lc, gammafl: grad(self.J_jax, argnums=0)(gammac, lc, gammafl))
        self.thisgrad1 = jit(lambda gammac, lc, gammafl: grad(self.J_jax, argnums=1)(gammac, lc, gammafl))
        self.thisgrad2 = jit(lambda gammac, lc, gammafl: grad(self.J_jax, argnums=2)(gammac, lc, gammafl))

        self.candidates = None
        super().__init__(depends_on=curves + [periodic_fieldline])

    def recompute_bell(self, parent=None):
        self.candidates = None

    def compute_candidates(self):
        if self.candidates is None:
            gammafl = self.periodic_fieldline.curve.gamma()
            self.candidates = sopp.get_pointclouds_closer_than_threshold_between_two_collections(
                [c.gamma() for c in self.curves],
                [gammafl],
                self.minimum_distance
            )

    def shortest_distance_among_candidates(self):
        self.compute_candidates()
        from scipy.spatial.distance import cdist
        gammafl = self.periodic_fieldline.curve.gamma()
        return min(
            [self.minimum_distance] +
            [np.min(cdist(self.curves[i].gamma(), gammafl)) for i, _ in self.candidates]
        )

    def shortest_distance(self):
        self.compute_candidates()
        if len(self.candidates) > 0:
            return self.shortest_distance_among_candidates()

        from scipy.spatial.distance import cdist
        gammafl = self.periodic_fieldline.curve.gamma()
        return min(np.min(cdist(c.gamma(), gammafl)) for c in self.curves)

    def J(self):
        """
        Returns the value of the quantity.
        """
        if self.periodic_fieldline.need_to_run_code:
            res = self.periodic_fieldline.res
            self.periodic_fieldline.run_code(res['length'])

        self.compute_candidates()

        gammafl = self.periodic_fieldline.curve.gamma()
        res = 0.0
        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()
            res += self.J_jax(gammac, lc, gammafl)
        return res

    @derivative_dec
    def dJ(self):
        """
        Returns the derivative with respect to the upstream optimization dofs.
        """
        if self.periodic_fieldline.need_to_run_code:
            res = self.periodic_fieldline.res
            self.periodic_fieldline.run_code(res['length'])

        self.compute_candidates()

        gammafl = self.periodic_fieldline.curve.gamma()

        # partial wrt explicit curves
        dgamma_by_dcoeff_vjp_vecs = [np.zeros_like(c.gamma()) for c in self.curves]
        dgammadash_by_dcoeff_vjp_vecs = [np.zeros_like(c.gammadash()) for c in self.curves]

        # accumulate derivative wrt periodic fieldline curve
        dgammafl_total = np.zeros_like(gammafl)

        for i, _ in self.candidates:
            gammac = self.curves[i].gamma()
            lc = self.curves[i].gammadash()

            dgamma_by_dcoeff_vjp_vecs[i] += self.thisgrad0(gammac, lc, gammafl)
            dgammadash_by_dcoeff_vjp_vecs[i] += self.thisgrad1(gammac, lc, gammafl)
            dgammafl_total += self.thisgrad2(gammac, lc, gammafl)

        res_curve = [
            self.curves[i].dgamma_by_dcoeff_vjp(dgamma_by_dcoeff_vjp_vecs[i]) +
            self.curves[i].dgammadash_by_dcoeff_vjp(dgammadash_by_dcoeff_vjp_vecs[i])
            for i in range(len(self.curves))
        ]
        res_curve = sum(res_curve)

        # map dJ/d(gammafl) to dJ/d(fieldline-state)
        res_pfl_curve = self.periodic_fieldline.curve.dgamma_by_dcoeff_vjp(dgammafl_total)
        res_pfl_curve = res_pfl_curve(self.periodic_fieldline.curve)

        pfl = self.periodic_fieldline
        P, L, U = pfl.res['PLU']
        dconstraint_dcoils_vjp = pfl.res['vjp']

        # tack on dJ_dlength = 0 at the end
        dJ_dc = np.zeros(L.shape[0])
        dj_dc = res_pfl_curve.copy()
        dJ_dc[:dj_dc.size] = dj_dc

        adj = forward_backward(P, L, U, dJ_dc)

        # same convention as your FieldLineVesselDistance
        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, pfl.biotsavart, pfl)

        res = res_curve - adj_times_dg_dcoil
        return res

    return_fn_map = {'J': J, 'dJ': dJ}
