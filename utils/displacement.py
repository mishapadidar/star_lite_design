import simsoptpp as sopp
from simsopt.field import BiotSavart
from simsopt.geo import ToroidalFlux, Volume, CurveXYZFourierSymmetries
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.objectives.utilities import forward_backward
import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacfwd

def distance_pure(gamma1, gamma2, maximum_distance):
    """
    This function is used in a Python+Jax implementation of the curve-surface distance
    formula.
    """
    dists = jnp.linalg.norm(gamma1-gamma2, axis=-1)
    return jnp.mean(jnp.maximum(dists-maximum_distance, 0)**2)


class FieldLineDistance(Optimizable):
    def __init__(self, fieldline1, fieldline2, maximum_distance):
        """A curve-vessel distance function

        Args:
            curves (list): list of curves
            vessel (array): (n, 3) array representing the vessel surface
            minimum_distance (float): minimum distance threshold for the distance calculation.
        """
        self.fieldline1 = fieldline1
        self.fieldline2 = fieldline2
        self.maximum_distance = maximum_distance

        self.J_jax = jit(lambda gamma1, gamma2: distance_pure(gamma1, gamma2, maximum_distance))
        self.thisgrad0 = jit(lambda gamma1, gamma2: grad(self.J_jax, argnums=0)(gamma1, gamma2))
        self.thisgrad1 = jit(lambda gamma1, gamma2: grad(self.J_jax, argnums=1)(gamma1, gamma2))
        super().__init__(depends_on=[fieldline1, fieldline2])  # Bharat's comment: Shouldn't we add surface here
    def max_distance(self):
        gamma1 = self.fieldline1.curve.gamma()
        gamma2 = self.fieldline2.curve.gamma()
        return np.max(np.linalg.norm(gamma1-gamma2, axis=-1))

    def J(self):
        """
        This returns the value of the quantity.
        """
        if self.fieldline1.need_to_run_code:
            res = self.fieldline1.res
            res = self.fieldline1.run_code(res['length'])
        if self.fieldline2.need_to_run_code:
            res = self.fieldline2.res
            res = self.fieldline2.run_code(res['length'])

        gamma1 = self.fieldline1.curve.gamma()
        gamma2 = self.fieldline2.curve.gamma()
        
        res = self.J_jax(gamma1, gamma2)
        return res

    @derivative_dec
    def dJ(self):
        """
        This returns the derivative of the quantity with respect to the curve dofs.
        """
        if self.fieldline1.need_to_run_code:
            res = self.fieldline1.res
            res = self.fieldline1.run_code(res['length'])
        if self.fieldline2.need_to_run_code:
            res = self.fieldline2.res
            res = self.fieldline2.run_code(res['length'])


        gamma1 = self.fieldline1.curve.gamma()
        gamma2 = self.fieldline2.curve.gamma()
        
        dJ_dgamma1 = self.thisgrad0(gamma1, gamma2)
        dJ_dgamma2 = self.thisgrad1(gamma1, gamma2)
        
        res = None
        for fieldline, dJ_dgamma in zip([self.fieldline1, self.fieldline2], [dJ_dgamma1, dJ_dgamma2]):
            if fieldline.need_to_run_code:
                res = self.fieldline.res
                res = self.fieldline.run_code(res['length'])

            curve = fieldline.curve
            res_curve = curve.dgamma_by_dcoeff_vjp(dJ_dgamma)
            res_curve = res_curve(curve)

            P, L, U = fieldline.res['PLU']
            dconstraint_dcoils_vjp = fieldline.res['vjp']

            # tack on dJ_dlength = 0 to the end of dJ_ds
            dJ_dc = np.zeros(L.shape[0])
            dj_dc = res_curve.copy()
            dJ_dc[:dj_dc.size] = dj_dc
            adj = forward_backward(P, L, U, dJ_dc)

            adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, fieldline.biotsavart, fieldline)
            if res is None:
                res = -1 * adj_times_dg_dcoil
            else:
                res +=-1 * adj_times_dg_dcoil
        return res

    return_fn_map = {'J': J, 'dJ': dJ}
