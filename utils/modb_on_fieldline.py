import simsoptpp as sopp
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.objectives import forward_backward
import numpy as np
import jax.numpy as jnp
from jax import jit, grad

class ModBOnFieldLine(Optimizable):
    def __init__(self, fieldline, biotsavart):
        """An optimizable class to compute the mean magnetic field strength along a field line,
            J = \int |B| dl / L.
        It is assumed that the periodic fieldline is unit speed.

        Args:
            fieldline (PeriodicFieldLine): instance of a periodic field line on which the magnetic field is computed.
            biotsavart (BiotSavart): instance of a Biot-Savart object that computes the magnetic field.
        """
        self.fieldline = fieldline
        self.biotsavart = biotsavart
        self.recompute_bell()
        super().__init__(depends_on=[fieldline])  # Bharat's comment: Shouldn't we add surface here

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        """
        Compute the mean magnetic field strength along the field line.

        Returns:
            float: The mean magnetic field strength along the field line.
        """
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ
    
    def compute(self):

        # make sure the fieldline reflects its dofs
        if self.fieldline.need_to_run_code:
            res = self.fieldline.res
            res = self.fieldline.run_code(res['length'])

        # compute the objective
        self.biotsavart.set_points(self.fieldline.curve.gamma())
        modB = self.biotsavart.AbsB()
        self._J = np.mean(modB)
        
        B = self.biotsavart.B()
        dmodB_dB = B/modB/B.shape[0]
        res_coils = self.biotsavart.B_vjp(dmodB_dB)
        
        dB_by_dX = self.biotsavart.dB_by_dX()
        dB_dc = np.einsum('ikl,ikm->ilm', dB_by_dX, self.fieldline.curve.dgamma_by_dcoeff(), optimize=True)
        res_curve = np.einsum('ik,ikl->l', dmodB_dB, dB_dc, optimize=True)

        fieldline = self.fieldline
        P, L, U = fieldline.res['PLU']
        dconstraint_dcoils_vjp = fieldline.res['vjp']
        
        # tack on dJ_dlength = 0 to the end of dJ_ds
        dJ_dc = np.zeros(L.shape[0])
        dj_dc = res_curve.copy()
        dJ_dc[:dj_dc.size] = dj_dc
        adj = forward_backward(P, L, U, dJ_dc)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, fieldline.biotsavart, fieldline)
        res = res_coils  - adj_times_dg_dcoil
        self._dJ = res


class ModBRippleOnFieldLine(Optimizable):
    def __init__(self, fieldline, biotsavart, threshold=0.05):
        r"""Differentiable penalty keeping \|B\| within +-threshold of its MEAN along a
        field line (used on the magnetic axis). It penalizes every point that strays
        outside the band [ (1-threshold), (1+threshold) ] * mean\|B\|:

            r_i = \|B\|_i / mean\|B\|
            J   = mean_i [ max(r_i - (1+threshold), 0)^2 + max((1-threshold) - r_i, 0)^2 ]

        i.e. it penalizes \|B\|_i > (1+threshold)*mean and \|B\|_i < (1-threshold)*mean.
        J is ~0 when \|B\| stays within +-threshold (e.g. 5%) of its mean, smooth, and
        dimensionless. The gradient flows through \|B\| at the field-line points
        (including the dependence of the mean on every point) and the field-line solve,
        mirroring ModBOnFieldLine.

        Args:
            fieldline (PeriodicFieldLine): the field line (e.g. the magnetic axis).
            biotsavart (BiotSavart): the field; uses its own set_points cache.
            threshold (float): allowed fractional band half-width (default 0.05 = 5%).
        """
        self.fieldline = fieldline
        self.biotsavart = biotsavart
        self.threshold = float(threshold)

        def _pure(modB):
            r = modB / jnp.mean(modB)
            over = jnp.maximum(r - (1.0 + self.threshold), 0.0)
            under = jnp.maximum((1.0 - self.threshold) - r, 0.0)
            return jnp.mean(over ** 2 + under ** 2)
        self._J_jax = jit(_pure)
        self._grad_jax = jit(grad(_pure))

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

    def max_deviation(self):
        """Current max fractional |B| deviation from the mean: max_i ||B|_i/mean - 1|.
        Compare against `threshold` for a constraint/escalation check."""
        if self.fieldline.need_to_run_code:
            self.fieldline.run_code(self.fieldline.res['length'])
        self.biotsavart.set_points(self.fieldline.curve.gamma())
        modB = self.biotsavart.AbsB()
        return float(np.max(np.abs(modB / np.mean(modB) - 1.0)))

    def compute(self):
        # make sure the fieldline reflects its dofs
        if self.fieldline.need_to_run_code:
            res = self.fieldline.res
            res = self.fieldline.run_code(res['length'])

        self.biotsavart.set_points(self.fieldline.curve.gamma())
        modB = self.biotsavart.AbsB()
        B = self.biotsavart.B()

        self._J = float(self._J_jax(jnp.asarray(modB)))
        # dJ/d|B|_i (same shape as modB), then chain to B: dJ/dB_i = (dJ/d|B|_i) B_i/|B|_i
        dJ_dmodB = np.asarray(self._grad_jax(jnp.asarray(modB)))
        dJ_dB = dJ_dmodB * B / modB

        res_coils = self.biotsavart.B_vjp(dJ_dB)

        dB_by_dX = self.biotsavart.dB_by_dX()
        dB_dc = np.einsum('ikl,ikm->ilm', dB_by_dX, self.fieldline.curve.dgamma_by_dcoeff(), optimize=True)
        res_curve = np.einsum('ik,ikl->l', dJ_dB, dB_dc, optimize=True)

        fieldline = self.fieldline
        P, L, U = fieldline.res['PLU']
        dconstraint_dcoils_vjp = fieldline.res['vjp']

        # tack on dJ_dlength = 0 to the end of dJ_ds
        dJ_dc = np.zeros(L.shape[0])
        dJ_dc[:res_curve.size] = res_curve
        adj = forward_backward(P, L, U, dJ_dc)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, fieldline.biotsavart, fieldline)
        self._dJ = res_coils - adj_times_dg_dcoil
