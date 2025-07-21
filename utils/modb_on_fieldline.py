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

