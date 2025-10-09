import simsoptpp as sopp
from simsopt.field import BiotSavart
from simsopt.geo import ToroidalFlux, Volume
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.objectives.utilities import forward_backward
import numpy as np
import jax.numpy as jnp
from jax import jit, grad

def pure_well(tfs, volume):
    tfs = jnp.abs(tfs)
    volume = jnp.abs(volume)
    tfs = tfs/tfs[-1]

    x1 = tfs[0]
    x2 = tfs[1]
    y1 = volume[0]
    y2 = volume[1]
    A = jnp.array([[x1**2, x1],
                  [x2**2, x2]])
    Y = jnp.array([y1, y2])
    
    a, b = jnp.linalg.solve(A, Y)
    return 2.*a

class MagneticWell(Optimizable):
    def __init__(self, boozer_surfaces):
        """A magnetic well evaluator

        Args:
        """
        super().__init__(depends_on=boozer_surfaces)
        self.boozer_surfaces = boozer_surfaces
        assert np.all([isinstance(boozer_surface.label, Volume) for boozer_surface in boozer_surfaces])
        assert np.all([np.abs(boozer_surface.surface.volume())<=np.abs(boozer_surfaces[-1].surface.volume()) for boozer_surface in boozer_surfaces])

        self.toroidal_fluxes = [ToroidalFlux(boozer_surface.surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]
        self.J_jax = jit(lambda tfs, volume: pure_well(tfs, volume))
        self.thisgrad0 = jit(lambda tfs, volume: grad(self.J_jax, argnums=0)(tfs, volume))
        
        # dvol_dcoils is 0, so we don't really need this
        self.thisgrad1 = jit(lambda tfs, volume: grad(self.J_jax, argnums=1)(tfs, volume))
    
    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def compute(self):
        for boozer_surface in self.boozer_surfaces:
            if boozer_surface.need_to_run_code:
                res = boozer_surface.res
                res = boozer_surface.run_code(res['iota'], G=res['G'])
        
        tfs = np.array([tf.J() for tf in self.toroidal_fluxes])
        vols = np.array([boozer_surface.surface.volume() for boozer_surface in self.boozer_surfaces])
        self._J = self.J_jax(tfs, vols)

        dJ_dtf = self.thisgrad0(tfs, vols)

        # tf.dJ(coils) - vjp(tf.dJ(surface))
        dJ_dcoils1 = sum([float(dj_dtf)*tf.dJ(partials=True)(tf.biotsavart, as_derivative=True) for (dj_dtf, tf) in zip(dJ_dtf, self.toroidal_fluxes)])
        
        self._dJ = dJ_dcoils1
        for idx, (boozer_surface, tf) in enumerate(zip(self.boozer_surfaces, self.toroidal_fluxes)):
            P, L, U = boozer_surface.res['PLU']
            iota = boozer_surface.res['iota']
            G = boozer_surface.res['G']

            dJ_ds = np.concatenate([tf.dJ(partials=True)(tf.surface), [0., 0.]])
            adj = forward_backward(P, L, U, dJ_ds)
            dJ_dcoils2 = float(dJ_dtf[idx])*boozer_surface.res['vjp'](adj, boozer_surface, iota, G)
            self._dJ = self._dJ-dJ_dcoils2

    def J(self):
        if self._J is None:
            self.compute()
        return self._J
    
    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    return_fn_map = {'J': J, 'dJ': dJ}
