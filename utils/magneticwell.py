import simsoptpp as sopp
from simsopt.field import BiotSavart
from simsopt.geo import ToroidalFlux, Volume
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.objectives.utilities import forward_backward
import numpy as np
import jax.numpy as jnp
from jax import jit, grad

#def pure_well(tfs, volume):
#    tfs = jnp.abs(tfs)
#    volume = jnp.abs(volume)
#    tfs = tfs/tfs[-1]
#
#    x1 = tfs[0]
#    x2 = tfs[1]
#    y1 = volume[0]
#    y2 = volume[1]
#    A = jnp.array([[x1**2, x1],
#                  [x2**2, x2]])
#    Y = jnp.array([y1, y2])
#    
#    a, b = jnp.linalg.solve(A, Y)
#    return 2.*a
#
#class MagneticWell(Optimizable):
#    def __init__(self, boozer_surfaces):
#        """A magnetic well evaluator
#
#        Args:
#        """
#        super().__init__(depends_on=boozer_surfaces)
#        self.boozer_surfaces = boozer_surfaces
#        assert np.all([isinstance(boozer_surface.label, Volume) for boozer_surface in boozer_surfaces])
#        assert np.all([np.abs(boozer_surface.surface.volume())<=np.abs(boozer_surfaces[-1].surface.volume()) for boozer_surface in boozer_surfaces])
#
#        self.toroidal_fluxes = [ToroidalFlux(boozer_surface.surface, BiotSavart(boozer_surface.biotsavart.coils)) for boozer_surface in boozer_surfaces]
#        self.J_jax = jit(lambda tfs, volume: pure_well(tfs, volume))
#        self.thisgrad0 = jit(lambda tfs, volume: grad(self.J_jax, argnums=0)(tfs, volume))
#        
#        # dvol_dcoils is 0, so we don't really need this
#        self.thisgrad1 = jit(lambda tfs, volume: grad(self.J_jax, argnums=1)(tfs, volume))
#    
#    def recompute_bell(self, parent=None):
#        self._J = None
#        self._dJ = None
#
#    def compute(self):
#        for boozer_surface in self.boozer_surfaces:
#            if boozer_surface.need_to_run_code:
#                res = boozer_surface.res
#                res = boozer_surface.run_code(res['iota'], G=res['G'])
#        
#        tfs = np.array([tf.J() for tf in self.toroidal_fluxes])
#        vols = np.array([boozer_surface.surface.volume() for boozer_surface in self.boozer_surfaces])
#        self._J = self.J_jax(tfs, vols)
#
#        dJ_dtf = self.thisgrad0(tfs, vols)
#
#        # tf.dJ(coils) - vjp(tf.dJ(surface))
#        dJ_dcoils1 = sum([float(dj_dtf)*tf.dJ(partials=True)(tf.biotsavart, as_derivative=True) for (dj_dtf, tf) in zip(dJ_dtf, self.toroidal_fluxes)])
#        
#        self._dJ = dJ_dcoils1
#        for idx, (boozer_surface, tf) in enumerate(zip(self.boozer_surfaces, self.toroidal_fluxes)):
#            P, L, U = boozer_surface.res['PLU']
#            iota = boozer_surface.res['iota']
#            G = boozer_surface.res['G']
#
#            dJ_ds = np.concatenate([tf.dJ(partials=True)(tf.surface), [0., 0.]])
#            adj = forward_backward(P, L, U, dJ_ds)
#            dJ_dcoils2 = float(dJ_dtf[idx])*boozer_surface.res['vjp'](adj, boozer_surface, iota, G)
#            self._dJ = self._dJ-dJ_dcoils2
#
#    def J(self):
#        if self._J is None:
#            self.compute()
#        return self._J
#    
#    @derivative_dec
#    def dJ(self):
#        if self._dJ is None:
#            self.compute()
#        return self._dJ
#
#    return_fn_map = {'J': J, 'dJ': dJ}

#def pure_well_polynomial(tf, vol, G, B0, B1):
#        # axis
#        V0 = 0.
#        Psi0 = 0
#        modB0 = jnp.linalg.norm(B0, axis=-1)
#        
#        # surface
#        V1 = jnp.abs(vol)
#        Psi1 = jnp.abs(tf)
#        modB1 = jnp.linalg.norm(B1, axis=-1)
#        
#        # G from boozer surfaces is multiplied by 2pi
#        V0p = jnp.abs(G) * jnp.mean(1/modB0**2)
#        V1p = jnp.abs(G) * jnp.mean(1/modB1**2)
#        
#        a = -((-Psi1*V0p + 2*V1 - Psi1*V1p)/Psi1**3)
#        b = -((2*Psi1*V0p - 3*V1 + Psi1*V1p)/Psi1**2)
#        c = V0p
#        d = 0
#        return a, b, c, d
#
#def mean_well_pure(tf, vol, G, B0, B1):
#    a, b, c, d = pure_well_polynomial(tf, vol, G, B0, B1)
#    Psi1 = tf
#    return 2*b+3*a*Psi1
#    
#
#class MagneticWell(Optimizable):
#    def __init__(self, axis, boozer_surface):
#        """A magnetic well evaluator
#
#        Args:
#        """
#        super().__init__(depends_on=[axis, boozer_surface])
#        self.axis = axis
#        self.boozer_surface = boozer_surface
#        
#        self.biotsavart0 = BiotSavart(boozer_surface.biotsavart.coils)
#        self.biotsavart1 = BiotSavart(boozer_surface.biotsavart.coils)
#        assert isinstance(boozer_surface.label, Volume)
#
#        self.toroidal_flux = ToroidalFlux(boozer_surface.surface, BiotSavart(boozer_surface.biotsavart.coils))
#        
#        self.J_jax = jit(lambda tf, volume, G, B0, B1: mean_well_pure(tf, volume, G, B0, B1))
#        self.thisgrad0 = jit(lambda tf, volume, G, B0, B1: grad(self.J_jax, argnums=0)(tf, volume, G, B0, B1))
#        # dvol_dcoils is 0, so we don't really need this
#        self.thisgrad1 = jit(lambda tf, volume, G, B0, B1: grad(self.J_jax, argnums=1)(tf, volume, G, B0, B1))
#        self.thisgrad2 = jit(lambda tf, volume, G, B0, B1: grad(self.J_jax, argnums=2)(tf, volume, G, B0, B1))
#        self.thisgrad3 = jit(lambda tf, volume, G, B0, B1: grad(self.J_jax, argnums=3)(tf, volume, G, B0, B1))
#    
#    def recompute_bell(self, parent=None):
#        self._J = None
#        self._dJ = None
#
#    def compute(self):
#        axis = self.axis
#        boozer_surface = self.boozer_surface 
#        if boozer_surface.need_to_run_code:
#             res = boozer_surface.res
#             res = boozer_surface.run_code(res['iota'], G=res['G'])
#        if axis.need_to_run_code:
#            res = axis.res
#            res = axis.run_code(res['length'])
#
#        tf = self.toroidal_flux.J()
#        vol = boozer_surface.surface.volume()
#        G = boozer_surface.res['G']
#        
#        axis_pts = self.axis.curve.gamma().reshape((-1, 3))
#        surface_pts = self.boozer_surface.surface.gamma().reshape((-1, 3))
#        self.biotsavart0.set_points(axis_pts)
#        self.biotsavart1.set_points(surface_pts)
#        B0 = self.biotsavart0.B()
#        B1 = self.biotsavart1.B()
#        
#        self.polynomial = np.array(pure_well_polynomial(tf, vol, G, B0, B1))
#        self._J = self.J_jax(tf, vol, G, B0, B1)
#        
#        import ipdb;ipdb.set_trace()
#        dJ_dtf = self.thisgrad0(tf, vol, G, B0, B1)
#        dJ_dvol = self.thisgrad1(tf, vol, G, B0, B1)
#        dJ_dG = self.thisgrad2(tf, vol, G, B0, B1)
#        dJ_dB0 = self.thisgrad3(tf, vol, G, B0, B1)
#        dJ_dB1 = self.thisgrad4(tf, vol, G, B0, B1)
#
#        # tf.dJ(coils) - vjp(tf.dJ(surface))
#        dJ_dcoils1 = float(dJ_dtf)*tf.dJ(partials=True)(self.toroidal_flux.biotsavart, as_derivative=True)
#
#        self._dJ = dJ_dcoils1
#        P, L, U = boozer_surface.res['PLU']
#        iota = boozer_surface.res['iota']
#        G = boozer_surface.res['G']
#
#        dJ_ds = np.concatenate([tf.dJ(partials=True)(tf.surface), [0., dJ_dG]])
#        adj = forward_backward(P, L, U, dJ_ds)
#        dJ_dcoils2 = float(dJ_dtf)*boozer_surface.res['vjp'](adj, boozer_surface, iota, G)
#        self._dJ = self._dJ-dJ_dcoils2
#
#    def J(self):
#        if self._J is None:
#            self.compute()
#        return self._J
#    
#    @derivative_dec
#    def dJ(self):
#        if self._dJ is None:
#            self.compute()
#        return self._dJ
#
#    return_fn_map = {'J': J, 'dJ': dJ}


# THIS ONE ONLY NEEDS ONE MAGNETIC SURFACE
#def pure_well_polynomial(tf, vol, G, B1):
#        # axis
#        V0 = 0.
#        Psi0 = 0
#        
#        # surface
#        V1 = jnp.abs(vol)
#        Psi1 = jnp.abs(tf)
#        modB1 = jnp.linalg.norm(B1, axis=-1)
#        
#        # G from boozer surfaces is multiplied by 2pi
#        V1p = jnp.abs(G) * jnp.mean(1/modB1**2)
#        
#        a = -((V1 - Psi1*V1p)/Psi1**2)
#        b = -((-2*V1 + Psi1*V1p)/Psi1)
#        c = 0
#        return a, b, c
#
#def mean_well_pure(tf, vol, G, B1):
#    a, b, c = pure_well_polynomial(tf, vol, G, B1)
#    return 2*a
#    
#
#class MagneticWell(Optimizable):
#    def __init__(self, boozer_surface):
#        """A magnetic well evaluator
#
#        Args:
#        """
#        super().__init__(depends_on=[boozer_surface])
#        self.boozer_surface = boozer_surface
#        assert isinstance(boozer_surface.label, Volume)
#
#        self.toroidal_flux = ToroidalFlux(boozer_surface.surface, BiotSavart(boozer_surface.biotsavart.coils))
#        self.J_jax = jit(lambda tf, volume, G, B1: mean_well_pure(tf, volume, G, B1))
#        self.thisgrad0 = jit(lambda tf, volume, G, B1: grad(self.J_jax, argnums=0)(tf, volume, G, B1))
#        # dvol_dcoils is 0, so we don't really need this
#        self.thisgrad1 = jit(lambda tf, volume, G, B1: grad(self.J_jax, argnums=1)(tf, volume, G, B1))
#        self.thisgrad2 = jit(lambda tf, volume, G, B1: grad(self.J_jax, argnums=2)(tf, volume, G, B1))
#        self.thisgrad3 = jit(lambda tf, volume, G, B1: grad(self.J_jax, argnums=3)(tf, volume, G, B1))
#    
#    def recompute_bell(self, parent=None):
#        self._J = None
#        self._dJ = None
#
#    def compute(self):
#        boozer_surface = self.boozer_surface 
#        if boozer_surface.need_to_run_code:
#            res = boozer_surface.res
#            boozer_surface.run_code(res['iota'], G=res['G'])
#            self.toroidal_flux.recompute_bell()
#
#        P, L, U = boozer_surface.res['PLU']
#        iota = boozer_surface.res['iota']
#        G = boozer_surface.res['G']
#        
#        tf = self.toroidal_flux.J()
#        vol = boozer_surface.surface.volume()
#        B1 = boozer_surface.biotsavart.B()
#        
#        self._polynomial = np.array(pure_well_polynomial(tf, vol, G, B1))
#        self._J = self.J_jax(tf, vol, G, B1)
#        
#        dJ_dtf = self.thisgrad0(tf, vol, G, B1)
#        dJ_dvol = self.thisgrad1(tf, vol, G, B1)
#        dJ_dG = self.thisgrad2(tf, vol, G, B1)
#        dJ_dB1 = self.thisgrad3(tf, vol, G, B1)
#                
#        # Toroidal flux contribution
#        # tf.dJ(coils) - vjp(tf.dJ(surface))
#        dJ_dcoils1 = float(dJ_dtf)*self.toroidal_flux.dJ(partials=True)(self.toroidal_flux.biotsavart, as_derivative=True)
#        dJ_ds = np.concatenate([self.toroidal_flux.dJ(partials=True)(self.toroidal_flux.surface), [0., 0.]])
#        adj = forward_backward(P, L, U, dJ_ds)
#        dJ_dcoils1 -= float(dJ_dtf)*boozer_surface.res['vjp'](adj, boozer_surface, iota, G)
#        
#        # B contribution
#        dJ_dcoils2 = boozer_surface.biotsavart.B_vjp(dJ_dB1)
#        ndofs = boozer_surface.surface.full_x.size
#        dx_dc = boozer_surface.surface.dgamma_by_dcoeff().reshape((-1, 3, ndofs))
#        dB_by_dX = boozer_surface.biotsavart.dB_by_dX()
#        dB1_ds = np.einsum('ikl,ikm->ilm', dB_by_dX, dx_dc)
#        rhs_ds = np.einsum('ik,ikm->m', dJ_dB1, dB1_ds)
#        dJ_ds = np.concatenate([rhs_ds, [0., 0.]])
#        adj = forward_backward(P, L, U, dJ_ds)
#        dJ_dcoils2 -= boozer_surface.res['vjp'](adj, boozer_surface, iota, G)
#        
#        dJ_ds = np.zeros((ndofs+2,))
#        dJ_ds[-1] = dJ_dG
#        adj = forward_backward(P, L, U, dJ_ds)
#        dJ_dcoils3 = (-1.)*boozer_surface.res['vjp'](adj, boozer_surface, iota, G)
#        self._dJ = dJ_dcoils1 + dJ_dcoils2 + dJ_dcoils3
#
#    def J(self):
#        if self._J is None:
#            self.compute()
#        return self._J
#    
#    @derivative_dec
#    def dJ(self):
#        if self._dJ is None:
#            self.compute()
#        return self._dJ
#
#    def polynomial(self):
#        if self._polynomial is None:
#            self.compute()
#        return self._polynomial
#
#    return_fn_map = {'J': J, 'dJ': dJ}


def pure_well_polynomial(tf, vol, G, B0, B1):
        # axis
        V0 = 0.
        Psi0 = 0
        modB0 = jnp.linalg.norm(B0, axis=-1)

        # surface
        V1 = jnp.abs(vol)
        Psi1 = jnp.abs(tf)
        modB1 = jnp.linalg.norm(B1, axis=-1)
        
        # G from boozer surfaces is multiplied by 2pi, Psi is also multiplied by 2pi, hence we don't need to have a (2pi)^2 in front
        #V0p = jnp.abs(G) * jnp.mean(1/modB0**2)
        # because the axis is not perfectly QS
        V0p = jnp.abs(G) * jnp.mean(1/modB0) / jnp.mean(modB0)
        V1p = jnp.abs(G) * jnp.mean(1/modB1**2)
        
        a = -((-Psi1*V0p + 2*V1 - Psi1*V1p)/Psi1**3)
        b = -((2*Psi1*V0p - 3*V1 + Psi1*V1p)/Psi1**2)
        c = V0p 
        d = 0.
        return a, b, c, d

def mean_well_pure(tf, vol, G, B0, B1):
    a, b, c, d = pure_well_polynomial(tf, vol, G, B0, B1)
    return 2*b + 3*a*tf
    

class MagneticWell(Optimizable):
    def __init__(self, axis, boozer_surface):
        """A magnetic well evaluator

        Args:
        """
        super().__init__(depends_on=[axis, boozer_surface])
        self.axis = axis
        self.boozer_surface = boozer_surface
        assert isinstance(boozer_surface.label, Volume)

        self.toroidal_flux = ToroidalFlux(boozer_surface.surface, BiotSavart(boozer_surface.biotsavart.coils))
        self.J_jax = jit(lambda tf, volume, G, B0, B1: mean_well_pure(tf, volume, G, B0, B1))
        self.thisgrad0 = jit(lambda tf, volume, G, B0, B1: grad(self.J_jax, argnums=0)(tf, volume, G, B0, B1))
        # dvol_dcoils is 0, so we don't really need this
        self.thisgrad1 = jit(lambda tf, volume, G, B0, B1: grad(self.J_jax, argnums=1)(tf, volume, G, B0, B1))
        self.thisgrad2 = jit(lambda tf, volume, G, B0, B1: grad(self.J_jax, argnums=2)(tf, volume, G, B0, B1))
        self.thisgrad3 = jit(lambda tf, volume, G, B0, B1: grad(self.J_jax, argnums=3)(tf, volume, G, B0, B1))
        self.thisgrad4 = jit(lambda tf, volume, G, B0, B1: grad(self.J_jax, argnums=4)(tf, volume, G, B0, B1))
    
    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None
        self._polynomial = None
    
    def compute(self):
        axis = self.axis
        boozer_surface = self.boozer_surface 
        if boozer_surface.need_to_run_code:
            res = boozer_surface.res
            boozer_surface.run_code(res['iota'], G=res['G'])
            self.toroidal_flux.recompute_bell()
        
        if axis.need_to_run_code:
            res = axis.res
            axis.run_code(res['length'])

        P, L, U = boozer_surface.res['PLU']
        iota = boozer_surface.res['iota']
        G = boozer_surface.res['G']
        
        B0 = axis.biotsavart.B()

        tf = self.toroidal_flux.J()
        vol = boozer_surface.surface.volume()
        B1 = boozer_surface.biotsavart.B()
        
        self._polynomial = np.array(pure_well_polynomial(tf, vol, G, B0, B1))
        self._J = self.J_jax(tf, vol, G, B0, B1)
        
        dJ_dtf = self.thisgrad0(tf, vol, G, B0, B1)
        dJ_dvol = self.thisgrad1(tf, vol, G, B0, B1)
        dJ_dG = self.thisgrad2(tf, vol, G, B0, B1)
        dJ_dB0 = self.thisgrad3(tf, vol, G, B0, B1)
        dJ_dB1 = self.thisgrad4(tf, vol, G, B0, B1)
        

        # B0 contribution
        Pc, Lc, Uc = axis.res['PLU']
        dJ_dcoils0 = axis.biotsavart.B_vjp(dJ_dB0)
        ndofs = axis.curve.full_x.size
        dx_dc = axis.curve.dgamma_by_dcoeff().reshape((-1, 3, ndofs))
        dB_by_dX = axis.biotsavart.dB_by_dX()
        dB0_ds = np.einsum('ikl,ikm->ilm', dB_by_dX, dx_dc)
        rhs_ds = np.einsum('ik,ikm->m', dJ_dB0, dB0_ds)
        dJ_ds = np.concatenate([rhs_ds, [0.]])
        adj = forward_backward(Pc, Lc, Uc, dJ_ds)
        dJ_dcoils0 -= axis.res['vjp'](adj, axis.biotsavart, axis)

        # Toroidal flux contribution
        # tf.dJ(coils) - vjp(tf.dJ(surface))
        dJ_dcoils1 = float(dJ_dtf)*self.toroidal_flux.dJ(partials=True)(self.toroidal_flux.biotsavart, as_derivative=True)
        dJ_ds = np.concatenate([self.toroidal_flux.dJ(partials=True)(self.toroidal_flux.surface), [0., 0.]])
        adj = forward_backward(P, L, U, dJ_ds)
        dJ_dcoils1 -= float(dJ_dtf)*boozer_surface.res['vjp'](adj, boozer_surface, iota, G)
        
        # B1 contribution
        dJ_dcoils2 = boozer_surface.biotsavart.B_vjp(dJ_dB1)
        ndofs = boozer_surface.surface.full_x.size
        dx_dc = boozer_surface.surface.dgamma_by_dcoeff().reshape((-1, 3, ndofs))
        dB_by_dX = boozer_surface.biotsavart.dB_by_dX()
        dB1_ds = np.einsum('ikl,ikm->ilm', dB_by_dX, dx_dc)
        rhs_ds = np.einsum('ik,ikm->m', dJ_dB1, dB1_ds)
        dJ_ds = np.concatenate([rhs_ds, [0., 0.]])
        adj = forward_backward(P, L, U, dJ_ds)
        dJ_dcoils2 -= boozer_surface.res['vjp'](adj, boozer_surface, iota, G)
        
        dJ_ds = np.zeros((ndofs+2,))
        dJ_ds[-1] = dJ_dG
        adj = forward_backward(P, L, U, dJ_ds)
        dJ_dcoils3 = (-1.)*boozer_surface.res['vjp'](adj, boozer_surface, iota, G)

        self._dJ = dJ_dcoils1 + dJ_dcoils2 + dJ_dcoils3 + dJ_dcoils0
    
    def polynomial(self):
        if self._polynomial is None:
            self.compute()
        return self._polynomial

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
