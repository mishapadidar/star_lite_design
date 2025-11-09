import simsoptpp as sopp
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.objectives import forward_backward
import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from .periodicfieldline import PeriodicFieldLine
from simsopt.geo import SurfaceRZFourier, Curve, BoozerSurface
from pyevtk.hl import gridToVTK  # pip install pyevtk

# > 0 if outside
# = 0 if on
# < 0 if inside
def toroidal_signed_level_set(pts, r, R, sign):
    col1 = jnp.sqrt(pts[:, 0]**2 + pts[:, 1]**2)-R
    col2 = pts[:, 2]
    val = jnp.concatenate([col1[:, None], col2[:, None]], axis=1)
    return sign*(jnp.linalg.norm(val, axis=-1)-r)

def quadratic_threshold(pts, r, R, sign, threshold):
    sls = toroidal_signed_level_set(pts, r, R, sign)
    return jnp.mean(jnp.maximum(threshold-sls, 0)**2)

class ToroidalVesselDistance(Optimizable):
    def __init__(self, r, R, entities, sign, minimum_distance):
        """A entity-vessel distance function

        Args:
            entities (list): list of entities
            vessel (array): (n, 3) array representing the vessel surface
            minimum_distance (float): minimum distance threshold for the distance calculation.
        """
        self.entities = entities
        self.sign = sign
        self.minimum_distance = minimum_distance
        
        self.J_jax = jit(lambda pts, r, R, sign: quadratic_threshold(pts, r, R, sign, minimum_distance))
        self.thisgrad0 = jit(lambda pts, r, R, sign: grad(self.J_jax, argnums=0)(pts, r, R, sign))
        self.thisgrad1 = jit(lambda pts, r, R, sign: grad(self.J_jax, argnums=1)(pts, r, R, sign))
        self.thisgrad2 = jit(lambda pts, r, R, sign: grad(self.J_jax, argnums=2)(pts, r, R, sign))
        self.candidates = None
        super().__init__(depends_on=entities, x0=np.array([r, R]), names=['r', 'R'])

    def shortest_distance(self):
        min_dist_curve = np.inf
        min_dist_fieldline = np.inf
        min_dist_bs = np.inf
        for sign, entity in zip(self.sign, self.entities):
            if isinstance(entity, PeriodicFieldLine):
                curve = entity.curve
                sd_tmp = toroidal_signed_level_set(curve.gamma(), self.local_full_x[0], self.local_full_x[1], sign)
                min_dist_tmp = sd_tmp.min()
                if min_dist_tmp < min_dist_fieldline:
                    min_dist_fieldline = min_dist_tmp
            elif isinstance(entity, Curve):
                sd_tmp = toroidal_signed_level_set(entity.gamma(), self.local_full_x[0], self.local_full_x[1], sign)
                min_dist_tmp = sd_tmp.min()
                if min_dist_tmp < min_dist_curve:
                    min_dist_curve = min_dist_tmp
            elif isinstance(entity, BoozerSurface):
                sd_tmp = toroidal_signed_level_set(entity.surface.gamma().reshape((-1, 3)), self.local_full_x[0], self.local_full_x[1], sign)
                min_dist_tmp = sd_tmp.min()
                if min_dist_tmp < min_dist_bs:
                    min_dist_bs = min_dist_tmp
            else:
                raise Exception('entity not supported')

        return min_dist_curve, min_dist_fieldline, min_dist_bs
    
    def to_vtk(self, name):
        surf = SurfaceRZFourier(mpol=1, ntor=1, nfp=1, quadpoints_phi=np.linspace(0, 1, 100), quadpoints_theta=np.linspace(0, 1, 100))
        
        r = self.local_full_x[0]
        R = self.local_full_x[1]

        surf.set_rc(0, 0, R)
        surf.set_rc(1, 0, r)
        surf.set_zs(1, 0, r)
        
        surf.to_vtk(name)

    def J(self):
        res = 0.
        r = self.local_full_x[0]
        R = self.local_full_x[1]
        for sign, entity in zip(self.sign, self.entities):
            if isinstance(entity, PeriodicFieldLine):
                # make sure the fieldline reflects its dofs
                if entity.need_to_run_code:
                    entity.run_code(entity.res['length'])
                curve = entity.curve
                gamma = curve.gamma()
            elif isinstance(entity, Curve):
                gamma = entity.gamma()
            elif isinstance(entity, BoozerSurface):
                if entity.need_to_run_code:
                    entity.run_code(entity.res['iota'], entity.res['G'])
                gamma = entity.surface.gamma().reshape((-1, 3))
            else:
                raise Exception('entity not supported')
            res += self.J_jax(gamma, r, R, sign)

        return res

    @derivative_dec
    def dJ(self):
        r = self.local_full_x[0]
        R = self.local_full_x[1]
        
        dres = Derivative({self: np.array([0., 0.])})

        dJ_dr = 0.
        dJ_dR = 0.
        for sign, entity in zip(self.sign, self.entities):
            if isinstance(entity, PeriodicFieldLine) or isinstance(entity, Curve):
                line = entity.curve if isinstance(entity, PeriodicFieldLine) else entity
                
                dJ_dgamma = self.thisgrad0(line.gamma(), r, R, sign)
                dres_curve = line.dgamma_by_dcoeff_vjp(dJ_dgamma)
                dJ_dr = float(self.thisgrad1(line.gamma(), r, R, sign))
                dJ_dR = float(self.thisgrad2(line.gamma(), r, R, sign))
                dres += dres_curve + Derivative({self:np.array([dJ_dr, dJ_dR])}) 
                
                if isinstance(entity, PeriodicFieldLine):
                    # make sure the fieldline reflects its dofs
                    if entity.need_to_run_code:
                        entity.run_code(entity.res['length'])

                    fieldline = entity
                    P, L, U = fieldline.res['PLU']
                    dconstraint_dcoils_vjp = fieldline.res['vjp']
                    
                    res_curve = dres_curve(entity.curve)

                    # tack on dJ_dlength = 0 to the end of dJ_ds
                    dJ_dc = np.zeros(L.shape[0])
                    dj_dc = res_curve.copy()
                    dJ_dc[:dj_dc.size] = dj_dc
                    adj = forward_backward(P, L, U, dJ_dc)

                    adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, fieldline.biotsavart, fieldline)
                    dres = dres  - adj_times_dg_dcoil
            elif isinstance(entity, BoozerSurface):
                if entity.need_to_run_code:
                    entity.run_code(entity.res['iota'], entity.res['G'])

                surface = entity.surface
                gamma = surface.gamma().reshape((-1, 3))
                
                dJ_dgamma = self.thisgrad0(gamma, r, R, sign).reshape(surface.gamma().shape)
                dres_surface = surface.dgamma_by_dcoeff_vjp(dJ_dgamma)
                dJ_dr = float(self.thisgrad1(gamma, r, R, sign))
                dJ_dR = float(self.thisgrad2(gamma, r, R, sign))
                
                dres += Derivative({self:np.array([dJ_dr, dJ_dR])}) 
                
                P, L, U = entity.res['PLU']
                dconstraint_dcoils_vjp = entity.res['vjp']
                

                # tack on dJ_dlength = 0 to the end of dJ_ds
                dJ_dc = np.zeros(L.shape[0])
                dj_dc = dres_surface.copy()
                dJ_dc[:dj_dc.size] = dj_dc
                adj = forward_backward(P, L, U, dJ_dc)

                adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, entity, entity.res['iota'], entity.res['G'])
                dres = dres  - adj_times_dg_dcoil

            else:
                raise Exception('entity not supported')




        return dres

    return_fn_map = {'J': J, 'dJ': dJ}

