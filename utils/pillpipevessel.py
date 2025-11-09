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
# ---------- 2D rounded rectangle SDF ----------
def sdf_round_rect(x, y, bx, by, r):
    ax = jnp.abs(x) - (bx - r)
    ay = jnp.abs(y) - (by - r)
    qx = jnp.maximum(ax, 0.0)
    qy = jnp.maximum(ay, 0.0)
    outside = jnp.sqrt(qx*qx + qy*qy)
    inside = jnp.minimum(jnp.maximum(ax, ay), 0.0)
    return outside + inside - r

# ---------- 3D circular pipe SDF ----------
def sdf_pill_pipe(pts, params, sign):
    bx, by, r, rr = params
    d2 = sdf_round_rect(pts[:, 0], pts[:, 1], bx, by, r)
    return sign*(jnp.sqrt(d2**2 + pts[:, 2]**2) - rr)

def quadratic_threshold(pts, params, sign, threshold):
    sls = sdf_pill_pipe(pts, params, sign)
    
    
    bx, by, r, rr = params
    cons1 = jnp.maximum(r-bx, 0)**2
    cons2 = jnp.maximum(r-by, 0)**2
    return jnp.mean(jnp.maximum(threshold-sls, 0)**2) + cons1 + cons2

class PillPipeVesselDistance(Optimizable):
    def __init__(self, bx, by, r, rr, entities, sign, minimum_distance):
        """A entity-vessel distance function

        Args:
            entities (list): list of entities
            vessel (array): (n, 3) array representing the vessel surface
            minimum_distance (float): minimum distance threshold for the distance calculation.
        """
        self.entities = entities
        self.sign = sign
        self.minimum_distance = minimum_distance
        
        self.J_jax = jit(lambda pts, params, sign: quadratic_threshold(pts, params, sign, minimum_distance))
        self.thisgrad0 = jit(lambda pts, params, sign: grad(self.J_jax, argnums=0)(pts, params, sign))
        self.thisgrad1 = jit(lambda pts, params, sign: grad(self.J_jax, argnums=1)(pts, params, sign))
        super().__init__(depends_on=entities, x0=np.array([bx, by, r, rr]), names=['bx', 'by', 'r', 'rr'])

    def shortest_distance(self):
        min_dist_curve = np.inf
        min_dist_fieldline = np.inf
        min_dist_bs = np.inf
        for sign, entity in zip(self.sign, self.entities):
            if isinstance(entity, PeriodicFieldLine):
                curve = entity.curve
                sd_tmp = sdf_pill_pipe(curve.gamma(), self.local_full_x, sign)
                min_dist_tmp = sd_tmp.min()
                if min_dist_tmp < min_dist_fieldline:
                    min_dist_fieldline = min_dist_tmp
            elif isinstance(entity, Curve):
                sd_tmp = sdf_pill_pipe(entity.gamma(), self.local_full_x, sign)
                min_dist_tmp = sd_tmp.min()
                if min_dist_tmp < min_dist_curve:
                    min_dist_curve = min_dist_tmp
            elif isinstance(entity, BoozerSurface):
                sd_tmp = sdf_pill_pipe(entity.surface.gamma().reshape((-1, 3)), self.local_full_x, sign)
                min_dist_tmp = sd_tmp.min()
                if min_dist_tmp < min_dist_bs:
                    min_dist_bs = min_dist_tmp
            else:
                raise Exception('entity not supported')

        return min_dist_curve, min_dist_fieldline, min_dist_bs
    
    def to_vtk(self, name):
        bx, by, r, rr = self.local_full_x
        pad = self.minimum_distance
        # domain extents
        x_min, x_max = -(bx+rr + pad), (bx+rr + pad)
        y_min, y_max = -(by+rr + pad), (by+rr + pad)
        z_min, z_max = -(rr + pad),  (rr + pad)
        
        nx, ny, nz = 20, 20, 20
        xs = np.linspace(x_min, x_max, nx).astype(np.float32)
        ys = np.linspace(y_min, y_max, ny).astype(np.float32)
        zs = np.linspace(z_min, z_max, nz).astype(np.float32)

        # meshgrid in ij indexing (nx,ny,nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        
        pts = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None], Z.flatten()[:, None]), axis=-1)
        D = np.array(sdf_pill_pipe(pts, self.local_full_x, 1.0).astype(np.float32).reshape((nx, ny, nz)))
        
        # write to .vts file
        gridToVTK(
            name,
            x=xs, y=ys, z=zs,
            cellData=None,
            pointData={"sdf": D},
        )


    def J(self):
        res = 0.
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
            res += self.J_jax(gamma, self.local_full_x, sign)

        return res

    @derivative_dec
    def dJ(self):
        dres = Derivative({self: np.zeros_like(self.local_full_x)})

        for sign, entity in zip(self.sign, self.entities):
            if isinstance(entity, PeriodicFieldLine) or isinstance(entity, Curve):
                line = entity.curve if isinstance(entity, PeriodicFieldLine) else entity
                
                dJ_dgamma = self.thisgrad0(line.gamma(), self.local_full_x, sign)
                dres_curve = line.dgamma_by_dcoeff_vjp(dJ_dgamma)
                dJ_dparams = self.thisgrad1(line.gamma(), self.local_full_x, sign)
                dres += dres_curve + Derivative({self:dJ_dparams}) 
                
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
                
                dJ_dgamma = self.thisgrad0(gamma, self.local_full_x, sign).reshape(surface.gamma().shape)
                dres_surface = surface.dgamma_by_dcoeff_vjp(dJ_dgamma)
                dJ_dparams = self.thisgrad1(gamma, self.local_full_x, sign)
                
                dres += Derivative({self:dJ_dparams}) 
                
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
