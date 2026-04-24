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
import jax
import jax.numpy as jnp

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

class PillPipeSDF(Optimizable):
    def __init__(self, bx, by, r, rr, **kwargs):
        super().__init__(depends_on=[], x0=np.array([bx, by, r, rr]), names=['bx', 'by', 'r', 'rr'], **kwargs)
        self.bx, self.by, self.r, self.rr = bx, by, r, rr

        self.pure = sdf_pill_pipe
        self.quadratic_threshold = quadratic_threshold
    def num_dofs(self):
        return 4

    def to_vtk(self, name):
        bx, by, r, rr = self.local_full_x
        pad = 0.1
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
        D = np.array(self.pure(pts, self.local_full_x, 1.0).astype(np.float32).reshape((nx, ny, nz)))
        
        # write to .vts file
        gridToVTK(
            name,
            x=xs, y=ys, z=zs,
            cellData=None,
            pointData={"sdf": D},
        )


class VesselDistance(Optimizable):
    def __init__(self, sdf, entities, sign, minimum_distance):
        """A entity-vessel distance function

        Args:
            entities (list): list of entities
            vessel (array): (n, 3) array representing the vessel surface
            minimum_distance (float): minimum distance threshold for the distance calculation.
        """
        self.entities = entities
        self.sign = sign
        self.minimum_distance = minimum_distance
        self.sdf = sdf

        self.J_jax = jit(lambda pts, params, sign: self.sdf.quadratic_threshold(pts, params, sign, minimum_distance))
        self.thisgrad0 = jit(lambda pts, params, sign: grad(self.J_jax, argnums=0)(pts, params, sign))
        self.thisgrad1 = jit(lambda pts, params, sign: grad(self.J_jax, argnums=1)(pts, params, sign))
        super().__init__(depends_on=entities + [sdf])

    def shortest_distance(self):
        min_dist_curve = np.inf
        min_dist_fieldline = np.inf
        min_dist_bs = np.inf
        for sign, entity in zip(self.sign, self.entities):
            if isinstance(entity, PeriodicFieldLine):
                curve = entity.curve
                sd_tmp = self.sdf.pure(curve.gamma(), self.sdf.local_full_x, sign)
                min_dist_tmp = sd_tmp.min()
                if min_dist_tmp < min_dist_fieldline:
                    min_dist_fieldline = min_dist_tmp
            elif isinstance(entity, Curve):
                sd_tmp = self.sdf.pure(entity.gamma(), self.sdf.local_full_x, sign)
                min_dist_tmp = sd_tmp.min()
                if min_dist_tmp < min_dist_curve:
                    min_dist_curve = min_dist_tmp
            elif isinstance(entity, BoozerSurface):
                sd_tmp = self.sdf.pure(entity.surface.gamma().reshape((-1, 3)), self.sdf.local_full_x, sign)
                min_dist_tmp = sd_tmp.min()
                if min_dist_tmp < min_dist_bs:
                    min_dist_bs = min_dist_tmp
            else:
                raise Exception('entity not supported')

        return min_dist_curve, min_dist_fieldline, min_dist_bs
    
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
            res += self.J_jax(gamma, self.sdf.local_full_x, sign)

        return res

    @derivative_dec
    def dJ(self):
        dres = Derivative({self.sdf: np.zeros_like(self.sdf.local_full_x)})

        for sign, entity in zip(self.sign, self.entities):
            if isinstance(entity, PeriodicFieldLine) or isinstance(entity, Curve):
                line = entity.curve if isinstance(entity, PeriodicFieldLine) else entity
                
                dJ_dgamma = self.thisgrad0(line.gamma(), self.sdf.local_full_x, sign)
                dres_curve = line.dgamma_by_dcoeff_vjp(dJ_dgamma)
                dJ_dparams = self.thisgrad1(line.gamma(), self.sdf.local_full_x, sign)
                dres += dres_curve + Derivative({self.sdf:dJ_dparams}) 

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
                
                dJ_dgamma = self.thisgrad0(gamma, self.sdf.local_full_x, sign).reshape(surface.gamma().shape)
                dres_surface = surface.dgamma_by_dcoeff_vjp(dJ_dgamma)
                dJ_dparams = self.thisgrad1(gamma, self.sdf.local_full_x, sign)
                
                dres += Derivative({self.sdf:dJ_dparams}) 
                
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


import jax
import jax.numpy as jnp


import jax
import jax.numpy as jnp


def pill_pipe_curve_pure(dofs, quadpoints, s, phi, sign=1.0):
    """
    Notes
    -----
    - JAX-friendly and jittable for scalar inputs.
    - Differentiable w.r.t. bx, by, r, rr almost everywhere.
    - Nonsmooth at rounded-rectangle segment junctions, as expected.
    """
    curve_radius = dofs[0]
    bx, by, r, rr = dofs[1:]

    phi *= jnp.pi*2
    w = bx - r
    h = by - r
    s0 = 4*(bx - r) + 3*(by - r) + 1.5*jnp.pi*r
    lengths = jnp.array([
        2.0 * w,
        0.5 * jnp.pi * r,
        2.0 * h,
        0.5 * jnp.pi * r,
        2.0 * w,
        0.5 * jnp.pi * r,
        2.0 * h,
        0.5 * jnp.pi * r,
    ], dtype=jnp.result_type(s, phi, bx, by, r, rr))

    cum = jnp.cumsum(lengths)
    L = cum[-1]
    s *= L
    s -= s0
    s = s%L

    starts = jnp.concatenate([jnp.array([0.0], dtype=s.dtype), cum[:-1]])
    idx = jnp.searchsorted(cum, s, side="right")
    u = s - starts[idx]

    def seg0(u):
        # top edge: (w, by) -> (-w, by)
        x = w - u
        y = by
        nx = jnp.zeros_like(u)
        ny = jnp.ones_like(u)
        tx = -jnp.ones_like(u)
        ty = jnp.zeros_like(u)
        dnx = jnp.zeros_like(u)
        dny = jnp.zeros_like(u)
        return x, y, nx, ny, tx, ty, dnx, dny

    def seg1(u):
        # top-left quarter circle
        th = 0.5 * jnp.pi + u / r
        x = -w + r * jnp.cos(th)
        y =  h + r * jnp.sin(th)
        nx = jnp.cos(th)
        ny = jnp.sin(th)
        tx = -jnp.sin(th)
        ty =  jnp.cos(th)
        dnx = tx / r
        dny = ty / r
        return x, y, nx, ny, tx, ty, dnx, dny

    def seg2(u):
        # left edge: (-bx, h) -> (-bx, -h)
        x = -bx
        y = h - u
        nx = -jnp.ones_like(u)
        ny = jnp.zeros_like(u)
        tx = jnp.zeros_like(u)
        ty = -jnp.ones_like(u)
        dnx = jnp.zeros_like(u)
        dny = jnp.zeros_like(u)
        return x, y, nx, ny, tx, ty, dnx, dny

    def seg3(u):
        # bottom-left quarter circle
        th = jnp.pi + u / r
        x = -w + r * jnp.cos(th)
        y = -h + r * jnp.sin(th)
        nx = jnp.cos(th)
        ny = jnp.sin(th)
        tx = -jnp.sin(th)
        ty =  jnp.cos(th)
        dnx = tx / r
        dny = ty / r
        return x, y, nx, ny, tx, ty, dnx, dny

    def seg4(u):
        # bottom edge: (-w, -by) -> (w, -by)
        x = -w + u
        y = -by
        nx = jnp.zeros_like(u)
        ny = -jnp.ones_like(u)
        tx = jnp.ones_like(u)
        ty = jnp.zeros_like(u)
        dnx = jnp.zeros_like(u)
        dny = jnp.zeros_like(u)
        return x, y, nx, ny, tx, ty, dnx, dny

    def seg5(u):
        # bottom-right quarter circle
        th = 1.5 * jnp.pi + u / r
        x =  w + r * jnp.cos(th)
        y = -h + r * jnp.sin(th)
        nx = jnp.cos(th)
        ny = jnp.sin(th)
        tx = -jnp.sin(th)
        ty =  jnp.cos(th)
        dnx = tx / r
        dny = ty / r
        return x, y, nx, ny, tx, ty, dnx, dny

    def seg6(u):
        # right edge: (bx, -h) -> (bx, h)
        x = bx
        y = -h + u
        nx = jnp.ones_like(u)
        ny = jnp.zeros_like(u)
        tx = jnp.zeros_like(u)
        ty = jnp.ones_like(u)
        dnx = jnp.zeros_like(u)
        dny = jnp.zeros_like(u)
        return x, y, nx, ny, tx, ty, dnx, dny

    def seg7(u):
        # top-right quarter circle
        th = u / r
        x =  w + r * jnp.cos(th)
        y =  h + r * jnp.sin(th)
        nx = jnp.cos(th)
        ny = jnp.sin(th)
        tx = -jnp.sin(th)
        ty =  jnp.cos(th)
        dnx = tx / r
        dny = ty / r
        return x, y, nx, ny, tx, ty, dnx, dny

    x0, y0, nx2, ny2, tx2, ty2, dnx2, dny2 = jax.lax.switch(idx, (seg0, seg1, seg2, seg3, seg4, seg5, seg6, seg7), u)

    c = jnp.cos(phi)
    sp = jnp.sin(phi)

    # Surface point
    p = jnp.array([x0 + rr * c * nx2, y0 + rr * c * ny2, rr * sp])

    # Outward normal
    n = jnp.array([c * nx2, c * ny2, sp])
    n = sign * n
    n = n / jnp.linalg.norm(n)

    # Tangent in s-direction: dp/ds
    dp_ds = jnp.array([tx2 + rr * c * dnx2, ty2 + rr * c * dny2, 0.0])
    t1 = dp_ds / jnp.linalg.norm(dp_ds)

    # Tangent in phi-direction: dp/dphi
    dp_dphi = jnp.array([-sp * nx2, -sp * ny2, c])
    t2 = dp_dphi / jnp.linalg.norm(dp_dphi)
   
    gamma = p[None, :] + curve_radius * (jnp.cos(2*jnp.pi*quadpoints[:, None]) * t1[None, :] + jnp.sin(2*jnp.pi*quadpoints[:, None]) * t2[None, :])
    return gamma
# coil class, dofs are (bx, by, r, rr, coil radius)
# to do:
#       write my own vjp dGamma_d(vessel, radius)
#                        dGammadash_d(vessel, radius)
import numpy as np
from jax import vjp, jacfwd, jvp
import jax.numpy as jnp
class PillCurve(sopp.Curve, Curve):
    def __init__(self, vessel, s_coord, phi_coord, quadpoints, **kwargs):
        self.vessel = vessel
        self.s_coord = s_coord
        self.phi_coord = phi_coord

        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        sopp.Curve.__init__(self, quadpoints)
        
        points = np.asarray(self.quadpoints)
        ones = jnp.ones_like(points)
        self.gamma_pure = lambda dofs, points: pill_pipe_curve_pure(dofs, points, s_coord, phi_coord)
        self.gamma_jax = jit(lambda dofs: self.gamma_pure(dofs, points))
        self.dgamma_by_dcoeff_jax = jit(jacfwd(self.gamma_jax))
        self.dgamma_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gamma_jax, x)[1](v)[0])

        self.gammadash_pure = lambda x, q: jvp(lambda p: self.gamma_pure(x, p), (q,), (ones,))[1]
        self.gammadash_jax = jit(lambda x: self.gammadash_pure(x, points))
        self.dgammadash_by_dcoeff_jax = jit(jacfwd(self.gammadash_jax))
        self.dgammadash_by_dcoeff_vjp_jax = jit(lambda x, v: vjp(self.gammadash_jax, x)[1](v)[0])

        Curve.__init__(self, names=['radius'], depends_on=[vessel], x0=np.zeros((1,)), **kwargs)


    def get_dofs(self):
        return self.local_full_x

    def num_dofs(self):
        return self.vessel.num_dofs()+1

    def gamma_impl(self, gamma, quadpoints):
        dofs = np.concatenate([self.local_full_x, self.vessel.local_full_x])
        gamma[:] = self.gamma_pure(dofs, quadpoints)

    def gammadash_impl(self, gammadash):
        dofs = np.concatenate([self.local_full_x, self.vessel.local_full_x])
        gammadash[:] = self.gammadash_jax(dofs)

#    def gammadashdash_impl(self, gammadashdash):
#        r"""
#        This function returns :math:`\Gamma''(\varphi)`, where :math:`\Gamma` are the x, y, z
#        coordinates of the curve.
#
#        """
#
#        gammadashdash[:] = self.curve.gammadashdash() @ self.rotmat
#
#    def gammadashdashdash_impl(self, gammadashdashdash):
#        r"""
#        This function returns :math:`\Gamma'''(\varphi)`, where :math:`\Gamma` are the x, y, z
#        coordinates of the curve.
#
#        """
#
#        gammadashdashdash[:] = self.curve.gammadashdashdash() @ self.rotmat
#
    def dgamma_by_dcoeff_impl(self, dgamma_by_dcoeff):
        dofs = np.concatenate([self.local_full_x, self.vessel.local_full_x])
        dgamma_by_dcoeff[:] = self.dgamma_by_dcoeff_jax(dofs)

    def dgammadash_by_dcoeff_impl(self, dgammadash_by_dcoeff):
        dofs = np.concatenate([self.local_full_x, self.vessel.local_full_x])
        dgammadash_by_dcoeff[:] = self.dgammadash_by_dcoeff_jax(dofs)

    def dgamma_by_dcoeff_vjp(self, v):
        dofs = np.concatenate([self.local_full_x, self.vessel.local_full_x])
        res = self.dgamma_by_dcoeff_vjp_jax(dofs, v)
        return Derivative({self: np.array([res[0]])}) + Derivative({self.vessel: np.array(res[1:])})

    def dgammadash_by_dcoeff_vjp(self, v):
        dofs = np.concatenate([self.local_full_x, self.vessel.local_full_x])
        res = self.dgammadash_by_dcoeff_vjp_jax(dofs, v)
        return Derivative({self: np.array([res[0]])}) + Derivative({self.vessel: np.array(res[1:])})

