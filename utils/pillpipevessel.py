import simsoptpp as sopp
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.geo import CurveXYZFourier
from simsopt.objectives import forward_backward
import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from .periodicfieldline import PeriodicFieldLine
from .singularperiodicfieldline import SingularPeriodicFieldline
from simsopt.geo import SurfaceRZFourier, Curve, BoozerSurface

# Field-line entity types accepted by VesselDistance: both follow the same
# interface (.curve, .biotsavart, run_code(res['length']), res['PLU'], res['vjp']).
_FIELDLINE_TYPES = (PeriodicFieldLine, SingularPeriodicFieldline)
from pyevtk.hl import gridToVTK  # pip install pyevtk
import jax
import jax.numpy as jnp

@jax.custom_jvp
def safe_norm2d(qx, qy):
    return jnp.sqrt(qx * qx + qy * qy)

@safe_norm2d.defjvp
def safe_norm2d_jvp(primals, tangents):
    qx, qy = primals
    dqx, dqy = tangents

    sq = qx * qx + qy * qy
    norm = jnp.sqrt(sq)

    # Choose a subgradient at the origin: zero.
    safe_denom = jnp.where(norm > 0.0, norm, 1.0)
    tangent_out = jnp.where(
        norm > 0.0,
        (qx * dqx + qy * dqy) / safe_denom,
        0.0,
    )

    return norm, tangent_out

# > 0 if outside
# = 0 if on
# < 0 if inside
# ---------- 2D rounded rectangle SDF ----------
def sdf_round_rect(x, y, bx, by, r):
    ax = jnp.abs(x) - (bx - r)
    ay = jnp.abs(y) - (by - r)
    qx = jnp.maximum(ax, 0.0)
    qy = jnp.maximum(ay, 0.0)
    outside = safe_norm2d(qx, qy)
    inside = jnp.minimum(jnp.maximum(ax, ay), 0.0)
    return outside + inside - r

# ---------- 3D circular pipe SDF ----------
def sdf_pill_pipe(pts, params, sign):
    bx, by, r, rr = params
    d2 = sdf_round_rect(pts[:, 0], pts[:, 1], bx, by, r)
    return sign*(jnp.sqrt(d2**2 + pts[:, 2]**2) - rr)

def quadratic_threshold_pill_pipe(pts, params, sign, threshold):
    sls = sdf_pill_pipe(pts, params, sign)
    bx, by, r, rr = params
    cons1 = jnp.maximum(r-bx, 0)**2
    cons2 = jnp.maximum(r-by, 0)**2
    # valid-tube regime: the pipe radius rr must be smaller than the rounded-corner
    # radius r (the corner curvature is 1/r, so rr*kappa = rr/r < 1), else the pipe
    # self-intersects at a corner and the SDF is no longer exact.
    cons3 = jnp.maximum(rr-r, 0)**2
    return jnp.mean(jnp.maximum(threshold-sls, 0)**2) + cons1 + cons2 + cons3

def quadratic_distance_pill_pipe(pts, params, sign, threshold):
    sls = sdf_pill_pipe(pts, params, sign)
    mean_value = jnp.mean(sls)
    dvalue = jnp.abs(sls-mean_value)
    return jnp.mean(jnp.maximum(dvalue-threshold, 0)**2)

class PillPipeSDF(Optimizable):
    def __init__(self, bx, by, r, rr, **kwargs):
        super().__init__(depends_on=[], x0=np.array([bx, by, r, rr]), names=['bx', 'by', 'r', 'rr'], **kwargs)
        self.bx, self.by, self.r, self.rr = bx, by, r, rr

        self.pure = sdf_pill_pipe
        self.quadratic_threshold = quadratic_threshold_pill_pipe
        self.quadratic_distance = quadratic_distance_pill_pipe
    
    def num_dofs(self):
        return 4
    
    def eval(self, x, y, z):
        pts = np.concatenate((x.flatten()[:, None], y.flatten()[:, None], z.flatten()[:, None]), axis=-1)
        return np.array(self.pure(pts, self.local_full_x, 1.0).astype(np.float32).reshape(x.shape))

    def rz_bounds(self):
        """Conservative (R_max, |Z|_max) extent of the vessel surface in the
        (R, Z) half-plane, from the vessel parameters. Used to size the
        cross-section sampling window / plot limits so the full vessel is never
        clipped. The surface lies within |x|<=bx+rr, |y|<=by+rr, |z|<=rr, so
        R = hypot(x, y) <= hypot(bx+rr, by+rr) and |Z| <= rr."""
        bx, by, r, rr = self.local_full_x
        return float(np.hypot(bx + rr, by + rr)), float(rr)

    def to_vtk(self, name, nx=20, ny=20, nz=20):
        bx, by, r, rr = self.local_full_x
        pad = 0.1
        # domain extents
        x_min, x_max = -(bx+rr + pad), (bx+rr + pad)
        y_min, y_max = -(by+rr + pad), (by+rr + pad)
        z_min, z_max = -(rr + pad),  (rr + pad)
        
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


def sdf_torus(pts, params, sign):
    r, R = params
    col1 = jnp.sqrt(pts[:, 0]**2 + pts[:, 1]**2)-R
    col2 = pts[:, 2]
    val = jnp.concatenate([col1[:, None], col2[:, None]], axis=1)
    return sign*(jnp.linalg.norm(val, axis=-1)-r)

def quadratic_threshold_torus(pts, params, sign, threshold):
    sls = sdf_torus(pts, params, sign)
    r, R = params
    # valid-torus regime: the minor radius must be smaller than the major radius
    # (the circle curvature is 1/R, so r*kappa = r/R < 1), else the torus
    # self-intersects through the hole and the SDF is no longer exact.
    cons = jnp.maximum(r-R, 0)**2
    return jnp.mean(jnp.maximum(threshold-sls, 0)**2) + cons

def quadratic_distance_torus(pts, params, sign, threshold):
    sls = sdf_torus(pts, params, sign)
    mean_value = jnp.mean(sls)
    dvalue = jnp.abs(sls-mean_value)
    return jnp.mean(jnp.maximum(dvalue-threshold, 0)**2)

class TorusSDF(Optimizable):
    def __init__(self, r, R, **kwargs):
        super().__init__(depends_on=[], x0=np.array([r, R]), names=['r', 'R'], **kwargs)
        self.r, self.R = r, R

        self.pure = sdf_torus
        self.quadratic_threshold = quadratic_threshold_torus
        self.quadratic_distance = quadratic_distance_torus
    
    def num_dofs(self):
        return 2
    
    def eval(self, x, y, z):
        pts = np.concatenate((x.flatten()[:, None], y.flatten()[:, None], z.flatten()[:, None]), axis=-1)
        return np.array(self.pure(pts, self.local_full_x, 1.0).astype(np.float32).reshape(x.shape))

    def rz_bounds(self):
        """(R_max, |Z|_max) extent of the torus surface from its parameters: the
        circular tube of minor radius r about the R=R circle reaches R<=R+r and
        |Z|<=r. See PillPipeSDF.rz_bounds."""
        r, R = self.local_full_x
        return float(R + r), float(r)

    def to_vtk(self, name, nx=20, ny=20, nz=20):
        r, R = self.local_full_x
        pad = 0.1

        d = R+r
        # domain extents
        x_min, x_max = -(d + pad), (d + pad)
        y_min, y_max = -(d + pad), (d + pad)
        z_min, z_max = -(r + pad), (r + pad)
        
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




def cyl_sdf(P, P0, u, rad):
    v = P - P0
    proj = jnp.sum(v * u, axis=-1, keepdims=True) * u
    return jnp.linalg.norm(v - proj, axis=-1) - rad

# ---------- 3D circular pipe SDF ----------
def sdf_rennaissance(pts, params, sign):
    d1, d2, rr = params
    
    pts = jnp.abs(pts)

    A = jnp.array([d1,  0.0, 0.0])
    B = jnp.array([d1,  d2,  0.0])
    C = jnp.array([0.0, d2,  0.0])

    AB = B - A
    AB /= jnp.linalg.norm(AB)
    
    BC = C - B
    BC /= jnp.linalg.norm(BC)
    
    # Bisector plane normal
    n = 0.5 * (AB + BC)
    n /= jnp.linalg.norm(n)

    sdf1 = cyl_sdf(pts, A, AB, rr)
    sdf2 = cyl_sdf(pts, B, BC, rr)
    s = jnp.sum((pts - B) * n, axis=-1)
    sdf = jnp.where(s < 0, sdf1, sdf2)
    return sign*sdf

def quadratic_threshold_rennaissance(pts, params, sign, threshold):
    sls = sdf_rennaissance(pts, params, sign)
    d1, d2, rr = params
    return jnp.mean(jnp.maximum(threshold-sls, 0)**2)

def quadratic_distance_rennaissance(pts, params, sign, threshold):
    sls = sdf_rennaissance(pts, params, sign)
    mean_value = jnp.mean(sls)
    dvalue = jnp.abs(sls-mean_value)
    return jnp.mean(jnp.maximum(dvalue-threshold, 0)**2)

class RennaissanceSDF(Optimizable):
    def __init__(self, d1, d2, rr, **kwargs):
        super().__init__(depends_on=[], x0=np.array([d1, d2, rr]), names=['d1', 'd2', 'rr'], **kwargs)
        self.d1, self.d2, self.rr = d1, d2, rr

        self.pure = sdf_rennaissance
        self.quadratic_threshold = quadratic_threshold_rennaissance
        self.quadratic_distance = quadratic_distance_rennaissance
    
    def num_dofs(self):
        return 3
    
    def eval(self, x, y, z):
        pts = np.concatenate((x.flatten()[:, None], y.flatten()[:, None], z.flatten()[:, None]), axis=-1)
        return np.array(self.pure(pts, self.local_full_x, 1.0).astype(np.float32).reshape(x.shape))

    def rz_bounds(self):
        """(R_max, |Z|_max) extent of the surface from its parameters: the two
        radius-rr pipes (reflected into all octants) lie within |x|<=d1+rr,
        |y|<=d2+rr, |z|<=rr, so R<=hypot(d1+rr, d2+rr) and |Z|<=rr. See
        PillPipeSDF.rz_bounds."""
        d1, d2, rr = self.local_full_x
        return float(np.hypot(d1 + rr, d2 + rr)), float(rr)

    def to_vtk(self, name, nx=20, ny=20, nz=20):
        d1, d2, rr = self.local_full_x
        pad = 0.1

        d = np.max([d1, d2])
        # domain extents
        x_min, x_max = -(d+rr + pad), (d + rr + pad)
        y_min, y_max = -(d+rr + pad), (d + rr + pad)
        z_min, z_max = -(rr + pad),  (rr + pad)
        
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
    def __init__(self, sdf, entities, sign, minimum_distance, metric='threshold'):
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
        
        if metric == 'threshold':
            self.J_jax = jit(lambda pts, params, sign: self.sdf.quadratic_threshold(pts, params, sign, minimum_distance))
        else:
            self.J_jax = jit(lambda pts, params, sign: self.sdf.quadratic_distance(pts, params, sign, minimum_distance))

        self.thisgrad0 = jit(lambda pts, params, sign: grad(self.J_jax, argnums=0)(pts, params, sign))
        self.thisgrad1 = jit(lambda pts, params, sign: grad(self.J_jax, argnums=1)(pts, params, sign))
        super().__init__(depends_on=entities + [sdf])

    def longest_distance(self):
        max_dist_curve = -np.inf
        max_dist_fieldline = -np.inf
        max_dist_bs = -np.inf
        for sign, entity in zip(self.sign, self.entities):
            if isinstance(entity, _FIELDLINE_TYPES):
                curve = entity.curve
                sd_tmp = self.sdf.pure(curve.gamma(), self.sdf.local_full_x, sign)
                max_dist_tmp = np.abs(sd_tmp-sd_tmp.mean()).max()
                if max_dist_tmp > max_dist_fieldline:
                    max_dist_fieldline = max_dist_tmp
            elif isinstance(entity, Curve):
                sd_tmp = self.sdf.pure(entity.gamma(), self.sdf.local_full_x, sign)
                max_dist_tmp = np.abs(sd_tmp-sd_tmp.mean()).max()
                if max_dist_tmp > max_dist_curve:
                    max_dist_curve = max_dist_tmp
            elif isinstance(entity, BoozerSurface):
                sd_tmp = self.sdf.pure(entity.surface.gamma().reshape((-1, 3)), self.sdf.local_full_x, sign)
                max_dist_tmp = np.abs(sd_tmp-sd_tmp.mean()).max()
                if max_dist_tmp > max_dist_bs:
                    max_dist_bs = max_dist_tmp
            else:
                raise Exception('entity not supported')

        return max_dist_curve, max_dist_fieldline, max_dist_bs
 

    def shortest_distance(self):
        min_dist_curve = np.inf
        min_dist_fieldline = np.inf
        min_dist_bs = np.inf
        for sign, entity in zip(self.sign, self.entities):
            if isinstance(entity, _FIELDLINE_TYPES):
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
            if isinstance(entity, _FIELDLINE_TYPES):
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
            if isinstance(entity, _FIELDLINE_TYPES) or isinstance(entity, Curve):
                line = entity.curve if isinstance(entity, _FIELDLINE_TYPES) else entity
                
                dJ_dgamma = self.thisgrad0(line.gamma(), self.sdf.local_full_x, sign)
                dres_curve = line.dgamma_by_dcoeff_vjp(dJ_dgamma)
                dJ_dparams = self.thisgrad1(line.gamma(), self.sdf.local_full_x, sign)
                dres += dres_curve + Derivative({self.sdf:dJ_dparams}) 

                if isinstance(entity, _FIELDLINE_TYPES):
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


def make_lattice(x0, t1, t2, r0, N=160):
    x0 = x0.reshape((-1, 3))
    t1 = t1.reshape((-1, 3))
    t2 = t2.reshape((-1, 3))
    r0 = r0.flatten()
    
    base_curves = []
    for xyz0, T, B, r0 in zip(x0, t1, t2, r0):
        curve = CurveXYZFourier(np.linspace(0, 1, 160, endpoint=False), N)
        curve.set('xc(0)', xyz0[0])
        curve.set('yc(0)', xyz0[1])
        curve.set('zc(0)', xyz0[2])
       
        curve.set('xc(1)', r0*T[0])
        curve.set('xs(1)', r0*B[0])

        curve.set('yc(1)', r0*T[1])
        curve.set('ys(1)', r0*B[1])
        
        curve.set('zc(1)', r0*T[2])
        curve.set('zs(1)', r0*B[2])

        curve.fix_all()
        base_curves.append(curve)

    return base_curves

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Shared: arclength-uniform phi reparametrisation
#
# For a tube surface  p(s, theta) = c(s) + rr*(cos_th*n(s) + sin_th*b(s))
# the toroidal speed is  v(s, theta) = |dp/ds|.
# We want s_i(theta) such that  integral_0^{s_i} v ds = i/nphi * L_surf(theta)
# where  L_surf(theta) = integral_0^{L_total} v ds.
#
# Strategy:
#   1. Analytically integrate v over each centreline segment (exact).
#   2. Build cumulative arclength array over segments → fast bisection to
#      find the right segment, then Newton to find u within the segment.
# ---------------------------------------------------------------------------

def _speed(u, seg_type, r, rr, cos_th):
    """
    Toroidal speed  |dp/ds|  at local offset u within a segment.

    seg_type : 's' = straight  (dn/ds = 0  →  speed = 1)
               'a' = arc       (dn/ds = tang/r  →  speed = |1 + rr*cos_th/r|)
    cos_th   : scalar, cos(theta) for this toroidal angle
    """
    if seg_type == 's':
        return 1.0
    else:
        return abs(1.0 + rr * cos_th / r)


def _seg_arclength(seg_type, length, r, rr, cos_th):
    """Exact integral of |dp/ds| over a full segment of given arc length."""
    return length * _speed(0., seg_type, r, rr, cos_th)  # speed is constant per seg


def _invert_arclength_phi(s_targets, seg_types, cum, seg_lens, r, rr, cos_th):
    """
    For a single theta value (cos_th fixed), invert the cumulative toroidal
    arclength to find the centreline arclength s for each target in s_targets.

    Parameters
    ----------
    s_targets  : (nphi,)  desired cumulative toroidal arclengths
    seg_types  : list of 's'/'a' per segment
    cum        : (nseg+1,) cumulative centreline arclengths (cum[0]=0)
    seg_lens   : (nseg,)
    r, rr      : pill-pipe parameters
    cos_th     : cos(theta) for this theta value

    Returns
    -------
    s_out : (nphi,) centreline arclength values
    """
    nseg = len(seg_lens)

    # Cumulative *toroidal* arclength at each segment boundary
    cum_surf = np.zeros(nseg + 1)
    for k in range(nseg):
        cum_surf[k+1] = cum_surf[k] + _seg_arclength(seg_types[k], seg_lens[k], r, rr, cos_th)
    L_surf = cum_surf[-1]

    s_out = np.empty(len(s_targets))
    for i, tgt in enumerate(s_targets):
        tgt = tgt % L_surf   # wrap
        # find segment
        k = np.searchsorted(cum_surf[1:], tgt, side='right')
        k = min(k, nseg - 1)
        # remaining toroidal arclength within segment
        rem = tgt - cum_surf[k]
        spd = _speed(0., seg_types[k], r, rr, cos_th)
        # u = rem / speed  (speed is constant within each segment)
        u = rem / spd if spd > 0. else 0.
        s_out[i] = cum[k] + u
    return s_out

# ---------------------------------------------------------------------------
# Shared: arclength-uniform phi reparametrisation
#
# For a tube surface  p(s, theta) = c(s) + rr*(cos_th*n(s) + sin_th*b(s))
# the toroidal speed is  v(s, theta) = |dp/ds|.
# We want s_i(theta) such that  integral_0^{s_i} v ds = i/nphi * L_surf(theta)
# where  L_surf(theta) = integral_0^{L_total} v ds.
#
# Strategy:
#   1. Analytically integrate v over each centreline segment (exact).
#   2. Build cumulative arclength array over segments → fast bisection to
#      find the right segment, then Newton to find u within the segment.
# ---------------------------------------------------------------------------

def _speed(u, seg_type, r, rr, cos_th):
    """
    Toroidal speed  |dp/ds|  at local offset u within a segment.

    seg_type : 's' = straight  (dn/ds = 0  →  speed = 1)
               'a' = arc       (dn/ds = tang/r  →  speed = |1 + rr*cos_th/r|)
    cos_th   : scalar, cos(theta) for this toroidal angle
    """
    if seg_type == 's':
        return 1.0
    else:
        return abs(1.0 + rr * cos_th / r)


def _seg_arclength(seg_type, length, r, rr, cos_th):
    """Exact integral of |dp/ds| over a full segment of given arc length."""
    return length * _speed(0., seg_type, r, rr, cos_th)  # speed is constant per seg


def _invert_arclength_phi(s_targets, seg_types, cum, seg_lens, r, rr, cos_th):
    """
    For a single theta value (cos_th fixed), invert the cumulative toroidal
    arclength to find the centreline arclength s for each target in s_targets.

    Parameters
    ----------
    s_targets  : (nphi,)  desired cumulative toroidal arclengths
    seg_types  : list of 's'/'a' per segment
    cum        : (nseg+1,) cumulative centreline arclengths (cum[0]=0)
    seg_lens   : (nseg,)
    r, rr      : pill-pipe parameters
    cos_th     : cos(theta) for this theta value

    Returns
    -------
    s_out : (nphi,) centreline arclength values
    """
    nseg = len(seg_lens)

    # Cumulative *toroidal* arclength at each segment boundary
    cum_surf = np.zeros(nseg + 1)
    for k in range(nseg):
        cum_surf[k+1] = cum_surf[k] + _seg_arclength(seg_types[k], seg_lens[k], r, rr, cos_th)
    L_surf = cum_surf[-1]

    s_out = np.empty(len(s_targets))
    for i, tgt in enumerate(s_targets):
        tgt = tgt % L_surf   # wrap
        # find segment
        k = np.searchsorted(cum_surf[1:], tgt, side='right')
        k = min(k, nseg - 1)
        # remaining toroidal arclength within segment
        rem = tgt - cum_surf[k]
        spd = _speed(0., seg_types[k], r, rr, cos_th)
        # u = rem / speed  (speed is constant within each segment)
        u = rem / spd if spd > 0. else 0.
        s_out[i] = cum[k] + u
    return s_out


# ---------------------------------------------------------------------------
# TorusSDF  –  analytic arclength reparametrisation
#
# Toroidal speed: |dp/dphi_angle| = R + r*cos_th  (phi_angle = 2pi * phi_norm)
# This is constant in phi_angle for fixed theta, so arclength-uniform phi
# just means uniform phi_angle — no reparametrisation needed per se, BUT
# the total toroidal length depends on theta, so we still need to
# re-scale. Since speed is constant in phi for each theta, uniform phi_angle
# IS arclength-uniform for each theta independently.
# ---------------------------------------------------------------------------

def gamma(self, quadpoints_phi, quadpoints_theta):
    r, R = self.local_full_x
    phi_2pi   = np.asarray(quadpoints_phi)   * 2. * np.pi
    theta_2pi = np.asarray(quadpoints_theta) * 2. * np.pi
    nphi = len(phi_2pi)

    c_phi, s_phi = np.cos(phi_2pi), np.sin(phi_2pi)
    c_th,  s_th  = np.cos(theta_2pi), np.sin(theta_2pi)

    e_r   = np.stack([ c_phi,  s_phi, np.zeros(nphi)], axis=-1)
    e_phi = np.stack([-s_phi,  c_phi, np.zeros(nphi)], axis=-1)
    e_z   = np.tile([0., 0., 1.], (nphi, 1))

    XYZ = (R * e_r[:, None, :]
           + r * (c_th[None, :, None] * e_r[:, None, :]
                + s_th[None, :, None] * e_z[:, None, :]))

    T = np.broadcast_to(e_phi[:, None, :], XYZ.shape).copy()

    B = (-s_th[None, :, None] * e_r[:, None, :]
        +  c_th[None, :, None] * e_z[:, None, :])

    N = np.cross(T, B)
    N = N / np.linalg.norm(N, axis=-1, keepdims=True)

    return XYZ, T, B, N

TorusSDF.gamma = gamma


# ---------------------------------------------------------------------------
# PillPipeSDF  –  arclength-uniform in phi for each theta
# ---------------------------------------------------------------------------

def gamma(self, quadpoints_phi, quadpoints_theta):
    bx, by, r, rr = self.local_full_x
    phi_norm  = np.asarray(quadpoints_phi)
    theta_2pi = np.asarray(quadpoints_theta) * 2. * np.pi
    nphi, ntheta = len(phi_norm), len(theta_2pi)

    w, h = bx - r, by - r

    seg_lens  = np.array([2*w, 0.5*np.pi*r, 2*h, 0.5*np.pi*r,
                          2*w, 0.5*np.pi*r, 2*h, 0.5*np.pi*r])
    seg_types = ['s','a','s','a','s','a','s','a']
    L_total   = seg_lens.sum()
    cum       = np.concatenate([[0.], np.cumsum(seg_lens)])

    e_z = np.array([0., 0., 1.])
    c_th = np.cos(theta_2pi)   # (ntheta,)
    s_th = np.sin(theta_2pi)

    XYZ = np.empty((nphi, ntheta, 3))
    T   = np.empty((nphi, ntheta, 3))
    B   = np.empty((nphi, ntheta, 3))
    N   = np.empty((nphi, ntheta, 3))

    for j, (cth, sth) in enumerate(zip(c_th, s_th)):
        # Toroidal arclength of the full loop for this theta
        L_surf = sum(_seg_arclength(st, sl, r, rr, cth)
                     for st, sl in zip(seg_types, seg_lens))
        # Phase shift so that (phi=0, theta=0) lands on the X-axis at (bx+rr, 0, 0).
        # s0_surf = surface arclength from old start (top of seg 0) to midpoint of
        # seg 6 (right side, y=0):  segs 0-5  +  half of seg 6 (length h).
        # Arc segs (1,3,5) have speed (1 + rr*cth/r); straight segs have speed 1.
        s0_surf = 4.*w + 3.*h + 3.*(np.pi * r / 2.) * (1. + rr * cth / r)
        s_tgt = (phi_norm * L_surf + s0_surf) % L_surf
        # Invert to centreline arclengths
        s_cl = _invert_arclength_phi(s_tgt, seg_types, cum, seg_lens, r, rr, cth)

        # Evaluate geometry at each centreline arclength
        seg  = np.searchsorted(cum[1:], s_cl, side='right')
        seg  = np.clip(seg, 0, 7)
        u    = s_cl - cum[seg]

        cx = np.empty(nphi); cy = np.empty(nphi)
        tx = np.empty(nphi); ty = np.empty(nphi)
        nx = np.empty(nphi); ny = np.empty(nphi)
        dn_ds_x = np.empty(nphi); dn_ds_y = np.empty(nphi)

        for i in range(nphi):
            sg, ui = seg[i], u[i]
            if sg == 0:
                cx[i],cy[i]         =  w-ui,  by
                tx[i],ty[i]         = -1.,  0.
                nx[i],ny[i]         =  0.,  1.
                dn_ds_x[i],dn_ds_y[i] =  0.,  0.
            elif sg == 1:
                th = 0.5*np.pi + ui/r
                cx[i],cy[i]         = -w+r*np.cos(th),  h+r*np.sin(th)
                tx[i],ty[i]         = -np.sin(th),  np.cos(th)
                nx[i],ny[i]         =  np.cos(th),  np.sin(th)
                dn_ds_x[i],dn_ds_y[i] = -np.sin(th)/r, np.cos(th)/r
            elif sg == 2:
                cx[i],cy[i]         = -bx,  h-ui
                tx[i],ty[i]         =  0., -1.
                nx[i],ny[i]         = -1.,  0.
                dn_ds_x[i],dn_ds_y[i] =  0.,  0.
            elif sg == 3:
                th = np.pi + ui/r
                cx[i],cy[i]         = -w+r*np.cos(th), -h+r*np.sin(th)
                tx[i],ty[i]         = -np.sin(th),  np.cos(th)
                nx[i],ny[i]         =  np.cos(th),  np.sin(th)
                dn_ds_x[i],dn_ds_y[i] = -np.sin(th)/r, np.cos(th)/r
            elif sg == 4:
                cx[i],cy[i]         = -w+ui, -by
                tx[i],ty[i]         =  1.,  0.
                nx[i],ny[i]         =  0., -1.
                dn_ds_x[i],dn_ds_y[i] =  0.,  0.
            elif sg == 5:
                th = 1.5*np.pi + ui/r
                cx[i],cy[i]         =  w+r*np.cos(th), -h+r*np.sin(th)
                tx[i],ty[i]         = -np.sin(th),  np.cos(th)
                nx[i],ny[i]         =  np.cos(th),  np.sin(th)
                dn_ds_x[i],dn_ds_y[i] = -np.sin(th)/r, np.cos(th)/r
            elif sg == 6:
                cx[i],cy[i]         =  bx, -h+ui
                tx[i],ty[i]         =  0.,  1.
                nx[i],ny[i]         =  1.,  0.
                dn_ds_x[i],dn_ds_y[i] =  0.,  0.
            else:
                th = ui/r
                cx[i],cy[i]         =  w+r*np.cos(th),  h+r*np.sin(th)
                tx[i],ty[i]         = -np.sin(th),  np.cos(th)
                nx[i],ny[i]         =  np.cos(th),  np.sin(th)
                dn_ds_x[i],dn_ds_y[i] = -np.sin(th)/r, np.cos(th)/r

        cline = np.stack([cx, cy, np.zeros(nphi)], axis=-1)
        tang3 = np.stack([tx, ty, np.zeros(nphi)], axis=-1)
        n3    = np.stack([nx, ny, np.zeros(nphi)], axis=-1)
        dn3   = np.stack([dn_ds_x, dn_ds_y, np.zeros(nphi)], axis=-1)
        b3    = np.tile(e_z, (nphi, 1))

        XYZ[:, j, :] = cline + rr * (cth * n3 + sth * b3)

        dpds = tang3 + rr * cth * dn3
        T[:, j, :] = dpds / np.linalg.norm(dpds, axis=-1, keepdims=True)

        Bj = -sth * n3 + cth * b3
        B[:, j, :] = Bj   # already unit

        N[:, j, :] = np.cross(T[:, j, :], B[:, j, :])
        N[:, j, :] /= np.linalg.norm(N[:, j, :], axis=-1, keepdims=True)

    return XYZ, T, B, N

PillPipeSDF.gamma = gamma


# ---------------------------------------------------------------------------
# RennaissanceSDF  –  arclength-uniform in phi for each theta
#
# The bisector-plane transition from cylinder 1 to cylinder 2 happens at a
# point shifted by rr*cos_theta from the polygon corner.  So the effective
# arm lengths are 2*(d2+rr*cos_th) (vertical) and 2*(d1+rr*cos_th)
# (horizontal) — theta-dependent, exactly as (R+r*cos_th) for the torus.
#
# L_surf(theta) = 4*(d1+d2) + 8*rr*cos_theta
#
# The four arms and their transitions are continuous:
#   ARM 0 right  (d1+rr*cth, y,            rr*sth), y: -(d2+rr*cth) → +(d2+rr*cth)
#   ARM 1 top    (x,          d2+rr*cth,   rr*sth), x: +(d1+rr*cth) → -(d1+rr*cth)
#   ARM 2 left  -(d1+rr*cth), y,            rr*sth), y: +(d2+rr*cth) → -(d2+rr*cth)
#   ARM 3 bottom (x,         -(d2+rr*cth), rr*sth), x: -(d1+rr*cth) → +(d1+rr*cth)
# ---------------------------------------------------------------------------

def gamma(self, quadpoints_phi, quadpoints_theta):
    d1, d2, rr = self.local_full_x
    phi_norm  = np.asarray(quadpoints_phi)
    theta_2pi = np.asarray(quadpoints_theta) * 2. * np.pi
    nphi, ntheta = len(phi_norm), len(theta_2pi)

    e_z = np.array([0., 0., 1.])

    # Arm tangents and normals are theta-independent
    arm_tangs = np.array([[ 0., 1., 0.],   # right: going +y
                           [-1., 0., 0.],   # top:   going -x
                           [ 0.,-1., 0.],   # left:  going -y
                           [ 1., 0., 0.]])  # bottom: going +x
    arm_norms = np.array([[ 1., 0., 0.],   # outward +x
                           [ 0., 1., 0.],   # outward +y
                           [-1., 0., 0.],   # outward -x
                           [ 0.,-1., 0.]]) # outward -y

    c_th = np.cos(theta_2pi)   # (ntheta,)
    s_th = np.sin(theta_2pi)

    XYZ = np.empty((nphi, ntheta, 3))
    T   = np.empty((nphi, ntheta, 3))
    B   = np.empty((nphi, ntheta, 3))
    N   = np.empty((nphi, ntheta, 3))

    for j, (cth, sth) in enumerate(zip(c_th, s_th)):
        # Arm lengths depend on theta (bisector transition shifts by rr*cth)
        len_v  = 2. * (d2 + rr * cth)   # vertical   arms (right, left)
        len_h  = 2. * (d1 + rr * cth)   # horizontal arms (top, bottom)
        L_surf = 2. * (len_v + len_h)   # = 4*(d1+d2) + 8*rr*cth
        cum = np.array([0., len_v, len_v + len_h, 2.*len_v + len_h, L_surf])

        # Centreline arm starts are also theta-dependent
        arm_starts = np.array([
            [ d1,            -(d2 + rr*cth), 0.],  # ARM 0
            [ d1 + rr*cth,    d2,            0.],  # ARM 1
            [-d1,             d2 + rr*cth,   0.],  # ARM 2
            [-(d1 + rr*cth), -d2,            0.],  # ARM 3
        ])

        s_req = phi_norm * L_surf
        # Phase shift so that (phi=0, theta=0) lands on the X-axis at (d1+rr, 0, 0).
        # s0_surf = surface arclength from old start (bottom of ARM 0) to midpoint
        # of ARM 0 (y=0), which is exactly half the vertical arm length.
        s0_surf = d2 + rr * cth
        s_req = (s_req + s0_surf) % L_surf
        arm   = np.searchsorted(cum[1:], s_req, side='right')
        arm   = np.clip(arm, 0, 3)
        u     = s_req - cum[arm]

        cline = arm_starts[arm] + u[:, None] * arm_tangs[arm]   # (nphi, 3)
        n3    = arm_norms[arm]                                   # (nphi, 3)

        XYZ[:, j, :] = cline + rr * (cth * n3 + sth * e_z[None, :])

        # dp/ds = tang  (dn/ds = 0 on all straight arms)
        T[:, j, :] = arm_tangs[arm]

        # dp/dtheta = rr*(-sth*n + cth*e_z);  |dp/dtheta| = rr  (uniform)
        B[:, j, :] = -sth * n3 + cth * e_z[None, :]

        Nj = np.cross(T[:, j, :], B[:, j, :])
        N[:, j, :] = Nj / np.linalg.norm(Nj, axis=-1, keepdims=True)

    return XYZ, T, B, N

RennaissanceSDF.gamma = gamma


def _torus_centerline_gamma(self, quadpoints_phi):
    r, R = self.local_full_x
    phi_2pi = np.asarray(quadpoints_phi) * 2. * np.pi
    nphi    = len(phi_2pi)
    c_phi, s_phi = np.cos(phi_2pi), np.sin(phi_2pi)

    XYZ = R * np.stack([ c_phi,  s_phi, np.zeros(nphi)], axis=-1)
    T   =     np.stack([-s_phi,  c_phi, np.zeros(nphi)], axis=-1)  # e_phi
    N   =     np.stack([ c_phi,  s_phi, np.zeros(nphi)], axis=-1)  # e_r (outward)
    B   = np.tile([0., 0., 1.], (nphi, 1))                          # e_z
    return XYZ, T, N, B


def _pill_pipe_centerline_gamma(self, quadpoints_phi):
    bx, by, r, rr = self.local_full_x
    phi_norm = np.asarray(quadpoints_phi)
    nphi     = len(phi_norm)

    w, h = bx - r, by - r
    seg_lens = np.array([2*w, np.pi*r/2, 2*h, np.pi*r/2,
                         2*w, np.pi*r/2, 2*h, np.pi*r/2])
    L_cline = seg_lens.sum()
    cum     = np.concatenate([[0.], np.cumsum(seg_lens)])

    # phi=0 at midpoint of right segment (bx, 0, 0)
    s0  = 4.*w + 3.*h + 3.*np.pi*r / 2.
    s   = (phi_norm * L_cline + s0) % L_cline
    seg = np.searchsorted(cum[1:], s, side='right')
    seg = np.clip(seg, 0, 7)
    u   = s - cum[seg]

    cx = np.empty(nphi); cy = np.empty(nphi)
    tx = np.empty(nphi); ty = np.empty(nphi)
    nx = np.empty(nphi); ny = np.empty(nphi)

    for i in range(nphi):
        sg, ui = seg[i], u[i]
        if sg == 0:
            cx[i],cy[i] =  w-ui,  by;  tx[i],ty[i] = -1., 0.;  nx[i],ny[i] =  0., 1.
        elif sg == 1:
            th = 0.5*np.pi + ui/r
            cx[i],cy[i] = -w+r*np.cos(th),  h+r*np.sin(th)
            tx[i],ty[i] = -np.sin(th),  np.cos(th);  nx[i],ny[i] = np.cos(th),  np.sin(th)
        elif sg == 2:
            cx[i],cy[i] = -bx,  h-ui;  tx[i],ty[i] =  0., -1.;  nx[i],ny[i] = -1., 0.
        elif sg == 3:
            th = np.pi + ui/r
            cx[i],cy[i] = -w+r*np.cos(th), -h+r*np.sin(th)
            tx[i],ty[i] = -np.sin(th),  np.cos(th);  nx[i],ny[i] = np.cos(th),  np.sin(th)
        elif sg == 4:
            cx[i],cy[i] = -w+ui, -by;  tx[i],ty[i] =  1., 0.;  nx[i],ny[i] =  0., -1.
        elif sg == 5:
            th = 1.5*np.pi + ui/r
            cx[i],cy[i] =  w+r*np.cos(th), -h+r*np.sin(th)
            tx[i],ty[i] = -np.sin(th),  np.cos(th);  nx[i],ny[i] = np.cos(th),  np.sin(th)
        elif sg == 6:
            cx[i],cy[i] =  bx, -h+ui;  tx[i],ty[i] =  0., 1.;  nx[i],ny[i] =  1., 0.
        else:
            th = ui/r
            cx[i],cy[i] =  w+r*np.cos(th),  h+r*np.sin(th)
            tx[i],ty[i] = -np.sin(th),  np.cos(th);  nx[i],ny[i] = np.cos(th),  np.sin(th)

    XYZ = np.stack([cx, cy, np.zeros(nphi)], axis=-1)
    T   = np.stack([tx, ty, np.zeros(nphi)], axis=-1)
    N   = np.stack([nx, ny, np.zeros(nphi)], axis=-1)
    B   = np.tile([0., 0., 1.], (nphi, 1))
    return XYZ, T, N, B


def _rennaissance_centerline_gamma(self, quadpoints_phi):
    d1, d2, rr = self.local_full_x
    phi_norm   = np.asarray(quadpoints_phi)
    nphi       = len(phi_norm)

    arm_lens   = np.array([2.*d2, 2.*d1, 2.*d2, 2.*d1])
    arm_tangs  = np.array([[ 0., 1., 0.], [-1., 0., 0.],
                           [ 0.,-1., 0.], [ 1., 0., 0.]])
    arm_norms  = np.array([[ 1., 0., 0.], [ 0., 1., 0.],
                           [-1., 0., 0.], [ 0.,-1., 0.]])
    arm_starts = np.array([[ d1, -d2, 0.],
                           [ d1,  d2, 0.],
                           [-d1,  d2, 0.],
                           [-d1, -d2, 0.]])

    L_cline = arm_lens.sum()   # 4*(d1 + d2)
    cum     = np.concatenate([[0.], np.cumsum(arm_lens)])

    # phi=0 at midpoint of right arm (d1, 0, 0)
    s0  = d2
    s   = (phi_norm * L_cline + s0) % L_cline
    arm = np.searchsorted(cum[1:], s, side='right')
    arm = np.clip(arm, 0, 3)
    u   = s - cum[arm]

    XYZ = arm_starts[arm] + u[:, None] * arm_tangs[arm]
    T   = arm_tangs[arm].copy()
    N   = arm_norms[arm].copy()
    B   = np.tile([0., 0., 1.], (nphi, 1))
    return XYZ, T, N, B


TorusSDF.centerline_gamma      = _torus_centerline_gamma
PillPipeSDF.centerline_gamma   = _pill_pipe_centerline_gamma
RennaissanceSDF.centerline_gamma = _rennaissance_centerline_gamma
