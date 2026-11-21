"""shapedvessel.py -- pill-pipe vessels with a NON-circular tube cross-section, drop-in replacements for
star_lite_design.utils.pillpipevessel.PillPipeSDF (same interface: pure / quadratic_threshold / quadratic_distance /
eval / rz_bounds / to_vtk; JAX, so VesselDistance gets its gradients automatically).

Both keep Andrew's planform: the tube centre line is the rounded rectangle in z = 0 with half-widths bx, by and corner
radius r. Local cross-section coordinates of a point: u = signed in-plane distance to that centre line (sdf_round_rect,
> 0 outward), v = z.
  EllipsePipeSDF (bx, by, r, a, b):        ellipse u^2/a^2 + v^2/b^2 = 1   (a = in-plane semi-axis, b = vertical)
  RectPipeSDF    (bx, by, r, ha, hb, rc):  rounded rectangle |u| <= ha, |v| <= hb, corner radius rc
Both reduce to Andrew's circular pipe of radius rr for a = b = rr, and for ha = hb = rc = rr, so a run can start from
an existing device without changing anything. The returned value is the signed distance in the (u, v) plane
(exact for the rectangle; for the ellipse the exact closest point, found by a few Newton steps from three starts).
This 2-D cross-section distance IS the exact 3-D Euclidean signed distance (as for PillPipeSDF): the straight parts
are extrusions and the corner parts surfaces of revolution about the vertical corner axis, and for both the closest
surface point lies in the cross-section plane through the point. This needs the validity constraints (in-plane half-width
< r, so the tube does not overlap itself). Checked against brute-force distances to a dense surface sampling: agreement
to the sampling error (median 0.004 mm), |grad| = 1 to 1e-10."""
import numpy as np
import jax
import jax.numpy as jnp
from simsopt._core import Optimizable
from star_lite_design.utils.pillpipevessel import sdf_round_rect, safe_norm2d

jax.config.update('jax_enable_x64', True)


# ---------------------------------------------------------------- ellipse: exact distance in the cross-section plane
def _ellipse_dist(p, q, a, b, n_iter=14):
    """signed distance of (p, q) to the ellipse with semi-axes a (p direction), b (q direction); vectorised"""
    P, Q = jnp.abs(p), jnp.abs(q)
    c = a * a - b * b

    def newton(t):
        for _ in range(n_iter):
            st, ct = jnp.sin(t), jnp.cos(t)
            f = c * st * ct - P * a * st + Q * b * ct
            fp = c * (ct * ct - st * st) - P * a * ct - Q * b * st
            fp = jnp.where(jnp.abs(fp) > 1e-14, fp, jnp.where(fp >= 0, 1e-14, -1e-14))
            t = jnp.clip(t - f / fp, 0.0, 0.5 * jnp.pi)
        return t

    best = None
    for t0 in (jnp.arctan2(a * Q, b * P), jnp.zeros_like(P), jnp.full_like(P, 0.5 * jnp.pi)):
        t = newton(t0)
        d = safe_norm2d(P - a * jnp.cos(t), Q - b * jnp.sin(t))
        best = d if best is None else jnp.minimum(best, d)
    inside = (P / a) ** 2 + (Q / b) ** 2 < 1.0
    return jnp.where(inside, -best, best)


def sdf_ellipse_pipe(pts, params, sign):
    bx, by, r, a, b = params
    u = sdf_round_rect(pts[:, 0], pts[:, 1], bx, by, r)
    return sign * _ellipse_dist(u, pts[:, 2], a, b)


def _cons_ellipse(params):
    bx, by, r, a, b = params
    return (jnp.maximum(r - bx, 0) ** 2 + jnp.maximum(r - by, 0) ** 2      # corner radius fits the box
            + jnp.maximum(a - r, 0) ** 2                                  # in-plane semi-axis < corner radius (no self-intersection)
            + jnp.maximum(-bx, 0) ** 2 + jnp.maximum(-by, 0) ** 2 + jnp.maximum(-r, 0) ** 2
            + jnp.maximum(-a, 0) ** 2 + jnp.maximum(-b, 0) ** 2)


def quadratic_threshold_ellipse_pipe(pts, params, sign, threshold):
    return jnp.mean(jnp.maximum(threshold - sdf_ellipse_pipe(pts, params, sign), 0) ** 2) + _cons_ellipse(params)


def quadratic_distance_ellipse_pipe(pts, params, sign, threshold):
    s = sdf_ellipse_pipe(pts, params, sign)
    return jnp.mean(jnp.maximum(jnp.abs(s - jnp.mean(s)) - threshold, 0) ** 2)


# ---------------------------------------------------------------- rounded-rectangle cross-section (exact)
def sdf_rect_pipe(pts, params, sign):
    bx, by, r, ha, hb, rc = params
    u = sdf_round_rect(pts[:, 0], pts[:, 1], bx, by, r)
    return sign * sdf_round_rect(u, pts[:, 2], ha, hb, rc)


def _cons_rect(params):
    bx, by, r, ha, hb, rc = params
    return (jnp.maximum(r - bx, 0) ** 2 + jnp.maximum(r - by, 0) ** 2
            + jnp.maximum(ha - r, 0) ** 2                                 # in-plane half-width < planform corner radius
            + jnp.maximum(rc - ha, 0) ** 2 + jnp.maximum(rc - hb, 0) ** 2  # corner radius fits the cross-section
            + jnp.maximum(-bx, 0) ** 2 + jnp.maximum(-by, 0) ** 2 + jnp.maximum(-r, 0) ** 2
            + jnp.maximum(-ha, 0) ** 2 + jnp.maximum(-hb, 0) ** 2 + jnp.maximum(-rc, 0) ** 2)


def quadratic_threshold_rect_pipe(pts, params, sign, threshold):
    return jnp.mean(jnp.maximum(threshold - sdf_rect_pipe(pts, params, sign), 0) ** 2) + _cons_rect(params)


def quadratic_distance_rect_pipe(pts, params, sign, threshold):
    s = sdf_rect_pipe(pts, params, sign)
    return jnp.mean(jnp.maximum(jnp.abs(s - jnp.mean(s)) - threshold, 0) ** 2)


class _ShapedPipe(Optimizable):
    NAMES = ()

    def num_dofs(self):
        return len(self.NAMES)

    def eval(self, x, y, z):
        pts = np.concatenate((np.ravel(x)[:, None], np.ravel(y)[:, None], np.ravel(z)[:, None]), axis=-1)
        return np.array(self.pure(pts, self.local_full_x, 1.0)).reshape(np.shape(x))

    def half_extent(self):
        """(in-plane, vertical) half-size of the cross-section"""
        raise NotImplementedError

    def rz_bounds(self):
        bx, by = self.local_full_x[:2]; hu, hv = self.half_extent()
        return float(np.hypot(bx + hu, by + hu)), float(hv)

    def to_vtk(self, name, nx=20, ny=20, nz=20):
        from pyevtk.hl import gridToVTK
        bx, by = self.local_full_x[:2]; hu, hv = self.half_extent(); pad = 0.1
        xs = np.linspace(-(bx + hu + pad), bx + hu + pad, nx); ys = np.linspace(-(by + hu + pad), by + hu + pad, ny); zs = np.linspace(-(hv + pad), hv + pad, nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        gridToVTK(name, x=xs, y=ys, z=zs, cellData=None, pointData={'sdf': np.ascontiguousarray(self.eval(X, Y, Z).astype(np.float32))})


class EllipsePipeSDF(_ShapedPipe):
    NAMES = ('bx', 'by', 'r', 'a', 'b')

    def __init__(self, bx, by, r, a, b, **kwargs):
        super().__init__(depends_on=[], x0=np.array([bx, by, r, a, b], float), names=list(self.NAMES), **kwargs)
        self.bx, self.by, self.r, self.a, self.b = bx, by, r, a, b
        self.pure = sdf_ellipse_pipe
        self.quadratic_threshold = quadratic_threshold_ellipse_pipe
        self.quadratic_distance = quadratic_distance_ellipse_pipe

    def half_extent(self):
        return float(abs(self.local_full_x[3])), float(abs(self.local_full_x[4]))

    def geometric_violation(self):
        bx, by, r, a, b = self.local_full_x
        rel = lambda x, y: max(x - y, 0.0) / abs(y) if y != 0 else max(x, 0.0)
        return max(rel(r, bx), rel(r, by), rel(a, r), max(-bx, 0.), max(-by, 0.), max(-r, 0.), max(-a, 0.), max(-b, 0.))

    @classmethod
    def from_pill(cls, pill):
        bx, by, r, rr = [float(v) for v in pill.local_full_x]
        return cls(bx, by, r, rr, rr)


class RectPipeSDF(_ShapedPipe):
    NAMES = ('bx', 'by', 'r', 'ha', 'hb', 'rc')

    def __init__(self, bx, by, r, ha, hb, rc, **kwargs):
        super().__init__(depends_on=[], x0=np.array([bx, by, r, ha, hb, rc], float), names=list(self.NAMES), **kwargs)
        self.bx, self.by, self.r, self.ha, self.hb, self.rc = bx, by, r, ha, hb, rc
        self.pure = sdf_rect_pipe
        self.quadratic_threshold = quadratic_threshold_rect_pipe
        self.quadratic_distance = quadratic_distance_rect_pipe

    def half_extent(self):
        return float(abs(self.local_full_x[3])), float(abs(self.local_full_x[4]))

    def geometric_violation(self):
        bx, by, r, ha, hb, rc = self.local_full_x
        rel = lambda x, y: max(x - y, 0.0) / abs(y) if y != 0 else max(x, 0.0)
        return max(rel(r, bx), rel(r, by), rel(ha, r), rel(rc, ha), rel(rc, hb),
                   max(-bx, 0.), max(-by, 0.), max(-r, 0.), max(-ha, 0.), max(-hb, 0.), max(-rc, 0.))

    @classmethod
    def from_pill(cls, pill):
        bx, by, r, rr = [float(v) for v in pill.local_full_x]
        return cls(bx, by, r, rr, rr, rr)
