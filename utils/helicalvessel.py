from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from simsopt._core import Optimizable
from pyevtk.hl import gridToVTK  # pip install pyevtk

# ---------- vessel SDF about a CurveXYZFourierSymmetries centerline ----------
# Centerline parametrized exactly as simsopt's CurveXYZFourierSymmetries, with
# field-period (nfp) and optional stellarator symmetry, parameter theta in [0,1):
#   xhat = sum_{m=0}^{N} xc[m] cos(2 pi nfp m t) + sum_{m=1}^{N} xs[m] sin(2 pi nfp m t)
#   yhat = sum_{m=0}^{N} yc[m] cos(2 pi nfp m t) + sum_{m=1}^{N} ys[m] sin(2 pi nfp m t)
#   z    = sum_{m=0}^{N} zc[m] cos(2 pi nfp m t) + sum_{m=1}^{N} zs[m] sin(2 pi nfp m t)
#   x = xhat cos(2 pi ntor t) - yhat sin(2 pi ntor t)
#   y = xhat sin(2 pi ntor t) + yhat cos(2 pi ntor t)
# Stell-sym is the special case xs = yc = zc = 0.   Exact SDF where R * kappa_max < 1.
# DOF vector: params = concat([xc(N+1), xs(N), yc(N+1), ys(N), zc(N+1), zs(N), rr])

def _unpack(params):
    N = (params.shape[0] - 4) // 6          # Fourier order
    nc, ns = N + 1, N
    o = 0
    xc = params[o:o+nc]; o += nc
    xs = params[o:o+ns]; o += ns
    yc = params[o:o+nc]; o += nc
    ys = params[o:o+ns]; o += ns
    zc = params[o:o+nc]; o += nc
    zs = params[o:o+ns]; o += ns
    return xc, xs, yc, ys, zc, zs, params[-1]

def _C(t, xc, xs, yc, ys, zc, zs, nfp, ntor):
    mc = jnp.arange(xc.shape[0])            # 0..N   (cosine, incl. constant)
    ms = jnp.arange(1, xs.shape[0] + 1)     # 1..N   (sine)
    cc = jnp.cos(2 * jnp.pi * nfp * mc * t)
    ss = jnp.sin(2 * jnp.pi * nfp * ms * t)
    xhat = xc @ cc + xs @ ss
    yhat = yc @ cc + ys @ ss
    z    = zc @ cc + zs @ ss
    c, s = jnp.cos(2 * jnp.pi * ntor * t), jnp.sin(2 * jnp.pi * ntor * t)
    return jnp.stack([xhat * c - yhat * s, xhat * s + yhat * c, z])

def sdf_helical_vessel(pts, params, sign, nfp, ntor, n_grid=256, n_newton=8):
    """Signed distance from each point in `pts` to the constant-radius tube.

    The exact SDF is  Phi(p) = ||p - C(t*)|| - R,  where C(t*) is the closest
    point on the centerline to p (the "foot point") and R is the tube radius.
    There is no closed form for t*, so for each query point we (1) seed the
    global basin with a coarse scan, (2) polish to machine precision with
    Newton, and (3) detach the solve so autodiff still returns the exact
    gradient (envelope theorem).

    Args:
        pts:      (M, 3) query points.
        params:   flat DOF vector (see _unpack): the Fourier harmonics + radius.
        sign:     +1 for outside-positive; -1 flips the inside/outside convention.
        nfp:      field-period number (multiplies the basis frequencies).
        ntor:     toroidal winding number (rotates xhat/yhat into x/y).
        n_grid:   coarse-scan resolution; use >~ 4*max(nfp*N, ntor) so the scan
                  lands in the correct global basin before Newton refines it.
        n_newton: number of Newton steps (quadratic convergence near the min).
    Returns:
        (M,) array of signed distances.
    """
    xc, xs, yc, ys, zc, zs, R = _unpack(params)        # split harmonics + radius

    def one(p):                                        # --- SDF for one point p ---
        # f(t): half the squared distance from p to the centerline point C(t).
        # Its minimizer over t is the foot point (nearest point on the curve).
        f = lambda t: 0.5 * jnp.sum((p - _C(t, xc, xs, yc, ys, zc, zs, nfp, ntor)) ** 2)

        # f'(t) and f''(t) by autodiff -- no hand-coded curve derivatives needed.
        g, h = jax.grad(f), jax.grad(jax.grad(f))

        # (1) GLOBAL SEARCH: sample f on a uniform grid over one period [0,1) and
        #     pick the parameter with the smallest distance. This chooses the
        #     correct basin so Newton can't fall into a wrong local minimum on a
        #     curve that wiggles or folds back near itself.
        grid = jnp.linspace(0.0, 1.0, n_grid, endpoint=False)
        t = grid[jnp.argmin(jax.vmap(f)(grid))]

        # (2) LOCAL POLISH: Newton on the normality condition f'(t) = 0, i.e.
        #     (p - C).C' = 0 (residual perpendicular to the tangent). In-regime
        #     f'' > 0, so jnp.maximum(., 1e-9) only guards floating-point
        #     underflow rather than changing the step; convergence is quadratic.
        t, _ = lax.scan(
            lambda t, _: (t - g(t) / jnp.maximum(h(t), 1e-9), None),
            t, None, length=n_newton)

        # (3) DETACH the foot point. By the envelope theorem df/dt = 0 at t*, so
        #     the chain-rule term that would flow through dt*/d(params) is exactly
        #     zero -- freezing t* loses nothing, makes grad(Phi) the exact normal
        #     and shape gradient, and skips backprop through argmin (which is not
        #     differentiable) and through the Newton iterations.
        t = lax.stop_gradient(t)

        # Signed distance: outward distance from p to its foot point, minus R.
        return sign * (jnp.linalg.norm(p - _C(t, xc, xs, yc, ys, zc, zs, nfp, ntor)) - R)

    return jax.vmap(one)(pts)                          # vectorize over all M points

def kappa_helical_vessel(t, params, nfp, ntor):
    """Curvature kappa(t) of the centerline at parameter(s) t in [0,1).

    Accepts a scalar or a 1-D array of parameters and returns kappa with the
    same shape, using kappa = ||C' x C''|| / ||C'||^3 (derivatives by autodiff).
    """
    xc, xs, yc, ys, zc, zs, _ = _unpack(params)
    C = lambda s: _C(s, xc, xs, yc, ys, zc, zs, nfp, ntor)
    def kap(s):
        Cp, Cpp = jax.jacfwd(C)(s), jax.jacfwd(jax.jacfwd(C))(s)
        return jnp.linalg.norm(jnp.cross(Cp, Cpp)) / jnp.linalg.norm(Cp) ** 3
    t = jnp.asarray(t)
    return kap(t) if t.ndim == 0 else jax.vmap(kap)(t)

def _kappa_max(params, nfp, ntor, n_grid=512):
    return jnp.max(kappa_helical_vessel(
        jnp.linspace(0.0, 1.0, n_grid, endpoint=False), params, nfp, ntor))

def quadratic_threshold_helical_vessel(pts, params, sign, threshold, nfp, ntor):
    sls = sdf_helical_vessel(pts, params, sign, nfp, ntor)
    *_, R = _unpack(params)
    reach = jnp.maximum(R * _kappa_max(params, nfp, ntor) - 1.0, 0.0) ** 2   # stay in regime
    return jnp.mean(jnp.maximum(threshold - sls, 0) ** 2) + reach

def quadratic_distance_helical_vessel(pts, params, sign, threshold, nfp, ntor):
    sls = sdf_helical_vessel(pts, params, sign, nfp, ntor)
    dvalue = jnp.abs(sls - jnp.mean(sls))
    return jnp.mean(jnp.maximum(dvalue - threshold, 0) ** 2)

class HelicalVesselSDF(Optimizable):
    """Tube of radius rr about a closed CurveXYZFourierSymmetries centerline
    (order N = len(xc)-1).  xc, ys, zs are the stell-sym harmonics; xs, yc, zc
    are the symmetry-breaking ones (zero by default -> stellarator symmetric)."""
    def __init__(self, xc, ys, zs, rr, nfp, ntor=1, xs=None, yc=None, zc=None, **kwargs):
        xc, ys, zs = (np.asarray(a, float) for a in (xc, ys, zs))
        N = len(xc) - 1
        xs = np.zeros(N)     if xs is None else np.asarray(xs, float)   # stell-sym: xs=0
        yc = np.zeros(N + 1) if yc is None else np.asarray(yc, float)   # stell-sym: yc=0
        zc = np.zeros(N + 1) if zc is None else np.asarray(zc, float)   # stell-sym: zc=0
        assert len(xs) == len(ys) == len(zs) == N, "sine arrays length N (m=1..N)"
        assert len(yc) == len(zc) == N + 1, "cosine arrays length N+1 (m=0..N)"
        x0 = np.concatenate([xc, xs, yc, ys, zc, zs, [rr]])
        parts = [('xc', xc, 0), ('xs', xs, 1), ('yc', yc, 0),
                 ('ys', ys, 1), ('zc', zc, 0), ('zs', zs, 1)]
        names = [f'{p}({m})' for p, a, m0 in parts
                 for m in range(m0, m0 + len(a))] + ['rr']
        super().__init__(depends_on=[], x0=x0, names=names, **kwargs)
        self.nfp, self.ntor = int(nfp), int(ntor)
        self.pure = partial(sdf_helical_vessel, nfp=self.nfp, ntor=self.ntor)
        self.quadratic_threshold = partial(quadratic_threshold_helical_vessel,
                                            nfp=self.nfp, ntor=self.ntor)
        self.quadratic_distance = partial(quadratic_distance_helical_vessel,
                                           nfp=self.nfp, ntor=self.ntor)

    @classmethod
    def from_curve_xyz_fourier_symmetries(cls, curve, rr, **kwargs):
        """Build directly from a simsopt CurveXYZFourierSymmetries, copying its
        harmonics, nfp, and ntor. Works whether or not the curve is stellarator
        symmetric: harmonics absent from the curve's DOF set (xs/yc/zc when
        stellsym=True) are simply read as zero."""
        # name -> value over ALL dofs (free and fixed); strip any 'ObjName:' prefix
        d = {n.split(':')[-1]: v for n, v in zip(curve.full_dof_names, curve.full_x)}
        N, nfp = curve.order, curve.nfp
        ntor = getattr(curve, 'ntor', 1)
        get = lambda p, m: d.get(f'{p}({m})', 0.0)
        cos = lambda p: [get(p, m) for m in range(N + 1)]     # m = 0..N
        sin = lambda p: [get(p, m) for m in range(1, N + 1)]  # m = 1..N
        return cls(cos('xc'), sin('ys'), sin('zs'), rr, nfp, ntor=ntor,
                   xs=sin('xs'), yc=cos('yc'), zc=cos('zc'), **kwargs)

    def num_dofs(self):
        return len(self.local_full_x)

    def kappa(self, t=None, n_grid=512):
        """Centerline curvature kappa.

        The exact-SDF regime requires rr * kappa < 1 everywhere; sample this to
        find where it is violated, e.g. ``t[vessel.kappa() * vessel.rr >= 1]``.

        Args:
            t: parameter(s) in [0,1) at which to evaluate.  If None, a uniform
               grid of ``n_grid`` points is used.
        Returns:
            (t, kappa) arrays when ``t`` is None, otherwise just kappa(t).
        """
        if t is None:
            t = np.linspace(0.0, 1.0, n_grid, endpoint=False)
        return np.asarray(kappa_helical_vessel(np.asarray(t, float),
                                               self.local_full_x, self.nfp, self.ntor))

    @property
    def rr(self):
        return float(self.local_full_x[-1])

    def eval(self, x, y, z):
        pts = np.concatenate((x.flatten()[:, None], y.flatten()[:, None],
                              z.flatten()[:, None]), axis=-1)
        return np.array(self.pure(pts, self.local_full_x, 1.0).astype(np.float32).reshape(x.shape))

    def rz_bounds(self):
        """Conservative (R_max, |Z|_max): rotation is norm-preserving so
        R = hypot(xhat, yhat); per-component mode sums bound the rest; tube adds rr."""
        xc, xs, yc, ys, zc, zs, R = _unpack(self.local_full_x)
        xb = np.sum(np.abs(xc)) + np.sum(np.abs(xs))
        yb = np.sum(np.abs(yc)) + np.sum(np.abs(ys))
        zb = np.sum(np.abs(zc)) + np.sum(np.abs(zs))
        return float(np.hypot(xb, yb) + R), float(zb + R)

    def to_vtk(self, name, nx=40, ny=40, nz=40):
        R_max, Z_max = self.rz_bounds(); pad = 0.1
        xs = np.linspace(-(R_max + pad), R_max + pad, nx).astype(np.float32)
        ys = np.linspace(-(R_max + pad), R_max + pad, ny).astype(np.float32)
        zs = np.linspace(-(Z_max + pad), Z_max + pad, nz).astype(np.float32)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.concatenate((X.flatten()[:, None], Y.flatten()[:, None],
                              Z.flatten()[:, None]), axis=-1)
        D = np.array(self.pure(pts, self.local_full_x, 1.0).astype(np.float32).reshape((nx, ny, nz)))
        gridToVTK(name, x=xs, y=ys, z=zs, cellData=None, pointData={"sdf": D})
