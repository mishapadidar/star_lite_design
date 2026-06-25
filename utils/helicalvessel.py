from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
from simsopt._core import Optimizable
from pyevtk.hl import gridToVTK  # pip install pyevtk

# Single coarse sampling resolution shared by every grid in this module: the SDF
# foot-point bracketing scan over [0,1) and the geometric metrics (kappa, dR/ds,
# arclength, radius) over one field period [0,1/nfp). One value suffices -- the
# binding requirement is that the SDF scan bracket the global basin, i.e.
# N_GRID >~ 4*max(nfp*N, ntor); the per-period geometric grids are then over-
# resolved by a factor nfp (harmless). Bump it if a higher centerline order needs
# finer bracketing.
N_GRID = 64

# Foot-point solve tolerance: the guarded Newton drives |phi'(t*)| below this. It is
# also the convergence test elsewhere -- a residual above FOOT_TOL means the solve hit
# maxiter (degenerate tube, R*kappa >= 1), which callers use to flag a bad config.
FOOT_TOL = 1e-11

# ---------- vessel SDF about a CurveXYZFourierSymmetries centerline ----------
# Centerline parametrized exactly as simsopt's CurveXYZFourierSymmetries, with
# field-period (nfp) and optional stellarator symmetry, parameter theta in [0,1):
#   xhat = sum_{m=0}^{N} xc[m] cos(2 pi nfp m t) + sum_{m=1}^{N} xs[m] sin(2 pi nfp m t)
#   yhat = sum_{m=0}^{N} yc[m] cos(2 pi nfp m t) + sum_{m=1}^{N} ys[m] sin(2 pi nfp m t)
#   z    = sum_{m=0}^{N} zc[m] cos(2 pi nfp m t) + sum_{m=1}^{N} zs[m] sin(2 pi nfp m t)
#   x = xhat cos(2 pi ntor t) - yhat sin(2 pi ntor t)
#   y = xhat sin(2 pi ntor t) + yhat cos(2 pi ntor t)
# Stell-sym is the special case xs = yc = zc = 0.
# The tube radius is itself a Fourier series R(t) in the toroidal parameter, of
# the same order N as the centerline (constant radius = only rc[0] nonzero):
#   R(t) = sum_{m=0}^{N} rc[m] cos(2 pi nfp m t) + sum_{m=1}^{N} rs[m] sin(2 pi nfp m t)
# The tube is the union of balls B(C(t), R(t)); its exact signed distance is
#   Phi(p) = min_t ( ||p - C(t)|| - R(t) ),
# valid (a smooth, non-self-intersecting tube) where max_t R*kappa < 1 AND
# max_t |dR/ds| < 1 (ds = ||C'|| dt arclength).
# The radius Fourier order `r_order` (Mr) is independent of the centerline order N
# and is a static flag too: Mr=0 is a constant-radius tube (only rc[0]); Mr>0 lets
# the radius vary with the toroidal parameter.
# DOF layout depends on `stellsym` and `r_order` (Mr); both are static flags threaded
# through every function:
#   stellsym=False:  params = concat([xc(N+1), xs(N), yc(N+1), ys(N), zc(N+1), zs(N),
#                                      rc(Mr+1), rs(Mr)])
#   stellsym=True:   params = concat([xc(N+1), ys(N), zs(N), rc(Mr+1)])  -- the symmetry-
#                    breaking harmonics xs, yc, zc (and the radius sines rs) are absent
#                    entirely (NOT just fixed); _unpack reconstructs them as zeros.

def _unpack(params, stellsym, r_order):
    Mr = int(r_order)
    if stellsym:
        # radius rc(Mr+1); centerline xc(N+1), ys(N), zs(N) = 3N+1
        n_cl = params.shape[0] - (Mr + 1)
        N = (n_cl - 1) // 3
        nc, ns = N + 1, N
        o = 0
        xc = params[o:o+nc]; o += nc
        ys = params[o:o+ns]; o += ns
        zs = params[o:o+ns]; o += ns
        rc = params[o:o+(Mr + 1)]; o += (Mr + 1)
        # The symmetry-breaking harmonics (and radius sines) do not exist as dofs.
        xs = jnp.zeros(ns)
        yc = jnp.zeros(nc)
        zc = jnp.zeros(nc)
        rs = jnp.zeros(Mr)
        return xc, xs, yc, ys, zc, zs, rc, rs
    # radius rc(Mr+1)+rs(Mr) = 2Mr+1; centerline 6N+3
    n_cl = params.shape[0] - (2 * Mr + 1)
    N = (n_cl - 3) // 6
    nc, ns = N + 1, N
    o = 0
    xc = params[o:o+nc]; o += nc
    xs = params[o:o+ns]; o += ns
    yc = params[o:o+nc]; o += nc
    ys = params[o:o+ns]; o += ns
    zc = params[o:o+nc]; o += nc
    zs = params[o:o+ns]; o += ns
    rc = params[o:o+(Mr + 1)]; o += (Mr + 1)
    rs = params[o:o+Mr]; o += Mr
    return xc, xs, yc, ys, zc, zs, rc, rs

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

def _R(t, rc, rs, nfp):
    """Tube radius R(t): a Fourier series in the toroidal parameter t in [0,1)."""
    mc = jnp.arange(rc.shape[0])            # 0..N   (cosine, incl. constant)
    ms = jnp.arange(1, rs.shape[0] + 1)     # 1..N   (sine)
    return rc @ jnp.cos(2 * jnp.pi * nfp * mc * t) + rs @ jnp.sin(2 * jnp.pi * nfp * ms * t)

def sdf_helical_vessel(pts, params, sign, nfp, ntor, stellsym, r_order, n_grid=N_GRID, tol=FOOT_TOL, maxiter=100, return_residual=False):
    """Signed distance from each point in `pts` to the variable-radius tube.

    The tube is the union of balls B(C(t), R(t)), so the exact SDF is
        Phi(p) = min_t ( ||p - C(t*)|| - R(t*) ),
    where t* is the "foot point" that minimizes the distance-to-ball over the
    centerline. There is no closed form for t*, so for each query point we
    (1) seed the global basin with a coarse scan + bracket, (2) polish phi'(t*)=0
    with bisection-guarded Newton ("rtsafe") run to |phi'| < tol, and (3) detach
    the solve so autodiff still returns the exact gradient (envelope theorem:
    phi'(t*) = 0 kills the dt*/d(params) term).

    Args:
        pts:      (M, 3) query points.
        params:   flat DOF vector (see _unpack): centerline + radius harmonics.
        sign:     +1 for outside-positive; -1 flips the inside/outside convention.
        nfp:      field-period number (multiplies the basis frequencies).
        ntor:     toroidal winding number (rotates xhat/yhat into x/y).
        n_grid:   coarse-scan resolution; use >~ 4*max(nfp*N, ntor) so the scan
                  brackets the correct global basin before the polish refines it.
        tol:      stop each foot-point solve once the residual |phi'(t)| < tol.
        maxiter:  hard cap on guarded-Newton iterations (degenerate-point safety).
    Returns:
        (M,) array of signed distances.
    """
    xc, xs, yc, ys, zc, zs, rc, rs = _unpack(params, stellsym, r_order)  # harmonics + radius

    def one(p):                                        # --- SDF for one point p ---
        # phi(t): signed distance from p to the ball at parameter t (||p - C(t)|| - R(t));
        # the tube SDF is min_t phi (union of balls).
        phi = lambda t: (jnp.linalg.norm(p - _C(t, xc, xs, yc, ys, zc, zs, nfp, ntor))
                         - _R(t, rc, rs, nfp))
        g, h = jax.grad(phi), jax.grad(jax.grad(phi))  # phi'(t), phi''(t) by autodiff

        # (1) GLOBAL SEARCH + BRACKET: scan phi over one period [0,1), pick the
        #     smallest sample (correct basin even if the curve wiggles), and bracket
        #     the foot point with the two neighbouring grid points (phi is 1-periodic,
        #     so endpoints just outside [0,1) are fine). Orient [xl, xh] so that
        #     phi'(xl) <= 0 <= phi'(xh): the root of phi' between them is the MINIMUM.
        grid = jnp.linspace(0.0, 1.0, n_grid, endpoint=False)
        t0 = grid[jnp.argmin(jax.vmap(phi)(grid))]
        dt = 1.0 / n_grid
        a, b = t0 - dt, t0 + dt
        xl = jnp.where(g(a) <= 0.0, a, b)
        xh = jnp.where(g(a) <= 0.0, b, a)

        # (2) LOCAL POLISH: bisection-guarded Newton ("rtsafe"), iterated as a
        #     while_loop UNTIL the residual |phi'(t)| < tol (or maxiter). A Newton
        #     step on phi'(t)=0 is taken only if it stays inside [xl,xh] and shrinks
        #     the bracket by >= half; otherwise it bisects. The bracket
        #     (phi'(xl) <= 0 <= phi'(xh)) is kept by the sign of phi' at each new
        #     iterate. The bisection branch never divides by phi'', so the solve is
        #     robust (degrading to linear) as phi'' -> 0 near R*kappa = 1, where plain
        #     Newton's step blows up.
        def _cond(st):
            gt, it = st[1], st[7]
            return (jnp.abs(gt) > tol) & (it < maxiter)

        def _step(st):
            t, gt, ht, xl, xh, dx, dxold, it = st
            out = ((t - xh) * ht - gt) * ((t - xl) * ht - gt) > 0.0   # Newton leaves bracket?
            slow = jnp.abs(2.0 * gt) > jnp.abs(dxold * ht)            # step not shrinking >= half?
            bisect = out | slow | (jnp.abs(ht) < 1e-14)               # or phi'' ~ 0
            dx_bis = 0.5 * (xh - xl)
            ht_safe = jnp.where(jnp.abs(ht) > 1e-300, ht, 1e-300)     # guard the unused-branch divide
            dx_newt = gt / ht_safe
            dx_step = jnp.where(bisect, dx_bis, dx_newt)
            t_prop = jnp.where(bisect, xl + dx_bis, t - dx_newt)
            done = jnp.abs(gt) <= tol                                 # freeze converged lanes (vmap)
            t_new = jnp.where(done, t, t_prop)
            dx_new = jnp.where(done, dx, dx_step)
            g_new, h_new = g(t_new), h(t_new)
            xl_new = jnp.where((g_new < 0.0) & (~done), t_new, xl)    # keep phi'(xl) <= 0 <= phi'(xh)
            xh_new = jnp.where((g_new >= 0.0) & (~done), t_new, xh)
            return (t_new, g_new, h_new, xl_new, xh_new, dx_new, dx, it + 1)

        dx0 = jnp.abs(xh - xl)
        st = lax.while_loop(_cond, _step, (t0, g(t0), h(t0), xl, xh, dx0, dx0, jnp.array(0)))
        t, resid = st[0], st[1]                         # foot point and its residual phi'(t*)

        # (3) DETACH the foot point: by the envelope theorem phi'(t*)=0, so freezing
        #     t* loses nothing and grad(Phi) = phi_q(t*) is exact (the implicit
        #     params->t* path contributes zero). stop_gradient also makes the
        #     non-reverse-differentiable while_loop / argmin invisible to autodiff --
        #     the backward pass only differentiates phi at the frozen t*, giving the
        #     exact outward normal (grad_p) and shape gradient (grad_params).
        t = lax.stop_gradient(t)
        sdf_val = sign * phi(t)
        return (sdf_val, resid) if return_residual else sdf_val

    # vectorize over all M points. With return_residual, also returns the (M,) array
    # of foot-point residuals |phi'(t*)| (for a convergence / line-search check).
    return jax.vmap(one)(pts)

def kappa_helical_vessel(t, params, nfp, ntor, stellsym, r_order):
    """Curvature kappa(t) of the centerline at parameter(s) t in [0,1).

    Accepts a scalar or a 1-D array of parameters and returns kappa with the
    same shape, using kappa = ||C' x C''|| / ||C'||^3 (derivatives by autodiff).
    """
    xc, xs, yc, ys, zc, zs, _rc, _rs = _unpack(params, stellsym, r_order)
    C = lambda s: _C(s, xc, xs, yc, ys, zc, zs, nfp, ntor)
    def kap(s):
        Cp, Cpp = jax.jacfwd(C)(s), jax.jacfwd(jax.jacfwd(C))(s)
        return jnp.linalg.norm(jnp.cross(Cp, Cpp)) / jnp.linalg.norm(Cp) ** 3
    t = jnp.asarray(t)
    return kap(t) if t.ndim == 0 else jax.vmap(kap)(t)

def radius_helical_vessel(t, params, nfp, stellsym, r_order):
    """Tube radius R(t) at parameter(s) t in [0,1) (scalar or 1-D array)."""
    *_, rc, rs = _unpack(params, stellsym, r_order)
    R = lambda s: _R(s, rc, rs, nfp)
    t = jnp.asarray(t)
    return R(t) if t.ndim == 0 else jax.vmap(R)(t)

def max_dr_ds_helical_vessel(params, nfp, ntor, stellsym, r_order, n_grid=N_GRID):
    """max_t |dR/ds| = max_t |R'(t)| / ||C'(t)|| over a uniform grid: the radius-
    slope regime metric (must stay < 1). Identically 0 for a constant radius."""
    xc, xs, yc, ys, zc, zs, rc, rs = _unpack(params, stellsym, r_order)
    C = lambda s: _C(s, xc, xs, yc, ys, zc, zs, nfp, ntor)
    R = lambda s: _R(s, rc, rs, nfp)
    # |dR/ds| is 1/nfp-periodic (see _geometric_penalty), so one field period suffices.
    tg = jnp.linspace(0.0, 1.0 / nfp, n_grid, endpoint=False)
    slope = lambda s: jnp.abs(jax.grad(R)(s)) / jnp.linalg.norm(jax.jacfwd(C)(s))
    return jnp.max(jax.vmap(slope)(tg))

def arclength_variation_helical_vessel(params, nfp, ntor, stellsym, r_order, n_grid=N_GRID):
    """Squared coefficient of variation of the centerline speed ||C'(t)||:
    mean((|C'| - <|C'|>)^2) / <|C'|>^2. Zero iff the centerline is parametrized
    at constant arclength; used as a penalty to keep the toroidal parameter
    tracking arclength (a uniform parametrization)."""
    xc, xs, yc, ys, zc, zs, _rc, _rs = _unpack(params, stellsym, r_order)
    C = lambda s: _C(s, xc, xs, yc, ys, zc, zs, nfp, ntor)
    # ||C'|| is 1/nfp-periodic (see _geometric_penalty), so the speed distribution
    # over one field period [0, 1/nfp) has the same mean and variance as over [0,1).
    tg = jnp.linspace(0.0, 1.0 / nfp, n_grid, endpoint=False)
    speeds = jax.vmap(lambda s: jnp.linalg.norm(jax.jacfwd(C)(s)))(tg)
    mean = jnp.mean(speeds)
    return jnp.mean((speeds - mean) ** 2) / mean ** 2

def _geometric_penalty(params, nfp, ntor, stellsym, r_order, n_grid=N_GRID):
    """Geometrical penalty on the tube, sampled on a uniform grid, combining:
      (1) the valid-tube regime  max_t R(t)*kappa(t) < 1  (no self-intersection
          across a bend) and  max_t |dR/ds| < 1  (radius grows slower than
          arclength; no swallowing along the axis), each as max(excess,0)^2;
      (2) radius positivity  R(t) > 0  (a negative radius is meaningless), as a
          mean of squared hinges over the grid; and
      (3) a constant-arclength penalty on the centerline: the squared coefficient
          of variation of the speed ||C'(t)||, which drives the parametrization
          toward uniform arclength.
    All terms are dimensionless and enter the vessel penalty additively."""
    xc, xs, yc, ys, zc, zs, rc, rs = _unpack(params, stellsym, r_order)
    C = lambda s: _C(s, xc, xs, yc, ys, zc, zs, nfp, ntor)
    R = lambda s: _R(s, rc, rs, nfp)
    # All these quantities are intrinsic (curvature, speed, R, dR/ds) and the
    # centerline profile uses only nfp*m frequencies, so they are exactly
    # 1/nfp-periodic in t (the ntor rotation is rigid and norm/cross-product
    # preserving). Sampling one field period [0, 1/nfp) therefore captures every
    # value -- nfp x finer resolution than [0,1) for the same n_grid.
    tg = jnp.linspace(0.0, 1.0 / nfp, n_grid, endpoint=False)
    def per_t(s):
        Cp, Cpp = jax.jacfwd(C)(s), jax.jacfwd(jax.jacfwd(C))(s)
        speed = jnp.linalg.norm(Cp)
        kap = jnp.linalg.norm(jnp.cross(Cp, Cpp)) / speed ** 3
        slope = jnp.abs(jax.grad(R)(s)) / speed       # |dR/ds|
        return speed, R(s) * kap, slope, R(s)
    speeds, Rkap, slopes, Rs = jax.vmap(per_t)(tg)
    curv = jnp.maximum(jnp.max(Rkap) - 1.0, 0.0) ** 2
    slp = jnp.maximum(jnp.max(slopes) - 1.0, 0.0) ** 2
    pos = jnp.mean(jnp.maximum(-Rs, 0.0) ** 2)             # radius must stay positive
    mean = jnp.mean(speeds)
    arclen = jnp.mean((speeds - mean) ** 2) / mean ** 2     # constant-arclength penalty
    return curv + slp + pos + arclen

def quadratic_threshold_helical_vessel(pts, params, sign, threshold, nfp, ntor, stellsym, r_order):
    sls = sdf_helical_vessel(pts, params, sign, nfp, ntor, stellsym, r_order)
    reach = _geometric_penalty(params, nfp, ntor, stellsym, r_order)   # regime + arclength
    return jnp.mean(jnp.maximum(threshold - sls, 0) ** 2) + reach

def quadratic_distance_helical_vessel(pts, params, sign, threshold, nfp, ntor, stellsym, r_order):
    sls = sdf_helical_vessel(pts, params, sign, nfp, ntor, stellsym, r_order)
    dvalue = jnp.abs(sls - jnp.mean(sls))
    return jnp.mean(jnp.maximum(dvalue - threshold, 0) ** 2)

class HelicalVesselSDF(Optimizable):
    """Tube about a closed CurveXYZFourierSymmetries centerline (order N =
    len(xc)-1), with a radius R(t) that is itself a Fourier series of order
    Mr = len(rc)-1 (Mr=0 -> constant radius rc[0]).  xc, ys, zs, rc are the
    stell-sym harmonics; xs, yc, zc, rs are the symmetry-breaking ones.

    With ``stellsym=True`` the symmetry-breaking harmonics (xs, yc, zc, rs) are
    NOT part of the DOF vector at all -- the centerline AND radius profile are
    exactly stellarator symmetric. With ``stellsym=False`` all families are
    present and free; xs/yc/zc/rs default to zero if not supplied."""
    def __init__(self, xc, ys, zs, rc, nfp, ntor=1, xs=None, yc=None, zc=None,
                 rs=None, stellsym=True, **kwargs):
        xc, ys, zs, rc = (np.asarray(a, float) for a in (xc, ys, zs, rc))
        N, Mr = len(xc) - 1, len(rc) - 1
        assert len(ys) == len(zs) == N, "sine arrays length N (m=1..N)"
        self.stellsym = bool(stellsym)
        self._N, self._Mr = N, Mr
        if self.stellsym:
            # No symmetry-breaking harmonics: layout is xc, ys, zs, rc.
            x0 = np.concatenate([xc, ys, zs, rc])
            parts = [('xc', xc, 0), ('ys', ys, 1), ('zs', zs, 1), ('rc', rc, 0)]
        else:
            xs = np.zeros(N)     if xs is None else np.asarray(xs, float)
            yc = np.zeros(N + 1) if yc is None else np.asarray(yc, float)
            zc = np.zeros(N + 1) if zc is None else np.asarray(zc, float)
            rs = np.zeros(Mr)    if rs is None else np.asarray(rs, float)
            assert len(xs) == N, "sine arrays length N (m=1..N)"
            assert len(yc) == len(zc) == N + 1, "cosine arrays length N+1 (m=0..N)"
            assert len(rs) == Mr, "radius sine array length Mr (m=1..Mr)"
            x0 = np.concatenate([xc, xs, yc, ys, zc, zs, rc, rs])
            parts = [('xc', xc, 0), ('xs', xs, 1), ('yc', yc, 0),
                     ('ys', ys, 1), ('zc', zc, 0), ('zs', zs, 1),
                     ('rc', rc, 0), ('rs', rs, 1)]
        names = [f'{p}({m})' for p, a, m0 in parts
                 for m in range(m0, m0 + len(a))]
        super().__init__(depends_on=[], x0=x0, names=names, **kwargs)
        if not self.stellsym:
            # zc(0) is a rigid vertical translation of the centerline (z is not rotated
            # by ntor), and it is a gauge freedom of the penalties -- the geometry terms
            # (kappa, dR/ds, arclength) are translation-invariant and the distance terms
            # are one-sided -- so the vessel could otherwise drift up/down. Pin it. (For
            # stellsym the zc family is absent entirely, so there is nothing to fix.)
            self.fix('zc(0)')
        self.nfp, self.ntor = int(nfp), int(ntor)
        _sj = dict(nfp=self.nfp, ntor=self.ntor, stellsym=self.stellsym, r_order=self._Mr)
        # JIT the EAGER SDF paths used by the diagnostics (shortest/longest_distance,
        # eval, to_vtk, foot_residual) and the per-grid metric helpers. Run eagerly,
        # the while_loop/vmap foot-point solve dispatches op-by-op and is ~1000x slower
        # than the compiled kernel (it made the optimizer callback take seconds; jitted
        # it is milliseconds). Static args (nfp/ntor/stellsym/r_order/n_grid) are bound
        # here, so only `params` (the dofs) varies -> each compiles once and is reused.
        # quadratic_threshold/_distance stay plain partials -- VesselDistance jits them
        # itself (self.J_jax), so they are already fast there.
        self.pure = jax.jit(partial(sdf_helical_vessel, **_sj))
        self._resid_fn = jax.jit(partial(sdf_helical_vessel, return_residual=True, **_sj))
        self.quadratic_threshold = partial(quadratic_threshold_helical_vessel, **_sj)
        self.quadratic_distance = partial(quadratic_distance_helical_vessel, **_sj)
        self._kappa_fn = jax.jit(partial(kappa_helical_vessel, nfp=self.nfp, ntor=self.ntor,
                                         stellsym=self.stellsym, r_order=self._Mr))
        self._radius_fn = jax.jit(partial(radius_helical_vessel, nfp=self.nfp,
                                          stellsym=self.stellsym, r_order=self._Mr))
        self._max_dr_ds_fn = jax.jit(partial(max_dr_ds_helical_vessel, n_grid=N_GRID, **_sj))
        self._arclen_fn = jax.jit(partial(arclength_variation_helical_vessel, n_grid=N_GRID, **_sj))

    @classmethod
    def from_curve_xyz_fourier_symmetries(cls, curve, rr, stellsym=None, num_modes=4,
                                          radius_num_modes=0, **kwargs):
        """Build directly from a simsopt CurveXYZFourierSymmetries, copying its
        harmonics, nfp, and ntor. ``stellsym`` defaults to the curve's own flag:
        a stellarator-symmetric curve yields a stellsym vessel (xs/yc/zc absent),
        a non-symmetric one yields a full vessel seeded with the curve's xs/yc/zc.

        ``num_modes`` TRUNCATES the centerline Fourier order: only modes
        m = 0..num_modes are kept (cosine families) and m = 1..num_modes (sine),
        so the resulting centerline has order min(num_modes, curve.order). The
        higher harmonics of the (typically order-16) axis are dropped, giving a
        smaller, smoother vessel dof set.

        ``radius_num_modes`` (Mr) sets the radius Fourier order: 0 -> constant
        radius rr; >0 -> the radius varies with the toroidal parameter (rc[0]=rr,
        higher radius modes start at 0 and are free design variables)."""
        # name -> value over ALL dofs (free and fixed); strip any 'ObjName:' prefix
        d = {n.split(':')[-1]: v for n, v in zip(curve.full_dof_names, curve.full_x)}
        nfp = curve.nfp
        N = min(int(num_modes), curve.order)   # truncated Fourier order
        Mr = int(radius_num_modes)
        ntor = getattr(curve, 'ntor', 1)
        if stellsym is None:
            stellsym = curve.stellsym
        get = lambda p, m: d.get(f'{p}({m})', 0.0)
        cos = lambda p: [get(p, m) for m in range(N + 1)]     # m = 0..N
        sin = lambda p: [get(p, m) for m in range(1, N + 1)]  # m = 1..N
        rc = [rr] + [0.0] * Mr   # constant radius rr in mode 0; higher modes free, start at 0
        if stellsym:
            return cls(cos('xc'), sin('ys'), sin('zs'), rc, nfp, ntor=ntor,
                       stellsym=True, **kwargs)
        return cls(cos('xc'), sin('ys'), sin('zs'), rc, nfp, ntor=ntor,
                   xs=sin('xs'), yc=cos('yc'), zc=cos('zc'), rs=[0.0] * Mr,
                   stellsym=False, **kwargs)

    # --- live harmonic views (read straight off the current dof vector) so the
    # default simsopt GSON serialization can reconstruct via __init__. xs/yc/zc
    # are None in stellsym mode (those args are then left at their defaults). ---
    def _slice(self, lo, hi):
        return [float(v) for v in self.local_full_x[lo:hi]]

    @property
    def xc(self):
        return self._slice(0, self._N + 1)

    @property
    def ys(self):
        N = self._N
        return self._slice(N + 1, 2 * N + 1) if self.stellsym else self._slice(3 * N + 2, 4 * N + 2)

    @property
    def zs(self):
        N = self._N
        return self._slice(2 * N + 1, 3 * N + 1) if self.stellsym else self._slice(5 * N + 3, 6 * N + 3)

    @property
    def xs(self):
        N = self._N
        return None if self.stellsym else self._slice(N + 1, 2 * N + 1)

    @property
    def yc(self):
        N = self._N
        return None if self.stellsym else self._slice(2 * N + 1, 3 * N + 2)

    @property
    def zc(self):
        N = self._N
        return None if self.stellsym else self._slice(4 * N + 2, 5 * N + 3)

    @property
    def _rad_base(self):
        # index where the radius block begins (centerline dofs precede it)
        N = self._N
        return 3 * N + 1 if self.stellsym else 6 * N + 3

    @property
    def rc(self):
        b = self._rad_base
        return self._slice(b, b + self._Mr + 1)

    @property
    def rs(self):
        if self.stellsym:
            return None
        b = self._rad_base + self._Mr + 1
        return self._slice(b, b + self._Mr)

    def num_dofs(self):
        return len(self.local_full_x)

    def kappa(self, t=None, n_grid=N_GRID):
        """Centerline curvature kappa.

        The valid-tube regime requires R(t)*kappa(t) < 1 everywhere; sample this
        to find where it is violated, e.g. ``t[vessel.kappa()*vessel.radius() >= 1]``.

        Args:
            t: parameter(s) in [0,1) at which to evaluate.  If None, a uniform
               grid of ``n_grid`` points is used.
        Returns:
            kappa(t).
        """
        if t is None:
            t = np.linspace(0.0, 1.0, n_grid, endpoint=False)
        return np.asarray(self._kappa_fn(np.asarray(t, float), self.local_full_x))

    def radius(self, t=None, n_grid=N_GRID):
        """Tube radius R(t) (constant when radius order Mr == 0)."""
        if t is None:
            t = np.linspace(0.0, 1.0, n_grid, endpoint=False)
        return np.asarray(self._radius_fn(np.asarray(t, float), self.local_full_x))

    def max_kappa_radius(self, n_grid=N_GRID):
        """max_t R(t)*kappa(t): the curvature regime metric (must stay < 1).

        R*kappa is 1/nfp-periodic (intrinsic + nfp-frequency profile), so sampling
        one field period [0, 1/nfp) captures the maximum at nfp x finer resolution."""
        t = np.linspace(0.0, 1.0 / self.nfp, n_grid, endpoint=False)
        x = self.local_full_x
        return float(np.max(np.asarray(self._kappa_fn(t, x)) * np.asarray(self._radius_fn(t, x))))

    def min_radius(self, n_grid=N_GRID):
        """min_t R(t): smallest tube radius over one field period (R is 1/nfp-periodic).
        Must stay > 0; used (eager) by the drivers' geometric-constraint escalation."""
        t = np.linspace(0.0, 1.0 / self.nfp, n_grid, endpoint=False)
        return float(np.min(self.radius(t)))

    def max_dr_ds(self):
        """max_t |dR/ds| = max_t |R'(t)|/||C'(t)||: the radius-slope regime metric
        (must stay < 1; 0 for a constant radius). Sampled on N_GRID points."""
        return float(self._max_dr_ds_fn(self.local_full_x))

    def arclength_variation(self):
        """Squared coefficient of variation of the centerline speed ||C'(t)||;
        0 iff the centerline is parametrized at constant arclength. N_GRID points."""
        return float(self._arclen_fn(self.local_full_x))

    def foot_residual(self, pts):
        """max_p |phi'(t*_p)| over the query points `pts`. The foot-point solve drives
        this below its tolerance, so a value well above tol means the solve hit maxiter
        without converging -- the tube is degenerate/invalid (e.g. R*kappa >= 1) and the
        SDF is unreliable there. Use it to detect a bad configuration and trigger an
        optimizer line-search retreat. (Uses the jitted solve.)"""
        _, resids = self._resid_fn(np.asarray(pts), self.local_full_x, 1.0)
        return float(np.max(np.abs(resids)))

    @property
    def rr(self):
        # mean tube radius = rc[0]
        return float(self.local_full_x[self._rad_base])

    def eval(self, x, y, z):
        pts = np.concatenate((x.flatten()[:, None], y.flatten()[:, None],
                              z.flatten()[:, None]), axis=-1)
        return np.array(self.pure(pts, self.local_full_x, 1.0).astype(np.float32).reshape(x.shape))

    def rz_bounds(self):
        """Conservative (R_max, |Z|_max): rotation is norm-preserving so
        R = hypot(xhat, yhat); per-component mode sums bound the rest; the tube
        adds the largest possible radius (sum of |radius harmonics|)."""
        xc, xs, yc, ys, zc, zs, rc, rs = _unpack(self.local_full_x, self.stellsym, self._Mr)
        xb = np.sum(np.abs(xc)) + np.sum(np.abs(xs))
        yb = np.sum(np.abs(yc)) + np.sum(np.abs(ys))
        zb = np.sum(np.abs(zc)) + np.sum(np.abs(zs))
        Rb = float(np.sum(np.abs(rc)) + np.sum(np.abs(rs)))
        return float(np.hypot(xb, yb) + Rb), float(zb + Rb)

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
