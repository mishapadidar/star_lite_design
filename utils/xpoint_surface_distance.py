import numpy as np
import jax.numpy as jnp
from jax import jit, grad

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.objectives import forward_backward


def xpoint_surface_dist_pure(gammafl, gammas, threshold, sign, p=2):
    """Pure JAX kernel for the X-point-to-surface distance inequality penalty.

    ``dists`` are the X-point-to-surface point distances. The penalty is

        J = mean( max(0, sign * (dists - threshold))^p ).

    With ``sign = +1`` it penalizes distances ABOVE ``threshold`` and is zero
    when every distance is below it (bounds the MAXIMUM distance, keeping the
    X-point within ``threshold`` of the surface). With ``sign = -1`` it
    penalizes distances BELOW ``threshold`` and is zero when every distance is
    above it (bounds the MINIMUM distance, keeping the X-point at least
    ``threshold`` from the surface).
    """
    dists = jnp.linalg.norm(gammafl[:, None, :] - gammas[None, :, :], axis=-1)
    return jnp.mean(jnp.maximum(sign * (dists - threshold), 0) ** p)


class XpointSurfaceDistance(Optimizable):
    r"""Inequality penalty on the distance between an X-point field line (a
    ``PeriodicFieldLine`` / ``SingularPeriodicFieldLine``) and a ``BoozerSurface``,
    in EITHER direction.

    ``kind='max'`` bounds the MAXIMUM distance: it keeps the X-point WITHIN
    ``threshold`` of the surface (penalizes distances greater than ``threshold``).
    ``kind='min'`` bounds the MINIMUM distance: it keeps the X-point at least
    ``threshold`` AWAY from the surface (penalizes distances less than
    ``threshold``).

    .. math::
        J = \frac{1}{N} \sum_{i,j} \max\!\big(0,\ s\,(\|\mathbf{r}_i^{fl}
            - \mathbf{s}_j\| - d)\big)^p, \quad s = +1\,(\text{max}),\ -1\,(\text{min})

    The gradient flows through BOTH solved objects: the X-point field-line solve
    and the Boozer-surface solve (adjoints via their ``res['PLU']`` /
    ``res['vjp']``). The X-point curve dofs and the surface dofs are slaved by
    their Newton solves, so the only free dofs reached are the coil dofs, exactly
    as in CurveBoozerSurfaceDistance and CurvesPeriodicFieldlineDistance.
    """

    def __init__(self, xpoint, boozer_surface, threshold, kind='max', p=2):
        if kind not in ('max', 'min'):
            raise ValueError("kind must be 'max' (bound the max distance) or "
                             "'min' (bound the min distance)")
        self.xpoint = xpoint
        self.boozer_surface = boozer_surface
        self.surface = boozer_surface.surface
        self.threshold = threshold
        self.kind = kind
        self.p = p
        sign = 1.0 if kind == 'max' else -1.0

        self.J_jax = jit(lambda gfl, gs: xpoint_surface_dist_pure(gfl, gs, threshold, sign, p))
        self.thisgrad0 = jit(lambda gfl, gs: grad(self.J_jax, argnums=0)(gfl, gs))   # d/d gammafl
        self.thisgrad1 = jit(lambda gfl, gs: grad(self.J_jax, argnums=1)(gfl, gs))   # d/d gammas
        super().__init__(depends_on=[xpoint, boozer_surface])

    def _ensure_solved(self):
        if self.xpoint.need_to_run_code:
            self.xpoint.run_code(self.xpoint.res['length'])
        if self.boozer_surface.need_to_run_code:
            r = self.boozer_surface.res
            self.boozer_surface.run_code(r['iota'], r['G'])

    def max_distance(self):
        """Maximum over the X-point of the nearest-surface distance (how far the
        X-point strays from the surface). Used for the kind='max' constraint."""
        from scipy.spatial.distance import cdist
        self._ensure_solved()
        gfl = self.xpoint.curve.gamma()
        gs = self.surface.gamma().reshape((-1, 3))
        return float(np.max(np.min(cdist(gfl, gs), axis=1)))

    def min_distance(self):
        """Closest approach of the X-point to the surface (min over all X-point /
        surface point pairs). Used for the kind='min' constraint."""
        from scipy.spatial.distance import cdist
        self._ensure_solved()
        gfl = self.xpoint.curve.gamma()
        gs = self.surface.gamma().reshape((-1, 3))
        return float(np.min(cdist(gfl, gs)))

    def J(self):
        self._ensure_solved()
        gfl = self.xpoint.curve.gamma()
        gs = self.surface.gamma().reshape((-1, 3))
        return float(self.J_jax(gfl, gs))

    @derivative_dec
    def dJ(self):
        self._ensure_solved()
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size

        gfl = self.xpoint.curve.gamma()
        gs = self.surface.gamma().reshape((-1, 3))

        dgfl = self.thisgrad0(gfl, gs)
        dgs = self.thisgrad1(gfl, gs).reshape((nphi, ntheta, -1))

        # ---- X-point field-line adjoint (curve dofs are slaved by the solve) ----
        pfl = self.xpoint
        res_fl = pfl.curve.dgamma_by_dcoeff_vjp(dgfl)
        res_fl = res_fl(pfl.curve)
        Pf, Lf, Uf = pfl.res['PLU']
        vjp_fl = pfl.res['vjp']
        rhs_fl = np.zeros(Lf.shape[0])
        dj_fl = res_fl.copy()
        rhs_fl[:dj_fl.size] = dj_fl
        adj_fl = forward_backward(Pf, Lf, Uf, rhs_fl)
        dJ_fl = vjp_fl(adj_fl, pfl.biotsavart, pfl)

        # ---- Boozer-surface adjoint (surface dofs are slaved by the solve) ----
        bs = self.boozer_surface
        iota = bs.res['iota']
        G = bs.res['G']
        res_s = self.surface.dgamma_by_dcoeff_vjp(dgs)
        Ps, Ls, Us = bs.res['PLU']
        vjp_s = bs.res['vjp']
        rhs_s = np.zeros(Ls.shape[0])
        dj_s = res_s.copy()
        rhs_s[:dj_s.size] = dj_s
        adj_s = forward_backward(Ps, Ls, Us, rhs_s)
        dJ_s = vjp_s(adj_s, bs, iota, G)
        
        # Derivative has no unary minus; use scalar multiply (both terms are
        # gradients w.r.t. the same coil dofs, so they add).
        return -1.0 * (dJ_fl + dJ_s)

    return_fn_map = {'J': J, 'dJ': dJ}
