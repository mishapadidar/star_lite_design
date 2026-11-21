"""Extra coils of the designB workflow (option --extra-coils of boozer_all.py).

  circ  4 circular coils (CircleCurve), dofs [r, X, Y, Z, a, b] each: radius, centre, and the
        orientation of the plane (rotation vector a e1 + b e2 applied to the initial frame).
        The circumference 2 pi r is bounded (CIRC_MAX, 4.0 m) by a penalty in the optimization
        scripts. (Not simsopt's CurvePlanarFourier: in this simsopt its derivatives ignore the
        normalization of its quaternion, so its gradients are wrong by ~2% as soon as the
        quaternion is moved; checked against finite differences on 2026-11-18.)
  T     design A's 2 T coils (coil 4 of designs/designA/designA_after_scaled.json, in the YZ
        plane, and its copy): SHAPE frozen, only a rigid motion is free (RigidCurve), dofs
        [tx, ty, tz, wx, wy, wz] (translation, rotation vector about the coil centroid).
  LL    design A's 4 L coils (its modular coils 0-3): shape frozen, rigid motion (RigidCurve).
  LT    per half period, one L coil (design A's coil 0, phi = 34 deg) then one T coil (phi = 135
        deg, as in T): 2 L + 2 T coils on the torus.
  TL    per half period, one T coil (phi = 45 deg: design A's T turned by -45 deg) then one L
        coil (design A's coil 3, phi = 146 deg): the mirror image of LT (the single-null device
        is not stellarator symmetric, so LT and TL differ).

The base coils (circ: 2, T: 1, LL, LT, TL: 2) are expanded by coils_via_symmetries(nfp=2, stellsym=False),
each with its copy rotated by pi about z, so the set keeps the nfp=2 symmetry of the single-null
Boozer surfaces; each base has its own current, starting at ~0 (EXTRA_I0 = 1 nA), so the initial field (and the
initial designB surface) is unchanged. The set is added to the device by
sn_setup.setup_single_null(..., extra_coils=coils): bases at indices 4, 5, ..., then the copies.

Initial placement (scans of 2026-11-18). Every coil encircles the plasma (linking number 1 with
the magnetic axis) and none is interlocked with a modular coil. No placement that encircles the
plasma is 15 cm from the designB coils: they cover every toroidal angle.
  circ  vertical, in the planes phi = 45, 135, 225, 315 deg, centre at R0 = 0.60 m, z = 0,
        r = 0.37 m (circumference 2.32 m): 8.6 cm from the modular coils, 12.9 cm from the
        surface, 10.2 cm outside the vessel, 23 cm from the z axis;
  T     design A's position rotated by 45 deg about z (phi = 135, 315 deg): 2.2 cm from the
        modular coils, 5.5 cm from the surface, 5.0 cm inside the vessel (at design A's own
        position, phi = 90 deg: 3.8 cm, 13 cm, 2.3 cm inside); the penalties push them out;
  LL    design A's own position (phi = +-34, +-146 deg): 3.7 cm from the modular coils, 18.9 cm
        from the surface, 10.7 cm outside the vessel (rotated by 45 or 90 deg they would be
        interlocked with designB's coils);
  LT,TL each coil where it is in LL or T (the T of TL at phi = 45 deg instead of 135).
"""
import functools
import os

import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo.curve import JaxCurve

EXTRA_KINDS = ('none', 'circ', 'T', 'LL', 'LT', 'TL')
CIRC_MAX = 4.0                      # bound on the circumference of a circular coil (m)
CIRC_PHIS = (45.0, 135.0)           # base circles: vertical, in these planes phi (deg); copies at +180
CIRC_R0 = 0.60                      # their centre: major radius (m), at z = 0
CIRC_RADIUS = 0.37                  # their radius (m)
DESIGN_A_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'designs', 'designA',
                             'designA_after_scaled.json')
T_COIL_INDEX = 4                    # design A: coil 4 at phi = +90 deg, coil 5 its pi-rotated copy
T_PHI_SHIFT = 45.0                  # initial rotation of the T coils about z (deg)
L_COIL_INDICES = (0, 1)             # design A: L coil 0 (phi = 34 deg), its stellarator image 1
                                    # (phi = -34 deg); 2 and 3 are their pi-rotated copies
L_PHI_SHIFT = 0.0                   # initial rotation of the L coils about z (deg)
# LT: L coil 0 (phi = 34 deg) + the T turned by +45 deg (phi = 135 deg); TL: the T turned by
# -45 deg (phi = 45 deg) + L coil 3 (phi = 146 deg, the pi-rotated copy of coil 1)
LT_BASES = (('L', 0, 0.0), ('T', T_COIL_INDEX, 45.0))
TL_BASES = (('T', T_COIL_INDEX, -45.0), ('L', 3, 0.0))
# Initial current of every extra base coil (A): 1 nA rather than exactly 0, because the gradient
# of simsopt's LpCurveForce is NaN for a target coil whose current is exactly 0 (the norm of a zero
# force; checked on 2026-11-18: finite for any current from 1 mA down to 1 pA), and the term is in
# the objective even with the force bound off (weight 0, and 0 x NaN = NaN). Against the 55 kA of
# the modular coils, 1 nA changes the field by ~2e-14 relative, the level of the Newton tolerance
# of the surfaces: the initial designB surface is unchanged.
EXTRA_I0 = 1e-9


def _rotation_matrix(w):
    """Rotation matrix of the rotation vector w (Rodrigues), smooth (and NaN-free) at w = 0."""
    th2 = jnp.dot(w, w)
    small = th2 < 1e-8
    th2s = jnp.where(small, 1.0, th2)
    ths = jnp.sqrt(th2s)
    a = jnp.where(small, 1.0 - th2 / 6.0, jnp.sin(ths) / ths)
    b = jnp.where(small, 0.5 - th2 / 24.0, (1.0 - jnp.cos(ths)) / th2s)
    K = jnp.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    return jnp.eye(3) + a * K + b * (K @ K)


def _fourier(cos_c, sin_c, points):
    """sum_m cos_c[m] cos(2 pi m t) + sin_c[m] sin(2 pi m t), shape (len(points), 3)."""
    ang = 2 * jnp.pi * jnp.outer(points, jnp.arange(cos_c.shape[0]))
    return jnp.cos(ang) @ cos_c + jnp.sin(ang) @ sin_c


class RigidCurve(JaxCurve):
    """A curve of FROZEN shape (an xyz Fourier series, constructor arguments, not dofs) moved
    rigidly: gamma = R(w) (gamma0 - c) + c + t, dofs [tx, ty, tz, wx, wy, wz], with c the
    centroid of gamma0 (its m = 0 coefficient) and R(w) the rotation of rotation vector w."""

    def __init__(self, quadpoints, cos_coeffs, sin_coeffs, dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        self.cos_coeffs = np.asarray(cos_coeffs, dtype=float)
        self.sin_coeffs = np.asarray(sin_coeffs, dtype=float)
        cc, sc = jnp.asarray(self.cos_coeffs), jnp.asarray(self.sin_coeffs)
        centroid = jnp.asarray(self.cos_coeffs[0])

        def pure(d, points):
            g0 = _fourier(cc, sc, points)
            return (g0 - centroid) @ _rotation_matrix(d[3:6]).T + centroid + d[0:3]

        self.dof_list = np.zeros(6)
        names = ['tx', 'ty', 'tz', 'wx', 'wy', 'wz']
        if dofs is None:
            super().__init__(quadpoints, pure, x0=self.dof_list, names=names,
                             external_dof_setter=RigidCurve.set_dofs_impl)
        else:
            super().__init__(quadpoints, pure, dofs=dofs, names=names,
                             external_dof_setter=RigidCurve.set_dofs_impl)

    def num_dofs(self):
        return 6

    def get_dofs(self):
        return np.array(self.dof_list)

    def set_dofs_impl(self, dofs):
        self.dof_list = np.array(dofs)


def _frame(normal):
    """Orthonormal (e1, e2, n) with n along normal, e1 horizontal when possible."""
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    e1 = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(e1) < 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)
    return e1, np.cross(n, e1), n


class CircleCurve(JaxCurve):
    """Circle: gamma = C + r (cos t u1 + sin t u2), (u1, u2) = R(a e1 + b e2) (e1, e2), with
    (e1, e2, normal0) the initial frame (a constructor argument, not a dof). Dofs
    [r, X, Y, Z, a, b]: radius, centre, orientation (normal = R(a e1 + b e2) normal0: every
    direction is reachable, no redundant dof, smooth at a = b = 0)."""

    def __init__(self, quadpoints, normal0=(1.0, 0.0, 0.0), dofs=None):
        if isinstance(quadpoints, int):
            quadpoints = np.linspace(0, 1, quadpoints, endpoint=False)
        self.normal0 = np.asarray(normal0, dtype=float)
        e1, e2, _ = _frame(self.normal0)
        E1, E2 = jnp.asarray(e1), jnp.asarray(e2)

        def pure(d, points):
            R = _rotation_matrix(d[4] * E1 + d[5] * E2)
            t = 2 * jnp.pi * points
            return d[1:4] + d[0] * (jnp.outer(jnp.cos(t), R @ E1) + jnp.outer(jnp.sin(t), R @ E2))

        self.dof_list = np.zeros(6)
        names = ['r', 'X', 'Y', 'Z', 'a', 'b']
        if dofs is None:
            super().__init__(quadpoints, pure, x0=self.dof_list, names=names,
                             external_dof_setter=CircleCurve.set_dofs_impl)
        else:
            super().__init__(quadpoints, pure, dofs=dofs, names=names,
                             external_dof_setter=CircleCurve.set_dofs_impl)

    def num_dofs(self):
        return 6

    def get_dofs(self):
        return np.array(self.dof_list)

    def set_dofs_impl(self, dofs):
        self.dof_list = np.array(dofs)

    def normal(self):
        e1, e2, n0 = _frame(self.normal0)
        d = self.get_dofs()
        return np.asarray(_rotation_matrix(jnp.asarray(d[4] * e1 + d[5] * e2))) @ n0


@functools.lru_cache(maxsize=None)
def _design_a_coils(path=DESIGN_A_JSON):
    from simsopt._core import load
    return load(path)[0][0].biotsavart.coils


def design_a_coeffs(index, order=None, path=DESIGN_A_JSON):
    """Fourier coefficients (cos, sin: (order+1, 3) each) of design A's coil `index`, from its
    gamma on its (uniform) quadrature points; order defaults to that of its underlying curve.
    Returns (cos, sin, gamma)."""
    curve = _design_a_coils(path)[index].curve
    qp = np.asarray(curve.quadpoints)
    n = len(qp)
    if not np.allclose(qp, np.arange(n) / n):
        raise RuntimeError(f"design A's coil {index} does not have uniform quadrature points")
    if order is None:
        order = getattr(getattr(curve, 'curve', curve), 'order', None) or (n - 1) // 2
    F = np.fft.rfft(curve.gamma(), axis=0) / n
    cos_c = np.zeros((order + 1, 3))
    sin_c = np.zeros((order + 1, 3))
    cos_c[0] = F[0].real
    cos_c[1:] = 2 * F[1:order + 1].real
    sin_c[1:] = -2 * F[1:order + 1].imag
    return cos_c, sin_c, curve.gamma()


def _design_a_rigid(index, quadpoints, phi_shift):
    """RigidCurve of design A's coil `index`, rotated by phi_shift (deg) about the z axis: the
    rotation about z through the origin is the rotation vector (0, 0, phi) about the centroid c
    plus the translation Rz c - c."""
    cos_c, sin_c, _ = design_a_coeffs(index)
    c = RigidCurve(quadpoints, cos_c, sin_c)
    th = np.radians(phi_shift)
    Rz = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    cen = cos_c[0]
    c.x = np.concatenate([Rz @ cen - cen, [0.0, 0.0, th]])
    return c


def make_extra_curves(kind, quadpoints):
    """The base curves of the extra coils, at their initial placement."""
    if kind == 'circ':
        curves = []
        for phi in np.radians(CIRC_PHIS):
            c = CircleCurve(quadpoints, (-np.sin(phi), np.cos(phi), 0.0))   # normal: toroidal direction
            c.x = np.array([CIRC_RADIUS, CIRC_R0 * np.cos(phi), CIRC_R0 * np.sin(phi), 0.0, 0.0, 0.0])
            curves.append(c)
        return curves
    if kind == 'T':
        return [_design_a_rigid(T_COIL_INDEX, quadpoints, T_PHI_SHIFT)]
    if kind == 'LL':
        return [_design_a_rigid(i, quadpoints, L_PHI_SHIFT) for i in L_COIL_INDICES]
    if kind in ('LT', 'TL'):
        return [_design_a_rigid(i, quadpoints, shift) for _, i, shift in (LT_BASES if kind == 'LT' else TL_BASES)]
    raise ValueError(f"unknown extra coil kind {kind!r} (expected one of {EXTRA_KINDS[1:]})")


def make_extra_coils(kind, quadpoints, scale):
    """The extra coils: the base coils, each with its own current (EXTRA_I0 = 1 nA at start, see
    above; ScaledCurrent(Current(EXTRA_I0 / scale), scale) like the modular coils, so CurrentBound
    works on it), then their copies rotated by pi about z (coils_via_symmetries order: bases, then
    copies). Returns (coils, base_curves, base_currents)."""
    curves = make_extra_curves(kind, quadpoints)
    currents = [ScaledCurrent(Current(EXTRA_I0 / scale), scale) for _ in curves]
    return coils_via_symmetries(curves, currents, 2, False), curves, currents


def describe(curve):
    """One line on the geometry of an extra base curve (for the callback)."""
    if isinstance(curve, RigidCurve):
        d = curve.get_dofs()
        ang = np.degrees(np.linalg.norm(d[3:6]))
        return (f"rigid coil: translation ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f}) m, "
                f"rotation vector ({d[3]:+.4f}, {d[4]:+.4f}, {d[5]:+.4f}) rad ({ang:.2f} deg)")
    if isinstance(curve, CircleCurve):
        d = curve.get_dofs()
        r, ctr, n = d[0], d[1:4], curve.normal()
        return (f"circle: r = {r:.4f} m (circumference {2 * np.pi * abs(r):.3f} m, bound {CIRC_MAX} m), "
                f"centre ({ctr[0]:+.4f}, {ctr[1]:+.4f}, {ctr[2]:+.4f}), normal ({n[0]:+.3f}, {n[1]:+.3f}, {n[2]:+.3f})")
    return type(curve).__name__
