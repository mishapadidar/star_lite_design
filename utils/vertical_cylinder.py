"""Infinite vertical cylinder (axis parallel to z) as a signed distance function: a keep-out zone
that the modular coils may not enter.

    sdf(x, y, z) = sqrt((x - x0)^2 + (y - y0)^2) - r          (positive outside the cylinder)

The radius r is fixed; the position (x0, y0) of the axis is free (two design variables, started
at the origin unless given). The class has the interface of the vessel SDFs of pillpipevessel.py
(pure, quadratic_threshold, quadratic_distance, local_full_x = (x0, y0, r)), so that the penalty
is the existing VesselDistance:

    cyl = VerticalCylinderSDF(0.125)                    # diameter 25 cm, axis at the origin
    J = VesselDistance(cyl, curves, [1.0] * len(curves), clearance)

which is mean_points max(clearance - sdf, 0)^2 summed over the curves, and depends on the curves
and on (x0, y0). shortest_distance()[0] is then the smallest sdf over the curves (negative when a
coil is inside the cylinder).
"""
import numpy as np
import jax.numpy as jnp
from simsopt._core import Optimizable

__all__ = ['VerticalCylinderSDF', 'sdf_vertical_cylinder']


def sdf_vertical_cylinder(pts, params, sign):
    """Signed distance from the points pts (n, 3) to the vertical cylinder of axis (x0, y0) and
    radius r, params = (x0, y0, r): positive outside, times sign. The 1e-300 keeps the gradient
    finite for a point exactly on the axis."""
    x0, y0, r = params[0], params[1], params[2]
    d = jnp.sqrt((pts[:, 0] - x0)**2 + (pts[:, 1] - y0)**2 + 1e-300)
    return sign * (d - r)


def quadratic_threshold_vertical_cylinder(pts, params, sign, threshold):
    """mean max(threshold - sdf, 0)^2: zero when every point is at least `threshold` outside."""
    sls = sdf_vertical_cylinder(pts, params, sign)
    return jnp.mean(jnp.maximum(threshold - sls, 0)**2)


def quadratic_distance_vertical_cylinder(pts, params, sign, threshold):
    """Spread of the distance about its mean (interface of the vessel SDFs; not used here)."""
    sls = sdf_vertical_cylinder(pts, params, sign)
    dvalue = jnp.abs(sls - jnp.mean(sls))
    return jnp.mean(jnp.maximum(dvalue - threshold, 0)**2)


class VerticalCylinderSDF(Optimizable):
    """Infinite vertical cylinder of radius r (fixed) whose axis (x0, y0) is free.

    Args:
        r: radius of the cylinder [m] (diameter 25 cm: r = 0.125).
        x0, y0: initial position of the axis [m] (default: the origin).
    """

    def __init__(self, r, x0=0.0, y0=0.0, **kwargs):
        super().__init__(depends_on=[], x0=np.array([float(x0), float(y0), float(r)]),
                         names=['x0', 'y0', 'r'], **kwargs)
        self.fix('r')
        self.pure = sdf_vertical_cylinder
        self.quadratic_threshold = quadratic_threshold_vertical_cylinder
        self.quadratic_distance = quadratic_distance_vertical_cylinder

    def center(self):
        """(x0, y0) of the axis."""
        return float(self.local_full_x[0]), float(self.local_full_x[1])

    def radius(self):
        return float(self.local_full_x[2])

    def eval(self, x, y, z):
        pts = np.concatenate((x.flatten()[:, None], y.flatten()[:, None], z.flatten()[:, None]), axis=-1)
        return np.array(self.pure(pts, self.local_full_x, 1.0)).reshape(x.shape)

    def to_vtk(self, name, zmin=-1.0, zmax=1.0, n_theta=64, n_z=2):
        """The cylinder surface between zmin and zmax, as a structured grid (name.vts)."""
        from pyevtk.hl import gridToVTK
        x0, y0 = self.center()
        r = self.radius()
        th = np.linspace(0.0, 2.0 * np.pi, n_theta)
        zs = np.linspace(zmin, zmax, n_z)
        TH, Z = np.meshgrid(th, zs, indexing='ij')
        X = np.ascontiguousarray((x0 + r * np.cos(TH))[:, :, None])
        Y = np.ascontiguousarray((y0 + r * np.sin(TH))[:, :, None])
        Z = np.ascontiguousarray(Z[:, :, None])
        gridToVTK(name, X, Y, Z)
