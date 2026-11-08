"""The plasma's own magnetic field as a simsopt MagneticField that lives inside a design JSON.

Finite-beta combined-stage devices (beta 1 %) are traced in the coils + plasma field: the coil field alone describes a
different plasma (hsx_scan_ext/c24_i0.25_A10_w3: the magnetic axis moves 13 cm).  The plasma field is the second
group of a coils + plasma mgrid (workflows/flare_analysis/plasma_mgrid.py): virtual casing of the VMEX boundary
outside the plasma, B_VMEX - B_coils inside, blended across a thin skin.  Two classes carry it into an archive:

GridPlasmaField(grid, data, meta)
    B_plasma on the mgrid's (phi, Z, R) grid over one field period, stored in the JSON as int16 (one scale per
    component, ~1e-6 T resolution) + zlib + base64 -- a few MB per device.  Cubic splines in (phi, Z, R), phi periodic,
    like FLARE's mgrid interpolation; first and second derivatives by central differences of the spline.
    ``GridPlasmaField.from_mgrid(path, group=1)``.

CoilsPlasmaField(coils, plasma)
    B_coils (BiotSavart) + B_plasma.  It quacks like a BiotSavart where the array scripts and BoozerSurface need it
    (``.coils``, ``._coils``, ``compute``, ``B_vjp`` w.r.t. the coil dofs), so a BoozerSurface, axis or X-point stored with
    it is solved in -- and re-solves in -- the coils + plasma field.  Code that rebuilds ``BiotSavart(bs.coils)`` gets
    the coils alone again; preprocessing.simsopt_io.load_design returns the coil part as ``bs`` and the plasma part
    as ``plasma``, so the FLARE workflows keep tracing through their 2-group mgrid.
"""
import base64
import zlib

import numpy as np
from scipy import ndimage
from simsopt.field import BiotSavart, MagneticField

_PAD = 3          # periodic phi padding for the splines
_EPS = 1e-4       # central-difference step for the derivatives [m]


def _encode(a):
    a = np.asarray(a, dtype=float)
    scale = float(np.max(np.abs(a))) / 32000.0 or 1.0
    q = np.rint(a / scale).astype("<i2")
    return dict(shape=list(a.shape), scale=scale, b64=base64.b64encode(zlib.compress(q.tobytes(), 6)).decode("ascii"))


def _decode(d):
    q = np.frombuffer(zlib.decompress(base64.b64decode(d["b64"])), dtype="<i2")
    return q.reshape(d["shape"]).astype(float) * d["scale"]


class GridPlasmaField(MagneticField):
    """B_plasma sampled on an mgrid grid; see the module docstring.

    Args:
        grid: dict(nfp, rmin, rmax, zmin, zmax, nr, nz, nphi) -- the mgrid's grid over one field period
        data: dict(br=, bp=, bz=) as written by ``_encode`` (arrays of shape (nphi, nz, nr), cylindrical components)
        meta: provenance (wout, its md5, beta, the mgrid file and its checks), kept verbatim
    """

    def __init__(self, grid, data, meta=None):
        self.grid = {k: (int(v) if k in ("nfp", "nr", "nz", "nphi") else float(v)) for k, v in grid.items()}
        self.data = data
        self.meta = meta or {}
        g = self.grid
        B = np.stack([_decode(data[c]) for c in ("br", "bp", "bz")], axis=-1)        # (nphi, nz, nr, 3)
        B = np.concatenate([B[-_PAD:], B, B[:_PAD]], axis=0)
        self._origin = np.array([-_PAD * 2 * np.pi / (g["nfp"] * g["nphi"]), g["zmin"], g["rmin"]])
        self._step = np.array([2 * np.pi / (g["nfp"] * g["nphi"]), (g["zmax"] - g["zmin"]) / (g["nz"] - 1),
                               (g["rmax"] - g["rmin"]) / (g["nr"] - 1)])
        self._coef = [ndimage.spline_filter(B[..., i], order=3, mode="nearest") for i in range(3)]
        MagneticField.__init__(self)

    @classmethod
    def from_mgrid(cls, path, group=1, meta=None):
        """The ``group``-th field group of an mgrid (plasma_mgrid.py writes coils = 0, plasma = 1)."""
        from simsopt.field import MGrid
        import netCDF4
        mg = MGrid.from_file(str(path))
        with netCDF4.Dataset(str(path)) as d:
            nfp = int(d["nfp"][:])
        grid = dict(nfp=nfp, rmin=mg.rmin, rmax=mg.rmax, zmin=mg.zmin, zmax=mg.zmax, nr=mg.nr, nz=mg.nz,
                    nphi=mg.nphi)
        data = {c: _encode(arr[group]) for c, arr in (("br", mg.br_arr), ("bp", mg.bp_arr), ("bz", mg.bz_arr))}
        return cls(grid, data, dict(meta or {}, mgrid=str(path), group=int(group)))

    def _cart(self, x):
        """B (cartesian) at points x (n, 3); zero outside the grid box."""
        g = self.grid
        R = np.hypot(x[:, 0], x[:, 1])
        phi = np.arctan2(x[:, 1], x[:, 0])
        idx = np.vstack([(np.mod(phi, 2 * np.pi / g["nfp"]) - self._origin[0]) / self._step[0],
                         (x[:, 2] - self._origin[1]) / self._step[1],
                         (R - self._origin[2]) / self._step[2]])
        br, bp, bz = (ndimage.map_coordinates(c, idx, order=3, mode="nearest", prefilter=False) for c in self._coef)
        out = (R < g["rmin"]) | (R > g["rmax"]) | (x[:, 2] < g["zmin"]) | (x[:, 2] > g["zmax"])
        c, s = np.cos(phi), np.sin(phi)
        B = np.stack([br * c - bp * s, br * s + bp * c, bz], axis=1)
        B[out] = 0.0
        return B

    def _dcart(self, x, f):
        """d f / dX by central differences: (n, 3, ...) with [:, j] = d/dx_j."""
        return np.stack([(f(x + _EPS * e) - f(x - _EPS * e)) / (2 * _EPS) for e in np.eye(3)], axis=1)

    def _B_impl(self, B):
        B[:] = self._cart(self.get_points_cart_ref())

    def _dB_by_dX_impl(self, dB):
        dB[:] = self._dcart(self.get_points_cart_ref(), self._cart)

    def _d2B_by_dXdX_impl(self, ddB):
        ddB[:] = self._dcart(self.get_points_cart_ref(), lambda y: self._dcart(y, self._cart))


class CoilsPlasmaField(MagneticField):
    """B_coils + B_plasma, with the BiotSavart interface the archive tools use; see the module docstring."""

    def __init__(self, coils, plasma):
        self.coils = list(coils)
        self._coils = self.coils
        self.plasma = plasma
        self.coil_field = BiotSavart(self.coils)
        MagneticField.__init__(self, depends_on=[self.coil_field, plasma])

    def _set_points_cb(self):
        self.coil_field.set_points_cart(self.get_points_cart_ref())

    def compute(self, derivatives=0):
        """BiotSavart.compute: BoozerSurface calls it before B(); here B() evaluates on demand."""
        self.coil_field.compute(derivatives)

    # the plasma part is evaluated straight from the grid at this field's points, so a GridPlasmaField shared by
    # several CoilsPlasmaField objects never mixes up their point sets
    def _B_impl(self, B):
        B[:] = self.coil_field.B() + self.plasma._cart(self.get_points_cart_ref())

    def _dB_by_dX_impl(self, dB):
        p = self.plasma
        dB[:] = self.coil_field.dB_by_dX() + p._dcart(self.get_points_cart_ref(), p._cart)

    def _d2B_by_dXdX_impl(self, ddB):
        p = self.plasma
        ddB[:] = self.coil_field.d2B_by_dXdX() + p._dcart(self.get_points_cart_ref(), lambda y: p._dcart(y, p._cart))

    def _A_impl(self, A):
        raise NotImplementedError("the vector potential of the gridded plasma field is not available")

    # derivatives w.r.t. the coil dofs: the plasma field has none (it is frozen with the equilibrium), so these are
    # the coil field's own
    def B_vjp(self, v):
        return self.coil_field.B_vjp(v)

    def B_and_dB_vjp(self, v, vgrad):
        return self.coil_field.B_and_dB_vjp(v, vgrad)
