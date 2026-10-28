from star_lite_design.continuous_winding_model.winding import PolylineCurve, spline_curve, spline_panel_curve, refine
from star_lite_design.finite_build.hex_finite_build_coil import create_hexagonal_filament_grid
from star_lite_design.utils.find_x_point import find_x_point
from simsopt._core import load
from simsopt.field import BiotSavart, Coil, Current
from simsopt.geo import CurveRZFourier, SurfaceXYZTensorFourier, BoozerSurface, Volume
import numpy as np
from scipy.spatial import cKDTree

"""
The coil models of design A and the field computations shared by analyze_winding.py,
quadrature_study.py and resolution_study.py.

Coil models:
- filament: the design's coils, one filament per coil carrying the coil current I,
- hex build: the hexagonal finite build model (finite_build/hex_finite_build_coil.py),
  18 closed filaments per coil in two layers around the spine, each carrying I/18,
- continuous: the continuously wound cable of each coil (wind_coils.py) carrying I/18,
  with its two 1 m leads. The path is closed by a straight piece (about 3 cm) between
  the far ends of the leads, which stands in for the power supply.

The currents of each model are scaled by one common factor so that the mean |B| on its
own magnetic axis is 0.0875 T, as in the design (utils/axis_field_strength_penalty.py).
The factor changes no field line, iota or Boozer surface, only |B| and G.

Quadrature in BiotSavart (equal weights at equally spaced parameter values): the
trapezoid rule on the Fourier curves of the filament and hex build models, the midpoint
rule on each segment of the continuous winding's polyline. When the winding is
resampled (quadrature_study.py): the trapezoid rule on a periodic spline through it
(spline_curve), or Gauss-Legendre panels on splines split at its corners
(spline_panel_curve).
"""

infile = "../designs/designA/designA_after_scaled.json"
outdir = "./output/designA/"
r = 0.0108
n_turns = 18
B_axis = 0.0875  # mean |B| on the magnetic axis [T]
n_line = 101  # points on the closed field lines, Fourier order (n_line - 1) / 2


def load_design(config):
    """Boozer surface, (iota, G), magnetic axis and X-point of design A in a current configuration."""
    data = load(infile)
    return data[0][config], data[1][config], data[2][config], data[3][config]


def full_torus(curve, n=n_line):
    """Points of an nfp-periodic CurveRZFourier on n points over the full torus."""
    full = CurveRZFourier(np.linspace(0, 1, n, endpoint=False), curve.order, curve.nfp, curve.stellsym)
    full.x = curve.x
    return full.gamma()


def closed_field_line(bs, guess):
    """Closed field line over one toroidal turn near the guess: its points on the phi grid of n_line points, and whether it converged."""
    xyz = full_torus(guess)
    _, line, success = find_x_point(bs, np.hypot(xyz[:, 0], xyz[:, 1]), xyz[:, 2], 1, (n_line - 1) // 2)
    return line.gamma(), success


# ------------------------------------------------------------------ coil models
def filament_coils(coils, n=None):
    """The design's coils, on n quadrature points each (None: the design's 160)."""
    return [Coil(c.curve if n is None else refine(c.curve, n), Current(c.current.get_value())) for c in coils]


def hex_build_coils(coils, n=None):
    """The 18 filaments per coil of the hex finite build, on n quadrature points each (None: the design's 160)."""
    out = []
    for c in coils:
        curve = c.curve if n is None else refine(c.curve, n)
        out += [Coil(f, Current(c.current.get_value() / n_turns)) for f in create_hexagonal_filament_grid(curve, n_layers=2, radius=r)]
    return out


def path_lengths(d):
    """Length of a continuous winding's whole path (cable and leads), and its mean turn length."""
    P = d["path"]
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    a, b = d["turns"][0][0], d["turns"][-1][1]
    return seg.sum(), seg[a:b].sum() / n_turns


def continuous_coils(coils, n=None, rule="adaptive"):
    """
    The continuous winding of each coil: the polyline from wind_coils.py (n = None, 1000
    points per turn), or a spline through it with n points per mean turn length, leads
    included, integrated with
    - rule "trapezoid": the trapezoid rule, the spline sampled uniformly in arc length,
    - rule "panels": Gauss-Legendre panels, the spline split at its corners (spline_panel_curve),
    - rule "adaptive": the same, with shorter panels in the tight bends near the leads
      (no longer than half the bend radius), so more than n points per turn on average.
    """
    out = []
    for i, c in enumerate(coils):
        d = np.load(outdir + f"coil{i}.npz")
        assert np.isclose(float(d["current"]), c.current.get_value())
        length, turn = path_lengths(d)
        if n is None:
            curve = PolylineCurve(d["path"])
        elif rule == "trapezoid":
            curve = spline_curve(d["path"], int(round(n * length / turn)))
        else:
            curve = spline_panel_curve(d["path"], turn / n, bend_fraction=0.5 if rule == "adaptive" else None)
        out.append(Coil(curve, Current(c.current.get_value() / n_turns)))
    return out


def scale_to_axis_field(coil_list, guess):
    """
    Scale the currents of coil_list by one factor so that the mean |B| on their magnetic
    axis (the closed field line near guess) is B_axis. Returns the BiotSavart, the axis
    points, whether the axis converged and the factor.
    """
    bs = BiotSavart(coil_list)
    axis, success = closed_field_line(bs, guess)
    bs.set_points(axis)
    factor = B_axis / bs.AbsB().mean()
    for c in coil_list:
        c.current.x = [factor * c.current.get_value()]
    return bs, axis, success, factor


# ------------------------------------------------------------------ Boozer surfaces
def full_torus_surface(x, mpol, ntor, nphi, ntheta):
    surf = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, nfp=1, stellsym=False,
                                   quadpoints_phi=np.linspace(0, 1, nphi, endpoint=False),
                                   quadpoints_theta=np.linspace(0, 1, ntheta, endpoint=False))
    surf.x = x
    return surf


def design_surface_full_torus(s, mpol, ntor):
    """The design's surface s fitted on the full torus (nfp 1, no symmetry) at resolution mpol, ntor."""
    phis = np.linspace(0, 1, 2 * ntor + 1, endpoint=False)
    thetas = np.linspace(0, 1, 2 * mpol + 1, endpoint=False)
    s_eval = SurfaceXYZTensorFourier(mpol=s.mpol, ntor=s.ntor, nfp=s.nfp, stellsym=s.stellsym,
                                     quadpoints_phi=phis, quadpoints_theta=thetas)
    s_eval.x = s.x
    full = SurfaceXYZTensorFourier(mpol=mpol, ntor=ntor, nfp=1, stellsym=False, quadpoints_phi=phis, quadpoints_theta=thetas)
    full.least_squares_fit(s_eval.gamma())
    return full


def boozer(bs, x0, iota, G, mpol, ntor, label):
    """
    Exact Boozer surface of bs on the full torus at volume label, started from the surface
    dofs x0, on (2 ntor + 1) x (2 mpol + 1) collocation points.
    """
    surf = full_torus_surface(x0, mpol, ntor, 2 * ntor + 1, 2 * mpol + 1)
    b = BoozerSurface(bs, surf, Volume(surf), label, options=dict(verbose=False))
    res = b.run_code(iota=iota, G=G)
    return res, surf


def continuation(start, target, x, iota, G, solve, max_step=0.25, min_step=1 / 64):
    """
    Boozer surface of the coils target by continuation from the coils start, whose surface
    (dofs x, iota, G) is known: the field of the current fraction 1 - lam in start and lam in
    target, for lam from 0 to 1, each solve started from the last one. The step in lam is
    halved after a failed solve and doubled (up to max_step) after a success.
    solve(bs, x, iota, G) returns (res, surf). Returns the last converged (res, surf) and
    the values of lam it went through; it failed if the last one is not 1.
    """
    a = [Coil(c.curve, Current(c.current.get_value())) for c in start]
    b = [Coil(c.curve, Current(c.current.get_value())) for c in target]
    bs = BiotSavart(a + b)
    lam, step, path, last = 0., max_step, [], None
    while lam < 1 and step >= min_step:
        trial = min(1., lam + step)
        for c, c0 in zip(a, start):
            c.current.x = [(1 - trial) * c0.current.get_value()]
        for c, c0 in zip(b, target):
            c.current.x = [trial * c0.current.get_value()]
        res, surf = solve(bs, x, iota, G)
        if res["success"]:
            lam, x, iota, G, last = trial, surf.x.copy(), res["iota"], res["G"], (res, surf)
            path.append(lam)
            step = min(2 * step, max_step)
        else:
            step /= 2
    return last, path


def surface_distance(xa, xb, mpol, ntor):
    """Largest distance from full torus surface a to surface b [m] (nearest point of b, then its tangent plane)."""
    pa = full_torus_surface(xa, mpol, ntor, 200, 100).gamma().reshape((-1, 3))
    fb = full_torus_surface(xb, mpol, ntor, 400, 200)
    pb, nb = fb.gamma().reshape((-1, 3)), fb.unitnormal().reshape((-1, 3))
    _, k = cKDTree(pb).query(pa)
    return np.abs(np.sum((pa - pb[k]) * nb[k], axis=1)).max()


def qs_error(surf, bs, sDIM=20):
    """
    QS error of simsopt's NonQuasiSymmetricRatio (quasi-axisymmetry), the square root
    of the ratio of the non-QS to the QS part of |B| on the surface, on 2 sDIM points
    per field period. Computed here because simsopt's objective also computes its
    gradient, which fails for surfaces without stellarator symmetry.
    """
    e = SurfaceXYZTensorFourier(mpol=surf.mpol, ntor=surf.ntor, stellsym=surf.stellsym, nfp=surf.nfp,
                                quadpoints_phi=np.linspace(0, 1 / surf.nfp, 2 * sDIM, endpoint=False),
                                quadpoints_theta=np.linspace(0, 1, 2 * sDIM, endpoint=False))
    e.x = surf.x
    bs.set_points(e.gamma().reshape((-1, 3)))
    modB = np.linalg.norm(bs.B(), axis=1).reshape(e.gamma().shape[:2])
    dS = np.linalg.norm(e.normal(), axis=2)
    B_QS = (np.mean(modB * dS, axis=0) / np.mean(dS, axis=0))[None, :]
    return np.sqrt(np.mean(dS * (modB - B_QS)**2) / np.mean(dS * B_QS**2))
