import os
import unittest
import numpy as np
from simsopt._core import Optimizable, load
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.field import BiotSavart
from simsopt.geo import CurveLength, CurveXYZFourierSymmetries
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.vmex_double_null import (FreeXLine, FreeXLineHyperbolicity, XLineFieldLineConstraint,
                                                     XpointHyperbolicity, XpointPlasmaDistance,
                                                     ensure_fieldline_solved, fieldline_restore,
                                                     fieldline_snapshot)
from star_lite_design.utils.augmented_lagrangian import AugmentedLagrangian
from star_lite_design.utils.vmex_combined_stage import WeightedSum
from star_lite_design.utils.finite_difference import taylor_test

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = os.path.join(HERE, "..", "..", "convert", "designA_after_scaled.json")


class FakePlasma(Optimizable):
    """Elliptic torus R = R0 + a cos(t), Z = z0 + k a sin(t) with dofs (a, k, z0),
    exposing the VmexPlasma interface used by XpointPlasmaDistance."""

    def __init__(self, R0=0.5, a=0.08, k=1.6, z0=0.0, nphi=64, ntheta=48):
        self.R0, self.nphi, self.ntheta, self.nfp = R0, nphi, ntheta, 2
        self.n_backward = 0
        Optimizable.__init__(self, x0=np.array([a, k, z0]), names=["a", "k", "z0"])

    def _grid(self):
        phi = np.linspace(0, 2 * np.pi, self.nphi, endpoint=False)[:, None]
        t = np.linspace(0, 2 * np.pi, self.ntheta, endpoint=False)[None, :]
        return phi, t

    def evaluate(self):
        a, k, z0 = self.local_full_x
        phi, t = self._grid()
        R = self.R0 + a * np.cos(t) + 0 * phi
        Z = z0 + k * a * np.sin(t) + 0 * phi
        pts = np.stack([R * np.cos(phi), R * np.sin(phi), Z])
        return {"accepted": True, "out": {"boundary_points": pts}}

    def pullback(self, ct):
        a, k, z0 = self.local_full_x
        phi, t = self._grid()
        c = np.asarray(ct["boundary_points"])
        dR_da, dZ_da, dZ_dk = np.cos(t) + 0 * phi, k * np.sin(t) + 0 * phi, a * np.sin(t) + 0 * phi
        g_a = np.sum(c[0] * dR_da * np.cos(phi) + c[1] * dR_da * np.sin(phi) + c[2] * dZ_da)
        g_k = np.sum(c[2] * dZ_dk)
        g_z0 = np.sum(c[2])
        self.n_backward += 1
        return Derivative({self: np.array([g_a, g_k, g_z0])})


@unittest.skipUnless(os.path.exists(DESIGN), "designA archive not available")
class TestDoubleNullTerms(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = load(DESIGN)
        cls.coils = data[0][0].biotsavart.coils
        xp = data[3][0]
        cls.xpoint = PeriodicFieldLine(BiotSavart(cls.coils), xp.curve,
                                       options={"newton_tol": 1e-13, "newton_maxiter": 40})
        res = cls.xpoint.run_code(CurveLength(xp.curve).J())
        assert res["success"]

    def test_designA_xpoint_is_hyperbolic(self):
        h = XpointHyperbolicity(self.xpoint, BiotSavart(self.coils), margin=0.2)
        print(f"\n[hyperbolicity] tr(M) = {h.trace():+.4f}, J(margin 0.2) = {h.J():.3e}")
        self.assertGreater(abs(h.trace()), 2.2)
        self.assertEqual(h.J(), 0.0)

    def test_hyperbolicity_taylor(self):
        h = XpointHyperbolicity(self.xpoint, BiotSavart(self.coils), margin=1.5)   # active: 3.5 > |tr M|
        self.assertGreater(h.J(), 0.0)
        x0 = h.x.copy()

        def fun(x):
            h.x = x
            return h.J(), h.dJ()
        err = taylor_test(fun, x0, order=4)
        h.x = x0
        self.assertLess(err, 1e-6)

    def test_distance_taylor_coils_and_plasma(self):
        plasma = FakePlasma()
        term = XpointPlasmaDistance(self.xpoint, plasma, d_min=0.07, d_max=0.09)   # both sides active
        d = term.distances()
        print(f"\n[distance] X-point to fake boundary: {d.min()*100:.2f} .. {d.max()*100:.2f} cm, J = {term.J():.3e}")
        self.assertGreater(term.J(), 0.0)
        obj = WeightedSum([(3.0, term)])
        x0 = obj.x.copy()

        def fun(x):
            obj.x = x
            return obj.J(), obj.dJ()
        err = taylor_test(fun, x0, order=4)
        obj.x = x0
        self.assertLess(err, 1e-6)

    def test_distance_shares_one_pullback(self):
        plasma = FakePlasma()
        t1 = XpointPlasmaDistance(self.xpoint, plasma, d_min=0.07, d_max=0.09)
        t2 = XpointPlasmaDistance(self.xpoint, plasma, d_min=0.06, d_max=0.08)
        obj = WeightedSum([(1.0, t1), (2.0, t2)])
        plasma.n_backward = 0
        obj.dJ()
        self.assertEqual(plasma.n_backward, 1)


class _Dot(Optimizable):
    """``w . c(x)`` of an EqualityConstraint block, to Taylor-test ``residuals_vjp``."""

    def __init__(self, constraint, key, w):
        self.constraint, self.key, self.w = constraint, key, w
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[constraint])

    def J(self):
        return float(self.w @ self.constraint.residuals()[self.key])

    @derivative_dec
    def dJ(self):
        return self.constraint.residuals_vjp({self.key: self.w})


@unittest.skipUnless(os.path.exists(DESIGN), "designA archive not available")
class TestFreeXLine(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        data = load(DESIGN)
        cls.coils = data[0][0].biotsavart.coils
        xp = data[3][0]
        cls.solved = PeriodicFieldLine(BiotSavart(cls.coils), xp.curve,
                                       options={"newton_tol": 1e-13, "newton_maxiter": 40})
        assert cls.solved.run_code(CurveLength(xp.curve).J())["success"]

    def _xline(self, perturb=0.0, seed=0):
        c = self.solved.curve
        curve = CurveXYZFourierSymmetries(c.quadpoints, c.order, c.nfp, c.stellsym, ntor=c.ntor)
        curve.x = c.x + perturb * np.random.default_rng(seed).standard_normal(c.x.size)
        return FreeXLine(curve, self.solved.res["length"] * (1 + perturb))

    def test_solved_xpoint_is_feasible_and_traces_agree(self):
        xl = self._xline()
        con = XLineFieldLineConstraint(xl, BiotSavart(self.coils))
        r = con.residuals()["fieldline"]
        self.assertEqual(r.size, xl.curve.x.size + 1)             # square: curve dofs + length
        self.assertLess(np.max(np.abs(r)), 1e-10)
        tr_free = FreeXLineHyperbolicity(xl, BiotSavart(self.coils), margin=0.2).trace()
        tr_track = XpointHyperbolicity(self.solved, BiotSavart(self.coils), margin=0.2).trace()
        print(f"\n[free X-line] residual {np.max(np.abs(r)):.2e}, tr M free {tr_free:+.6f} tracked {tr_track:+.6f}")
        self.assertAlmostEqual(tr_free, tr_track, places=8)

    def test_constraint_vjp_taylor(self):
        xl = self._xline(perturb=1e-3)
        con = XLineFieldLineConstraint(xl, BiotSavart(self.coils))
        w = np.random.default_rng(1).standard_normal(con.residuals()["fieldline"].size)
        obj = _Dot(con, "fieldline", w)
        x0 = obj.x.copy()

        def fun(x):
            obj.x = x
            return obj.J(), obj.dJ()
        err = taylor_test(fun, x0, order=4)
        obj.x = x0
        self.assertLess(err, 1e-6)

    def test_hyperbolicity_taylor(self):
        xl = self._xline(perturb=1e-3)
        h = FreeXLineHyperbolicity(xl, BiotSavart(self.coils), margin=1.5)   # active: 3.5 > |tr M|
        self.assertGreater(h.J(), 0.0)
        x0 = h.x.copy()

        def fun(x):
            h.x = x
            return h.J(), h.dJ()
        err = taylor_test(fun, x0, order=4)
        h.x = x0
        self.assertLess(err, 1e-6)

    def test_signed_distance_is_negative_inside(self):
        xl = self._xline()
        c = xl.curve
        # the same curve shrunk halfway onto the fake plasma's circular axis (R0 = 0.5, z = 0) lies inside
        g = c.gamma()
        R = np.hypot(g[:, 0], g[:, 1])
        inside = CurveXYZFourierSymmetries(c.quadpoints, c.order, c.nfp, c.stellsym, ntor=c.ntor)
        scale = (0.5 + 0.3 * (R - 0.5)) / R
        inside.least_squares_fit(np.column_stack([g[:, 0] * scale, g[:, 1] * scale, 0.3 * g[:, 2]]))
        plasma = FakePlasma(a=0.12, k=1.6)
        d_in = XpointPlasmaDistance(FreeXLine(inside, 1.0), plasma, signed=True).distances()
        d_unsigned = XpointPlasmaDistance(FreeXLine(inside, 1.0), plasma, signed=False).distances()
        self.assertTrue(np.all(d_in < 0.0))
        np.testing.assert_allclose(np.abs(d_in), d_unsigned)

    def test_distance_taylor_free_line_and_plasma(self):
        xl = self._xline(perturb=1e-3)
        term = XpointPlasmaDistance(xl, FakePlasma(), d_min=0.07, d_max=0.09, signed=True)
        self.assertGreater(term.J(), 0.0)
        obj = WeightedSum([(3.0, term)])
        x0 = obj.x.copy()

        def fun(x):
            obj.x = x
            return obj.J(), obj.dJ()
        err = taylor_test(fun, x0, order=4)
        obj.x = x0
        self.assertLess(err, 1e-6)


class _BrokenCurve:
    x = np.zeros(3)

    def gamma(self):
        return np.zeros((4, 3))


class _BrokenLine:
    """A field line whose Newton solve hits |B| = 0 (as on a coil-crossing trial step)."""
    def __init__(self):
        self.need_to_run_code, self.res, self.curve = True, {"length": 1.0, "success": True}, _BrokenCurve()

    def run_code(self, length):
        raise ValueError("array must not contain infs or NaNs")


class TestFieldLineFailurePath(unittest.TestCase):

    def test_numerical_breakdown_is_a_failed_solve(self):
        line = _BrokenLine()
        self.assertFalse(ensure_fieldline_solved(line))
        self.assertFalse(line.res["success"])

    @unittest.skipUnless(os.path.exists(DESIGN), "designA archive not available")
    def test_restore_after_garbage_dofs(self):
        data = load(DESIGN)
        coils = data[0][0].biotsavart.coils
        xp = PeriodicFieldLine(BiotSavart(coils), data[3][0].curve, options={"newton_tol": 1e-13, "newton_maxiter": 40})
        self.assertTrue(xp.run_code(CurveLength(xp.curve).J())["success"])
        snap = fieldline_snapshot(xp)
        xp.curve.x = xp.curve.x + 0.05          # what a diverged Newton leaves behind
        fieldline_restore(xp, snap)
        np.testing.assert_allclose(xp.curve.x, snap["dofs"])
        self.assertTrue(ensure_fieldline_solved(xp))


if __name__ == "__main__":
    unittest.main()
