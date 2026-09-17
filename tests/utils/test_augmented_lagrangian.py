import unittest
import numpy as np
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from star_lite_design.utils.augmented_lagrangian import (
    AugmentedLagrangian, EqualityConstraint, InequalityConstraint, solve_augmented_lagrangian)
from star_lite_design.utils.finite_difference import taylor_test


class Point(Optimizable):
    def __init__(self, x0):
        Optimizable.__init__(self, x0=np.asarray(x0, dtype=float),
                             names=[f"x{i}" for i in range(len(x0))])


class Distance(Optimizable):
    """f(x) = 1/2 |x - a|^2"""
    def __init__(self, point, a):
        self.point, self.a = point, np.asarray(a, dtype=float)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[point])

    def J(self):
        return 0.5 * np.sum((self.point.x - self.a) ** 2)

    @derivative_dec
    def dJ(self):
        return Derivative({self.point: self.point.x - self.a})


class LinearAndSphere(EqualityConstraint):
    """Block 'lin': A x - b; block 'sphere': |x|^2 - R^2."""
    def __init__(self, point, A, b, R=None):
        self.point, self.A, self.b, self.R = point, np.asarray(A, float), np.asarray(b, float), R
        EqualityConstraint.__init__(self, x0=np.asarray([]), depends_on=[point])

    def residuals(self):
        x = self.point.x
        out = {"lin": self.A @ x - self.b}
        if self.R is not None:
            out["sphere"] = np.array([x @ x - self.R ** 2])
        return out

    def residuals_vjp(self, ct):
        x = self.point.x
        g = self.A.T @ ct["lin"]
        if "sphere" in ct:
            g = g + 2.0 * x * ct["sphere"][0]
        return Derivative({self.point: g})


class SharedOwner(Optimizable):
    """Stands in for an expensive shared pullback (e.g. one VMEX adjoint)."""
    def __init__(self, point):
        self.point, self.calls = point, 0
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[point])

    def pullback(self, ct):
        self.calls += 1
        return Derivative({self.point: np.asarray(ct["v"], dtype=float)})


class LinearViaOwner(EqualityConstraint):
    def __init__(self, point, owner, A, b):
        self.point, self.owner, self.A, self.b = point, owner, np.asarray(A, float), np.asarray(b, float)
        EqualityConstraint.__init__(self, x0=np.asarray([]), depends_on=[point, owner])

    def residuals(self):
        return {"lin": self.A @ self.point.x - self.b}

    def residuals_vjp_parts(self, ct):
        return Derivative({}), {self.owner: {"v": self.A.T @ ct["lin"]}}


class TestAugmentedLagrangian(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.a = rng.standard_normal(4)
        self.A = rng.standard_normal((2, 4))
        self.b = rng.standard_normal(2)

    def test_linear_kkt(self):
        """Converges to the analytic KKT point and multipliers at a finite penalty."""
        p = Point(np.zeros(4))
        al = AugmentedLagrangian(Distance(p, self.a), [LinearAndSphere(p, self.A, self.b)], penalty=10.0)
        x, rec = solve_augmented_lagrangian(al, ctol=1e-9, gtol=1e-7, outer_maxiter=30)
        AAT = self.A @ self.A.T
        lam_star = np.linalg.solve(AAT, self.b - self.A @ self.a)
        x_star = self.a + self.A.T @ lam_star
        self.assertTrue(rec["converged"])
        np.testing.assert_allclose(x, x_star, atol=1e-7)
        np.testing.assert_allclose(al.multipliers["0:lin"], lam_star, atol=1e-6)
        self.assertLess(al.penalties["0:lin"], 1e4)

    def test_nonlinear_converges(self):
        # feasible by construction: sphere radius twice the plane's distance from 0
        R = 2.0 * abs(self.b[0]) / np.linalg.norm(self.A[0])
        p = Point(np.full(4, 0.3))
        al = AugmentedLagrangian(Distance(p, self.a),
                                 [LinearAndSphere(p, self.A[:1], self.b[:1], R=R)], penalty=10.0)
        x, rec = solve_augmented_lagrangian(al, ctol=1e-9, gtol=1e-7, outer_maxiter=40)
        self.assertTrue(rec["converged"])
        self.assertLess(abs(x @ x - R ** 2), 1e-8)
        self.assertLess(abs(self.A[0] @ x - self.b[0]), 1e-8)
        self.assertLess(max(al.penalties.values()), 1e5)

    def test_taylor(self):
        p = Point(np.full(4, 0.2))
        al = AugmentedLagrangian(Distance(p, self.a), [LinearAndSphere(p, self.A, self.b, R=0.7)], penalty=3.0)
        al.blocks()
        al.multipliers["0:lin"][:] = [0.4, -1.1]
        al.multipliers["0:sphere"][:] = [2.5]

        def fun(x):
            al.x = x
            return al.J(), al.dJ()
        self.assertLess(taylor_test(fun, al.x.copy(), order=6), 1e-9)

    def test_shared_pullback_called_once(self):
        p = Point(np.full(4, 0.1))
        owner = SharedOwner(p)
        c1 = LinearViaOwner(p, owner, self.A[:1], self.b[:1])
        c2 = LinearViaOwner(p, owner, self.A[1:], self.b[1:])
        al = AugmentedLagrangian(Distance(p, self.a), [c1, c2], penalty=5.0)
        al.blocks()
        al.multipliers["0:lin"][:] = 0.3
        al.multipliers["1:lin"][:] = -0.2
        g = al.dJ()
        self.assertEqual(owner.calls, 1)
        ref = AugmentedLagrangian(Distance(p, self.a), [LinearAndSphere(p, self.A, self.b)], penalty=5.0)
        ref.blocks()
        ref.multipliers["0:lin"][:] = [0.3, -0.2]
        np.testing.assert_allclose(g, ref.dJ(), rtol=1e-12, atol=1e-14)

    def test_small_initial_penalty(self):
        """A far-too-small rho0 must still converge (schedules keep contracting)."""
        p = Point(np.zeros(4))
        al = AugmentedLagrangian(Distance(p, self.a), [LinearAndSphere(p, self.A, self.b)], penalty=1e-3)
        x, rec = solve_augmented_lagrangian(al, ctol=1e-9, gtol=1e-7, outer_maxiter=40)
        self.assertTrue(rec["converged"])
        self.assertLess(np.max(np.abs(self.A @ x - self.b)), 1e-9)

    def _penalty_sequence(self, eta0, n=4):
        """Penalties after n outer updates of a block frozen at residual -0.4 (no inner solve)."""
        p = Point(np.full(4, 0.2))
        al = AugmentedLagrangian(Distance(p, self.a), [LinearAndSphere(p, [[1.0, 0.0, 0.0, 0.0]], [0.6])],
                                 penalty=10.0, penalty_growth=2.0, eta0=eta0)
        return [al.update(1.0)["penalties"]["0:lin"] for _ in range(n)]

    def test_penalty_reset_keeps_eta0_scale(self):
        """With eta0 given, a block that stays violated grows its penalty at EVERY outer update; the
        textbook reset (eta = 1/rho^0.1 = 0.74 at rho 20) let an O(0.1) residual pass every other time."""
        self.assertEqual(self._penalty_sequence(eta0=1e-3), [20.0, 40.0, 80.0, 160.0])
        self.assertEqual(self._penalty_sequence(eta0=None), [10.0, 20.0, 20.0, 40.0])

    def test_stall_safeguard_updates_multipliers_at_the_cap(self):
        """An unreachable feasibility target must not freeze the multipliers at zero.

        Combined-stage runs 816 / 817 / 821 / 823 ended with lambda = 0 and rho = penalty_max on their B.n block: the
        schedule demanded 5e-4 from coil sets whose measured floor is ~1.2e-3, so every outer took the penalty branch
        and the method degenerated into a pure (capped) penalty method.
        """
        R = 0.25 * abs(self.b[0]) / np.linalg.norm(self.A[0])        # sphere too small for the plane: infeasible
        p = Point(np.full(4, 0.3))
        al = AugmentedLagrangian(Distance(p, self.a), [LinearAndSphere(p, self.A[:1], self.b[:1], R=R)],
                                 penalty=10.0, penalty_growth=10.0, penalty_max=1e3, eta0=1e-3)
        _, rec = solve_augmented_lagrangian(al, ctol=1e-9, gtol=1e-7, outer_maxiter=12)
        self.assertFalse(rec["converged"])
        self.assertEqual(max(al.penalties.values()), 1e3)                                   # capped, as before
        self.assertTrue(any("stalled" in a for h in al.history for a in h["actions"].values()))
        self.assertGreater(max(float(np.max(np.abs(v))) for v in al.multipliers.values()), 0.0)

    def test_infeasible_grows_penalty(self):
        """An infeasible constraint set is reported by penalty growth, not false convergence."""
        R = 0.25 * abs(self.b[0]) / np.linalg.norm(self.A[0])
        p = Point(np.full(4, 0.3))
        al = AugmentedLagrangian(Distance(p, self.a),
                                 [LinearAndSphere(p, self.A[:1], self.b[:1], R=R)],
                                 penalty=10.0, penalty_max=1e6)
        _, rec = solve_augmented_lagrangian(al, ctol=1e-9, gtol=1e-7, outer_maxiter=12)
        self.assertFalse(rec["converged"])
        self.assertGreater(max(al.penalties.values()), 1e3)


class UpperBound(InequalityConstraint):
    """Block 'bound': c(x) = limit - x[0] >= 0."""
    def __init__(self, point, limit):
        self.point, self.limit = point, float(limit)
        InequalityConstraint.__init__(self, x0=np.asarray([]), depends_on=[point])

    def residuals(self):
        return {"bound": np.array([self.limit - self.point.x[0]])}

    def residuals_vjp(self, ct):
        g = np.zeros(self.point.x.size)
        g[0] = -float(ct["bound"][0])
        return Derivative({self.point: g})


class TestInequalityConstraint(unittest.TestCase):
    """Powell-Hestenes-Rockafellar blocks: an active bound reproduces its KKT multiplier, a slack one costs nothing."""

    def test_active_bound_recovers_kkt(self):
        # min 1/2 |x - (2, 0)|^2 s.t. x0 <= 1  ->  x = (1, 0), multiplier 1
        p = Point([0.0, 0.0])
        al = AugmentedLagrangian(Distance(p, [2.0, 0.0]), [UpperBound(p, 1.0)], penalty=10.0)
        x, rec = solve_augmented_lagrangian(al, ctol=1e-9, gtol=1e-7, outer_maxiter=30)
        self.assertTrue(rec["converged"])
        np.testing.assert_allclose(x, [1.0, 0.0], atol=1e-7)
        self.assertAlmostEqual(float(al.multipliers["0:bound"][0]), 1.0, places=5)

    def test_slack_bound_is_free(self):
        # the unconstrained optimum already satisfies the bound: no force, no multiplier, no penalty growth
        p = Point([0.0, 0.0])
        al = AugmentedLagrangian(Distance(p, [-1.0, 0.0]), [UpperBound(p, 1.0)], penalty=10.0)
        x, rec = solve_augmented_lagrangian(al, ctol=1e-9, gtol=1e-7, outer_maxiter=20)
        self.assertTrue(rec["converged"])
        np.testing.assert_allclose(x, [-1.0, 0.0], atol=1e-7)
        self.assertAlmostEqual(float(al.multipliers["0:bound"][0]), 0.0, places=9)
        self.assertEqual(al.penalties["0:bound"], 10.0)

    def test_taylor_on_the_active_branch(self):
        p = Point([0.6, 0.2])                      # c = -0.1 < lambda/rho = 0.267: active, smooth locally
        al = AugmentedLagrangian(Distance(p, [2.0, 0.0]), [UpperBound(p, 0.5)], penalty=3.0)
        al.blocks()
        al.multipliers["0:bound"][:] = [0.8]

        def fun(x):
            al.x = x
            return al.J(), al.dJ()
        self.assertLess(taylor_test(fun, al.x.copy(), order=6), 1e-9)


if __name__ == "__main__":
    unittest.main()
