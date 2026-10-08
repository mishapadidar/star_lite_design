"""Tests of the net-poloidal-current (Boozer G) field-strength condition and the bootstrap consistency block.

Slow (VMEX solves), data-dependent, opt-in like test_vmex_combined_stage.py:

    VMEX_COMBINED_STAGE_TESTS=1 python -m pytest -s tests/utils/test_vmex_current_bootstrap.py

Seed: the HSX-size nfp 4 band-run final (beta 0.93 %, p = p0 (1 - s)) and its coils, overridable with
``VMEX_BOOT_SEED`` / ``VMEX_BOOT_COILS``. The seed gets a nonzero current (AC = [1, -1], CURTOR = 20 kA) so the current
dofs have a gradient, and kinetic profiles consistent with its pressure (ne flat, Te = Ti linear).
"""
import os
import unittest
from dataclasses import replace

import numpy as np

RUNS = "/data/common/raffael.wendlinger/simflare/runs/combined_stage/dn_32coils_vc_band"
SEED = os.environ.get("VMEX_BOOT_SEED", f"{RUNS}/inputs/input.combined_stage_final")
COILS = os.environ.get("VMEX_BOOT_COILS", f"{RUNS}/coils/coils_final.json")
RUN = os.environ.get("VMEX_COMBINED_STAGE_TESTS") == "1"
E = 1.602176634e-19
MU0 = 4e-7 * np.pi


@unittest.skipUnless(RUN, "set VMEX_COMBINED_STAGE_TESTS=1 (slow VMEX tests)")
class TestNetPoloidalCurrentAndBootstrap(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import vmex as vj
        from vmex import optimize as opt
        from simsopt._core import load
        from simsopt.field import BiotSavart
        from star_lite_design.utils.vmex_combined_stage import (
            BootstrapConsistency, PlasmaCoilInterface, VmexPlasma, VmexQuasisymmetry, WeightedSum,
            bootstrap_mismatch_output, kinetic_pressure_coeffs, kinetic_profiles, net_poloidal_current_output)
        from star_lite_design.utils.augmented_lagrangian import AugmentedLagrangian

        inp = vj.VmecInput.from_file(SEED)
        p0 = float(inp.pres_scale) * float(np.asarray(inp.am)[0])
        n0 = 2e19
        T0 = p0 / (2 * E * n0)
        ne, Te, Ti = [n0], [T0, -T0], [T0, -T0]
        am = np.zeros(21)
        c = kinetic_pressure_coeffs(ne, Te, Ti)
        am[:c.size] = c
        ac = np.zeros(21)
        ac[:2] = [1.0, -1.0]
        inp = replace(inp, am=am, pres_scale=1.0, ncurr=1, pcurr_type="power_series", ac=ac, curtor=2.0e4,
                      ns_array=np.array([16]), ftol_array=np.array([1e-11]), niter_array=np.array([20000]))
        cls.eq0 = opt.solve_equilibrium(inp)
        qs = opt.QuasisymmetryRatioResidual(np.linspace(0.1, 0.9, 4), helicity_m=1, helicity_n=0)
        terms = [(qs, 0.0, 1.0), (opt.aspect_ratio, float(cls.eq0.wout.aspect), 1.0)]
        cls.boot_rows = bootstrap_mismatch_output(kinetic_profiles(ne, Te, Ti), 0, np.linspace(0.1, 0.9, 6))
        extra = {"rbtor": net_poloidal_current_output(), "bootstrap": cls.boot_rows}
        cls.plasma = VmexPlasma(inp, terms, max_mode=1, nphi=16, ntheta=16, plasma_field="vacuum",
                                current_dofs=2, restart_from=cls.eq0, extra_outputs=extra)
        coils = load(COILS)
        cls.interface = PlasmaCoilInterface(cls.plasma, BiotSavart(coils), mode="fourier", mpol=3, ntor=3,
                                            field_strength="net_poloidal_current")
        cls.boot = BootstrapConsistency(cls.plasma)
        objective = WeightedSum([(1.0, VmexQuasisymmetry(cls.plasma))])
        cls.al = AugmentedLagrangian(objective, [cls.interface, cls.boot], penalty=10.0)
        rng = np.random.default_rng(5)
        for name, _, _, r in cls.al.blocks():
            cls.al.multipliers[name] = rng.standard_normal(r.shape)
        cls.x0 = cls.al.x.copy()
        names = cls.al.dof_names
        cls.plasma_mask = np.array([n.startswith(cls.plasma.name + ":") for n in names])
        scale = np.ones(cls.x0.size)
        scale[cls.plasma_mask] = np.asarray(cls.plasma.problem.scales)[cls.plasma.dofs_free_status]
        cls.scale = scale
        print(f"\n[setup] dofs {cls.x0.size} (plasma {cls.plasma_mask.sum()}, incl. current dofs); "
              f"G residual {cls.interface.residuals()['net_poloidal_current'][0]:+.3e}; "
              f"bootstrap rows {cls.boot.residuals()['bootstrap']}")

    def tearDown(self):
        self.al.x = self.x0

    def test_rbtor_matches_wout(self):
        bvco = np.asarray(self.eq0.wout.bvco, dtype=float)
        G_wout = 1.5 * bvco[-1] - 0.5 * bvco[-2]
        G = float(self.plasma.outputs()["rbtor"])
        print(f"[rbtor] state {G:+.12e}  wout {G_wout:+.12e}")
        self.assertLess(abs(G - G_wout) / abs(G_wout), 1e-6)

    def test_linked_current_matches_G(self):
        I_coil, I_eq = self.interface.current_seed
        print(f"[linked current] coils {I_coil:+.6e} A  equilibrium 2 pi G / mu0 {I_eq:+.6e} A")
        self.assertLess(abs(abs(I_coil) - abs(I_eq)) / abs(I_eq), 5e-3)

    def _directional(self, direction, key, h):
        vals = []
        for sgn in (+1.0, -1.0):
            self.al.x = self.x0 + sgn * h * direction
            res = self.interface.residuals() if key == "net_poloidal_current" else self.boot.residuals()
            vals.append(np.asarray(res[key], dtype=float))
        self.al.x = self.x0
        return (vals[0] - vals[1]) / (2 * h)

    def _adjoint(self, key, weights):
        self.al.x = self.x0
        constraint = self.interface if key == "net_poloidal_current" else self.boot
        d, parts = constraint.residuals_vjp_parts({key: weights})
        for owner, ct in parts.items():
            d += owner.pullback(ct)
        return d(self.al)

    def _check(self, key, mask, h, tol):
        rng = np.random.default_rng(7)
        direction = np.where(mask, rng.standard_normal(self.x0.size), 0.0) * self.scale
        direction /= np.linalg.norm(direction)
        self.al.x = self.x0
        r0 = self.interface.residuals()[key] if key == "net_poloidal_current" else self.boot.residuals()[key]
        weights = rng.standard_normal(np.asarray(r0).size)
        fd = float(weights @ self._directional(direction, key, h))
        adj = float(self._adjoint(key, weights) @ direction)
        rel = abs(adj - fd) / max(abs(fd), 1e-300)
        print(f"[{key}] adjoint {adj:+.8e} FD {fd:+.8e} rel {rel:.2e}")
        self.assertLess(rel, tol)

    def test_G_condition_coil_gradient(self):
        self._check("net_poloidal_current", ~self.plasma_mask, 1e-6, 1e-6)

    def test_G_condition_plasma_gradient(self):
        # re-solve FD; the adjoint matched VMEX's frozen-path reference to 2.8e-5 and naive FD to 1.5e-4 (job 2119)
        self._check("net_poloidal_current", self.plasma_mask, 1e-5, 2e-3)

    def _current_mask(self):
        names = self.al.dof_names
        return np.array([self.plasma_mask[i] and (("AC(" in n) or ("CURTOR" in n)) for i, n in enumerate(names)])

    def _check_frozen(self, mask, h, tol):
        """Bootstrap block vs VMEX's frozen-path FD, the valid reference for solver-state quantities.

        A naive re-solve FD is NOT: along the direction below it gave +16.9 / +1.4 / -4.04 at h = 1e-6 / 3e-6 / 1e-5
        against an adjoint of -4.84, while the frozen path converges to it (4.2e-3 at h = 1e-5; job 2310) -- the same
        path dependence as virtual casing. The mismatch is also genuinely only piecewise smooth: Redl's trapped fraction
        takes hard grid max/min of |B| (vmex.core.bootstrap.compute_trapped_fraction), and a switch of the min|B| grid
        point changed the slope 3x within t = 2.5e-5 (job 2309).
        """
        import jax
        import jax.numpy as jnp
        from vmex.core import implicit as imp
        rng = np.random.default_rng(7)
        direction = np.where(mask, rng.standard_normal(self.x0.size), 0.0) * self.scale
        direction /= np.linalg.norm(direction)
        self.al.x = self.x0
        weights = rng.standard_normal(np.asarray(self.boot.residuals()["bootstrap"]).size)
        adj = float(self._adjoint("bootstrap", weights) @ direction)
        problem = self.plasma.problem
        xp0 = np.asarray(self.plasma.local_full_x, dtype=float)
        dp = direction[self.plasma_mask]

        def params_at(x):
            return imp.params_from_input(problem.input_from_x(np.asarray(x)))
        tangent = jax.tree_util.tree_map(lambda a, b: (jnp.asarray(a) - jnp.asarray(b)) / (2 * h),
                                         params_at(xp0 + h * dp), params_at(xp0 - h * dp))
        rows = self.boot_rows
        frozen = float(imp.frozen_path_directional_fd(
            params_at(xp0), problem.metadata["config"], lambda st, rt: jnp.dot(jnp.asarray(weights), rows(st, rt)),
            tangent, h=h)[0])
        rel = abs(adj - frozen) / abs(frozen)
        print(f"[bootstrap vs frozen path] adjoint {adj:+.8e} frozen {frozen:+.8e} rel {rel:.2e}")
        self.assertLess(rel, tol)

    def test_bootstrap_current_gradient(self):
        self._check_frozen(self._current_mask(), 1e-5, 1e-2)

    def test_bootstrap_boundary_gradient(self):
        # 4.2e-3 run alone (job 2310), 2.2e-2 after the other tests moved VMEX's warm-start state (job 2311): the
        # frozen-path FD of this grid-point quantity is itself only good to ~1-2 %
        self._check_frozen(self.plasma_mask & ~self._current_mask(), 1e-5, 5e-2)

    def test_one_adjoint_per_gradient(self):
        self.al.x = self.x0
        self.al.J()
        before = self.plasma.n_backward
        self.al.dJ()
        self.assertEqual(self.plasma.n_backward - before, 1)


if __name__ == "__main__":
    unittest.main()
