import os
import unittest

import numpy as np
from simsopt._core import load
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import create_equally_spaced_curves
from star_lite_design.utils.vmex_combined_stage import coil_stellsym_error, fix_self_symmetric_coil_parity

DESIGN_A = os.path.join(os.path.dirname(__file__), "..", "..", "convert", "designA_after_scaled.json")


def base_curves_of(coils):
    base, seen = [], set()
    for c in coils:
        for o in c.curve.unique_dof_lineage:
            if o.local_dof_size > 0 and id(o) not in seen and "Curve" in type(o).__name__:
                seen.add(id(o))
                base.append(o)
    return base


class CoilStellsymTests(unittest.TestCase):

    def test_designA_self_symmetric_coil_is_pinned(self):
        """designA's phi = +-90 deg coils are their own stellarator image: exactly their 50 odd-parity coefficients
        are fixed, and afterwards NO change of the remaining free dofs can break the coil set's symmetry."""
        coils = load(DESIGN_A)[0][0].biotsavart.coils
        base = base_curves_of(coils)
        self.assertLess(coil_stellsym_error(coils), 1e-12)
        free_before = [b.local_dof_size for b in base]
        n_fixed, notes = fix_self_symmetric_coil_parity(base)
        self.assertEqual(n_fixed, 50)
        self.assertEqual(sorted(free_before[i] - b.local_dof_size for i, b in enumerate(base)), [0, 50])
        pinned = next(b for b in base if b.local_dof_size < b.local_full_dof_size)
        self.assertTrue(all(n.startswith(("xs(", "yc(", "zc(")) for n in pinned.local_full_dof_names
                            if not pinned.is_free(n)))
        rng = np.random.default_rng(0)
        for b in base:
            b.x = b.x + 1e-3 * rng.standard_normal(b.x.size)
        self.assertLess(coil_stellsym_error(coils), 1e-12)
        # idempotent
        self.assertEqual(fix_self_symmetric_coil_parity(base)[0], 0)

    def test_unfixed_coil_drifts_off_symmetry(self):
        """Without the fix, a perturbation of the free dofs breaks the symmetry, and the error measures it."""
        coils = load(DESIGN_A)[0][0].biotsavart.coils
        base = base_curves_of(coils)
        rng = np.random.default_rng(0)
        for b in base:
            b.x = b.x + 1e-3 * rng.standard_normal(b.x.size)
        self.assertGreater(coil_stellsym_error(coils), 1e-4)
        # a curve that is no longer its own image is left alone
        self.assertEqual(fix_self_symmetric_coil_parity(base)[0], 0)

    def test_circular_coils_untouched(self):
        """Circular coils from create_equally_spaced_curves sit between symmetry planes; their many zero coefficients
        must stay free."""
        base = create_equally_spaced_curves(4, 2, stellsym=True, R0=1.0, R1=0.5, order=6)
        coils = coils_via_symmetries(base, [Current(1.0) for _ in base], 2, True)
        self.assertLess(coil_stellsym_error(coils), 1e-12)
        sizes = [b.local_dof_size for b in base]
        self.assertEqual(fix_self_symmetric_coil_parity(base)[0], 0)
        self.assertEqual([b.local_dof_size for b in base], sizes)


if __name__ == "__main__":
    unittest.main()
