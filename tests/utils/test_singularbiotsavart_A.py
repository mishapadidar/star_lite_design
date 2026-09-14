import os
import unittest
import numpy as np
from star_lite_design.utils.singularbiotsavart import SingularBiotSavart
from simsopt._core import load
from simsopt.geo import ToroidalFlux

# Regression test for the campaign VMEC crash "RuntimeError: _A_impl was not
# implemented": archives whose BoozerSurface carries a SingularBiotSavart died
# in ToroidalFlux(surf, bs).J() because the vector potential was missing.
ARCHIVE = (
    "/data/common/raffael.wendlinger/data/rusty/star_lite/00/"
    "margin=0p06_well=-100.0_Z=0_onvessel=0_distance=0_configID=2_"
    "vesselID=3_mono=1_null=SN_num_aux=5_AR=0_attempt=0/"
    "design_unpolished_final_1148982759.json"
)


class TestSingularBiotSavartVectorPotential(unittest.TestCase):

    def test_A_impl_defined(self):
        """The MagneticField _A_impl / _dA_by_dX_impl hooks must be overridden,
        otherwise the pybind base class raises at every .A() call."""
        self.assertTrue(hasattr(SingularBiotSavart, '_A_impl'))
        self.assertTrue(hasattr(SingularBiotSavart, '_dA_by_dX_impl'))

    @unittest.skipUnless(os.path.exists(ARCHIVE),
                         "campaign archive with SingularBiotSavart not available")
    def test_toroidal_flux_on_singular_archive(self):
        data = load(ARCHIVE)
        bsurf = data[0][0] if isinstance(data[0], (list, tuple)) else data[0]
        bs = bsurf.biotsavart
        surf = bsurf.surface
        self.assertIsInstance(bs, SingularBiotSavart)

        tf = ToroidalFlux(surf, bs).J()
        self.assertTrue(np.isfinite(tf))
        self.assertGreater(abs(tf), 0.0)

        # A is the superposition of the modular and aux fields
        pts = surf.gamma().reshape(-1, 3)[:32].copy()
        bs.set_points(pts)
        A = bs.A()
        self.assertEqual(A.shape, (32, 3))
        self.assertTrue(np.all(np.isfinite(A)))
        bs.modular.set_points(pts)
        bs._aux_bs.set_points(pts)
        np.testing.assert_allclose(A, bs.modular.A() + bs._aux_bs.A(),
                                   rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
