import numpy as np
import unittest
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.finite_difference import finite_difference
from simsopt.geo import CurveXYZFourierSymmetries, CurveLength
from simsopt._core import load
from simsopt.field import BiotSavart

class TestModBOnFieldline(unittest.TestCase):

    def test_derivative(self):
        """Test the dJ attribute of ModBOnFieldLine class with finite difference."""
        data = load(f"./designs/designB_after_scaled.json")
        bsurf = data[0][0]
        coils = bsurf.biotsavart.coils
        axis_RZ = data[2][0] # magnetic axis CurveRZFouriers

        # compute the magnetic axis
        stellsym=True
        nfp = 2
        order=16
        tmp = CurveXYZFourierSymmetries(axis_RZ.quadpoints, order, nfp, stellsym)
        tmp.least_squares_fit(axis_RZ.gamma())
        quadpoints = np.linspace(0, 1/nfp, 2*order+1, endpoint=False)
        axis = CurveXYZFourierSymmetries(quadpoints, order, nfp, stellsym)
        axis.x = tmp.x
        axis_fl = PeriodicFieldLine(BiotSavart(coils), axis)
        axis_fl.run_code(CurveLength(axis_fl.curve).J())

        obj = ModBOnFieldLine(axis_fl, BiotSavart(coils))

        # determine dofs
        for c in coils:
            c.unfix_all()
        x0 = obj.x
        
        # check derivative with finite difference
        J = obj.J()
        dJ = obj.dJ()

        def fun(x):
            obj.x = x
            return obj.J()
        dJ_fd = finite_difference(fun, obj.x, eps=1e-4)

        err = np.max(np.abs(dJ - dJ_fd))
        print(err)
        self.assertTrue(err < 1e-5), f"Derivative check failed: {err}"

if __name__ == '__main__':
    unittest.main()