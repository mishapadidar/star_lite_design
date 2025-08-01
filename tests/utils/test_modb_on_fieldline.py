import numpy as np
import unittest
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.finite_difference import finite_difference, taylor_test
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
        res = axis_fl.run_code(CurveLength(axis_fl.curve).J())
        self.assertTrue(res['success']), f"curve convergence failed"
        
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

    def test_taylor(self):
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
        res = axis_fl.run_code(CurveLength(axis_fl.curve).J())
        self.assertTrue(res['success']), f"curve convergence failed"

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
            return obj.J(), obj.dJ()
        
        min_rel_error = taylor_test(fun, x0, order=6)
        self.assertTrue(min_rel_error < 1e-10) # 10 digits in the gradient

    def test_modB_on_axis(self):
        """
        Test that modB computed by ModBOnFieldLine is actually the mean modB on
        the fieldline.
        """
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
        res = axis_fl.run_code(CurveLength(axis_fl.curve).J())
        self.assertTrue(res['success']), f"curve convergence failed"

        obj = ModBOnFieldLine(axis_fl, BiotSavart(coils))
        bs = BiotSavart(coils)
        bs.set_points(axis.gamma())

        modB1 = obj.J()
        modB2 = np.mean(bs.AbsB())
        
        self.assertTrue(np.abs(modB1-modB2)/modB2 < 1e-10), f'modB on axis is not what it should be'

if __name__ == '__main__':
    unittest.main()
