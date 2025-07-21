import numpy as np
import unittest
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine, field_line_residual
from star_lite_design.utils.finite_difference import finite_difference, taylor_test
from simsopt.geo import CurveXYZFourierSymmetries, CurveLength
from simsopt._core import load
from simsopt.field import BiotSavart

class TestPeriodicFieldline(unittest.TestCase):

    def test_magnetic_axis(self):
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
        axis = CurveXYZFourierSymmetries(quadpoints, order, nfp, stellsym, options={})
        axis.x = tmp.x
        axis_fl1 = PeriodicFieldLine(BiotSavart(coils), axis, options={'verbose':True})
        res = axis_fl1.run_code(CurveLength(axis_fl1.curve).J())
        self.assertTrue(res['success'])
        dofs = axis_fl1.curve.x.copy()
        xc1 = dofs[:order+1]
        ys1 = dofs[order+1:2*order+1]
        zs1 = dofs[2*order+1:]

        stellsym=False
        nfp = 2
        order=16
        tmp = CurveXYZFourierSymmetries(axis_RZ.quadpoints, order, nfp, stellsym)
        tmp.least_squares_fit(axis_RZ.gamma())
        quadpoints = np.linspace(0, 1/nfp, 2*order+1, endpoint=False)
        axis = CurveXYZFourierSymmetries(quadpoints, order, nfp, stellsym)
        axis.x = tmp.x
        axis_fl2 = PeriodicFieldLine(BiotSavart(coils), axis, options={'verbose':True})
        res = axis_fl2.run_code(CurveLength(axis_fl2.curve).J())
        self.assertTrue(res['success'])
        
        dofs = axis_fl2.curve.x.copy()
        xc2 = dofs[0: order+1]
        xs2 = dofs[order+1: 2*order+1]
        yc2 = dofs[2*order+1: 3*order+2]
        ys2 = dofs[3*order+2: 4*order+2]
        zc2 = dofs[4*order+2: 5*order+3]
        zs2 = dofs[5*order+3:]

        self.assertTrue(np.linalg.norm(xc1-xc2) < 1e-15), f"xc check failed"
        self.assertTrue(np.linalg.norm(ys1-ys2) < 1e-15), f"ys check failed"
        self.assertTrue(np.linalg.norm(zs1-zs2) < 1e-15), f"zs check failed"
        
        self.assertTrue(np.linalg.norm(xs2) < 1e-15), f"xs check failed"
        self.assertTrue(np.linalg.norm(yc2) < 1e-15), f"yc check failed"
        self.assertTrue(np.linalg.norm(zc2) < 1e-15), f"zc check failed"

    def test_x_point(self):
        """Test the dJ attribute of ModBOnFieldLine class with finite difference."""
        data = load(f"./designs/designB_after_scaled.json")
        bsurf = data[0][0]
        coils = bsurf.biotsavart.coils
        axis_RZ = data[3][0] # magnetic axis CurveRZFouriers


        stellsym=False
        nfp = 2
        order=16
        tmp = CurveXYZFourierSymmetries(axis_RZ.quadpoints, order, nfp, stellsym)
        tmp.least_squares_fit(axis_RZ.gamma())
        quadpoints = np.linspace(0, 1/nfp, 2*order+1, endpoint=False)
        axis = CurveXYZFourierSymmetries(quadpoints, order, nfp, stellsym)
        axis.x = tmp.x
        axis_fl = PeriodicFieldLine(BiotSavart(coils), axis, options={'verbose':True})
        res = axis_fl.run_code(CurveLength(axis_fl.curve).J())
        self.assertTrue(res['success'])

        dofs = axis_fl.curve.x.copy()
        xc = dofs[0: order+1]
        xs = dofs[order+1: 2*order+1]
        yc = dofs[2*order+1: 3*order+2]
        ys = dofs[3*order+2: 4*order+2]
        zc = dofs[4*order+2: 5*order+3]
        zs = dofs[5*order+3:]
        
        self.assertTrue(np.linalg.norm(xs) > 1e-10)
        self.assertTrue(np.linalg.norm(yc) > 1e-10)
        self.assertTrue(np.linalg.norm(zc) > 1e-10)

    def test_stellsym(self):
        """Test that in the stellarator symmetric case nfp=1
        
        rx(-tk) = -rx(-tk)
        ry(-tk) = ry(tk)
        rz(-tk) = rz(tk)

        for nfp=2 you have to apply a rotation to the residual, but the
        same result holds.
        """

        data = load(f"./designs/designB_after_scaled.json")
        bsurf = data[0][0]
        field = bsurf.biotsavart
        axis_RZ = data[2][0] # magnetic axis CurveRZFouriers
        
        stellsym=True
        nfp = 2
        order=16
        tmp = CurveXYZFourierSymmetries(axis_RZ.quadpoints, order, nfp, stellsym)
        tmp.least_squares_fit(axis_RZ.gamma())
        quadpoints = np.linspace(0, 1/nfp, 2*order+1, endpoint=False)
        axis = CurveXYZFourierSymmetries(quadpoints, order, nfp, stellsym)
        axis.x = tmp.x
        out = field_line_residual(axis, CurveLength(axis).J(), field)
        residual = out[0].reshape((-1, 3))
        angle = 2*np.pi/nfp
        R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        r1 = residual[1]
        r2 = R@residual[-1]
        self.assertTrue(np.abs(r1[0]+r2[0]) < 1e-15)
        self.assertTrue(np.linalg.norm(r1[1:]-r2[1:]) < 1e-15)

if __name__ == '__main__':
    unittest.main()
