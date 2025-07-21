import numpy as np
import unittest
from star_lite_design.utils.curve_vessel_distance import CurveVesselDistance
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.finite_difference import finite_difference, taylor_test
from simsopt.geo import CurveXYZFourierSymmetries, CurveLength
from simsopt._core import load
from simsopt.field import BiotSavart
import pandas as pd

class TestCurveVesselDistance(unittest.TestCase):
    def test_taylor(self):
        """Test the dJ attribute of CurveVesselDistance class with finite difference."""
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

        # coil-to-vessel distance
        df = pd.read_csv("./designs/sheetmetal_chamber.csv")
        X_vessel = df.values
        # this penalty has a min(something(dofs), 0)**2 in it.  This is not differentiable globally.
        # we set the CV_THRESHOLD to be really large here so that the min is essentially not active
        CV_THRESHOLD = 1e1
        obj = CurveVesselDistance([c.curve for c in coils], X_vessel, CV_THRESHOLD)

        # determine dofs
        for c in coils:
            c.unfix_all()
        x0 = obj.x
        
        def fun(x):
            obj.x = x
            return obj.J(), obj.dJ()
        
        min_rel_error = taylor_test(fun, x0, order=6)
        assert min_rel_error < 1e-10 # 10 digits in the gradient

if __name__ == '__main__':
    unittest.main()
