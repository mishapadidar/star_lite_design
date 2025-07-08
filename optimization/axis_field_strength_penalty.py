from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
import numpy as np

class AxisFieldStrengthPenalty(Optimizable):
    
    def __init__(self, bs, xyz, B_target):
        """A penalty on the mean field strength on the axis
            J = (B_mean - B_target)^2.

        Args:
            bs (BiotSavart):
            xyz (array): (n, 3) array of points
            B_target (float): target mean field strength on the axis
        """
        self.bs = bs
        self.xyz = xyz
        self.B_target = B_target

        super().__init__(depends_on=[bs])

    def J(self):
        """
        Compute the penalty for the field strength on the axis.
        """
        # compute B on axis
        self.bs.set_points(self.xyz)
        B_axis = self.bs.B()
        B_norm = np.linalg.norm(B_axis, axis=1)
        mean_B_norm = np.mean(B_norm)
        
        # return the penalty
        return (mean_B_norm - self.B_target)**2

    @derivative_dec
    def dJ(self):
        """
        Compute the derivative of the penalty for the field strength on the axis.
        """
        # compute B on axis
        self.bs.set_points(self.xyz)
        B_axis = self.bs.B()
        B_norm = np.linalg.norm(B_axis, axis=1)
        mean_B_norm = np.mean(B_norm)

        # val = (2 * (B_norm - self.B_target) / len(B_norm) / B_norm)[:, None] * B_axis
        val = (2 * (mean_B_norm - self.B_target) / len(B_norm) / B_norm)[:, None] * B_axis
        dJ_by_dbs = self.bs.B_vjp(val)
        return dJ_by_dbs
    

def test_axis_field_strength_penalty():
    """ Test the AxisFieldStrengthPenalty class """
    from simsopt._core import load
    data = load(f"../designs/designB_after_scaled.json")
    boozer_surfaces = data[0] # BoozerSurfaces
    iota_Gs = data[1] # (iota, G) pairs
    axis_curves = data[2] # magnetic axis CurveRZFouriers
    
    bs = boozer_surfaces[0].biotsavart
    xyz = axis_curves[0].gamma()

    pen = AxisFieldStrengthPenalty(bs, xyz, 0.0875)

    print("J:", pen.J())
    dJ = pen.dJ()

    from star_lite_design.utils.finite_difference import finite_difference
    def fun(x):
        bs.x = x
        return pen.J()
    x0 = bs.x
    print('Taylor test')
    for eps in [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]:
        dJ_fd = finite_difference(fun, x0, eps=eps)
        err = dJ - dJ_fd
        print(f'eps {eps}: err = {np.max(np.abs(err))}')

if __name__ == "__main__":
    test_axis_field_strength_penalty()