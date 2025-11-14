import numpy as np
from simsopt.field import BiotSavart



class Inductance:
    """
    Compute the inductance matrix for a set of coils. The accuracy of the computation
    depends on the number of quadrature points used to represent each coil: 
    at least 128 points is recommended.

    Args:
        coils (list of Coil): List of Coil objects representing the coils.
        a (float): The minor radius of a coil.
    """

    def __init__(self, coils, a):
        self.coils = coils
        self.a = a

    def mutual_inductance(self, coil1, coil2):
        """
        Compute the mutual inductance between two distinct coils.

        This computation assumes that coils are far enough from one another that they 
        can be modeled as filaments, ignoring their cross-sectional area. 
        The inductance is defined as
            M12 = Phi / I1
        where I1 is the current in coil1 and Phi is the magnetic flux through coil 2.
        The flux is,
            Phi = \int B1 * da2 = \oint A1 * dl2
        where A1 is the magnetic vector potential due to coil1.

        Args:
            coil1 (Coil): The first coil.
            coil2 (Coil): The second coil.

        Returns:
            float: The mutual inductance between the two coils.
        """
        bs = BiotSavart([coil1])

        xyz = coil2.curve.gamma() # (n, 3)
        dl = coil2.curve.gammadash()  # (n, 3)
        bs.set_points(xyz)
        A = bs.A()  # (n, 3)

        A_dot_dl = np.sum(A * dl, axis=1)
        Phi = np.mean(A_dot_dl)

        M = Phi / coil1.current.get_value()

        return M
    
    def self_inductance(self, coil):
        """Compute the self-inductance of a coil.

        Args:
            coil (Coil): The coil for which to compute self-inductance.
        
        Returns:
            float: The self-inductance of the coil.
        """
        return self.self_inductance_unmodified(coil)

    def self_inductance_modified(self, coil):
        """
        Compute the self-inductance of a coil using eq (27) from [1].

        [1] Hurwitz, Efficient Calculation of the Self-Magnetic Field, Self-Force, and Self-Inductance for Electromagnetic Coils

        Args:
            coil (Coil): The coil for which to compute self-inductance.
        
        Returns:
            float: The self-inductance of the coil.
        """
        raise NotImplementedError("Self-inductance calculation is incorrect.")
        mu0 = 4e-7 * np.pi
        coeff = mu0 / (4 * np.pi)
        xyz = coil.curve.gamma()  # (n, 3)
        gd = coil.curve.gammadash()  # (n, 3)
        dl = np.linalg.norm(gd, axis=1)  # (n,)

        # first integral
        first = coeff  * np.mean(dl * (2 * np.log(8 * dl / self.a) + 1/2))

        # second integral
        term = self.a**2 / np.sqrt(np.e)
        dr = np.linalg.norm(xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :], axis=2)  # (n, n)
        phi = coil.curve.quadpoints # (n,)
        dphi = 2*np.pi * (phi[:, np.newaxis] - phi[np.newaxis, :])  # (n, n)
        integrand1 = gd @ gd.T / np.sqrt(dr**2 + term) # (n, n)
        integrand2 = dl**2 / np.sqrt(2 * (1 - np.cos(dphi))* dl**2  + term)  # (n, n)
        integrand = integrand1 - integrand2
        second = coeff * np.mean(np.mean(integrand, axis=1))

        L = first + second
        return L

    def self_inductance_unmodified(self, coil):
        """
        Compute the self-inductance of a coil using eq (10) from [1].

        [1] Hurwitz, Efficient Calculation of the Self-Magnetic Field, Self-Force, and Self-Inductance for Electromagnetic Coils

        Args:
            coil (Coil): The coil for which to compute self-inductance.
        
        Returns:
            float: The self-inductance of the coil.
        """
        mu0 = 4e-7 * np.pi
        coeff = mu0 / (4 * np.pi)
        xyz = coil.curve.gamma()  # (n, 3)
        gd = coil.curve.gammadash()  # (n, 3)
        term = self.a**2 / np.sqrt(np.e)
        dot = gd @ gd.T  # (n, n)
        dr = np.linalg.norm(xyz[:, np.newaxis, :] - xyz[np.newaxis, :], axis=2)  # (n, n)
        integrand = dot / np.sqrt(dr**2 + term)  # (n, n)
        L = coeff * np.mean(np.mean(integrand, axis=1))
        return L

    def calculate(self):
        """
        Calculate the inductance matrix for the set of coils.

        Returns:
            np.ndarray: The inductance matrix.
        """
        n = len(self.coils)
        L = np.zeros((n, n))

        for i in range(n):
            L[i, i] = self.self_inductance(self.coils[i])
            for j in range(i + 1, n):
                M = self.mutual_inductance(self.coils[i], self.coils[j])
                L[i, j] = M
                L[j, i] = M

        return L


def test_inductance():

    from simsopt.field import Coil, Current
    from simsopt.geo import CurveXYZFourier

    """
    Test the self-inductance of a circular loop against the analytic formula.
    """
    curve1 = CurveXYZFourier(256, order=1)
    R = 1.283
    I1 = 3.76
    a = 0.05  # minor radius
    curve1.set("xc(1)", R)
    curve1.set("ys(1)", R)
    current1 = Current(I1)
    coil1 = Coil(curve1, current1)

    # analytic formula for self-inductance of circular loop
    L_circular = 4e-7 * np.pi * R * (np.log(8 * R / a) - 7/4)
    print(f"Analytic self-inductance of circular loop: {L_circular}")

    coils = [coil1]

    ind = Inductance(coils, a)
    L = ind.self_inductance(coil1)
    print(f"Self-inductance (modified): {L} H")

    """Test the mutual inductance converges to the selfinductance as the coils are brought together."""
    curve1 = CurveXYZFourier(512, order=3)
    curve1.set("xc(1)", R)
    curve1.set("ys(1)", R)
    curve1.set("xc(2)", 0.324)
    curve1.set("ys(2)", 0.0234)
    curve1.set('xc(3)', 0.0123)

    curve2 = CurveXYZFourier(512, order=3)
    curve2.x = curve1.x

    # from simsopt.geo import plot
    # curve2.x = 0.99 * curve1.x
    # plot([curve1, curve2], show=True)

    I = 1.234
    current1 = Current(I)
    current2 = Current(I)
    coil1 = Coil(curve1, current1)
    coil2 = Coil(curve2, current2)

    alphas = np.linspace(0.8,0.897,10, endpoint=True)
    for alpha in alphas:
        curve2.x = alpha * curve1.x
        # make sure curves (with finite thickness) do not intersect
        dist = np.min(np.linalg.norm(curve1.gamma()[:, np.newaxis, :] - curve2.gamma()[np.newaxis, :, :], axis=2))
        if dist < 2 * a:
            print(f"Skipping alpha={alpha:.5f} due to intersection (dist={dist})")
            continue


        coils = [coil1, coil2]
        ind = Inductance(coils, a)
        L_matrix = ind.calculate()
        M = L_matrix[0,1]
        L_self = L_matrix[0,0]
        print(f"alpha: {alpha:.5f}, Mutual Inductance: {M} H, Self Inductance: {L_self} H")



if __name__ == "__main__":
    test_inductance()