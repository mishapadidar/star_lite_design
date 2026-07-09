import numpy as np

from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt.objectives import forward_backward
from simsopt.geo.surfaceobjectives import boozer_surface_dexactresidual_dcoils_dcurrents_vjp

__all__ = ['NonQuasiSymmetricRatio']


class NonQuasiSymmetricRatio(Optimizable):
    r"""
    This objective decomposes the field magnitude :math:`B(\varphi,\theta)` into quasisymmetric and
    non-quasisymmetric components.  For quasi-axisymmetry, we compute

    .. math::
        B_{\text{QS}} &= \frac{\int_0^1 B \|\mathbf n\| ~d\varphi}{\int_0^1 \|\mathbf n\| ~d\varphi} \\
        B_{\text{non-QS}} &= B - B_{\text{QS}}

    where :math:`B = \| \mathbf B(\varphi,\theta) \|_2`.
    For quasi-poloidal symmetry, an analagous formula is used, but the integral is computed in the :math:`\theta` direction.
    The objective computed by this penalty is

    .. math::
        J &= \frac{\int_{\Gamma_{s}} B_{\text{non-QS}}^2~dS}{\int_{\Gamma_{s}} B_{\text{QS}}^2~dS} \\

    When :math:`J` is zero, then there is perfect QS on the given boozer surface. The ratio of the QS and non-QS components
    of the field is returned to avoid dependence on the magnitude of the field strength.  Note that this penalty is computed
    on an auxilliary surface with quadrature points that are different from those on the input Boozer surface.  This is to allow
    for a spectrally accurate evaluation of the above integrals. Note that if boozer_surface.surface.stellsym == True,
    computing this term on the half-period with shifted quadrature points is ~not~ equivalent to computing on the full-period
    with unshifted points.  This is why we compute on an auxilliary surface with quadrature points on the full period.

    Args:
        boozer_surface: input boozer surface on which the penalty term is evaluated,
        biotsavart: biotsavart object (not necessarily the same as the one used on the Boozer surface).
        sDIM: integer that determines the resolution of the quadrature points placed on the auxilliary surface.
        quasi_poloidal: `False` for quasiaxisymmetry and `True` for quasipoloidal symmetry
    """

    def __init__(self, boozer_surface, bs, sDIM=20, quasi='QA', sign='+'):
        # only BoozerExact surfaces work for now
        assert boozer_surface.res['type'] == 'exact'
        # only SurfaceXYZTensorFourier for now
        assert type(boozer_surface.surface) is SurfaceXYZTensorFourier
        assert (quasi == 'QA') or (quasi == 'QH') or (quasi == 'QP')

        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface

        surface = in_surface
        phis = np.linspace(0, 1/in_surface.nfp, 2*sDIM+1, endpoint=False)
        thetas = np.linspace(0, 1., 2*sDIM+1, endpoint=False)
        surface = SurfaceXYZTensorFourier(mpol=in_surface.mpol, ntor=in_surface.ntor, stellsym=in_surface.stellsym, nfp=in_surface.nfp, quadpoints_phi=phis, quadpoints_theta=thetas, dofs=in_surface.dofs)

        self.quasi=quasi
        def make_QA_matrix(in_nphi, in_ntheta):
            idx = np.arange(in_nphi)
            jdx = np.arange(in_ntheta)
            idx, jdx = np.meshgrid(idx, jdx, indexing='ij')
            return idx, jdx

        def make_QP_matrix(in_nphi, in_ntheta):
            idx = np.arange(in_nphi)
            jdx = np.arange(in_ntheta)
            idx, jdx = np.meshgrid(idx, jdx, indexing='ij')
            return jdx, idx

        def make_QH_matrix(in_nphi, in_ntheta):
            idx = np.arange(in_nphi)
            jdx = np.arange(in_ntheta)
            idx, jdx = np.meshgrid(idx, jdx, indexing='ij')
            idx, jdx = np.mod(idx-jdx, in_nphi), np.mod(idx+jdx, in_ntheta)

            if sign == '-':
                idx=idx.T
                jdx=jdx.T
            return idx, jdx

        if quasi == 'QH':
            self.idx, self.jdx = make_QH_matrix(phis.size, thetas.size)
        elif quasi == 'QP':
            self.idx, self.jdx = make_QP_matrix(phis.size, thetas.size)
        else:
            self.idx, self.jdx = make_QA_matrix(phis.size, thetas.size)

        self.in_surface = in_surface
        self.surface = surface
        self.biotsavart = bs
        self.recompute_bell()


    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def J(self):
        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        if self._dJ is None:
            self.compute()
        return self._dJ

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.solve_residual_equation_exactly_newton(tol=1e-13, maxiter=20, iota=res['iota'], G=res['G'])
        idx = self.idx
        jdx = self.jdx

        pts = self.surface.gamma()[idx, jdx, :]
        self.biotsavart.set_points(pts.reshape((-1, 3)))

        # compute J
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)

        nor = surface.normal()[idx, jdx, :]
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
        B_QS = B_QS[None, :]
        B_nonQS = modB - B_QS
        self._J = np.mean(dS * B_nonQS**2) / np.mean(dS * B_QS**2)

        booz_surf = self.boozer_surface
        iota = booz_surf.res['iota']
        G = booz_surf.res['G']
        P, L, U = booz_surf.res['PLU']
        dconstraint_dcoils_vjp = boozer_surface_dexactresidual_dcoils_dcurrents_vjp

        dJ_by_dB = self.dJ_by_dB().reshape((-1, 3))
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # tack on dJ_diota = dJ_dG = 0 to the end of dJ_ds
        dJ_ds = np.concatenate((self.dJ_by_dsurfacecoefficients(), [0., 0.]))
        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
        self._dJ = dJ_by_dcoils-adj_times_dg_dcoil

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        idx = self.idx
        jdx = self.jdx

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))

        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        nor = surface.normal()[idx, jdx, :]
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)

        denom = np.mean(dS, axis=0)
        B_QS = np.mean(modB * dS, axis=0) / denom

        B_QS = B_QS[None, :]
        B_nonQS = modB - B_QS

        dmodB_dB = B / modB[..., None]
        dnum_by_dB = B_nonQS[..., None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta)  # d J_nonQS / dB_ijk
        ddenom_by_dB = B_QS[..., None] * dmodB_dB * dS[:, :, None] / (nphi * ntheta)  # dJ_QS/dB_ijk
        num = 0.5*np.mean(dS * B_nonQS**2)
        denom = 0.5*np.mean(dS * B_QS**2)
        return (denom * dnum_by_dB - num * ddenom_by_dB) / denom**2

    def dJ_by_dsurfacecoefficients(self):
        """
        Return the partial derivative of the objective with respect to the surface coefficients
        """
        surface = self.surface
        nphi = surface.quadpoints_phi.size
        ntheta = surface.quadpoints_theta.size
        idx = self.idx
        jdx = self.jdx

        B = self.biotsavart.B()
        B = B.reshape((nphi, ntheta, 3))
        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)

        nor = surface.normal()[idx, jdx, :]
        dnor_dc = surface.dnormal_by_dcoeff()[idx, jdx, :, :]
        dS = np.sqrt(nor[:, :, 0]**2 + nor[:, :, 1]**2 + nor[:, :, 2]**2)
        dS_dc = (nor[:, :, 0, None]*dnor_dc[:, :, 0, :] + nor[:, :, 1, None]*dnor_dc[:, :, 1, :] + nor[:, :, 2, None]*dnor_dc[:, :, 2, :])/dS[:, :, None]

        B_QS = np.mean(modB * dS, axis=0) / np.mean(dS, axis=0)
        B_QS = B_QS[None, :]

        B_nonQS = modB - B_QS

        dB_by_dX = self.biotsavart.dB_by_dX().reshape((nphi, ntheta, 3, 3))
        dx_dc = surface.dgamma_by_dcoeff()[idx, jdx, :, :]
        dB_dc = np.einsum('ijkl,ijkm->ijlm', dB_by_dX, dx_dc, optimize=True)

        modB = np.sqrt(B[:, :, 0]**2 + B[:, :, 1]**2 + B[:, :, 2]**2)
        dmodB_dc = (B[:, :, 0, None] * dB_dc[:, :, 0, :] + B[:, :, 1, None] * dB_dc[:, :, 1, :] + B[:, :, 2, None] * dB_dc[:, :, 2, :])/modB[:, :, None]

        num = np.mean(modB * dS, axis=0)
        denom = np.mean(dS, axis=0)
        dnum_dc = np.mean(dmodB_dc * dS[..., None] + modB[..., None] * dS_dc, axis=0)
        ddenom_dc = np.mean(dS_dc, axis=0)
        B_QS_dc = (dnum_dc * denom[:, None] - ddenom_dc * num[:, None])/denom[:, None]**2

        B_QS_dc = B_QS_dc[None, :, :]
        B_nonQS_dc = dmodB_dc - B_QS_dc

        num = 0.5*np.mean(dS * B_nonQS**2)
        denom = 0.5*np.mean(dS * B_QS**2)
        dnum_by_dc = np.mean(0.5*dS_dc * B_nonQS[..., None]**2 + dS[..., None] * B_nonQS[..., None] * B_nonQS_dc, axis=(0, 1))
        ddenom_by_dc = np.mean(0.5*dS_dc * B_QS[..., None]**2 + dS[..., None] * B_QS[..., None] * B_QS_dc, axis=(0, 1))
        dJ_by_dc = (denom * dnum_by_dc - num * ddenom_by_dc) / denom**2
        return dJ_by_dc
