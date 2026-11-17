#!/usr/bin/env python3
"""boozer_risk_opt.py

THE two-configuration (double) optimization of the designB pipeline: config 1 = X-point with
the aux coils off, config 2 = snowflake with the aux coils. It replaces boozer_double_opt.py
(archive/old_scripts), whose docstring follows below. Two possible inputs, told apart from
the design json itself:

  * a boozer_all design (design_opt_final_<ID>.json, one configuration): the aux coils are
    placed (--num-aux), the snowflake is made by the singular polish and the current
    continuation, then both configurations are optimized. Without --ec-max this is exactly
    what boozer_double_opt.py did;
  * a FINISHED double (design_double_final_<TAG>.json): both configurations, the aux coils
    and the weights (the escalated ones of the double) are loaded from it and its yaml, and
    the optimization goes on.

In both cases the outputs keep the names of a double (design_double_final_<TAG>.json,
config1/, config2/, summary.txt), written next to the input or in --out-dir.

--ec-max adds one more inequality constraint (risk-neutral bound); without it E[C] is only
reported (start lines, tables, summary.txt) and nothing of it enters the objective:

    E[C] <= --ec-max,

The E[C] that the bound applies to is chosen with --ec-section: E[C] in ONE toroidal
section, given by the fraction T in [0, 1) of the full turn of the snowflake field line
(default 0, the section at phi = 0), or the MEAN of E[C] over the sections ('mean'). E[C] in a
section is ExactMeanUnfoldingC with its phi0; the profile over the sections and their mean are
PhiMeanUnfoldingC, transported from that section. Both are always reported (start lines, table,
summary.txt: unfolding_EC = the bounded one, unfolding_EC_section, unfolding_EC_phi_mean / _min /
_max) and checked by --check-gradient.

Here C is the unfolding constant of the snowflake under a Gaussian-process perturbation of
the modular coils (utils/discriminant.py: the two X-points born from the snowflake separate by
C * sqrt(sigma) for a perturbation of amplitude sigma; the aux coils are not perturbed),
measured in the (R, Z) section at phi = 0, and E[.] is the expectation over the Gaussian
process, computed exactly (ExactMeanUnfoldingC: the response of the return map is a Gaussian
2-vector, so the expectation is a one-dimensional integral; no realizations are drawn). The
constraint is a one-sided quadratic penalty with its own escalating weight, and its relative
excess enters the 0.1% gate (summary.txt: unfolding_EC).

The docstring of boozer_double_opt.py follows.

Continue a boozer_all.py optimization as an "all-in-one" optimization of TWO configurations
that share the modular-coil GEOMETRY and the vessel SDF, with their own modular currents
(independent design variables, both started at boozer_all's values):

  config 1  hyperbolic X-point: the boozer_all device itself, auxiliary coils off (aux
            currents = 0), ordinary PeriodicFieldLine X-point, with the penalized constraint
            tr M >= 2 + MONODROMY_THRESHOLD (TangentMap mtype='trace_min'; the threshold is
            boozer_all's --mono-thr, read from the yaml);
  config 2  snowflake: generated here from the same device by the singular polish with
            --num-aux auxiliary coils, exactly as boozer_singular_opt.py does (below).

Everything else is boozer_singular_opt.py, of which this file is a copy: the objective is the
mean non-QS ratio of the two Boozer surfaces, every physics constraint (iota with the yaml
target, major radius, |B|, on-axis ripple, well, plasma-vessel margin, Boozer residual,
currents, forces, field-line constraints) applies to EACH configuration, and the coil
geometry constraints once. Lists (boozer_surfaces, axes, xpoints) hold config 1 then config 2.

Takes as input a design json produced by boozer_all.py
([boozer_surfaces, iota_Gs, axes, xpoints, sdf]).  For the X-point a
SingularPeriodicFieldline is created with --num-aux planar circular
auxiliary coils:

  1) initially ALL aux currents and radii are DEPENDENT (fixed dofs, solved by
     the under-determined pinv Newton polish; z is independent/held);
  2) then the partition is re-drawn: everything is unfixed and only the
     dependent currents are fixed (1 current for the 'trace' system; for
     'identity', 3 in general but 2 when the X-point is stellarator
     symmetric, since M[0,0] == M[1,1] is then redundant), giving a square
     LU Newton polish.  The remaining mu (radii, z, other currents) become
     free dofs, i.e. design variables.

The optimization then bounds ALL aux currents, dependent and independent
alike, by |I| <= AUX_CURRENT_EPS (5e3 A), with one shared escalating weight
AUX_CURRENT_WEIGHT (J_aux_current):
  * dependent currents through the DependentMu optimizable + QuadraticPenalty
    (their value is an implicit function of the design variables, so the
    gradient flows through the adjoint);
  * independent currents (free mu dofs) through MuBound.

It also keeps the aux coils clear of the modular coils: AuxCoilDistance adds a
one-sided quadratic clearance penalty (minimum separation
AUX_COIL_DISTANCE_THRESHOLD, weight AUX_COIL_DISTANCE_WEIGHT escalated after each
BFGS run) between every modular coil and the planar circular aux coils. The aux
coils are mu dofs (not Curves), so the penalty reads the aux radii/Z from fl.mu
and its gradient lands on both the modular-coil dofs and the aux (mu) dofs.
"""
import argparse
import os
import zlib

import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from scipy.optimize import minimize

from simsopt._core import load, save
from simsopt.field import BiotSavart, Coil, Current, RegularizedCoil, coils_via_symmetries
from simsopt.field.force import LpCurveForce
from simsopt.field.selffield import regularization_circ
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import (
    ArclengthVariation,
    BoozerSurface,
    CurveCurveDistance,
    CurveLength,
    CurveXYZFourier,
    CurveXYZFourierSymmetries,
    Volume,
    LpCurveCurvature,
    MajorRadius,
    MeanSquaredCurvature,
    NonQuasiSymmetricRatio,
    Iotas,
    RotatedCurve,
    SurfaceXYZTensorFourier,
    curves_to_vtk,
)
from simsopt.objectives import QuadraticPenalty, Weight

from star_lite_design.utils.boozer_surface_utils import BoozerResidual
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.current_bound import CurrentBound
from star_lite_design.utils.displacement import FieldLineMeanZ
from star_lite_design.utils.magneticwell import MagneticWell
from star_lite_design.utils.modb_on_fieldline import ModBOnFieldLine, ModBRippleOnFieldLine
from star_lite_design.utils.pillpipevessel import RennaissanceSDF, PillPipeSDF, TorusSDF, VesselDistance
from star_lite_design.utils.helicalvessel import HelicalVesselSDF, FOOT_TOL
from star_lite_design.utils.helicalcylindervessel import HelicalCylinderVesselSDF
from star_lite_design.utils.singularperiodicfieldline import (
    SingularPeriodicFieldline, DependentMu, AuxCoilDistance, _mu_names, _CURRENT_SCALE)
from star_lite_design.utils.singularbiotsavart import SingularBiotSavart
from star_lite_design.utils.mubound import MuBound
from star_lite_design.utils.tangent_map import TangentMap, Monodromy, AxisIota, AxisElongation
from star_lite_design.utils.discriminant import (SnowflakeDiscriminant, CoilPerturbationField, DSigma,
                                                 unfolding_solutions)
from star_lite_design.utils.total_field_coil_force import TotalFieldLpCurveForce
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec


class _WeightedPerturbation:
    r"""A perturbation of the modular coils with an ARBITRARY displacement (_d, _dp: the
    displacement of each coil and its theta-derivative), with the part of the interface of a
    CoilPerturbationField that DSigma uses (dB_by_dsigma, dgradB_by_dsigma, dB_by_dsigma_vjp).

    delta_B = dB/dsigma at sigma = 0 is LINEAR in the displacement of the coils, hence so is
    D_sigma: ExactMeanUnfoldingC takes its Jacobian w.r.t. the displacement, and runs its
    adjoint for the displacements it sets here. The field is the formula of
    CoilPerturbationField._Bfield with the displacement as an ARGUMENT of the compiled
    functions, so that changing it does not recompile anything. Built from a list of
    CoilPerturbationField (their unit samples are kept in _d_all, _dp_all; set_weights(c)
    sets the displacement to their combination)."""

    def __init__(self, perturbations):
        import jax
        import jax.numpy as jnp
        self._jnp = jnp
        p0 = perturbations[0]
        self.biotsavart = p0.biotsavart
        self.coils = p0.coils
        self._coil_data = p0._coil_data   # live (gamma, gamma', current) of the modular coils
        # (realization, coil, quadrature point, 3): unit displacements and their theta-derivatives
        self._d_all = np.array([[np.asarray(us._sample[0]) for us in p._unit_samples] for p in perturbations])
        self._dp_all = np.array([[np.asarray(us._sample[1]) for us in p._unit_samples] for p in perturbations])
        nq = self._d_all.shape[2]
        mu0_4pi = 1.0e-7

        def _Bfield(points, gammas, gammadashs, currents, d, dprime, sigma):
            g = gammas + sigma * d
            gd = gammadashs + sigma * dprime
            R = points[:, None, None, :] - g[None, :, :, :]
            invR3 = jnp.sum(R * R, axis=-1) ** (-1.5)
            cross = jnp.cross(gd[None, :, :, :], R, axis=-1)
            return mu0_4pi * jnp.sum(currents[None, :, None, None] * cross * invR3[..., None], axis=(1, 2)) / nq

        def _gradB(points, gammas, gammadashs, currents, d, dprime, sigma):
            Jac = jax.vmap(lambda p: jax.jacfwd(
                lambda q: _Bfield(q[None, :], gammas, gammadashs, currents, d, dprime, sigma)[0])(p))(points)
            return jnp.transpose(Jac, (0, 2, 1))          # SIMSOPT convention [i,k,j] = dB_j/dx_k

        self._dBdsigma = jax.jit(lambda p, g, gd, I, d, dp: jax.jacfwd(lambda s: _Bfield(p, g, gd, I, d, dp, s))(0.0))
        self._dgradBdsigma = jax.jit(lambda p, g, gd, I, d, dp: jax.jacfwd(lambda s: _gradB(p, g, gd, I, d, dp, s))(0.0))
        self._dBdsigma_grad = jax.jit(jax.grad(
            lambda p, g, gd, I, d, dp, v: jnp.sum(v * jax.jacfwd(lambda s: _Bfield(p, g, gd, I, d, dp, s))(0.0)),
            argnums=(1, 2, 3)))
        self.set_weights(np.zeros(len(perturbations)))

    def set_weights(self, c):
        """Weights c_k of the realizations in the combined displacement."""
        c = np.asarray(c, dtype=float)
        self._d = self._jnp.asarray(np.einsum('k,kcqi->cqi', c, self._d_all))
        self._dp = self._jnp.asarray(np.einsum('k,kcqi->cqi', c, self._dp_all))

    def dB_by_dsigma(self, points):
        g, gd, I = self._coil_data()
        return np.asarray(self._dBdsigma(self._jnp.asarray(points), g, gd, I, self._d, self._dp))

    def dgradB_by_dsigma(self, points):
        g, gd, I = self._coil_data()
        return np.asarray(self._dgradBdsigma(self._jnp.asarray(points), g, gd, I, self._d, self._dp))

    def dB_by_dsigma_vjp(self, points, v):
        g, gd, I = self._coil_data()
        vg, vgd, vI = self._dBdsigma_grad(self._jnp.asarray(points), g, gd, I, self._d, self._dp,
                                          self._jnp.asarray(v))
        vg, vgd, vI = np.asarray(vg), np.asarray(vgd), np.asarray(vI)
        deriv = None
        for k, c in enumerate(self.coils):
            dk = (c.curve.dgamma_by_dcoeff_vjp(vg[k])
                  + c.curve.dgammadash_by_dcoeff_vjp(vgd[k])
                  + c.current.vjp(np.array([vI[k]])))
            deriv = dk if deriv is None else deriv + dk
        return deriv


class _GroupedDSigma:
    """Holder of the compiled evaluation of D_sigma for a batch of perturbations (used by
    ExactMeanUnfoldingC, on an object with _weighted and _dsigma_weighted)."""

    def _build_grouped_forward(self, frame):
        """Compiled function of D_sigma for a batch of field perturbations delta_B.

        D_sigma solves the collocation system of dsigma_pure (utils/discriminant.py), whose
        operator only depends on the unperturbed field, the length and the grid, with a
        right-hand side that is linear in delta_B. The operator is built and factorized once
        and solved with one right-hand side per delta_B of the batch (first axis of dB_all)."""
        import jax
        import jax.numpy as jnp
        from star_lite_design.utils.tangent_map import A_pure, reduced_frame_pure
        D = jnp.asarray(self._dsigma_weighted.D_full)

        def dsigma_all(B, gradB_ss, dB_all, L, gamma):
            # dsigma_pure for dB_all of shape (realization, N, 3) -> D_sigma's (realization, 2)
            gradB = jnp.transpose(gradB_ss, (0, 2, 1))
            N = B.shape[0]
            A = A_pure(B, gradB)
            modB = jnp.linalg.norm(B, axis=-1)
            bhat = B / modB[:, None]

            def resid0(eta):                      # the operator of dsigma_pure, without source
                r = jnp.matmul(D, eta) / L - jnp.einsum('nij,nj->ni', A, eta)
                return jnp.concatenate([eta[0][None, :], r[1:]], axis=0)   # first row: eta(0) = 0

            Lop = jax.jacfwd(resid0)(jnp.zeros_like(B)).reshape((3 * N, 3 * N))
            source = (dB_all - bhat[None] * jnp.sum(bhat[None] * dB_all, axis=-1)[..., None]) / modB[None, :, None]
            rhs = source.at[:, 0, :].set(0.0).reshape((source.shape[0], 3 * N)).T    # (3N, realization)
            eta_end = jnp.linalg.solve(Lop, rhs)[-3:, :]                             # eta at the last node
            _, Pt, _ = reduced_frame_pure(B, gamma, D, frame)
            return (Pt[-1] @ eta_end).T

        dBdsigma = self._weighted._dBdsigma
        self._dB_chunk = jax.jit(lambda p, g, gd, I, dc, dpc: jax.vmap(
            lambda d, dp: dBdsigma(p, g, gd, I, d, dp))(dc, dpc))
        self._dsigma_all = jax.jit(dsigma_all)


class ExactMeanUnfoldingC(Optimizable):
    r"""E[C], the mean unfolding constant of a snowflake under Gaussian-process perturbations
    of the modular coils, computed EXACTLY (no realizations).

    D_sigma (2-vector) is linear in the displacement of the coils, which is Gaussian (per coil
    and per component, sample = L_s z with z standard normal and L_s the factor of the
    covariance of simsopt's GaussianSampler). So D_sigma ~ N(0, Sigma), Sigma = M M^T with
    M = G L_s and G the Jacobian of D_sigma w.r.t. the displacement and its derivative. And
    C(K, D) is homogeneous of degree 1/2 in D. With Sigma = S S^T (Cholesky) and
    D = r S u(theta), r Rayleigh and theta uniform:

        E[C] = E[sqrt(r)] * mean over theta of C(K, S u(theta)),  E[sqrt(r)] = 2^(1/4) Gamma(5/4).

    The mean over theta is a midpoint rule with n_theta angles (converged to 6 digits with 256).
    J() raises when a direction has no real split. dJ(): C depends on K (adjoint of the jet,
    disc.jet_cotangent_vjp) and on Sigma; d(Sigma) only involves dG contracted with two FIXED
    displacements, i.e. two adjoints of D_sigma for those displacements (_WeightedPerturbation).
    values() returns E[sqrt(r)] * C(theta): its mean is E[C]."""

    def __init__(self, fieldline, biotsavart, length_scale=0.5, frame='RZ', phi0=0.0, n_theta=256):
        import jax
        import jax.numpy as jnp
        from math import gamma
        self.fieldline = fieldline
        self.nfp = int(fieldline.curve.nfp)
        self.disc = SnowflakeDiscriminant(fieldline, biotsavart, frame=frame, phi0=phi0)
        # one perturbation object: it supplies the sampler (the covariance) and the coil data
        pert = CoilPerturbationField(biotsavart, length_scale=length_scale, seed=0)
        self._Ls = np.asarray(pert._unit_samples[0].sampler.L)
        self._weighted = _WeightedPerturbation([pert])
        self._dsigma_weighted = DSigma(fieldline, self._weighted, frame=frame, biotsavart=biotsavart, phi0=phi0)
        _GroupedDSigma._build_grouped_forward(self, frame)       # self._dsigma_all
        dBdsigma, dsigma_all = self._weighted._dBdsigma, self._dsigma_all

        def D_of(d, dp, p, g, gd, I, B, gB, L):
            return dsigma_all(B, gB, dBdsigma(p, g, gd, I, d, dp)[None], L, p)[0]

        self._jacD = jax.jit(jax.jacrev(D_of, argnums=(0, 1)))
        self._chol_vjp = lambda Sigma, cot: np.asarray(jax.vjp(jnp.linalg.cholesky, jnp.asarray(Sigma))[1](jnp.asarray(cot))[0])
        th = (np.arange(int(n_theta)) + 0.5) * 2.0 * np.pi / int(n_theta)
        self._u = np.stack([np.cos(th), np.sin(th)], axis=1)     # (n_theta, 2)
        self._c_half = 2.0 ** 0.25 * gamma(1.25)
        self._solution = None
        super().__init__(depends_on=[self.disc, self._dsigma_weighted])

    def recompute_bell(self, parent=None):
        self._solution = None

    def _solve(self):
        """K, the rows M of Sigma = M M^T, its Cholesky factor S and (C, xi) per angle."""
        if self._solution is None:
            import jax.numpy as jnp
            K = self.nfp * np.asarray(self.disc.jet())
            pts, B, gradB, _, L = self._dsigma_weighted._field_data()
            w = self._weighted
            g, gd, I = w._coil_data()
            zero = jnp.zeros(w._d_all[0].shape)
            Gd, Gdp = self._jacD(zero, zero, jnp.asarray(pts), g, gd, I, jnp.asarray(B), jnp.asarray(gradB), L)
            Gd, Gdp = np.asarray(Gd), np.asarray(Gdp)            # (2, coil, quadrature point, 3)
            nq, m = Gd.shape[2], self._Ls.shape[0]
            G = np.zeros((2, Gd.shape[1], m, 3))                 # on (d, d', higher derivatives)
            G[:, :, :nq], G[:, :, nq:2 * nq] = Gd, Gdp
            M = np.einsum('icaj,ab->icbj', G, self._Ls)          # D = sum over (c, b, j) of M z
            Sigma = np.einsum('icbj,kcbj->ik', M, M)
            S = np.linalg.cholesky(Sigma)
            out = []
            for u in self._u:
                sols = unfolding_solutions(K, S @ u)
                out.append((float(sols[0]['C']), np.asarray(sols[0]['xi'])) if sols else (float('nan'), None))
            self._solution = (K, M, Sigma, S, out)
        return self._solution

    def values(self):
        return self._c_half * np.array([C for C, _ in self._solve()[4]])

    def sigma(self):
        """Covariance (2x2) of D_sigma."""
        return self._solve()[2]

    def std(self):
        """Standard deviation of C: E[C^2] = E[r] * mean over theta of C(theta)^2, E[r] = sqrt(pi/2)."""
        C = np.array([C for C, _ in self._solve()[4]])
        return float(np.sqrt(max(np.sqrt(np.pi / 2.0) * np.mean(C ** 2) - (self._c_half * np.mean(C)) ** 2, 0.0)))

    def J(self):
        v = self.values()
        if not np.all(np.isfinite(v)):
            raise RuntimeError(f"E[C] undefined: {int(np.sum(~np.isfinite(v)))} of {len(v)} directions "
                               f"have no real split")
        return float(np.mean(v))

    @derivative_dec
    def dJ(self):
        out = self._solve()[4]
        if any(xi is None for _, xi in out):
            raise RuntimeError("E[C] undefined (a direction has no real split); dJ is undefined.")
        a = self._c_half / len(out)
        # E[C] = a * sum over theta of C, C = 2 |xi|: dE/dxi = 2 a xi / |xi|
        return self._adjoint_xi(np.array([2.0 * a * xi / np.linalg.norm(xi) for _, xi in out]))

    def _adjoint_xi(self, q):
        """Derivative (simsopt Derivative) of a scalar whose partial derivatives w.r.t. the split
        directions xi(theta) at phi = 0 are q, of shape (n_theta, 2): through xi(K, D = S u)."""
        import jax.numpy as jnp
        K, M, Sigma, S, out = self._solve()
        cK = np.zeros((2, 2, 2))
        dS = np.zeros((2, 2))
        for (_, xi), u, qk in zip(out, self._u, q):
            # IFT on 0.5 K[xi, xi] + D = 0 (as SnowflakeUnfolding.dC): d(q . xi) = lam . dD
            # + 0.5 lam_i xi_a xi_b dK_iab with lam = -(K[., xi])^-T q; here D = S u
            Jmat = np.einsum('icb,b->ic', K, xi)
            lam = -np.linalg.solve(Jmat.T, qk)
            cK += self.nfp * 0.5 * np.einsum('i,a,b->iab', lam, xi, xi)
            dS += np.outer(lam, u)
        Q = self._chol_vjp(Sigma, dS)                            # dE = sum_ik Q_ik dSigma_ik
        # dSigma_ik = dM_i . M_k + M_i . dM_k and dM = dG L_s: dE = sum_i dG_i . delta_i with the
        # displacement delta_i = [(Q + Q^T)_ik M_k] L_s^T held fixed, i.e. d(e_i . D_sigma[delta_i])
        R = np.einsum('ik,kcbj->icbj', Q + Q.T, M)
        delta = np.einsum('icbj,ab->icaj', R, self._Ls)
        nq = self._weighted._d_all.shape[2]
        dJ = None
        for i in range(2):
            self._weighted._d = jnp.asarray(delta[i][:, :nq])
            self._weighted._dp = jnp.asarray(delta[i][:, nq:2 * nq])
            term = self._dsigma_weighted.d_sigma_vjp(np.eye(2)[i])
            dJ = term if dJ is None else dJ + term
        return dJ + self.disc.jet_cotangent_vjp(cK)

    return_fn_map = {'J': J, 'dJ': dJ}


class PhiMeanUnfoldingC(Optimizable):
    r"""Mean over the toroidal sections of E[C], the exact mean unfolding constant of a
    snowflake (ExactMeanUnfoldingC gives it in the section at phi = 0).

    The two X-points born from the snowflake are field lines, so their half-separation xi in the
    (R, Z) section through the point t of the snowflake field line is the one at t = 0
    transported by the flow: xi(t) = T(t) xi(0), with T(t) the 2x2 section-to-section tangent
    map (utils/tangent_map.py, monodromy_pure, at every node of the full-torus Chebyshev grid of
    DSigma; T = I after a full turn since the monodromy of a snowflake is the identity). Hence

        E[C](t) = E[sqrt(r)] * mean over theta of 2 |T(t) xi(0; theta)|,

    with xi(0; theta) the split directions of ExactMeanUnfoldingC, and nothing is solved per
    section. J() is the mean of E[C](t) over the field-line parameter t in [0, 1] (Clenshaw-
    Curtis weights; t / (2 pi) is close to the toroidal angle), profile() gives (t, E[C](t)).
    dJ(): through xi(0) (ExactMeanUnfoldingC._adjoint_xi) and through T (the coil-dof adjoint of
    DSigma, run on the scalar sum over nodes of <G, T> instead of v . D_sigma)."""

    def __init__(self, ec):
        import jax
        import jax.numpy as jnp
        from star_lite_design.utils.tangent_map import cheb, monodromy_pure
        self.ec = ec
        ds = ec._dsigma_weighted
        frame = ds.frame
        D = jnp.asarray(ds.D_full)
        _, t, w = cheb(ds.D_full.shape[0], 0.0, 1.0)
        self._t, self._w = np.asarray(t), np.asarray(w) / np.sum(w)

        def T_all(B, gradB_ss, L, gamma):          # (node, 2, 2), gradB in SIMSOPT convention
            return monodromy_pure(B, jnp.transpose(gradB_ss, (0, 2, 1)), L, gamma, D, frame)

        self._T_all = jax.jit(T_all)
        # a DSigma of its own, on the same grid, whose scalar is sum(v * T) (v of shape
        # (node, 2, 2)) instead of v . D_sigma: its d_sigma_vjp then is the adjoint of T
        self._ds_T = DSigma(ec.fieldline, ec._weighted, frame=frame, biotsavart=ds.biotsavart, phi0=ds.phi0)
        self._ds_T._dsig_grad = jax.jit(jax.grad(
            lambda B, gB, dB, L, g, v: jnp.sum(v * T_all(B, gB, L, g)), argnums=(0, 1, 2, 3, 4)))
        self._solution = None
        super().__init__(depends_on=[ec, self._ds_T])

    def recompute_bell(self, parent=None):
        self._solution = None

    def _solve(self):
        """T at the nodes, xi(0; theta), xi at every (node, theta) and its norm."""
        if self._solution is None:
            import jax.numpy as jnp
            out = self.ec._solve()[4]
            if any(xi is None for _, xi in out):
                raise RuntimeError("E[C] undefined (a direction has no real split)")
            xi0 = np.array([xi for _, xi in out])                       # (theta, 2)
            pts, B, gradB, _, L = self._ds_T._field_data()
            T = np.asarray(self._T_all(jnp.asarray(B), jnp.asarray(gradB), L, jnp.asarray(pts)))
            xi = np.einsum('sab,tb->sta', T, xi0)                       # (node, theta, 2)
            self._solution = (T, xi0, xi, np.linalg.norm(xi, axis=-1))
        return self._solution

    def profile(self):
        """(t, E[C](t)) at the nodes of the grid."""
        nrm = self._solve()[3]
        return self._t, self.ec._c_half * 2.0 * np.mean(nrm, axis=1)

    def J(self):
        return float(np.sum(self._w * self.profile()[1]))

    @derivative_dec
    def dJ(self):
        T, xi0, xi, nrm = self._solve()
        a = 2.0 * self.ec._c_half / xi0.shape[0]
        xihat = xi / nrm[:, :, None]
        G = a * self._w[:, None, None] * np.einsum('sta,tb->sab', xihat, xi0)   # cotangent of T
        q = a * np.einsum('s,sab,sta->tb', self._w, T, xihat)                   # cotangent of xi(0)
        return self.ec._adjoint_xi(q) + self._ds_T.d_sigma_vjp(G)

    return_fn_map = {'J': J, 'dJ': dJ}


def _rel_vio(a, b):
    """Violation of the inequality a < b for the weight-escalation check: the raw
    excess max(a-b, 0) made RELATIVE to the target |b| (so it is comparable to the
    0.1% escalation gate), or ABSOLUTE when the target b == 0 (e.g. positivity)."""
    v = max(a - b, 0.0)
    return v / abs(b) if b != 0.0 else v


# The boozer_all design json is the only required input; everything else
# (weights, thresholds, monodromy constraint, config id, well state, ...) is
# read from the sibling yaml of the same basename in the same directory. The
# number of auxiliary coils is given on the command line (prefix.sh passes it),
# defaulting to NUM_AUX_DEFAULT. Outputs are written in place, next to the input.
NUM_AUX_DEFAULT = 10

parser = argparse.ArgumentParser(
    description="Singular polish + coil optimization, continuing a boozer_all "
                "design. Reads all run parameters from the design json's sibling "
                ".yaml; writes its outputs next to the input.")
parser.add_argument("design_json",
                    help="path to a boozer_all design_opt_final_<ID>.json (the aux coils are placed and the "
                         "snowflake is made by the singular polish, then both configurations are optimized) "
                         "or to the design_double_final_<TAG>.json of a finished double (it is reloaded and "
                         "optimized further); the matching .yaml in the same directory supplies all parameters")
parser.add_argument("--ec-max", type=float, default=None,
                    help="upper bound on E[C], the mean unfolding constant of the snowflake [m^(1/2)], in the "
                         "section or averaged over the sections (--ec-section). Default: no bound (E[C] is "
                         "then only reported, and the optimization is the plain double optimization)")
parser.add_argument("--risk-length-scale", type=float, default=0.5,
                    help="GP correlation length of the coil perturbation (default 0.5)")
parser.add_argument("--ec-section", default=None, metavar="mean|T",
                    help="which E[C] the bound --ec-max applies to: 'mean' for the mean of E[C] over the "
                         "toroidal sections, or a number T in [0, 1) for E[C] in the section through the "
                         "point of the snowflake field line at the fraction T of its full turn (default: the "
                         "yaml's EC_SECTION if a previous run wrote it, else 0, the section at phi = 0)")
parser.add_argument("--check-gradient", action="store_true",
                    help="compare the gradient of E[C] with finite differences, then stop")
parser.add_argument("--out-dir", default=None,
                    help="directory of the outputs (default: next to the input design json)")
parser.add_argument("--num-aux", type=int, default=None,
                    help=f"number of planar circular auxiliary coils, for a start from a boozer_all design "
                         f"(default {NUM_AUX_DEFAULT}); ignored for a finished double, which keeps its own")
parser.add_argument("--discriminant", type=int, default=None, choices=(0, 1),
                    help="1 for the snowflake-discriminant inequality constraint "
                         "Delta > DISCRIMINANT_THRESHOLD, 0 off (default: 0 for a start from a boozer_all "
                         "design; as in the loaded double, its yaml's DISCRIMINANT, for a finished double). "
                         "Recorded in the output filename (_disc1 suffix when on).")
parser.add_argument("--stages", type=int, default=10,
                    help="number of outer BFGS stages (default 10; lower it for a quick end-to-end test)")
parser.add_argument("--maxiter", type=int, default=1000,
                    help="BFGS iterations per stage (default 1000; lower it for a quick end-to-end test)")
parser.add_argument("--trace-min", type=float, default=None,
                    help="lower bound on tr M of the config 1 X-point (default: 2 + the yaml's "
                         "MONODROMY_THRESHOLD, i.e. boozer_all's --mono-thr)")
parser.add_argument("--target", default="double", choices=("double", "snowflake"),
                    help="what is optimized: 'double' (default) = both configurations, config 1 X-point "
                         "with the aux coils off and config 2 snowflake; 'snowflake' = the snowflake "
                         "configuration alone (no config 1, no bound on the trace of its X-point). The "
                         "outputs keep the names of a double, with config2/ only")
parser.add_argument("--aux-current-max", type=float, default=None,
                    help="bound on the current of each auxiliary (planar circular, PF) coil, in amperes: "
                         "|I| <= this (default: the yaml's AUX_CURRENT_EPS if a previous run wrote it, "
                         "else 5000)")
args = parser.parse_args()
SNOW_ONLY = args.target == "snowflake"

_in = os.path.abspath(args.design_json)
_yaml_path = os.path.splitext(_in)[0] + '.yaml'
config = yaml.safe_load(open(_yaml_path, 'r'))
data = load(_in)

# run parameters: num_aux from the command line, everything else from the yaml
config_id = int(config['CONFIG_ID'])
mon_constraint = config['MONODROMY_CONSTRAINT']
if mon_constraint not in ('trace', 'identity'):
    raise SystemExit(f"config MONODROMY_CONSTRAINT must be 'trace' or 'identity', "
                     f"got {mon_constraint!r}")
# The input is either a FINISHED double (two configurations, the second one with its snowflake:
# it is reloaded) or a boozer_all design (one configuration, no aux coils: the snowflake is
# built by the singular polish, as boozer_double_opt.py did).
IS_RESTART = len(data[0]) == 2 and isinstance(data[3][1], SingularPeriodicFieldline)
if not IS_RESTART and len(data[0]) != 1:
    raise SystemExit(f"expected a boozer_all design (one configuration) or a finished double (X-point + "
                     f"snowflake), got {len(data[0])} configuration(s) in {_in}")
if not IS_RESTART and isinstance(data[3][0], SingularPeriodicFieldline):
    raise SystemExit(f"{_in} is a finished snowflake-only design (--target snowflake): restarting from "
                     f"one is not supported; start from the boozer_all design or from a finished double")
config['TARGET'] = args.target         # 'double' or 'snowflake', recorded in the (output) yaml
if IS_RESTART:
    num_aux = int(config['NUM_AUX'])   # the loaded double's (checked against its snowflake below)
    # 1 = keep the snowflake-discriminant constraint: as in the loaded double unless --discriminant
    discriminant = int(args.discriminant if args.discriminant is not None else config.get('DISCRIMINANT', 0))
else:
    num_aux = int(args.num_aux if args.num_aux is not None else NUM_AUX_DEFAULT)
    discriminant = int(args.discriminant if args.discriminant is not None else 0)
config['NUM_AUX'] = num_aux            # record it in the (output) yaml
config['DISCRIMINANT'] = discriminant  # record it in the (output) yaml
# bound on E[C]: --ec-max, or none (E[C] is then reported but not part of the objective)
EC_ENABLED = args.ec_max is not None
EC_MAX = float(args.ec_max) if EC_ENABLED else None
if EC_ENABLED:
    config['EC_MAX'] = EC_MAX
else:
    config.pop('EC_MAX', None)
config['RISK_LENGTH_SCALE'] = float(args.risk_length_scale)
# N_DEP_CURRENTS (number of dependent aux currents = number of monodromy constraints
# solved for) is chosen after the X-point stellsym-defect check below: a stellsym
# singular field line has one fewer identity constraint (M[0,0] == M[1,1]).

# outputs in --out-dir, or in place (next to the input design json)
OUT_DIR = os.path.join(os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(_in), '')
TASK_NAME = os.path.basename(os.path.dirname(OUT_DIR)) or os.path.basename(_in)
# Device ID = crc32 of the folder name (matches boozer_all.py / device_browser.py),
# embedded in the output json/yaml names: design_double_final_<DEVICE_ID>.json.
DEVICE_ID = zlib.crc32(TASK_NAME.encode())
# Tag used in the OUTPUT filenames: a run WITH the snowflake-discriminant constraint
# (--discriminant 1) is written as design_..._<DEVICE_ID>_disc1 so it does not overwrite
# the default run, which keeps the plain design_..._<DEVICE_ID> names expected downstream.
DEVICE_TAG = f'{DEVICE_ID}_disc1' if discriminant else f'{DEVICE_ID}'
print(f"Task: {TASK_NAME}")
print(f"Input: {_in}")
print(f"Output dir: {OUT_DIR}")
print(f"constraint={mon_constraint}  num_aux={num_aux}  discriminant={discriminant}  config_id={config_id}")

def _copy_curve(xc):
    c = CurveXYZFourierSymmetries(xc.quadpoints, xc.order, xc.nfp, xc.stellsym, ntor=xc.ntor)
    c.set_dofs(xc.get_dofs())
    return c


def _leaf_currents(cur):
    """ids of the plain Current objects a (possibly scaled) coil current is built on."""
    return frozenset(id(o) for o in [cur] + list(cur.ancestors) if type(o).__name__ == 'Current')


if IS_RESTART:
    print("Running RISK-NEUTRAL double optimization (restart from a double design)")
    print("================================")
    # The input is a FINISHED double: design_double_final_<TAG>.json, with its two configurations
    #   boozer surfaces [config 1 on its modular coils, config 2 on explicit modular + aux coils],
    #   axes            [config 1, config 2 (explicit coils)],
    #   x-points        [config 1 PeriodicFieldLine, config 2 SingularPeriodicFieldline (snowflake)].
    # The optimization below works with the snowflake configuration on
    # SingularBiotSavart(field line): its Boozer surface and axis are rebuilt on that field here.
    (bs_x, _bs2), _iota_Gs_in, (axis_x, _ax2), (xp_x, _fl), sdf = data

    # ---- config 1: hyperbolic X-point, aux coils off, its own modular currents ----------------
    coils_x = bs_x.biotsavart.coils
    axis_x.options = {'newton_tol': 1e-15, 'newton_maxiter': 20, 'verbose': False}
    xp_x.options = {'newton_tol': 1e-15, 'newton_maxiter': 20, 'verbose': False}
    bs_x.options = {'newton_tol': 1e-16 if bs_x.surface.stellsym else 1e-14, 'newton_maxiter': 20, 'verbose': False}
    _res_ax = axis_x.run_code(CurveLength(axis_x.curve).J())
    _res_xp = xp_x.run_code(CurveLength(xp_x.curve).J())
    _res_bs = bs_x.run_code(_iota_Gs_in[0][0], _iota_Gs_in[0][1])
    print(f"config 1 (X-point, aux off): Boozer surface success={_res_bs['success']} iota={_res_bs['iota']:+.8f}, "
          f"axis success={_res_ax['success']}, X-point success={_res_xp['success']}")
    if not (_res_bs['success'] and _res_ax['success'] and _res_xp['success']):
        raise SystemExit("ERROR: could not re-solve configuration 1 of the double")

    # ---- config 2: the snowflake, modular coils with the SAME curves as config 1 ---------------
    coils = _fl.biotsavart.coils
    curves = [c.curve for c in coils]
    if len(coils) != len(coils_x) or any(a.curve is not b.curve for a, b in zip(coils_x, coils)):
        raise SystemExit("ERROR: the two configurations of the loaded double do not share their coil curves")
    # designB (4 coils). DN (stellsym surface): 2 independent base coils -> [0, 2].
    # SN (rebuilt by sn_setup from bases 0 and 2): 2 -> [0, 1].
    base_curve_idx = [0, 2] if bs_x.surface.stellsym else [0, 1]
    base_curves = [curves[i] for i in base_curve_idx]

    num_aux = (len(np.asarray(_fl.mu)) - 1) // 2   # from the design itself: mu = (I_1..I_N, r_1..r_N, Z)
    config['NUM_AUX'] = num_aux
    # The snowflake field line keeps the dependent / independent partition of its mu saved with the
    # design; it is re-solved (square Newton solve) so that its adjoint data is available.
    # (update, not replace: the constructor filled in the other options, e.g. target_monodromy)
    _fl.options.update({'newton_tol': 1e-12, 'newton_maxiter': 10, 'verbose': False,
                        'monodromy_constraint': mon_constraint})
    _fl.need_to_run_code = True
    _res_fl = _fl.run_code(_fl.length if _fl.length is not None else CurveLength(_fl.curve).J())
    print(f"config 2 snowflake field line: success={_res_fl['success']} square={_res_fl['square']} iter={_res_fl['iter']} "
          f"||r||_inf={np.linalg.norm(_res_fl['residual'], ord=np.inf):.3e}")
    if not (_res_fl['success'] and _res_fl['square']):
        raise SystemExit("ERROR: could not re-solve the snowflake field line of the double (square solve)")
    _surf2 = _bs2.surface
    bs_s = BoozerSurface(SingularBiotSavart(_fl), _surf2, Volume(_surf2), bs_x.targetlabel,
                         constraint_weight=bs_x.constraint_weight, options=dict(bs_x.options))
    _res_bs2 = bs_s.run_code(_iota_Gs_in[1][0], _iota_Gs_in[1][1])
    ax_s = PeriodicFieldLine(SingularBiotSavart(_fl), _ax2.curve, options=dict(axis_x.options))
    _res_ax2 = ax_s.run_code(CurveLength(_ax2.curve).J())
    print(f"config 2 (snowflake, {num_aux} aux coils) on SingularBiotSavart: Boozer surface success={_res_bs2['success']} "
          f"iota={_res_bs2['iota']:+.8f}, axis success={_res_ax2['success']}")
    if not (_res_bs2['success'] and _res_ax2['success']) or _surf2.is_self_intersecting():
        raise SystemExit("ERROR: could not rebuild the snowflake configuration on SingularBiotSavart")

    sing_fls = [_fl]
    if SNOW_ONLY:      # the snowflake configuration alone: config 1 of the double is left out
        boozer_surfaces, axes, xpoints = [bs_s], [ax_s], [_fl]
    else:
        boozer_surfaces = [bs_x, bs_s]
        axes = [axis_x, ax_s]
        xpoints = [xp_x, _fl]
else:
    print("Running SINGULAR-POLISH Optimization")
    print("================================")
    boozer_surfaces = data[0]   # BoozerSurfaces
    iota_Gs = data[1]           # (iota, G) pairs
    axes = data[2]              # magnetic axis field lines
    xpoints = data[3]           # X-point field lines
    sdf = data[4]               # vessel signed-distance function

    for ii, (boozer_surface, axis, xpoint) in enumerate(zip(boozer_surfaces, axes, xpoints)):
        axis.options = {'newton_tol': 1e-15, 'newton_maxiter': 20, 'verbose': False}
        xpoint.options = {'newton_tol': 1e-15, 'newton_maxiter': 20, 'verbose': False}
        boozer_surface.options = {'newton_tol': 1e-16 if boozer_surface.surface.stellsym else 1e-14,
                                  'newton_maxiter': 20, 'verbose': False}
        axis.run_code(CurveLength(axis.curve).J())
        xpoint.run_code(CurveLength(xpoint.curve).J())
        boozer_surface.run_code(iota_Gs[ii][0], iota_Gs[ii][1])

    # get the base curves
    biotsavart = boozer_surfaces[0].biotsavart
    coils = biotsavart.coils
    curves = [c.curve for c in coils]
    # designB (4 coils). DN (stellsym surface): 2 independent base coils -> [0, 2].
    # SN (rebuilt by sn_setup from bases 0 and 2): 2 -> [0, 1].
    base_curve_idx = [0, 2] if boozer_surfaces[0].surface.stellsym else [0, 1]
    base_curves = [curves[i] for i in base_curve_idx]

    # ---- config 1: the boozer_all device itself (hyperbolic X-point, aux coils off) ------------
    # Same coil CURVES as the loaded device, but its OWN modular currents (independent design
    # variables, started at the loaded values), and its own copies of the Boozer surface, the
    # axis and the X-point, solved on those coils. The loaded objects are left untouched: the
    # snowflake (config 2) is generated from them below, as in boozer_singular_opt.py.

    # one own current per base coil; a symmetry partner takes its base coil's, with the same ratio
    _own_currents = {}
    for i in base_curve_idx:
        _cur = coils[i].current
        _own_currents[i] = ScaledCurrent(Current(float(_cur.get_value()) / float(_cur.scale)), float(_cur.scale))
    coils_x = []
    for j, c in enumerate(coils):
        _match = [i for i in base_curve_idx if _leaf_currents(coils[i].current) == _leaf_currents(c.current)]
        if len(_match) != 1:
            raise SystemExit(f"cannot tell which base coil the current of modular coil {j} comes from")
        _ratio = float(c.current.get_value()) / float(coils[_match[0]].current.get_value())
        coils_x.append(Coil(c.curve, _own_currents[_match[0]] if _ratio == 1.0
                            else ScaledCurrent(_own_currents[_match[0]], _ratio)))

    _bs0, _s0 = boozer_surfaces[0], boozer_surfaces[0].surface
    surf_x = SurfaceXYZTensorFourier(mpol=_s0.mpol, ntor=_s0.ntor, nfp=_s0.nfp, stellsym=_s0.stellsym,
                                     quadpoints_phi=_s0.quadpoints_phi, quadpoints_theta=_s0.quadpoints_theta)
    surf_x.x = _s0.x.copy()
    bs_x = BoozerSurface(BiotSavart(coils_x), surf_x, Volume(surf_x), _bs0.targetlabel,
                         constraint_weight=_bs0.constraint_weight, options=dict(_bs0.options))
    _res_bs = bs_x.run_code(iota_Gs[0][0], iota_Gs[0][1])
    axis_x = PeriodicFieldLine(BiotSavart(coils_x), _copy_curve(axes[0].curve), options=dict(axes[0].options))
    _res_ax = axis_x.run_code(axes[0].res['length'])
    xp_x = PeriodicFieldLine(BiotSavart(coils_x), _copy_curve(xpoints[0].curve), options=dict(xpoints[0].options))
    _res_xp = xp_x.run_code(xpoints[0].res['length'])
    print(f"config 1 (X-point, aux off): Boozer surface success={_res_bs['success']} iota={_res_bs['iota']:+.8f}, "
          f"axis success={_res_ax['success']}, X-point success={_res_xp['success']}")
    if not (_res_bs['success'] and _res_ax['success'] and _res_xp['success']):
        raise SystemExit("ERROR: could not re-solve the boozer_all device on its own copy of the coils")

# ALL thresholds AND weights are read STRAIGHT from the config yaml (boozer_all
# wrote them); the polish does not set its own values. This includes the well
# state: WELL_ACTIVE False / WELL_WEIGHT == 0 means the well penalty was OFF in
# the boozer_all run and is loaded as off here.
COIL_ON_VESSEL_THRESHOLD = config['COIL_ON_VESSEL_THRESHOLD']
COIL_CLEARANCE_THRESHOLD = config['COIL_CLEARANCE_THRESHOLD']
COIL_ON_VESSEL_WEIGHT = Weight(config['COIL_ON_VESSEL_WEIGHT'])
COIL_CLEARANCE_WEIGHT = Weight(config['COIL_CLEARANCE_WEIGHT'])

PLASMA_VESSEL_MARGIN_THRESHOLD = config['PLASMA_VESSEL_MARGIN_THRESHOLD']

CURRENT_THRESHOLD = config['CURRENT_THRESHOLD']
CURRENT_WEIGHT = Weight(config['CURRENT_WEIGHT'])

well_active = bool(config['WELL_ACTIVE'])
WELL_THRESHOLD = float(config['WELL_THRESHOLD'])
WELL_WEIGHT = Weight(config['WELL_WEIGHT'])
print(f"WELL loaded from config: active={well_active}  threshold={WELL_THRESHOLD}  weight={WELL_WEIGHT.value}")

FIELDLINE_MEANZ_WEIGHT = Weight(config['FIELDLINE_MEANZ_WEIGHT'])
FIELDLINE_MEANZ_THRESHOLD = config['FIELDLINE_MEANZ_THRESHOLD']

FIELDLINE_MEANDIST_WEIGHT = Weight(config['FIELDLINE_MEANDIST_WEIGHT'])
FIELDLINE_MEANDIST_THRESHOLD = config['FIELDLINE_MEANDIST_THRESHOLD']



MODB_TARGET = config['MODB_TARGET']
MR_TARGET = config['MAJOR_RADIUS_TARGET']
COIL_TO_COIL_THRESHOLD = config['COIL_TO_COIL_THRESHOLD']
CURVATURE_THRESHOLD = config['CURVATURE_THRESHOLD']
MEAN_SQUARED_CURVATURE_THRESHOLD = config['MEAN_SQUARED_CURVATURE_THRESHOLD']
LENGTH_THRESHOLD = config['LENGTH_THRESHOLD']
IOTAS_TARGET  = config['IOTAS_TARGET'][config_id:config_id+1]
COIL_MINOR_RADIUS = config['COIL_MINOR_RADIUS']

COIL_TO_COIL_WEIGHT = Weight(config['COIL_TO_COIL_WEIGHT'])
CURVATURE_WEIGHT = Weight(config['CURVATURE_WEIGHT'])
MEAN_SQUARED_CURVATURE_WEIGHT = Weight(config['MEAN_SQUARED_CURVATURE_WEIGHT'])
LENGTH_WEIGHT = Weight(config['LENGTH_WEIGHT'])
IOTAS_WEIGHT=Weight(config['IOTAS_WEIGHT'])
# On-axis iota constraint: active only for --AR 1/2 devices (boozer_all.py recorded the AR
# knob in config['AR']). For --AR 0 the on-axis iota is still REPORTED in summary.txt but
# not penalized. When active, reuse the weight boozer_all wrote (the pre-scaling below
# shrinks it if the total-field term starts above 1), falling back to IOTAS_WEIGHT.
axis_iota_enabled = int(config.get('AR', 0)) in (1, 2)
if axis_iota_enabled:
    config.setdefault('AXIS_IOTA_WEIGHT', config['IOTAS_WEIGHT'])
    AXIS_IOTA_WEIGHT=Weight(config['AXIS_IOTA_WEIGHT'])
else:
    AXIS_IOTA_WEIGHT=Weight(0.0)
MAJOR_RADIUS_WEIGHT=Weight(config['MAJOR_RADIUS_WEIGHT'])

BOOZER_RESIDUAL_WEIGHT=Weight(config['BOOZER_RESIDUAL_WEIGHT'])


PLASMA_VESSEL_MARGIN_WEIGHT = Weight(config['PLASMA_VESSEL_MARGIN_WEIGHT'])
MODB_WEIGHT = Weight(config['MODB_WEIGHT'])
ARCLENGTH_WEIGHT = Weight(config['ARCLENGTH_WEIGHT'])

# Aux-coil clearance threshold (an existing yaml may override it).
config.setdefault('AUX_COIL_DISTANCE_THRESHOLD', 0.15)
AUX_COIL_DISTANCE_THRESHOLD = config['AUX_COIL_DISTANCE_THRESHOLD']
# Initial aux-coil radii, tried IN ORDER until the singular polish (stages 1-2) and the
# total-field current continuation both succeed (see the placement loop below):
#   'near_xpoint': radii centred on the X-point's mean major radius R_X, spaced by the
#                  aux-to-aux clearance (so that constraint holds from the start), >= 0.1 m;
#   'default':     linspace(0.5, 1.2), the original placement brought inside AUX_RADIUS_MAX.
# On the 60 accepted devices of campaign 7456743, 'near_xpoint' brought 19 to the
# optimization against 9 for 'default', and the two together 21.
AUX_PLACEMENTS = ('near_xpoint', 'default')


def _aux_radii0(placement, xpoint):
    if placement == 'near_xpoint':
        g = xpoint.curve.gamma()
        R_X = float(np.mean(np.sqrt(g[:, 0] ** 2 + g[:, 1] ** 2)))
        return np.maximum(R_X + AUX_COIL_DISTANCE_THRESHOLD * (np.arange(num_aux) - (num_aux - 1) / 2), 0.1)
    return np.linspace(0.5, 1.2, num_aux)


MU_NAMES = _mu_names(2 * num_aux + 1)

# A loaded X-point is always parametrized stellsym=False, but it may in fact be
# stellarator symmetric, i.e. gamma(t) = diag(1, -1, -1) gamma(-t) for all t. If
# so we project it onto a stellsym=True curve below so the singular field line is
# exactly stellarator symmetric. STELLSYM_TOL is the defect threshold (relative
# to the curve scale); the separation between a truly symmetric X-point (defect
# ~ machine eps) and a single-null one (defect ~ O(0.1 m)) is enormous, so the
# exact value is not delicate.
STELLSYM_TOL = 1e-9


def _stellsym_defect(curve):
    """max_t | gamma(t) - diag(1, -1, -1) gamma(-t) |, which is ~0 iff the curve
    is stellarator symmetric. gamma is 1-periodic in the parametrization, so
    gamma(-t) is evaluated on a copy of the curve at the reflected quadpoints."""
    qp = np.asarray(curve.quadpoints)
    refl = CurveXYZFourierSymmetries(np.mod(-qp, 1.0), curve.order, curve.nfp,
                                     curve.stellsym, ntor=curve.ntor)
    refl.set_dofs(curve.get_dofs())
    flip = np.array([1.0, -1.0, -1.0])
    return float(np.max(np.abs(curve.gamma() - flip * refl.gamma())))


# Number of dependent aux currents = number of monodromy constraints solved for:
# 1 for 'trace'; for 'identity', 3 in general but only 2 when the singular field line
# is stellarator symmetric (stellsym forces M[0,0] == M[1,1], so SingularPeriodicFieldline
# drops the redundant second diagonal equation). From a stage-1 device, decide from the
# X-points' stellsym defect -- the SAME test used per-X-point in the loop below -- assuming
# (and enforcing) that all X-points share the same stellsym; from a finished double, read
# it from the snowflake field line.
def _is_stellsym(curve):
    return _stellsym_defect(curve) <= STELLSYM_TOL * float(np.max(np.abs(curve.gamma())))


if IS_RESTART:
    SING_STELLSYM = bool(_fl.curve.stellsym)
else:
    _xp_stellsym = [_is_stellsym(xp.curve) for xp in xpoints]
    if len(set(_xp_stellsym)) > 1:
        raise SystemExit(f"X-points have mixed stellsym {_xp_stellsym}; N_DEP_CURRENTS "
                         f"assumes a single value across all field lines.")
    SING_STELLSYM = bool(_xp_stellsym[0])
N_DEP_CURRENTS = 1 if mon_constraint == 'trace' else (2 if SING_STELLSYM else 3)
if num_aux < N_DEP_CURRENTS:
    raise SystemExit(f"NUM_AUX must be >= {N_DEP_CURRENTS} for the {mon_constraint!r} "
                     f"constraint (stellsym={SING_STELLSYM}), got {num_aux}")
DEP_CURRENT_NAMES = [f'I{k+1}' for k in range(N_DEP_CURRENTS)]
INDEP_CURRENT_NAMES = [f'I{k+1}' for k in range(N_DEP_CURRENTS, num_aux)]


def _singular_polish(idx, xpoint, boozer_surface, radii0):
    """Singular polish (stages 1-2) of one X-point with the aux coils started at radii0:
    returns the polished SingularPeriodicFieldline, or None if stage 1 or 2 fails."""
    mu0 = np.concatenate([np.zeros(num_aux), radii0, [max_Z + AUX_COIL_DISTANCE_THRESHOLD]])
    # The polish gets its OWN copy of the X-point curve: the plain `xpoint`
    # field line keeps re-solving its curve every objective evaluation, so
    # sharing dofs would have the two Newton solvers fight each other.
    #
    # The loaded X-point is always stellsym=False. If it is in fact stellarator
    # symmetric, project it onto a stellsym=True curve so the singular field line
    # is exactly stellarator symmetric: the stellsym=True parametrization keeps
    # only the symmetric Fourier coefficients (xc, ys, zs); copying those by name
    # drops the antisymmetric ones (xs, yc, zc), which are zero up to `defect`.
    xc = xpoint.curve
    defect = _stellsym_defect(xc)
    scale = float(np.max(np.abs(xc.gamma())))
    if defect <= STELLSYM_TOL * scale:
        fl_curve = CurveXYZFourierSymmetries(xc.quadpoints, xc.order, xc.nfp, True, ntor=xc.ntor)
        sym_names = ([f'xc({i})' for i in range(xc.order + 1)]
                     + [f'ys({i})' for i in range(1, xc.order + 1)]
                     + [f'zs({i})' for i in range(1, xc.order + 1)])
        for nm in sym_names:
            fl_curve.set(nm, xc.get(nm))
        print(f"[idx={idx}] X-point IS stellarator symmetric "
              f"(defect={defect:.2e} <= {STELLSYM_TOL:.0e}*scale={STELLSYM_TOL * scale:.2e}); "
              f"projected onto a stellsym=True singular field line")
    else:
        fl_curve = CurveXYZFourierSymmetries(xc.quadpoints, xc.order, xc.nfp, xc.stellsym, ntor=xc.ntor)
        fl_curve.set_dofs(xc.get_dofs())
        print(f"[idx={idx}] X-point is NOT stellarator symmetric "
              f"(defect={defect:.2e} > {STELLSYM_TOL:.0e}*scale={STELLSYM_TOL * scale:.2e}); "
              f"keeping the stellsym=False parametrization")
    fl = SingularPeriodicFieldline(
        BiotSavart(boozer_surface.biotsavart.coils), fl_curve, mu0,
        options={'newton_tol': 1e-12, 'newton_maxiter': 10, 'verbose': False,
                 'monodromy_constraint': mon_constraint},
        stellsym_aux=boozer_surface.surface.stellsym)

    # (1) initial polish: ALL aux currents and radii are DEPENDENT (fixed),
    # z is independent (held at max_Z + AUX_COIL_DISTANCE_THRESHOLD). Under-determined -> pinv Newton.
    for k in range(num_aux):
        fl.fix(f'I{k+1}')
        fl.fix(f'r{k+1}')
    # Stage 1 starts far from the polished state: give it 40 Newton iterations (was 10)
    # and the damped (line-search) step, so full steps cannot throw the field line away;
    # both options are restored right after, for stage 2 and the optimization.
    _maxiter0 = fl.options['newton_maxiter']
    fl.options['newton_maxiter'] = 40
    fl.options['newton_linesearch'] = True
    res = fl.run_code(CurveLength(fl_curve).J())
    fl.options['newton_maxiter'] = _maxiter0
    fl.options.pop('newton_linesearch', None)
    print(f"[idx={idx}] stage-1 polish (all I, r dependent; pinv): success={res['success']}  "
          f"iter={res['iter']}  square={res['square']}  "
          f"||r||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e}")
    if not res['success']:
        print(f"[idx={idx}] stage-1 singular polish failed", flush=True)
        return None

    # (2) re-partition: unfix the currents, radii and z, then fix only the
    # dependent currents (1 for trace, 3 for identity). Square -> LU Newton.
    fl.unfix_all()
    for name in DEP_CURRENT_NAMES:
        fl.fix(name)
    fl.need_to_run_code = True
    res = fl.run_code(res['length'])
    print(f"[idx={idx}] stage-2 polish (dependent: {DEP_CURRENT_NAMES}; LU): success={res['success']}  "
          f"iter={res['iter']}  square={res['square']}  "
          f"||r||_inf={np.linalg.norm(res['residual'], ord=np.inf):.3e}")
    if not (res['success'] and res['square']):
        print(f"[idx={idx}] stage-2 (square) singular polish failed", flush=True)
        return None
    print(f"[idx={idx}] mu: " + '  '.join(
        f"{nm}={v * _CURRENT_SCALE:+.4e}A" if k < num_aux else f"{nm}={v:+.4f}"
        for k, (nm, v) in enumerate(zip(MU_NAMES, fl.mu))))
    return fl


# Recompute the Boozer surfaces and magnetic axes in the TOTAL field (modular +
# auxiliary): wrap each polished singular field line in a SingularBiotSavart and
# REPLACE the loaded (modular-field) boozer_surfaces / axes with ones solved on
# the total field. Every field-dependent objective below is then built against the
# total field too. Each object that drives its own set_points gets its OWN
# SingularBiotSavart(fl) wrapper (all sharing fl), mirroring the original
# per-objective BiotSavart(coils) so their point caches stay independent.
#
# CURRENT CONTINUATION: instead of one Newton solve from the modular-field surface
# straight into the full aux field, ramp the aux currents s*I_k, s: 0 -> 1, re-solving
# the Boozer surface (target volume) and the magnetic axis at each step from the
# previous converged state. Adaptive step (x1.5 on success, up to S_CONT_DSMAX; halve
# and restore on failure). A step is accepted only if the surface converges, does not
# self-intersect (checked at phi = 0) and the axis converges. If the step falls below
# S_CONT_DSMIN the target-volume surface is lost: report s* and stop.
S_CONT_DS0, S_CONT_DSMAX, S_CONT_DSMIN = 0.05, 0.25, 1e-4


def _self_intersecting(surf):
    try:
        return surf.is_self_intersecting()
    except Exception:   # raised when a cross section folds back on itself
        return True


def _scaled_aux_field(fl, s):
    """Modular + s * aux field: the aux coils of SingularBiotSavart(fl) (same curves
    and currents, synced from fl.mu, stellsym partners included), each current
    scaled by s."""
    sbs = SingularBiotSavart(fl)
    aux = [Coil(c.curve, ScaledCurrent(c.current, s)) for c in sbs._aux_bs.coils]
    return BiotSavart(list(fl.biotsavart.coils) + aux)


def _current_continuation(idx, boozer_surface, axis, fl):
    """Current continuation of the Boozer surface and magnetic axis into the total field
    of the polished field line fl: returns (BoozerSurface, PeriodicFieldLine) solved on
    SingularBiotSavart(fl), or None if the target-volume surface is lost."""
    surf = boozer_surface.surface
    # state = (surface dofs, iota, G, axis dofs, axis length) at the last accepted s
    state = (surf.get_dofs().copy(), boozer_surface.res['iota'], boozer_surface.res['G'],
             axis.curve.get_dofs().copy(), axis.res['length'])
    s, ds = 0.0, S_CONT_DS0
    print(f"[idx={idx}] current continuation: s=0  iota={state[1]:+.8f}  G={state[2]:.8e}", flush=True)
    while s < 1.0:
        st = min(s + ds, 1.0)
        surf.set_dofs(state[0])
        axis.curve.set_dofs(state[3])
        # the final step (s = 1) uses SingularBiotSavart itself
        field = SingularBiotSavart(fl) if st == 1.0 else _scaled_aux_field(fl, st)
        nbs = BoozerSurface(field, surf, boozer_surface.label, boozer_surface.targetlabel,
                            constraint_weight=boozer_surface.constraint_weight,
                            options=dict(boozer_surface.options))
        bs_res = nbs.run_code(state[1], state[2])
        surf_ok = bool(bs_res['success']) and not _self_intersecting(surf)
        ax_ok = False
        if surf_ok:
            nax = PeriodicFieldLine(SingularBiotSavart(fl) if st == 1.0 else _scaled_aux_field(fl, st),
                                    axis.curve, options=axis.options)
            ax_res = nax.run_code(state[4])
            ax_ok = bool(ax_res['success'])
        if surf_ok and ax_ok:
            s = st
            state = (surf.get_dofs().copy(), bs_res['iota'], bs_res['G'],
                     axis.curve.get_dofs().copy(), ax_res['length'])
            print(f"[idx={idx}]   s={s:.5f} ok    it={bs_res['iter']:2d}  iota={bs_res['iota']:+.8f}  "
                  f"G={bs_res['G']:.8e}", flush=True)
            ds = min(1.5 * ds, S_CONT_DSMAX)
        else:
            ds /= 2
            why = 'axis did not converge' if surf_ok else (
                'surface self-intersecting' if bs_res['success'] else 'surface did not converge')
            print(f"[idx={idx}]   s={st:.5f} FAIL  ({why}); ds -> {ds:.2e}", flush=True)
            if ds < S_CONT_DSMIN:
                surf.set_dofs(state[0])
                axis.curve.set_dofs(state[3])
                print(f"[idx={idx}] total-field BoozerSurface lost at s*={s:.5f} "
                      f"(last converged iota={state[1]:+.8f})", flush=True)
                return None
    # s = 1 reached: the last accepted solves are on SingularBiotSavart(fl)
    print(f"[idx={idx}] current continuation reached s=1: iota={state[1]:+.8f}", flush=True)
    return nbs, nax


if IS_RESTART:
    _fixed = [nm for nm, free in zip(MU_NAMES, _fl.mu_free_status()) if not free]
    print(f"snowflake mu: dependent (fixed) {_fixed}; expected {DEP_CURRENT_NAMES}")
    if _fixed != DEP_CURRENT_NAMES:
        raise SystemExit("ERROR: the dependent aux currents of the loaded snowflake are not the expected ones")
else:
    print("""
################################################################################
### Singular polish: SingularPeriodicFieldline per X-point ###############
################################################################################
""")
    # Auxiliary-coil parameters mu = (I_1..I_N, r_1..r_N, Z) become local dofs of
    # the SingularPeriodicFieldline. FIXED mu dofs are DEPENDENT (solved by
    # the Newton polish), FREE mu dofs are INDEPENDENT design variables.
    max_Z = np.max([np.abs(c.curve.gamma()[:, 2]) for c in boozer_surfaces[0].biotsavart.coils])
    # Aux-coil placement loop: for each X-point, try AUX_PLACEMENTS in order and keep the
    # first one whose singular polish (stages 1-2) AND current continuation both succeed.
    # The Boozer surface and the axis are put back in their modular-field state before
    # every attempt (a failed continuation leaves them at its last accepted s).
    sing_fls, new_boozer_surfaces, new_axes, aux_placements = [], [], [], []
    for idx, (xpoint, boozer_surface, axis) in enumerate(zip(xpoints, boozer_surfaces, axes)):
        surf_dofs0 = boozer_surface.surface.get_dofs().copy()
        axis_dofs0 = axis.curve.get_dofs().copy()
        for placement in AUX_PLACEMENTS:
            boozer_surface.surface.set_dofs(surf_dofs0)
            axis.curve.set_dofs(axis_dofs0)
            radii0 = _aux_radii0(placement, xpoint)
            print(f"[idx={idx}] aux placement '{placement}': initial radii = "
                  + ", ".join(f"{r:.3f}" for r in radii0), flush=True)
            fl = _singular_polish(idx, xpoint, boozer_surface, radii0)
            if fl is None:
                print(f"[idx={idx}] aux placement '{placement}' failed (singular polish)", flush=True)
                continue
            solved = _current_continuation(idx, boozer_surface, axis, fl)
            if solved is None:
                print(f"[idx={idx}] aux placement '{placement}' failed (surface lost)", flush=True)
                continue
            print(f"[idx={idx}] aux placement '{placement}' kept", flush=True)
            break
        else:
            print(f"ERROR: idx={idx}: no aux placement in {AUX_PLACEMENTS} gave a polished X-point "
                  f"with a total-field Boozer surface")
            raise SystemExit(1)
        sing_fls.append(fl)
        new_boozer_surfaces.append(solved[0])
        new_axes.append(solved[1])
        aux_placements.append(placement)
    boozer_surfaces, axes = new_boozer_surfaces, new_axes
    config['AUX_PLACEMENT'] = aux_placements   # recorded in the output yaml

    # ---- the two configurations: config 1 (X-point, aux off) then config 2 (snowflake) ---------
    # From here on the POLISHED singular field lines stand in for the X-points of config 2
    # (vessel penalties, meanZ, tangent maps, callback, snapshots/restores, VTK dumps and
    # saves). boozer_surfaces / axes / xpoints hold both configurations; sing_fls keeps the
    # singular field line(s) of config 2 only (aux-coil penalties, discriminant).
    # --target snowflake: the snowflake configuration alone (config 1, built above, is left out)
    if SNOW_ONLY:
        xpoints = sing_fls
    else:
        boozer_surfaces = [bs_x] + boozer_surfaces
        axes = [axis_x] + axes
        xpoints = [xp_x] + sing_fls

FIELDS = ([] if SNOW_ONLY else [lambda: BiotSavart(coils_x)]) + [(lambda fl=fl: SingularBiotSavart(fl)) for fl in sing_fls]
iota_Gs = [[bs.res['iota'], bs.res['G']] for bs in boozer_surfaces]
IOTAS_TARGET = list(IOTAS_TARGET) * len(boozer_surfaces)   # the yaml target, for each configuration
# config 1: tr M >= TRACE_MIN; --trace-min, else the bound recorded by a previous run of this
# script (yaml TRACE_MIN), else 2 + boozer_all's --mono-thr. boozer_double_opt.py does not
# record its --trace-min: pass it for a double that was run with one.
TRACE_MIN = (args.trace_min if args.trace_min is not None
             else float(config.get('TRACE_MIN', 2.0 + float(config['MONODROMY_THRESHOLD']))))
config['TRACE_MIN'] = float(TRACE_MIN)
if SNOW_ONLY:
    print(f"--target snowflake: ONE configuration, the snowflake ({num_aux} aux coils); "
          f"iota targets {IOTAS_TARGET}")
else:
    print(f"two configurations: config 1 X-point (aux off, tr M >= {TRACE_MIN}), config 2 snowflake "
          f"({num_aux} aux coils); iota targets {IOTAS_TARGET}")

# Snapshot the INITIAL device WITH the auxiliary coils, BEFORE any design polishing.
# This device carries the aux coils but has not been optimized, so it may not satisfy
# all the design constraints; it is saved purely as an unpolished reference. prefix.sh
# copies it off scratch regardless of whether the polished device passes its gate.
os.makedirs(OUT_DIR, exist_ok=True)
save([boozer_surfaces, iota_Gs, axes, xpoints, sdf],
     OUT_DIR + f'design_double_unpolished_{DEVICE_TAG}.json')
with open(OUT_DIR + f'design_double_unpolished_{DEVICE_TAG}.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print(f"wrote {OUT_DIR}design_double_unpolished_{DEVICE_TAG}.json (initial device with aux coils, unpolished)")


## SET UP THE OPTIMIZATION PROBLEM AS A SUM OF OPTIMIZABLES ##
mrs = [MajorRadius(boozer_surface) for boozer_surface in boozer_surfaces]   # one per configuration
ls = [CurveLength(c) for c in base_curves]
brs = [BoozerResidual(boozer_surface, mk()) for boozer_surface, mk in zip(boozer_surfaces, FIELDS)]
J_major_radius = sum(QuadraticPenalty(mr, MR_TARGET, 'identity') for mr in mrs)  # same target for both configurations

IOTAS_LIST = [Iotas(boozer_surface) for boozer_surface in boozer_surfaces]
J_iotas = sum([QuadraticPenalty(IOTAS, IOT_TARGET, 'identity') for IOTAS, IOT_TARGET in zip(IOTAS_LIST, IOTAS_TARGET)]) # target rotational transform is that computed on the initial surface

# On-axis rotational transform via the tangent map on the magnetic axis (utils/tangent_map.py):
# AxisIota.J()/dJ() return the on-axis iota and its coil gradient. The diagnostic is ALWAYS
# built so it can be reported in summary.txt regardless of the knob; only --AR 1/2
# (axis_iota_enabled) PENALIZE it. When active the target is loaded from the yaml
# (config['AXIS_IOTA_TARGET'], written by boozer_all as the original unperturbed-config
# value); a fresh value is computed only as a fallback for an old yaml that lacks the key.
axis_tmos = [TangentMap(axis, mk(), 0.0, mtype='identity')
             for axis, mk in zip(axes, FIELDS)]
AXIS_IOTA_LIST = [AxisIota(tmo) for tmo in axis_tmos]
if axis_iota_enabled:
    if 'AXIS_IOTA_TARGET' not in config:
        config['AXIS_IOTA_TARGET'] = [float(AXIS_IOTA_LIST[0].J())]
    # the yaml holds ONE target (boozer_all's configuration): use it for each configuration
    AXIS_IOTA_TARGET = list(config['AXIS_IOTA_TARGET'])[:1] * len(axes)
    J_axis_iota = sum([QuadraticPenalty(aio, tgt, 'identity')
                       for aio, tgt in zip(AXIS_IOTA_LIST, AXIS_IOTA_TARGET)])
else:
    AXIS_IOTA_TARGET, J_axis_iota = [], None

# Near-axis elongation bound, as in boozer_all.py (its --elongation, read here from the yaml
# as ELONGATION_MAX; None or absent: no bound): the elongation of the flux-surface
# cross-section in the plane normal to the magnetic axis is kept <= ELONGATION_MAX at
# N_ELONGATION toroidal angles (a full field period, or half of one for a stellarator-symmetric
# axis), for BOTH configurations, each with its own field.
N_ELONGATION = 16
ELONGATION_MAX = config.get('ELONGATION_MAX')
elongation_enabled = ELONGATION_MAX is not None
if elongation_enabled:
    ELONGATION_MAX = float(ELONGATION_MAX)
    ELONGATION_WEIGHT = Weight(config.setdefault('ELONGATION_WEIGHT', 1.0))
    elong_tmos = []
    for axis, mk in zip(axes, FIELDS):
        _nfp_e, _stellsym_e = axis.curve.nfp, axis.curve.stellsym
        _phis_e = (np.linspace(0., 0.5/_nfp_e, N_ELONGATION) if _stellsym_e
                   else np.linspace(0., 1./_nfp_e, N_ELONGATION, endpoint=False))
        elong_tmos += [TangentMap(axis, mk(), 0.0, mtype='identity', phi=float(phi)) for phi in _phis_e]
    ELONGATION_LIST = [AxisElongation(tmo) for tmo in elong_tmos]
    J_elongation = sum([QuadraticPenalty(e, ELONGATION_MAX, 'max') for e in ELONGATION_LIST])
else:
    ELONGATION_WEIGHT, ELONGATION_LIST, J_elongation = Weight(0.0), [], None

# Risk-neutral bound on the unfolding of the snowflake: E[C] <= EC_MAX, with E[C] the mean of
# the unfolding constant C over the Gaussian-process perturbations of the modular coils, in the
# (R, Z) section at phi = 0, computed exactly (ExactMeanUnfoldingC: no realizations). One-sided
# quadratic penalty with its own escalating weight, like the other inequality constraints.
# Without --ec-max there is no bound: E[C] is only reported (tables, summary), the penalty is
# not built and nothing of it enters the objective.
EC_WEIGHT = Weight(config.setdefault('EC_WEIGHT', 10.0)) if EC_ENABLED else Weight(0.0)
# --ec-section: the bound applies to E[C] in ONE section (the fraction T of the full turn of the
# snowflake field line, default 0: phi = 0) or to its MEAN over the sections. Recorded in the
# yaml, so that a restart keeps the choice.
_sec = args.ec_section if args.ec_section is not None else config.get('EC_SECTION', 0.0)
EC_USE_MEAN = str(_sec).strip().lower() == 'mean'
EC_T = 0.0 if EC_USE_MEAN else float(_sec)
if not 0.0 <= EC_T < 1.0:
    raise SystemExit(f"ERROR: --ec-section must be 'mean' or a number in [0, 1), got {_sec!r}")
config['EC_SECTION'] = 'mean' if EC_USE_MEAN else EC_T
EC_LABEL = 'mean over the sections' if EC_USE_MEAN else f'section t = {EC_T:g}'
# EC: E[C] in the section at t = EC_T; EC_PHI: its profile over the sections and their mean
# (transported from that section); EC_OBJ: the one of the two that is bounded.
EC = ExactMeanUnfoldingC(sing_fls[0], sing_fls[0].biotsavart,
                         length_scale=float(args.risk_length_scale), frame='RZ', phi0=EC_T)
EC_PHI = PhiMeanUnfoldingC(EC)
EC_OBJ = EC_PHI if EC_USE_MEAN else EC
J_EC = QuadraticPenalty(EC_OBJ, EC_MAX, 'max') if EC_ENABLED else None
_C0 = EC.values()
print(f"E[C], exact expectation over the GP perturbations (L = {args.risk_length_scale}), in the section "
      f"t = {EC_T:g} at the start: {np.nanmean(_C0):.6f}; std[C] = {EC.std():.5f}; "
      f"covariance of D_sigma = {EC.sigma().ravel()}; "
      f"{int(np.isfinite(_C0).sum())} of {len(_C0)} directions with a real split")
if EC_ENABLED:
    print(f"bound E[C] <= {EC_MAX} applied to: {EC_LABEL}")
else:
    print("no bound on E[C] (--ec-max not given): E[C] is reported only")


def ec_phi_stats():
    """(mean, min, max) over the sections of E[C], or nans when it is undefined."""
    try:
        _p = EC_PHI.profile()[1]
        return EC_PHI.J(), float(np.min(_p)), float(np.max(_p))
    except Exception:
        return float('nan'), float('nan'), float('nan')


def ec_bounded_value():
    """The bounded E[C] (EC_OBJ: one section or the mean), or nan when it is undefined."""
    try:
        return float(EC_OBJ.J())
    except Exception:
        return float('nan')


print("E[C] over the toroidal sections at the start: mean %.6f, min %.6f, max %.6f" % ec_phi_stats())
if args.check_gradient:
    # --check-gradient: central finite differences of E[C] along a random direction of its
    # dofs (modular-coil dofs and free aux dofs) against dJ, then stop.
    import time as _time
    _x0 = EC.x.copy()
    _t = _time.time(); _J0 = EC.J(); _tJ = _time.time() - _t
    _t = _time.time(); _g = EC.dJ(); _tg = _time.time() - _t
    _h = np.random.default_rng(1).standard_normal(_x0.size); _h /= np.linalg.norm(_h)
    print(f"gradient check of E[C]: {_x0.size} dofs, J = {_J0:.8f} ({_tJ:.1f} s), dJ.h = {_g @ _h:.10e} ({_tg:.1f} s)")
    # timing of a value and of a gradient, once their functions are compiled
    EC._solution = None
    _t = _time.time(); EC.J(); _tv = _time.time() - _t
    _t = _time.time(); EC.dJ(); _tg2 = _time.time() - _t
    print(f"   exact E[C]: value {_tv:.2f} s, gradient {_tg2:.2f} s (after compilation)")
    # small steps only: the snowflake solution moves by ~4e3 per unit step of the dofs, so
    # larger steps are not in the linear regime (and the Newton solve may fail)
    for _eps in (1e-9, 2e-9, 5e-9, 1e-8, 2e-8):
        _vals = []
        for _s in (+1.0, -1.0):
            EC.x = _x0; EC.values()   # every evaluation starts from the nominal snowflake solution
            EC.x = _x0 + _s * _eps * _h
            _Cs = EC.values()
            _r = sing_fls[0].res
            _vals.append((np.mean(_Cs), int(np.isfinite(_Cs).sum()), bool(_r['success']),
                          float(np.abs(np.asarray(_r['monodromy_matrix']) - np.eye(2)).max())))
        _fd = (_vals[0][0] - _vals[1][0]) / (2 * _eps)
        print(f"   eps = {_eps:.0e}: FD = {_fd:.10e}, relative error = {abs(_fd - _g @ _h) / abs(_g @ _h):.2e} | "
              f"real splits {_vals[0][1]}/{_vals[1][1]} of {len(_Cs)}, snowflake solved {_vals[0][2]}/{_vals[1][2]}, "
              f"|M - I| {_vals[0][3]:.1e}/{_vals[1][3]:.1e}")
    EC.x = _x0
    # ---- the same for the mean over the toroidal sections (PhiMeanUnfoldingC)
    _xp = EC_PHI.x.copy()
    _hp = np.random.default_rng(2).standard_normal(_xp.size); _hp /= np.linalg.norm(_hp)
    EC_PHI.J(); EC_PHI.dJ()                                   # compilation
    EC_PHI._solution = None; EC._solution = None
    _t = _time.time(); _Jp = EC_PHI.J(); _tJp = _time.time() - _t
    _t = _time.time(); _gp = EC_PHI.dJ(); _tgp = _time.time() - _t
    _tt, _prof = EC_PHI.profile()
    print(f"gradient check of the mean over the sections of E[C]: {_xp.size} dofs, J = {_Jp:.8f} ({_tJp:.2f} s), "
          f"dJ.h = {_gp @ _hp:.10e} ({_tgp:.2f} s); profile: {len(_tt)} nodes, min {_prof.min():.6f}, "
          f"max {_prof.max():.6f}, at the first node {_prof[0]:.6f} and the last {_prof[-1]:.6f} (E[C] at phi = 0: {_J0:.6f})")
    for _eps in (1e-9, 2e-9, 5e-9, 1e-8, 2e-8):
        _vp = []
        for _s in (+1.0, -1.0):
            EC_PHI.x = _xp; EC_PHI.J()                        # start from the nominal snowflake solution
            EC_PHI.x = _xp + _s * _eps * _hp
            _vp.append(EC_PHI.J())
        _fd = (_vp[0] - _vp[1]) / (2 * _eps)
        print(f"   eps = {_eps:.0e}: FD = {_fd:.10e}, relative error = {abs(_fd - _gp @ _hp) / abs(_gp @ _hp):.2e}")
    EC_PHI.x = _xp
    # the transported profile against E[C] computed directly in a few sections (an
    # ExactMeanUnfoldingC with its own phi0: jet and covariance of that section)
    _tt, _prof = EC_PHI.profile()
    for _k in sorted({len(_tt) // 8, len(_tt) // 4, len(_tt) // 2, (3 * len(_tt)) // 4}):
        _phi0 = (EC_T + float(_tt[_k])) % 1.0     # the nodes are counted from the section of EC
        _direct = ExactMeanUnfoldingC(sing_fls[0], sing_fls[0].biotsavart,
                                      length_scale=float(args.risk_length_scale), frame='RZ', phi0=_phi0).J()
        print(f"   section at t = {_phi0:.5f} (node {_k}): transported E[C] = {_prof[_k]:.8f}, direct = {_direct:.8f}, "
              f"difference {abs(_prof[_k] - _direct):.2e}")
    raise SystemExit(0)
nonQS_list = [NonQuasiSymmetricRatio(boozer_surface, mk()) for boozer_surface, mk in zip(boozer_surfaces, FIELDS)]
print([J.J()**0.5 for J in nonQS_list])
J_nonQSRatio = (1./len(boozer_surfaces)) * sum(nonQS_list)

Jls = [CurveLength(c) for c in base_curves]
Jccdist = CurveCurveDistance(curves, COIL_TO_COIL_THRESHOLD)
Jcs = [LpCurveCurvature(c, 2, CURVATURE_THRESHOLD) for c in base_curves]
Jmscs = [MeanSquaredCurvature(c) for c in base_curves]
Jal = sum(ArclengthVariation(curve) for curve in base_curves)

length_penalty = sum([QuadraticPenalty(Jl, LENGTH_THRESHOLD, "max") for Jl in Jls])
curvature_penalty = sum(Jcs)
msc_penalty = sum(QuadraticPenalty(J, MEAN_SQUARED_CURVATURE_THRESHOLD, "max") for J in Jmscs)
Jbrs = sum(brs)

# penalty on deviation from target mean field strength
modBs = [ModBOnFieldLine(axis, mk()) for axis, mk in zip(axes, FIELDS)]
JmodB = sum([QuadraticPenalty(modB, MODB_TARGET, 'identity') for modB, axis, boozer_surface in zip(modBs, axes, boozer_surfaces)])

# penalty keeping |B| within +-MODB_RIPPLE_THRESHOLD of its mean ALONG THE AXIS, in
# the TOTAL (modular + aux) field. NEW config keys (boozer_all did not write them);
# an existing yaml may override so escalation survives a restart.
config.setdefault('MODB_RIPPLE_THRESHOLD', 0.005)  # 0.5% (0.1% up to campaign 7463849); the yaml written by boozer_all wins
config.setdefault('MODB_RIPPLE_WEIGHT', 1e2)
MODB_RIPPLE_THRESHOLD = config['MODB_RIPPLE_THRESHOLD']
MODB_RIPPLE_WEIGHT = Weight(config['MODB_RIPPLE_WEIGHT'])
modB_ripples = [ModBRippleOnFieldLine(axis, mk(), MODB_RIPPLE_THRESHOLD)
                for axis, mk in zip(axes, FIELDS)]
J_modB_ripple = sum(modB_ripples)

J_curr = None
for boozer_surface in boozer_surfaces:
    for idx in base_curve_idx:
        curr = boozer_surface.biotsavart.coils[idx].current
        if J_curr is None:
            J_curr = CurrentBound(curr.current_to_scale, CURRENT_THRESHOLD/curr.scale)
        else:
            J_curr += CurrentBound(curr.current_to_scale, CURRENT_THRESHOLD/curr.scale)


# Maximum MODULAR coil force per unit length (designB), as in boozer_all.py but in the
# TOTAL field: |dF/dl| = |I t x (B_self,reg + B_other modular coils + B_aux)| on every
# modular coil must stay below FORCE_THRESHOLD (N/m). Threshold and (possibly
# escalated) weight are read from the boozer_all yaml. TotalFieldLpCurveForce
# (utils/total_field_coil_force.py) adds the field of the auxiliary coils of each
# singular field line (its mu dofs) and differentiates through it: directly for the
# free mu, and through the adjoint of the singular polish for the dependent aux
# currents. The self-force is regularized for a circular cross section of radius
# COIL_MINOR_RADIUS. Its threshold is in MN/m (as simsopt's LpCurveForce), hence the /1e6.
FORCE_THRESHOLD = config['FORCE_THRESHOLD']
FORCE_WEIGHT = Weight(config['FORCE_WEIGHT'])
force_objs = [TotalFieldLpCurveForce([coils[i] for i in base_curve_idx], coils, fl,
                                     regularization_circ(COIL_MINOR_RADIUS), p=2,
                                     threshold=FORCE_THRESHOLD / 1e6)
              for fl in sing_fls]
# Config 1 has no aux field: boozer_all.py's force objective (LpCurveForce on the base coils,
# regularized self-force + the other modular coils), with config 1's own currents.
_force_reg = regularization_circ(COIL_MINOR_RADIUS)
force_targets_x = [RegularizedCoil(coils_x[i].curve, coils_x[i].current, _force_reg) for i in base_curve_idx]
force_sources_x = [c for j, c in enumerate(coils_x) if j not in base_curve_idx]
J_force_x = LpCurveForce(force_targets_x, force_sources_x, p=2, threshold=FORCE_THRESHOLD / 1e6)
J_force = sum(force_objs) if SNOW_ONLY else J_force_x + sum(force_objs)


def max_coil_force():
    """Largest modular-coil Lorentz force per unit length |dF/dl| (N/m) over both
    configurations (config 2: total field; config 1: modular field), or over the snowflake
    configuration alone (--target snowflake)."""
    if SNOW_ONLY:
        return max(fo.max_force() for fo in force_objs)
    f_x = max(float(np.max(np.linalg.norm(t.force([c for j, c in enumerate(coils_x) if j != i]), axis=1)))
              for t, i in zip(force_targets_x, base_curve_idx))
    return max(f_x, max(fo.max_force() for fo in force_objs))


# AUX coil currents: BOTH dependent and independent aux currents are constrained
# to -AUX_CURRENT_EPS <= I <= AUX_CURRENT_EPS, sharing ONE penalty weight
# (AUX_CURRENT_WEIGHT) in a single objective J_aux_current. Dependent currents go
# through DependentMu (their value is an implicit function of the design vars, so
# the gradient must flow through the adjoint); independent currents are free mu
# dofs handled directly by MuBound. (mu units: EPS/_CURRENT_SCALE; EPS is in A.)
if args.aux_current_max is not None:       # --aux-current-max wins over the yaml
    config['AUX_CURRENT_EPS'] = float(args.aux_current_max)
config.setdefault('AUX_CURRENT_EPS', 5e3)  # Amperes
config.setdefault('AUX_CURRENT_WEIGHT', 1e2)
AUX_CURRENT_EPS = config['AUX_CURRENT_EPS']
AUX_CURRENT_WEIGHT = Weight(config['AUX_CURRENT_WEIGHT'])
_aux_eps_mu = AUX_CURRENT_EPS / _CURRENT_SCALE   # bound in mu units
dependent_mus = [DependentMu(fl, name) for fl in sing_fls for name in DEP_CURRENT_NAMES]
J_aux_current = sum(QuadraticPenalty(dm, +_aux_eps_mu, 'max')
                    + QuadraticPenalty(dm, -_aux_eps_mu, 'min')
                    for dm in dependent_mus)
for fl in sing_fls:
    for name in INDEP_CURRENT_NAMES:
        J_aux_current += MuBound(fl, name, _aux_eps_mu)


# AUX coil <-> MODULAR coil clearance: keep every modular coil at least
# AUX_COIL_DISTANCE_THRESHOLD away from the planar circular aux coils, sharing one
# escalating weight (AUX_COIL_DISTANCE_WEIGHT) via J_aux_coil_distance. The aux
# coils are mu dofs (not Curves), so AuxCoilDistance reads their radii/Z from
# fl.mu directly (closed-form point-to-circle distance) and returns gradients to
# BOTH the modular-coil dofs and the aux (mu) dofs. AUX_COIL_DISTANCE_THRESHOLD is
# set above (it also fixes the initial aux-coil Z); here we just default/read the
# starting weight, letting an existing yaml override so escalation survives a restart.
config.setdefault('AUX_COIL_DISTANCE_WEIGHT', 1e5)
AUX_COIL_DISTANCE_WEIGHT = Weight(config['AUX_COIL_DISTANCE_WEIGHT'])
aux_coil_distances = [AuxCoilDistance(fl, curves, AUX_COIL_DISTANCE_THRESHOLD) for fl in sing_fls]
J_aux_coil_distance = sum(aux_coil_distances)


# AUX coil RADII: cap each planar circular aux coil's radius at AUX_RADIUS_MAX
# metres. The radii r1..rN are free mu design vars (stage-2 fixes only the
# dependent currents), so MuBound applies directly; MuBound penalizes |r| > max,
# which for a (positive) radius enforces r <= AUX_RADIUS_MAX. They share one
# escalating weight (AUX_RADIUS_WEIGHT) in J_aux_radius. NEW keys (boozer_all did
# not write them); an existing yaml may override so escalation survives a restart.
config.setdefault('AUX_RADIUS_MAX', 1.2)
config.setdefault('AUX_RADIUS_WEIGHT', 1e2)
AUX_RADIUS_MAX = config['AUX_RADIUS_MAX']
AUX_RADIUS_WEIGHT = Weight(config['AUX_RADIUS_WEIGHT'])
J_aux_radius = sum(MuBound(fl, f'r{k+1}', AUX_RADIUS_MAX)
                   for fl in sing_fls for k in range(num_aux))





plasma_boundary_entities = xpoints + boozer_surfaces
plasma_boundary_signs = np.array([-1.0 for _ in xpoints] + [-1.0 for _ in boozer_surfaces])
J_plasma_to_vessel_margin = VesselDistance(sdf, plasma_boundary_entities, plasma_boundary_signs, PLASMA_VESSEL_MARGIN_THRESHOLD)

coil_on_vessel_entities = base_curves + base_curves
coil_on_vessel_signs = np.array(
    [-1.0 for _ in base_curves] + [1.0 for _ in base_curves]
)
J_coil_on_vessel = VesselDistance(
    sdf,
    coil_on_vessel_entities,
    coil_on_vessel_signs,
    COIL_ON_VESSEL_THRESHOLD,
)

coil_clearance_entities = base_curves
coil_clearance_signs = np.array([1.0 for _ in base_curves])
J_coil_clearance = VesselDistance(
    sdf,
    coil_clearance_entities,
    coil_clearance_signs,
    COIL_CLEARANCE_THRESHOLD,
)

magnetic_wells = [MagneticWell(axis, boozer_surface, WELL_THRESHOLD) for axis, boozer_surface in zip(axes, boozer_surfaces)]
J_wells = sum(magnetic_wells)

meanzs = [FieldLineMeanZ(xpoint, FIELDLINE_MEANZ_THRESHOLD) for xpoint in xpoints]
J_meanz = sum(meanzs)

fieldline_mean_distance_signs = np.array([-1.0 for _ in xpoints])
fieldline_mean_distance_entities = xpoints

J_fieldline_mean_distance = VesselDistance(
    sdf,
    fieldline_mean_distance_entities,
    fieldline_mean_distance_signs,
    FIELDLINE_MEANDIST_THRESHOLD,
    metric='distance',
)

# Snowflake discriminant Delta of the reduced return-map quadratic jet on each
# singular field line (utils/discriminant.py). Delta > 0 <-> 6-legged snowflake;
# its sign/size decides the leg structure. Built for ALL polishes (reported in the
# callback), but CONSTRAINED only for the identity (monodromy = I) polish, where a
# genuine snowflake is the target: keep Delta >= DISCRIMINANT_THRESHOLD via a
# one-sided QuadraticPenalty (penalizes Delta < threshold). SnowflakeDiscriminant
# takes the MODULAR field (it adds the aux field internally and needs BiotSavart's
# d2B/d3B vjps); each gets its own BiotSavart(coils) so point caches stay independent.
discriminants = [SnowflakeDiscriminant(fl, BiotSavart(fl.biotsavart.coils), frame='RZ')
                 for fl in sing_fls]
# The constraint is opt-in (--discriminant 1) and only defined for the identity polish.
if discriminant == 1 and mon_constraint != 'identity':
    print("WARNING: --discriminant 1 ignored: the snowflake-discriminant constraint is "
          "only defined for the identity (monodromy = I) polish.")
discriminant_enabled = (discriminant == 1 and mon_constraint == 'identity')
if discriminant_enabled:
    config.setdefault('DISCRIMINANT_THRESHOLD', 1e4)
    # 1e-6 (was 1e-8): a collapsed snowflake (Delta ~ 0) now costs ~0.5 * 1e8 * 1e-6 = 50, more
    # than any other pre-scaled term (<= 1), so the optimizer protects Delta first; with 1e-8 it
    # cost only 0.5 and 59 of the 177 out-of-spec polishes of campaign 7463849 ended at Delta ~ 0.
    config.setdefault('DISCRIMINANT_WEIGHT', 1e-6)
    DISCRIMINANT_THRESHOLD = config['DISCRIMINANT_THRESHOLD']
    DISCRIMINANT_WEIGHT = Weight(config['DISCRIMINANT_WEIGHT'])
    J_discriminant = sum(QuadraticPenalty(d, DISCRIMINANT_THRESHOLD, 'min') for d in discriminants)
else:
    DISCRIMINANT_THRESHOLD, DISCRIMINANT_WEIGHT, J_discriminant = None, None, None

# Config 1: keep the ordinary X-point HYPERBOLIC, tr M >= TRACE_MIN = 2 + MONODROMY_THRESHOLD,
# with boozer_all.py's approach to its monodromy constraint (a TangentMap penalty on the
# X-point field line, weight MONODROMY_WEIGHT read from the yaml and escalated below), here
# with mtype='trace_min': penalty max(TRACE_MIN - tr M, 0)**2, zero once the bound holds.
config.setdefault('MONODROMY_WEIGHT', 1e-2)
if SNOW_ONLY:     # no config 1: no X-point trace, and no weight to escalate
    MONODROMY_WEIGHT = Weight(0.0)
    trace_tmo, J_trace = None, None
else:
    MONODROMY_WEIGHT = Weight(config['MONODROMY_WEIGHT'])
    trace_tmo = TangentMap(xp_x, BiotSavart(coils_x), TRACE_MIN, mtype='trace_min')
    J_trace = Monodromy(trace_tmo)


def xpoint_trace():
    """tr M of config 1's X-point (nan with --target snowflake: there is no config 1)."""
    if SNOW_ONLY:
        return float('nan')
    return float(np.trace(np.asarray(trace_tmo.matrix)))


# sum the objectives together (config 2 has no monodromy penalty: its singular polish
# enforces M = I exactly).
JF = (J_nonQSRatio
    + IOTAS_WEIGHT * J_iotas
    + MAJOR_RADIUS_WEIGHT * J_major_radius
    + LENGTH_WEIGHT * length_penalty
    + COIL_TO_COIL_WEIGHT * Jccdist
    + CURVATURE_WEIGHT * curvature_penalty
    + MEAN_SQUARED_CURVATURE_WEIGHT * msc_penalty
    + ARCLENGTH_WEIGHT * Jal
    + BOOZER_RESIDUAL_WEIGHT * Jbrs
    + MODB_WEIGHT * JmodB
    + MODB_RIPPLE_WEIGHT * J_modB_ripple
    + PLASMA_VESSEL_MARGIN_WEIGHT * J_plasma_to_vessel_margin
    + COIL_ON_VESSEL_WEIGHT * J_coil_on_vessel
    + COIL_CLEARANCE_WEIGHT * J_coil_clearance
    + CURRENT_WEIGHT * J_curr
    + WELL_WEIGHT * J_wells
    + FIELDLINE_MEANZ_WEIGHT * J_meanz
    + FIELDLINE_MEANDIST_WEIGHT * J_fieldline_mean_distance
    + AUX_CURRENT_WEIGHT * J_aux_current
    + AUX_COIL_DISTANCE_WEIGHT * J_aux_coil_distance
    + AUX_RADIUS_WEIGHT * J_aux_radius
    + FORCE_WEIGHT * J_force
    )
# config 1 X-point trace bound (not with --target snowflake: there is no config 1)
if not SNOW_ONLY:
    JF = JF + MONODROMY_WEIGHT * J_trace
# On-axis iota constraint (only for --AR 1/2 devices).
if axis_iota_enabled:
    JF = JF + AXIS_IOTA_WEIGHT * J_axis_iota
# Near-axis elongation bound of both configurations (only when the yaml sets ELONGATION_MAX).
if elongation_enabled:
    JF = JF + ELONGATION_WEIGHT * J_elongation
# Risk-neutral bound E[C] <= EC_MAX on the unfolding constant of the snowflake (--ec-max).
if EC_ENABLED:
    JF = JF + EC_WEIGHT * J_EC

# Snowflake discriminant constraint Delta > DISCRIMINANT_THRESHOLD (--discriminant 1, identity only).
if discriminant_enabled:
    JF = JF + DISCRIMINANT_WEIGHT * J_discriminant


penalties = {'nonQS': J_nonQSRatio,
        'iotas':IOTAS_WEIGHT * J_iotas,
        'length':LENGTH_WEIGHT * length_penalty,
        'coil-to-coil': COIL_TO_COIL_WEIGHT * Jccdist,
        'curvature':CURVATURE_WEIGHT * curvature_penalty,
        'mean-squared curvature': MEAN_SQUARED_CURVATURE_WEIGHT * msc_penalty,
        'arclength':ARCLENGTH_WEIGHT * Jal,
        'Boozer residual': BOOZER_RESIDUAL_WEIGHT * Jbrs,
        'modB': MODB_WEIGHT * JmodB,
        'modB axis ripple': MODB_RIPPLE_WEIGHT * J_modB_ripple,
        'plasma-boundary-to-vessel': PLASMA_VESSEL_MARGIN_WEIGHT * J_plasma_to_vessel_margin,
        'coil-on-vessel': COIL_ON_VESSEL_WEIGHT * J_coil_on_vessel,
        'coil-clearance': COIL_CLEARANCE_WEIGHT * J_coil_clearance,
        'current':CURRENT_WEIGHT * J_curr,
        'well':WELL_WEIGHT * J_wells,
        'fieldline meanz': FIELDLINE_MEANZ_WEIGHT * J_meanz,
        'fieldline mean dist':FIELDLINE_MEANDIST_WEIGHT * J_fieldline_mean_distance,
        'major radius': MAJOR_RADIUS_WEIGHT * J_major_radius,
        'aux current': AUX_CURRENT_WEIGHT * J_aux_current,
        'aux coil distance': AUX_COIL_DISTANCE_WEIGHT * J_aux_coil_distance,
        'aux radius': AUX_RADIUS_WEIGHT * J_aux_radius,
        'max force': FORCE_WEIGHT * J_force,
        **({} if SNOW_ONLY else {'xpoint trace': MONODROMY_WEIGHT * J_trace}),
        }

# Weight object backing each penalty above, keyed identically. 'nonQS' has an
# implicit weight of 1 (no Weight object) and is the primary objective, so it is
# omitted and never pre-scaled.
penalty_weights = {
        'iotas': IOTAS_WEIGHT,
        'length': LENGTH_WEIGHT,
        'coil-to-coil': COIL_TO_COIL_WEIGHT,
        'curvature': CURVATURE_WEIGHT,
        'mean-squared curvature': MEAN_SQUARED_CURVATURE_WEIGHT,
        'arclength': ARCLENGTH_WEIGHT,
        'Boozer residual': BOOZER_RESIDUAL_WEIGHT,
        'modB': MODB_WEIGHT,
        'modB axis ripple': MODB_RIPPLE_WEIGHT,
        'plasma-boundary-to-vessel': PLASMA_VESSEL_MARGIN_WEIGHT,
        'coil-on-vessel': COIL_ON_VESSEL_WEIGHT,
        'coil-clearance': COIL_CLEARANCE_WEIGHT,
        'current': CURRENT_WEIGHT,
        'well': WELL_WEIGHT,
        'fieldline meanz': FIELDLINE_MEANZ_WEIGHT,
        'fieldline mean dist': FIELDLINE_MEANDIST_WEIGHT,
        'major radius': MAJOR_RADIUS_WEIGHT,
        'aux current': AUX_CURRENT_WEIGHT,
        'aux coil distance': AUX_COIL_DISTANCE_WEIGHT,
        'aux radius': AUX_RADIUS_WEIGHT,
        'max force': FORCE_WEIGHT,
        **({} if SNOW_ONLY else {'xpoint trace': MONODROMY_WEIGHT}),
        }

states = {
        'iotas': IOTAS_LIST,
        'modB': modBs,
        'lengths':Jls,
        'major radius': [MajorRadius(boozer_surface) for boozer_surface in boozer_surfaces],
        'Boozer residuals': brs,
        'mean-squared curvature': Jmscs
        }

# On-axis iota enters the objective / escalation / callback table ONLY when penalized
# (--AR 1/2). For --AR 0 it is still reported in summary.txt (computed once at the end),
# but kept out of these so its tangent-map adjoint is never run per BFGS iteration.
if axis_iota_enabled:
    penalties['on-axis iota'] = AXIS_IOTA_WEIGHT * J_axis_iota
    penalty_weights['on-axis iota'] = AXIS_IOTA_WEIGHT
    states['on-axis iota'] = AXIS_IOTA_LIST
# Likewise the near-axis elongation bound (its maxima are printed in the callback table).
if elongation_enabled:
    penalties['elongation'] = ELONGATION_WEIGHT * J_elongation
    penalty_weights['elongation'] = ELONGATION_WEIGHT
# the risk-neutral bound on E[C] (its value is printed in the callback table, see below)
if EC_ENABLED:
    penalties['E[C]'] = EC_WEIGHT * J_EC
    penalty_weights['E[C]'] = EC_WEIGHT

# Snowflake discriminant constraint enters the objective / escalation / weight-scaling
# tables ONLY when enabled (--discriminant 1, identity polish). Its VALUE is still
# printed in the callback either way.
if discriminant_enabled:
    penalties['snowflake discriminant'] = DISCRIMINANT_WEIGHT * J_discriminant
    penalty_weights['snowflake discriminant'] = DISCRIMINANT_WEIGHT

# fix some currents
for bbsurf in boozer_surfaces:
    for coil in bbsurf.biotsavart.coils:
        coil.current.unfix_all()
    dn = bbsurf.biotsavart.dof_names
    print('free currents:', [c for c in dn if 'current' in c.lower() ])

# make sure coils are stellarator symmetric (DN only). In designB BOTH base coils
# are their own stellsym image, so the stellsym-violating dofs of both are fixed.
# For SN designs the base coils are deliberately general (the device is no longer
# stellsym).
if boozer_surfaces[0].surface.stellsym:
    for ii in base_curve_idx:
        c = boozer_surfaces[0].biotsavart.coils[ii].curve
        if isinstance(c, RotatedCurve):
            c = c.curve
        for df in c.local_dof_names:
            if ('xs' in df) or ('yc' in df) or ('zc' in df):
                c.fix(df)

print(JF.dof_names, JF.x.size)


# Directory for output
os.makedirs(OUT_DIR, exist_ok=True)

curves_to_vtk(curves, OUT_DIR + "curves_init")
curves_to_vtk([fl.curve for fl in sing_fls], OUT_DIR + "xpoint_singular_curves_init")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_init_{idx}")

# save these as a backup in case the boozer surface Newton solve fails
res_list = [{'sdofs': boozer_surface.surface.x.copy() , 'iota': boozer_surface.res['iota'], 'G': boozer_surface.res['G']} for boozer_surface in boozer_surfaces]
axes_res_list = [{'adofs': axis.curve.x.copy() , 'length': axis.res['length']} for axis in axes]
xpoints_res_list = [{'xdofs': xpoint.curve.x.copy() , 'length': xpoint.res['length']} for xpoint in xpoints]
# the singular-polish field lines re-solve their own curve copies and the
# dependent mu each evaluation; snapshot/restore them like the axis/x-point.
sing_res_list = [{'xdofs': fl.curve.x.copy(), 'mu': fl.mu.copy(), 'length': fl.res['length']} for fl in sing_fls]
dat_dict = {'iter':0, 'J': JF.J(), 'dJ': JF.dJ().copy(), 'x': JF.x.copy()}
# last successful evaluation of fun(): its point and its gradient, so that callback() does not
# recompute the gradient at that same point
_last_eval = {}


def _add_ec_rows(table):
    """Rows of E[C] of the per-iteration tables: E[C] in the section when it carries the bound,
    and always its mean, minimum and maximum over the sections."""
    _bnd = f'bound: <= {EC_MAX}' if EC_ENABLED else 'not bounded'
    if EC_ENABLED and not EC_USE_MEAN:    # with the bound on the mean, only the row of the sections is shown
        _Cs = EC.values()
        table.add_row(f'E[C] unfolding (t={EC_T:g})', f'{np.nanmean(_Cs):.5f}  min {np.nanmin(_Cs):.4f}  max {np.nanmax(_Cs):.4f}  '
                      f'({int(np.isfinite(_Cs).sum())} of {len(_Cs)} directions with a real split; {_bnd})')
    table.add_row('E[C] over the sections', 'mean %.5f  min %.4f  max %.4f' % ec_phi_stats()
                  + f'  ({"mean " + _bnd if EC_USE_MEAN and EC_ENABLED else "not bounded"})')


def callback(dofs, full=False):
    for res, boozer_surface in zip(res_list, boozer_surfaces):
        res['sdofs'] = boozer_surface.surface.x.copy()
        res['iota'] =  boozer_surface.res['iota']
        res['G'] = boozer_surface.res['G']
    for res, axis in zip(axes_res_list, axes):
        res['adofs'] = axis.curve.x.copy()
        res['length'] =  axis.res['length']
    for res, axis in zip(xpoints_res_list, xpoints):
        res['xdofs'] = axis.curve.x.copy()
        res['length'] =  axis.res['length']
    for res, fl in zip(sing_res_list, sing_fls):
        res['xdofs'] = fl.curve.x.copy()
        res['mu'] = fl.mu.copy()
        res['length'] = fl.res['length']

    dist = np.linalg.norm(dofs - dat_dict['x'])

    dat_dict['x'] = JF.x.copy()
    dat_dict['J'] = JF.J()
    # same point AND same objective value: the weights may have changed since fun() ran (the
    # pre-scaling, an escalation), and its gradient is then stale
    if ('x' in _last_eval and _last_eval['J'] == dat_dict['J']
            and np.array_equal(_last_eval['x'], dat_dict['x'])):
        dat_dict['dJ'] = _last_eval['grad'].copy()     # the gradient fun() computed at this point
    else:
        dat_dict['dJ'] = JF.dJ().copy()

    currents_list = [np.abs(boozer_surface.biotsavart.coils[idx].current.get_value()) for boozer_surface in boozer_surfaces for idx in base_curve_idx]
    kappas = [np.max(c.kappa()) for c in base_curves]

    console = Console(width=250)
    table1 = Table(show_header=False)
    table1.add_row(*['iter', 'J', 'dJ', 'dist'])
    table1.add_row(*[f'{dat_dict["iter"]}', f'{dat_dict["J"]:.2e}', f'{np.max(np.abs(dat_dict["dJ"])):.2e}', f'{dist:.2e}'])
    console.print(table1)

    # The two full tables (every penalty, every state) are printed every 10 iterations, with
    # the saves at the end of this function, and when asked (full=True). In between, only the
    # line above and the rows of E[C].
    if not full and dat_dict["iter"] % 10 != 0:
        table2 = Table(expand=False, show_header=False)
        _add_ec_rows(table2)
        console.print(table2)
        dat_dict["iter"] += 1
        return

    console = Console(width=250)
    # Split the penalties over two stacked tables so the many columns don't get
    # truncated ("…") to fit the console width.
    pen_items = list(penalties.items())
    half = (len(pen_items) + 1) // 2
    for chunk in (pen_items[:half], pen_items[half:]):
        table1 = Table(expand=False, show_header=False)
        table1.add_row(*[k for k, _ in chunk])
        table1.add_row(*[f"{v.J():.4e}" for _, v in chunk])
        console.print(table1)

    table2 = Table(expand=False, show_header=False)
    for k in states.keys():
        table2.add_row(k, ' '.join([f'{J.J():.4e}' for J in states[k]]))
    table2.add_row('aspect ratio', ' '.join([f'{bsurf.surface.aspect_ratio():.4e}' for bsurf in boozer_surfaces]))
    table2.add_row('modB axis deviation', ' '.join([f'{r.max_deviation():.4e}' for r in modB_ripples]))
    if elongation_enabled:
        _el = [e.J() for e in ELONGATION_LIST]
        table2.add_row('axis elongation max', ' '.join([f'{max(_el[k:k+N_ELONGATION]):.4f}' for k in range(0, len(_el), N_ELONGATION)])
                       + f'  (bound: <= {ELONGATION_MAX})')
    _add_ec_rows(table2)
    table2.add_row('xpoint_top(0)', ' '.join([f'{np.array2string(xpoint.curve.gamma()[0])}' for xpoint in xpoints]))
    # auxiliary-coil parameters, per fl: summarize the currents (scaled to
    # Amperes) by min/max, separated into DEPENDENT (solved, bounded to
    # +-AUX_CURRENT_EPS) and INDEPENDENT (free design vars); list the radii + z.
    for fi, fl in enumerate(sing_fls):
        mu = np.asarray(fl.mu)
        dep_I = np.abs(mu[:N_DEP_CURRENTS]) * _CURRENT_SCALE
        indep_I = np.abs(mu[N_DEP_CURRENTS:num_aux]) * _CURRENT_SCALE
        radii = mu[num_aux:2 * num_aux]
        z = float(mu[-1])
        table2.add_row(f'aux dep |I| [{fi}] [A]',
                       f'min={dep_I.min():.4e}  max={dep_I.max():.4e}')
        table2.add_row(f'aux indep |I| [{fi}] [A]',
                       (f'min={indep_I.min():.4e}  max={indep_I.max():.4e}'
                        if indep_I.size else '(none)'))
        table2.add_row(f'aux r,z [{fi}]',
                       '  '.join([f'r{k+1}={radii[k]:+.4f}' for k in range(num_aux)] + [f'z={z:+.4f}']))
    table2.add_row('singular polish monodromy', ' '.join(
        [f'{np.array2string(np.asarray(fl.res["monodromy_matrix"]))}' for fl in sing_fls]))
    if not SNOW_ONLY:
        table2.add_row('config 1 X-point tr M', f'{xpoint_trace():.6f}  (bound: >= {TRACE_MIN})')
    # snowflake discriminant Delta per singular field line (value only; constrained
    # to Delta > DISCRIMINANT_THRESHOLD in the identity polish).
    table2.add_row('snowflake discriminant', ' '.join([f'{d.value():.4e}' for d in discriminants]))
    table2.add_row('well', ' '.join([f'{w.well().max():.3e}' for w in magnetic_wells]))
    table2.add_row('currents', ' '.join([f'{curr:.3e}' for curr in currents_list]))
    table2.add_row('curvatures', ' '.join([f'{curv:.3e}' for curv in kappas]))
    _, min_xpoint_to_vessel, min_boozer_surface_to_vessel = J_plasma_to_vessel_margin.shortest_distance()
    table2.add_row('minimum X-point-to-vessel distance', f'{min_xpoint_to_vessel:.3e}')
    table2.add_row('minimum Boozer surface-to-vessel distance', f'{min_boozer_surface_to_vessel:.3e}')

    min_coil_on_vessel_distance, _, _ = J_coil_on_vessel.shortest_distance()
    min_coil_clearance_distance, _, _ = J_coil_clearance.shortest_distance()
    table2.add_row('minimum coil-on-vessel distance', f'{min_coil_on_vessel_distance:.3e}')
    table2.add_row('minimum coil-clearance distance', f'{min_coil_clearance_distance:.3e}')

    # Zip the FULL dof names against the FULL dof vector: local_dof_names lists only
    # the FREE dofs, so if any dof is fixed (e.g. n0z for the SN welded-pipe vessel,
    # zc(0) for the non-stellsym helical vessel) enumerate-indexing local_full_x would
    # shift every entry past the fixed one by one (making rr read as a node coord).
    table2.add_row('vessel dimensions', ' '.join([f'{name}={val:.6e} ' for name, val in zip(sdf.local_full_dof_names, sdf.local_full_x)]))
    if isinstance(sdf, HelicalVesselSDF):
        # exact-SDF regime requires max_t kappa(t)*R(t) < 1 on the centerline;
        # arclength variation should stay near 0 (uniform parametrization)
        table2.add_row('max kappa*R (helical vessel)', f'{sdf.max_kappa_radius():.4e}')
        table2.add_row('max |dR/ds| (helical vessel)', f'{sdf.max_dr_ds():.4e}')
        table2.add_row('min radius (helical vessel)', f'{sdf.min_radius():.4e}')
        table2.add_row('centerline arclength variation', f'{sdf.arclength_variation():.4e}')
        # foot-point optimality residual max_p |phi'(t*)| on the plasma boundary;
        # ~0 (< FOOT_TOL) means every foot-point solve converged.
        _foot_gammas = np.concatenate(
            [xp.curve.gamma() for xp in xpoints]
            + [bsurf.surface.gamma().reshape((-1, 3)) for bsurf in boozer_surfaces])
        table2.add_row('foot-point residual max|phi prime|', f'{sdf.foot_residual(_foot_gammas):.4e}')
    table2.add_row('minimum coil-to-coil distance', f'{Jccdist.shortest_distance():.3e}')
    table2.add_row('maximum modular coil force, total field [N/m]', f'{max_coil_force():.4e}')
    table2.add_row('minimum aux-to-modular coil distance',
                   ' '.join([f'{a.shortest_distance():.3e}' for a in aux_coil_distances]))
    table2.add_row('minimum aux-to-aux coil distance',
                   ' '.join([f'{a.shortest_aux_distance():.3e}' for a in aux_coil_distances]))

    table2.add_row('fieldline mean-z', ' '.join([f'{Jfl.max_distance():.3e}' for Jfl in meanzs]))
    _, max_fieldline_mean_distance, _ = J_fieldline_mean_distance.longest_distance()
    table2.add_row('fieldline mean distance', f'{max_fieldline_mean_distance:.3e}')

    console.print(table2)

    if dat_dict["iter"] % 10 == 0:
        sdf.to_vtk(OUT_DIR + 'vessel_tmp')
        curves_to_vtk(curves, OUT_DIR + "curves_tmp")
        curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_tmp")
        curves_to_vtk([fl.curve for fl in sing_fls], OUT_DIR + "xpoint_singular_curves_tmp")
        curves_to_vtk([axis.curve for axis in axes], OUT_DIR + "ma_tmp")
        for idx, boozer_surface in enumerate(boozer_surfaces):
            boozer_surface.surface.to_vtk(OUT_DIR + f"surf_tmp_{idx}")
        # standard 5-entry layout, with the singular-polish field lines in the
        # x-point slot (xpoints is sing_fls).
        save([boozer_surfaces, iota_Gs, axes, xpoints, sdf], OUT_DIR + f'design_double_tmp.json')

    dat_dict["iter"] += 1

def _restore_state():
    """Roll boozer surfaces, axis, xpoint and singular-polish curves, and JF
    dofs back to the last successful callback. Called after a failed fun()
    evaluation so the next call warm-starts from a good point."""
    for res, bs in zip(res_list, boozer_surfaces):
        bs.surface.x = res['sdofs']
        bs.res['iota'] = res['iota']
        bs.res['G'] = res['G']
    for res, ax in zip(axes_res_list, axes):
        ax.curve.x = res['adofs']
        ax.res['length'] = res['length']
    for res, xp in zip(xpoints_res_list, xpoints):
        xp.curve.x = res['xdofs']
        xp.res['length'] = res['length']
    for res, fl in zip(sing_res_list, sing_fls):
        fl.curve.x = res['xdofs']
        fl.mu = res['mu']
        fl.res['length'] = res['length']
        fl.need_to_run_code = True
    # set JF.x LAST: it owns the independent mu (free dofs of the sing_fls), so
    # this re-imposes the good design vector after the mu reset above.
    JF.x = dat_dict['x']


# Failed-evaluation barrier: aim the 1-D parabola minimum at ~this fraction of
# the failed step, so the line search retreats to a short feasible step rather
# than all the way to zero (which would stall BFGS).
BARRIER_RETREAT = 0.1


def fun(dofs):
    JF.x = dofs

    fail_reasons = []
    try:
        J = JF.J()
        grad = JF.dJ()
    except Exception as e:
        fail_reasons.append(f'JF.J()/dJ() raised: {e!r}')

    for i, bs in enumerate(boozer_surfaces):
        if not bs.res['success']:
            fail_reasons.append(f'boozer_surface[{i}] Newton solve did not converge')
        # is_self_intersecting() can itself raise; treat a raise as self-intersecting.
        try:
            if bs.surface.is_self_intersecting():
                fail_reasons.append(f'boozer_surface[{i}] surface is self-intersecting')
        except Exception as e:
            fail_reasons.append(f'boozer_surface[{i}] is_self_intersecting() raised: {e!r} (assumed self-intersecting)')
    for i, fl in enumerate(axes):
        if not fl.res['success']:
            fail_reasons.append(f'axis[{i}] Newton solve did not converge')
    for i, fl in enumerate(xpoints):
        if not fl.res['success']:
            fail_reasons.append(f'xpoint[{i}] Newton solve did not converge')
    for i, fl in enumerate(sing_fls):
        if not fl.res.get('success', False):
            fail_reasons.append(f'singular polish[{i}] Newton solve did not converge')
        elif not fl.res.get('square', False):
            fail_reasons.append(f'singular polish[{i}] system is not square (bad mu partition)')

    # Helical vessel: foot-point solve unconverged (residual >> tol) => degenerate tube /
    # unreliable SDF; treat as a failed evaluation so the quadratic barrier makes the line
    # search retreat (mirrors boozer_all.py).
    if isinstance(sdf, HelicalVesselSDF):
        _gammas = np.concatenate(
            [fl.curve.gamma() for fl in xpoints]
            + [bs.surface.gamma().reshape((-1, 3)) for bs in boozer_surfaces])
        _foot_res = sdf.foot_residual(_gammas)
        if _foot_res > FOOT_TOL:
            fail_reasons.append(f'helical SDF foot-point unconverged (max|phi prime|={_foot_res:.2e})')

    if fail_reasons:
        print('failed — quadratic barrier toward last good point: ' + '; '.join(fail_reasons))
        _restore_state()
        # The trial step left the solvable region (Newton solves diverged /
        # surface self-intersects). Return a SMOOTH, CONSISTENT quadratic barrier
        # anchored at the last good point (x0, f0, g0):
        #     f(x) = f0 + g0.(x-x0) + (K/2)||x-x0||^2 ,   g(x) = g0 + K (x-x0)
        # Because g == grad f exactly and, along the BFGS search direction, f is a
        # convex parabola, the Wolfe line search backtracks to a short step just
        # inside the feasible region instead of failing on a flat/inconsistent
        # wall. K places the parabola minimum at ~BARRIER_RETREAT of this failed
        # step (scale-aware, floored to stay strongly convex).
        x0, f0, g0 = dat_dict['x'], dat_dict['J'], dat_dict['dJ']
        dx = dofs - x0
        nrm2 = float(dx @ dx) + 1e-300
        gdx = float(g0 @ dx)
        K = max(abs(gdx) / (BARRIER_RETREAT * nrm2), 1.0)
        return f0 + gdx + 0.5 * K * nrm2, g0 + K * dx

    _last_eval['x'] = np.array(dofs, copy=True)
    _last_eval['J'] = J
    _last_eval['grad'] = np.array(grad, copy=True)
    return J, grad

print("""
################################################################################
### Initial performance #######################################################
################################################################################
""")

J0, dJ0 = fun(JF.x.copy())
print("Norm gradient", np.linalg.norm(dJ0))
print("Norm QS gradient", np.linalg.norm(J_nonQSRatio.dJ()))

# print the currents
for bbsurf in boozer_surfaces:
    for ii in base_curve_idx:
        print(f"Coil {ii} current: {bbsurf.biotsavart.coils[ii].current.full_x} A")

# The weights were loaded from the yaml for the (modular-field) boozer_all run and
# may be too large now that every penalty is evaluated in the TOTAL (modular + aux)
# field. Before optimizing, shrink any weight whose CURRENT contribution
# weight*penalty.J() exceeds 1 -- by repeated factors of 10 -- so each weighted
# term starts below 1. The in-loop x10 escalation can raise it back as the
# constraints demand. Modifying each Weight in place is seen by JF (same objects).
print("Pre-scaling penalty weights so each weighted term starts below 1 in the total field:")
for _name, _scaled in penalties.items():
    _w = penalty_weights.get(_name)
    if _w is None or _w.value == 0.:
        continue
    _prod = _scaled.J()
    if not np.isfinite(_prod) or _prod <= 1.0:
        continue
    _bare = _prod / _w.value          # the unweighted penalty value (dofs fixed here)
    while _w.value * _bare > 1.0:
        _w *= 0.1
    print(f"  {_name}: weight -> {_w.value:.3e}  (penalty={_bare:.3e}, product {_prod:.3e} -> {_w.value * _bare:.3e})")


print("""
################################################################################
### Run the optimization #######################################################
################################################################################
""")
# Number of iterations to perform:
MAXITER = int(args.maxiter)   # --maxiter, default 1000
# Outer BFGS stages (each up to MAXITER iterations, followed by the x10 escalation of the
# weights of the constraints still violated by more than 0.1%). Was 5: in campaign 7463849
# most out-of-spec polishes missed by small margins (0.2-1%) after the 5 stages.
N_POLISH_STAGES = int(args.stages)   # --stages, default 10

dofs = JF.x.copy()
callback(dofs, full=True)

for j in range(N_POLISH_STAGES):
    dat_dict["iter"] = 0

    try:
        res = minimize(fun, dofs, jac=True, method='BFGS', options={'maxiter': MAXITER}, tol=1e-15, callback=callback)
        msg = res.message
    except Exception as e:
        msg = f'caught exception: {e}, restarting from last successful callback.'

    # Resume from the last ACCEPTED state, not res.x. dat_dict['x'] already equals
    # res.x in the good case (callback saved it), but BFGS can return having left
    # the surface/axis/mu warm-start on a DIFFERENT branch (it may stop on a
    # feasible line-search probe, not a _restore_state), so re-solving fun(res.x)
    # from that stale warm-start can land on a different (e.g. self-intersecting)
    # surface. Reset dofs to the accepted point AND _restore_state() so the live
    # warm-start matches it -- making the post-min eval and the next run start
    # consistent and feasible.
    dofs = dat_dict['x'].copy()
    _restore_state()

    print(msg)
    J0, dJ0 = fun(dofs)
    callback(dofs, full=True)

    # MODULAR coil currents (A): bounded by CURRENT_THRESHOLD; weight CURRENT_WEIGHT.
    modular_currents = [np.abs(boozer_surface.biotsavart.coils[idx].current.get_value())
                        for boozer_surface in boozer_surfaces for idx in base_curve_idx]
    curr_err = max([max(c - CURRENT_THRESHOLD, 0.) / CURRENT_THRESHOLD for c in modular_currents])
    # AUX currents (A): BOTH dependent and independent are bounded by AUX_CURRENT_EPS
    # and share ONE weight (AUX_CURRENT_WEIGHT) via J_aux_current. dm.J() is the mu
    # value -> *_CURRENT_SCALE for A; independent currents are read straight from fl.mu.
    dep_aux_amps = [np.abs(float(dm.J())) * _CURRENT_SCALE for dm in dependent_mus]
    indep_aux_amps = [np.abs(float(fl.mu[k])) * _CURRENT_SCALE
                      for fl in sing_fls for k in range(N_DEP_CURRENTS, num_aux)]
    aux_amps = dep_aux_amps + indep_aux_amps
    aux_err = (max([max(c - AUX_CURRENT_EPS, 0.) / AUX_CURRENT_EPS for c in aux_amps])
               if aux_amps else 0.)
    iota_err = max([np.abs(IOTAS.J() - IOT_TARGET)/np.abs(IOT_TARGET) for IOTAS, IOT_TARGET in zip(IOTAS_LIST, IOTAS_TARGET)])
    axis_iota_err = max([np.abs(aio.J() - tgt)/np.abs(tgt) for aio, tgt in zip(AXIS_IOTA_LIST, AXIS_IOTA_TARGET)]) if axis_iota_enabled else 0.0
    # near-axis elongation bound: relative excess of the largest elongation (both configurations)
    elongation_max = max(e.J() for e in ELONGATION_LIST) if elongation_enabled else None
    elongation_err = max(elongation_max - ELONGATION_MAX, 0.)/ELONGATION_MAX if elongation_enabled else 0.0
    # risk-neutral bound: relative excess of the bounded E[C] (one section or the mean over the
    # sections, --ec-section; a direction without a real split makes it undefined and counts
    # as a full violation, so that such a design is never reported in spec)
    ec_value = ec_bounded_value()
    if not EC_ENABLED:
        ec_err = 0.0      # no bound: E[C] is reported only
    else:
        ec_err = max(ec_value - EC_MAX, 0.)/EC_MAX if np.isfinite(ec_value) else 1.0
    mr_err =max(np.abs(mr.J()-MR_TARGET)/MR_TARGET for mr in mrs)
    clen_err = max([max(Jl.J() - LENGTH_THRESHOLD, 0)/np.abs(LENGTH_THRESHOLD) for Jl in Jls])

    cc_err = max(COIL_TO_COIL_THRESHOLD-Jccdist.shortest_distance(), 0)/np.abs(COIL_TO_COIL_THRESHOLD)
    aux_coil_mod_err = max([max(AUX_COIL_DISTANCE_THRESHOLD - a.shortest_distance(), 0)
                            / np.abs(AUX_COIL_DISTANCE_THRESHOLD) for a in aux_coil_distances])
    aux_coil_aux_err = max([max(AUX_COIL_DISTANCE_THRESHOLD - a.shortest_aux_distance(), 0)
                            / np.abs(AUX_COIL_DISTANCE_THRESHOLD) for a in aux_coil_distances])
    # one shared weight drives BOTH the modular<->aux and aux<->aux clearance
    aux_coil_dist_err = max(aux_coil_mod_err, aux_coil_aux_err)
    _, min_xpoint_to_vessel, min_boozer_surface_to_vessel = J_plasma_to_vessel_margin.shortest_distance()
    plasma_vessel_margin_err = max(PLASMA_VESSEL_MARGIN_THRESHOLD-np.min([min_xpoint_to_vessel, min_boozer_surface_to_vessel]), 0)/np.abs(PLASMA_VESSEL_MARGIN_THRESHOLD)

    # Geometric validity of the loaded vessel, fed into the plasma-vessel-margin
    # weight escalation below (the helical geometric penalty rides inside the vessel
    # penalty, so its weight IS that weight). Worst (max) of all constraint errors;
    # relative for nonzero targets, absolute for target-0 (positivity / arclength).
    vessel_shape_err = 0.
    if isinstance(sdf, HelicalVesselSDF):
        #   max_t R*kappa < 1, max_t |dR/ds| < 1, min_t R > 0, arclength variation ~ 0.
        vessel_shape_err = max([_rel_vio(sdf.max_kappa_radius(), 1.0),
                                _rel_vio(sdf.max_dr_ds(), 1.0),
                                max(-sdf.min_radius(), 0.0),
                                sdf.arclength_variation()])
    elif isinstance(sdf, PillPipeSDF):
        # rounded-rect pipe validity 0 < rr < r <= (bx, by).
        bx, by, r, rr = sdf.local_full_x
        vessel_shape_err = max([_rel_vio(r, bx), _rel_vio(r, by), _rel_vio(rr, r),
                                max(-bx, 0.), max(-by, 0.), max(-r, 0.), max(-rr, 0.)])
    elif isinstance(sdf, RennaissanceSDF):
        # renaissance validity 0 < rr < (d1, d2).
        d1, d2, rr = sdf.local_full_x
        vessel_shape_err = max([_rel_vio(rr, d1), _rel_vio(rr, d2),
                                max(-d1, 0.), max(-d2, 0.), max(-rr, 0.)])
    elif isinstance(sdf, TorusSDF):
        # torus validity 0 < r < R.
        r, R = sdf.local_full_x
        vessel_shape_err = max([_rel_vio(r, R), max(-r, 0.), max(-R, 0.)])
    elif isinstance(sdf, HelicalCylinderVesselSDF):
        # welded-pipe validity (rr > 0, turns <= MAX_TURN_DEG, no mitre interference).
        vessel_shape_err = sdf.geometric_violation()

    min_coil_on_vessel_distance, _, _ = J_coil_on_vessel.shortest_distance()
    coil_on_vessel_err = (
        max(COIL_ON_VESSEL_THRESHOLD - min_coil_on_vessel_distance, 0)
        / np.abs(COIL_ON_VESSEL_THRESHOLD)
    )

    min_coil_clearance_distance, _, _ = J_coil_clearance.shortest_distance()
    coil_clearance_err = (
        max(COIL_CLEARANCE_THRESHOLD - min_coil_clearance_distance, 0)
        / np.abs(COIL_CLEARANCE_THRESHOLD)
    )

    msc = [J.J() for J in Jmscs]
    msc_err = max(np.max(msc) - MEAN_SQUARED_CURVATURE_THRESHOLD, 0)/np.abs(MEAN_SQUARED_CURVATURE_THRESHOLD)

    curv_err = max(max([np.max(c.kappa()) for c in base_curves]) - CURVATURE_THRESHOLD, 0)/np.abs(CURVATURE_THRESHOLD)
    alen_err = np.max([ArclengthVariation(c).J() for c in base_curves])

    modB_err = max([np.abs(modB.J()-MODB_TARGET)/MODB_TARGET for modB in modBs])
    # axis |B| ripple: max fractional deviation from the mean, relative to the band
    modB_ripple_err = max([max(r.max_deviation() - MODB_RIPPLE_THRESHOLD, 0)/np.abs(MODB_RIPPLE_THRESHOLD)
                           for r in modB_ripples])

    well_err = 0.
    if WELL_THRESHOLD != 0. and WELL_WEIGHT.value != 0.:
        well_err = max([max(Jl.well().max() - WELL_THRESHOLD, 0)/np.abs(WELL_THRESHOLD) for Jl in magnetic_wells])
    elif WELL_THRESHOLD == 0. and WELL_WEIGHT.value != 0.:  # take absolute error in case of WELL_THRESHOLD == 0
        well_err = max([max(Jl.well().max() - WELL_THRESHOLD, 0) for Jl in magnetic_wells])

    meanz_err = max(max([Jfl.max_distance() for Jfl in meanzs]) - FIELDLINE_MEANZ_THRESHOLD, 0) / np.abs(FIELDLINE_MEANZ_THRESHOLD)
    _, max_fieldline_mean_distance, _ = J_fieldline_mean_distance.longest_distance()
    fieldline_mean_distance_err = (
        max(max_fieldline_mean_distance - FIELDLINE_MEANDIST_THRESHOLD, 0)
        / np.abs(FIELDLINE_MEANDIST_THRESHOLD)
    )

    # maximum modular coil force per unit length (N/m) vs FORCE_THRESHOLD (inequality, so
    # the error is 0 when satisfied)
    max_force = max_coil_force()
    force_err = max(max_force - FORCE_THRESHOLD, 0) / np.abs(FORCE_THRESHOLD)

    # check which constraints are violated and increase weight if violated by more than 0.1%
    if force_err > 0.001 and FORCE_WEIGHT.value != 0.:
        FORCE_WEIGHT *= 10
        print("MAX COIL FORCE ERROR", force_err)
    if curr_err > 0.001 and CURRENT_WEIGHT.value != 0.:
        CURRENT_WEIGHT*=10
        print("MODULAR CURRENT ERROR", curr_err)
    if aux_err > 0.001 and AUX_CURRENT_WEIGHT.value != 0.:
        AUX_CURRENT_WEIGHT *= 10
        print("AUX CURRENT ERROR (dependent + independent)", aux_err)
    if (plasma_vessel_margin_err > 0.001 or vessel_shape_err > 0.001) and PLASMA_VESSEL_MARGIN_WEIGHT.value != 0.:
        PLASMA_VESSEL_MARGIN_WEIGHT*=10
        print("PLASMA-VESSEL MARGIN ERROR", plasma_vessel_margin_err, "VESSEL SHAPE ERROR", vessel_shape_err)
    # on-vessel vs clearance is encoded by which weight is nonzero (the other is
    # 0 in the config), so the weight check alone selects the active constraint.
    if coil_on_vessel_err > 0.001 and COIL_ON_VESSEL_WEIGHT.value != 0.:
        COIL_ON_VESSEL_WEIGHT *= 10
        print("COIL-ON-VESSEL ERROR", coil_on_vessel_err)
    if coil_clearance_err > 0.001 and COIL_CLEARANCE_WEIGHT.value != 0.:
        COIL_CLEARANCE_WEIGHT *= 10
        print("COIL-CLEARANCE ERROR", coil_clearance_err)
    if iota_err > 0.001 and IOTAS_WEIGHT.value != 0.:
        IOTAS_WEIGHT*=10
        print("IOTA ERROR", iota_err)
    if axis_iota_err > 0.001 and AXIS_IOTA_WEIGHT.value != 0.:
        AXIS_IOTA_WEIGHT*=10
        print("ON-AXIS IOTA ERROR", axis_iota_err)
    if elongation_err > 0.001 and ELONGATION_WEIGHT.value != 0.:
        ELONGATION_WEIGHT*=10
        print("ELONGATION ERROR", elongation_err)
    if ec_err > 0.001 and EC_WEIGHT.value != 0.:
        EC_WEIGHT*=10
        print("UNFOLDING EC ERROR", ec_err)
    if mr_err > 0.001 and MAJOR_RADIUS_WEIGHT.value !=0.:
        MAJOR_RADIUS_WEIGHT*=10
        print("MR ERROR", mr_err)
    if clen_err > 0.001 and LENGTH_WEIGHT.value !=0.:
        LENGTH_WEIGHT*=10
        print("COIL LENGTH ERROR", clen_err)
    if cc_err > 0.001 and COIL_TO_COIL_WEIGHT.value != 0.:
        COIL_TO_COIL_WEIGHT*=10
        print("COIL TO COIL ERROR", cc_err)
    if aux_coil_dist_err > 0.001 and AUX_COIL_DISTANCE_WEIGHT.value != 0.:
        AUX_COIL_DISTANCE_WEIGHT*=10
        print("AUX COIL DISTANCE ERROR", aux_coil_dist_err)
    # AUX coil radii must not exceed AUX_RADIUS_MAX (radii are mu[num_aux:2*num_aux])
    aux_radius_err = max([max(abs(float(fl.mu[num_aux + k])) - AUX_RADIUS_MAX, 0.) / np.abs(AUX_RADIUS_MAX)
                          for fl in sing_fls for k in range(num_aux)], default=0.)
    if aux_radius_err > 0.001 and AUX_RADIUS_WEIGHT.value != 0.:
        AUX_RADIUS_WEIGHT*=10
        print("AUX RADIUS ERROR", aux_radius_err)
    if modB_ripple_err > 0.001 and MODB_RIPPLE_WEIGHT.value != 0.:
        MODB_RIPPLE_WEIGHT*=10
        print("MODB AXIS RIPPLE ERROR", modB_ripple_err)
    if modB_err > 0.001 and MODB_WEIGHT.value != 0.:
        MODB_WEIGHT*=10
        print("MODB ERROR", modB_err)
    if msc_err > 0.001 and MEAN_SQUARED_CURVATURE_WEIGHT.value != 0.:
        MEAN_SQUARED_CURVATURE_WEIGHT*=10
        print("MEAN SQUARED ERROR", msc_err)
    if curv_err > 0.001 and CURVATURE_WEIGHT.value !=0.:
        CURVATURE_WEIGHT*=10
        print("CURVATURE ERROR", curv_err)
    if alen_err > 0.001 and ARCLENGTH_WEIGHT.value != 0.:
        ARCLENGTH_WEIGHT*=10
        print("ARCLENGTH ERROR", alen_err)
    if well_err > 0.001 and WELL_WEIGHT.value != 0.:
        WELL_WEIGHT*=10
        print("WELL ERROR", well_err)
    if meanz_err > 0.001 and FIELDLINE_MEANZ_WEIGHT.value != 0.:
        print("MEANZ ERROR", meanz_err)
        FIELDLINE_MEANZ_WEIGHT*=10
    if fieldline_mean_distance_err > 0.001 and FIELDLINE_MEANDIST_WEIGHT.value !=0.:
        print("FIELDLINE MEAN DIST ERROR", fieldline_mean_distance_err)
        FIELDLINE_MEANDIST_WEIGHT*=10
    # snowflake discriminant Delta > DISCRIMINANT_THRESHOLD (identity polish only):
    # relative shortfall below the threshold, max over the singular field lines.
    if discriminant_enabled and DISCRIMINANT_WEIGHT.value != 0.:
        discriminant_err = max([max(DISCRIMINANT_THRESHOLD - d.value(), 0.) / np.abs(DISCRIMINANT_THRESHOLD)
                                for d in discriminants], default=0.)
        if discriminant_err > 0.001:
            DISCRIMINANT_WEIGHT *= 10
            print("SNOWFLAKE DISCRIMINANT ERROR", discriminant_err)
    # config 1 X-point must stay hyperbolic: relative shortfall of tr M below TRACE_MIN
    trace_err = 0.0 if SNOW_ONLY else max(TRACE_MIN - xpoint_trace(), 0.) / TRACE_MIN
    if trace_err > 0.001 and MONODROMY_WEIGHT.value != 0.:
        MONODROMY_WEIGHT *= 10
        print("XPOINT TRACE ERROR", trace_err)

    # Refresh the failed-evaluation barrier anchor with the JUST-ESCALATED weights.
    # callback() above captured dat_dict['J']/['dJ'] BEFORE these weight bumps, so on
    # the NEXT BFGS run's first line search the barrier (anchored at dat_dict) would
    # be built from STALE, pre-escalation weights -- inconsistent with the new-weight
    # phi(0)/phi'(0) the line search measures at this same point, which makes the
    # barrier the "inconsistent wall" it was designed to avoid (every probe fails ->
    # line search gives up with no accepted step). Re-evaluating fun() here at the
    # unchanged dofs rebuilds (J, dJ) under the new weights so the barrier is again a
    # faithful local model. fun() reuses the warm-started (feasible) solves, so this
    # is one cheap extra evaluation per outer iteration.
    J0, dJ0 = fun(dofs)
    dat_dict['x'] = dofs.copy()
    dat_dict['J'] = J0
    dat_dict['dJ'] = dJ0.copy()

    sdf.to_vtk(OUT_DIR + f'vessel_opt_{j}', nx=40, ny=40, nz=40)
    curves_to_vtk(curves, OUT_DIR + f"curves_opt_{j}")
    curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_opt_{j}")
    curves_to_vtk([fl.curve for fl in sing_fls], OUT_DIR + f"xpoint_singular_curves_opt_{j}")
    curves_to_vtk([axis.curve for axis in axes], OUT_DIR + f"ma_opt_{j}")
    for idx, boozer_surface in enumerate(boozer_surfaces):
        boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_{j}")
    # standard 5-entry layout, with the singular-polish field lines in the
    # x-point slot (xpoints is sing_fls).
    save([boozer_surfaces, iota_Gs, axes, xpoints, sdf], OUT_DIR + f'design_double_{j}.json')

    # save the weights in a yaml file
    config['CURRENT_WEIGHT'] = CURRENT_WEIGHT.value
    config['AUX_CURRENT_WEIGHT'] = AUX_CURRENT_WEIGHT.value
    config['COIL_TO_COIL_WEIGHT'] = COIL_TO_COIL_WEIGHT.value
    config['AUX_COIL_DISTANCE_WEIGHT'] = AUX_COIL_DISTANCE_WEIGHT.value
    config['AUX_COIL_DISTANCE_THRESHOLD'] = AUX_COIL_DISTANCE_THRESHOLD
    config['AUX_RADIUS_WEIGHT'] = AUX_RADIUS_WEIGHT.value
    config['AUX_RADIUS_MAX'] = AUX_RADIUS_MAX
    config['CURVATURE_WEIGHT'] = CURVATURE_WEIGHT.value
    config['MEAN_SQUARED_CURVATURE_WEIGHT'] = MEAN_SQUARED_CURVATURE_WEIGHT.value
    config['LENGTH_WEIGHT'] = LENGTH_WEIGHT.value
    config['IOTAS_WEIGHT'] = IOTAS_WEIGHT.value
    # On-axis iota weight/target only persist when the constraint is active (--AR 1/2).
    if axis_iota_enabled:
        config['AXIS_IOTA_WEIGHT'] = AXIS_IOTA_WEIGHT.value
        config['AXIS_IOTA_TARGET'] = list(AXIS_IOTA_TARGET)[:1]
    # Elongation bound (None: no bound) and, when there is one, its escalated weight.
    config['ELONGATION_MAX'] = ELONGATION_MAX
    if elongation_enabled:
        config['ELONGATION_WEIGHT'] = ELONGATION_WEIGHT.value
    # escalated weight of the bound on E[C] (the bound itself, EC_MAX, is recorded at start-up)
    if EC_ENABLED:
        config['EC_WEIGHT'] = EC_WEIGHT.value
    config['MONODROMY_WEIGHT'] = MONODROMY_WEIGHT.value
    config['MAJOR_RADIUS_WEIGHT'] = MAJOR_RADIUS_WEIGHT.value
    config['BOOZER_RESIDUAL_WEIGHT'] = BOOZER_RESIDUAL_WEIGHT.value
    config['PLASMA_VESSEL_MARGIN_WEIGHT'] = PLASMA_VESSEL_MARGIN_WEIGHT.value
    config['MODB_WEIGHT'] = MODB_WEIGHT.value
    config['MODB_RIPPLE_WEIGHT'] = MODB_RIPPLE_WEIGHT.value
    config['MODB_RIPPLE_THRESHOLD'] = MODB_RIPPLE_THRESHOLD
    config['ARCLENGTH_WEIGHT'] = ARCLENGTH_WEIGHT.value
    config['COIL_ON_VESSEL_WEIGHT'] = COIL_ON_VESSEL_WEIGHT.value
    config['COIL_CLEARANCE_WEIGHT'] = COIL_CLEARANCE_WEIGHT.value
    config['WELL_WEIGHT'] = WELL_WEIGHT.value
    config['FIELDLINE_MEANZ_WEIGHT'] = FIELDLINE_MEANZ_WEIGHT.value
    config['FIELDLINE_MEANDIST_WEIGHT'] = FIELDLINE_MEANDIST_WEIGHT.value
    config['FORCE_WEIGHT'] = FORCE_WEIGHT.value
    config['FORCE_THRESHOLD'] = FORCE_THRESHOLD
    if discriminant_enabled:
        config['DISCRIMINANT_WEIGHT'] = DISCRIMINANT_WEIGHT.value

    # Save to YAML
    with open(OUT_DIR + f'design_double_{j}.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

sdf.to_vtk(OUT_DIR + f'vessel_opt_final', nx=40, ny=40, nz=40)
curves_to_vtk(curves, OUT_DIR + f"curves_opt_final")
curves_to_vtk([xpoint.curve for xpoint in xpoints], OUT_DIR + f"xpoint_curves_opt_final")
curves_to_vtk([fl.curve for fl in sing_fls], OUT_DIR + f"xpoint_singular_curves_opt_final")
curves_to_vtk([axis.curve for axis in axes], OUT_DIR + f"ma_opt_final")
for idx, boozer_surface in enumerate(boozer_surfaces):
    boozer_surface.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_final")
with open(OUT_DIR + f'design_double_final_{DEVICE_TAG}.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
# design_double_final_<DEVICE_ID>.json is written by the FINALIZE step at the end of the
# script (boozer surfaces and axes re-solved on the COMBINED modular+aux coil
# set), so downstream tracing (mk_manifolds.py) sees the polished field.

# ---- final summary of constraints, values, and relative errors ----
final_nonqs_pct = 100. * J_nonQSRatio.J()**0.5

# mirror ratio max|B|/min|B| on the magnetic (Boozer) surface, in the TOTAL
# (modular + aux) field, via a fresh SingularBiotSavart so no point cache is disturbed.
mirror_ratios = []
for _bsurf, _mk in zip(boozer_surfaces, FIELDS):   # each configuration in its own field
    _bs_mirror = _mk()
    _bs_mirror.set_points(_bsurf.surface.gamma().reshape((-1, 3)))
    _modB_surf = _bs_mirror.AbsB()
    mirror_ratios.append(float(np.max(_modB_surf) / np.min(_modB_surf)))
mirror_ratio = max(mirror_ratios)
# the configuration whose major radius is furthest from the target
_mr_worst = max((mr.J() for mr in mrs), key=lambda v: abs(v - MR_TARGET))

# Recompute all error metrics one more time with final dofs
final_metrics = {
    'nonQS_percent':              (final_nonqs_pct,                                None,                          None),
    'mirror_ratio':               (mirror_ratio,                                   None,                          None),
    'aspect_ratio':               (boozer_surfaces[0].surface.aspect_ratio(),      None,                          None),
    'current':                    (max(modular_currents),                          CURRENT_THRESHOLD,             curr_err if CURRENT_WEIGHT.value != 0. else 0.0),
    'aux_current_A':              (max(aux_amps) if aux_amps else 0.0,             AUX_CURRENT_EPS,               aux_err if AUX_CURRENT_WEIGHT.value != 0. else 0.0),
    'iotas':                      (max(IOTAS.J() for IOTAS in IOTAS_LIST),         max(IOTAS_TARGET),             iota_err if IOTAS_WEIGHT.value != 0. else 0.0),
    'on_axis_iota':               (max(aio.J() for aio in AXIS_IOTA_LIST),         (max(AXIS_IOTA_TARGET) if axis_iota_enabled else None),   (axis_iota_err if (axis_iota_enabled and AXIS_IOTA_WEIGHT.value != 0.) else None)),
    'major_radius':               (_mr_worst,                                         MR_TARGET,                     mr_err if MAJOR_RADIUS_WEIGHT.value != 0. else 0.0),
    'coil_length':                (max(Jl.J() for Jl in Jls),                      LENGTH_THRESHOLD,              clen_err if LENGTH_WEIGHT.value != 0. else 0.0),
    'coil_to_coil':               (Jccdist.shortest_distance(),                    COIL_TO_COIL_THRESHOLD,        cc_err if COIL_TO_COIL_WEIGHT.value != 0. else 0.0),
    'max_force':                  (max_force,                                      FORCE_THRESHOLD,               force_err if FORCE_WEIGHT.value != 0. else 0.0),
    'aux_coil_distance':          (min(a.shortest_distance() for a in aux_coil_distances), AUX_COIL_DISTANCE_THRESHOLD, aux_coil_mod_err if AUX_COIL_DISTANCE_WEIGHT.value != 0. else 0.0),
    'aux_coil_to_aux_distance':   (min(a.shortest_aux_distance() for a in aux_coil_distances), AUX_COIL_DISTANCE_THRESHOLD, aux_coil_aux_err if AUX_COIL_DISTANCE_WEIGHT.value != 0. else 0.0),
    'aux_radius':                 (max((abs(float(fl.mu[num_aux + k])) for fl in sing_fls for k in range(num_aux)), default=0.0), AUX_RADIUS_MAX, aux_radius_err if AUX_RADIUS_WEIGHT.value != 0. else 0.0),
    'plasma_vessel_margin':       (min(min_xpoint_to_vessel, min_boozer_surface_to_vessel),
                                                                                   PLASMA_VESSEL_MARGIN_THRESHOLD, plasma_vessel_margin_err if PLASMA_VESSEL_MARGIN_WEIGHT.value != 0. else 0.0),
    'coil_on_vessel':             (min_coil_on_vessel_distance,                    COIL_ON_VESSEL_THRESHOLD,      coil_on_vessel_err if COIL_ON_VESSEL_WEIGHT.value != 0. else 0.0),
    'coil_clearance':             (min_coil_clearance_distance,                    COIL_CLEARANCE_THRESHOLD,      coil_clearance_err if COIL_CLEARANCE_WEIGHT.value != 0. else 0.0),
    'msc':                        (max(msc),                                       MEAN_SQUARED_CURVATURE_THRESHOLD, msc_err if MEAN_SQUARED_CURVATURE_WEIGHT.value != 0. else 0.0),
    'curvature':                  (max(np.max(c.kappa()) for c in base_curves),    CURVATURE_THRESHOLD,           curv_err if CURVATURE_WEIGHT.value != 0. else 0.0),
    'arclength':                  (alen_err,                                       0.0,                           alen_err if ARCLENGTH_WEIGHT.value != 0. else 0.0),
    'modB':                       (max(modB.J() for modB in modBs),                MODB_TARGET,                   modB_err if MODB_WEIGHT.value != 0. else 0.0),
    'modB_axis_ripple':           (max(r.max_deviation() for r in modB_ripples),   MODB_RIPPLE_THRESHOLD,         modB_ripple_err if MODB_RIPPLE_WEIGHT.value != 0. else 0.0),
    'well':                       (max(w.well().max() for w in magnetic_wells),    WELL_THRESHOLD,                well_err if WELL_WEIGHT.value != 0. else 0.0),
    'fieldline_meanz':            (max(Jfl.max_distance() for Jfl in meanzs),      FIELDLINE_MEANZ_THRESHOLD,     meanz_err if FIELDLINE_MEANZ_WEIGHT.value != 0. else 0.0),
    'fieldline_meandist':         (max_fieldline_mean_distance,                    FIELDLINE_MEANDIST_THRESHOLD,  fieldline_mean_distance_err if FIELDLINE_MEANDIST_WEIGHT.value != 0. else 0.0),
}

# Two-configuration entries. The entries above are the worst case over both configurations
# (or config 1's value for the descriptive aspect_ratio). Added here: the trace bound of
# config 1's X-point, which is gated, and descriptive per-configuration values
# (config 1 = X-point, aux off; config 2 = snowflake).
if not SNOW_ONLY:
    final_metrics['xpoint_trace'] = (xpoint_trace(), TRACE_MIN, trace_err if MONODROMY_WEIGHT.value != 0. else 0.0)
# Near-axis elongation bound (only when the yaml sets ELONGATION_MAX): the largest elongation
# over the N_ELONGATION angles of both configurations; its relative excess feeds the 0.1% gate.
if elongation_enabled:
    final_metrics['elongation_bound'] = (elongation_max, ELONGATION_MAX, elongation_err if ELONGATION_WEIGHT.value != 0. else 0.0)
# Risk-neutral bound on the unfolding constant of the snowflake: the bounded E[C] (exact
# expectation over the Gaussian-process perturbations; one section or the mean over the sections,
# --ec-section) against EC_MAX; its relative excess feeds the 0.1% gate.
if EC_ENABLED:
    final_metrics['unfolding_EC'] = (ec_value, EC_MAX, ec_err if EC_WEIGHT.value != 0. else 0.0)
else:
    final_metrics['unfolding_EC'] = (ec_value, None, None)   # no bound: reported only
# what the bound applied to: 1 = the mean over the sections, 0 = the section at unfolding_EC_section_t
final_metrics['unfolding_EC_bound_on_mean'] = (1.0 if EC_USE_MEAN else 0.0, None, None)
final_metrics['unfolding_EC_section_t'] = (EC_T, None, None)
# E[C] in that section, and the exact standard deviation of C there (its own name:
# mk_unfolding.py writes unfolding_C_std, over samples)
try:
    final_metrics['unfolding_EC_section'] = (float(EC.J()), None, None)
except Exception:
    final_metrics['unfolding_EC_section'] = (float('nan'), None, None)
final_metrics['unfolding_EC_std'] = (EC.std(), None, None)
# E[C] over the toroidal sections (PhiMeanUnfoldingC): mean, smallest and largest section
_ecp_mean, _ecp_min, _ecp_max = ec_phi_stats()
final_metrics['unfolding_EC_phi_mean'] = (_ecp_mean, None, None)
final_metrics['unfolding_EC_phi_min'] = (_ecp_min, None, None)
final_metrics['unfolding_EC_phi_max'] = (_ecp_max, None, None)
for _k, (_bsurf, _nq, _io, _mrat) in enumerate(zip(boozer_surfaces, nonQS_list, IOTAS_LIST, mirror_ratios), start=2 if SNOW_ONLY else 1):
    final_metrics[f'nonQS_percent_config{_k}'] = (100. * _nq.J()**0.5, None, None)
    final_metrics[f'iota_config{_k}'] = (_io.J(), None, None)
    final_metrics[f'aspect_ratio_config{_k}'] = (_bsurf.surface.aspect_ratio(), None, None)
    final_metrics[f'mirror_ratio_config{_k}'] = (_mrat, None, None)

# Helical vessel geometry: report the regime metrics (both must stay < 1) and the
# centerline arclength variation in summary.txt. The two regime constraints carry a
# relative violation (so a geometrically invalid tube fails the 0.1% gate); the
# arclength variation is descriptive (escalated during the optimization, not gated).
if isinstance(sdf, HelicalVesselSDF):
    _kr = sdf.max_kappa_radius()
    _drds = sdf.max_dr_ds()
    final_metrics['vessel_max_kappa_radius'] = (_kr, 1.0, max(_kr - 1.0, 0.0))
    final_metrics['vessel_max_dr_ds'] = (_drds, 1.0, max(_drds - 1.0, 0.0))
    # radius positivity R(t) > 0 (paper section 5): absolute violation (target 0)
    _rmin = sdf.min_radius()
    final_metrics['vessel_min_radius'] = (_rmin, 0.0, max(-_rmin, 0.0))
    final_metrics['vessel_arclength_variation'] = (sdf.arclength_variation(), None, None)

# Welded-pipe vessel geometry: the mitre validity feeds the 0.1% gate (target 0);
# the largest turn angle is descriptive.
if isinstance(sdf, HelicalCylinderVesselSDF):
    _viol = sdf.geometric_violation()
    final_metrics['vessel_mitre_violation'] = (_viol, 0.0, _viol)
    final_metrics['vessel_max_turn_angle_deg'] = (sdf.max_turn_angle(), None, None)

# Pill-pipe / renaissance / torus geometric validity (paper section 5: eq. (20) for the
# pill pipe, r < R for the torus): the worst violation of the shape inequalities,
# relative between dimensions and absolute for positivity, i.e. the same
# vessel_shape_err that drives the plasma-vessel-margin escalation. Gated, like the
# helical-vessel regime metrics, so a geometrically invalid vessel fails the 0.1% check.
if isinstance(sdf, (PillPipeSDF, RennaissanceSDF, TorusSDF)):
    final_metrics['vessel_shape_violation'] = (vessel_shape_err, 0.0, vessel_shape_err)

# Snowflake discriminant Delta (worst = smallest over the singular field lines). Always
# reported; gated on its relative shortfall below DISCRIMINANT_THRESHOLD only when the
# constraint is enabled (--discriminant 1, identity polish).
_disc_min = min(d.value() for d in discriminants)
if discriminant_enabled:
    final_metrics['snowflake_discriminant_Delta'] = (
        _disc_min, DISCRIMINANT_THRESHOLD,
        max(DISCRIMINANT_THRESHOLD - _disc_min, 0.) / abs(DISCRIMINANT_THRESHOLD)
        if DISCRIMINANT_WEIGHT.value != 0. else 0.0)
else:
    final_metrics['snowflake_discriminant_Delta'] = (_disc_min, None, None)

# record the solved auxiliary-coil parameters (currents in Amperes) per fl.
for fi, fl in enumerate(sing_fls):
    for k, nm in enumerate(MU_NAMES):
        v = float(fl.mu[k])
        if k < num_aux:
            v *= _CURRENT_SCALE
        final_metrics[f'aux_{nm}_idx{fi}'] = (v, None, None)

# record the full 2x2 monodromy matrix of the SINGULAR POLISH per X-point so
# the device browser can display it; entries are descriptive (no threshold/error).
# Greene's residue R = (2 - tr(M))/4 is recorded for every X-point in all cases.
for ti, fl in enumerate(sing_fls):
    Mm = np.asarray(fl.res['monodromy_matrix'])
    for a in range(2):
        for b in range(2):
            final_metrics[f'monodromy_M{a}{b}_idx{ti}'] = (float(Mm[a, b]), None, None)
    final_metrics[f'greene_residue_idx{ti}'] = (
        (2.0 - float(np.trace(Mm))) / 4.0, None, None)

# write a human-readable summary
with open(OUT_DIR + 'summary.txt', 'w') as f:
    f.write(f"# {'metric':<30s} {'value':>16s} {'threshold':>16s} {'rel_error':>16s}\n")
    for name, (value, threshold, rel_err) in final_metrics.items():
        thr_str = f"{threshold:.6e}" if threshold is not None else "n/a"
        err_str = f"{rel_err:.6e}"   if rel_err   is not None else "n/a"
        f.write(f"  {name:<30s} {value:.6e}   {thr_str:>16s}   {err_str:>16s}\n")

# write a machine-readable max relative error for the prefix.sh check
max_rel_err = max(abs(v[2]) for v in final_metrics.values() if v[2] is not None)
with open(OUT_DIR + 'max_rel_error.txt', 'w') as f:
    f.write(f"{max_rel_err:.18e}\n")

print("""
################################################################################
### Finalize: rebuild aux coils + re-solve on the COMBINED coil set ###########
################################################################################
""")
# Mirror boozer_singular.py: build simsopt coils from each fl's solved mu and
# re-solve the Boozer surface and magnetic axis with modular+aux coils, so the
# saved design is self-consistent for downstream tracing -- mk_manifolds.py
# uses boozer_surface.biotsavart, which must contain the aux field.
out_boozer_surfaces = []
out_iota_Gs = []
out_axes = []
# Only config 2 (the snowflake, entries 1.. of boozer_surfaces / axes) has aux coils to
# rebuild; config 1 (entry 0) already lives on plain modular coils and is saved as it is.
_c2 = 0 if SNOW_ONLY else 1     # position of the snowflake configuration in the lists
for idx, (fl, boozer_surface, ax) in enumerate(zip(sing_fls, boozer_surfaces[_c2:], axes[_c2:]), start=1):
    mu_sol = fl.mu
    Naux = (len(mu_sol) - 1) // 2
    Zc = float(mu_sol[-1])
    aux_base_curves = []
    aux_base_currents = []
    for k in range(Naux):
        Ik = float(mu_sol[k])
        rk = float(mu_sol[Naux + k])
        c = CurveXYZFourier(np.linspace(0, 1, 160, endpoint=False), 1)
        c.x = c.x * 0.
        c.set('zc(0)', Zc)
        c.set('xc(1)', rk)
        c.set('ys(1)', rk)
        aux_base_curves.append(c)
        aux_base_currents.append(ScaledCurrent(Current(Ik), _CURRENT_SCALE))
    if fl.stellsym_aux:
        aux_coils = coils_via_symmetries(aux_base_curves, aux_base_currents, 1, True)
    else:
        aux_coils = [Coil(c, I) for c, I in zip(aux_base_curves, aux_base_currents)]
    combined_coils = boozer_surface.biotsavart.coils + aux_coils

    # Save the polished design on an EXPLICIT combined modular+aux BiotSavart
    # (real CurveXYZFourier aux coils), NOT SingularBiotSavart: the total field is
    # identical (~1e-17) but pure-C++, so downstream tracing (mk_manifolds.py's
    # compute_fieldlines) runs at native speed instead of calling back into
    # Python/jax per integration step. The optimization above still uses
    # SingularBiotSavart; only the saved/finalized field is the combined coil set.
    field = BiotSavart(combined_coils)
    bs_out = BoozerSurface(field, boozer_surface.surface,
                           Volume(boozer_surface.surface), boozer_surface.surface.volume())
    bs_res = bs_out.run_code(boozer_surface.res['iota'], boozer_surface.res['G'])
    if not bs_res['success'] or bs_out.surface.is_self_intersecting():
        print(f"ERROR: idx={idx}: BoozerSurface re-solve on the combined coil set failed")
        print(f"ABORT: design_double_final_{DEVICE_TAG}.json will not be written.")
        raise SystemExit(1)

    new_ax = PeriodicFieldLine(field, ax.curve)
    ax_res = new_ax.run_code(CurveLength(ax.curve).J())
    if not ax_res['success']:
        print(f"ERROR: idx={idx}: magnetic-axis re-solve on the combined coil set failed")
        print(f"ABORT: design_double_final_{DEVICE_TAG}.json will not be written.")
        raise SystemExit(1)

    out_boozer_surfaces.append(bs_out)
    out_iota_Gs.append([bs_res['iota'], bs_res['G']])
    out_axes.append(new_ax)
    curves_to_vtk([coil.curve for coil in aux_coils], OUT_DIR + f'aux_coils_{idx}')
    bs_out.surface.to_vtk(OUT_DIR + f"surf_opt_{idx}_final")   # refresh with the re-solved surface

# standard 5-entry layout consumed by mk_manifolds.py: the boozer surfaces and
# axes carry the COMBINED (modular + aux) coils, and the singular-polish field
# lines sit in the x-point slot.
# Both configurations: config 1 (X-point, its own modular currents) then config 2 (snowflake,
# combined modular + aux coils); the fixed points are config 1's X-point and config 2's
# singular field line.
# (--target snowflake: the snowflake configuration alone, in the same layout)
if not SNOW_ONLY:
    out_boozer_surfaces = [boozer_surfaces[0]] + out_boozer_surfaces
    out_iota_Gs = [[boozer_surfaces[0].res['iota'], boozer_surfaces[0].res['G']]] + out_iota_Gs
    out_axes = [axes[0]] + out_axes
save([out_boozer_surfaces, out_iota_Gs, out_axes, xpoints, sdf], OUT_DIR + f'design_double_final_{DEVICE_TAG}.json')
print(f"wrote {OUT_DIR}design_double_final_{DEVICE_TAG}.json (combined modular+aux coil set)")

# One single-configuration copy per configuration, in the standard 5-entry layout, each in
# its own sub-directory, so that the render scripts (run_render.sh: mk_manifolds.py,
# plot_manifolds.py), which take ONE configuration and write next to their input, can be
# run on each: config1/ = X-point (aux off), config2/ = snowflake (combined coils).
for _k, _name in enumerate(('config2',) if SNOW_ONLY else ('config1', 'config2')):
    os.makedirs(OUT_DIR + _name, exist_ok=True)
    save([[out_boozer_surfaces[_k]], [out_iota_Gs[_k]], [out_axes[_k]], [xpoints[_k]], sdf],
         OUT_DIR + f'{_name}/design_double_final_{DEVICE_TAG}.json')
    print(f"wrote {OUT_DIR}{_name}/design_double_final_{DEVICE_TAG}.json")
