"""Lp force penalty on the MODULAR coils in the TOTAL field (modular + auxiliary coils).

TotalFieldLpCurveForce is the analogue of simsopt's LpCurveForce for the singular
polish (boozer_singular_opt.py). The Lorentz force per unit length on each target
(base) modular coil i is

    dF/dl = I_i t_i x (B_self,reg + B_modular,mutual + B_aux),

where
  * B_self,reg is the regularized self field of coil i (Landreman and Hurwitz method,
    simsopt.field.selffield.B_regularized_pure, exactly as in simsopt's RegularizedCoil);
  * B_modular,mutual is the Biot-Savart field of all the OTHER modular coils;
  * B_aux is the field of the planar circular auxiliary coils of a
    SingularPeriodicFieldline (utils/singularperiodicfieldline._B_aux), whose
    parameters mu = (I_1..I_N, r_1..r_N, Z) are the local dofs of that field line
    (with their stellarator-symmetric images when fl.stellsym_aux is True).

The objective is the same as simsopt's LpCurveForce:

    J = (1/p) sum_i (1/n) sum_k max(|dF/dl|_ik / 1e6 - F0, 0)^p |gamma'_ik|

with |dF/dl| converted to MN/m and the threshold F0 given in MN/m.

The gradient is taken with respect to
  * the modular-coil dofs (geometry and currents, of the targets and of the sources),
  * the FREE (independent) mu dofs of the field line (aux radii, Z, independent currents),
  * the DEPENDENT mu (aux currents solved by the Newton polish to enforce the
    monodromy), through the adjoint of the singular polish:
    dJ/d(design) += sum_q dJ/dmu_q * dmu_q/d(design), with dmu_q/d(design) from
    SingularPeriodicFieldline.dmu_dep_by_ddesign().
"""
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, grad

from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.geo import CurveLength
from simsopt.field.selffield import B_regularized_pure

from .singularperiodicfieldline import _B_aux, _mu_names

__all__ = ['TotalFieldLpCurveForce']

# mu_0 / (4 pi) in SI units (T m / A)
_MU0_4PI = 1.0e-7


def _biot_savart_pure(pts, gammas, gammadashs, currents):
    """Biot-Savart field (T) at pts (n, 3) of m closed filaments sampled at q uniform
    quadrature points: gammas, gammadashs (m, q, 3), with gammadash = d gamma / d t for
    t in [0, 1); currents (m,) in A."""
    dr = pts[:, None, None, :] - gammas[None, :, :, :]              # (n, m, q, 3)
    num = jnp.cross(gammadashs[None, :, :, :], dr)                   # (n, m, q, 3)
    inv_r3 = jnp.sum(dr * dr, axis=-1) ** -1.5                       # (n, m, q)
    return _MU0_4PI * jnp.einsum('m,nmqk,nmq->nk', currents, num, inv_r3) / gammas.shape[1]


def _force_density_pure(gammas_t, gammadashs_t, gammadashdashs_t, currents_t, quadpoints,
                        regularizations, gammas_m, gammadashs_m, currents_m, src_idx,
                        mu, stellsym):
    """|dF/dl| (N/m) at the quadrature points of each target coil, shape (m_t, n).

    src_idx[i] (static tuple of ints) lists the modular coils acting as mutual sources
    on target i: all of them EXCEPT the target's own coil, whose contribution is the
    regularized self field instead. (Removing it, rather than zeroing its current,
    avoids 0 * inf = nan from its own singular Biot-Savart kernel.)"""
    out = []
    for i in range(gammas_t.shape[0]):      # static loop over the (few) target coils
        g = gammas_t[i]
        gd = gammadashs_t[i]
        tangent = gd / jnp.linalg.norm(gd, axis=-1)[:, None]
        idx = jnp.array(src_idx[i])
        B = (B_regularized_pure(g, gd, gammadashdashs_t[i], quadpoints, currents_t[i], regularizations[i])
             + _biot_savart_pure(g, gammas_m[idx], gammadashs_m[idx], currents_m[idx])
             + _B_aux(g, mu, stellsym=stellsym))
        out.append(jnp.linalg.norm(jnp.cross(currents_t[i] * tangent, B), axis=-1))
    return jnp.stack(out)


class TotalFieldLpCurveForce(Optimizable):
    r"""Lp penalty on the Lorentz force per unit length of the target MODULAR coils in
    the TOTAL (modular + auxiliary) field of a SingularPeriodicFieldline.

    Args:
        target_coils (list of Coil): coils on which the force is penalized (typically
            the base modular coils). They must be among ``modular_coils`` (same objects).
        modular_coils (list of Coil): all the modular coils (sources of the mutual field).
        fieldline (SingularPeriodicFieldline): converged singular polish whose aux coils
            (mu) add their field; its dependent mu enter the gradient through the adjoint.
        regularization (float): self-field regularization of the coil cross section,
            e.g. simsopt.field.selffield.regularization_circ(a).
        p (float): power of the penalty.
        threshold (float): force per unit length threshold F0 in MN/m (as LpCurveForce).
    """

    def __init__(self, target_coils, modular_coils, fieldline, regularization, p=2.0, threshold=0.0):
        self.target_coils = list(target_coils)
        self.modular_coils = list(modular_coils)
        self.fl = fieldline
        self.p = float(p)
        self.threshold = float(threshold)
        tidx = []
        for t in self.target_coils:
            matches = [j for j, c in enumerate(self.modular_coils) if c is t]
            if len(matches) != 1:
                raise ValueError("each target coil must appear exactly once in modular_coils")
            tidx.append(matches[0])
        nm = len(self.modular_coils)
        src_idx = tuple(tuple(j for j in range(nm) if j != k) for k in tidx)
        regs = jnp.array([float(regularization)] * len(self.target_coils))
        quadpoints = jnp.asarray(self.target_coils[0].curve.quadpoints)
        stellsym = bool(fieldline.stellsym_aux)
        p_, thr = self.p, self.threshold

        def fdens(gt, gdt, gddt, It, gm, gdm, Im, mu):
            return _force_density_pure(gt, gdt, gddt, It, quadpoints, regs, gm, gdm, Im,
                                       src_idx, mu, stellsym)

        def J(gt, gdt, gddt, It, gm, gdm, Im, mu):
            f = fdens(gt, gdt, gddt, It, gm, gdm, Im, mu) / 1e6      # MN/m
            dl = jnp.linalg.norm(gdt, axis=-1)                         # |gamma'|
            return jnp.sum(jnp.maximum(f - thr, 0.0) ** p_ * dl) / gt.shape[1] / p_

        self._fdens = jit(fdens)
        self._J = jit(J)
        self._dJ = jit(grad(J, argnums=(0, 1, 2, 3, 4, 5, 6, 7)))
        Optimizable.__init__(self, depends_on=self.modular_coils + [fieldline])

    def _ensure_solved(self):
        """Re-run the singular polish if anything upstream changed (as DependentMu does),
        so the dependent aux currents correspond to the current design. A field line
        reloaded from a json carries only a partial res (no 'mask', needed by the
        adjoint) with need_to_run_code False, so it is re-solved too."""
        fl = self.fl
        if getattr(fl, 'res', None) is None or 'mask' not in fl.res:
            fl.need_to_run_code = True
        if fl.need_to_run_code:
            length = None
            if getattr(fl, 'res', None) is not None:
                length = fl.res.get('length')
            if length is None:
                length = CurveLength(fl.curve).J()
            fl.run_code(length)

    def _args(self):
        self._ensure_solved()
        T, M = self.target_coils, self.modular_coils
        return (jnp.array([c.curve.gamma() for c in T]),
                jnp.array([c.curve.gammadash() for c in T]),
                jnp.array([c.curve.gammadashdash() for c in T]),
                jnp.array([c.current.get_value() for c in T]),
                jnp.array([c.curve.gamma() for c in M]),
                jnp.array([c.curve.gammadash() for c in M]),
                jnp.array([c.current.get_value() for c in M]),
                jnp.asarray(self.fl.mu))

    def force_density(self):
        """|dF/dl| (N/m) at the quadrature points of each target coil, shape (m_t, n)."""
        return np.asarray(self._fdens(*self._args()))

    def max_force(self):
        """Largest Lorentz force per unit length (N/m) over the target coils."""
        return float(np.max(self.force_density()))

    def J(self):
        return float(self._J(*self._args()))

    @derivative_dec
    def dJ(self):
        args = self._args()
        dgt, dgdt, dgddt, dIt, dgm, dgdm, dIm, dmu = self._dJ(*args)
        T, M = self.target_coils, self.modular_coils
        d = sum([c.curve.dgamma_by_dcoeff_vjp(dgt[i])
                 + c.curve.dgammadash_by_dcoeff_vjp(dgdt[i])
                 + c.curve.dgammadashdash_by_dcoeff_vjp(dgddt[i])
                 + c.current.vjp(jnp.asarray([dIt[i]])) for i, c in enumerate(T)])
        d = d + sum([c.curve.dgamma_by_dcoeff_vjp(dgm[j])
                     + c.curve.dgammadash_by_dcoeff_vjp(dgdm[j])
                     + c.current.vjp(jnp.asarray([dIm[j]])) for j, c in enumerate(M)])
        # mu part: FREE mu directly; DEPENDENT mu through the adjoint of the polish.
        dmu = np.asarray(dmu)
        free = np.asarray(self.fl.local_dofs_free_status, dtype=bool)
        d = d + Derivative({self.fl: np.where(free, dmu, 0.0)})
        dep = np.where(~free)[0]
        if dep.size and np.any(dmu[dep] != 0.0):
            names = _mu_names(len(dmu))
            dmu_dep = self.fl.dmu_dep_by_ddesign()
            for k in dep:
                if dmu[k] != 0.0:
                    d = d + float(dmu[k]) * dmu_dep[names[k]]
        return d

    return_fn_map = {'J': J, 'dJ': dJ}
