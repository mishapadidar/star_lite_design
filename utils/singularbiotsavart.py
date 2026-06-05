"""
singularperiodicfieldline.py
============================

``SingularBiotSavart`` -- a :class:`simsopt.field.MagneticField` that presents
the TOTAL field

    B_total(x) = B_modular(x; coil dofs) + B_aux(x; mu)

produced by the modular coils PLUS the planar circular auxiliary coils of a
:class:`~star_lite_design.utils.singularperiodicfieldline.SingularPeriodicFieldline`
(``fl``).  Because it is a ``MagneticField`` it can be handed straight to
``BoozerSurface`` and ``PeriodicFieldLine`` so those classes solve in the total
field instead of the modular field alone::

    fl = SingularPeriodicFieldline(BiotSavart(modular_coils), curve, mu0, ...)
    fl.fix('I1'); fl.run_code(length)           # square (stage-2) polish
    field = SingularBiotSavart(fl)
    bs = BoozerSurface(field, surface, Volume(surface), surface.volume())
    bs.run_code(iota, G)                         # surface in the TOTAL field
    pfl = PeriodicFieldLine(field, axis_curve)
    pfl.run_code(CurveLength(axis_curve).J())

Re-solve on coil change
-----------------------
The aux field depends on ``mu``, and the DEPENDENT components of ``mu`` are an
implicit function of the modular-coil dofs (solved by the singular polish).  So
whenever a modular-coil dof changes, ``fl`` must be re-polished before the total
field is evaluated.  ``fl`` is an ``Optimizable`` depending on the modular
``BiotSavart``; its ``recompute_bell`` sets ``need_to_run_code = True`` when a
coil dof changes, and :meth:`_prepare` re-runs the polish lazily (once) before
each field evaluation.  ``SingularBiotSavart`` depends on ``fl``, so its own
value cache is cleared by the same dependency change.

Adjoints (consistent total derivative)
--------------------------------------
``B_vjp`` / ``B_and_dB_vjp`` return the TOTAL derivative of the field with
respect to the modular-coil dofs and the free (independent) ``mu`` dofs.  For a
seed ``v`` (and ``vgrad``), with ``a = v . dB_aux/dmu (+ vgrad . dgradB_aux/dmu)``
a length-``nmu`` covector, the result is

    modular.B_vjp(v)                                   # dB_modular/dc
  + Derivative({fl: a on the FREE mu slots})           # dB_aux/dmu_indep
  + sum_q a[dep_q] * fl.dmu_by_dindependent()[name_q]  # dB_aux/dmu_dep . dmu_dep/d(indep,c)

the last term being the implicit sensitivity of the dependent ``mu`` through the
polish (so the gradient is consistent with the re-solve).  This requires the
square (stage-2) partition for which ``dmu_by_dindependent`` is defined.
"""

import numpy as np

from simsopt.field import MagneticField, BiotSavart, Coil, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt._core.derivative import Derivative
from simsopt.geo import CurveLength, CurveXYZFourier

# Forward (B / gradB / grad-gradB) is computed with a pure-C++ BiotSavart of the
# aux circles (built once, dofs synced from mu) for speed; only the mu-ADJOINT
# keeps the jax aux derivatives (_dB_aux_by_dmu, _dgradB_aux_by_dmu).
from star_lite_design.utils.singularperiodicfieldline import (
    _dB_aux_by_dmu, _dgradB_aux_by_dmu, _mu_names, _CURRENT_SCALE)

__all__ = ['SingularBiotSavart']


class SingularBiotSavart(MagneticField):
    """Total (modular + auxiliary) magnetic field of a
    :class:`SingularPeriodicFieldline`, usable anywhere a ``BiotSavart`` is.

    Parameters
    ----------
    fl : SingularPeriodicFieldline
        The polished singular field line carrying the modular ``BiotSavart``
        (``fl.biotsavart``) and the auxiliary-coil parameters (``fl.mu``).  Its
        partition must be the square stage-2 one (only the dependent currents
        fixed) for the adjoints to be available.
    """

    def __init__(self, fl):
        MagneticField.__init__(self, depends_on=[fl])
        self.fl = fl
        # mirror BiotSavart's attribute so BoozerSurface can read the modular
        # currents (it uses both `.coils` and `._coils`) to form G.
        self._coils = fl.biotsavart.coils
        self._aux_dirty = True            # aux coil dofs need (re)syncing from mu
        self._build_aux()

    def recompute_bell(self, parent=None):
        # Idiomatic simsopt invalidation: a change in a depends_on object (the
        # modular coils, or fl's free mu) means the aux BiotSavart dofs are stale
        # and must be re-synced from mu before the next forward eval. Mark them
        # dirty and let the base clear the cached B / gradB / grad-gradB.
        self._aux_dirty = True
        super().recompute_bell(parent)

    # ---- BiotSavart-like coil access (modular coils only) --------------------
    @property
    def coils(self):
        return self.fl.biotsavart.coils

    @property
    def modular(self):
        """The underlying modular-coil BiotSavart field."""
        return self.fl.biotsavart

    # ---- auxiliary-coil C++ BiotSavart (forward only; synced from mu) --------
    def _build_aux(self):
        """Build a C++ BiotSavart of the auxiliary circular coils ONCE; its dofs
        are re-synced from fl.mu before each forward evaluation. It is a fast
        forward calculator only and is NOT in this Optimizable's depends_on, so
        its (free) dofs never enter the optimization graph; the mu-adjoint is
        handled separately by the jax derivatives in B_vjp / B_and_dB_vjp."""
        mu = np.asarray(self.fl.mu)
        self._naux = (mu.shape[0] - 1) // 2
        qp = np.linspace(0.0, 1.0, 160, endpoint=False)
        self._aux_base_curves = []
        self._aux_currents = []                  # inner Current objects (raw mu units)
        scaled = []
        for k in range(self._naux):
            c = CurveXYZFourier(qp, 1)
            c.x = c.x * 0.
            self._aux_base_curves.append(c)
            cur = Current(float(mu[k]))
            self._aux_currents.append(cur)
            scaled.append(ScaledCurrent(cur, _CURRENT_SCALE))
        if self.fl.stellsym_aux:
            aux_coils = coils_via_symmetries(self._aux_base_curves, scaled, 1, True)
        else:
            aux_coils = [Coil(c, sc) for c, sc in zip(self._aux_base_curves, scaled)]
        self._aux_bs = BiotSavart(aux_coils)
        self._aux_dirty = True
        self._sync_aux()

    def _sync_aux(self):
        """Write the current fl.mu (radii r_k, height Z, currents I_k) into the
        aux BiotSavart's coil dofs, but ONLY when marked dirty by recompute_bell
        (mu is constant within a single Newton solve, so this skips the per-call
        .set / recompute_bell overhead). The stellsym -Z partners track the base
        curves automatically (coils_via_symmetries)."""
        if not self._aux_dirty:
            return
        mu = np.asarray(self.fl.mu)
        N = self._naux
        Z = float(mu[-1])
        for k in range(N):
            rk = float(mu[N + k])
            c = self._aux_base_curves[k]
            c.set('xc(1)', rk)
            c.set('ys(1)', rk)
            c.set('zc(0)', Z)
            self._aux_currents[k].x = np.array([float(mu[k])])
        self._aux_dirty = False

    # ---- re-solve + point bookkeeping ---------------------------------------
    def _ensure_solved(self):
        """Re-run the singular polish if anything upstream (modular coils or the
        free mu) changed since the last solve."""
        fl = self.fl
        if fl.need_to_run_code:
            length = None
            if getattr(fl, 'res', None) is not None:
                length = fl.res.get('length')
            if length is None:
                length = CurveLength(fl.curve).J()
            fl.run_code(length)

    def _prepare(self):
        """Ensure the polish is current, re-sync the aux coils from mu, and point
        BOTH the modular and aux BiotSavart fields at this wrapper's evaluation
        points (the polish leaves the modular field pointed at the field-line's
        own quadrature points), then return the points."""
        pts = self.get_points_cart_ref()
        self._ensure_solved()
        self._sync_aux()
        self.modular.set_points(pts)
        self._aux_bs.set_points(pts)
        return pts

    def _set_points_cb(self):
        # propagate points to the modular + aux fields (mirrors MagneticFieldSum)
        pts = self.get_points_cart_ref()
        self.modular.set_points_cart(pts)
        self._aux_bs.set_points_cart(pts)

    # ---- forward field (modular + aux), both pure C++ ------------------------
    def _B_impl(self, B):
        self._prepare()
        B[:] = self.modular.B() + self._aux_bs.B()

    def _dB_by_dX_impl(self, dB):
        self._prepare()
        dB[:] = self.modular.dB_by_dX() + self._aux_bs.dB_by_dX()

    def _d2B_by_dXdX_impl(self, ddB):
        self._prepare()
        ddB[:] = self.modular.d2B_by_dXdX() + self._aux_bs.d2B_by_dXdX()

    def compute(self, derivatives=0):
        """Populate the B (and up to ``derivatives`` spatial-derivative) caches.

        ``BoozerSurface``'s residual calls ``biotsavart.compute(derivatives)``;
        that method lives on the C++ ``BiotSavart`` but not on the generic
        ``MagneticField`` base, so we provide it by triggering the lazy
        accessors (each routes through the _impl hooks and is then cached)."""
        self.B()
        if derivatives >= 1:
            self.dB_by_dX()
        if derivatives >= 2:
            self.d2B_by_dXdX()
        return self

    # ---- adjoints ------------------------------------------------------------
    def _mu_to_derivative(self, a):
        """Turn a length-nmu covector ``a = seed . dB_aux/dmu`` into a simsopt
        Derivative over the FREE mu dofs (direct) plus the implicit dependent-mu
        sensitivity through the polish (over free mu + modular coils)."""
        fl = self.fl
        a = np.asarray(a, dtype=float)
        nmu = a.size
        free_mask = np.asarray(fl.local_dofs_free_status, dtype=bool)  # True = independent
        # direct: free (independent) mu slots
        g = np.zeros(nmu)
        g[free_mask] = a[free_mask]
        deriv = Derivative({fl: g})
        # implicit: dependent (fixed) mu slots propagate through dmu/d(indep, c)
        dep_idx = np.where(~free_mask)[0]
        if dep_idx.size:
            dmu = fl.dmu_by_dindependent()        # {name: Derivative}
            names = _mu_names(nmu)
            for q in dep_idx:
                name = names[q]
                if name in dmu and a[q] != 0.0:
                    deriv = deriv + float(a[q]) * dmu[name]
        return deriv

    def B_vjp(self, v):
        pts = self._prepare()
        fl = self.fl
        vv = np.ascontiguousarray(np.asarray(v).reshape((-1, 3)))
        # aux sensitivity wrt mu at the evaluation points
        dBaux_dmu = np.asarray(_dB_aux_by_dmu(pts, fl.mu, stellsym=fl.stellsym_aux))  # (N,3,nmu)
        a = np.einsum('ij,ijk->k', vv, dBaux_dmu, optimize=True)
        # modular part FIRST (modular points are the eval points here); the mu
        # term below calls dmu_by_dindependent, which re-points the modular field.
        deriv = self.modular.B_vjp(vv)
        deriv = deriv + self._mu_to_derivative(a)
        return deriv

    def B_and_dB_vjp(self, v, vgrad):
        pts = self._prepare()
        fl = self.fl
        vv = np.ascontiguousarray(np.asarray(v).reshape((-1, 3)))
        vg = np.ascontiguousarray(np.asarray(vgrad).reshape((-1, 3, 3)))
        dBaux_dmu = np.asarray(_dB_aux_by_dmu(pts, fl.mu, stellsym=fl.stellsym_aux))       # (N,3,nmu)
        dgradBaux_dmu = np.asarray(_dgradB_aux_by_dmu(pts, fl.mu, stellsym=fl.stellsym_aux))  # (N,3,3,nmu) [i,k,j,l]
        a = np.einsum('ij,ijk->k', vv, dBaux_dmu, optimize=True)
        a = a + np.einsum('ikj,ikjl->l', vg, dgradBaux_dmu, optimize=True)
        mod0, mod1 = self.modular.B_and_dB_vjp(vv, vg)
        # consumers always sum the returned 2-tuple; lump the aux/implicit part
        # (covering both seeds) into the first element.
        return (mod0 + self._mu_to_derivative(a), mod1)
