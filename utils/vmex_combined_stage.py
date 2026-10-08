"""VMEX-backed simsopt Optimizables for combined stage-1/2 optimization.

Implements the building blocks of the combined plasma + coil optimization of
Jorge, Goodman, Landreman, Rodrigues & Wechsung, PPCF 65 074003 (2023), with
two changes proposed by A. Giuliani:

1. the fixed-boundary equilibrium is solved with VMEX, so every plasma quantity
   has an exact (implicit-adjoint) derivative instead of finite differences;
2. plasma/coil consistency is an equality constraint enforced with an augmented
   Lagrangian (``utils/augmented_lagrangian.py``) instead of a large fixed weight.

Objects
-------
:class:`VmexPlasma`
    Owns the plasma degrees of freedom ``s`` (boundary Fourier modes up to
    ``max_mode`` and, optionally, current-profile dofs) and caches ONE jitted
    VMEX forward evaluation per dof vector.  That evaluation returns the
    quasisymmetry cost and all virtual-casing interface data; its JAX pullback is
    kept, so every consumer below shares a single adjoint per gradient.
:class:`VmexQuasisymmetry`
    ``f_QS(s)``: the VMEX least-squares cost ``0.5 |r|^2`` of the objective terms
    the plasma was built with (quasisymmetry, aspect, iota, beta, ...).
:class:`PlasmaCoilInterface`
    The consistency constraints between the equilibrium and a coil field:
    ``bnormal`` -- the vacuum-side normal field ``(B_plasma + B_coil).n / B_ref``
    as low-order Fourier modes (default) or area-weighted point values -- plus
    ONE field-strength condition pinning the field component ``B.n`` cannot
    see (the harmonic toroidal field set by the total poloidal coil current;
    without it a vacuum problem is solved by switching the coils off):
    ``toroidal_flux`` -- the coil flux through the boundary cross-section at
    ``phi = 0`` equals PHIEDGE (exact gradients) -- or ``pressure_balance`` --
    ``<|B_out|^2 - |B_in|^2 - 2 mu0 p_edge> / B_ref^2`` (finite beta).
:class:`CoilPlasmaDistance`
    One-sided quadratic penalty keeping the coils ``minimum_distance`` away from
    the (moving) plasma boundary.
:class:`WeightedSum`
    A sum of weighted objectives that merges shared pullbacks, so e.g.
    ``f_QS + w * coil-plasma distance`` still costs one VMEX adjoint.

Array layouts: VMEX interface arrays are ``(3, nphi, ntheta)`` over ONE field
period ``phi in [0, 2 pi / nfp)``; simsopt fields take ``(npoints, 3)``.
"""
import numpy as np
import jax
import jax.numpy as jnp

from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec

from .augmented_lagrangian import EqualityConstraint, InequalityConstraint, merge_cotangents

__all__ = ["VmexPlasma", "VmexQuasisymmetry", "PlasmaCoilInterface",
           "CoilPlasmaDistance", "WeightedSum", "bnormal_fourier_modes", "coil_stellsym_error",
           "fix_self_symmetric_coil_parity", "net_poloidal_current_output", "bootstrap_mismatch_output",
           "kinetic_profiles", "kinetic_pressure_coeffs", "BootstrapConsistency"]

MU0 = 4e-7 * np.pi


def _to_points(a):
    """(3, nphi, ntheta) -> (nphi, ntheta, 3)."""
    return np.ascontiguousarray(np.moveaxis(np.asarray(a, dtype=float), 0, -1))


def _to_jax_layout(a):
    """(nphi, ntheta, 3) -> (3, nphi, ntheta)."""
    return np.ascontiguousarray(np.moveaxis(np.asarray(a, dtype=float), -1, 0))


class VmexPlasma(Optimizable):
    """Plasma dofs ``s`` and the cached fixed-boundary VMEX equilibrium they define.

    Args:
        inp: seed :class:`vmex.VmecInput` (its resolution, profiles and multigrid
            ladder are used for every solve).
        objective_terms: VMEX ``(function, target, weight)`` tuples defining
            ``f_QS``; see :meth:`vmex.optimize.VmecProblem.from_tuples`.
        max_mode: largest boundary mode number varied.
        nphi, ntheta: interface grid per field period for virtual casing.
        vc_digits: virtual-casing accuracy (the quadrature plan is chosen once,
            on the seed boundary, so it stays differentiable).
        plasma_field: ``"virtual_casing"`` -- the plasma's own boundary field
            from virtual casing (finite beta); ``"vacuum"`` -- neglect it
            (``B_plasma = 0``), the vacuum single-stage formulation.
        nsec: points on the ``phi = 0`` boundary cross-section (outputs
            ``xsec``/``xsec_tangent``, used for the toroidal-flux condition),
            evaluated directly from the boundary Fourier coefficients.
        nfine: ``(nphi, ntheta)`` of ``boundary_points``, the boundary sampled over
            the FULL torus straight from the Fourier coefficients (used for
            distances to X-points and coils; exact in the boundary dofs).
        current_dofs: number of current-profile dofs (``None``: fixed profile).
        vary_major_radius: also vary ``RBC(0,0)``.
        restart_from: converged seed equilibrium (solved if omitted).
        extra_outputs: ``{name: fn(state, runtime) -> array}``, further equilibrium quantities evaluated in the same
            jitted forward and pulled back through the same adjoint, e.g. :func:`net_poloidal_current_output`
            (``"rbtor"``) or :func:`bootstrap_mismatch_output` (``"bootstrap"``).
        **problem_kwargs: forwarded to ``VmecProblem.from_tuples`` (e.g.
            ``adjoint_tol``, ``forward_ftol``, ``use_ess``).

    Gradient caveat: each trial is a hot-restarted host solve, while VMEX's
    implicit adjoint differentiates the fixed point of its *frozen* residual.
    The discrete equilibrium is not unique along a near-null direction (lambda /
    m=1 content; hot and cold solves of the same boundary differ there), so
    quantities read at VMEC grid points -- ``B_plasma``, ``Bn_plasma`` and
    ``Bin_mag2`` -- have adjoint gradients that differ from re-solve finite
    differences by ~0.5-4 % (up to ~15 % in field direction).  The adjoint
    matches VMEX's frozen-path FD to <= 1e-6.  ``f_QS`` and the boundary
    geometry (``gamma``, ``normal``, ``weights``, ``xsec``) are unaffected.
    See combined_stage/README.md.
    """

    OUTPUT_KEYS = ("fqs", "rows", "gamma", "normal", "Bn_plasma", "B_plasma", "weights", "Bin_mag2",
                   "xsec", "xsec_tangent", "boundary_points")

    def __init__(self, inp, objective_terms, *, max_mode, nphi=16, ntheta=16, vc_digits=4,
                 plasma_field="virtual_casing", nsec=128, nfine=(96, 48), current_dofs=None,
                 vary_major_radius=False, restart_from=None, extra_outputs=None, **problem_kwargs):
        from vmex import optimize as opt
        from vmex.core import virtual_casing as vc

        if plasma_field not in ("virtual_casing", "vacuum"):
            raise ValueError("plasma_field must be 'virtual_casing' or 'vacuum'")
        if restart_from is None:
            restart_from = opt.solve_equilibrium(inp)
        self.inp = inp
        self.nfp = int(inp.nfp)
        self.stellsym = not bool(inp.lasym)
        self.nphi, self.ntheta, self.vc_digits = int(nphi), int(ntheta), int(vc_digits)
        self.plasma_field = plasma_field
        self.problem = opt.VmecProblem.from_tuples(
            inp, objective_terms, max_mode=max_mode, current_dofs=current_dofs,
            vary_major_radius=vary_major_radius, restart_from=restart_from, **problem_kwargs)
        md = self.problem.metadata
        state_runtime_status = md["jax_state_runtime_status"]
        residual_rows = md["jax_residual_from_state"]
        self.term_slices = md.get("term_slices")

        x0 = np.asarray(self.problem.x0, dtype=float)
        self.precision = None
        if plasma_field == "virtual_casing":
            state0, runtime0 = md["jax_state_runtime"](jnp.asarray(x0))
            surface0 = vc.surface_field_data_from_state(inp, state0, runtime=runtime0,
                                                        nphi=self.nphi, ntheta=self.ntheta)
            self.precision = vc.plan_vc_precision(surface0, digits=self.vc_digits)
        nphi_, ntheta_, digits_, precision_ = self.nphi, self.ntheta, self.vc_digits, self.precision
        use_vc = plasma_field == "virtual_casing"
        boundary_from_x = self.problem.boundary_from_x
        theta_sec = jnp.linspace(0.0, 2.0 * jnp.pi, int(nsec), endpoint=False)
        phi_fine = jnp.linspace(0.0, 2.0 * jnp.pi, int(nfine[0]), endpoint=False)
        theta_fine = jnp.linspace(0.0, 2.0 * jnp.pi, int(nfine[1]), endpoint=False)
        nfp_ = self.nfp
        extra = dict(extra_outputs or {})
        clash = set(extra) & set(self.OUTPUT_KEYS)
        if clash:
            raise ValueError(f"extra_outputs may not reuse the built-in output names {sorted(clash)}")
        self.output_keys = self.OUTPUT_KEYS + tuple(extra)

        def graph(x):
            state, runtime, status = state_runtime_status(x)
            rows = residual_rows(state, runtime)
            surface = vc.surface_field_data_from_state(inp, state, runtime=runtime,
                                                       nphi=nphi_, ntheta=ntheta_)
            if use_vc:
                interface = vc.PlasmaVacuumInterface.from_surface_data(
                    surface, digits=digits_, precision=precision_)
                gamma, normal, weights = interface.gamma, interface.normal, interface.weights
                B_plasma, Bn_plasma, Bin_mag2 = interface.B_plasma, interface.Bn_plasma, interface.Bin_mag2
            else:
                gamma, normal = jnp.asarray(surface.gamma), jnp.asarray(surface.normal)
                area = jnp.linalg.norm(jnp.asarray(surface.area_vector), axis=0)
                weights = area / jnp.sum(area)
                B_plasma = jnp.zeros_like(gamma)
                Bn_plasma = jnp.zeros_like(area)
                Bin_mag2 = jnp.sum(jnp.asarray(surface.B_total) ** 2, axis=0)
            # phi = 0 cross-section straight from the (prescribed) boundary coefficients:
            # R = sum rbc cos(m theta), Z = sum zbs sin(m theta), summed over n
            rbc, zbs = boundary_from_x(x)[:2]
            m = jnp.arange(rbc.shape[1], dtype=float)
            rc, zs = jnp.sum(rbc, axis=0), jnp.sum(zbs, axis=0)
            cos_mt, sin_mt = jnp.cos(theta_sec[:, None] * m), jnp.sin(theta_sec[:, None] * m)
            R, Z = cos_mt @ rc, sin_mt @ zs
            zero = jnp.zeros_like(R)
            # full-torus boundary: R = sum rbc[n,m] cos(m theta - n nfp phi), Z = sum zbs sin(...)
            n = jnp.arange(rbc.shape[0], dtype=float) - (rbc.shape[0] - 1) // 2
            ang = (m[None, None, None, :] * theta_fine[None, :, None, None]
                   - n[None, None, :, None] * nfp_ * phi_fine[:, None, None, None])
            Rf = jnp.einsum("pqnm,nm->pq", jnp.cos(ang), rbc)
            Zf = jnp.einsum("pqnm,nm->pq", jnp.sin(ang), zbs)
            boundary_points = jnp.stack([Rf * jnp.cos(phi_fine)[:, None], Rf * jnp.sin(phi_fine)[:, None], Zf])
            out = dict(fqs=0.5 * jnp.vdot(rows, rows), rows=rows, gamma=gamma, normal=normal,
                       Bn_plasma=Bn_plasma, B_plasma=B_plasma, weights=weights, Bin_mag2=Bin_mag2,
                       xsec=jnp.stack([R, zero, Z]),
                       xsec_tangent=jnp.stack([-(sin_mt * m) @ rc, zero, (cos_mt * m) @ zs]),
                       boundary_points=boundary_points)
            for name, fn in extra.items():
                out[name] = jnp.asarray(fn(state, runtime))
            return out, (status, rows)

        self._graph = jax.jit(graph)
        self._failure_value_and_grad = jax.jit(jax.value_and_grad(md["jax_failure_value"]))
        self._cache = None
        self.n_forward = 0
        self.n_backward = 0
        Optimizable.__init__(self, x0=x0, names=list(self.problem.names))

    # -- evaluation cache -------------------------------------------------------

    def recompute_bell(self, parent=None):
        self._cache = None

    def evaluate(self):
        """Run (or return the cached) forward evaluation at the current dofs."""
        if self._cache is not None:
            return self._cache
        x = jnp.asarray(self.local_full_x)
        out, vjp_fn, (status, rows) = jax.vjp(self._graph, x, has_aux=True)
        self.n_forward += 1
        out = {k: np.asarray(v, dtype=float) for k, v in out.items()}
        status = int(status)
        accepted = status == 0 and all(np.all(np.isfinite(v)) for v in out.values())
        cache = dict(out=out, vjp=vjp_fn, status=status, accepted=accepted,
                     rows=np.asarray(rows, dtype=float))
        if not accepted:
            value, grad = self._failure_value_and_grad(x)
            cache["failure"] = (float(value), np.asarray(grad, dtype=float))
        self._cache = cache
        return cache

    @property
    def accepted(self):
        return self.evaluate()["accepted"]

    def outputs(self):
        return self.evaluate()["out"]

    def pullback(self, cotangents):
        """``Derivative`` of ``sum_k <cotangents[k], outputs[k]>`` w.r.t. ``s``."""
        cache = self.evaluate()
        if not cache["accepted"]:
            return Derivative({self: np.zeros(self.local_full_dof_size)})
        ct = {}
        for key in self.output_keys:
            ref = cache["out"][key]
            ct[key] = jnp.asarray(np.broadcast_to(np.asarray(cotangents[key], dtype=float), ref.shape)
                                  if key in cotangents else np.zeros_like(ref))
        grad, = cache["vjp"](ct)
        self.n_backward += 1
        return Derivative({self: np.asarray(grad, dtype=float)})

    # -- conveniences -----------------------------------------------------------

    def term_costs(self):
        """``{term_name: 0.5 |rows|^2}`` of the VMEX objective terms."""
        rows = self.evaluate()["rows"]
        if not self.term_slices:
            return {"f_QS": 0.5 * float(rows @ rows)}
        return {name: 0.5 * float(rows[a:b] @ rows[a:b]) for name, a, b in self.term_slices}

    def vmec_input(self):
        return self.problem.input_from_x(np.asarray(self.local_full_x))

    def equilibrium(self):
        """Converged equilibrium at the current dofs (reuses the accepted state)."""
        return self.problem.equilibrium_from_x(np.asarray(self.local_full_x))


def _term_mask(plasma, names):
    """Boolean mask of the residual rows belonging to the named VMEX objective terms."""
    rows = plasma.evaluate()["rows"]
    mask = np.zeros(rows.size, dtype=bool)
    slices = {name: (a, b) for name, a, b in (plasma.term_slices or [])}
    for name in names:
        if name not in slices:
            raise KeyError(f"no VMEX objective term {name!r}; terms: {sorted(slices)}")
        a, b = slices[name]
        mask[a:b] = True
    return mask


class VmexQuasisymmetry(Optimizable):
    """``f_QS(s)`` -- the VMEX least-squares cost of ``plasma``'s objective terms.

    ``exclude`` drops named terms (e.g. one weighted separately by :class:`VmexTermCost`).
    At a rejected (non-converged) trial it returns VMEX's smooth failure wall and
    its gradient, so line searches retreat instead of stalling.
    """

    def __init__(self, plasma, exclude=()):
        self.plasma, self.exclude = plasma, tuple(exclude)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[plasma])

    def J(self):
        cache = self.plasma.evaluate()
        if not cache["accepted"]:
            return cache["failure"][0]
        if not self.exclude:
            return cache["out"]["fqs"].item()
        rows = cache["out"]["rows"][~_term_mask(self.plasma, self.exclude)]
        return 0.5 * float(rows @ rows)

    def dJ_parts(self):
        cache = self.plasma.evaluate()
        if not cache["accepted"]:
            return Derivative({self.plasma: cache["failure"][1]}), {}
        if not self.exclude:
            return Derivative({}), {self.plasma: {"fqs": 1.0}}
        rows = cache["out"]["rows"]
        return Derivative({}), {self.plasma: {"rows": np.where(_term_mask(self.plasma, self.exclude), 0.0, rows)}}

    @derivative_dec
    def dJ(self):
        derivative, parts = self.dJ_parts()
        for owner, ct in parts.items():
            derivative += owner.pullback(ct)
        return derivative


class VmexTermCost(Optimizable):
    """``0.5 |rows|^2`` of ONE named VMEX objective term, so it can carry its own (escalated) weight.

    Zero at a rejected trial (the failure wall is carried by :class:`VmexQuasisymmetry`).
    """

    def __init__(self, plasma, term_name):
        # not `self.name`: Optimizable.__init__ overwrites that with its own instance name ("VmexTermCost1")
        self.plasma, self.term_name = plasma, term_name
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[plasma])

    def J(self):
        cache = self.plasma.evaluate()
        if not cache["accepted"]:
            return 0.0
        rows = cache["out"]["rows"][_term_mask(self.plasma, [self.term_name])]
        return 0.5 * float(rows @ rows)

    def dJ_parts(self):
        cache = self.plasma.evaluate()
        if not cache["accepted"]:
            return Derivative({}), {}
        rows = cache["out"]["rows"]
        return Derivative({}), {self.plasma: {"rows": np.where(_term_mask(self.plasma, [self.term_name]), rows, 0.0)}}

    @derivative_dec
    def dJ(self):
        derivative, parts = self.dJ_parts()
        for owner, ct in parts.items():
            derivative += owner.pullback(ct)
        return derivative


def bnormal_fourier_modes(mpol, ntor, stellsym):
    """``[(family, m, n)]`` of the constrained ``B.n`` harmonics.

    ``B.n`` on a stellarator-symmetric boundary is odd, so only
    ``sin(m theta - n nfp phi)`` survives; ``cos`` modes are added when the
    configuration is not stellarator symmetric.
    """
    modes = []
    for m in range(mpol + 1):
        for n in range(-ntor, ntor + 1):
            if m == 0 and n <= 0:
                continue
            modes.append(("sin", m, n))
    if not stellsym:
        for m in range(mpol + 1):
            for n in range(-ntor, ntor + 1):
                if m == 0 and n < 0:
                    continue
                modes.append(("cos", m, n))
    return modes


def net_poloidal_current_output():
    """``VmexPlasma`` extra output ``"rbtor"``: the edge ``G = bvco(edge)`` [T m] of the live equilibrium state.

    ``2 pi G / mu0`` is the net poloidal current outside the boundary -- the current the coils must link through the
    torus hole (:class:`PlasmaCoilInterface` ``field_strength="net_poloidal_current"``). VMEC's ``rbtor``
    extrapolation ``1.5 bvco(ns) - 0.5 bvco(ns-1)``; converged in ns already at ns = 16 (1.7e-6 vs ns = 101 on the
    band-run final, job 2119), adjoint vs VMEX frozen-path FD 2.8e-5.
    """
    from vmex.core.fields import surface_currents
    from vmex.core.statephysics import _field_chain

    def rbtor(state, runtime):
        fields = _field_chain(state, runtime)[3]
        return surface_currents(bsubu=fields.bsubu, bsubv=fields.bsubv, trig=runtime.trig,
                                s=jnp.asarray(runtime.setup.s_full), signgs=runtime.setup.signgs).rbtor
    return rbtor


def kinetic_profiles(ne, Te, Ti, Zeff=(1.0,)):
    """``vmex.core.bootstrap.KineticProfiles`` from polynomial coefficients in ``s`` (lowest order first).

    Units: ``ne`` [1/m^3], ``Te``/``Ti`` [eV], ``Zeff`` dimensionless (Redl's fit: hydrogenic main ions plus an
    effective charge -- a pure argon plasma is outside what it was fitted to).
    """
    from vmex.core.bootstrap import KineticProfiles
    return KineticProfiles(ne_coeffs=np.asarray(ne, dtype=float), Te_coeffs=np.asarray(Te, dtype=float),
                           Ti_coeffs=np.asarray(Ti, dtype=float), Zeff_coeffs=np.asarray(Zeff, dtype=float))


def kinetic_pressure_coeffs(ne, Te, Ti):
    """Power-series coefficients (lowest order first) of ``p(s) = e ne (Te + Ti)`` [Pa], ``n_i = n_e``.

    The bootstrap current is computed from the kinetic profiles while the equilibrium uses VMEC's pressure: they must
    describe the same plasma, so with bootstrap on, the seed's ``AM``/``PRES_SCALE`` are replaced by this.
    """
    e = 1.602176634e-19
    return e * np.polynomial.polynomial.polyadd(np.polynomial.polynomial.polymul(ne, Te),
                                                np.polynomial.polynomial.polymul(ne, Ti))


def bootstrap_mismatch_output(profiles, helicity_n, surfaces=None):
    """``VmexPlasma`` extra output ``"bootstrap"``: VMEX's ``RedlBootstrapMismatch`` residual rows of the live state.

    ``R_j = (<J.B>_vmec(s_j) - <J.B>_Redl(s_j)) / sqrt(sum_k (<J.B>_vmec + <J.B>_Redl)^2)`` on ``surfaces``: zero when
    the equilibrium's parallel current is exactly the Redl (2021) bootstrap current of ``profiles``. Only the
    bootstrap drive is modelled -- no ohmic or driven current -- and the equilibrium pressure must match the kinetic
    pressure (:func:`kinetic_pressure_coeffs`). ``helicity_n`` is the quasisymmetry helicity (0 = QA) Redl's
    tokamak-to-QS isomorphism uses.
    """
    from vmex.core.bootstrap import RedlBootstrapMismatch
    mismatch = RedlBootstrapMismatch(profiles, int(helicity_n), surfaces)
    return mismatch.residuals_state


class BootstrapConsistency(EqualityConstraint):
    """Bootstrap self-consistency as an augmented-Lagrangian equality block ``"bootstrap"``.

    The residual rows are :class:`VmexPlasma`'s ``"bootstrap"`` output (:func:`bootstrap_mismatch_output`); the design
    variables that can meet them are the current-profile dofs (``VmexPlasma(current_dofs=k)``: ``k`` AC coefficients +
    CURTOR). Its gradient is one more cotangent on the shared VMEX adjoint.
    """

    def __init__(self, plasma):
        if "bootstrap" not in getattr(plasma, "output_keys", ()):
            raise ValueError("BootstrapConsistency needs VmexPlasma(..., extra_outputs={'bootstrap': "
                             "bootstrap_mismatch_output(...)})")
        self.plasma = plasma
        EqualityConstraint.__init__(self, x0=np.asarray([]), depends_on=[plasma])

    def residuals(self):
        return {"bootstrap": np.asarray(self.plasma.outputs()["bootstrap"], dtype=float).ravel()}

    def residuals_vjp_parts(self, cotangents):
        if not self.plasma.accepted:
            return Derivative({}), {}
        return Derivative({}), {self.plasma: {"bootstrap": np.asarray(cotangents["bootstrap"], dtype=float)}}


class PlasmaCoilInterface(EqualityConstraint):
    """Plasma/coil consistency residuals on the VMEX boundary.

    Args:
        plasma: :class:`VmexPlasma`.
        field: simsopt ``BiotSavart`` of the coils (give it its OWN instance: the
            evaluation points are reset on every call).
        mode: ``"fourier"`` -- constrain ``B.n`` harmonics with ``m <= mpol`` and
            ``|n| <= ntor`` (keeps the constraint count below the dof count, so
            exact feasibility is possible); ``"points"`` -- one residual per grid
            point, weighted by ``sqrt(area weight)``.
        mpol, ntor: harmonic truncation for ``mode="fourier"``; must satisfy
            ``mpol < ntheta/2`` and ``ntor < nphi/2`` (discrete orthogonality).
        B_ref: normalization [T]; default ``sqrt(<|B|^2>)`` on the SEED boundary,
            then frozen (a constant, so it adds no derivative terms).
        p_edge: pressure at the boundary [Pa] (0 at a true plasma edge).
        field_strength: ``"toroidal_flux"`` -- ``(Phi_coil - Phi_target)/|Phi_target|``
            with ``Phi_coil`` the coil flux ``oint A.dl`` through the ``phi = 0``
            boundary cross-section and ``|Phi_target| = |PHIEDGE|`` (vacuum; every
            gradient analytic); ``"pressure_balance"`` -- mean total-pressure jump
            (finite beta; uses ``|B_in|^2`` at the boundary); ``"net_poloidal_current"`` --
            ``(I_coil - 2 pi G/mu0) / |2 pi G_seed/mu0|``: the current the coils link through the torus hole,
            ``I_coil = oint B_coil . dl / mu0`` around a fixed circle through the plasma, equals the net poloidal
            current the equilibrium needs outside its boundary, ``G = bvco(edge)`` (Boozer G; VMEX ``rbtor``). Exact at
            any beta (the plasma's own poloidal currents lie inside the boundary and drop out), linear in the coil
            currents, and its plasma gradient is one scalar through the shared adjoint. Needs ``plasma`` built with
            ``extra_outputs={"rbtor": net_poloidal_current_output()}``; ``None`` -- omit.
        nloop: points on the circle for ``oint B_coil . dl`` (``net_poloidal_current``).
    """

    def __init__(self, plasma, field, *, mode="fourier", mpol=4, ntor=4, B_ref=None,
                 p_edge=0.0, field_strength="toroidal_flux", nloop=512):
        if mode not in ("fourier", "points"):
            raise ValueError("mode must be 'fourier' or 'points'")
        if field_strength not in ("toroidal_flux", "pressure_balance", "net_poloidal_current", None):
            raise ValueError("field_strength must be 'toroidal_flux', 'pressure_balance', 'net_poloidal_current' "
                             "or None")
        if field_strength == "net_poloidal_current" and "rbtor" not in getattr(plasma, "output_keys", ()):
            raise ValueError("field_strength 'net_poloidal_current' needs VmexPlasma(..., extra_outputs="
                             "{'rbtor': net_poloidal_current_output()})")
        self.plasma, self.field = plasma, field
        self.mode, self.p_edge, self.field_strength = mode, float(p_edge), field_strength
        nphi, ntheta, nfp = plasma.nphi, plasma.ntheta, plasma.nfp
        if mode == "fourier":
            if not (2 * mpol < ntheta and 2 * ntor < nphi):
                raise ValueError(f"need 2*mpol < ntheta and 2*ntor < nphi, got mpol={mpol}, "
                                 f"ntor={ntor}, ntheta={ntheta}, nphi={nphi}")
            self.modes = bnormal_fourier_modes(mpol, ntor, plasma.stellsym)
            theta = np.linspace(0.0, 2 * np.pi, ntheta, endpoint=False)
            phi = np.linspace(0.0, 2 * np.pi / nfp, nphi, endpoint=False)
            PHI, THETA = np.meshgrid(phi, theta, indexing="ij")
            npts = nphi * ntheta
            rows = []
            for family, m, n in self.modes:
                angle = (m * THETA - n * nfp * PHI).ravel()
                basis = np.sin(angle) if family == "sin" else np.cos(angle)
                rows.append(basis * (1.0 if (family == "cos" and m == 0 and n == 0) else 2.0) / npts)
            self.projection = np.array(rows)
        else:
            self.modes, self.projection = None, None
        if B_ref is None:
            out = plasma.outputs()
            B_ref = float(np.sqrt(np.sum(out["weights"] * out["Bin_mag2"])))
        self.B_ref = float(B_ref)
        if field_strength == "toroidal_flux":
            flux0 = self._coil_flux(plasma.outputs())[0]
            phiedge = float(plasma.inp.phiedge)
            # the orientation of oint A.dl depends on the theta direction; match signs
            self.flux_seed, self.flux_target = flux0, float(np.sign(flux0)) * abs(phiedge)
        elif field_strength == "net_poloidal_current":
            # Fixed circle z = 0 at the seed boundary's R00, through the plasma: every modular coil links it once, so
            # oint B_coil . dl / mu0 is the linked coil current whatever the coils' shape (it only changes if a coil
            # crossed the circle). Do NOT sum coil currents instead: simsopt's stellarator-symmetric copies carry
            # flipped signs, and the plain sum over a symmetric coil set is exactly 0 (band-run final: 0 vs 6.29 MA).
            R0 = float(np.asarray(plasma.inp.rbc)[int(plasma.inp.ntor), 0])
            t = np.linspace(0.0, 2.0 * np.pi, int(nloop), endpoint=False)
            self.loop = np.ascontiguousarray(np.column_stack([R0 * np.cos(t), R0 * np.sin(t), 0.0 * t]))
            self.loop_dl = np.ascontiguousarray(np.column_stack([-R0 * np.sin(t), R0 * np.cos(t), 0.0 * t])
                                                * (2.0 * np.pi / int(nloop)))
            I_coil0 = self._linked_current()
            I_eq0 = 2.0 * np.pi * float(np.asarray(plasma.outputs()["rbtor"])) / MU0
            # the sign relating I_coil and G depends on the loop / angle orientation: fix it at the seed
            self.current_sign = float(np.sign(I_coil0 * I_eq0)) or 1.0
            self.current_norm = abs(I_eq0)
            self.current_seed = (I_coil0, I_eq0)
        self._cache = None
        EqualityConstraint.__init__(self, x0=np.asarray([]), depends_on=[plasma, field])

    def _linked_current(self):
        """``oint B_coil . dl / mu0`` around the fixed circle, A."""
        self.field.set_points(self.loop)
        return float(np.sum(self.field.B() * self.loop_dl) / MU0)

    def _coil_flux(self, out):
        xs, ts = _to_points(out["xsec"]), _to_points(out["xsec_tangent"])
        self.field.set_points(xs)
        A, dA = self.field.A().copy(), self.field.dA_by_dX().copy()
        wq = 2.0 * np.pi / xs.shape[0]
        return float(wq * np.sum(A * ts)), xs, ts, A, dA, wq

    def recompute_bell(self, parent=None):
        self._cache = None

    def _compute(self):
        if self._cache is not None:
            return self._cache
        cache_p = self.plasma.evaluate()
        out = cache_p["out"]
        nphi, ntheta = self.plasma.nphi, self.plasma.ntheta
        gamma = _to_points(out["gamma"])
        normal = _to_points(out["normal"])
        B_plasma = _to_points(out["B_plasma"])
        points = gamma.reshape(-1, 3)
        self.field.set_points(points)
        B = self.field.B().reshape(nphi, ntheta, 3).copy()
        dB = self.field.dB_by_dX().reshape(nphi, ntheta, 3, 3).copy()
        Bn_out = out["Bn_plasma"] + np.sum(B * normal, axis=-1)
        B_out = B_plasma + B
        jump = np.sum(B_out ** 2, axis=-1) - out["Bin_mag2"] - 2.0 * MU0 * self.p_edge
        w = out["weights"]
        f = Bn_out / self.B_ref
        if self.mode == "fourier":
            bnormal = self.projection @ f.ravel()
        else:
            bnormal = (np.sqrt(w) * f).ravel()
        residuals = {"bnormal": bnormal}
        cache = dict(accepted=cache_p["accepted"], points=points, B=B, dB=dB, normal=normal,
                     B_out=B_out, jump=jump, f=f, w=w, residuals=residuals)
        if self.field_strength == "pressure_balance":
            residuals["pressure_balance"] = np.array([np.sum(w * jump) / self.B_ref ** 2])
        elif self.field_strength == "toroidal_flux":
            flux, xs, ts, A, dA, wq = self._coil_flux(out)
            residuals["toroidal_flux"] = np.array([(flux - self.flux_target) / abs(self.flux_target)])
            cache.update(flux=flux, xs=xs, ts=ts, A=A, dA=dA, wq=wq)
        elif self.field_strength == "net_poloidal_current":
            I_coil = self._linked_current()
            I_eq = 2.0 * np.pi * float(np.asarray(out["rbtor"])) / MU0
            residuals["net_poloidal_current"] = np.array([(I_coil - self.current_sign * I_eq) / self.current_norm])
            cache.update(I_coil=I_coil, I_eq=I_eq)
        self._cache = cache
        return self._cache

    def residuals(self):
        return self._compute()["residuals"]

    def rms_bnormal_over_B(self):
        """Area-weighted RMS of ``(B_plasma + B_coil).n / |B_in|`` (a diagnostic)."""
        c = self._compute()
        Bin = np.sqrt(self.plasma.outputs()["Bin_mag2"])
        return float(np.sqrt(np.sum(c["w"] * (c["f"] * self.B_ref / Bin) ** 2)))

    def residuals_vjp_parts(self, cotangents):
        c = self._compute()
        if not c["accepted"]:
            return Derivative({}), {}
        nphi, ntheta = self.plasma.nphi, self.plasma.ntheta
        w = c["w"]
        d_weights = np.zeros((nphi, ntheta))
        g_f = np.zeros((nphi, ntheta))
        if "bnormal" in cotangents:
            v = np.asarray(cotangents["bnormal"], dtype=float)
            if self.mode == "fourier":
                g_f = (self.projection.T @ v).reshape(nphi, ntheta)
            else:
                v = v.reshape(nphi, ntheta)
                g_f = np.sqrt(w) * v
                d_weights += v * c["f"] * 0.5 / np.sqrt(w)
        g_Bn = g_f / self.B_ref                                   # d/d (B_out . n)
        h = np.zeros((nphi, ntheta))                              # d/d |B_out|^2
        if "pressure_balance" in cotangents:
            v = float(np.asarray(cotangents["pressure_balance"]).ravel()[0])
            h = v * w / self.B_ref ** 2
            d_weights += v * c["jump"] / self.B_ref ** 2
        v_B = g_Bn[..., None] * c["normal"] + 2.0 * h[..., None] * c["B_out"]   # d/d B_coil
        self.field.set_points(c["points"])
        d_coils = self.field.B_vjp(np.ascontiguousarray(v_B.reshape(-1, 3)))
        # B_coil(gamma): simsopt dB_by_dX[..., j, k] = d B_k / d x_j
        ct_gamma = np.einsum("pqjk,pqk->pqj", c["dB"], v_B)
        plasma_ct = dict(
            Bn_plasma=g_Bn,
            normal=_to_jax_layout(g_Bn[..., None] * c["B"]),
            gamma=_to_jax_layout(ct_gamma),
            B_plasma=_to_jax_layout(2.0 * h[..., None] * c["B_out"]),
            Bin_mag2=-h,
            weights=d_weights,
        )
        if "toroidal_flux" in cotangents:
            v = float(np.asarray(cotangents["toroidal_flux"]).ravel()[0]) * c["wq"] / abs(self.flux_target)
            self.field.set_points(c["xs"])
            d_coils += self.field.A_vjp(np.ascontiguousarray(v * c["ts"]))
            # simsopt dA_by_dX[..., j, k] = d A_k / d x_j
            plasma_ct["xsec"] = _to_jax_layout(v * np.einsum("pjk,pk->pj", c["dA"], c["ts"]))
            plasma_ct["xsec_tangent"] = _to_jax_layout(v * c["A"])
        if "net_poloidal_current" in cotangents:
            v = float(np.asarray(cotangents["net_poloidal_current"]).ravel()[0]) / self.current_norm
            self.field.set_points(self.loop)
            d_coils += self.field.B_vjp(np.ascontiguousarray(v * self.loop_dl / MU0))
            plasma_ct["rbtor"] = np.asarray(-v * self.current_sign * 2.0 * np.pi / MU0)
        return d_coils, {self.plasma: plasma_ct}


class CoilPlasmaDistance(Optimizable):
    """``J = sum_i mean_{q,p} max(d_min - |x_iq - y_p|, 0)^2`` over coils ``i``.

    ``y_p`` are the VMEX boundary points of one field period rotated to all
    ``nfp`` periods; ``x_iq`` the quadrature points of ``curves``.
    """

    def __init__(self, plasma, curves, minimum_distance):
        self.plasma, self.curves = plasma, list(curves)
        self.minimum_distance = float(minimum_distance)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[plasma] + self.curves)

    def _rotations(self):
        nfp = self.plasma.nfp
        mats = []
        for k in range(nfp):
            a = 2 * np.pi * k / nfp
            mats.append(np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]]))
        return mats

    def _terms(self):
        cache = self.plasma.evaluate()
        if not cache["accepted"]:
            return None
        gamma = _to_points(cache["out"]["gamma"]).reshape(-1, 3)
        rots = self._rotations()
        Y = np.concatenate([gamma @ R.T for R in rots])
        return gamma, rots, Y

    def J(self):
        terms = self._terms()
        if terms is None:
            return 0.0
        _, _, Y = terms
        J = 0.0
        for curve in self.curves:
            X = curve.gamma()
            D = np.linalg.norm(X[:, None, :] - Y[None, :, :], axis=-1)
            J += np.mean(np.maximum(self.minimum_distance - D, 0.0) ** 2)
        return float(J)

    def shortest_distance(self):
        """Closest approach of any coil to the full-torus boundary (inf if the equilibrium is rejected)."""
        terms = self._terms()
        if terms is None:
            return np.inf
        _, _, Y = terms
        return float(min(np.min(np.linalg.norm(curve.gamma()[:, None, :] - Y[None, :, :], axis=-1))
                         for curve in self.curves))

    def dJ_parts(self):
        terms = self._terms()
        if terms is None:
            return Derivative({}), {}
        gamma, rots, Y = terms
        derivative = Derivative({})
        gY = np.zeros_like(Y)
        for curve in self.curves:
            X = curve.gamma()
            diff = X[:, None, :] - Y[None, :, :]
            D = np.linalg.norm(diff, axis=-1)
            coef = -2.0 * np.maximum(self.minimum_distance - D, 0.0) / (D.size * np.maximum(D, 1e-300))
            gX = np.einsum("qp,qpj->qj", coef, diff)
            gY -= np.einsum("qp,qpj->pj", coef, diff)
            derivative += curve.dgamma_by_dcoeff_vjp(gX)
        npts = gamma.shape[0]
        g_gamma = sum(gY[k * npts:(k + 1) * npts] @ R for k, R in enumerate(rots))
        g_gamma = g_gamma.reshape(self.plasma.nphi, self.plasma.ntheta, 3)
        return derivative, {self.plasma: {"gamma": _to_jax_layout(g_gamma)}}

    @derivative_dec
    def dJ(self):
        derivative, parts = self.dJ_parts()
        for owner, ct in parts.items():
            derivative += owner.pullback(ct)
        return derivative


class WeightedSum(Optimizable):
    """``sum_k w_k J_k`` that merges shared pullbacks (one VMEX adjoint in total).

    ``terms`` is a list of ``(weight, optimizable)``; a weight may be a float or a
    ``simsopt.objectives.Weight`` (read at evaluation time, so escalation works).
    """

    def __init__(self, terms):
        self.terms = list(terms)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[t for _, t in self.terms])

    @staticmethod
    def _w(weight):
        return float(weight.value) if hasattr(weight, "value") else float(weight)

    def J(self):
        return float(sum(self._w(w) * t.J() for w, t in self.terms))

    def dJ_parts(self):
        derivative, pending = Derivative({}), {}
        for weight, term in self.terms:
            w = self._w(weight)
            if w == 0.0:
                continue
            if hasattr(term, "dJ_parts"):
                d, parts = term.dJ_parts()
                derivative += w * d if d.data else d
                merge_cotangents(pending, {owner: {k: w * np.asarray(v, dtype=float) for k, v in ct.items()}
                                           for owner, ct in parts.items()})
            else:
                derivative += w * term.dJ(partials=True)
        return derivative, {owner: ct for owner, ct in pending.values()}

    @derivative_dec
    def dJ(self):
        derivative, parts = self.dJ_parts()
        for owner, ct in parts.items():
            derivative += owner.pullback(ct)
        return derivative


def coil_stellsym_error(coils):
    """Largest distance from the stellarator image ``(x, -y, -z)`` of a coil point to the coil set, m (0: symmetric).

    :class:`PlasmaCoilInterface` constrains only the stellarator-odd B.n harmonics, so it cannot see the even B.n of a
    coil set that lost its symmetry -- log this next to the B.n residuals.
    """
    g = np.array([c.curve.gamma() for c in coils])
    points = g.reshape(-1, 3)
    image = points * np.array([1.0, -1.0, -1.0])
    worst = 0.0
    for chunk in np.array_split(np.arange(image.shape[0]), max(1, image.shape[0] // 512)):
        d = np.linalg.norm(image[chunk][:, None, :] - points[None, :, :], axis=-1)
        worst = max(worst, float(np.max(np.min(d, axis=1))))
    return worst


# CurveXYZFourier coefficients that are odd under the stellarator symmetry (x, y, z)(t) -> (x, -y, -z)(-t): a curve that
# is its own image, with t = 0 on the symmetry line, has x even and y, z odd in t, i.e. these coefficients are all zero
_STELLSYM_ODD_PREFIXES = ("xs(", "yc(", "zc(")


def fix_self_symmetric_coil_parity(base_curves, tol=1e-9):
    """Fix the symmetry-breaking coefficients of base curves that are their own stellarator image.

    A coil lying across a symmetry plane (designA's two coils at phi = +-90 deg) is its own stellarator image. Its free
    ``CurveXYZFourier`` stays symmetric only while the coefficients named in ``_STELLSYM_ODD_PREFIXES`` stay zero, and
    nothing else holds them there: with them free, the B.n constraints (odd harmonics only) let the optimizer break the
    symmetry at no cost -- the final coils of designA_L2_beta1 / _vc_beta1 / _vc_free_beta1 are 18 / 4.3 / 9.3 mm off
    symmetry, and the example run 2149 grew an even B.n of 1.9e-2 while its odd part fell to 9e-4.

    Only curves whose own points map onto themselves under ``(x, -y, -z)`` (to ``tol`` m) are touched, so circular
    coils from ``create_equally_spaced_curves`` (not on a symmetry plane, many coefficients zero but free) are left
    alone. Returns ``(n_fixed, notes)``: the number of coefficients fixed and a list of human-readable notes.
    """
    n_fixed, notes = 0, []
    for curve in base_curves:
        if type(curve).__name__ != "CurveXYZFourier":
            continue
        g = curve.gamma()
        image = g * np.array([1.0, -1.0, -1.0])
        err = float(np.max(np.min(np.linalg.norm(image[:, None, :] - g[None, :, :], axis=-1), axis=1)))
        if err > tol:
            continue                                      # not its own image: its symmetry partner is another coil
        odd = [(name, value) for name, value in zip(curve.local_full_dof_names, curve.local_full_x)
               if name.startswith(_STELLSYM_ODD_PREFIXES)]
        scale = float(np.max(np.abs(curve.local_full_x)))
        if any(abs(value) > 1e-12 * scale for _, value in odd):
            notes.append(f"{curve.name}: its own stellarator image, but not parametrized with t = 0 on the symmetry "
                         "line (odd-parity coefficients nonzero); left free -- watch the coil symmetry error")
            continue
        newly = [name for name, _ in odd if curve.is_free(name)]
        for name in newly:
            curve.fix(name)
        n_fixed += len(newly)
        notes.append(f"{curve.name}: its own stellarator image; fixed {len(newly)} odd-parity coefficients")
    return n_fixed, notes


def _polyline_self_intersects(R, Z):
    """True if the closed polyline (R[i], Z[i]) crosses itself (non-adjacent segments intersect)."""
    P = np.column_stack([R, Z])
    Q = np.roll(P, -1, axis=0)
    d = Q - P                                                   # segment directions
    n = P.shape[0]
    i, j = np.triu_indices(n, k=2)
    keep = ~((i == 0) & (j == n - 1))                           # first and last segments share a point
    i, j = i[keep], j[keep]
    den = d[i, 0] * d[j, 1] - d[i, 1] * d[j, 0]
    ok = np.abs(den) > 1e-300
    i, j, den = i[ok], j[ok], den[ok]
    w = P[j] - P[i]
    t = (w[:, 0] * d[j, 1] - w[:, 1] * d[j, 0]) / den
    u = (w[:, 0] * d[i, 1] - w[:, 1] * d[i, 0]) / den
    return bool(np.any((t > 0.0) & (t < 1.0) & (u > 0.0) & (u < 1.0)))


def boundary_regularity(plasma, nphi=8, ntheta=512, check_self_intersection=True):
    """Quality of the VMEX boundary's poloidal parametrization at the plasma's current dofs (no equilibrium solve).

    Returns ``(min_speed_ratio, (phi/2pi, theta/2pi), self_intersecting)``: the smallest ``|dx/dtheta|`` relative to the
    median on its cross-section, over ``nphi`` cross-sections of one field period (the half period for a stellarator-
    symmetric boundary; exact Fourier derivatives), where it occurs, and whether any of those cross-sections crosses
    itself. A collapsing speed or a self-intersection makes VMEC's Jacobian degenerate: trial E's boundary carried a
    0.2 mm loop at the phi = 0 outboard tip (speed ratio 7e-3, sqrt_g 500x below its median), which broke virtual casing,
    pressure balance and cold solves (combined_stage/README.md).
    """
    x = np.asarray(plasma.local_full_x, dtype=float)
    rbc, zbs = (np.asarray(a, dtype=float) for a in plasma.problem.boundary_from_x(x)[:2])
    m = np.arange(rbc.shape[1], dtype=float)[None, None, :]
    n = (np.arange(rbc.shape[0], dtype=float) - (rbc.shape[0] - 1) // 2)[None, :, None]
    theta = np.linspace(0.0, 2.0 * np.pi, int(ntheta), endpoint=False)[:, None, None]
    span = (0.5 if plasma.stellsym else 1.0) * 2.0 * np.pi / plasma.nfp
    worst, where, crossing = np.inf, (0.0, 0.0), False
    for phi in np.linspace(0.0, span, int(nphi), endpoint=False):
        ang = m * theta - n * plasma.nfp * phi
        c, s = np.cos(ang), np.sin(ang)
        R, Z = np.einsum("tnm,nm->t", c, rbc), np.einsum("tnm,nm->t", s, zbs)
        Rt = np.einsum("tnm,nm->t", -s * m, rbc)
        Zt = np.einsum("tnm,nm->t", c * m, zbs)
        speed = np.hypot(Rt, Zt)
        ratio = speed / np.median(speed)
        k = int(np.argmin(ratio))
        if ratio[k] < worst:
            worst, where = float(ratio[k]), (float(phi / (2.0 * np.pi)), float(k / ntheta))
        if check_self_intersection and not crossing:
            crossing = _polyline_self_intersects(R, Z)
    return worst, where, crossing


class BoundaryRegularityConstraint(InequalityConstraint):
    """``|dx/dtheta| / mean_theta(|dx/dtheta|) >= min_speed_ratio`` on a grid of the boundary's symmetry period.

    The smooth counterpart of ``combined_stage_vmex.py``'s boundary guard. The guard REJECTS a trial whose poloidal
    parametrization is collapsing (see :func:`boundary_regularity`): correct, but invisible to the optimizer, so once
    the boundary reaches the limit every step is rejected and the run stalls with 0 inner iterations (runs 816, 817,
    823). As an augmented-Lagrangian inequality block the same requirement has a gradient: the optimizer can slide
    along the limit, and the multiplier learns the force the B.n and X-line blocks are pushing the tip with.

    The ratio is taken against each cross-section's MEAN speed (its perimeter / 2 pi), which is smooth, where the guard
    uses the median. The residual depends only on the prescribed boundary Fourier coefficients, so a value and its vjp
    are one jitted Fourier sum -- no equilibrium solve and no adjoint. Keep the hard guard enabled at a lower floor as a
    safety net: this block discourages a fold, it cannot forbid one during a line search.
    """

    def __init__(self, plasma, min_speed_ratio=0.15, nphi=8, ntheta=256):
        self.plasma = plasma
        self.min_speed_ratio = float(min_speed_ratio)
        self.nphi, self.ntheta = int(nphi), int(ntheta)
        nfp, r_min = int(plasma.nfp), self.min_speed_ratio
        span = (0.5 if plasma.stellsym else 1.0) * 2.0 * jnp.pi / nfp
        phi = jnp.linspace(0.0, span, self.nphi, endpoint=False)[:, None, None, None]
        theta = jnp.linspace(0.0, 2.0 * jnp.pi, self.ntheta, endpoint=False)[None, :, None, None]
        boundary_from_x = plasma.problem.boundary_from_x

        def ratio_minus_floor(x):
            rbc, zbs = boundary_from_x(x)[:2]                     # [n + ntor, m]
            m = jnp.arange(rbc.shape[1], dtype=float)[None, None, None, :]
            n = (jnp.arange(rbc.shape[0], dtype=float) - (rbc.shape[0] - 1) // 2)[None, None, :, None]
            ang = m * theta - n * nfp * phi
            cos_a, sin_a = jnp.cos(ang), jnp.sin(ang)
            Rt = jnp.einsum("ptnm,nm->pt", -sin_a * m, rbc)       # exact dR/dtheta
            Zt = jnp.einsum("ptnm,nm->pt", cos_a * m, zbs)
            speed = jnp.hypot(Rt, Zt)
            return (speed / jnp.mean(speed, axis=1, keepdims=True) - r_min).ravel()

        self._value = jax.jit(ratio_minus_floor)
        self._pullback = jax.jit(lambda x, v: jax.vjp(ratio_minus_floor, x)[1](v)[0])
        InequalityConstraint.__init__(self, x0=np.asarray([]), depends_on=[plasma])

    def residuals(self):
        return {"boundary_regularity": np.asarray(self._value(jnp.asarray(self.plasma.local_full_x)), dtype=float)}

    def residuals_vjp_parts(self, cotangents):
        v = jnp.asarray(np.asarray(cotangents["boundary_regularity"], dtype=float))
        g = self._pullback(jnp.asarray(self.plasma.local_full_x), v)
        return Derivative({self.plasma: np.asarray(g, dtype=float)}), {}

    def worst_ratio(self):
        """Smallest speed ratio at the current dofs and where it sits, ``(ratio, (phi/2pi, theta/2pi))`` -- logging."""
        r = self.residuals()["boundary_regularity"]
        k = int(np.argmin(r))
        i, j = divmod(k, self.ntheta)
        span = (0.5 if self.plasma.stellsym else 1.0) / self.plasma.nfp
        return float(r[k] + self.min_speed_ratio), (i * span / self.nphi, j / self.ntheta)
