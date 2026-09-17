"""Augmented Lagrangian method (ALM) for simsopt Optimizables.

Solves equality-constrained problems

    min_x f(x)   subject to   c_b(x) = 0   for every constraint block b,

by repeatedly minimizing the augmented Lagrangian

    L(x; lam, rho) = f(x) + sum_b [ -lam_b . c_b(x) + (rho_b / 2) |c_b(x)|^2 ]

and updating the multipliers ``lam_b <- lam_b - rho_b c_b`` between inner solves.
Unlike a pure quadratic penalty (the ``WEIGHT *= 10`` escalation used in
``array/boozer_all.py``), the constraints are met to any tolerance at a FINITE
penalty ``rho``, so the inner problems stay well conditioned: the multipliers
carry the force a diverging weight would otherwise have to supply.

The outer schedule is Algorithm 17.4 of Nocedal & Wright, *Numerical
Optimization* (2nd ed.), applied per constraint block:

* block met to its tolerance ``eta_b``  ->  update ``lam_b``, tighten ``eta_b``;
* block not met                          ->  grow ``rho_b``, reset ``eta_b``;
* the inner gradient tolerance ``omega`` tightens with the largest penalty.

Constraint protocol
-------------------
A constraint is any :class:`EqualityConstraint` (an Optimizable) that implements

* ``residuals() -> dict[str, np.ndarray]``: named 1-D residual blocks, and
* ``residuals_vjp(cotangents: dict[str, np.ndarray]) -> Derivative``.

A constraint (or the objective) whose pullback is expensive and shared -- e.g.
several quantities differentiated through ONE VMEX adjoint -- may instead
implement ``residuals_vjp_parts`` (``dJ_parts`` for the objective), returning
``(Derivative, {owner: cotangent_dict})``.  :class:`AugmentedLagrangian` sums the
cotangents per owner and calls ``owner.pullback(cotangent_dict)`` exactly once
per gradient evaluation.

Use the residual SCALING to express what "small" means: all block tolerances
are infinity norms of the residual vectors as returned.
"""
import numpy as np
from scipy.optimize import minimize

from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec

__all__ = ["EqualityConstraint", "AugmentedLagrangian", "merge_cotangents",
           "solve_augmented_lagrangian"]


def merge_cotangents(pending, parts):
    """Add ``parts = {owner: {name: array}}`` into ``pending`` (keyed by id)."""
    for owner, ct in parts.items():
        slot = pending.setdefault(id(owner), (owner, {}))[1]
        for name, value in ct.items():
            value = np.asarray(value, dtype=float)
            if name in slot:
                slot[name] = slot[name] + value
            else:
                slot[name] = value.copy()
    return pending


class EqualityConstraint(Optimizable):
    """Base class for vector-valued equality constraints ``c(x) = 0``."""

    def residuals(self):
        raise NotImplementedError

    def residuals_vjp(self, cotangents):
        derivative, parts = self.residuals_vjp_parts(cotangents)
        for owner, ct in parts.items():
            derivative += owner.pullback(ct)
        return derivative

    def residuals_vjp_parts(self, cotangents):
        return self.residuals_vjp(cotangents), {}


class InequalityConstraint(EqualityConstraint):
    """Base class for vector-valued inequality constraints ``c(x) >= 0``.

    Powell-Hestenes-Rockafellar form: the augmented term uses the clipped residual ``min(c, lambda/rho)`` wherever the
    equality term uses ``c``, so an entry that is comfortably satisfied contributes no value and no gradient, an entry
    at or past the limit acts like a spring, and the multiplier update ``lambda <- max(0, lambda - rho c)`` keeps the
    multipliers non-negative. ``residuals``/``residuals_vjp_parts`` are as in :class:`EqualityConstraint`.
    """


class AugmentedLagrangian(Optimizable):
    """The augmented Lagrangian of ``objective`` subject to ``constraints``.

    Args:
        objective: scalar Optimizable with ``J``/``dJ`` (or ``None`` for a pure
            feasibility problem).
        constraints: list of :class:`EqualityConstraint`.
        penalty: initial penalty ``rho_0`` for every block (float or dict
            ``{block_name: rho}``; block names are ``"<i>:<key>"`` with ``i``
            the constraint index).
        penalty_growth: factor ``tau`` applied to ``rho_b`` when a block misses
            its tolerance (Nocedal & Wright use 100; 10 is gentler).
        penalty_max: cap on ``rho_b``.
        eta0, omega0: initial feasibility / gradient tolerances. Default to the
            textbook ``1/rho0**0.1`` and ``1/rho0``.
        eta_min, omega_min: floors on the tolerance schedules.
    """

    def __init__(self, objective, constraints, penalty=10.0, penalty_growth=10.0,
                 penalty_max=1e12, eta0=None, omega0=None, eta_min=0.0, omega_min=0.0):
        self.objective = objective
        self.constraints = list(constraints)
        self.penalty0 = penalty
        self.penalty_growth = float(penalty_growth)
        self.penalty_max = float(penalty_max)
        self.eta0 = eta0
        self.eta_min = float(eta_min)
        self.omega_min = float(omega_min)
        rho_ref = penalty if np.isscalar(penalty) else max(dict(penalty).values())
        self.omega = float(omega0) if omega0 is not None else 1.0 / self._rho_eff(rho_ref)
        self.multipliers = {}
        self.penalties = {}
        self.tolerances = {}
        self.last_norms = {}          # previous outer's block norms, for the stall safeguard in update()
        self.history = []
        deps = ([objective] if objective is not None else []) + self.constraints
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=deps)

    # -- block bookkeeping ----------------------------------------------------

    def _rho_eff(self, rho):
        """Penalty used by the tolerance schedules. Nocedal & Wright divide the
        tolerances by rho, which only tightens them for rho > 1; flooring at the
        growth factor keeps the schedule contracting for small initial penalties."""
        return max(float(rho), self.penalty_growth)

    def _initial_penalty(self, name):
        if np.isscalar(self.penalty0):
            return float(self.penalty0)
        return float(dict(self.penalty0)[name])

    def _eta_scale(self, name):
        """Scale of the feasibility tolerance reset after a penalty increase, ``eta = scale / rho^0.1``.

        The textbook reset (scale 1) assumes O(1) constraints. When ``eta0`` is given the reset keeps
        its scale, ``eta0 * rho0^0.1``: with the textbook value a block at rho 20 got eta 0.74, so an
        O(0.1) residual counted as met and its penalty grew only every other outer iteration.
        """
        if self.eta0 is None:
            return 1.0
        return float(self.eta0) * self._rho_eff(self._initial_penalty(name)) ** 0.1

    def _is_inequality(self, i):
        return isinstance(self.constraints[i], InequalityConstraint)

    def _effective(self, name, i, r):
        """Residual entering the augmented term: ``c`` for an equality, ``min(c, lambda/rho)`` for an inequality."""
        if not self._is_inequality(i):
            return r
        return np.minimum(r, self.multipliers[name] / self.penalties[name])

    def blocks(self):
        """List of ``(name, constraint_index, key, residual)`` at the current x."""
        out = []
        for i, constraint in enumerate(self.constraints):
            for key, r in constraint.residuals().items():
                name = f"{i}:{key}"
                r = np.asarray(r, dtype=float).ravel()
                if name not in self.multipliers:
                    rho = self._initial_penalty(name)
                    self.multipliers[name] = np.zeros_like(r)
                    self.penalties[name] = rho
                    self.tolerances[name] = (float(self.eta0) if self.eta0 is not None
                                             else 1.0 / self._rho_eff(rho) ** 0.1)
                if self.multipliers[name].shape != r.shape:
                    raise ValueError(f"constraint block {name} changed size "
                                     f"{self.multipliers[name].shape} -> {r.shape}")
                out.append((name, i, key, r))
        return out

    def constraint_norms(self):
        """Infinity norm of every residual block, ``{name: float}``.

        An inequality block reports ``min(c, lambda/rho)``, which measures violation and complementarity together and
        vanishes exactly at a KKT point (a satisfied inactive entry contributes nothing).
        """
        return {name: (float(np.max(np.abs(self._effective(name, i, r)))) if r.size else 0.0)
                for name, i, _, r in self.blocks()}

    # -- value and gradient ---------------------------------------------------

    def J(self):
        value = float(self.objective.J()) if self.objective is not None else 0.0
        for name, i, _, r in self.blocks():
            lam, rho = self.multipliers[name], self.penalties[name]
            r = self._effective(name, i, r)
            value += -lam @ r + 0.5 * rho * (r @ r)
        return value

    @derivative_dec
    def dJ(self):
        pending = {}
        derivative = Derivative({})
        if self.objective is not None:
            if hasattr(self.objective, "dJ_parts"):
                d_obj, parts = self.objective.dJ_parts()
                derivative += d_obj
                merge_cotangents(pending, parts)
            else:
                derivative += self.objective.dJ(partials=True)
        cotangents = {}
        for name, i, key, r in self.blocks():
            # an inequality uses the clipped residual, so its inactive entries contribute exactly zero here
            cotangents.setdefault(i, {})[key] = (-self.multipliers[name]
                                                 + self.penalties[name] * self._effective(name, i, r))
        for i, cts in cotangents.items():
            d_c, parts = self.constraints[i].residuals_vjp_parts(cts)
            derivative += d_c
            merge_cotangents(pending, parts)
        for owner, ct in pending.values():
            derivative += owner.pullback(ct)
        return derivative

    # -- outer iteration --------------------------------------------------------

    def update(self, grad_norm, ctol=1e-8, gtol=1e-8):
        """One Nocedal & Wright 17.4 outer update after an approximate inner solve.

        Args:
            grad_norm: infinity norm of the gradient of L at the inner solution.
            ctol: final feasibility tolerance (infinity norm, every block).
            gtol: final stationarity tolerance.

        Returns:
            dict with the block norms, the action taken per block
            (``"multiplier"`` or ``"penalty"``), and ``converged``.
        """
        actions, norms = {}, {}
        # Never tighten the schedules below the final tolerances (eta*, omega*):
        # past them the inner solver only chases round-off, and a block that sits
        # at its noise floor would trigger spurious penalty growth.
        eta_floor = max(self.eta_min, float(ctol))
        omega_floor = max(self.omega_min, float(gtol))
        for name, i, _, r in self.blocks():
            nrm = float(np.max(np.abs(self._effective(name, i, r)))) if r.size else 0.0
            norms[name] = nrm
            rho = self.penalties[name]
            # Safeguard against an unreachable feasibility target: a block whose tolerance the inner solve can never
            # meet takes the penalty branch every time, so its multiplier stays at ZERO and the method degenerates into
            # a pure penalty method that stops at the cap (runs 816/817/821/823: lambda = 0, rho = penalty_max = 1e6,
            # B.n stuck at ~1.2e-3 against a 5e-4 gate their coils cannot reach -- measured floor 1.2e-3). Once the
            # penalty is capped and the residual stopped improving, update the multiplier anyway and relax the
            # tolerance to what is actually achievable, so the multipliers can accumulate the force the block needs.
            stalled = (rho >= self.penalty_max and name in self.last_norms
                       and nrm >= 0.99 * self.last_norms[name] and nrm > self.tolerances[name])
            if nrm <= self.tolerances[name] or stalled:
                lam = self.multipliers[name] - rho * r
                self.multipliers[name] = np.maximum(lam, 0.0) if self._is_inequality(i) else lam
                if stalled:
                    self.tolerances[name] = max(nrm, eta_floor)     # aim at the reachable level, not below it
                    actions[name] = "multiplier (stalled at penalty_max, tolerance relaxed)"
                else:
                    self.tolerances[name] = max(self.tolerances[name] / self._rho_eff(rho) ** 0.9, eta_floor)
                    actions[name] = "multiplier"
            else:
                rho = min(rho * self.penalty_growth, self.penalty_max)
                self.penalties[name] = rho
                self.tolerances[name] = max(self._eta_scale(name) / self._rho_eff(rho) ** 0.1, eta_floor)
                actions[name] = "penalty"
        self.last_norms = dict(norms)
        rho_max = self._rho_eff(max(self.penalties.values()) if self.penalties else 1.0)
        if all(a == "multiplier" for a in actions.values()):
            self.omega = max(self.omega / rho_max, omega_floor)
        else:
            self.omega = max(1.0 / rho_max, omega_floor)
        converged = (all(n <= ctol for n in norms.values()) and grad_norm <= gtol)
        record = dict(norms=norms, actions=actions, grad_norm=float(grad_norm),
                      penalties=dict(self.penalties), tolerances=dict(self.tolerances),
                      omega=self.omega, converged=bool(converged))
        self.history.append(record)
        return record

    # -- checkpointing ----------------------------------------------------------

    def state_dict(self):
        return dict(multipliers={k: v.tolist() for k, v in self.multipliers.items()},
                    penalties=dict(self.penalties), tolerances=dict(self.tolerances),
                    last_norms=dict(self.last_norms), omega=self.omega)

    def load_state_dict(self, state):
        self.multipliers = {k: np.asarray(v, dtype=float) for k, v in state["multipliers"].items()}
        self.penalties = {k: float(v) for k, v in state["penalties"].items()}
        self.tolerances = {k: float(v) for k, v in state["tolerances"].items()}
        self.last_norms = {k: float(v) for k, v in (state.get("last_norms") or {}).items()}
        self.omega = float(state["omega"])


def solve_augmented_lagrangian(al, fun=None, outer_maxiter=20, inner_maxiter=500,
                               ctol=1e-8, gtol=1e-8, method="L-BFGS-B", callback=None):
    """Plain ALM driver: inner scipy solve to ``gtol=al.omega``, then :meth:`update`.

    ``fun(x) -> (L, dL)`` defaults to evaluating ``al`` itself; drivers with a
    failure barrier (see ``combined_stage/``) pass their own.
    """
    if fun is None:
        def fun(x):
            al.x = x
            return al.J(), al.dJ()
    x = al.x.copy()
    record = None
    for k in range(outer_maxiter):
        options = {"maxiter": inner_maxiter}
        if method == "L-BFGS-B":
            options.update(gtol=al.omega, ftol=1e-15)
        else:
            options.update(gtol=al.omega)
        res = minimize(fun, x, jac=True, method=method, options=options)
        x = res.x
        _, grad = fun(x)
        record = al.update(float(np.max(np.abs(grad))) if grad.size else 0.0, ctol=ctol, gtol=gtol)
        record.update(outer=k, inner_message=str(res.message), inner_nit=int(res.nit), L=float(res.fun))
        if callback is not None:
            callback(record)
        if record["converged"]:
            break
    return x, record
