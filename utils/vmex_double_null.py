"""Double-null (X-point) terms for the VMEX combined stage.

The double-null divertor of the Star_Lite designs is carried by a pair of
hyperbolic periodic field lines (the upper X-point line and its
stellarator-symmetric image) in the COIL field.  In vacuum they are features of
the coil field only, so their derivatives are simsopt/star_lite_design adjoints
through the periodic field-line Newton solve (``utils/periodicfieldline.py``).

:class:`XpointHyperbolicity`
    ``max(2 + margin - |tr M|, 0)^2`` with ``M`` the one-field-period return map
    (monodromy) of the X-point line: keeps it a genuine X-point
    (``|tr M| > 2``) with a margin.
:class:`XpointPlasmaDistance`
    one-sided quadratic penalties keeping every point of the X-point line within
    ``[d_min, d_max]`` of the moving VMEX plasma boundary (nearest distance to
    ``VmexPlasma``'s full-torus ``boundary_points``).  An X-point cannot lie inside
    the nested-surface region, so an unsigned distance suffices.

Both return their plasma dependence as cotangents (``dJ_parts``), so they share
the single VMEX adjoint of :class:`~star_lite_design.utils.vmex_combined_stage.WeightedSum`.

Free X-line (a double null that does not exist yet)
---------------------------------------------------
Tracking needs a solved X-point.  A coil set without one (e.g. grown from circular
coils) gets a :class:`FreeXLine` instead: a one-period closed curve plus its length,
both design variables.  :class:`XLineFieldLineConstraint` makes it a field line as an
augmented-Lagrangian equality (the ``periodicfieldline.py`` residual
``gamma'/L - B/|B|`` plus the ``y(0) = 0`` label; as many equations as curve + length
dofs), :class:`FreeXLineHyperbolicity` evaluates the same margin penalty directly on
the curve, and :class:`XpointPlasmaDistance` accepts either kind of line.  At
feasibility the free line is the X-point line the tracked terms would see.
"""
import numpy as np
import jax.numpy as jnp
from jax import grad, jit

from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.geo import CurveLength, CurveXYZFourierSymmetries
from simsopt.objectives import forward_backward

from .augmented_lagrangian import EqualityConstraint
from .periodicfieldline import field_line_residual
from .tangent_map import TangentMap, cheb, monodromy_pure

__all__ = ["XpointHyperbolicity", "XpointPlasmaDistance", "fieldline_gamma_vjp", "ensure_fieldline_solved",
           "fieldline_snapshot", "fieldline_restore", "FreeXLine", "XLineFieldLineConstraint",
           "FreeXLineHyperbolicity", "free_xline_from_plasma", "fit_xline_to_field"]


def ensure_fieldline_solved(fieldline):
    """Re-solve a :class:`PeriodicFieldLine` if its coils moved; return success.

    A Newton solve that breaks down numerically (|B| = 0 on the trial curve, a
    singular or non-finite Jacobian) is a failed solve, not a crash.  The Newton
    loop overwrites the curve dofs in place, so after a failure restore them with
    :func:`fieldline_restore` before the next evaluation.
    """
    if fieldline.need_to_run_code:
        try:
            with np.errstate(divide="ignore", invalid="ignore"):
                fieldline.run_code(fieldline.res["length"])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError):
            fieldline.res = dict(fieldline.res, success=False)
            return False
    return bool(fieldline.res["success"]) and bool(np.all(np.isfinite(fieldline.curve.gamma())))


def fieldline_snapshot(fieldline):
    """State of a solved field line (curve dofs + solve record) to return to after a failed trial."""
    return dict(dofs=np.asarray(fieldline.curve.x).copy(), res=dict(fieldline.res))


def fieldline_restore(fieldline, snapshot):
    """Put back a :func:`fieldline_snapshot`; the next evaluation re-solves from it."""
    fieldline.curve.x = snapshot["dofs"]
    fieldline.res = dict(snapshot["res"])
    fieldline.need_to_run_code = True


def fieldline_gamma_vjp(fieldline, dJ_dgamma):
    """Coil ``Derivative`` of ``<dJ_dgamma, gamma>`` through the periodic field-line solve.

    Same adjoint as ``FieldLineMeanZ.dJ`` (``utils/displacement.py``): the curve dofs
    are slaved to the coils by the Newton solve, so the explicit curve gradient is
    pulled back with the stored LU factors of the residual Jacobian.
    """
    curve = fieldline.curve
    res_curve = curve.dgamma_by_dcoeff_vjp(np.ascontiguousarray(dJ_dgamma))(curve)
    P, L, U = fieldline.res["PLU"]
    rhs = np.zeros(L.shape[0])
    rhs[:res_curve.size] = res_curve
    adj = forward_backward(P, L, U, rhs)
    return -1.0 * fieldline.res["vjp"](adj, fieldline.biotsavart, fieldline)


def _line_solved(line):
    """A free X-line has no solve; a tracked field line must re-solve on the current coils."""
    return True if isinstance(line, FreeXLine) else ensure_fieldline_solved(line)


def _line_gamma_vjp(line, dJ_dgamma):
    """``Derivative`` of ``<dJ_dgamma, gamma>``: direct for a free line, through the solve for a tracked one."""
    return line.gamma_vjp(dJ_dgamma) if isinstance(line, FreeXLine) else fieldline_gamma_vjp(line, dJ_dgamma)


def hyperbolicity_margin_pure(B, gradB, L, gamma, D, margin, frame="NB"):
    Rf = monodromy_pure(B, gradB, L, gamma, D, frame)[-1]
    trace = Rf[0, 0] + Rf[1, 1]
    return jnp.maximum(2.0 + margin - jnp.abs(trace), 0.0) ** 2


class XpointHyperbolicity(TangentMap):
    """``J = max(2 + margin - |tr M|, 0)^2`` for the X-point line's return map.

    Reuses the :class:`TangentMap` machinery (collocated tangent-map solve, field
    and field-gradient VJPs, field-line adjoint) with the monodromy kernel replaced.
    """

    def __init__(self, xpoint, biotsavart, margin=0.2, frame="NB"):
        super().__init__(xpoint, biotsavart, threshold=0.0, mtype="jordan", frame=frame)
        self.margin = float(margin)
        self.monodromy_jax = lambda B, gradB, L, gamma: hyperbolicity_margin_pure(
            B, gradB, L, gamma, self.D, self.margin, self.frame)
        self.recompute_bell()

    def trace(self):
        M = np.asarray(self.matrix)
        return float(M[0, 0] + M[1, 1])

    def J(self):
        return float(self.monodromy)

    @derivative_dec
    def dJ(self):
        return self.dmonodromy_dcoils


class XpointPlasmaDistance(Optimizable):
    """Keep the X-point field line within ``[d_min, d_max]`` of the plasma boundary.

    ``J = mean_q [ max(d_min - d_q, 0)^2 + max(d_q - d_max, 0)^2 ]`` with ``d_q`` the
    distance from X-point quadrature point ``q`` to the nearest boundary point.
    ``plasma`` must provide ``evaluate()`` (dict with ``accepted`` and ``out``),
    a full-torus ``boundary_points`` output ``(3, nphi, ntheta)`` and ``pullback``.
    """

    def __init__(self, xpoint, plasma, d_min=0.0, d_max=np.inf, signed=False):
        self.xpoint, self.plasma = xpoint, plasma
        self.d_min, self.d_max = float(d_min), float(d_max)
        # A solved X-point cannot sit inside the nested surfaces, but a free X-line can (a field-line fit on
        # frozen coils slid onto the magnetic axis, 10 cm inside): signed distances make "inside" violate d_min.
        self.signed = bool(signed)
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[xpoint, plasma])

    def _geometry(self):
        if not _line_solved(self.xpoint):
            return None
        cache = self.plasma.evaluate()
        if not cache["accepted"]:
            return None
        pts = np.moveaxis(np.asarray(cache["out"]["boundary_points"], dtype=float), 0, -1)
        shape = pts.shape
        Y = pts.reshape(-1, 3)
        X = self.xpoint.curve.gamma()
        diff = X[:, None, :] - Y[None, :, :]
        dist2 = np.einsum("qpj,qpj->qp", diff, diff)
        idx = np.argmin(dist2, axis=1)
        q = np.arange(X.shape[0])
        d = np.sqrt(dist2[q, idx])
        unit = diff[q, idx] / np.maximum(d, 1e-300)[:, None]
        if self.signed:
            # outside <=> the offset points away from the centroid of the nearest point's cross-section
            # (rows of boundary_points are planes of constant phi); the sign is locally constant
            centre = pts.mean(axis=1)
            outward = Y[idx] - centre[idx // shape[1]]
            sign = np.where(np.einsum("qj,qj->q", diff[q, idx], outward) < 0.0, -1.0, 1.0)
            d, unit = sign * d, sign[:, None] * unit
        return shape, idx, d, unit

    def distances(self):
        geo = self._geometry()
        return None if geo is None else geo[2]

    def J(self):
        geo = self._geometry()
        if geo is None:
            return 0.0
        d = geo[2]
        return float(np.mean(np.maximum(self.d_min - d, 0.0) ** 2 + np.maximum(d - self.d_max, 0.0) ** 2))

    def dJ_parts(self):
        geo = self._geometry()
        if geo is None:
            return Derivative({}), {}
        shape, idx, d, unit = geo
        g = (-2.0 * np.maximum(self.d_min - d, 0.0) + 2.0 * np.maximum(d - self.d_max, 0.0)) / d.size
        dJ_dX = g[:, None] * unit
        dJ_dY = np.zeros((int(np.prod(shape[:2])), 3))
        np.add.at(dJ_dY, idx, -dJ_dX)
        derivative = _line_gamma_vjp(self.xpoint, dJ_dX)
        return derivative, {self.plasma: {"boundary_points": np.moveaxis(dJ_dY.reshape(shape), -1, 0)}}

    @derivative_dec
    def dJ(self):
        derivative, parts = self.dJ_parts()
        for owner, ct in parts.items():
            derivative += owner.pullback(ct)
        return derivative


class FreeXLine(Optimizable):
    """A candidate X-point line: a one-period ``CurveXYZFourierSymmetries`` (its dofs) plus its length.

    Mirrors the parts of :class:`PeriodicFieldLine` the double-null terms read
    (``curve``, ``res["length"]``), but nothing is solved: the curve and length are
    design variables made consistent by :class:`XLineFieldLineConstraint`.
    """

    def __init__(self, curve, length):
        self.curve = curve
        Optimizable.__init__(self, x0=np.array([float(length)]), names=["length"], depends_on=[curve])

    @property
    def length(self):
        return float(self.local_full_x[0])

    @property
    def res(self):
        return {"length": self.length, "success": True}

    def gamma_vjp(self, dJ_dgamma):
        return self.curve.dgamma_by_dcoeff_vjp(np.ascontiguousarray(dJ_dgamma))


class XLineFieldLineConstraint(EqualityConstraint):
    """``gamma'/L - B/|B| = 0`` at the curve's quadrature points (and ``y(0) = 0`` unless stellsym).

    The residual and its Jacobians are ``periodicfieldline.field_line_residual``; the
    coil part of the VJP is ``field.B_vjp`` of the cotangent contracted with ``dr/dB``.
    Use a dedicated ``BiotSavart`` (``set_points`` moves its evaluation points).
    """

    def __init__(self, xline, field):
        self.xline, self.field = xline, field
        EqualityConstraint.__init__(self, x0=np.asarray([]), depends_on=[xline, field])

    def residuals(self):
        r, _, _ = field_line_residual(self.xline.curve, self.xline.length, self.field)
        return {"fieldline": r}

    def residuals_vjp_parts(self, cotangents):
        lam = np.asarray(cotangents["fieldline"], dtype=float)
        curve = self.xline.curve
        _, dres, dres_dB = field_line_residual(curve, self.xline.length, self.field)
        g = lam @ dres                                         # [curve coefficients..., length]
        derivative = Derivative({curve: g[:-1]}) + Derivative({self.xline: g[-1:]})
        nq = curve.gamma().shape[0]
        ct_B = np.einsum("qk,qkj->qj", lam[:3 * nq].reshape(-1, 3), dres_dB[:3 * nq].reshape(-1, 3, 3))
        derivative += self.field.B_vjp(np.ascontiguousarray(ct_B))
        return derivative, {}


class FreeXLineHyperbolicity(Optimizable):
    """``J = max(2 + margin - |tr M|, 0)^2`` of the one-period return map along a :class:`FreeXLine`.

    Same kernel and collocation as :class:`XpointHyperbolicity`, but ``B``, ``grad B``,
    ``L`` and ``gamma`` are explicit functions of the coils and the free curve, so the
    gradient needs no field-line adjoint: coil part through ``B_and_dB_vjp``, curve part
    through ``grad B`` and ``d2B_by_dXdX``.
    """

    def __init__(self, xline, field, margin=0.2, frame="NB"):
        self.xline, self.field, self.margin, self.frame = xline, field, float(margin), frame
        curve = xline.curve
        self.D, xh, _ = cheb(5 * curve.order + 1, 0.0, 1.0 / curve.nfp)
        self.cheb_curve = CurveXYZFourierSymmetries(xh, curve.order, curve.nfp, curve.stellsym,
                                                    ntor=curve.ntor, dofs=curve.dofs)
        self._J = jit(lambda B, gradB, L, gamma: hyperbolicity_margin_pure(B, gradB, L, gamma, self.D,
                                                                           self.margin, self.frame))
        self._grad = jit(grad(self._J, argnums=(0, 1, 2, 3)))
        self._trace = jit(lambda B, gradB, L, gamma: jnp.trace(monodromy_pure(B, gradB, L, gamma, self.D,
                                                                              self.frame)[-1]))
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[xline, field])

    def _inputs(self):
        gamma = self.cheb_curve.gamma()
        self.field.set_points(gamma)
        return self.field.B(), self.field.dB_by_dX(), self.xline.length, gamma

    def trace(self):
        return float(self._trace(*self._inputs()))

    def J(self):
        return float(self._J(*self._inputs()))

    @derivative_dec
    def dJ(self):
        B, gradB, L, gamma = self._inputs()
        gB, ggradB, gL, ggamma = (np.asarray(v, dtype=float) for v in self._grad(B, gradB, L, gamma))
        derivative = sum(self.field.B_and_dB_vjp(np.ascontiguousarray(gB), np.ascontiguousarray(ggradB)))
        # total d/dgamma: explicit + through B (dB_by_dX[i,j,k] = d_j B_k) + through grad B (d2B[i,j,k,l] = d_j d_k B_l)
        dgamma = (ggamma + np.einsum("ik,ijk->ij", gB, gradB)
                  + np.einsum("iab,iacb->ic", ggradB, self.field.d2B_by_dXdX()))
        dcoeff = np.einsum("ij,ijm->m", dgamma, self.cheb_curve.dgamma_by_dcoeff())
        return derivative + Derivative({self.xline.curve: dcoeff}) + Derivative({self.xline: np.array([float(gL)])})


def fit_xline_to_field(xline, field, maxiter=2000):
    """Least-squares field-line fit of a :class:`FreeXLine` with the coils frozen; returns the max residual.

    Minimizes ``|gamma'/L - B/|B||^2`` (+ the label row) over the curve and length only, so the
    augmented Lagrangian starts from the field line closest to the initial guess instead of
    moving the coils to meet an arbitrary curve.
    """
    from scipy.optimize import minimize
    curve = xline.curve

    def fun(x):
        curve.x = x[:-1]
        r, dres, _ = field_line_residual(curve, x[-1], field)
        return 0.5 * float(r @ r), dres.T @ r

    res = minimize(fun, np.concatenate([curve.x, [xline.length]]), jac=True, method="L-BFGS-B",
                   options=dict(maxiter=int(maxiter), maxcor=200))
    curve.x = res.x[:-1]
    xline.local_full_x = res.x[-1:]
    return float(np.max(np.abs(field_line_residual(curve, xline.length, field)[0])))


def free_xline_from_plasma(plasma, order=10, offset=0.06):
    """Initial :class:`FreeXLine`: the top of every boundary cross-section over one field period, raised by ``offset``.

    Its stellarator-symmetric image (the bottom of each cross-section) is the second null.
    The curve is not a field line yet; :class:`XLineFieldLineConstraint` makes it one.
    """
    x, y, z = np.asarray(plasma.evaluate()["out"]["boundary_points"], dtype=float)   # [phi, theta] rows
    rows = np.arange(x.shape[0])
    top = np.argmax(z, axis=1)
    phi = np.mod(np.arctan2(y[rows, top], x[rows, top]), 2 * np.pi)
    R, Z = np.hypot(x[rows, top], y[rows, top]), z[rows, top]
    order_phi = np.argsort(phi)
    nfp = int(plasma.nfp)
    qp = np.linspace(0.0, 1.0 / nfp, 2 * order + 1, endpoint=False)
    phq = 2 * np.pi * qp
    Rq = np.interp(phq, phi[order_phi], R[order_phi], period=2 * np.pi)
    Zq = np.interp(phq, phi[order_phi], Z[order_phi], period=2 * np.pi) + offset
    curve = CurveXYZFourierSymmetries(qp, order, nfp, False, ntor=1)
    curve.least_squares_fit(np.column_stack([Rq * np.cos(phq), Rq * np.sin(phq), Zq]))
    return FreeXLine(curve, float(CurveLength(curve).J()))
