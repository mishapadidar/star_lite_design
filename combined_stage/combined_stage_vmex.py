#!/usr/bin/env python3
"""Combined stage-1/2 optimization of a Star_Lite device: VMEX + augmented Lagrangian.

    min_{s, c}  f_QS(s) + coil engineering penalties(c, s)
    subject to  B.n harmonics of (B_plasma(s) + B_coil(c)) = 0   on the VMEX boundary
                <|B_out|^2 - |B_in|^2 - 2 mu0 p_edge> = 0         (field strength)

``s`` = VMEX boundary modes (+ optional current-profile dofs), ``c`` = coil
shapes and currents.  Every gradient is exact: VMEX's implicit adjoint for the
plasma, simsopt's analytic derivatives for the coils; ONE VMEX adjoint per
gradient evaluation.  See README.md for the method and its current limits.

Usage (from this directory, on a compute node):

    ./combined_stage_vmex.py --config config.yaml --vmec-input <seed namelist> --outdir output/<tag>
    ./combined_stage_vmex.py ... --smoke            # tiny end-to-end check
    ./combined_stage_vmex.py ... --resume            # continue from <outdir>/checkpoint.npz
    ./combined_stage_vmex.py ... --coils <coils.json> --xpoint <xpoint.json>   # seed from a previous stage
    ./combined_stage_vmex.py ... --set plasma_objective.aspect_target=7.5 --set vmex.max_mode=3

Double null (``double_null`` config section): the upper X-point periodic field line
(its stellarator-symmetric image is the lower one) must keep solving, stay hyperbolic
(|tr M| >= 2 + margin) and stay within [distance_min, distance_max] of the moving
plasma boundary.
"""
import argparse
import glob
import os
import tempfile
import time
from dataclasses import replace

import numpy as np
import yaml
from scipy.optimize import minimize

from simsopt._core import load, save
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.geo import (CurveCurveDistance, CurveLength, CurveRZFourier, LpCurveCurvature,
                         MeanSquaredCurvature, SurfaceRZFourier, create_equally_spaced_curves)
from star_lite_design.utils.periodicfieldline import PeriodicFieldLine
from star_lite_design.utils.vmex_double_null import (FreeXLine, FreeXLineHyperbolicity, XLineFieldLineConstraint,
                                                     XpointHyperbolicity, XpointPlasmaDistance,
                                                     ensure_fieldline_solved, fieldline_restore,
                                                     fieldline_snapshot, fit_xline_to_field, free_xline_from_plasma)
from simsopt.objectives import QuadraticPenalty, Weight

from star_lite_design.utils.augmented_lagrangian import (AugmentedLagrangian, BandConstraint, EqualityConstraint,
                                                         PenaltyConstraint, SubsetConstraint)
import run_layout as layout   # coils/, xpoints/, inputs/, state/, wout/ subfolders (reads old flat runs too)
from star_lite_design.utils.current_bound import CurrentBound
from star_lite_design.utils.vmex_combined_stage import (
    BoundaryRegularityConstraint, ClosestApproachConstraint, CoilPlasmaDistance, CoilXLineDistance, PlasmaCoilInterface, PlasmaVesselDistance, VmexPlasma,
    VmexQuasisymmetry,
    VmexTermCost, WeightedSum, boundary_regularity, coil_stellsym_error, fix_self_symmetric_coil_parity,
    BootstrapConsistency, bootstrap_mismatch_output, kinetic_pressure_coeffs, kinetic_profiles,
    interface_check, net_poloidal_current_output, _to_points)
from simsopt.geo import ArclengthVariation, CurveXYZFourierSymmetries
from star_lite_design.utils.helicalvessel import HelicalVesselSDF
from star_lite_design.utils.helicalcylindervessel import HelicalCylinderVesselSDF
from star_lite_design.utils.pillpipevessel import TorusSDF, VesselDistance
from simsopt._core import Optimizable
from simsopt._core.derivative import derivative_dec

# plasma_objective sections that become VMEX terms with their own escalated weight: (term/function name, config key)
ESCALATED_PLASMA_TERMS = (("magnetic_well_floor", "magnetic_well"), ("axis_ripple", "axis_ripple"), ("mercier", "mercier"))


class XLineFlatness(Optimizable):
    """array/boozer_all.py's FieldLineMeanZ on a :class:`FreeXLine`: ``mean(max(|z - mean z| - threshold, 0)^2)``."""

    def __init__(self, xline, threshold):
        import jax
        from star_lite_design.utils.displacement import mean_z_pure
        self.xline = xline
        self.f = jax.jit(lambda g: mean_z_pure(g, threshold))
        self.g = jax.jit(jax.grad(self.f))
        Optimizable.__init__(self, x0=np.asarray([]), depends_on=[xline])

    def max_deviation(self):
        z = self.xline.curve.gamma()[:, 2]
        return float(np.max(np.abs(z - z.mean())))

    def J(self):
        return float(self.f(self.xline.curve.gamma()))

    @derivative_dec
    def dJ(self):
        return self.xline.gamma_vjp(np.asarray(self.g(self.xline.curve.gamma())))



class XLineFlatnessConstraint(EqualityConstraint):
    """X-line flatness as augmented-Lagrangian rows, one per X-line point: ``max(|z_i - mean z| - threshold, 0) / threshold``.

    The escalated ``XLineFlatness`` (mean of the squared excess, the same averaging that diluted the clearances) left
    HSX-class X-lines 24-50 mm from flat in vessel scan 2 (user requirement: <= 10 mm, ALL one-stage runs); here the
    inf-norm tolerance bounds the worst point. Upper null only; the lower one is its stellarator image.
    """

    def __init__(self, xline, threshold):
        self.xline, self.threshold = xline, float(threshold)
        EqualityConstraint.__init__(self, x0=np.asarray([]), depends_on=[xline])

    def residuals(self):
        z = self.xline.curve.gamma()[:, 2]
        return {"hard_xline_flatness": np.maximum(np.abs(z - z.mean()) - self.threshold, 0.0) / self.threshold}

    def residuals_vjp_parts(self, cotangents):
        v = np.asarray(cotangents["hard_xline_flatness"], dtype=float).ravel()
        gamma = self.xline.curve.gamma()
        dev = gamma[:, 2] - gamma[:, 2].mean()
        w = np.where(np.abs(dev) > self.threshold, v * np.sign(dev), 0.0) / self.threshold   # d row_i / d z_i
        G = np.zeros_like(gamma)
        G[:, 2] = w - w.mean()                         # through z_i - mean z
        return self.xline.gamma_vjp(G), {}

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"))
    p.add_argument("--vmec-input", default=None, help="seed fixed-boundary VMEC namelist")
    p.add_argument("--design", default=None, help="simsopt archive with the seed coils")
    p.add_argument("--config-id", type=int, default=None)
    p.add_argument("--outdir", default="output/combined_stage")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--coils", default=None, help="seed coils (simsopt json list of Coil), e.g. a previous stage's coils_final.json")
    p.add_argument("--xpoint", default=None, help="seed X-point curve json (a previous stage's xpoint_final.json)")
    p.add_argument("--plasma-target", default=None,
                   help="namelist whose boundary the plasma walks to (warm-started) before the optimization starts: "
                        "for starting points that do not solve cold, e.g. an optimizer checkpoint")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override a config entry (YAML-parsed value); repeatable")
    p.add_argument("--smoke", action="store_true", help="max_mode 1, 2 outer x 3 inner iterations")
    return p.parse_args()


def resolve(path, base):
    return path if (path is None or os.path.isabs(path)) else os.path.normpath(os.path.join(base, path))


def build_plasma(cfg, seed_path):
    import vmex as vj
    from vmex import optimize as opt
    import jax.numpy as jnp

    vcfg, pcfg = cfg["vmex"], cfg["plasma_objective"]
    inp = vj.VmecInput.from_file(seed_path)
    if vcfg.get("mpol") or vcfg.get("ntor"):
        # reduced equilibrium resolution for speed (boundary harmonics above it are dropped);
        # polish at full resolution afterwards
        mp = int(vcfg.get("mpol") or inp.mpol)
        nt = int(vcfg.get("ntor") or inp.ntor)
        inp = inp.change_resolution(mpol=mp, ntor=nt, ntheta=2 * mp + 6, nzeta=2 * nt + 4)
        print(f"equilibrium resolution reduced to mpol={mp}, ntor={nt}")
    inp = replace(inp, ns_array=np.array([vcfg["ns"]]), ftol_array=np.array([vcfg["ftol"]]),
                  niter_array=np.array([vcfg["niter"]]))
    bcfg = cfg.get("bootstrap") or {}
    extra_outputs = {}
    if cfg["consistency"].get("field_strength") == "net_poloidal_current":
        extra_outputs["rbtor"] = net_poloidal_current_output()
    if bcfg.get("enabled"):
        prof = bcfg["profiles"]
        profiles = kinetic_profiles(prof["ne"], prof["Te"], prof["Ti"], prof.get("Zeff", [1.0]))
        if bcfg.get("pressure_from_profiles", True):
            # the Redl current is computed from the kinetic profiles, the equilibrium from AM/PRES_SCALE: make them the
            # same plasma (n_i = n_e), otherwise the constraint pairs a current with a pressure it was not driven by
            am = np.zeros(max(21, len(np.atleast_1d(inp.am))))
            coeffs = kinetic_pressure_coeffs(prof["ne"], prof["Te"], prof["Ti"])
            am[:coeffs.size] = coeffs
            print(f"bootstrap: equilibrium pressure replaced by e ne (Te + Ti): p(0) = {coeffs[0]:.4g} Pa "
                  f"(seed {float(inp.pres_scale) * float(np.atleast_1d(inp.am)[0]):.4g} Pa)")
            inp = replace(inp, am=am, pres_scale=1.0, pmass_type="power_series")
        helicity_n = bcfg.get("helicity_n")
        helicity_n = pcfg["helicity_n"] if helicity_n is None else helicity_n
        smin, smax, nsurf = bcfg.get("surfaces", [0.1, 0.9, 8])
        extra_outputs["bootstrap"] = bootstrap_mismatch_output(profiles, helicity_n,
                                                               np.linspace(smin, smax, int(nsurf)))
        if not vcfg.get("current_dofs"):
            print("WARNING bootstrap: vmex.current_dofs is off -- no design variable can carry the bootstrap current")
        if int(inp.ncurr) != 1:
            raise SystemExit("bootstrap needs NCURR = 1 (prescribed current profile) in the seed namelist")
        if not np.any(np.asarray(inp.ac, dtype=float)):
            print("WARNING bootstrap: the seed AC profile is all zero, so VMEC's shape normalization is degenerate and "
                  "the AC dofs have no gradient until CURTOR moves; seed with tools/make_bootstrap_seed.py")
    t0 = time.time()
    try:
        eq0 = opt.solve_equilibrium(inp)
    except Exception as e:  # noqa: BLE001
        # a ladder stage's seed is the previous stage's final boundary, which the optimizer reached by warm starts:
        # strongly shaped ones (W7-X size, pinched X-line boundaries) raise INITIAL JACOBIAN CHANGED SIGN cold even
        # with a re-guessed axis, multigrid or a smaller DELT (2026-10-19). VMEX's hot restart from the previous
        # stage's wout solves them in seconds.
        restart = vcfg.get("restart_wout") or _stage_wout_beside(seed_path)
        if not restart or not os.path.exists(restart):
            raise
        print(f"seed cold solve failed ({type(e).__name__}: {e}); hot restart from {restart}", flush=True)
        eq0 = opt.solve_equilibrium(inp, restart_from=restart)
    w0 = eq0.wout
    print(f"seed equilibrium: {time.time() - t0:.1f}s  aspect={float(w0.aspect):.4f}  "
          f"beta={float(w0.betatotal):.3e}  iota(axis,edge)=({float(w0.iotaf[0]):+.4f},{float(w0.iotaf[-1]):+.4f})")

    smin, smax, ns = pcfg["qs_surfaces"]
    if pcfg.get("symmetry", "qs") == "qi":
        # W7-X class: VMEX's traceable smooth quasi-isodynamic surrogate (Boozer |B| spectrum + level-set Goodman
        # construction) on the same surfaces, in place of quasi-symmetry; term name "qi"
        from vmex.core.omnigenity import QIResidual
        qs = QIResidual(np.linspace(smin, smax, int(ns)), **(pcfg.get("qi") or {}))
    else:
        qs = opt.QuasisymmetryRatioResidual(np.linspace(smin, smax, int(ns)),
                                            helicity_m=pcfg["helicity_m"], helicity_n=pcfg["helicity_n"])
    terms = [(qs, 0.0, 1.0)]
    aspect_target = pcfg["aspect_target"] if pcfg["aspect_target"] is not None else float(w0.aspect)
    terms.append((opt.aspect_ratio, aspect_target, pcfg["aspect_weight"]))
    if pcfg["iota_floor"] is not None:
        floor = float(pcfg["iota_floor"])

        def iota_floor(state, runtime):          # the term name ("iota_floor") is how VmexTermCost finds it
            return jnp.maximum(floor - opt.min_abs_iota(state, runtime), 0.0)
        # with iota_escalate the weight is applied (and escalated) outside VMEX, see main()
        terms.append((iota_floor, 0.0, 1.0 if pcfg.get("iota_escalate") else pcfg["iota_weight"]))
    if pcfg["beta_target"] is not None:
        bw = pcfg["beta_weight"] if pcfg["beta_weight"] is not None else 1.0 / pcfg["beta_target"]
        terms.append((opt.volume_average_beta, pcfg["beta_target"], bw))
    # array/boozer_all.py counterparts (Andrew's star_lite_browser campaign); each is a unit-weight VMEX term whose
    # weight is applied and escalated outside VMEX like iota_floor (ESCALATED_PLASMA_TERMS, main()). The term name is
    # the function name.
    wcfg = pcfg.get("magnetic_well") or {}
    if wcfg.get("enabled"):
        # boozer_all: MagneticWell, d2V/dpsi2 <= threshold along psi (vacuum, Boozer surfaces). Here, on the finite-beta
        # equilibrium: V'(s) must fall everywhere (V'' <= 0, Andrew's threshold 0) AND the well depth
        # W = (V'(0) - V'(1)) / V'(0) (simsopt's Vmec.vacuum_well proxy) must reach `floor`.
        from vmex.core.optimize import _field_chain
        wfloor = float(wcfg.get("floor", 0.0))

        def magnetic_well_floor(state, runtime):
            vp = jnp.abs(jnp.asarray(_field_chain(state, runtime)[4].vp))[1:]
            v0 = 1.5 * vp[0] - 0.5 * vp[1]
            v1 = 1.5 * vp[-1] - 0.5 * vp[-2]
            depth = jnp.maximum(wfloor - (v0 - v1) / v0, 0.0)
            slope = jnp.maximum(jnp.diff(vp) / v0 * vp.size, 0.0)
            return jnp.concatenate([depth[None], slope])
        terms.append((magnetic_well_floor, 0.0, 1.0))
    rcfg_ = pcfg.get("axis_ripple") or {}
    if rcfg_.get("enabled"):
        # boozer_all: ModBRippleOnFieldLine, |B| within +-threshold of its mean along the magnetic axis. The axis
        # |B|(phi) is the theta average on the two innermost half-mesh surfaces, extrapolated linearly in s to s = 0.
        # NOT opt.mirror_ratio near the axis: that is the full-surface modulation, whose poloidal 1/R part (~2 % on
        # the first half-mesh surface) no coil shape can remove (dn_32coils_vc_G_andrew_badripple stalled on it).
        rthr = float(rcfg_.get("threshold", 1e-3))
        from vmex.core.optimize import _field_chain

        def axis_ripple(state, runtime):
            fields = _field_chain(state, runtime)[3]
            bsq = jnp.asarray(fields.total_pressure) - jnp.asarray(fields.pressure)[:, None, None]
            bmag = jnp.sqrt(jnp.maximum(2.0 * bsq[1:3], jnp.finfo(bsq.dtype).tiny))
            w = jnp.ones(bmag.shape[1]).at[jnp.array([0, -1])].set(0.5)   # stellsym grid: theta in [0, pi] incl.
            bt = jnp.einsum("jtz,t->jz", bmag, w) / jnp.sum(w)
            r = 1.5 * bt[0] - 0.5 * bt[1]
            r = r / jnp.mean(r)
            return jnp.concatenate([jnp.maximum(r - (1.0 + rthr), 0.0),
                                    jnp.maximum((1.0 - rthr) - r, 0.0)]) / jnp.sqrt(r.size)
        terms.append((axis_ripple, 0.0, 1.0))
    mcfg = pcfg.get("mercier") or {}
    if mcfg.get("enabled"):
        # not in boozer_all (vacuum): Mercier at finite beta, DMerc[2:-1] >= margin (smooth one-sided residual)
        from vmex.core.stability import d_merc_state, mercier_stability_residual
        mmargin, msmooth = float(mcfg.get("margin", 0.0)), float(mcfg.get("smoothing", 1e-3))
        # s_min (None = VMEX's DMerc[2:-1]): at ns >= 31 the near-axis DMerc diverges with resolution (c24: min over
        # [2:-1] -45 / -96 / -160 / -757 at ns 31 / 41 / 51 / 101) while s >= 0.1 converges, so a higher-resolution
        # polish penalises s >= s_min only (runs/vmex/c24_ns_test, 2026-10-26)
        ms_min = mcfg.get("s_min")
        if int(cfg["vmex"]["ns"]) < 51:
            print(f"WARNING: Mercier term on a ns = {cfg['vmex']['ns']} equilibrium -- DMerc is not converged below "
                  f"ns ~ 51 (its verdict was wrong for 107 of 646 beta rungs at ns = 16); set vmex.ns >= 51", flush=True)

        # normalize: phi_edge (2026-11-07) -- penalise DMerc * PHIEDGE^2 instead of raw DMerc. VMEC's DMerc carries
        # 1/Phi^2 (DShear = (d iota / d Phi)^2 / 4 ...), so the raw residual of one and the same instability is ~400x
        # smaller on a W7-X-size device (PHIEDGE 2.4-3.7 Wb) than on designA x 10 (0.15 Wb): with weight 1e-4 and the
        # 100x escalation cap the term never acted in vessel_scan3 (violation flat over 25 outers, 30/31 beta-2 %
        # finals unstable at ns 51, runs/vmex/mercier_survey_scan3). DMerc * PHIEDGE^2 is invariant under scaling size
        # and field, so margin / smoothing / weight mean the same in every class. PHIEDGE is a fixed input here.
        mnorm = mcfg.get("normalize")
        if mnorm not in (None, False, "phi_edge"):
            raise SystemExit(f"plasma_objective.mercier.normalize: {mnorm!r} (choose phi_edge or leave unset)")
        mscale = float(inp.phiedge) ** 2 if mnorm == "phi_edge" else 1.0
        if mnorm:
            print(f"Mercier term: DMerc x PHIEDGE^2 = DMerc x {mscale:.4e} (margin {mmargin:g}, smoothing {msmooth:g} in "
                  f"those units)", flush=True)

        def mercier(state, runtime):
            if ms_min is None and not mnorm:
                return mercier_stability_residual(state, runtime, margin=mmargin, smoothing=msmooth)
            dm = d_merc_state(state, runtime) * mscale
            if ms_min is None:
                return msmooth * jnp.logaddexp(0.0, (mmargin - dm[2:-1]) / msmooth)
            k0 = max(2, int(np.ceil(float(ms_min) * (dm.shape[0] - 1))))     # static under jit
            return msmooth * jnp.logaddexp(0.0, (mmargin - dm[k0:-1]) / msmooth)     # smooth max(., 0)
        terms.append((mercier, 0.0, 1.0))
    plasma = VmexPlasma(inp, terms, max_mode=vcfg["max_mode"], nphi=vcfg["nphi"], ntheta=vcfg["ntheta"],
                        vc_digits=vcfg["vc_digits"], plasma_field=vcfg.get("plasma_field", "vacuum"),
                        current_dofs=vcfg["current_dofs"],
                        vary_major_radius=vcfg["vary_major_radius"], restart_from=eq0,
                        extra_outputs=extra_outputs, **(vcfg.get("problem_kwargs") or {}))
    return plasma, aspect_target


def _stage_wout_beside(seed_path):
    """``<stage>/wout/wout_combined_stage_final.nc`` for a seed ``<stage>/inputs/input.combined_stage_final`` (a ladder
    stage's seed is the previous stage's final), else None. Lets ladders started before vmex.restart_wout existed use
    the hot-restart fallback too."""
    if os.path.basename(seed_path) != "input.combined_stage_final":
        return None
    wout = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(seed_path))), "wout", "wout_combined_stage_final.nc")
    return wout if os.path.exists(wout) else None


def walk_plasma_to(plasma, cfg, target_path):
    """Move the plasma dofs from the cold-solved seed to the boundary of ``target_path`` in warm-started fractions.

    Optimizer checkpoints are often not cold-solvable VMEX seeds (trial E's final and outer 10-24 boundaries all raise
    INITIAL JACOBIAN CHANGED SIGN cold) while the seed they were reached from solves. Steps of up to 0.25, halved when
    VMEX rejects the trial, as the driver's resume walk; the boundary dofs are ``vmex.core.optimize.pack_boundary`` of
    the target at the run's resolution, current-profile dofs (if any) keep the seed's values.
    """
    import vmex as vj
    from vmex.core.optimize import pack_boundary

    vcfg = cfg["vmex"]
    inp = vj.VmecInput.from_file(target_path)
    if vcfg.get("mpol") or vcfg.get("ntor"):
        mp, nt = int(vcfg.get("mpol") or inp.mpol), int(vcfg.get("ntor") or inp.ntor)
        inp = inp.change_resolution(mpol=mp, ntor=nt, ntheta=2 * mp + 6, nzeta=2 * nt + 4)
    x0 = np.asarray(plasma.local_full_x, dtype=float).copy()
    xb = pack_boundary(inp, int(vcfg["max_mode"]), vary_major_radius=bool(vcfg["vary_major_radius"]))
    seed_xb = pack_boundary(plasma.inp, int(vcfg["max_mode"]), vary_major_radius=bool(vcfg["vary_major_radius"]))
    if not np.allclose(seed_xb, x0[:xb.size]):
        raise SystemExit("plasma dof layout differs from pack_boundary(seed); cannot map the target boundary")
    xt = x0.copy()
    xt[:xb.size] = xb
    print(f"walking the plasma boundary to {target_path}: max |dx| {np.max(np.abs(xt - x0)):.3e}", flush=True)
    t, step = 0.0, 0.25
    while t < 1.0:
        t_try = min(1.0, t + step)
        plasma.local_full_x = x0 + t_try * (xt - x0)
        if plasma.accepted:
            t, step = t_try, min(0.25, 2.0 * step)
            print(f"  reached t = {t:.4f}", flush=True)
        else:
            step /= 2.0
            if step < 1e-3:
                raise SystemExit(f"could not walk the plasma boundary to {target_path} (stuck at t = {t:.4f})")
    return plasma


def apply_overrides(cfg, overrides):
    for item in overrides:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def build_xpoint(cfg, design_path, config_id, coils, xpoint_path=None):
    """Upper X-point periodic field line re-bound to ``coils`` and solved."""
    dcfg = cfg["double_null"]
    if xpoint_path is not None:
        seed = load(xpoint_path)
        curve, length = seed["curve"], float(seed["length"])
    else:
        entry = load(design_path)[3][config_id]
        if not hasattr(entry, "curve"):
            raise SystemExit(f"{design_path} entry 3 holds {type(entry).__name__}, not a PeriodicFieldLine; "
                             "use an archive with solved X-points (e.g. convert/designA_after_scaled.json)")
        curve, length = entry.curve, float(CurveLength(entry.curve).J())
    xpoint = PeriodicFieldLine(BiotSavart(coils), curve,
                               options={"newton_tol": float(dcfg["newton_tol"]),
                                        "newton_maxiter": int(dcfg["newton_maxiter"]), "verbose": False})
    res = xpoint.run_code(length)
    if not res["success"]:
        raise SystemExit("the seed X-point field line does not solve on the seed coils")
    return xpoint


def circular_coils(scfg, seed_path):
    """Planar circular modular coils around the seed boundary's axis, stellarator symmetric with the
    seed's nfp; every coil carries the current that gives B0 = |PHIEDGE| / (pi a^2) at R0."""
    import vmex as vj
    inp = vj.VmecInput.from_file(seed_path)
    nt, nfp, ncoils = int(inp.ntor), int(inp.nfp), int(scfg["ncoils"])
    R0, a = float(inp.rbc[nt, 0]), float(inp.rbc[nt, 1])     # rbc[n + ntor, m]
    B0 = abs(float(inp.phiedge)) / (np.pi * a ** 2)
    R1 = float(scfg["R1"]) if scfg.get("R1") is not None else float(scfg["minor_radius_factor"]) * a
    I0 = 2 * np.pi * R0 * B0 / (4e-7 * np.pi * 2 * nfp * ncoils)
    base = create_equally_spaced_curves(ncoils, nfp, stellsym=True, R0=R0, R1=R1, order=int(scfg["order"]))
    coils = coils_via_symmetries(base, [Current(1.0) * I0 for _ in base], nfp, True)
    print(f"circular coil seed: nfp={nfp}, {len(coils)} coils (R0={R0:.3f} m, R1={R1:.3f} m, "
          f"order {scfg['order']}), {I0 / 1e3:.1f} kA each for B0={B0:.3f} T")
    return coils


def build_coils(cfg, design_path, config_id, coils_path=None, seed_path=None):
    if coils_path:
        coils = load(coils_path)
    elif cfg.get("coil_seed", {}).get("type", "archive") == "circular":
        coils = circular_coils(cfg["coil_seed"], seed_path)
    else:
        coils = load(design_path)[0][config_id].biotsavart.coils
    curves = [c.curve for c in coils]
    base_curves, seen = [], set()
    for c in coils:                         # curves owning free dofs (not RotatedCurve copies)
        for opt_ in c.curve.unique_dof_lineage:
            if opt_.local_dof_size > 0 and id(opt_) not in seen and "Curve" in type(opt_).__name__:
                seen.add(id(opt_))
                base_curves.append(opt_)
    # The B.n constraints see only stellarator-odd harmonics, so a coil set that loses its symmetry is invisible to
    # them: pin the symmetry-breaking coefficients of coils that are their own stellarator image (designA's phi = +-90
    # deg coils; utils/vmex_combined_stage.fix_self_symmetric_coil_parity). Changes the dof set, so a run checkpointed
    # without it resumes only with coils.fix_stellsym_parity: false.
    if cfg["coils"].get("fix_stellsym_parity", True):
        _, notes = fix_self_symmetric_coil_parity(base_curves)
        for note in notes:
            print(f"coil symmetry: {note}")
    print(f"coil symmetry error at the start: {coil_stellsym_error(coils):.2e} m", flush=True)
    # Coil currents are ScaledCurrent chains (e.g. a sign flip -1 wrapped around the
    # normalized base current); bound each underlying Current by the threshold over
    # the largest |product of scales| among the coils that use it.
    leaves = {}
    for c in coils:
        current, scale = c.current, 1.0
        while hasattr(current, "current_to_scale"):
            scale *= float(current.scale)
            current = current.current_to_scale
        prev = leaves.get(id(current), (current, 0.0))
        leaves[id(current)] = (current, max(prev[1], abs(scale)))
    base_currents = list(leaves.values())
    if cfg["coils"]["fix_currents"]:
        for current, _ in base_currents:
            current.fix_all()
    return coils, curves, base_curves, base_currents


def vmex_axis_curve(plasma, npts=129):
    """Magnetic axis of the current VMEX equilibrium (its accepted state, no new solve) as a CurveRZFourier.

    VMEC convention R = sum_n raxis_cc[n] cos(n nfp phi), Z = -sum_n zaxis_cs[n] sin(n nfp phi); on trial E's final
    state this matches the coil-field axis (PeriodicFieldLine) to 0.7 cm, the opposite sign would be 7 cm off.
    """
    wout = plasma.equilibrium().wout
    rc = np.asarray(wout.raxis_cc, dtype=float)
    zs = np.asarray(wout.zaxis_cs, dtype=float)
    curve = CurveRZFourier(np.linspace(0, 1, npts, endpoint=False), rc.size - 1, int(plasma.nfp), True)
    curve.x = np.concatenate([rc, -zs[1:]])
    return curve


def jacobi_scaling(al, constraints, names, step, cap):
    """Per-dof optimizer scale D_j = step / ||dc/dx_j|| (constraint-Jacobian column norm), capped.

    Fixed per-group steps can make one group inert in the optimizer's variables: on
    designA a coil step of 0.01 shrank the coil columns 100x, leaving 0.6 % of the
    objective's descent feasible (16 % in physical variables) and the optimizer pulled
    the plasma back to the seed instead of moving the coils.  Equilibrating the columns
    removes that bias; ``cap`` limits dofs the constraints barely see to ``cap`` x the
    median scale.
    """
    sq = np.zeros(len(names))
    for constraint in constraints:                       # every AL block, e.g. B.n + a free X-line
        res = constraint.residuals()
        for key, value in res.items():
            for i in range(value.size):
                ct = {k: np.zeros_like(v) for k, v in res.items()}
                ct[key][i] = 1.0
                d, parts = constraint.residuals_vjp_parts(ct)
                for owner, pc in parts.items():
                    d += owner.pullback(pc)
                sq += d(al) ** 2
    cols = np.sqrt(sq)
    inv = 1.0 / np.maximum(cols, 1e-300)
    med = float(np.median(inv))
    return step * np.minimum(inv, cap * med), cols


def main():
    args = parse_args()
    cfg_dir = os.path.dirname(os.path.abspath(args.config))
    cfg = apply_overrides(yaml.safe_load(open(args.config)), args.set)
    if args.smoke:
        cfg["vmex"]["max_mode"] = 1
        cfg["augmented_lagrangian"].update(outer_maxiter=2, inner_maxiter=3)
    seed_path = args.vmec_input or resolve(cfg["vmec_input"], cfg_dir)
    if seed_path is None:
        raise SystemExit("a seed VMEC namelist is required (--vmec-input or vmec_input in the config)")
    design_path = args.design or resolve(cfg["design"], cfg_dir)
    config_id = cfg["config_id"] if args.config_id is None else args.config_id
    os.makedirs(args.outdir, exist_ok=True)
    yaml.safe_dump(dict(cfg, vmec_input=seed_path, design=design_path, config_id=config_id),
                   open(os.path.join(args.outdir, "config_used.yaml"), "w"), sort_keys=False)

    # ---- objects --------------------------------------------------------------------------
    plasma, aspect_target = build_plasma(cfg, seed_path)
    from vmex.core import implicit as vmex_implicit
    vmex_cfg = plasma.problem.metadata.get("config")   # key of VMEX's process-local solve / hot-restart caches
    if args.plasma_target:
        walk_plasma_to(plasma, cfg, args.plasma_target)
    # virtual-casing sanity gate: at beta of a few percent the plasma's own field is ~1 % of |B|. Run 911 (2026-09-17)
    # started with a corrupt virtual-casing evaluation (B.n 1.25e8, pressure balance 2.3e18) that the SAME inputs do not
    # reproduce four days later -- cause unexplained -- and burned an hour at L ~ 1e28 before it was noticed.
    if plasma.plasma_field == "virtual_casing":
        out0 = plasma.outputs()
        bp = np.linalg.norm(np.asarray(out0["B_plasma"], dtype=float).reshape(3, -1), axis=0)
        b_in = np.sqrt(np.asarray(out0["Bin_mag2"], dtype=float).ravel())
        vc_ratio = bp / np.maximum(b_in, 1e-300)
        vc_max = float(cfg.get("vc_sanity_max_ratio", 0.2))
        # the MAXIMUM matters too: run 1072 passed a median-only gate at 2.4 % while one grid point read 30x |B|, and that
        # local spike alone corrupted the Jacobi scaling (plasma column norms 1.6e17) -- the run never moved
        vc_max_point = float(cfg.get("vc_sanity_max_point_ratio", 1.0))
        print(f"virtual casing at the start: |B_plasma|/|B_in| median {np.median(vc_ratio):.3e}, max {np.max(vc_ratio):.3e} "
              f"(gate: median <= {vc_max}, max <= {vc_max_point})", flush=True)
        if not np.all(np.isfinite(vc_ratio)) or np.median(vc_ratio) > vc_max or np.max(vc_ratio) > vc_max_point:
            raise SystemExit("virtual-casing field is implausible at the start (see the line above); refusing to optimize "
                             "with it -- rerun, or check the boundary regularity and the VC grid")
    # boundary guard: trials whose VMEX boundary self-intersects or whose poloidal speed |dx/dtheta| collapses
    # (degenerate VMEC Jacobian) are rejected before any solve; min_speed_ratio 0 switches it off
    gcfg = dict(dict(min_speed_ratio=0.1, nphi=8, ntheta=512), **(cfg.get("boundary_guard") or {}))
    guard_ratio = float(gcfg["min_speed_ratio"])
    ratio0, where0, cross0 = boundary_regularity(plasma, int(gcfg["nphi"]), int(gcfg["ntheta"]))
    print(f"boundary guard (min speed ratio {guard_ratio}): start boundary speed ratio {ratio0:.3g} at "
          f"(phi, theta)/2pi = ({where0[0]:.3f}, {where0[1]:.3f}), self-intersecting {cross0}", flush=True)
    coils, curves, base_curves, base_currents = build_coils(cfg, design_path, config_id, args.coils, seed_path)
    dn_cfg = cfg.get("double_null", {"enabled": False})
    dn_mode = dn_cfg.get("mode", "track")
    xpoint = None
    if dn_cfg.get("enabled") and dn_mode == "track":
        xpoint = build_xpoint(cfg, design_path, config_id, coils, args.xpoint)
    elif dn_cfg.get("enabled") and dn_mode == "free":
        if args.xpoint:                                   # a previous stage's X-line (curve + length)
            seed = load(args.xpoint)
            xpoint = FreeXLine(seed["curve"], float(seed["length"]))
        else:
            xpoint = free_xline_from_plasma(plasma, order=int(dn_cfg.get("free_order", 10)),
                                            offset=float(dn_cfg.get("free_offset", 0.06)))
            if int(dn_cfg.get("free_prefit_maxiter", 0)) > 0 and not args.resume:
                t0 = time.time()
                resid = fit_xline_to_field(xpoint, BiotSavart(coils), maxiter=dn_cfg["free_prefit_maxiter"])
                print(f"free X-line pre-fit on the frozen coils ({time.time() - t0:.0f}s): "
                      f"max field-line residual {resid:.3e}", flush=True)
    elif dn_cfg.get("enabled"):
        raise SystemExit(f"double_null.mode must be 'track' or 'free', got {dn_mode!r}")
    tracked = xpoint is not None and dn_mode == "track"
    ccfg, kcfg = cfg["consistency"], cfg["coils"]
    interface = PlasmaCoilInterface(plasma, BiotSavart(coils), mode=ccfg["mode"], mpol=ccfg["mpol"],
                                    ntor=ccfg["ntor"], field_strength=ccfg["field_strength"],
                                    p_edge=ccfg["p_edge"])
    J_length = sum(QuadraticPenalty(CurveLength(c), kcfg["length_threshold"], "max") for c in base_curves)
    J_curv = sum(LpCurveCurvature(c, 2, kcfg["curvature_threshold"]) for c in base_curves)
    J_msc = sum(QuadraticPenalty(MeanSquaredCurvature(c), kcfg["msc_threshold"], "max") for c in base_curves)
    J_cc = CurveCurveDistance(curves, kcfg["coil_coil_threshold"])
    J_cp = CoilPlasmaDistance(plasma, curves, kcfg["coil_plasma_threshold"])
    weights = {key: Weight(kcfg[f"{key}_weight"])
               for key in ("length", "curvature", "msc", "coil_coil", "coil_plasma", "current")}
    pcfg = cfg["plasma_objective"]
    iota_escalate = bool(pcfg.get("iota_escalate")) and pcfg["iota_floor"] is not None
    # VMEX terms weighted (and escalated) outside f_QS: (term name, plasma_objective key)
    extra_plasma = [(name, key) for name, key in ESCALATED_PLASMA_TERMS if (pcfg.get(key) or {}).get("enabled")]
    excluded = (("iota_floor",) if iota_escalate else ()) + tuple(name for name, _ in extra_plasma)
    terms = [(1.0, VmexQuasisymmetry(plasma, exclude=excluded)),
             (weights["length"], J_length), (weights["curvature"], J_curv),
             (weights["msc"], J_msc), (weights["coil_coil"], J_cc),
             (weights["coil_plasma"], J_cp)]
    if not kcfg["fix_currents"]:
        J_curr = sum(CurrentBound(current, kcfg["current_threshold"] / scale)
                     for current, scale in base_currents)
        terms.append((weights["current"], J_curr))
    J_iota = None
    if iota_escalate:
        # a double null pulls iota down persistently; a fixed quadratic weight only balances that at a shortfall
        # (trial C, weight 1e3: min |iota| 0.249 -> 0.239 over 3 outer iterations), so escalate it like the ranges
        weights["iota_floor"] = Weight(pcfg["iota_weight"])
        J_iota = VmexTermCost(plasma, "iota_floor")
        terms.append((weights["iota_floor"], J_iota))
    J_extra_plasma = {}
    for name, key in extra_plasma:
        weights[name] = Weight(pcfg[key]["weight"])
        J_extra_plasma[name] = VmexTermCost(plasma, name)
        terms.append((weights[name], J_extra_plasma[name]))
        print(f"plasma term {name} ({key}): weight {pcfg[key]['weight']:.3g}, unit-weight cost now "
              f"{J_extra_plasma[name].J():.4e}", flush=True)
    if float(kcfg.get("arclength_weight", 0.0)) > 0.0:
        # boozer_all: ArclengthVariation (uniform quadrature-point spacing along each coil), fixed weight
        weights["arclength"] = Weight(kcfg["arclength_weight"])
        terms.append((weights["arclength"], sum(ArclengthVariation(c) for c in base_curves)))
    # B.n is blind to scaling every coil current by the same factor: with free currents a field-strength condition is
    # required (pointwise pressure balance as a penalty lives only in examples/combined_stage_simple.py;
    # docs/field_strength_condition.md)
    if ccfg["field_strength"] is None and not kcfg["fix_currents"]:
        raise SystemExit("consistency.field_strength is null with free coil currents: nothing fixes the coil-current "
                         "scale (B.n is blind to it); use net_poloidal_current at finite beta")
    J_hyp = J_xdist = xline_constraint = J_coil_xline = J_flat = None
    # consistency.bnormal_band = tau: the B.n harmonics become a BAND |h| <= tau (two inequality rows each) instead of
    # equalities h = 0. As equalities their multipliers keep pulling toward an unreachable zero after the target is met,
    # and quasi-symmetry pays (v2a / v2b: f_QS 0.024 -> 0.116 and 0.022 -> 0.21 once B.n had met its target); inside the
    # band there is no force at all. The field-strength condition stays an equality (SubsetConstraint), index 0 as before.
    band_tau = ccfg.get("bnormal_band")
    constraints = [SubsetConstraint(interface, exclude=("bnormal",))] if band_tau else [interface]
    if xpoint is not None:
        if tracked:
            J_hyp = XpointHyperbolicity(xpoint, BiotSavart(coils), margin=dn_cfg["hyperbolicity_margin"])
        else:
            J_hyp = FreeXLineHyperbolicity(xpoint, BiotSavart(coils), margin=dn_cfg["hyperbolicity_margin"])
            xline_constraint = XLineFieldLineConstraint(xpoint, BiotSavart(coils))
            constraints.append(xline_constraint)
        J_xdist = XpointPlasmaDistance(xpoint, plasma, d_min=dn_cfg["distance_min"], d_max=dn_cfg["distance_max"],
                                       signed=not tracked)
        # escalated like the engineering ranges: a new X-line starts far from |tr M| >= 2 + margin, and designA-sized
        # weights (1e2) let that term alone bend the coils to 39 /m and blow B.n up 100x within 3 iterations
        weights["hyperbolicity"] = Weight(dn_cfg["hyperbolicity_weight"])
        weights["xpoint_distance"] = Weight(dn_cfg["distance_weight"])
        terms += [(weights["hyperbolicity"], J_hyp), (weights["xpoint_distance"], J_xdist)]
        if float(dn_cfg.get("flatness_weight", 0.0)) > 0.0 and tracked:
            print("X-line flatness needs a free X-line (double_null.mode: free); not applied", flush=True)
        elif float(dn_cfg.get("flatness_weight", 0.0)) > 0.0:
            # boozer_all --Z: FieldLineMeanZ, max |z - mean z| along the X-line <= flatness_threshold, escalated like the
            # ranges (user, 2026-10-18: a flat X-line is a design requirement, 10 mm, the flatter the better), plus a
            # fixed-weight pull on the whole deviation (threshold 0) that keeps flattening once inside the band
            weights["xline_flatness"] = Weight(dn_cfg["flatness_weight"])
            J_flat = XLineFlatness(xpoint, float(dn_cfg.get("flatness_threshold", 1e-2)))
            terms.append((weights["xline_flatness"], J_flat))
            if float(dn_cfg.get("flatness_pull_weight", 0.0)) > 0.0:
                weights["xline_flatness_pull"] = Weight(dn_cfg["flatness_pull_weight"])
                terms.append((weights["xline_flatness_pull"], XLineFlatness(xpoint, 0.0)))
            print(f"X-line flatness: max |z - mean z| = {J_flat.max_deviation():.4f} m "
                  f"(threshold {dn_cfg.get('flatness_threshold', 1e-2)}, pull weight "
                  f"{dn_cfg.get('flatness_pull_weight', 0.0)})", flush=True)
        if dn_cfg.get("coil_distance_min") is not None:
            # conformal-vessel room: coils >= margin + clearance from the X-lines (and, via coils.coil_plasma_threshold,
            # from the plasma); the vessel is then an offset of plasma + X-lines, built after the optimization
            weights["coil_xline"] = Weight(dn_cfg["coil_distance_weight"])
            J_coil_xline = CoilXLineDistance(curves, xpoint, float(dn_cfg["coil_distance_min"]))
            terms.append((weights["coil_xline"], J_coil_xline))
            print(f"coil-X-line distance now {J_coil_xline.shortest_distance():.4f} m (>= {dn_cfg['coil_distance_min']})",
                  flush=True)
        dists = J_xdist.distances()
        if dists is None:
            # XpointPlasmaDistance has no geometry: the X-line did not solve, or VMEX rejected the start equilibrium
            error = vmex_implicit._LAST_STATUS_ERROR.get(vmex_cfg)
            raise SystemExit("double null: no X-line/boundary distance at the start ("
                             + ("X-line field line not solved" if tracked and not ensure_fieldline_solved(xpoint)
                                else f"VMEX start equilibrium rejected, status {plasma.evaluate()['status']}"
                                + (f": {type(error).__name__}: {str(error)[:160]}" if error is not None else "")) + ")")
        print(f"double null ({dn_mode}): X-point tr(M) = {J_hyp.trace():+.4f}; distance to boundary "
              f"{dists.min():.4f} .. {dists.max():.4f} m (range {dn_cfg['distance_min']} .. {dn_cfg['distance_max']})"
              + (f"; field-line residual max {np.max(np.abs(xline_constraint.residuals()['fieldline'])):.3e}"
                 if xline_constraint is not None else ""))
    # ---- vessel (array/boozer_all.py, vessel_id 4, on_vessel 0, --distance 1): a helical tube whose centre line and
    # radius R(t) are dofs; the plasma boundary and the X-line stay >= margin inside it, the coils >= coil_clearance
    # outside it, and the X-line keeps a constant gap to it (|d - mean d| <= xline_distance_threshold). The margin
    # and clearance escalate like the ranges; the tube's shape limits (kappa R < 1, |dR/ds| < 1, R > 0, uniform
    # arclength) ride inside every vessel penalty and in the margin violation, as in boozer_all.
    vscfg = cfg.get("vessel") or {}
    sdf = J_vmargin = J_vxline = J_vclear = J_vxdist = None
    if vscfg.get("enabled"):
        # families (array/boozer_all.py vessel_id): helical (3: radius_modes 0, 4: radius R(t)), pipe (5000 + nseg: a
        # closed loop of nseg straight pipes mitred at the joints, node positions + radius free), torus (2)
        family = vscfg.get("family", "helical")
        if family not in ("helical", "pipe", "torus"):
            raise SystemExit(f"vessel family {family!r} not implemented (helical, pipe, torus)")
        # centre line seeded on the VMEX magnetic axis: R = sum raxis_cc cos(n nfp phi), Z = -sum zaxis_cs sin(...)
        # is CurveXYZFourierSymmetries(ntor=1) with xc = raxis_cc, ys = 0, zs = -zaxis_cs, truncated to num_modes
        wout = plasma.equilibrium().wout
        nv = int(vscfg.get("num_modes", 2))
        rax, zax = np.asarray(wout.raxis_cc, float), np.asarray(wout.zaxis_cs, float)
        xc = [float(rax[m]) if m < rax.size else 0.0 for m in range(nv + 1)]
        zs = [-float(zax[m]) if m < zax.size else 0.0 for m in range(1, nv + 1)]
        mr = int(vscfg.get("radius_modes", 2))

        def tube(radius):
            if family == "torus":
                return TorusSDF(float(radius), float(rax[0]))
            if family == "pipe":
                # nseg legs, a multiple of nfp; stellsym: an odd count per period puts node 0 on the phi = 0 plane
                nseg, nfp_v = int(vscfg["nseg"]), int(plasma.nfp)
                if nseg % nfp_v:
                    raise SystemExit(f"vessel.nseg {nseg} must be a multiple of nfp {nfp_v}")
                q = nseg // nfp_v
                axis = CurveXYZFourierSymmetries(64, nv, nfp_v, True, ntor=1)
                axis.set_dofs(np.concatenate([xc, [0.0] * nv, zs]))
                return HelicalCylinderVesselSDF(num_nodes=(q + 1) // 2 if q % 2 else q // 2, rr=float(radius),
                                                curve=axis, stellsym=True, boundary_node=bool(q % 2))
            return HelicalVesselSDF(xc, [0.0] * nv, zs, [float(radius)] + [0.0] * mr, int(plasma.nfp), ntor=1,
                                    stellsym=True)
        radius_dof = dict(helical="rc(0)", pipe="rr", torus="r")[family]
        rr = vscfg.get("radius")
        if vscfg.get("seed_file"):
            # continue a previous stage's vessel (run_ladder.py passes its vessel_final.json)
            sdf = load(vscfg["seed_file"])
            rr = float(sdf.local_full_x[sdf.local_full_dof_names.index(radius_dof)])
        elif rr is None:
            # start with the margin met: radius = deepest plasma/X-line point from the centre line + margin
            probe = tube(1.0)
            pts = [np.asarray(_to_points(plasma.evaluate()["out"]["gamma"])).reshape(-1, 3)]
            if xpoint is not None:
                pts.append(xpoint.curve.gamma())
            reach = max(float(np.max(probe.pure(p, probe.local_full_x, 1.0))) + 1.0 for p in pts)
            rr = reach + float(vscfg["margin"])
        if sdf is None:
            sdf = tube(rr)
        weights["vessel_margin"] = Weight(vscfg["margin_weight"])
        weights["coil_clearance"] = Weight(vscfg["clearance_weight"])
        J_vmargin = PlasmaVesselDistance(plasma, sdf, float(vscfg["margin"]))
        J_vclear = VesselDistance(sdf, list(base_curves), np.ones(len(base_curves)), float(vscfg["coil_clearance"]))
        terms += [(weights["vessel_margin"], J_vmargin), (weights["coil_clearance"], J_vclear)]
        if xpoint is not None:
            J_vxline = VesselDistance(sdf, [xpoint.curve], np.array([-1.0]), float(vscfg["margin"]))
            terms.append((weights["vessel_margin"], J_vxline))
            if float(vscfg.get("xline_distance_weight", 0.0)) > 0.0:
                weights["xline_vessel_distance"] = Weight(vscfg["xline_distance_weight"])
                J_vxdist = VesselDistance(sdf, [xpoint.curve], np.array([-1.0]),
                                          float(vscfg.get("xline_distance_threshold", 1e-3)), metric="distance")
                terms.append((weights["xline_vessel_distance"], J_vxdist))
        shape_desc = dict(helical=f"helical tube, centre line order {nv}, radius order {mr}",
                          pipe=f"welded pipe, {vscfg.get('nseg')} legs", torus="torus")[family]
        print(f"vessel: {shape_desc}, radius {rr:.4f} m, {len(sdf.local_full_x)} dofs; plasma depth "
              f"{J_vmargin.shortest_distance():.4f} m (margin {vscfg['margin']}), X-line depth "
              f"{J_vxline.shortest_distance()[0] if J_vxline is not None else float('nan'):.4f} m, coil clearance "
              f"{J_vclear.shortest_distance()[0]:.4f} m (>= {vscfg['coil_clearance']})", flush=True)

    # Smooth boundary regularity as an augmented-Lagrangian INEQUALITY, the counterpart of the hard guard below: the
    # guard only rejects (816/817/823 then stalled with 0 inner iterations once the phi = 0 tip reached it), while this
    # block has a gradient, so the optimizer slides along the limit and its multiplier learns the force the B.n and
    # X-line blocks push the tip with. Appended LAST so the existing block names ("0:bnormal", "1:fieldline") keep their
    # indices and --resume still finds them in al_state.yaml.
    rcfg = dict(dict(enabled=False, min_speed_ratio=0.15, nphi=8, ntheta=64), **(cfg.get("boundary_regularity") or {}))
    regularity = None
    if rcfg.get("enabled"):
        regularity = BoundaryRegularityConstraint(plasma, min_speed_ratio=float(rcfg["min_speed_ratio"]),
                                                  nphi=int(rcfg["nphi"]), ntheta=int(rcfg["ntheta"]))
        constraints.append(regularity)
        ratio_now, where_now = regularity.worst_ratio()
        print(f"boundary regularity constraint: {regularity.residuals()['boundary_regularity'].size} rows on "
              f"{rcfg['nphi']} x {rcfg['ntheta']}, floor {rcfg['min_speed_ratio']}; worst ratio now {ratio_now:.3f} at "
              f"(phi, theta)/2pi = ({where_now[0]:.3f}, {where_now[1]:.3f})", flush=True)
    band = None
    if band_tau:
        band = BandConstraint(interface, "bnormal", float(band_tau))
        constraints.append(band)
        h0 = np.asarray(interface.residuals()["bnormal"], dtype=float)
        print(f"B.n band: |harmonic| <= {float(band_tau):.2e} on {h0.size} harmonics ({2 * h0.size} inequality rows); "
              f"max |harmonic| now {np.max(np.abs(h0)):.2e} ({int(np.sum(np.abs(h0) > float(band_tau)))} outside)",
              flush=True)
    # Bootstrap self-consistency (section `bootstrap`): the equilibrium's <J.B> must equal the Redl bootstrap current of
    # the kinetic profiles. Appended LAST so earlier block names keep their indices for --resume.
    bootstrap = None
    if (cfg.get("bootstrap") or {}).get("enabled"):
        bootstrap = BootstrapConsistency(plasma)
        constraints.append(bootstrap)
        r0 = bootstrap.residuals()["bootstrap"]
        print(f"bootstrap consistency: {r0.size} Redl mismatch rows, max |row| now {np.max(np.abs(r0)):.3e}; "
              f"CURTOR now {float(plasma.vmec_input().curtor):.4e} A", flush=True)
    # Hard engineering limits (coils.hard): the listed hinge penalties leave the escalated weighted sum and become
    # augmented-Lagrangian rows sqrt(J) / limit = 0 (PenaltyConstraint), one per coil for length and curvature. Escalation
    # capped at 1e4x froze the solve (v2), at 100x it let curvature run to 2-7x and the coils into the plasma's clearance
    # (v3). Appended LAST so earlier block names keep their indices for --resume.
    hard = [str(key) for key in (kcfg.get("hard") or [])]
    if hard:
        available = dict(length=([QuadraticPenalty(CurveLength(c), kcfg["length_threshold"], "max") for c in base_curves],
                                 float(kcfg["length_threshold"]) / np.sqrt(2.0), J_length),
                         curvature=([LpCurveCurvature(c, 2, kcfg["curvature_threshold"]) for c in base_curves],
                                    float(kcfg["curvature_threshold"]), J_curv),
                         coil_coil=([J_cc], float(kcfg["coil_coil_threshold"]), J_cc),
                         coil_plasma=([J_cp], float(kcfg["coil_plasma_threshold"]), J_cp))
        if J_coil_xline is not None:
            available["coil_xline"] = ([J_coil_xline], float(dn_cfg["coil_distance_min"]), J_coil_xline)
        if sdf is not None:      # plasma and X-line >= margin inside the vessel, coils >= coil_clearance outside it
            available["vessel_margin"] = ([J_vmargin] + ([J_vxline] if J_vxline is not None else []),
                                          float(vscfg["margin"]), J_vmargin)
            available["vessel_clearance"] = ([J_vclear], float(vscfg["coil_clearance"]), J_vclear)
            if J_vxline is not None:
                terms = [(w, t) for w, t in terms if t is not J_vxline or "vessel_margin" not in hard]
        if J_flat is not None:
            available["xline_flatness"] = ([J_flat], float(dn_cfg.get("flatness_threshold", 1e-2)), J_flat)
        if J_hyp is not None:
            # |tr M| >= 2 + margin (one scalar: sqrt(J) / (2 + margin) is the relative shortfall); vessel scan 3: with
            # flatness hard and hyperbolicity soft, every HSX-class X-line went elliptic (|tr M| 1.7-1.95)
            available["hyperbolicity"] = ([J_hyp], 2.0 + float(dn_cfg["hyperbolicity_margin"]), J_hyp)
            available["xpoint_distance"] = ([J_xdist], float(dn_cfg["distance_min"]), J_xdist)
        if "mercier" in J_extra_plasma:
            # Mercier as an AL row (2026-11-07): sqrt(J) / (hard_scale / sqrt 2) = |softplus rows| / hard_scale, i.e.
            # the root-sum-square instability over s >= s_min in units of hard_scale (pair with normalize: phi_edge
            # so the unit is class-independent). As an escalated term it stalled at the 100x cap (vessel_scan3).
            mscale_hard = float((pcfg.get("mercier") or {}).get("hard_scale", 1.0))
            available["mercier"] = ([J_extra_plasma["mercier"]], mscale_hard / np.sqrt(2.0), J_extra_plasma["mercier"])
        optional = ("coil_xline", "vessel_margin", "vessel_clearance", "xline_flatness", "hyperbolicity",
                    "xpoint_distance", "mercier")   # absent without an X-line / a vessel / flatness / Mercier
        unknown = [key for key in hard if key not in available and key not in optional]
        if unknown:
            raise SystemExit(f"coils.hard: {unknown} not available here (choose from {sorted(available)})")
        dropped = {id(available[key][2]) for key in hard if key in available}
        terms = [(w, t) for w, t in terms if id(t) not in dropped]
        for key in (key for key in hard if key in available):
            rows, scale, _ = available[key]
            if key == "xline_flatness":
                block = XLineFlatnessConstraint(xpoint, scale)
            elif key in ("coil_plasma", "coil_xline"):
                # one row per coil point to its nearest target point: the averaged penalty diluted a 30 % closest-
                # approach violation to a 2 % row (ClosestApproachConstraint)
                block = ClosestApproachConstraint(f"hard_{key}", rows[0])
            else:
                block = PenaltyConstraint(f"hard_{key}", rows, scale)
            constraints.append(block)
            r_now = block.residuals()[f"hard_{key}"]
            print(f"hard limit {key}: {r_now.size} augmented-Lagrangian row(s), max relative violation now "
                  f"{np.max(r_now):.3e}", flush=True)
    initial_weights = {key: float(w.value) for key, w in weights.items()}   # config values, before resume/escalation
    objective = WeightedSum(terms)
    acfg = cfg["augmented_lagrangian"]
    penalty = acfg["penalty"]
    if xline_constraint is not None and dn_cfg.get("fieldline_penalty") is not None:
        # its own starting penalty: an O(0.1) field-line residual at rho(B.n) would throw the coils around
        penalty = {f"{i}:{key}": float(acfg["penalty"]) for i, c in enumerate(constraints) for key in c.residuals()}
        penalty[f"{len(constraints) - 1}:fieldline"] = float(dn_cfg["fieldline_penalty"])
    if acfg.get("penalty_blocks"):
        # starting penalties by block key (bnormal, bnormal_band, ...): a B.n block starting at the global rho stayed at
        # 2 % through 40 outer iterations of dn_converge while the escalated engineering weights grew past it
        if np.isscalar(penalty):
            penalty = {f"{i}:{key}": float(penalty) for i, c in enumerate(constraints) for key in c.residuals()}
        for name in penalty:
            if acfg["penalty_blocks"].get(name.split(":", 1)[1]) is not None:
                penalty[name] = float(acfg["penalty_blocks"][name.split(":", 1)[1]])
    al = AugmentedLagrangian(objective, constraints, penalty=penalty,
                             penalty_growth=acfg["penalty_growth"], penalty_max=acfg["penalty_max"],
                             eta0=acfg.get("eta0"), omega0=acfg.get("omega0"))

    # ---- scaled variables  x = x_seed + D * u -----------------------------------------------
    names = list(al.dof_names)
    plasma_mask = np.array([n.startswith(plasma.name + ":") for n in names])
    x_seed = al.x.copy()
    scfg = cfg["scaling"]
    D_saved = layout.find(args.outdir, "scaling_D.npy")
    if scfg.get("mode", "fixed") == "jacobi" and args.resume and os.path.exists(D_saved):
        D = np.load(D_saved)                     # the resumed run must keep the variables it started with
        print(f"Jacobi scaling reloaded from {D_saved}")
    elif scfg.get("mode", "fixed") == "jacobi":
        t0 = time.time()
        # the regularity block is excluded: its hundreds of nearly parallel rows would dominate the column norms and
        # shrink every plasma step, and the scaling is meant to equilibrate the physical consistency constraints
        # scaled on the EQUALITY formulation -- the interface itself plus the X-line block: a B.n band has the same
        # columns as the harmonics it bounds (twice, up to sign), and the regularity rows would dominate the column norms
        D, cols = jacobi_scaling(al, [interface] + ([xline_constraint] if xline_constraint is not None else [])
                                 + ([bootstrap] if bootstrap is not None else []), names,
                                 float(scfg.get("jacobi_step", 1e-2)), float(scfg.get("jacobi_cap", 100.0)))
        # smaller plasma steps: an L-BFGS step moves plasma and coil dofs together, and when its boundary part is a
        # shape VMEX cannot start from (INITIAL JACOBIAN CHANGED SIGN) the whole step, coil motion included, is rejected
        D[plasma_mask] *= float(scfg.get("plasma_factor", 1.0))
        print(f"Jacobi scaling ({time.time() - t0:.0f}s): column norm median plasma {np.median(cols[plasma_mask]):.3e}, "
              f"coil {np.median(cols[~plasma_mask]):.3e}; scale median plasma {np.median(D[plasma_mask]):.3e}, "
              f"coil {np.median(D[~plasma_mask]):.3e}", flush=True)
        np.save(layout.path(args.outdir, "scaling_D.npy", create=True), D)
    else:
        D = np.full(x_seed.size, float(scfg["coil_step"]))
        D[plasma_mask] = scfg["plasma_step"] * np.asarray(plasma.problem.scales)[plasma.dofs_free_status]
        if scfg.get("current_step") is not None:
            # coil currents relative to their seed value: coil_step is a length (m), and on a normalized Current dof of
            # ~0.1 it would allow 100 % current changes per unit step (the working designA example uses 0.1 x |I|)
            current_mask = np.array([n.startswith("Current") for n in names])
            D[current_mask] = float(scfg["current_step"]) * np.maximum(np.abs(x_seed[current_mask]), 1e-12)
    if sdf is not None:
        # the vessel enters no constraint block, so its Jacobi column is zero and would get the cap: give it the
        # coil step (metres) instead
        D[np.array([n.startswith(sdf.name + ":") for n in names])] = float(scfg["coil_step"])
    u = np.zeros_like(x_seed)
    k_start = 0
    ckpt = layout.find(args.outdir, "checkpoint.npz")
    if args.resume and os.path.exists(ckpt):
        data = np.load(ckpt, allow_pickle=True)
        if list(data["names"]) != names:
            raise SystemExit("checkpoint dof names do not match this configuration")
        u = (data["x"] - x_seed) / D
        # outer_maxiter is a budget for the whole stage: continue counting after the last saved outer iteration
        done = layout.matches(args.outdir, "coils_outer[0-9][0-9][0-9].json")
        k_start = int(os.path.basename(done[-1])[len("coils_outer"):-len(".json")]) + 1 if done else 0
        al.load_state_dict(yaml.safe_load(open(layout.find(args.outdir, "al_state.yaml"))))
        weights_file = layout.find(args.outdir, "coil_weights.yaml")
        if os.path.exists(weights_file):                  # keep escalated engineering weights
            for key, value in yaml.safe_load(open(weights_file)).items():
                if key in weights:                        # a term added since the checkpoint keeps its config weight
                    weights[key].value = float(value)
        print(f"resumed from {ckpt}")
        if tracked:                                    # a free X-line is part of the checkpoint dofs
            saved = layout.matches(args.outdir, "xpoint_outer*.json")
            if saved:
                al.x = x_seed + D * u              # coils at the checkpoint, then the X-point solved on them
                snap = load(saved[-1])
                xpoint.curve.x = snap["curve"].x
                xpoint.res = dict(xpoint.res, length=float(snap["length"]))
                xpoint.need_to_run_code = True
                if not ensure_fieldline_solved(xpoint):
                    raise SystemExit(f"X-point from {saved[-1]} does not solve on the checkpoint coils")
                print(f"resumed X-point from {saved[-1]}")
    print(f"dofs: {x_seed.size} total = {plasma_mask.sum()} plasma + {(~plasma_mask).sum()} coil; "
          f"constraints: {len(interface.residuals()['bnormal'])} B.n harmonics"
          f"{' + 1 ' + ccfg['field_strength'] if ccfg['field_strength'] else ''}"
          + (f" + {xline_constraint.residuals()['fieldline'].size} X-line field-line equations"
             if xline_constraint is not None else ""))

    # ---- objective with failed-solve barrier (same construction as array/boozer_all.py) ----
    good = {}
    best = {}      # lowest-L successful evaluation of the current inner solve (the fallback point)

    def restore_vmex(point):
        """Put back VMEX's stored solve of ``point`` so re-evaluating it returns that equilibrium.

        VMEX seeds each trial from its latest converged state (then cold); a boundary reached only through a chain of
        warm starts can fail from a later, different seed (dn_converge outer 2: the best point of the inner solve raised
        INITIAL JACOBIAN CHANGED SIGN on re-evaluation). A stored (params, result) pair is returned without solving.
        """
        if point.get("solve") is not None:
            vmex_implicit._LAST_SOLVE[vmex_cfg] = point["solve"]
        if point.get("hot") is not None:
            vmex_implicit._HOT_CACHE[vmex_cfg] = point["hot"]
        if point.get("xsnap") is not None:
            # the tracked X-point is Newton-solved in place, so re-evaluating this point from whatever curve the last
            # trials left can land on another branch or not solve at all (job 826 crashed exactly there at outer 7)
            fieldline_restore(xpoint, point["xsnap"])
    xsnap = fieldline_snapshot(xpoint) if tracked else None
    progress = dict(n=0, t0=time.time(), every=int(cfg.get("progress_every", 10)))

    def fun(uu):
        x = x_seed + D * uu
        al.x = x
        reasons = []
        nonlocal xsnap
        if guard_ratio > 0.0:
            ratio, where, crossing = boundary_regularity(plasma, int(gcfg["nphi"]), int(gcfg["ntheta"]))
            if crossing or ratio < guard_ratio:
                reasons.append(f"boundary guard: {'self-intersecting cross-section, ' if crossing else ''}poloidal speed "
                               f"ratio {ratio:.3g} at (phi, theta)/2pi = ({where[0]:.3f}, {where[1]:.3f})")
        try:
            if reasons:
                pass
            elif tracked and not ensure_fieldline_solved(xpoint):
                reasons.append("X-point field line did not solve")
            else:
                J, g = al.J(), al.dJ() * D
                if not np.isfinite(J) or not np.all(np.isfinite(g)):
                    bad = ~np.isfinite(g)
                    reasons.append(f"non-finite value/gradient (L {J:.3e}; non-finite gradient entries: "
                                   f"{int(bad[plasma_mask].sum())} plasma, {int(bad[~plasma_mask].sum())} coil/X-line)")
                if not plasma.accepted:
                    # status 1 = failed solve (typed VmecError, e.g. a Jacobian sign change), 2 = iteration
                    # budget exhausted above max_fsq_ratio (vmex/core/implicit.py); log WHY the trial failed
                    # (no import in here: it would make vmex_implicit local to fun and break best.update below)
                    error = vmex_implicit._LAST_STATUS_ERROR.get(vmex_cfg)
                    reasons.append(f"VMEX solve rejected (status {plasma.evaluate()['status']}"
                                   + (f": {type(error).__name__}: {str(error)[:160]}" if error is not None else "") + ")")
                elif plasma.plasma_field == "virtual_casing":
                    # per-evaluation virtual-casing check: with many cores JAX/XLA CPU returned nondeterministic garbage
                    # from virtual_casing_jax (|B_plasma| up to 1e10 T at random points; jobs 2324-2326, 2418: fine on
                    # 2-4 cores, garbage with 12) -- a start-up gate alone cannot catch it, so reject such trials
                    out_vc = plasma.outputs()
                    vc_ratio = (np.linalg.norm(np.asarray(out_vc["B_plasma"], dtype=float).reshape(3, -1), axis=0)
                                / np.sqrt(np.maximum(np.asarray(out_vc["Bin_mag2"], dtype=float).ravel(), 1e-300)))
                    if not np.all(np.isfinite(vc_ratio)) or np.max(vc_ratio) > vc_max_point:
                        reasons.append(f"virtual casing implausible (max |B_plasma|/|B_in| {np.max(vc_ratio):.3e})")
        except Exception as e:  # noqa: BLE001 -- any solver failure becomes a barrier
            reasons.append(f"evaluation raised {e!r}")
        progress["n"] += 1
        if reasons and tracked:
            fieldline_restore(xpoint, xsnap)     # Newton overwrote the curve dofs in place
        if not reasons:
            good.update(u=uu.copy(), J=J, g=g.copy())
            if not best or J < best["J"]:
                # keep VMEX's solve of this point too: re-solving it later from a different warm start can fail
                best.update(u=uu.copy(), J=J, solve=vmex_implicit._LAST_SOLVE.get(vmex_cfg),
                            hot=vmex_implicit._HOT_CACHE.get(vmex_cfg),
                            xsnap=fieldline_snapshot(xpoint) if tracked else None)
            if tracked:
                xsnap = fieldline_snapshot(xpoint)
            if progress["every"] > 0 and progress["n"] % progress["every"] == 0:
                norms = al.constraint_norms()
                print(f"  eval {progress['n']:5d}  {time.time() - progress['t0']:7.0f}s  L={J:.6e}  "
                      f"|grad|inf={np.max(np.abs(g)):.2e}  " + "  ".join(f"{kk}={vv:.2e}" for kk, vv in norms.items()),
                      flush=True)
            return J, g
        if not good:
            raise RuntimeError("failure at the starting point: " + "; ".join(reasons))
        print("  barrier: " + "; ".join(reasons))
        du = uu - good["u"]
        nrm2 = float(du @ du) + 1e-300
        gdu = float(good["g"] @ du)
        K = max(abs(gdu) / (cfg["barrier_retreat"] * nrm2), 1.0)
        return good["J"] + gdu + 0.5 * K * nrm2, good["g"] + K * du

    def report(k, record=None):
        al.x = x_seed + D * u
        res = interface.residuals()
        costs = plasma.term_costs()
        lengths = [CurveLength(c).J() for c in base_curves]
        kappa = max(float(np.max(c.kappa())) for c in base_curves)
        row = dict(outer=k, L=float(al.J()), f_QS=float(sum(costs.values())), terms=costs,
                   bnormal_inf=float(np.max(np.abs(res["bnormal"]))),
                   rms_Bn_over_B=interface.rms_bnormal_over_B(),
                   # pointwise (|B_out|^2 - |B_in|^2 - 2 mu0 p_edge)/|B_in|^2 on the optimisation grid: a diagnostic,
                   # never a constraint here (docs/field_strength_condition.md)
                   pressure_jump=interface.pressure_balance_stats(),
                   field_strength=next((float(res[k][0]) for k in ("toroidal_flux", "pressure_balance", "net_poloidal_current")
                                        if k in res), None),
                   coil_lengths=[float(v) for v in lengths], max_curvature=kappa,
                   coil_coil_penalty=float(J_cc.J()), coil_plasma_penalty=float(J_cp.J()),
                   vmex_forward=plasma.n_forward, vmex_backward=plasma.n_backward,
                   coil_violations=coil_violations(),
                   coil_symmetry_error=coil_stellsym_error(coils),
                   boundary_min_speed_ratio=float(boundary_regularity(plasma, int(gcfg["nphi"]), int(gcfg["ntheta"]))[0]),
                   coil_weights={key: float(w.value) for key, w in weights.items()})
        if xpoint is not None:
            dists = J_xdist.distances()
            row.update(xpoint_trace=J_hyp.trace(), xpoint_distance_min=float(dists.min()),
                       xpoint_distance_max=float(dists.max()), xpoint_penalty=float(J_xdist.J()))
            if J_flat is not None:
                row["xline_z_deviation"] = J_flat.max_deviation()
            if J_coil_xline is not None:
                row["coil_xline_distance"] = J_coil_xline.shortest_distance()
            if xline_constraint is not None:
                row["xline_residual_max"] = float(np.max(np.abs(xline_constraint.residuals()["fieldline"])))

        if sdf is not None:
            row.update(vessel_plasma_depth=float(J_vmargin.shortest_distance()),
                       vessel_xline_depth=float(J_vxline.shortest_distance()[0]) if J_vxline is not None else None,
                       vessel_coil_clearance=float(J_vclear.shortest_distance()[0]),
                       vessel_xline_gap_spread=float(J_vxdist.longest_distance()[0]) if J_vxdist is not None else None,
                       vessel_shape_violation=float(vessel_shape_violation()),
                       vessel_dofs=dict(zip(sdf.local_full_dof_names, map(float, sdf.local_full_x))))
        if bootstrap is not None:
            row.update(bootstrap_residual_max=float(np.max(np.abs(bootstrap.residuals()["bootstrap"]))),
                       curtor=float(plasma.vmec_input().curtor))
        if record is not None:
            row.update(penalties=record["penalties"], actions=record["actions"],
                       grad_norm=record["grad_norm"], omega=record["omega"], converged=record["converged"])
        print(f"[outer {k}] L={row['L']:.6e} f_QS={row['f_QS']:.6e} |B.n modes|inf={row['bnormal_inf']:.3e} "
              f"rms(B.n/B)={row['rms_Bn_over_B']:.3e} field_strength={row['field_strength']} "
              + "".join(f"{n}={costs[n]:.3e} " for n in costs if n in dict(extra_plasma)) +
              f"pressure jump rms={row['pressure_jump']['rms']:.2e} "
              f"max kappa={kappa:.2f} lengths={np.round(lengths, 3).tolist()} "
              f"coil symmetry error={row['coil_symmetry_error']:.1e} m "
              f"rho={row.get('penalties')} VMEX fwd/bwd={plasma.n_forward}/{plasma.n_backward} "
              f"boundary speed ratio={row['boundary_min_speed_ratio']:.3g}"
              + (f" X-point trM={row['xpoint_trace']:+.3f} d=[{row['xpoint_distance_min']:.4f},{row['xpoint_distance_max']:.4f}]"
                 if xpoint is not None else "")
              + (f" X-line residual={row['xline_residual_max']:.3e}" if "xline_residual_max" in row else "")
              + (f" coil-X-line={row['coil_xline_distance']:.4f}" if "coil_xline_distance" in row else "")
              + (f" X-line dz={row['xline_z_deviation']:.4f}" if "xline_z_deviation" in row else "")
              + f" coil-plasma={J_cp.shortest_distance():.4f}"
              + (f" vessel: plasma depth={row['vessel_plasma_depth']:.4f} X-line depth={row['vessel_xline_depth']:.4f}"
                 f" coil clearance={row['vessel_coil_clearance']:.4f} X-line gap spread="
                 f"{row['vessel_xline_gap_spread'] if row['vessel_xline_gap_spread'] is not None else float('nan'):.4f}"
                 f" shape={row['vessel_shape_violation']:.2e} radius={next((float(v) for n_, v in zip(sdf.local_full_dof_names, sdf.local_full_x) if n_ in ('rc(0)', 'rr', 'r')), float('nan')):.4f}"
                 if sdf is not None else "")
              + (f" bootstrap mismatch={row['bootstrap_residual_max']:.3e} CURTOR={row['curtor']:.4e} A"
                 if "bootstrap_residual_max" in row else ""), flush=True)
        with open(os.path.join(args.outdir, "history.yaml"), "a") as fh:
            yaml.safe_dump([row], fh, sort_keys=False)
        return row

    def checkpoint(tag):
        al.x = x_seed + D * u
        np.savez(layout.path(args.outdir, "checkpoint.npz", create=True), x=al.x, names=np.array(names, dtype=object))
        yaml.safe_dump(al.state_dict(), open(layout.path(args.outdir, "al_state.yaml", create=True), "w"))
        save(coils, layout.path(args.outdir, f"coils_{tag}.json", create=True))
        if xpoint is not None:
            save({"curve": xpoint.curve, "length": float(xpoint.res["length"])},
                 layout.path(args.outdir, f"xpoint_{tag}.json", create=True))
        plasma.vmec_input().to_indata(layout.path(args.outdir, f"input.combined_stage_{tag}", create=True))
        if sdf is not None:
            save(sdf, layout.path(args.outdir, f"vessel_{tag}.json", create=True))
        yaml.safe_dump({key: float(w.value) for key, w in weights.items()},
                       open(layout.path(args.outdir, "coil_weights.yaml", create=True), "w"))

    def coil_violations():
        """Relative violation of each engineering range (0 when it holds)."""
        def thr(key):
            return float(kcfg[f"{key}_threshold"])
        v = dict(length=max(CurveLength(c).J() for c in base_curves) / thr("length") - 1.0,
                 curvature=max(float(np.max(c.kappa())) for c in base_curves) / thr("curvature") - 1.0,
                 msc=max(MeanSquaredCurvature(c).J() for c in base_curves) / thr("msc") - 1.0,
                 coil_coil=1.0 - J_cc.shortest_distance() / thr("coil_coil"),
                 coil_plasma=1.0 - J_cp.shortest_distance() / thr("coil_plasma"))
        if not kcfg["fix_currents"]:
            v["current"] = max(abs(float(current.get_value())) * scale
                               for current, scale in base_currents) / thr("current") - 1.0
        if J_iota is not None:                           # unit-weight VMEX row = floor - min|iota|
            v["iota_floor"] = float(np.sqrt(2.0 * J_iota.J())) / float(pcfg["iota_floor"])
        for name, key in extra_plasma:                  # rms of the unit-weight rows over a natural scale
            scale = dict(magnetic_well=max(float(pcfg[key].get("floor", 0.0)), 1e-2),
                         axis_ripple=float(pcfg[key].get("threshold", 1e-3)),
                         mercier=float(pcfg[key].get("hard_scale", 1.0)))[key]
            v[name] = float(np.sqrt(2.0 * J_extra_plasma[name].J())) / scale
        if xpoint is not None:                           # double-null ranges escalate the same way
            margin = 2.0 + float(dn_cfg["hyperbolicity_margin"])
            v["hyperbolicity"] = 1.0 - abs(J_hyp.trace()) / margin
            dists = J_xdist.distances()
            if dists is not None:
                v["xpoint_distance"] = max(1.0 - dists.min() / float(dn_cfg["distance_min"]),
                                           dists.max() / float(dn_cfg["distance_max"]) - 1.0)
            if J_flat is not None:
                v["xline_flatness"] = J_flat.max_deviation() / float(dn_cfg.get("flatness_threshold", 1e-2)) - 1.0
            if J_coil_xline is not None:
                v["coil_xline"] = 1.0 - J_coil_xline.shortest_distance() / float(dn_cfg["coil_distance_min"])
        if sdf is not None:                              # boozer_all's margin / clearance / X-line-gap errors
            depth = min([J_vmargin.shortest_distance()] + ([J_vxline.shortest_distance()[0]] if J_vxline else []))
            margin = float(vscfg["margin"])
            v["vessel_margin"] = max((margin - depth) / margin, vessel_shape_violation())
            v["coil_clearance"] = 1.0 - J_vclear.shortest_distance()[0] / float(vscfg["coil_clearance"])
            if J_vxdist is not None:
                thr_d = float(vscfg.get("xline_distance_threshold", 1e-3))
                v["xline_vessel_distance"] = J_vxdist.longest_distance()[0] / thr_d - 1.0
        return {key: float(max(val, 0.0)) for key, val in v.items()}

    def vessel_shape_violation():
        """boozer_all's vessel validity: helical tube kappa R < 1, |dR/ds| < 1, R > 0, arclength variation; welded
        pipe geometric_violation (radius, leg length, turn cap, mitre interference); a torus has none."""
        if isinstance(sdf, HelicalCylinderVesselSDF):
            return float(sdf.geometric_violation())
        if isinstance(sdf, TorusSDF):
            return 0.0
        return max(max(sdf.max_kappa_radius() - 1.0, 0.0), max(sdf.max_dr_ds() - 1.0, 0.0),
                   max(-sdf.min_radius(), 0.0), float(sdf.arclength_variation()))

    def escalate_coil_weights():
        """array/boozer_all.py practice: fixed penalty weights cannot hold a range once the augmented
        Lagrangian terms grow, so multiply the weight of every range violated by more than
        escalation_tol (relative) by escalation_factor after each outer iteration."""
        if not kcfg.get("escalate", False):
            return {}
        actions = {}
        max_factor = float(kcfg.get("escalation_max_factor", np.inf))
        for key, viol in coil_violations().items():
            if key in hard or (key == "coil_clearance" and "vessel_clearance" in hard):   # held by an AL block
                continue
            cap = min(float(kcfg["weight_max"]), max_factor * initial_weights[key])
            if viol > float(kcfg["escalation_tol"]) and weights[key].value < cap:
                weights[key] *= float(kcfg["escalation_factor"])
                actions[key] = dict(violation=viol, weight=float(weights[key].value))
        return actions

    # ---- history snapshots: coils, VMEX boundary, magnetic axis, X-point line (snapshots/, run_layout.py) ----
    history_every = int(cfg.get("history_every", 5))
    snapshot_index = layout.path(args.outdir, "snapshots.yaml", create=True)
    n_snapshots = len(layout.matches(args.outdir, "snapshot_[0-9]*.json"))

    def snapshot(label, outer=None, inner=None, L=None):
        """Save the geometry at the current al.x as snapshots/snapshot_<n>.json (a simsopt archive of a dict) and index it.

        The axis comes from VMEX's accepted state; VMEX's warm-start caches are put back afterwards so taking a
        snapshot never changes the optimization path. A failure is logged and never stops the run.
        """
        nonlocal n_snapshots
        t0 = time.time()
        saved = dict(solve=vmex_implicit._LAST_SOLVE.get(vmex_cfg), hot=vmex_implicit._HOT_CACHE.get(vmex_cfg))
        try:
            if not plasma.accepted:
                print(f"  snapshot {label}: equilibrium not accepted, skipped", flush=True)
                return
            with tempfile.TemporaryDirectory() as tmp:
                namelist = os.path.join(tmp, "input.snapshot")
                plasma.vmec_input().to_indata(namelist)
                vmec_text = open(namelist).read()
                boundary = SurfaceRZFourier.from_vmec_input(namelist)
            try:
                axis = vmex_axis_curve(plasma)
            except Exception as e:  # noqa: BLE001
                axis = None
                print(f"  snapshot {label}: magnetic axis unavailable ({e!r})", flush=True)
            name = f"snapshot_{n_snapshots:05d}.json"
            save(dict(label=label, outer=outer, inner_iteration=inner, L=None if L is None else float(L),
                      coils=coils, boundary=boundary, vmec_input=vmec_text, magnetic_axis=axis,
                      xpoint=(dict(curve=xpoint.curve, length=float(xpoint.res["length"]))
                              if xpoint is not None else None)),
                 layout.path(args.outdir, name, create=True))
            with open(snapshot_index, "a") as fh:
                yaml.safe_dump([dict(file=name, label=label, outer=outer, inner_iteration=inner,
                                     L=None if L is None else float(L), wall_s=round(time.time() - t0, 2),
                                     time=time.strftime("%Y-%m-%d %H:%M:%S"))], fh, sort_keys=False)
            n_snapshots += 1
        except Exception as e:  # noqa: BLE001
            print(f"  snapshot {label} failed: {e!r}", flush=True)
        finally:
            restore_vmex(saved)

    inner_it = dict(n=0)

    def history_callback(uk):
        """L-BFGS-B iteration callback: a snapshot every ``history_every`` inner iterations."""
        inner_it["n"] += 1
        if history_every <= 0 or inner_it["n"] % history_every:
            return
        x = x_seed + D * uk
        if not np.array_equal(al.x, x):        # normally the accepted iterate is the last evaluation already
            al.x = x
        snapshot(f"outer{k:03d}_it{inner_it['n']:03d}", outer=k, inner=inner_it["n"], L=good.get("J"))

    if k_start:
        # the checkpoint boundary may not solve from the stage seed's state (reached only through warm starts):
        # walk x_seed -> checkpoint in fractions, each solve warm-started from the previous one
        try:
            good.clear()
            fun(u)
        except RuntimeError as e:
            print(f"checkpoint does not solve from the seed state ({str(e)[:120]}); walking to it", flush=True)
            u_target, t, step = u.copy(), 0.0, 0.25
            while t < 1.0:
                t_try = min(1.0, t + step)
                try:
                    good.clear()
                    fun(t_try * u_target)
                    t, step = t_try, min(0.25, 2.0 * step)
                    print(f"  reached t = {t:.4f}", flush=True)
                except RuntimeError:
                    step /= 2.0
                    if step < 1e-3:
                        raise SystemExit(f"could not reach the checkpoint from the seed equilibrium (stuck at t = {t:.4f})")
            u = u_target
        good.clear()
    report(-1)
    if k_start:
        print(f"continuing at outer iteration {k_start} of {acfg['outer_maxiter']}", flush=True)
        snapshot(f"resume_outer{k_start:03d}", outer=k_start)
    else:
        # the stage's input state, then snapshot 0 of its history
        save(coils, layout.path(args.outdir, "coils_seed.json", create=True))
        if xpoint is not None:
            save({"curve": xpoint.curve, "length": float(xpoint.res["length"])},
                 layout.path(args.outdir, "xpoint_seed.json", create=True))
        plasma.vmec_input().to_indata(layout.path(args.outdir, "input.combined_stage_seed", create=True))
        snapshot("initial", outer=-1, inner=0)
    for k in range(k_start, acfg["outer_maxiter"]):
        good.clear()
        best.clear()
        # this outer's accepted starting point with its VMEX and X-line state: the last fallback when even the inner
        # solve's best point no longer evaluates (dn_introduce, 5 HSX devices 2026-10-20: non-finite on re-evaluation)
        start = dict(u=u.copy(), solve=vmex_implicit._LAST_SOLVE.get(vmex_cfg), hot=vmex_implicit._HOT_CACHE.get(vmex_cfg),
                     xsnap=fieldline_snapshot(xpoint) if tracked else None)
        t0 = time.time()
        inner_it["n"] = 0
        res = minimize(fun, u, jac=True, method="L-BFGS-B", callback=history_callback,
                       options=dict(maxiter=acfg["inner_maxiter"], gtol=al.omega, ftol=1e-15))
        # L-BFGS-B returns its last iterate, which can be a REJECTED trial (a barrier value):
        # re-evaluate it without an anchor, and fall back to the last accepted point if it fails.
        # Fall back to the LOWEST-L successful evaluation, not the most recent one: the most recent can be a trial the
        # line search had already rejected for insufficient decrease (dn_converge outer 2 jumped L 0.08 -> 5.7 that way).
        best_point = dict(best)
        good.clear()
        try:
            J_end, g = fun(res.x)
            u = res.x
            if best_point and J_end > best_point["J"]:
                print(f"  inner solve ended above its best point (L {J_end:.6e} > {best_point['J']:.6e}); "
                      "continuing from the best point", flush=True)
                u = best_point["u"]
                good.clear()
                restore_vmex(best_point)
                _, g = fun(u)
        except RuntimeError:
            if not best_point:
                raise
            print("  inner solve ended on a rejected point; continuing from the best accepted evaluation", flush=True)
            u = best_point["u"]
            good.clear()
            restore_vmex(best_point)
            try:
                _, g = fun(u)
            except RuntimeError as e:
                print(f"  best point no longer evaluates ({e}); continuing from this outer's starting point", flush=True)
                u = start["u"]
                good.clear()
                restore_vmex(start)
                _, g = fun(u)
        record = al.update(float(np.max(np.abs(g))), ctol=acfg["ctol"], gtol=acfg["gtol"])
        escalated = escalate_coil_weights()
        if escalated:
            print(f"[outer {k}] coil weights escalated: " + ", ".join(
                f"{key} (violation {a['violation']:.3g}) -> {a['weight']:.3g}" for key, a in escalated.items()), flush=True)
        print(f"[outer {k}] inner: {res.nit} it, {res.message}, {time.time() - t0:.0f}s")
        row_k = report(k, record)
        checkpoint(f"outer{k:03d}")
        snapshot(f"outer{k:03d}_end", outer=k, inner=int(res.nit), L=row_k["L"])
        if record["converged"]:
            print("augmented Lagrangian converged")
            break

    # ---- final outputs ----------------------------------------------------------------------
    import vmex as vj
    al.x = x_seed + D * u
    checkpoint("final")
    snapshot("final")
    try:
        wout_path = vj.write_wout(layout.path(args.outdir, "wout_combined_stage_final.nc", create=True),
                                  plasma.equilibrium().wout)
        print(f"wrote {wout_path}")
    except Exception as e:  # noqa: BLE001
        print(f"could not write final wout: {e!r}")

    # ---- final free-boundary consistency check (reported, not optimised) ----------------------
    # The constraints only hold B.n harmonics and ONE field-strength scalar on the optimisation grid. Re-evaluate the
    # pointwise interface conditions -- B.n and total-pressure balance |B_out|^2 = |B_in|^2 + 2 mu0 p_edge -- with
    # virtual casing on an independent, finer grid (as VMEX's single_stage_optimization_finite_beta.py does). Why
    # pressure balance is a check here and not a constraint: docs/field_strength_condition.md.
    fcfg = cfg.get("final_check") or {}
    if fcfg.get("enabled", True) and plasma.accepted:
        t0 = time.time()
        try:
            check = interface_check(plasma, BiotSavart(coils), nphi=int(fcfg.get("nphi", 32)),
                                    ntheta=int(fcfg.get("ntheta", 32)), digits=int(fcfg.get("digits", 4)),
                                    p_edge=float(ccfg["p_edge"]))
            check["optimisation_grid"] = interface.pressure_balance_stats()
            check["field_strength"] = {k: float(v[0]) for k, v in interface.residuals().items() if k != "bnormal"}
            yaml.safe_dump(check, open(os.path.join(args.outdir, "final_check.yaml"), "w"), sort_keys=False)
            print(f"final check ({check['nphi']}x{check['ntheta']}, {time.time() - t0:.0f}s): "
                  f"rms(B.n/B)={check['bnormal_rms']:.2e} max={check['bnormal_max']:.2e}; "
                  f"pressure jump (|B_out|^2-|B_in|^2-2mu0 p)/|B_in|^2 mean={check['pressure_mean']:+.2e} "
                  f"rms={check['pressure_rms']:.2e} max={check['pressure_max']:.2e} -> final_check.yaml", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"final check failed: {e!r}")


if __name__ == "__main__":
    main()
