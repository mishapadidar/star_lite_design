from star_lite_design.continuous_winding_model.field_models import (
    load_design, filament_coils, hex_build_coils, continuous_coils, scale_to_axis_field,
    design_surface_full_torus, boozer, continuation, qs_error, outdir)
from simsopt.geo import SurfaceXYZTensorFourier, BoozerSurface, Volume
import numpy as np
import argparse, glob, json, os, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

"""
Convergence of the exact Boozer surface with its resolution, for each coil model.

The resolution is mpol = ntor = k per field period, as in the design (k = 10). The
continuous winding has no symmetry, so every model is solved on the full torus (nfp 1,
no stellarator symmetry), where k per field period is mpol = k, ntor = 2k, on
(4k + 1) x (2k + 1) collocation points, at the volume of the design's surface:
- filament: started from the design's surface,
- filament, design's solve: the design's own solve for comparison, on one field period
  with stellarator symmetry, mpol = ntor = k on (2k + 1) x (2k + 1) points (k = 10 is
  the design's surface),
- hex build: started from the design's surface, or by continuation from the filament
  at the same k if that fails,
- continuous: by continuation from the hex build at the same k.
Currents scaled to 0.0875 T on each model's axis (field_models.py).

With --from_k, every model is instead started from its own surface at resolution
from_k, fitted at resolution k. At k = 10 the collocation equations have a second,
spurious solution next to the true one (a large Boozer residual between the
collocation points), which is the design's stored surface; started from it, the solve
stays on it. Started from the k = 11 surfaces it finds the true one.

Run one or more k per process, then plot:
    python resolution_study.py --k 10 11 12
    python resolution_study.py --k 10 --from_k 11
    python resolution_study.py --plot
Writes output/designA/resolution/config{c}_k{k}[_from{from_k}].json (and the surfaces
in .npz) and output/designA/resolution_study_config{c}.png.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=int, default=0, help="current configuration (0, 1, 2)")
parser.add_argument("--k", type=int, nargs="*", default=[], help="mpol = ntor per field period")
parser.add_argument("--from_k", type=int, default=None, help="start from the surfaces found at this resolution")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

resdir = outdir + "resolution/"
os.makedirs(resdir, exist_ok=True)
models = ["filament", "filament, design's solve", "hex build", "continuous"]


def fit(x, surf_from, surf_to):
    """Dofs of surf_to fitted to the surface with dofs x and the shape of surf_from, on surf_to's points."""
    on_grid = SurfaceXYZTensorFourier(mpol=surf_from.mpol, ntor=surf_from.ntor, nfp=surf_from.nfp, stellsym=surf_from.stellsym,
                                      quadpoints_phi=surf_to.quadpoints_phi, quadpoints_theta=surf_to.quadpoints_theta)
    on_grid.x = x
    surf_to.least_squares_fit(on_grid.gamma())
    return surf_to.x.copy()


def full_torus_shape(k):
    return SurfaceXYZTensorFourier(mpol=k, ntor=2 * k, nfp=1, stellsym=False, quadpoints_phi=np.linspace(0, 1, 4 * k + 1, endpoint=False),
                                   quadpoints_theta=np.linspace(0, 1, 2 * k + 1, endpoint=False))


def solve_all(k, from_k=None):
    bsurf, (iota0, G0), ma, xp = load_design(args.config)
    s, label = bsurf.surface, bsurf.targetlabel
    coils = bsurf.biotsavart.coils
    bs = {}
    for name, build in [("filament", filament_coils), ("hex build", hex_build_coils), ("continuous", continuous_coils)]:
        bs[name], _, ok, factor = scale_to_axis_field(build(coils), ma)
        assert ok, name
    mpol, ntor = k, 2 * k

    def solve(b, x, iota, G):
        return boozer(b, x, iota, G, mpol, ntor, label)

    results, xs = {}, {}

    def record(name, res, surf, b, t, **extra):
        ok = res is not None and bool(res["success"])
        results[name] = dict(iota=float(res["iota"]) if ok else None, G=float(res["G"]) if ok else None, success=ok,
                             qs=float(qs_error(surf, b, sDIM=20 * s.nfp // surf.nfp)) if ok else None,
                             time=time.time() - t, **extra)
        if ok:
            xs[name] = surf.x.copy()
        print(f"k {k:2d} {name:25s}: iota {results[name]['iota']}, converged {ok} ({results[name]['time']:.0f} s)", flush=True)

    # the design's own solve: one field period, stellarator symmetric
    t = time.time()
    phis = np.linspace(0, 1 / s.nfp, 2 * k + 1, endpoint=False)
    thetas = np.linspace(0, 1, 2 * k + 1, endpoint=False)
    s_eval = SurfaceXYZTensorFourier(mpol=s.mpol, ntor=s.ntor, nfp=s.nfp, stellsym=True, quadpoints_phi=phis, quadpoints_theta=thetas)
    s_eval.x = s.x
    sym = SurfaceXYZTensorFourier(mpol=k, ntor=k, nfp=s.nfp, stellsym=True, quadpoints_phi=phis, quadpoints_theta=thetas)
    sym.least_squares_fit(s_eval.gamma())
    G_fil = G0 * bs["filament"].coils[0].current.get_value() / coils[0].current.get_value()
    if from_k is not None:
        # start every model from its own surface at resolution from_k
        prev = json.load(open(resdir + f"config{args.config}_k{from_k}.json"))["models"]
        prev_x = np.load(resdir + f"config{args.config}_k{from_k}.npz")
        sym_from = SurfaceXYZTensorFourier(mpol=from_k, ntor=from_k, nfp=s.nfp, stellsym=True)
        sym.x = fit(prev_x["filament, design's solve"], sym_from, sym)
        p = prev["filament, design's solve"]
        res = BoozerSurface(bs["filament"], sym, Volume(sym), label, options=dict(verbose=False)).run_code(iota=p["iota"], G=p["G"])
        record("filament, design's solve", res, sym, bs["filament"], t, start=f"from k = {from_k}")
        for name in ["filament", "hex build", "continuous"]:
            t = time.time()
            x = fit(prev_x[name], full_torus_shape(from_k), full_torus_shape(k))
            res, surf = solve(bs[name], x, prev[name]["iota"], prev[name]["G"])
            record(name, res, surf, bs[name], t, start=f"from k = {from_k}")
        with open(resdir + f"config{args.config}_k{k}_from{from_k}.json", "w") as f:
            json.dump({"k": k, "from k": from_k, "mpol": mpol, "ntor (full torus)": ntor, "models": results}, f, indent=2)
        np.savez(resdir + f"config{args.config}_k{k}_from{from_k}.npz", **xs)
        return

    res = BoozerSurface(bs["filament"], sym, Volume(sym), label, options=dict(verbose=False)).run_code(iota=iota0, G=G_fil)
    record("filament, design's solve", res, sym, bs["filament"], t)

    # full torus
    x0 = design_surface_full_torus(s, mpol, ntor).x
    t = time.time()
    res, surf = solve(bs["filament"], x0, iota0, G_fil)
    record("filament", res, surf, bs["filament"], t)

    t = time.time()
    res, surf = solve(bs["hex build"], x0, iota0, G_fil)
    how = "from the design's surface"
    if not res["success"] and results["filament"]["success"]:
        f = results["filament"]
        last, path = continuation(bs["filament"].coils, bs["hex build"].coils, xs["filament"], f["iota"], f["G"], solve)
        res, surf = last if last is not None and path[-1] == 1 else (None, None)
        how = "by continuation from the filament"
    record("hex build", res, surf, bs["hex build"], t, start=how)

    t = time.time()
    if results["hex build"]["success"]:
        h = results["hex build"]
        last, path = continuation(bs["hex build"].coils, bs["continuous"].coils, xs["hex build"], h["iota"], h["G"], solve)
        res, surf = last if last is not None and path[-1] == 1 else (None, None)
        record("continuous", res, surf, bs["continuous"], t, continuation=path)
    else:
        record("continuous", None, None, None, t, continuation=[])

    with open(resdir + f"config{args.config}_k{k}.json", "w") as f:
        json.dump({"k": k, "mpol": mpol, "ntor (full torus)": ntor, "models": results}, f, indent=2)
    np.savez(resdir + f"config{args.config}_k{k}.npz", **xs)


for k in args.k:
    solve_all(k, args.from_k)

if args.plot:
    files = glob.glob(resdir + f"config{args.config}_k*.json")
    main = {d["k"]: d for d in (json.load(open(f)) for f in files if "_from" not in f)}
    restart = {d["k"]: d for d in (json.load(open(f)) for f in files if "_from" in f)}
    ks = np.array(sorted(main))
    style = {"filament": dict(color="#2a78d6", ls="-", marker="o"),
             "filament, design's solve": dict(color="#2a78d6", ls="--", marker="o", mfc="white"),
             "hex build": dict(color="#eb6834", ls="-", marker="s"),
             "continuous": dict(color="#1baf7a", ls="-", marker="D")}

    def iota_of(d, name):
        m = d["models"][name]
        return m["iota"] if m["success"] else np.nan

    fig, axs = plt.subplots(2, 1, figsize=(8.5, 8.5), sharex=True)
    failed, spurious = [], False
    for name in models:
        # where the solve was repeated from the next resolution's surface, that is the solution
        iota = np.array([iota_of(restart[k], name) if k in restart else iota_of(main[k], name) for k in ks], dtype=float)
        failed += [f"{name} at k = {k}" for k, i in zip(ks, iota) if np.isnan(i)]
        ref = iota[~np.isnan(iota)][-1]
        axs[0].plot(ks, iota, lw=2, ms=6, label=name, **style[name])
        err = np.abs(iota - ref)
        axs[1].semilogy(ks[:-1], np.where(err[:-1] > 0, err[:-1], np.nan), lw=2, ms=6, label=name, **style[name])
        for k in restart:
            other = iota_of(main[k], name)
            if abs(other - iota[ks == k][0]) > 1e-7:
                for ax, y in zip(axs, [other, abs(other - ref)]):
                    ax.plot([k], [y], ls="none", marker=style[name]["marker"], ms=9, mfc="white", mec=style[name]["color"], mew=1.5)
                spurious = True
    axs[0].axvline(10, color="0.6", lw=1, ls=":")
    axs[0].text(10.1, 0.5, "design's\nresolution", color="0.35", fontsize=8, transform=axs[0].get_xaxis_transform())
    axs[0].set_ylabel("iota")
    if spurious:
        axs[0].plot([], [], ls="none", marker="o", ms=9, mfc="white", mec="0.4", mew=1.5,
                    label=f"k = {min(restart)} started from the design's stored surface:\na spurious solution "
                          "(continuous: no converged surface)")
    axs[0].legend(fontsize=8, frameon=False, loc="center right")
    restarted = ", ".join(f"k = {k} started from the k = {restart[k]['from k']} surfaces" for k in sorted(restart))
    axs[0].set_title("Exact Boozer surface at the design's volume, design A, current configuration "
                     f"{args.config}\nmpol = ntor = k per field period (full torus: mpol = k, ntor = 2k)"
                     + (f"; {restarted}" if restarted else ""), fontsize=10)
    axs[1].set_ylabel(f"|iota(k) - iota(k = {ks[-1]})|")
    axs[1].set_xlabel("k")
    if failed:
        axs[1].text(0.01, 0.02, "no converged surface: " + "; ".join(failed), fontsize=7, color="0.35", transform=axs[1].transAxes)
    for ax in axs:
        ax.grid(True, color="0.9", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir + f"resolution_study_config{args.config}.png", dpi=130)
