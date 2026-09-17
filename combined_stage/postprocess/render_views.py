#!/usr/bin/env python3
"""Isometric, top, left and right renders of a device folder, with array/mk_paraview.py's actors and colours.

usage (envs/vtkrender, headless): VTK_DEFAULT_OPENGL_WINDOW=vtkEGLRenderWindow python render_views.py <device dir> [out dir]

Draws the vessel (if any), the optimization surface (one field period, as in the campaign renders), the coils, the
magnetic axis and the X-point line, and writes scene_iso.png, scene_top.png, scene_left.png and scene_right.png into
the out dir (default: the device folder). Cameras look at the origin with orthographic projection: iso from (3, 3, 3), top from +z, left
from +x (the campaign's scene_left), right from -x. Each view is fitted to the visible geometry (render_iso.py's
fit), so large devices are not cropped.
"""
import math
import sys
from pathlib import Path

MK_PARAVIEW = Path(__file__).resolve().parents[2] / "array" / "mk_paraview.py"
CUT = "renderer = vtkRenderer()"          # mk_paraview.py builds its scene from this line on
SIZE = (1800, 1400)
VIEWS = {"iso": ((3.0, 3.0, 3.0), (0.0, 0.0, 1.0)),
         "top": ((0.0, 0.0, 5.0), (0.0, 1.0, 0.0)),
         "left": ((5.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
         "right": ((-5.0, 0.0, 0.0), (0.0, 0.0, 1.0))}


def load_helpers(device_dir):
    """Exec mk_paraview.py's imports, file paths and show_* helpers (everything before the scene) for device_dir."""
    text = MK_PARAVIEW.read_text()
    prefix = text[:text.index(CUT)]
    ns = {"__name__": "__mk_paraview_prefix__", "__file__": str(MK_PARAVIEW)}
    argv, sys.argv = sys.argv, ["mk_paraview.py", str(device_dir)]
    try:
        exec(compile(prefix, str(MK_PARAVIEW), "exec"), ns)
    finally:
        sys.argv = argv
    return ns


def fit_parallel_scale(renderer, position, view_up, margin=1.05):
    """Half-height that keeps the visible bounds' 8 corners in frame (exact for orthographic projection)."""
    b = renderer.ComputeVisiblePropBounds()

    def norm(v):
        n = math.sqrt(sum(c * c for c in v))
        return [c / n for c in v]

    fwd = norm([-c for c in position])
    dot = sum(u * f for u, f in zip(view_up, fwd))
    up = norm([view_up[i] - dot * fwd[i] for i in range(3)])
    right = norm([fwd[1] * up[2] - fwd[2] * up[1], fwd[2] * up[0] - fwd[0] * up[2], fwd[0] * up[1] - fwd[1] * up[0]])
    half_h = half_w = 0.0
    for x in b[0:2]:
        for y in b[2:4]:
            for z in b[4:6]:
                half_h = max(half_h, abs(x * up[0] + y * up[1] + z * up[2]))
                half_w = max(half_w, abs(x * right[0] + y * right[1] + z * right[2]))
    return max(half_h, half_w * SIZE[1] / SIZE[0]) * margin


def main():
    device_dir = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else device_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ns = load_helpers(device_dir)
    renderer = ns["vtkRenderer"]()
    renderer.SetBackground(1.0, 1.0, 1.0)
    renwin = ns["vtkRenderWindow"]()
    renwin.AddRenderer(renderer)
    renwin.SetSize(*SIZE)
    renwin.SetOffScreenRendering(1)
    renwin.SetMultiSamples(8)

    if ns["vessel_file"].exists():
        renderer.AddActor(ns["show_vessel_zero_levelset"](ns["vessel_file"]))
    renderer.AddActor(ns["show_surface"](ns["surface_file"], [0.1, 0.35, 0.9], 0.55))
    renderer.AddActor(ns["show_curve_as_tube"](ns["coils_file"], [0.85, 0.15, 0.05], 0.006))
    renderer.AddActor(ns["show_curve_as_tube"](ns["axis_file"], [0.0, 0.0, 0.0], 0.004))
    if ns["xpoint_file"].exists():
        renderer.AddActor(ns["show_curve_as_tube"](ns["xpoint_file"], [0.1, 0.6, 0.1], 0.004))

    for name, (position, view_up) in VIEWS.items():
        camera = renderer.GetActiveCamera()
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetPosition(*position)
        camera.SetViewUp(*view_up)
        camera.SetParallelProjection(1)
        camera.SetParallelScale(fit_parallel_scale(renderer, position, view_up))
        renderer.ResetCameraClippingRange()
        renwin.Render()
        grab = ns["vtkWindowToImageFilter"]()
        grab.SetInput(renwin)
        grab.ReadFrontBufferOff()
        grab.Update()
        writer = ns["vtkPNGWriter"]()
        out = out_dir / f"scene_{name}.png"
        writer.SetFileName(str(out))
        writer.SetInputConnection(grab.GetOutputPort())
        writer.Write()
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
