from pathlib import Path
import sys
import glob
import struct
import xml.etree.ElementTree as ET
from array import array

from vtkmodules.vtkCommonCore import vtkDoubleArray, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkPolyData, vtkRectilinearGrid, vtkStructuredGrid
from vtkmodules.vtkFiltersCore import vtkContourFilter, vtkTubeFilter
from vtkmodules.vtkFiltersGeometry import vtkDataSetSurfaceFilter
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkRenderWindow,
    vtkRenderer,
    vtkWindowToImageFilter,
)

import vtkmodules.vtkRenderingOpenGL2  # noqa: F401


outdir = Path(sys.argv[1])
idx = 'final'

coils_file = outdir / f"curves_opt_{idx}.vtu"
surface_file = outdir / f"surf_opt_0_{idx}.vts"
axis_file = outdir / f"ma_opt_{idx}.vtu"
xpoint_file = outdir / f"xpoint_curves_opt_{idx}.vtu"
vessel_file = outdir / f"vessel_opt_{idx}.vtr"


def parse_vtk_raw(path):
    data = Path(path).read_bytes()
    marker = b'<AppendedData encoding="raw">'
    pos = data.find(marker)
    if pos < 0:
        raise RuntimeError(f"{path} does not contain raw appended VTK XML data")

    start = data.find(b"_", pos + len(marker))
    end = data.rfind(b"</AppendedData>")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"{path} does not contain a valid appended data block")

    root = ET.fromstring(data[:pos] + marker + b"</AppendedData></VTKFile>")
    appended = data[start + 1 : end]
    byte_order = root.attrib.get("byte_order", "LittleEndian")
    header_type = root.attrib.get("header_type", "UInt32")
    return root, appended, byte_order, header_type


def first_data_array(parent):
    if parent is None:
        return None
    return parent.find("DataArray")


def decode_data_array(appended, da, byte_order, header_type):
    if da is None:
        raise RuntimeError("missing DataArray")

    offset = int(da.attrib.get("offset", "0"))
    vtk_type = da.attrib.get("type", "Float64")
    endian = "<" if byte_order.lower().startswith("little") else ">"
    header_fmt = "Q" if header_type == "UInt64" else "I"
    header_nbytes = 8 if header_type == "UInt64" else 4
    nbytes = struct.unpack(endian + header_fmt, appended[offset : offset + header_nbytes])[0]
    raw = appended[offset + header_nbytes : offset + header_nbytes + nbytes]

    typecode = {
        "Float64": "d",
        "Float32": "f",
        "Int64": "q",
        "Int32": "i",
        "UInt64": "Q",
        "UInt32": "I",
        "UInt16": "H",
        "UInt8": "B",
        "Int16": "h",
        "Int8": "b",
    }.get(vtk_type)
    if typecode is None:
        raise RuntimeError(f"Unsupported VTK array type: {vtk_type}")

    arr = array(typecode)
    arr.frombytes(raw)
    if arr.itemsize > 1 and ((sys.byteorder == "little") != byte_order.lower().startswith("little")):
        arr.byteswap()
    return arr


def extent_to_dims(extent):
    x0, x1, y0, y1, z0, z1 = map(int, extent.split())
    return x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1


def make_double_array(values, name=None):
    arr = vtkDoubleArray()
    if name is not None:
        arr.SetName(name)
    arr.SetNumberOfTuples(len(values))
    for i, v in enumerate(values):
        arr.SetValue(i, float(v))
    return arr


def make_points(flat_xyz):
    pts = vtkPoints()
    for i in range(0, len(flat_xyz), 3):
        pts.InsertNextPoint(float(flat_xyz[i]), float(flat_xyz[i + 1]), float(flat_xyz[i + 2]))
    return pts


def poly_actor(polydata, color, opacity=1.0):
    mapper = vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOff()

    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(*color)
    actor.GetProperty().SetOpacity(opacity)
    return actor


def show_vessel_zero_levelset(path):
    root, appended, byte_order, header_type = parse_vtk_raw(path)
    rect = root.find(".//RectilinearGrid")
    piece = root.find(".//Piece")
    coords = root.find(".//Coordinates")
    point_data = root.find(".//PointData")

    if rect is None or piece is None or coords is None:
        raise RuntimeError(f"{path}: not a valid rectilinear grid")

    coord_arrays = coords.findall("DataArray")
    if len(coord_arrays) < 3:
        raise RuntimeError(f"{path}: expected three coordinate arrays")

    x = decode_data_array(appended, coord_arrays[0], byte_order, header_type)
    y = decode_data_array(appended, coord_arrays[1], byte_order, header_type)
    z = decode_data_array(appended, coord_arrays[2], byte_order, header_type)

    sdf_da = point_data.find("DataArray[@Name='sdf']") if point_data is not None else None
    if sdf_da is None:
        sdf_da = first_data_array(point_data)
    sdf = decode_data_array(appended, sdf_da, byte_order, header_type)

    grid = vtkRectilinearGrid()
    grid.SetDimensions(*extent_to_dims(piece.attrib.get("Extent", rect.attrib.get("WholeExtent"))))
    grid.SetXCoordinates(make_double_array(x))
    grid.SetYCoordinates(make_double_array(y))
    grid.SetZCoordinates(make_double_array(z))

    sdf_vtk = make_double_array(sdf, "sdf")
    grid.GetPointData().SetScalars(sdf_vtk)

    contour = vtkContourFilter()
    contour.SetInputData(grid)
    contour.SetInputArrayToProcess(0, 0, 0, 0, "sdf")
    contour.SetValue(0, 0.0)
    contour.Update()

    return poly_actor(contour.GetOutput(), [0.7, 0.7, 0.7], 0.20)


def show_surface(path, color, opacity=1.0):
    root, appended, byte_order, header_type = parse_vtk_raw(path)
    grid = root.find(".//StructuredGrid")
    piece = root.find(".//Piece")
    points_node = root.find(".//Points")

    if grid is None or piece is None or points_node is None:
        raise RuntimeError(f"{path}: not a valid structured grid")

    points_da = first_data_array(points_node)
    pts = decode_data_array(appended, points_da, byte_order, header_type)

    vtk_pts = make_points(pts)
    sgrid = vtkStructuredGrid()
    sgrid.SetDimensions(*extent_to_dims(piece.attrib.get("Extent", grid.attrib.get("WholeExtent"))))
    sgrid.SetPoints(vtk_pts)

    surf = vtkDataSetSurfaceFilter()
    surf.SetInputData(sgrid)
    surf.Update()
    return poly_actor(surf.GetOutput(), color, opacity)


def show_curve_as_tube(path, color, radius=0.005):
    root, appended, byte_order, header_type = parse_vtk_raw(path)
    ugrid = root.find(".//UnstructuredGrid")
    points_node = root.find(".//Points")
    cells_node = root.find(".//Cells")

    if ugrid is None or points_node is None or cells_node is None:
        raise RuntimeError(f"{path}: not a valid unstructured grid")

    points_da = first_data_array(points_node)
    conn_da = cells_node.find("DataArray[@Name='connectivity']")
    offs_da = cells_node.find("DataArray[@Name='offsets']")
    if conn_da is None or offs_da is None:
        raise RuntimeError(f"{path}: missing connectivity/offsets arrays")

    pts = decode_data_array(appended, points_da, byte_order, header_type)
    conn = decode_data_array(appended, conn_da, byte_order, header_type)
    offs = decode_data_array(appended, offs_da, byte_order, header_type)

    vtk_pts = make_points(pts)
    lines = vtkCellArray()
    start = 0
    for end in offs:
        end = int(end)
        ids = conn[start:end]
        lines.InsertNextCell(len(ids))
        for pid in ids:
            lines.InsertCellPoint(int(pid))
        start = end

    poly = vtkPolyData()
    poly.SetPoints(vtk_pts)
    poly.SetLines(lines)

    tube = vtkTubeFilter()
    tube.SetInputData(poly)
    tube.SetRadius(radius)
    tube.SetNumberOfSides(24)
    tube.CappingOn()
    tube.Update()

    return poly_actor(tube.GetOutput(), color, 1.0)


renderer = vtkRenderer()
renderer.SetBackground(1.0, 1.0, 1.0)

renwin = vtkRenderWindow()
renwin.AddRenderer(renderer)
renwin.SetSize(1800, 1400)
renwin.SetOffScreenRendering(1)
renwin.SetAlphaBitPlanes(1)
renwin.SetMultiSamples(0)

renderer.AddActor(show_vessel_zero_levelset(vessel_file))
renderer.AddActor(show_surface(surface_file, [0.1, 0.35, 0.9], 0.55))
renderer.AddActor(show_curve_as_tube(coils_file, [0.85, 0.15, 0.05], 0.006))
renderer.AddActor(show_curve_as_tube(axis_file, [0.0, 0.0, 0.0], 0.004))
renderer.AddActor(show_curve_as_tube(xpoint_file, [0.1, 0.6, 0.1], 0.004))

# Auxiliary coils added by the polish step (one .vtu per xpoint index).
for aux_file in sorted(glob.glob(str(outdir / "aux_coils_*.vtu"))):
    renderer.AddActor(show_curve_as_tube(aux_file, [0.95, 0.6, 0.05], 0.006))

def save_view(png_name, position, view_up, parallel_scale):
    renderer.ResetCamera()
    camera = renderer.GetActiveCamera()
    camera.SetPosition(*position)
    camera.SetFocalPoint(0.0, 0.0, 0.0)
    camera.SetViewUp(*view_up)
    camera.SetParallelProjection(1)
    camera.SetParallelScale(parallel_scale)
    camera.SetClippingRange(0.1, 100.0)

    renwin.Render()

    w2i = vtkWindowToImageFilter()
    w2i.SetInput(renwin)
    w2i.ReadFrontBufferOff()
    w2i.Update()

    writer = vtkPNGWriter()
    writer.SetFileName(str(png_name))
    writer.SetInputConnection(w2i.GetOutputPort())
    writer.Write()


# A slightly larger scale helps avoid cropping of wide structures.
save_view(outdir / "scene_top.png", (0.0, 0.0, 5.0), (0.0, 1.0, 0.0), 1.8)
save_view(outdir / "scene_left.png", (5.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.8)

