import numpy as np

from simsopt.geo import FramedCurve, FrameRotation, ZeroRotation, FramedCurveCentroid, FramedCurveFrenet, CurveFilament

def _hex_grid_of_circles(n_layers, radius):
    """
    Compute the centers of a hexagonal grid of circles in 2D.

    Args:
        n_layers: number of layers of filaments around the center spine.
        radius: radius of a filamentary coil.
    Returns:
        shifts_normal: list of shifts in normal direction.
        shifts_binormal: list of shifts in binormal direction.
    """

    n_layers += 1
    v1 = np.array([2*radius, 0])
    v2 = np.array([radius, radius*np.sqrt(3)])
    shifts_normal = []
    shifts_binormal = []
    for m in range(-n_layers+1, n_layers):
        for n in range(-n_layers+1, n_layers):
            layer = max(abs(m), abs(n), abs(-m - n))
            if layer == 0 or layer > n_layers - 1:
                continue
            r_vec = m * v1 + n * v2
            shifts_normal.append(r_vec[0])
            shifts_binormal.append(r_vec[1])
    return shifts_normal, shifts_binormal

def create_hexagonal_filament_grid(curve, n_layers, radius, 
                              rotation_order=None, frame='centroid'):
    """
    Create a finite-build coil approximation using a hexagonal grid
    of filaments around a center spine. The cross section of the winding pack is centered
    around a metal spine. The first layer of filamentals is placed around the spine to form
    a hexagon. Subsequent layers of filaments are placed around the previous layers to
    create a hexagonal grid of filaments. The number of filaments in layer n is 6*n.

    Note that "normal" and "binormal" in the function arguments here
    refer to either the Frenet frame or the "coil centroid
    frame" defined by Singh et al., before rotation.

    Args:
        curve: The underlying curve.
        n_layers: number of layers of filaments around the center spine.
        radius: radius of a filamentary coil.
        rotation_order: Fourier order (maximum mode number) to use in the expression for the rotation
                        of the filament pack. ``None`` means that the rotation is not optimized.
        frame: orthonormal frame to define normal and binormal before rotation (either 'centroid' or 'frenet')
    """
    assert frame in ['centroid', 'frenet']
    assert n_layers >= 1
    assert radius > 0

    shifts_normal, shifts_binormal = _hex_grid_of_circles(n_layers, radius)
    n_total_filaments = len(shifts_normal)

    if rotation_order is None:
        rotation = ZeroRotation(curve.quadpoints)
    else:
        rotation = FrameRotation(curve.quadpoints, rotation_order, scale=1.0)
    if frame == 'frenet':
        framedcurve = FramedCurveFrenet(curve, rotation)
    else:
        framedcurve = FramedCurveCentroid(curve, rotation)

    filaments = []
    for i in range(n_total_filaments):
        filaments.append(CurveFilament(framedcurve, shifts_normal[i], shifts_binormal[i]))
    return filaments


