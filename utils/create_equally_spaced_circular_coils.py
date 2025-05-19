import numpy as np
from simsopt.field import CircularCoil

def create_equally_spaced_circular_coils(ncurves, nfp, stellsym, R0=1.0, R1=0.5, current=1e5):
    """
    Create ``ncurves`` curves of type
    :obj:`~simsopt.field.CircularCoil` that will result in circular equally spaced coils (major
    radius ``R0`` and minor radius ``R1``) after applying
    :obj:`~simsopt.field.coil.coils_via_symmetries`.

    Parameters
    ----------
    ncurves : int
        Number of curves to create.
    nfp : int
        Number of field periods.
    stellsym : bool
        If True, the coils will be stellarator symmetric.
    R0 : float
        Major radius of the coils.
    R1 : float
        Minor radius of the coils.
    current : float
        Current in the coils.

    Returns
    -------
    curves : list
        List of :obj:`~simsopt.field.CircularCoil` objects.
    """
    curves = []
    for i in range(ncurves):
        angle = (i+0.5)*(2*np.pi)/((1+int(stellsym))*nfp*ncurves)
        center = [np.cos(angle)*R0, np.sin(angle)*R0, 0.0]
        normal = [-np.sin(angle)*R0, np.cos(angle)*R0, 0.0]
        curve = CircularCoil(r0=R1, center=center, normal=normal, I=current)
        curve.x = curve.x  # need to do this to transfer data to C++
        curves.append(curve)
    return curves
