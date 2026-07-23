"""star_lite_design.utils package initialization.

Enable JAX double precision (float64) for the whole package.

Many modules here (the singular/periodic field lines, the tangent map, the
vessel signed-distance functions, several penalties) do their numerics in JAX,
which defaults to float32 and silently DOWNCASTS float64 inputs to float32 when
``jax_enable_x64`` is off. ``jax_enable_x64`` is a process-global flag that must
be set BEFORE any JAX array is created; importing this package runs __init__
before any submodule, so setting it here guarantees float64 regardless of import
order. Previously only singularperiodicfieldline.py set it (and simsopt.geo /
simsopt.field set it as a side effect), so a module imported in isolation could
silently run in single precision. Setting it in the package __init__ removes that
order dependence.
"""
import jax
jax.config.update("jax_enable_x64", True)
