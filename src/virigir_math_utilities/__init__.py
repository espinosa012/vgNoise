"""
vgMath - A math library for Python.

Package structure:
    - core: Base classes and enumerations
    - generators: Noise generation algorithms
    - matrix: Matrix data structures
    - utils: Utility functions
    - pathfinding: Pathfinding algorithms (A*, etc.)
"""

# TODO: Fix vgmath imports when the module is properly set up
# # Core
# from vgmath.noise.core import (
#     NoiseGenerator,
#     NoiseType,
#     FractalType,
#     CellularDistanceFunction,
#     CellularReturnType,
#     DomainWarpType,
# )
#
# # Generators
# from vgmath.noise.generators import (
#     NoiseGenerator2D,
#     NOISE_JSON_EXTENSION,
#     PerlinNoise2D,
#     OpenSimplexNoise2D,
#     CellularNoise2D,
#     ValueCubicNoise2D,
#     ValueNoise2D,
#     SimplexSmoothNoise2D,
# )

# Matrix
try:
    from .matrix import Matrix2D
except ImportError:
    Matrix2D = None

# Pathfinding
from .pathfinding import (
    astar,
    astar_grid_2d,
    astar_with_callbacks,
    Manhattan,
    Heuristic,
    PathResult,
)

__all__ = [
    # # Core
    # "NoiseGenerator",
    # "NoiseType",
    # "FractalType",
    # "CellularDistanceFunction",
    # "CellularReturnType",
    # "DomainWarpType",
    # # Generators
    # "NoiseGenerator2D",
    # "NOISE_JSON_EXTENSION",
    # "PerlinNoise2D",
    # "OpenSimplexNoise2D",
    # "CellularNoise2D",
    # "ValueCubicNoise2D",
    # "ValueNoise2D",
    # "SimplexSmoothNoise2D",
    # Matrix
    "Matrix2D",
    # Pathfinding
    "astar",
    "astar_grid_2d",
    "astar_with_callbacks",
    "Manhattan",
    "Heuristic",
    "PathResult",
]
