"""
vgNoise Matrix - Matrix data structures for numerical operations.
"""

from .matrix2d import Matrix2D
from .filters import MatrixFilters, BlurType, EdgeDetectionType
from .spline import apply_spline_points

__all__ = [
    "Matrix2D",
    "MatrixFilters",
    "BlurType",
    "EdgeDetectionType",
    "apply_spline_points",
]
