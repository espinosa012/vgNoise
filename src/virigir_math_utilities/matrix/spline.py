"""
Spline-based value remapping for Matrix2D.

Provides functions to remap matrix values through a curve defined by
a set of control points, using cubic spline interpolation.
"""

from typing import List, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from virigir_math_utilities.matrix.matrix2d import Matrix2D


def apply_spline_points(
    spline_points: List[Tuple[float, float]],
    matrix: Matrix2D,
) -> Matrix2D:
    """
    Remap every value in *matrix* through a cubic spline defined by
    *spline_points* and return a new Matrix2D with the results.

    Each point in *spline_points* is an (input, output) pair that the
    curve must pass through. Values between control points are
    interpolated with a natural cubic spline; values outside the range
    of control points are extrapolated following the spline.

    The operation is fully vectorised (no Python loops over cells) and
    works directly on the internal NumPy arrays for maximum performance.

    Args:
        spline_points: Control points as ``[(x0, y0), (x1, y1), ...]``.
                        Must contain **at least 2 points** and the x
                        values must be **strictly increasing** (they will
                        be sorted automatically if they are not).
        matrix:        Source Matrix2D whose values will be remapped.
                        Values are expected in [0, 1] but the function
                        works with any range.

    Returns:
        A new Matrix2D with the same shape and mask as *matrix*, where
        each assigned value has been remapped through the spline curve.

    Raises:
        ValueError: If fewer than 2 control points are provided or if
                    there are duplicate x values.

    Example::

        >>> from virigir_math_utilities.matrix import Matrix2D
        >>> m = Matrix2D((4, 4), 0.5)
        >>> # identity curve (no change)
        >>> result = apply_spline_points([(0, 0), (1, 1)], m)
        >>> result.get_value_at(0, 0)
        0.5
        >>> # boost mid-tones
        >>> result = apply_spline_points([(0, 0), (0.5, 0.8), (1, 1)], m)
    """
    # ------------------------------------------------------------------ #
    # Validate
    # ------------------------------------------------------------------ #
    if len(spline_points) < 2:
        raise ValueError(
            f"At least 2 spline points are required, got {len(spline_points)}."
        )

    # Sort by x to guarantee monotonicity
    sorted_points = sorted(spline_points, key=lambda p: p[0])
    xs = np.array([p[0] for p in sorted_points], dtype=np.float64)
    ys = np.array([p[1] for p in sorted_points], dtype=np.float64)

    # Check for duplicate x values (CubicSpline requires strictly increasing)
    if np.any(np.diff(xs) <= 0):
        raise ValueError(
            "Spline control points must have strictly increasing x values. "
            "Duplicate or non-monotonic x values were found after sorting."
        )

    # ------------------------------------------------------------------ #
    # Build spline (natural boundary: second derivative = 0 at ends)
    # ------------------------------------------------------------------ #
    spline = CubicSpline(xs, ys, bc_type="natural")

    # ------------------------------------------------------------------ #
    # Apply – operate on raw numpy arrays, avoid per-cell Python calls
    # ------------------------------------------------------------------ #
    src_data = matrix.data       # read-only np view
    src_mask = matrix.mask       # read-only np view

    # Evaluate the spline for *all* cells at once (vectorised)
    remapped_data = spline(src_data)

    # Preserve unassigned cells: copy zeros where mask is False
    out_data = np.where(src_mask, remapped_data, 0.0)

    return Matrix2D.from_numpy(out_data, mask=src_mask)

