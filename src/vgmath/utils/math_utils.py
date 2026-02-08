"""
Mathematical utility functions for noise generation.
"""

from typing import Union
import numpy as np
from numpy.typing import NDArray


# Type alias for values that can be scalar or array
NumericType = Union[np.float64, NDArray[np.float64]]


def fade(t: NumericType) -> NumericType:
    """
    Smoothstep fade function for interpolation.

    Uses the improved Perlin noise fade curve: 6t^5 - 15t^4 + 10t^3

    Args:
        t: Input value(s) in range [0, 1]. Can be scalar or array.

    Returns:
        Smoothed value(s).
    """
    return t * t * t * (t * (t * 6 - 15) + 10)


def lerp(a: NumericType, b: NumericType, t: NumericType) -> NumericType:
    """
    Linear interpolation between two values.

    Args:
        a: Start value(s).
        b: End value(s).
        t: Interpolation factor(s) in range [0, 1].

    Returns:
        Interpolated value(s).
    """
    return a + t * (b - a)
