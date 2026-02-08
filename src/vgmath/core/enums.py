"""
Enumerations for vgNoise library.

This module contains all enum types used across the noise generation library,
compatible with Godot's FastNoiseLite where applicable.
"""

from enum import Enum


class NoiseType(Enum):
    """Supported noise algorithms."""
    PERLIN = 0
    SIMPLEX = 1
    SIMPLEX_SMOOTH = 2
    CELLULAR = 3
    VALUE_CUBIC = 4
    VALUE = 5


class FractalType(Enum):
    """Fractal combination types compatible with Godot FastNoiseLite."""
    NONE = 0        # Single octave, no fractal
    FBM = 1         # Fractal Brownian Motion (standard layering)
    RIDGED = 2      # Ridged multifractal (creates ridge-like features)
    PING_PONG = 3   # Ping-pong effect (creates terraced/banded features)


class CellularDistanceFunction(Enum):
    """Distance functions for cellular/Worley noise."""
    EUCLIDEAN = 0
    EUCLIDEAN_SQUARED = 1
    MANHATTAN = 2
    HYBRID = 3


class CellularReturnType(Enum):
    """Return value types for cellular noise."""
    CELL_VALUE = 0
    DISTANCE = 1
    DISTANCE_2 = 2
    DISTANCE_2_ADD = 3
    DISTANCE_2_SUB = 4
    DISTANCE_2_MUL = 5
    DISTANCE_2_DIV = 6


class DomainWarpType(Enum):
    """Domain warp types compatible with Godot FastNoiseLite."""
    SIMPLEX = 0
    SIMPLEX_REDUCED = 1
    BASIC_GRID = 2
