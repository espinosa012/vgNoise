"""
Value Noise 2D implementation with Numba JIT acceleration.

This module implements Value noise algorithm for 2D with Numba JIT compilation.
Value noise uses random values at lattice points with bilinear interpolation,
making it simpler and faster than Perlin or Value Cubic noise.
Compatible with Godot's FastNoiseLite.
"""

from typing import Optional, Tuple, Sequence
import numpy as np
from numpy.typing import NDArray

from ..core.base import NoiseGenerator
from ..core.enums import FractalType
from .kernels import (
    value_single_2d,
    value_fbm_2d,
    value_fbm_2d_weighted,
    value_ridged_2d,
    value_pingpong_2d,
)


class ValueNoise2D(NoiseGenerator):
    """
    2D Value Noise Generator compatible with Godot FastNoiseLite.

    Value noise assigns random values to lattice points and interpolates
    between them using bilinear interpolation with a quintic fade function.
    This is simpler and faster than gradient-based noise like Perlin.

    Uses Numba JIT compilation for high-performance noise generation.

    Attributes:
        seed: Random seed for reproducible noise generation.
        _frequency: Base frequency for the first octave.
        _offset: Domain offset (x, y) applied before sampling.
        _fractal_type: Type of fractal combination (NONE, FBM, RIDGED, PING_PONG).
        _octaves: Number of noise layers to sample (1-9).
        _lacunarity: Factor by which frequency increases for each successive octave.
        _persistence: Factor by which amplitude decreases for each successive octave.
        _weighted_strength: Strength of octave weighting based on previous octave's value.
        _ping_pong_strength: Strength of the ping-pong effect.
    """

    MAX_OCTAVES = 9

    def __init__(
        self,
        frequency: float = 0.01,
        offset: Tuple[float, float] = (0.0, 0.0),
        fractal_type: FractalType = FractalType.FBM,
        octaves: int = 5,
        lacunarity: float = 2.0,
        persistence: float = 0.5,
        weighted_strength: float = 0.0,
        ping_pong_strength: float = 2.0,
        seed: Optional[int] = None
    ) -> None:
        """
        Initialize the 2D Value noise generator with Godot-compatible parameters.

        Args:
            frequency: Base frequency. Higher values = more detail. Default 0.01.
            offset: Domain offset (x, y) applied before noise sampling.
            fractal_type: Type of fractal combination (NONE, FBM, RIDGED, PING_PONG).
            octaves: Number of noise layers to sample (clamped 1-9). Default 5.
            lacunarity: Frequency multiplier between octaves. Default 2.0.
            persistence: Amplitude multiplier between octaves (gain). Default 0.5.
            weighted_strength: Octave weighting strength (0.0-1.0). Default 0.0.
            ping_pong_strength: Ping-pong effect strength. Default 2.0.
            seed: Optional random seed for reproducibility.
        """
        super().__init__(seed)

        self._frequency = frequency
        self._offset = offset
        self._fractal_type = fractal_type
        self._octaves = max(1, min(octaves, self.MAX_OCTAVES))
        self._lacunarity = lacunarity
        self._persistence = persistence
        self._weighted_strength = max(0.0, min(weighted_strength, 1.0))
        self._ping_pong_strength = ping_pong_strength

        # Precompute fractal bounding for normalization
        self._fractal_bounding = self._calculate_fractal_bounding()

    def _calculate_fractal_bounding(self) -> float:
        """Calculate the fractal bounding value for normalization."""
        gain = abs(self._persistence)
        amp = gain
        amp_fractal = 1.0

        for _ in range(1, self._octaves):
            amp_fractal += amp
            amp *= gain

        return 1.0 / amp_fractal

    # Properties
    @property
    def frequency(self) -> float:
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        self._frequency = value

    @property
    def offset(self) -> Tuple[float, float]:
        return self._offset

    @offset.setter
    def offset(self, value: Tuple[float, float]) -> None:
        self._offset = value

    @property
    def fractal_type(self) -> FractalType:
        return self._fractal_type

    @fractal_type.setter
    def fractal_type(self, value: FractalType) -> None:
        self._fractal_type = value

    @property
    def octaves(self) -> int:
        return self._octaves

    @octaves.setter
    def octaves(self, value: int) -> None:
        self._octaves = max(1, min(value, self.MAX_OCTAVES))
        self._fractal_bounding = self._calculate_fractal_bounding()

    @property
    def lacunarity(self) -> float:
        return self._lacunarity

    @lacunarity.setter
    def lacunarity(self, value: float) -> None:
        self._lacunarity = value

    @property
    def persistence(self) -> float:
        return self._persistence

    @persistence.setter
    def persistence(self, value: float) -> None:
        self._persistence = value
        self._fractal_bounding = self._calculate_fractal_bounding()

    @property
    def weighted_strength(self) -> float:
        return self._weighted_strength

    @weighted_strength.setter
    def weighted_strength(self, value: float) -> None:
        self._weighted_strength = max(0.0, min(value, 1.0))

    @property
    def ping_pong_strength(self) -> float:
        return self._ping_pong_strength

    @ping_pong_strength.setter
    def ping_pong_strength(self, value: float) -> None:
        self._ping_pong_strength = value

    @property
    def dimensions(self) -> int:
        return 2

    def get_value_at(self, position: Tuple[float, ...]) -> np.float64:
        if len(position) != 2:
            raise ValueError(f"Position must have 2 elements, got {len(position)}")

        x = np.array([position[0] + self._offset[0]], dtype=np.float64) * self._frequency
        y = np.array([position[1] + self._offset[1]], dtype=np.float64) * self._frequency

        result = self._generate_noise(x, y)
        return np.float64(result[0])

    def _generate_noise(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Generate noise using Numba JIT kernels."""
        seed = self.seed if self.seed is not None else 0

        if self._fractal_type == FractalType.NONE:
            return value_single_2d(x, y, seed)

        elif self._fractal_type == FractalType.FBM:
            if self._weighted_strength > 0:
                return value_fbm_2d_weighted(
                    x, y, seed,
                    self._octaves,
                    self._lacunarity,
                    self._persistence,
                    self._weighted_strength,
                    self._fractal_bounding
                )
            else:
                return value_fbm_2d(
                    x, y, seed,
                    self._octaves,
                    self._lacunarity,
                    self._persistence,
                    self._fractal_bounding
                )

        elif self._fractal_type == FractalType.RIDGED:
            return value_ridged_2d(
                x, y, seed,
                self._octaves,
                self._lacunarity,
                self._persistence,
                self._weighted_strength,
                self._fractal_bounding
            )

        elif self._fractal_type == FractalType.PING_PONG:
            return value_pingpong_2d(
                x, y, seed,
                self._octaves,
                self._lacunarity,
                self._persistence,
                self._weighted_strength,
                self._ping_pong_strength,
                self._fractal_bounding
            )

        # Fallback
        return value_fbm_2d(
            x, y, seed,
            self._octaves,
            self._lacunarity,
            self._persistence,
            self._fractal_bounding
        )

    def generate_region(
        self,
        region: Sequence[Tuple[float, float, int]]
    ) -> NDArray[np.float64]:
        if len(region) != self.dimensions:
            raise ValueError(
                f"Region must have {self.dimensions} dimensions, got {len(region)}"
            )

        x_coords = np.linspace(region[0][0], region[0][1], region[0][2], dtype=np.float64)
        y_coords = np.linspace(region[1][0], region[1][1], region[1][2], dtype=np.float64)

        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        shape = xx.shape

        x_flat = (xx.ravel() + self._offset[0]) * self._frequency
        y_flat = (yy.ravel() + self._offset[1]) * self._frequency

        result = self._generate_noise(x_flat, y_flat)
        return result.reshape(shape)
