"""
Simplex Smooth Noise 2D implementation with Numba JIT acceleration.

This module implements the Simplex Smooth noise algorithm for 2D with Numba JIT
compilation. It's a variant of Simplex noise with a smoother falloff function,
producing higher quality results with fewer artifacts.
Compatible with Godot's FastNoiseLite.
"""

from typing import Optional, Tuple, Sequence
import numpy as np
from numpy.typing import NDArray

from ..core.base import NoiseGenerator
from ..core.enums import FractalType
from .kernels import (
    simplex_smooth_single_2d,
    simplex_smooth_fbm_2d,
    simplex_smooth_fbm_2d_weighted,
    simplex_smooth_ridged_2d,
    simplex_smooth_pingpong_2d,
)


class SimplexSmoothNoise2D(NoiseGenerator):
    """
    2D Simplex Smooth Noise Generator compatible with Godot FastNoiseLite.

    Simplex Smooth is a variant of Simplex noise that uses a larger falloff
    radius (0.6 instead of 0.5) and more gradient directions, resulting in
    smoother, higher quality noise with fewer directional artifacts.

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
        _permutation: Permutation table for hash function.
    """

    PERM_SIZE = 256
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
        Initialize the 2D Simplex Smooth noise generator.

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

        # Initialize permutation table
        self._init_permutation_table()

        # Precompute fractal bounding for normalization
        self._fractal_bounding = self._calculate_fractal_bounding()

    def _init_permutation_table(self) -> None:
        """Initialize the permutation table based on seed."""
        rng = np.random.default_rng(self.seed if self.seed is not None else 0)
        perm = np.arange(self.PERM_SIZE, dtype=np.int16)
        rng.shuffle(perm)
        # Duplicate for wrapping
        self._permutation = np.concatenate([perm, perm]).astype(np.int16)

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
        if self._fractal_type == FractalType.NONE:
            return simplex_smooth_single_2d(x, y, self._permutation)

        elif self._fractal_type == FractalType.FBM:
            if self._weighted_strength > 0:
                return simplex_smooth_fbm_2d_weighted(
                    x, y, self._permutation,
                    self._octaves,
                    self._lacunarity,
                    self._persistence,
                    self._weighted_strength,
                    self._fractal_bounding
                )
            else:
                return simplex_smooth_fbm_2d(
                    x, y, self._permutation,
                    self._octaves,
                    self._lacunarity,
                    self._persistence,
                    self._fractal_bounding
                )

        elif self._fractal_type == FractalType.RIDGED:
            return simplex_smooth_ridged_2d(
                x, y, self._permutation,
                self._octaves,
                self._lacunarity,
                self._persistence,
                self._weighted_strength,
                self._fractal_bounding
            )

        elif self._fractal_type == FractalType.PING_PONG:
            return simplex_smooth_pingpong_2d(
                x, y, self._permutation,
                self._octaves,
                self._lacunarity,
                self._persistence,
                self._weighted_strength,
                self._ping_pong_strength,
                self._fractal_bounding
            )

        # Fallback
        return simplex_smooth_fbm_2d(
            x, y, self._permutation,
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
