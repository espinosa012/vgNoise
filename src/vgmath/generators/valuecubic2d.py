"""
Value Cubic Noise 2D implementation with Numba JIT acceleration.

This module implements the Value Cubic noise algorithm for 2D with Numba JIT
compilation for maximum performance. Value Cubic uses Catmull-Rom cubic
interpolation for smoother results than standard Value noise.
Compatible with Godot's FastNoiseLite.
"""

from typing import Optional, Tuple, Sequence
import numpy as np
from numpy.typing import NDArray

from ..core.base import NoiseGenerator
from ..core.enums import FractalType
from .kernels import (
    value_cubic_single_2d,
    value_cubic_fbm_2d,
    value_cubic_fbm_2d_weighted,
    value_cubic_ridged_2d,
    value_cubic_pingpong_2d,
)


class ValueCubicNoise2D(NoiseGenerator):
    """
    2D Value Cubic Noise Generator compatible with Godot FastNoiseLite.

    Value Cubic noise uses cubic (Catmull-Rom) interpolation between
    random values at lattice points, producing smoother results than
    standard Value noise while being faster than Perlin noise.

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
        Initialize the 2D Value Cubic noise generator with Godot-compatible parameters.

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
        """Get the base frequency of the noise."""
        return self._frequency

    @frequency.setter
    def frequency(self, value: float) -> None:
        """Set the base frequency of the noise."""
        self._frequency = value

    @property
    def offset(self) -> Tuple[float, float]:
        """Get the domain offset."""
        return self._offset

    @offset.setter
    def offset(self, value: Tuple[float, float]) -> None:
        """Set the domain offset."""
        self._offset = value

    @property
    def fractal_type(self) -> FractalType:
        """Get the fractal type."""
        return self._fractal_type

    @fractal_type.setter
    def fractal_type(self, value: FractalType) -> None:
        """Set the fractal type."""
        self._fractal_type = value

    @property
    def octaves(self) -> int:
        """Get the number of octaves."""
        return self._octaves

    @octaves.setter
    def octaves(self, value: int) -> None:
        """Set the number of octaves (clamped between 1 and MAX_OCTAVES)."""
        self._octaves = max(1, min(value, self.MAX_OCTAVES))
        self._fractal_bounding = self._calculate_fractal_bounding()

    @property
    def lacunarity(self) -> float:
        """Get the lacunarity (frequency multiplier between octaves)."""
        return self._lacunarity

    @lacunarity.setter
    def lacunarity(self, value: float) -> None:
        """Set the lacunarity."""
        self._lacunarity = value

    @property
    def persistence(self) -> float:
        """Get the persistence (amplitude multiplier between octaves)."""
        return self._persistence

    @persistence.setter
    def persistence(self, value: float) -> None:
        """Set the persistence."""
        self._persistence = value
        self._fractal_bounding = self._calculate_fractal_bounding()

    @property
    def weighted_strength(self) -> float:
        """Get the weighted strength for fractal octaves."""
        return self._weighted_strength

    @weighted_strength.setter
    def weighted_strength(self, value: float) -> None:
        """Set the weighted strength (clamped 0.0-1.0)."""
        self._weighted_strength = max(0.0, min(value, 1.0))

    @property
    def ping_pong_strength(self) -> float:
        """Get the ping-pong strength."""
        return self._ping_pong_strength

    @ping_pong_strength.setter
    def ping_pong_strength(self, value: float) -> None:
        """Set the ping-pong strength."""
        self._ping_pong_strength = value

    @property
    def dimensions(self) -> int:
        """Return the number of dimensions (2 for this generator)."""
        return 2

    def get_value_at(self, position: Tuple[float, ...]) -> np.float64:
        """
        Get the noise value at a specific 2D position.

        Args:
            position: A tuple (x, y) containing the 2D coordinates.

        Returns:
            A noise value normalized to the range [0, 1].

        Raises:
            ValueError: If the position doesn't have exactly 2 elements.
        """
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
        """
        Generate noise using Numba JIT kernels.

        Args:
            x: Flattened array of X coordinates (with offset and frequency applied).
            y: Flattened array of Y coordinates (with offset and frequency applied).

        Returns:
            Array of noise values normalized to [0, 1].
        """
        seed = self.seed if self.seed is not None else 0

        if self._fractal_type == FractalType.NONE:
            return value_cubic_single_2d(x, y, seed)

        elif self._fractal_type == FractalType.FBM:
            if self._weighted_strength > 0:
                return value_cubic_fbm_2d_weighted(
                    x, y, seed,
                    self._octaves,
                    self._lacunarity,
                    self._persistence,
                    self._weighted_strength,
                    self._fractal_bounding
                )
            else:
                return value_cubic_fbm_2d(
                    x, y, seed,
                    self._octaves,
                    self._lacunarity,
                    self._persistence,
                    self._fractal_bounding
                )

        elif self._fractal_type == FractalType.RIDGED:
            return value_cubic_ridged_2d(
                x, y, seed,
                self._octaves,
                self._lacunarity,
                self._persistence,
                self._weighted_strength,
                self._fractal_bounding
            )

        elif self._fractal_type == FractalType.PING_PONG:
            return value_cubic_pingpong_2d(
                x, y, seed,
                self._octaves,
                self._lacunarity,
                self._persistence,
                self._weighted_strength,
                self._ping_pong_strength,
                self._fractal_bounding
            )

        # Fallback to FBM
        return value_cubic_fbm_2d(
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
        """
        Generate noise values over a defined region (optimized).

        Args:
            region: A sequence defining the region to generate. Each element is a tuple
                    of (start, end, num_points) for each dimension.

        Returns:
            A NumPy array of noise values normalized to the range [0, 1].
        """
        if len(region) != self.dimensions:
            raise ValueError(
                f"Region must have {self.dimensions} dimensions, got {len(region)}"
            )

        # Create coordinate arrays
        x_coords = np.linspace(region[0][0], region[0][1], region[0][2], dtype=np.float64)
        y_coords = np.linspace(region[1][0], region[1][1], region[1][2], dtype=np.float64)

        # Create meshgrid and flatten
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        shape = xx.shape

        # Apply offset and frequency
        x_flat = (xx.ravel() + self._offset[0]) * self._frequency
        y_flat = (yy.ravel() + self._offset[1]) * self._frequency

        # Generate noise using JIT kernel
        result = self._generate_noise(x_flat, y_flat)

        return result.reshape(shape)
