"""
FastNoise2D — unified 2D noise generator backed by FastNoiseLite-compatible kernels.

This class handles all noise types (OpenSimplex2, OpenSimplex2S, Cellular,
Perlin, ValueCubic, Value) and fractal modes (None, FBM, Ridged, PingPong).
"""

from typing import Optional, Tuple, Sequence, Dict, Any
import numpy as np
from numpy.typing import NDArray

from ..core import (
    NoiseGenerator, NoiseType, FractalType,
    CellularDistanceFunction, CellularReturnType,
)
from .kernels import noise2d_batch, calc_fractal_bounding


class FastNoise2D(NoiseGenerator):
    """
    2D noise generator matching FastNoiseLite output exactly.

    All parameters are forwarded to the Numba-JIT batch kernel.
    Output is normalized from FastNoiseLite's ~[-1, 1] range to [0, 1].
    """

    def __init__(
        self,
        seed: int = 0,
        noise_type: NoiseType = NoiseType.PERLIN,
        frequency: float = 0.01,
        offset: Tuple[float, float] = (0.0, 0.0),
        fractal_type: FractalType = FractalType.FBM,
        octaves: int = 5,
        lacunarity: float = 2.0,
        gain: float = 0.5,
        weighted_strength: float = 0.0,
        ping_pong_strength: float = 2.0,
        cellular_distance_function: CellularDistanceFunction = CellularDistanceFunction.EUCLIDEAN_SQUARED,
        cellular_return_type: CellularReturnType = CellularReturnType.DISTANCE,
        cellular_jitter: float = 1.0,
    ) -> None:
        super().__init__(seed)
        self._noise_type = noise_type
        self._frequency = frequency
        self._offset = offset
        self._fractal_type = fractal_type
        self._octaves = max(1, min(octaves, 9))
        self._lacunarity = lacunarity
        self._gain = gain
        self._weighted_strength = weighted_strength
        self._ping_pong_strength = ping_pong_strength
        self._cellular_dist_func = cellular_distance_function
        self._cellular_return_type = cellular_return_type
        self._cellular_jitter = cellular_jitter
        self._fractal_bounding = calc_fractal_bounding(self._octaves, self._gain)

    @property
    def dimensions(self) -> int:
        return 2

    def get_value_at(self, position: Tuple[float, ...]) -> np.float64:
        x = np.array([position[0] + self._offset[0]], dtype=np.float64)
        y = np.array([position[1] + self._offset[1]], dtype=np.float64)
        raw = self._run_batch(x, y)
        return np.float64(np.clip((raw[0] + 1.0) * 0.5, 0.0, 1.0))

    def get_values_vectorized(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        x_flat = (x.ravel() + self._offset[0]).astype(np.float64)
        y_flat = (y.ravel() + self._offset[1]).astype(np.float64)
        raw = self._run_batch(x_flat, y_flat)
        result = np.clip((raw + 1.0) * 0.5, 0.0, 1.0)
        return result.reshape(x.shape)

    def generate_region(
        self, region: Sequence[Tuple[float, float, int]]
    ) -> NDArray[np.float64]:
        x_coords = np.linspace(region[0][0], region[0][1], region[0][2])
        y_coords = np.linspace(region[1][0], region[1][1], region[1][2])
        xx, yy = np.meshgrid(x_coords, y_coords, indexing='ij')
        return self.get_values_vectorized(xx, yy)

    def _run_batch(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return noise2d_batch(
            x, y,
            seed=int(self.seed) if self.seed is not None else 0,
            frequency=float(self._frequency),
            noise_type=int(self._noise_type.value),
            fractal_type=int(self._fractal_type.value),
            octaves=int(self._octaves),
            lacunarity=float(self._lacunarity),
            gain=float(self._gain),
            weighted_strength=float(self._weighted_strength),
            ping_pong_strength=float(self._ping_pong_strength),
            fractal_bounding=float(self._fractal_bounding),
            dist_func=int(self._cellular_dist_func.value),
            return_type=int(self._cellular_return_type.value),
            jitter=float(self._cellular_jitter),
        )