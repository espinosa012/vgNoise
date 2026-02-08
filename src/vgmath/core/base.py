"""
Base classes and interfaces for noise generators.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Sequence
import numpy as np
from numpy.typing import NDArray


class NoiseGenerator(ABC):
    """
    Abstract base class for all noise generators.

    This class defines the common interface that all noise generators must implement.
    Child classes must implement get_value_at() for sampling noise at specific positions
    and can use generate_region() to generate noise over a defined region.

    Attributes:
        seed: Random seed for reproducible noise generation.
        _rng: NumPy random number generator instance.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the noise generator.

        Args:
            seed: Optional random seed for reproducibility. If None, a random seed is used.
                  Negative seeds are automatically converted to positive values.
        """
        # Convert negative seeds to positive values for NumPy compatibility
        if seed is not None and seed < 0:
            seed = abs(seed)

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the number of dimensions this generator supports."""
        pass

    @abstractmethod
    def get_value_at(self, position: Tuple[float, ...]) -> np.float64:
        """
        Get the noise value at a specific position.

        Args:
            position: A tuple containing the coordinates. The number of elements
                      must match the number of dimensions of the noise generator.

        Returns:
            A noise value normalized to the range [0, 1] as a numpy float64.

        Raises:
            ValueError: If the position tuple length doesn't match the generator's dimensions.
        """
        pass

    def generate_region(
        self,
        region: Sequence[Tuple[float, float, int]]
    ) -> NDArray[np.float64]:
        """
        Generate noise values over a defined region.

        Args:
            region: A sequence defining the region to generate. Each element is a tuple
                    of (start, end, num_points) for each dimension.
                    For example, for 2D noise:
                    [(0.0, 10.0, 100), (0.0, 10.0, 100)]
                    This would generate a 100x100 grid from (0,0) to (10,10).

        Returns:
            A NumPy array of noise values normalized to the range [0, 1].
            The shape of the array corresponds to the number of points in each dimension.

        Raises:
            ValueError: If the region length doesn't match the generator's dimensions.
        """
        if len(region) != self.dimensions:
            raise ValueError(
                f"Region must have {self.dimensions} dimensions, got {len(region)}"
            )

        # Create coordinate arrays for each dimension
        coords = [
            np.linspace(start, end, num_points)
            for start, end, num_points in region
        ]

        # Create a meshgrid for all dimensions
        grids = np.meshgrid(*coords, indexing='ij')

        # Get the shape of the output array
        shape = grids[0].shape

        # Flatten all grids and stack them to create position tuples
        flat_grids = [grid.flatten() for grid in grids]

        # Generate noise values for all positions
        result = np.empty(flat_grids[0].shape, dtype=np.float64)
        for i in range(len(flat_grids[0])):
            position = tuple(grid[i] for grid in flat_grids)
            result[i] = self.get_value_at(position)

        # Reshape to the original grid shape
        return result.reshape(shape)

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the random number generator.

        Args:
            seed: Optional new seed. If None, uses the original seed.
        """
        if seed is not None:
            self.seed = seed
        self._rng = np.random.default_rng(self.seed)
