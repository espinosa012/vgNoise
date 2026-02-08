"""
Noise generator factory for vgNoise Viewer.

This module provides a factory for creating noise generators based on type.
"""

from typing import Dict, Any, Type

from vgmath import (
    PerlinNoise2D,
    OpenSimplexNoise2D,
    CellularNoise2D,
    ValueCubicNoise2D,
    ValueNoise2D,
    SimplexSmoothNoise2D,
    NoiseType,
    FractalType,
    CellularDistanceFunction,
    CellularReturnType,
)


class NoiseGeneratorFactory:
    """Factory for creating noise generators."""

    # Mapping of noise types to generator classes
    _generators: Dict[NoiseType, Type] = {
        NoiseType.PERLIN: PerlinNoise2D,
        NoiseType.SIMPLEX: OpenSimplexNoise2D,
        NoiseType.SIMPLEX_SMOOTH: SimplexSmoothNoise2D,
        NoiseType.CELLULAR: CellularNoise2D,
        NoiseType.VALUE_CUBIC: ValueCubicNoise2D,
        NoiseType.VALUE: ValueNoise2D,
    }

    # Currently implemented noise types
    IMPLEMENTED_TYPES = [
        NoiseType.PERLIN,
        NoiseType.SIMPLEX,
        NoiseType.SIMPLEX_SMOOTH,
        NoiseType.CELLULAR,
        NoiseType.VALUE_CUBIC,
        NoiseType.VALUE,
    ]

    @classmethod
    def create(cls, noise_type: NoiseType, **kwargs) -> Any:
        """
        Create a noise generator of the specified type.

        Args:
            noise_type: The type of noise generator to create.
            **kwargs: Parameters to pass to the generator constructor.

        Returns:
            A noise generator instance.

        Raises:
            ValueError: If the noise type is not implemented.
        """
        if noise_type not in cls._generators:
            raise ValueError(
                f"Noise type {noise_type.name} is not implemented. "
                f"Available types: {[t.name for t in cls.IMPLEMENTED_TYPES]}"
            )

        generator_class = cls._generators[noise_type]
        return generator_class(**kwargs)

    @classmethod
    def create_from_params(cls, params: 'NoiseParameters') -> Any:
        """
        Create a noise generator from a NoiseParameters object.

        Args:
            params: The parameters for the generator.

        Returns:
            A noise generator instance.
        """
        # Base parameters for all noise types
        base_kwargs = {
            'frequency': params.frequency,
            'offset': params.offset,
            'fractal_type': params.fractal_type,
            'octaves': params.octaves,
            'lacunarity': params.lacunarity,
            'persistence': params.persistence,
            'weighted_strength': params.weighted_strength,
            'ping_pong_strength': params.ping_pong_strength,
            'seed': params.seed,
        }

        # Add cellular-specific parameters
        if params.noise_type == NoiseType.CELLULAR:
            base_kwargs['distance_function'] = params.cellular_distance_function
            base_kwargs['return_type'] = params.cellular_return_type
            base_kwargs['jitter'] = params.cellular_jitter

        return cls.create(noise_type=params.noise_type, **base_kwargs)

    @classmethod
    def register(cls, noise_type: NoiseType, generator_class: Type) -> None:
        """
        Register a new noise generator type.

        Args:
            noise_type: The noise type enum value.
            generator_class: The generator class to register.
        """
        cls._generators[noise_type] = generator_class
        if noise_type not in cls.IMPLEMENTED_TYPES:
            cls.IMPLEMENTED_TYPES.append(noise_type)


class NoiseParameters:
    """Container for noise generation parameters."""

    def __init__(
        self,
        noise_type: NoiseType = NoiseType.PERLIN,
        seed: int = 0,
        frequency: float = 0.01,
        offset: tuple = (0.0, 0.0),
        fractal_type: FractalType = FractalType.FBM,
        octaves: int = 5,
        lacunarity: float = 2.0,
        persistence: float = 0.5,
        weighted_strength: float = 0.0,
        ping_pong_strength: float = 2.0,
        # Cellular-specific parameters
        cellular_distance_function: CellularDistanceFunction = CellularDistanceFunction.EUCLIDEAN_SQUARED,
        cellular_return_type: CellularReturnType = CellularReturnType.DISTANCE,
        cellular_jitter: float = 1.0
    ):
        """
        Initialize noise parameters.

        Args:
            noise_type: Type of noise algorithm.
            seed: Random seed.
            frequency: Base frequency.
            offset: Domain offset (x, y).
            fractal_type: Type of fractal combination.
            octaves: Number of octaves.
            lacunarity: Frequency multiplier per octave.
            persistence: Amplitude multiplier per octave.
            weighted_strength: Octave weighting strength.
            ping_pong_strength: Ping-pong effect strength.
            cellular_distance_function: Distance function for cellular noise.
            cellular_return_type: Return type for cellular noise.
            cellular_jitter: Jitter amount for cellular noise feature points.
        """
        self.noise_type = noise_type
        self.seed = seed
        self.frequency = frequency
        self.offset = offset
        self.fractal_type = fractal_type
        self.octaves = octaves
        self.lacunarity = lacunarity
        self.persistence = persistence
        self.weighted_strength = weighted_strength
        self.ping_pong_strength = ping_pong_strength
        # Cellular-specific
        self.cellular_distance_function = cellular_distance_function
        self.cellular_return_type = cellular_return_type
        self.cellular_jitter = cellular_jitter

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            'noise_type': self.noise_type,
            'seed': self.seed,
            'frequency': self.frequency,
            'offset': self.offset,
            'fractal_type': self.fractal_type,
            'octaves': self.octaves,
            'lacunarity': self.lacunarity,
            'persistence': self.persistence,
            'weighted_strength': self.weighted_strength,
            'ping_pong_strength': self.ping_pong_strength,
            'cellular_distance_function': self.cellular_distance_function,
            'cellular_return_type': self.cellular_return_type,
            'cellular_jitter': self.cellular_jitter,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NoiseParameters':
        """Create parameters from dictionary."""
        return cls(**data)

    def copy(self) -> 'NoiseParameters':
        """Create a copy of the parameters."""
        return NoiseParameters(**self.to_dict())
