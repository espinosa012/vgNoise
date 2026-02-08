"""
Noise Viewer module - Application for visualizing procedural noise.

Import components directly:
    from noise_viewer.app import NoiseViewer
    from noise_viewer.factory import NoiseGeneratorFactory
"""

# Only export factory items that don't cause circular imports
from .factory import NoiseGeneratorFactory, NoiseParameters

__all__ = [
    "NoiseGeneratorFactory",
    "NoiseParameters",
]

