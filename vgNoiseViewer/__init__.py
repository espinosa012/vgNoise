"""
vgNoiseViewer - Visual tools for noise and matrix manipulation.
This package provides graphical interfaces for:
- Visualizing procedural noise with various algorithms
- Creating and editing VGMatrix2D matrices
- Applying filters and transformations
Submodules:
    - core: Configuration, theme, and shared utilities
    - widgets: Reusable UI components
    - noise_viewer: Noise visualization application
    - matrix_editor: Matrix editing application
    - tests: Unit tests
"""
__version__ = "1.0.0"
# Lazy imports to avoid circular dependencies
def get_noise_viewer():
    """Get the NoiseViewer application class."""
    from .noise_viewer import NoiseViewer
    return NoiseViewer
def get_matrix_editor():
    """Get the MatrixEditor application class."""
    from .matrix_editor import MatrixEditor
    return MatrixEditor
__all__ = [
    "get_noise_viewer",
    "get_matrix_editor",
]
