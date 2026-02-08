"""
Matrix Editor module - Application for editing VGMatrix2D matrices.

Import components directly to avoid circular imports:
    from matrix_editor.app import MatrixEditor
    from matrix_editor.config import MatrixThemeColors
"""

# Only export config items that don't have dependencies
from .config import (
    MatrixThemeColors,
    MatrixWindowConfig,
    MAX_DISPLAY_SIZE,
    MATRIX_SIZES,
)

__all__ = [
    "MatrixThemeColors",
    "MatrixWindowConfig",
    "MAX_DISPLAY_SIZE",
    "MATRIX_SIZES",
]

