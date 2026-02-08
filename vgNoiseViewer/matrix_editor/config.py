"""
Configuration classes and constants for Matrix Editor App.

This module contains dataclasses for parameter configuration and
application constants specific to the matrix editor.
"""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class MatrixThemeColors:
    """Color scheme for the matrix editor application."""
    background: str = "#1e1e1e"
    foreground: str = "#ffffff"
    card: str = "#2d2d2d"
    accent: str = "#4a9eff"
    accent_hover: str = "#5aafff"
    muted: str = "#888888"
    success: str = "#4caf50"
    warning: str = "#ff9800"
    error: str = "#f44336"
    transparent_pattern: str = "#404040"  # Checkerboard pattern for transparent


@dataclass(frozen=True)
class MatrixWindowConfig:
    """Window configuration for matrix editor."""
    title: str = "VGMatrix2D Editor"
    width: int = 1200
    height: int = 800
    min_width: int = 900
    min_height: int = 600


@dataclass
class FilterParameterConfig:
    """Configuration for a filter parameter."""
    name: str
    label: str
    param_type: str  # "int", "float", "choice", "bool"
    default: any
    min_value: float = None
    max_value: float = None
    step: float = None
    choices: List[str] = None

    def __post_init__(self):
        if self.choices is None:
            self.choices = []


# Default matrix sizes available
MATRIX_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]

# Maximum display size for the image preview
MAX_DISPLAY_SIZE = 600

# Cell size for direct matrix editing (in pixels)
EDIT_CELL_SIZE = 40

# Maximum matrix size for direct cell editing (larger matrices use image-only view)
MAX_EDITABLE_SIZE = 32

# Supported image formats for import
SUPPORTED_IMAGE_FORMATS = [
    ("PNG files", "*.png"),
    ("JPEG files", "*.jpg *.jpeg"),
    ("All image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
    ("All files", "*.*"),
]

# Export formats
EXPORT_FORMATS = [
    ("PNG files", "*.png"),
    ("JPEG files", "*.jpg"),
    ("NumPy array", "*.npy"),
]

