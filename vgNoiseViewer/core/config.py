"""
Configuration classes and constants for vgNoise Viewer.

This module contains dataclasses for parameter configuration and
application constants.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ThemeColors:
    """Color scheme for the application."""
    background: str = "#1e1e1e"
    foreground: str = "#ffffff"
    card: str = "#2d2d2d"
    accent: str = "#4a9eff"
    accent_hover: str = "#5aafff"
    muted: str = "#888888"


@dataclass(frozen=True)
class WindowConfig:
    """Window configuration."""
    title: str = "vgNoise Viewer"
    width: int = 900
    height: int = 650
    min_width: int = 800
    min_height: int = 600


@dataclass
class ParameterConfig:
    """Configuration for a single parameter control."""
    name: str
    label: str
    default: float
    min_value: float
    max_value: float
    step: float = 0.01
    format_str: str = "{:.2f}"
    control_type: str = "stepper"  # stepper, spinbox, combobox

    def validate(self, value: float) -> float:
        """Validate and clamp value to valid range."""
        return max(self.min_value, min(self.max_value, value))

    def round_to_step(self, value: float) -> float:
        """Round value to nearest step."""
        return round(value / self.step) * self.step


@dataclass
class ParameterGroup:
    """A group of related parameters."""
    title: str
    parameters: List[ParameterConfig] = field(default_factory=list)


# Default parameter configurations
BASIC_PARAMETERS = ParameterGroup(
    title="Basic Parameters",
    parameters=[
        ParameterConfig(
            name="seed",
            label="Seed",
            default=0,
            min_value=-999999,
            max_value=999999,
            step=1,
            format_str="{:.0f}",
            control_type="spinbox"
        ),
        ParameterConfig(
            name="noise_type",
            label="Noise Type",
            default=0,
            min_value=0,
            max_value=1,
            control_type="combobox"
        ),
        ParameterConfig(
            name="frequency",
            label="Frequency",
            default=0.01,
            min_value=0.001,
            max_value=0.1,
            step=0.001,
            format_str="{:.3f}"
        ),
        ParameterConfig(
            name="offset_x",
            label="Offset X",
            default=0.0,
            min_value=-10000,
            max_value=10000,
            step=10,
            format_str="{:.0f}",
            control_type="spinbox"
        ),
        ParameterConfig(
            name="offset_y",
            label="Offset Y",
            default=0.0,
            min_value=-10000,
            max_value=10000,
            step=10,
            format_str="{:.0f}",
            control_type="spinbox"
        ),
    ]
)

FRACTAL_PARAMETERS = ParameterGroup(
    title="Fractal Parameters",
    parameters=[
        ParameterConfig(
            name="fractal_type",
            label="Fractal Type",
            default=0,
            min_value=0,
            max_value=3,
            control_type="combobox"
        ),
        ParameterConfig(
            name="octaves",
            label="Octaves",
            default=5,
            min_value=1,
            max_value=9,
            step=1,
            format_str="{:.0f}"
        ),
        ParameterConfig(
            name="lacunarity",
            label="Lacunarity",
            default=2.0,
            min_value=1.0,
            max_value=4.0,
            step=0.1,
            format_str="{:.1f}"
        ),
        ParameterConfig(
            name="persistence",
            label="Persistence",
            default=0.5,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format_str="{:.2f}"
        ),
        ParameterConfig(
            name="weighted_strength",
            label="Weighted Strength",
            default=0.0,
            min_value=0.0,
            max_value=1.0,
            step=0.05,
            format_str="{:.2f}"
        ),
        ParameterConfig(
            name="ping_pong_strength",
            label="Ping Pong Strength",
            default=2.0,
            min_value=0.0,
            max_value=4.0,
            step=0.1,
            format_str="{:.1f}"
        ),
    ]
)

IMAGE_PARAMETERS = ParameterGroup(
    title="Image Settings",
    parameters=[
        ParameterConfig(
            name="image_size",
            label="Image Size",
            default=512,
            min_value=128,
            max_value=4096,
            control_type="combobox"
        ),
    ]
)

# Available image sizes
IMAGE_SIZES = [128, 256, 512, 1024, 2048, 4096]

# Display configuration
MAX_DISPLAY_SIZE = 768
