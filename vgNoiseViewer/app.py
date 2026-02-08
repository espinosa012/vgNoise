"""
vgNoise Viewer - A visual noise generator tool using Tkinter.

This application provides a graphical interface to visualize and experiment
with different noise generation parameters compatible with Godot's FastNoiseLite.
"""

import sys
from pathlib import Path

# Add parent directory to path to import vgnoise
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import tkinter as tk
from tkinter import ttk
from typing import Optional
import numpy as np

from vgnoise.enums import NoiseType, FractalType, CellularDistanceFunction, CellularReturnType

# Handle both package and direct execution imports
try:
    from .config import (
        ThemeColors,
        WindowConfig,
        IMAGE_SIZES,
        MAX_DISPLAY_SIZE,
    )
    from .theme import ThemeManager
    from .widgets import (
        StepperControl,
        LabeledCombobox,
        LabeledSpinbox,
        Card,
        ScrollableFrame,
        ParameterConfig,
    )
    from .noise_factory import NoiseGeneratorFactory, NoiseParameters
    from .image_utils import NoiseImageRenderer
except ImportError:
    from config import (
        ThemeColors,
        WindowConfig,
        IMAGE_SIZES,
        MAX_DISPLAY_SIZE,
    )
    from theme import ThemeManager
    from widgets import (
        StepperControl,
        LabeledCombobox,
        LabeledSpinbox,
        Card,
        ScrollableFrame,
        ParameterConfig,
    )
    from noise_factory import NoiseGeneratorFactory, NoiseParameters
    from image_utils import NoiseImageRenderer


class NoiseViewer:
    """
    Main application class for the noise viewer.

    This class manages the UI and coordinates noise generation and display.
    """

    def __init__(
        self,
        root: tk.Tk,
        config: Optional[WindowConfig] = None,
        theme: Optional[ThemeColors] = None
    ):
        """
        Initialize the noise viewer application.

        Args:
            root: The root Tkinter window.
            config: Optional window configuration.
            theme: Optional theme colors.
        """
        self.root = root
        self.config = config or WindowConfig()
        self.theme_colors = theme or ThemeColors()

        # Initialize managers
        self._theme_manager = ThemeManager(self.theme_colors)
        self._renderer = NoiseImageRenderer(MAX_DISPLAY_SIZE)

        # State
        self._initializing = True
        self._photo_image: Optional[tk.PhotoImage] = None
        self._image_label: Optional[ttk.Label] = None

        # Initialize UI variables
        self._init_variables()

        # Configure window
        self._configure_window()

        # Apply theme
        self._theme_manager.apply(ttk.Style())

        # Build UI
        self._build_ui()

        # Mark initialization complete
        self._initializing = False

        # Generate initial image
        self.update_image()

    def _init_variables(self) -> None:
        """Initialize Tkinter variables for parameters."""
        self.seed = tk.IntVar(value=0)
        self.noise_type = tk.StringVar(value=NoiseType.PERLIN.name)
        self.frequency = tk.DoubleVar(value=0.01)
        self.offset_x = tk.DoubleVar(value=0.0)
        self.offset_y = tk.DoubleVar(value=0.0)
        self.fractal_type = tk.StringVar(value=FractalType.FBM.name)
        self.octaves = tk.IntVar(value=5)
        self.lacunarity = tk.DoubleVar(value=2.0)
        self.persistence = tk.DoubleVar(value=0.5)
        self.weighted_strength = tk.DoubleVar(value=0.0)
        self.ping_pong_strength = tk.DoubleVar(value=2.0)
        self.image_size = tk.IntVar(value=512)
        # Cellular-specific
        self.cellular_distance_func = tk.StringVar(value=CellularDistanceFunction.EUCLIDEAN_SQUARED.name)
        self.cellular_return_type = tk.StringVar(value=CellularReturnType.DISTANCE.name)
        self.cellular_jitter = tk.DoubleVar(value=1.0)

    def _configure_window(self) -> None:
        """Configure the main window."""
        self.root.title(self.config.title)
        self.root.geometry(f"{self.config.width}x{self.config.height}")
        self.root.minsize(self.config.min_width, self.config.min_height)
        self.root.configure(bg=self.theme_colors.background)

    def _build_ui(self) -> None:
        """Build the main UI layout."""
        self._build_header()
        self._build_main_content()

    def _build_header(self) -> None:
        """Build the header section."""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(
            header_frame,
            text="vgNoise Viewer",
            style="Header.TLabel"
        ).pack(side=tk.LEFT)

        ttk.Label(
            header_frame,
            text="Godot FastNoiseLite Compatible",
            style="Muted.TLabel"
        ).pack(side=tk.RIGHT)

    def _build_main_content(self) -> None:
        """Build the main content area."""
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Left panel - Controls
        self._build_controls_panel(main_frame)

        # Right panel - Image display
        self._build_image_panel(main_frame)

    def _build_controls_panel(self, parent: ttk.Frame) -> None:
        """Build the controls panel."""
        # Container with fixed width
        left_frame = ttk.Frame(parent, width=320)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        # Scrollable frame
        scroll_frame = ScrollableFrame(left_frame, bg_color=self.theme_colors.background)
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Build control sections
        self._build_basic_controls(scroll_frame.scrollable_frame)
        self._build_fractal_controls(scroll_frame.scrollable_frame)
        self._build_cellular_controls(scroll_frame.scrollable_frame)
        self._build_image_controls(scroll_frame.scrollable_frame)

    def _build_image_panel(self, parent: ttk.Frame) -> None:
        """Build the image display panel."""
        right_frame = ttk.Frame(parent, style="Card.TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._image_label = ttk.Label(right_frame, background=self.theme_colors.card)
        self._image_label.pack(expand=True)

    def _build_basic_controls(self, parent: ttk.Frame) -> None:
        """Build basic parameter controls."""
        card = Card(parent, "Basic Parameters")
        card.pack(fill=tk.X, pady=(0, 10))

        content = card.content_frame

        # Seed
        LabeledSpinbox(
            content,
            config=ParameterConfig(
                name="seed",
                label="Seed",
                default=0,
                min_value=-999999,
                max_value=999999,
                step=1
            ),
            variable=self.seed,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Noise Type
        LabeledCombobox(
            content,
            label="Noise Type",
            variable=self.noise_type,
            values=[t.name for t in NoiseGeneratorFactory.IMPLEMENTED_TYPES],
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Frequency
        StepperControl(
            content,
            config=ParameterConfig(
                name="frequency",
                label="Frequency",
                default=0.01,
                min_value=0.001,
                max_value=0.1,
                step=0.001,
                format_str="{:.3f}"
            ),
            variable=self.frequency,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Offset X
        LabeledSpinbox(
            content,
            config=ParameterConfig(
                name="offset_x",
                label="Offset X",
                default=0.0,
                min_value=-10000,
                max_value=10000,
                step=10
            ),
            variable=self.offset_x,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Offset Y
        LabeledSpinbox(
            content,
            config=ParameterConfig(
                name="offset_y",
                label="Offset Y",
                default=0.0,
                min_value=-10000,
                max_value=10000,
                step=10
            ),
            variable=self.offset_y,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

    def _build_fractal_controls(self, parent: ttk.Frame) -> None:
        """Build fractal parameter controls."""
        card = Card(parent, "Fractal Parameters")
        card.pack(fill=tk.X, pady=(0, 10))

        content = card.content_frame

        # Fractal Type
        LabeledCombobox(
            content,
            label="Fractal Type",
            variable=self.fractal_type,
            values=[t.name for t in FractalType],
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Octaves
        StepperControl(
            content,
            config=ParameterConfig(
                name="octaves",
                label="Octaves",
                default=5,
                min_value=1,
                max_value=9,
                step=1,
                format_str="{:.0f}"
            ),
            variable=self.octaves,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Lacunarity
        StepperControl(
            content,
            config=ParameterConfig(
                name="lacunarity",
                label="Lacunarity",
                default=2.0,
                min_value=1.0,
                max_value=4.0,
                step=0.1,
                format_str="{:.1f}"
            ),
            variable=self.lacunarity,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Persistence
        StepperControl(
            content,
            config=ParameterConfig(
                name="persistence",
                label="Persistence",
                default=0.5,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format_str="{:.2f}"
            ),
            variable=self.persistence,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Weighted Strength
        StepperControl(
            content,
            config=ParameterConfig(
                name="weighted_strength",
                label="Weighted Strength",
                default=0.0,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format_str="{:.2f}"
            ),
            variable=self.weighted_strength,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Ping Pong Strength
        StepperControl(
            content,
            config=ParameterConfig(
                name="ping_pong_strength",
                label="Ping Pong Strength",
                default=2.0,
                min_value=0.0,
                max_value=4.0,
                step=0.1,
                format_str="{:.1f}"
            ),
            variable=self.ping_pong_strength,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

    def _build_cellular_controls(self, parent: ttk.Frame) -> None:
        """Build cellular noise specific controls."""
        self._cellular_card = Card(parent, "Cellular Parameters")
        self._cellular_card.pack(fill=tk.X, pady=(0, 10))

        content = self._cellular_card.content_frame

        # Distance Function
        LabeledCombobox(
            content,
            label="Distance Function",
            variable=self.cellular_distance_func,
            values=[t.name for t in CellularDistanceFunction],
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Return Type
        LabeledCombobox(
            content,
            label="Return Type",
            variable=self.cellular_return_type,
            values=[t.name for t in CellularReturnType],
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Jitter
        StepperControl(
            content,
            config=ParameterConfig(
                name="cellular_jitter",
                label="Jitter",
                default=1.0,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                format_str="{:.2f}"
            ),
            variable=self.cellular_jitter,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

    def _build_image_controls(self, parent: ttk.Frame) -> None:
        """Build image settings controls."""
        card = Card(parent, "Image Settings")
        card.pack(fill=tk.X, pady=(0, 10))

        content = card.content_frame

        # Image Size
        LabeledCombobox(
            content,
            label="Image Size",
            variable=self.image_size,
            values=IMAGE_SIZES,
            on_change=self.update_image
        ).pack(fill=tk.X, pady=2)

        # Buttons
        btn_frame = ttk.Frame(content, style="Card.TFrame")
        btn_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Button(
            btn_frame,
            text="🔄 Regenerate",
            command=self.update_image,
            style="Accent.TButton"
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            btn_frame,
            text="🎲 Random Seed",
            command=self._randomize_seed,
            style="Accent.TButton"
        ).pack(fill=tk.X, pady=2)

    def _get_noise_parameters(self) -> NoiseParameters:
        """Get current noise parameters from UI."""
        return NoiseParameters(
            noise_type=NoiseType[self.noise_type.get()],
            seed=self.seed.get(),
            frequency=self.frequency.get(),
            offset=(self.offset_x.get(), self.offset_y.get()),
            fractal_type=FractalType[self.fractal_type.get()],
            octaves=self.octaves.get(),
            lacunarity=self.lacunarity.get(),
            persistence=self.persistence.get(),
            weighted_strength=self.weighted_strength.get(),
            ping_pong_strength=self.ping_pong_strength.get(),
            # Cellular-specific
            cellular_distance_function=CellularDistanceFunction[self.cellular_distance_func.get()],
            cellular_return_type=CellularReturnType[self.cellular_return_type.get()],
            cellular_jitter=self.cellular_jitter.get()
        )

    def _create_generator(self):
        """Create noise generator with current parameters."""
        params = self._get_noise_parameters()
        return NoiseGeneratorFactory.create_from_params(params)

    def update_image(self, *args) -> None:
        """Generate and display the noise image."""
        if self._image_label is None or self._initializing:
            return

        # Create generator and generate noise
        generator = self._create_generator()
        size = self.image_size.get()

        region = generator.generate_region([
            (0, size, size),
            (0, size, size)
        ])

        # Render to image
        self._photo_image = self._renderer.render(region)
        self._image_label.config(image=self._photo_image)

    def _randomize_seed(self) -> None:
        """Set a random seed and regenerate."""
        new_seed = np.random.randint(0, 999999)
        self.seed.set(new_seed)
        self.update_image()

    # Public properties for testing
    @property
    def photo_image(self):
        """Get the current photo image."""
        return self._photo_image

    @property
    def image_label(self):
        """Get the image label widget."""
        return self._image_label


def main():
    """Main entry point."""
    root = tk.Tk()
    app = NoiseViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
