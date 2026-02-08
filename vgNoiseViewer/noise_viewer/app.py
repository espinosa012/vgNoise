"""
vgNoise Viewer - A visual noise generator tool using Tkinter.

This application provides a graphical interface to visualize and experiment
with different noise generation parameters compatible with Godot's FastNoiseLite.
"""

import sys
from pathlib import Path

# Add parent directory to path to import vgmath
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional
import numpy as np

from vgmath import NoiseType, FractalType, CellularDistanceFunction, CellularReturnType

# Handle both package and direct execution imports
try:
    from ..core import ThemeColors, WindowConfig, ParameterConfig, IMAGE_SIZES, ThemeManager
    from ..widgets.common import StepperControl, LabeledCombobox, LabeledSpinbox, ScrollableFrame, Card
    from .factory import NoiseGeneratorFactory, NoiseParameters
    from .image_utils import NoiseImageRenderer
except ImportError:
    # Direct execution - add paths
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core import ThemeColors, WindowConfig, ParameterConfig, IMAGE_SIZES, ThemeManager
    from widgets.common import StepperControl, LabeledCombobox, LabeledSpinbox, ScrollableFrame, Card
    from noise_viewer.factory import NoiseGeneratorFactory, NoiseParameters
    from noise_viewer.image_utils import NoiseImageRenderer

# Constant for display
MAX_DISPLAY_SIZE = 800


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
        self._noise_region: Optional[np.ndarray] = None  # Store original noise data
        self._zoom_level: float = 1.0  # Current zoom level
        self._zoom_min: float = 0.25  # Minimum zoom (25%)
        self._zoom_max: float = 4.0   # Maximum zoom (400%)
        self._zoom_step: float = 0.1  # Zoom step per scroll

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
        # Maximize the window (cross-platform)
        try:
            # Windows
            self.root.state('zoomed')
        except tk.TclError:
            # Linux/macOS - use attributes
            self.root.attributes('-zoomed', True)

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
        """Build the image display panel with zoom support."""
        right_frame = ttk.Frame(parent, style="Card.TFrame")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Create a canvas for the image with scrollbars
        self._canvas_frame = ttk.Frame(right_frame)
        self._canvas_frame.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        h_scrollbar = ttk.Scrollbar(self._canvas_frame, orient=tk.HORIZONTAL)
        v_scrollbar = ttk.Scrollbar(self._canvas_frame, orient=tk.VERTICAL)

        # Canvas
        self._image_canvas = tk.Canvas(
            self._canvas_frame,
            bg=self.theme_colors.card,
            highlightthickness=0,
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=v_scrollbar.set
        )

        # Configure scrollbars
        h_scrollbar.config(command=self._image_canvas.xview)
        v_scrollbar.config(command=self._image_canvas.yview)

        # Grid layout for canvas and scrollbars
        self._image_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        self._canvas_frame.grid_rowconfigure(0, weight=1)
        self._canvas_frame.grid_columnconfigure(0, weight=1)

        # Bind mouse wheel for zoom
        self._image_canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows/macOS
        self._image_canvas.bind("<Button-4>", self._on_mouse_wheel)    # Linux scroll up
        self._image_canvas.bind("<Button-5>", self._on_mouse_wheel)    # Linux scroll down

        # Bind for panning with middle mouse button
        self._image_canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self._image_canvas.bind("<B2-Motion>", self._on_pan_move)

        # Zoom indicator label
        self._zoom_label = ttk.Label(
            right_frame,
            text="Zoom: 100%",
            style="Small.TLabel"
        )
        self._zoom_label.pack(side=tk.BOTTOM, pady=5)

        # For backward compatibility
        self._image_label = None

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
            text="🎲 Randomize All",
            command=self._randomize_all,
            style="Accent.TButton"
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            btn_frame,
            text="🌱 Random Seed",
            command=self._randomize_seed,
            style="Accent.TButton"
        ).pack(fill=tk.X, pady=2)

        # File operations
        file_frame = ttk.Frame(content, style="Card.TFrame")
        file_frame.pack(fill=tk.X, pady=(10, 0))

        ttk.Label(
            file_frame,
            text="Noise Preset",
            style="Small.TLabel"
        ).pack(anchor=tk.W)

        ttk.Button(
            file_frame,
            text="💾 Save Noise Preset",
            command=self._save_noise_preset,
            style="TButton"
        ).pack(fill=tk.X, pady=2)

        ttk.Button(
            file_frame,
            text="📂 Load Noise Preset",
            command=self._load_noise_preset,
            style="TButton"
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
        if self._initializing:
            return

        # Check if canvas exists
        if not hasattr(self, '_image_canvas'):
            return

        # Create generator and generate noise
        generator = self._create_generator()
        size = self.image_size.get()

        self._noise_region = generator.generate_region([
            (0, size, size),
            (0, size, size)
        ])

        # Reset zoom when generating new image
        self._zoom_level = 1.0

        # Render and display
        self._render_zoomed_image()

    def _render_zoomed_image(self) -> None:
        """Render the noise image with current zoom level."""
        if self._noise_region is None:
            return

        # Import PIL for resizing
        from PIL import Image, ImageTk

        # Convert noise to image data
        image_data = (np.clip(self._noise_region, 0, 1) * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_data, mode='L')

        # Calculate zoomed size
        original_size = pil_image.size
        zoomed_width = int(original_size[0] * self._zoom_level)
        zoomed_height = int(original_size[1] * self._zoom_level)

        # Resize image
        if self._zoom_level != 1.0:
            resample = Image.NEAREST if self._zoom_level > 1.0 else Image.LANCZOS
            pil_image = pil_image.resize((zoomed_width, zoomed_height), resample)

        # Convert to PhotoImage
        self._photo_image = ImageTk.PhotoImage(pil_image)

        # Update canvas
        self._image_canvas.delete("all")
        self._image_canvas.create_image(0, 0, anchor=tk.NW, image=self._photo_image)

        # Update scroll region
        self._image_canvas.config(scrollregion=(0, 0, zoomed_width, zoomed_height))

        # Update zoom label
        zoom_percent = int(self._zoom_level * 100)
        self._zoom_label.config(text=f"Zoom: {zoom_percent}%")

    def _on_mouse_wheel(self, event) -> None:
        """Handle mouse wheel for zoom."""
        # Get scroll direction
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            # Scroll up - zoom in
            delta = self._zoom_step
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            # Scroll down - zoom out
            delta = -self._zoom_step
        else:
            return

        # Calculate new zoom level
        new_zoom = self._zoom_level + delta
        new_zoom = max(self._zoom_min, min(self._zoom_max, new_zoom))

        if new_zoom != self._zoom_level:
            # Get mouse position on canvas
            canvas_x = self._image_canvas.canvasx(event.x)
            canvas_y = self._image_canvas.canvasy(event.y)

            # Calculate relative position (0-1)
            if self._noise_region is not None:
                old_width = self._noise_region.shape[0] * self._zoom_level
                old_height = self._noise_region.shape[1] * self._zoom_level

                rel_x = canvas_x / old_width if old_width > 0 else 0.5
                rel_y = canvas_y / old_height if old_height > 0 else 0.5

                # Update zoom level
                self._zoom_level = new_zoom

                # Render with new zoom
                self._render_zoomed_image()

                # Calculate new scroll position to keep the same point under cursor
                new_width = self._noise_region.shape[0] * self._zoom_level
                new_height = self._noise_region.shape[1] * self._zoom_level

                # Scroll to maintain position
                new_canvas_x = rel_x * new_width - event.x
                new_canvas_y = rel_y * new_height - event.y

                self._image_canvas.xview_moveto(new_canvas_x / new_width if new_width > 0 else 0)
                self._image_canvas.yview_moveto(new_canvas_y / new_height if new_height > 0 else 0)
            else:
                self._zoom_level = new_zoom
                self._render_zoomed_image()

    def _on_pan_start(self, event) -> None:
        """Start panning with middle mouse button."""
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self._image_canvas.config(cursor="fleur")

    def _on_pan_move(self, event) -> None:
        """Handle panning motion."""
        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y

        self._image_canvas.xview_scroll(-dx, "units")
        self._image_canvas.yview_scroll(-dy, "units")

        self._pan_start_x = event.x
        self._pan_start_y = event.y

    def _on_pan_end(self, event) -> None:
        """End panning."""
        self._image_canvas.config(cursor="")

    def _randomize_seed(self) -> None:
        """Set a random seed and regenerate."""
        new_seed = np.random.randint(0, 999999)
        self.seed.set(new_seed)
        self.update_image()

    def _randomize_all(self) -> None:
        """Randomize all noise parameters and regenerate."""
        # Temporarily disable updates
        self._initializing = True

        try:
            # Seed
            self.seed.set(np.random.randint(0, 999999))

            # Noise type
            noise_types = [t.name for t in NoiseGeneratorFactory.IMPLEMENTED_TYPES]
            self.noise_type.set(np.random.choice(noise_types))

            # Frequency (0.005 to 0.05 for useful range)
            self.frequency.set(round(np.random.uniform(0.005, 0.05), 3))

            # Offsets
            self.offset_x.set(round(np.random.uniform(-1000, 1000), 0))
            self.offset_y.set(round(np.random.uniform(-1000, 1000), 0))

            # Fractal type
            fractal_types = [t.name for t in FractalType]
            self.fractal_type.set(np.random.choice(fractal_types))

            # Octaves (1-8)
            self.octaves.set(np.random.randint(1, 9))

            # Lacunarity (1.5 to 3.0)
            self.lacunarity.set(round(np.random.uniform(1.5, 3.0), 1))

            # Persistence (0.3 to 0.7)
            self.persistence.set(round(np.random.uniform(0.3, 0.7), 2))

            # Weighted strength (0 to 0.5)
            self.weighted_strength.set(round(np.random.uniform(0.0, 0.5), 2))

            # Ping pong strength (1.0 to 3.0)
            self.ping_pong_strength.set(round(np.random.uniform(1.0, 3.0), 1))

            # Cellular parameters
            dist_funcs = [t.name for t in CellularDistanceFunction]
            self.cellular_distance_func.set(np.random.choice(dist_funcs))

            return_types = [t.name for t in CellularReturnType]
            self.cellular_return_type.set(np.random.choice(return_types))

            # Jitter (0.5 to 1.0)
            self.cellular_jitter.set(round(np.random.uniform(0.5, 1.0), 2))

        finally:
            self._initializing = False

        # Update the image with new settings
        self.update_image()

    def _get_config_dict(self) -> dict:
        """Get current noise configuration as a dictionary."""
        return {
            "noise_type": self.noise_type.get(),
            "seed": self.seed.get(),
            "frequency": self.frequency.get(),
            "offset_x": self.offset_x.get(),
            "offset_y": self.offset_y.get(),
            "fractal_type": self.fractal_type.get(),
            "octaves": self.octaves.get(),
            "lacunarity": self.lacunarity.get(),
            "persistence": self.persistence.get(),
            "weighted_strength": self.weighted_strength.get(),
            "ping_pong_strength": self.ping_pong_strength.get(),
            "cellular_distance_function": self.cellular_distance_func.get(),
            "cellular_return_type": self.cellular_return_type.get(),
            "cellular_jitter": self.cellular_jitter.get(),
        }

    def _set_config_from_dict(self, config: dict) -> None:
        """Set UI values from a configuration dictionary."""
        # Temporarily disable updates
        self._initializing = True

        try:
            # Set all values from config
            if "noise_type" in config:
                self.noise_type.set(config["noise_type"])
            if "seed" in config:
                self.seed.set(config["seed"])
            if "frequency" in config:
                self.frequency.set(config["frequency"])
            if "offset_x" in config:
                self.offset_x.set(config["offset_x"])
            if "offset_y" in config:
                self.offset_y.set(config["offset_y"])
            if "fractal_type" in config:
                self.fractal_type.set(config["fractal_type"])
            if "octaves" in config:
                self.octaves.set(config["octaves"])
            if "lacunarity" in config:
                self.lacunarity.set(config["lacunarity"])
            if "persistence" in config:
                self.persistence.set(config["persistence"])
            if "weighted_strength" in config:
                self.weighted_strength.set(config["weighted_strength"])
            if "ping_pong_strength" in config:
                self.ping_pong_strength.set(config["ping_pong_strength"])
            if "cellular_distance_function" in config:
                self.cellular_distance_func.set(config["cellular_distance_function"])
            if "cellular_return_type" in config:
                self.cellular_return_type.set(config["cellular_return_type"])
            if "cellular_jitter" in config:
                self.cellular_jitter.set(config["cellular_jitter"])
        finally:
            self._initializing = False

        # Update the image with new settings
        self.update_image()

    def _save_noise_preset(self) -> None:
        """Save current noise configuration to a JSON file."""
        from vgmath.generators import NoiseGenerator2D, NOISE_JSON_EXTENSION

        filepath = filedialog.asksaveasfilename(
            title="Save Noise Preset",
            defaultextension=NOISE_JSON_EXTENSION,
            filetypes=[
                ("Noise Preset Files", f"*{NOISE_JSON_EXTENSION}"),
                ("JSON Files", "*.json"),
                ("All Files", "*.*")
            ],
            initialfile="noise_preset"
        )

        if not filepath:
            return  # User cancelled

        try:
            # Create a NoiseGenerator2D with current config and save it
            config = self._get_config_dict()
            generator = NoiseGenerator2D(config=config)
            generator.save_to_json(filepath)

            messagebox.showinfo(
                "Success",
                f"Noise preset saved successfully to:\n{filepath}"
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to save noise preset:\n{str(e)}"
            )

    def _load_noise_preset(self) -> None:
        """Load noise configuration from a JSON file."""
        from vgmath.generators import NoiseGenerator2D, NOISE_JSON_EXTENSION

        filepath = filedialog.askopenfilename(
            title="Load Noise Preset",
            filetypes=[
                ("Noise Preset Files", f"*{NOISE_JSON_EXTENSION}"),
                ("JSON Files", "*.json"),
                ("All Files", "*.*")
            ]
        )

        if not filepath:
            return  # User cancelled

        try:
            # Load the noise generator from JSON
            generator = NoiseGenerator2D.load_from_json(filepath)

            # Apply the configuration to the UI
            self._set_config_from_dict(generator.config)

            messagebox.showinfo(
                "Success",
                f"Noise preset loaded successfully from:\n{filepath}"
            )
        except FileNotFoundError:
            messagebox.showerror(
                "Error",
                f"File not found:\n{filepath}"
            )
        except Exception as e:
            messagebox.showerror(
                "Error",
                f"Failed to load noise preset:\n{str(e)}"
            )

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
