"""
Noise Generator Dialog for Matrix Editor.

This module provides a dialog to generate VGMatrix2D from noise,
reusing components from the NoiseViewer.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vgmath.matrix import VGMatrix2D
from vgmath import NoiseType, FractalType, CellularDistanceFunction, CellularReturnType

# Handle both package and direct execution imports
try:
    from ..noise_viewer.factory import NoiseGeneratorFactory, NoiseParameters
    from ..widgets.common import StepperControl, LabeledCombobox
    from ..core import ParameterConfig
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from noise_viewer.factory import NoiseGeneratorFactory, NoiseParameters
    from widgets.common import StepperControl, LabeledCombobox
    from core import ParameterConfig


def _create_section(parent: ttk.Frame, title: str) -> ttk.Frame:
    """Create a section frame with title."""
    frame = ttk.Frame(parent)
    frame.pack(fill=tk.X, pady=5, padx=5)

    ttk.Label(frame, text=title, font=('TkDefaultFont', 10, 'bold')).pack(anchor=tk.W, pady=(5, 5), padx=5)

    content = ttk.Frame(frame)
    content.pack(fill=tk.X, padx=5, pady=(0, 5))

    return content


class NoiseGeneratorDialog(tk.Toplevel):
    """
    Dialog for generating matrices from noise.

    Provides controls for all noise parameters and real-time preview.
    """

    def __init__(
        self,
        parent: tk.Widget,
        target_shape: tuple,
        on_generate: Callable[[VGMatrix2D], None],
        theme_bg: str = "#1e1e1e",
        theme_card: str = "#2d2d2d",
    ):
        """
        Initialize the noise generator dialog.

        Args:
            parent: Parent widget.
            target_shape: Target matrix shape (rows, cols).
            on_generate: Callback when user confirms generation.
            theme_bg: Background color.
            theme_card: Card background color.
        """
        super().__init__(parent)

        self._target_shape = target_shape
        self._on_generate = on_generate
        self._theme_bg = theme_bg
        self._theme_card = theme_card
        self._preview_image = None
        self._generated_matrix: Optional[VGMatrix2D] = None

        self._configure_window()
        self._init_variables()
        self._build_ui()
        # Delay preview update to allow canvas to get dimensions
        self.after(100, self._update_preview)

    def _configure_window(self) -> None:
        """Configure the dialog window."""
        self.title("Generate Matrix from Noise")
        self.geometry("950x750")
        self.minsize(850, 550)
        self.configure(bg=self._theme_bg)
        self.transient(self.master)
        self.grab_set()

    def _init_variables(self) -> None:
        """Initialize control variables."""
        # Region size (editable)
        self.region_rows = tk.IntVar(value=self._target_shape[0])
        self.region_cols = tk.IntVar(value=self._target_shape[1])

        # Noise parameters
        self.seed = tk.IntVar(value=0)
        self.noise_type = tk.StringVar(value=NoiseType.PERLIN.name)
        self.frequency = tk.DoubleVar(value=0.02)
        self.offset_x = tk.DoubleVar(value=0.0)
        self.offset_y = tk.DoubleVar(value=0.0)
        self.fractal_type = tk.StringVar(value=FractalType.FBM.name)
        self.octaves = tk.IntVar(value=5)
        self.lacunarity = tk.DoubleVar(value=2.0)
        self.persistence = tk.DoubleVar(value=0.5)
        self.weighted_strength = tk.DoubleVar(value=0.0)
        self.ping_pong_strength = tk.DoubleVar(value=2.0)
        # Cellular-specific
        self.cellular_distance_func = tk.StringVar(value=CellularDistanceFunction.EUCLIDEAN_SQUARED.name)
        self.cellular_return_type = tk.StringVar(value=CellularReturnType.DISTANCE.name)
        self.cellular_jitter = tk.DoubleVar(value=1.0)

    def _build_ui(self) -> None:
        """Build the dialog UI."""
        # Main container
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left panel - Controls (wider to fit all controls properly)
        left_frame = ttk.Frame(main_frame, width=380)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_frame.pack_propagate(False)

        # Scrollable controls
        canvas = tk.Canvas(left_frame, bg=self._theme_bg, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient=tk.VERTICAL, command=canvas.yview)
        scroll_frame = ttk.Frame(canvas)

        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")

        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        self._build_noise_controls(scroll_frame)

        # Right panel - Preview
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_preview_panel(right_frame)

        # Bottom buttons
        self._build_buttons()

    def _build_noise_controls(self, parent: ttk.Frame) -> None:
        """Build noise parameter controls."""
        # Region Size section
        size_content = _create_section(parent, "Region Size")

        size_frame = ttk.Frame(size_content)
        size_frame.pack(fill=tk.X, pady=2)

        ttk.Label(size_frame, text="Rows:", width=6).pack(side=tk.LEFT)
        rows_entry = ttk.Entry(size_frame, textvariable=self.region_rows, width=8)
        rows_entry.pack(side=tk.LEFT, padx=2)
        rows_entry.bind("<Return>", lambda e: self._update_preview())
        rows_entry.bind("<FocusOut>", lambda e: self._update_preview())

        ttk.Label(size_frame, text="Cols:", width=6).pack(side=tk.LEFT, padx=(10, 0))
        cols_entry = ttk.Entry(size_frame, textvariable=self.region_cols, width=8)
        cols_entry.pack(side=tk.LEFT, padx=2)
        cols_entry.bind("<Return>", lambda e: self._update_preview())
        cols_entry.bind("<FocusOut>", lambda e: self._update_preview())

        # Basic Parameters
        basic_content = _create_section(parent, "Basic Parameters")

        # Seed with randomize button
        seed_frame = ttk.Frame(basic_content)
        seed_frame.pack(fill=tk.X, pady=2)
        ttk.Label(seed_frame, text="Seed:", width=12).pack(side=tk.LEFT)
        ttk.Entry(seed_frame, textvariable=self.seed, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(seed_frame, text="🎲", width=3, command=self._randomize_seed).pack(side=tk.LEFT)

        # Noise Type
        LabeledCombobox(
            basic_content,
            label="Noise Type",
            variable=self.noise_type,
            values=[t.name for t in NoiseGeneratorFactory.IMPLEMENTED_TYPES],
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

        # Frequency
        StepperControl(
            basic_content,
            config=ParameterConfig("frequency", "Frequency", 0.02, 0.001, 0.2, 0.005, "{:.3f}"),
            variable=self.frequency,
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

        # Offset X/Y
        StepperControl(
            basic_content,
            config=ParameterConfig("offset_x", "Offset X", 0, -10000, 10000, 10, "{:.0f}"),
            variable=self.offset_x,
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

        StepperControl(
            basic_content,
            config=ParameterConfig("offset_y", "Offset Y", 0, -10000, 10000, 10, "{:.0f}"),
            variable=self.offset_y,
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

        # Fractal Parameters
        fractal_content = _create_section(parent, "Fractal Parameters")

        LabeledCombobox(
            fractal_content,
            label="Fractal Type",
            variable=self.fractal_type,
            values=[t.name for t in FractalType],
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

        StepperControl(
            fractal_content,
            config=ParameterConfig("octaves", "Octaves", 5, 1, 9, 1, "{:.0f}"),
            variable=self.octaves,
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

        StepperControl(
            fractal_content,
            config=ParameterConfig("lacunarity", "Lacunarity", 2.0, 1.0, 4.0, 0.1, "{:.1f}"),
            variable=self.lacunarity,
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

        StepperControl(
            fractal_content,
            config=ParameterConfig("persistence", "Persistence", 0.5, 0.0, 1.0, 0.05, "{:.2f}"),
            variable=self.persistence,
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

        # Cellular Parameters
        cellular_content = _create_section(parent, "Cellular Parameters")

        LabeledCombobox(
            cellular_content,
            label="Distance Function",
            variable=self.cellular_distance_func,
            values=[t.name for t in CellularDistanceFunction],
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

        LabeledCombobox(
            cellular_content,
            label="Return Type",
            variable=self.cellular_return_type,
            values=[t.name for t in CellularReturnType],
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

        StepperControl(
            cellular_content,
            config=ParameterConfig("jitter", "Jitter", 1.0, 0.0, 1.0, 0.05, "{:.2f}"),
            variable=self.cellular_jitter,
            on_change=self._update_preview
        ).pack(fill=tk.X, pady=2)

    def _build_preview_panel(self, parent: ttk.Frame) -> None:
        """Build the preview panel."""
        preview_card = ttk.Frame(parent)
        preview_card.pack(fill=tk.BOTH, expand=True)

        ttk.Label(preview_card, text="Preview", font=('TkDefaultFont', 11, 'bold')).pack(pady=5)

        # Preview canvas with minimum size
        self._preview_canvas = tk.Canvas(
            preview_card,
            bg=self._theme_card,
            highlightthickness=0,
            width=400,
            height=400
        )
        self._preview_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def _build_buttons(self) -> None:
        """Build dialog buttons."""
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Generate", command=self._on_confirm).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="🎲 Randomize All", command=self._randomize_all).pack(side=tk.LEFT, padx=5)

    def _get_noise_parameters(self) -> NoiseParameters:
        """Get current noise parameters."""
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
            cellular_distance_function=CellularDistanceFunction[self.cellular_distance_func.get()],
            cellular_return_type=CellularReturnType[self.cellular_return_type.get()],
            cellular_jitter=self.cellular_jitter.get()
        )

    def _update_preview(self, *args) -> None:
        """Update the preview image."""
        try:
            # Get region size from inputs
            rows = max(8, min(2048, self.region_rows.get()))
            cols = max(8, min(2048, self.region_cols.get()))

            # Generate noise
            params = self._get_noise_parameters()
            generator = NoiseGeneratorFactory.create_from_params(params)

            noise_data = generator.generate_region([
                (0, rows, rows),
                (0, cols, cols)
            ])

            # Create matrix
            self._generated_matrix = VGMatrix2D.from_numpy(noise_data)

            # Render preview
            self._render_preview(noise_data)
        except Exception as e:
            import traceback
            print(f"Preview error: {e}")
            traceback.print_exc()

    def _render_preview(self, noise_data: np.ndarray) -> None:
        """Render noise data to preview canvas."""
        from PIL import Image, ImageTk

        # Convert to image
        data = np.clip(noise_data, 0, 1)
        image_data = (data * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_data, mode='L')

        # Fit to canvas - use update to ensure dimensions are available
        self._preview_canvas.update()
        canvas_w = self._preview_canvas.winfo_width()
        canvas_h = self._preview_canvas.winfo_height()

        # Fallback to reasonable defaults if canvas not sized yet
        if canvas_w < 50:
            canvas_w = 400
        if canvas_h < 50:
            canvas_h = 400

        # Scale to fit
        img_w, img_h = pil_image.size
        scale = min(canvas_w / img_w, canvas_h / img_h) * 0.95  # 95% to leave margin
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resample = Image.Resampling.NEAREST if scale > 1 else Image.Resampling.LANCZOS
        pil_image = pil_image.resize((new_w, new_h), resample)

        # Display
        self._preview_image = ImageTk.PhotoImage(pil_image)
        self._preview_canvas.delete("all")
        self._preview_canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            anchor=tk.CENTER,
            image=self._preview_image
        )

    def _randomize_seed(self) -> None:
        """Randomize the seed."""
        self.seed.set(np.random.randint(0, 999999))
        self._update_preview()

    def _randomize_all(self) -> None:
        """Randomize all parameters."""
        self.seed.set(np.random.randint(0, 999999))
        noise_types = [t.name for t in NoiseGeneratorFactory.IMPLEMENTED_TYPES]
        self.noise_type.set(np.random.choice(noise_types))
        self.frequency.set(round(np.random.uniform(0.005, 0.05), 3))
        self.octaves.set(np.random.randint(1, 8))
        self.lacunarity.set(round(np.random.uniform(1.5, 3.0), 1))
        self.persistence.set(round(np.random.uniform(0.3, 0.7), 2))
        self._update_preview()

    def _on_confirm(self) -> None:
        """Handle confirm button click."""
        if self._generated_matrix is not None:
            self._on_generate(self._generated_matrix)
        self.destroy()

