"""
VGMatrix2D Editor - A visual matrix editor and filter application.

This application provides a graphical interface to create, edit, and manipulate
VGMatrix2D matrices, apply filters, import/export images, and visualize results.
"""

import sys
from pathlib import Path

# Add parent directory to path to import vgmath
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image

from vgmath.matrix import VGMatrix2D
from vgmath.matrix.filters import MatrixFilters

# Handle both package and direct execution imports
try:
    from ..core import ThemeManager, ZoomableImageViewer
    from ..widgets.common import ScrollableFrame
    from ..widgets.matrix_widgets import MatrixCellEditor, Card, StatusBar
    from .config import (
        MatrixThemeColors,
        MatrixWindowConfig,
        MAX_DISPLAY_SIZE,
        MATRIX_SIZES,
        MAX_EDITABLE_SIZE,
        SUPPORTED_IMAGE_FORMATS,
        EXPORT_FORMATS,
    )
    from .filter_panel import FilterPanel, QuickFilterBar
    from .image_utils import (
        MatrixImageRenderer,
        ImageToMatrixConverter,
        MatrixImageGenerator,
    )
    from .noise_dialog import NoiseGeneratorDialog
except ImportError:
    # Direct execution - add paths
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core import ThemeManager, ZoomableImageViewer
    from widgets.common import ScrollableFrame
    from widgets.matrix_widgets import MatrixCellEditor, Card, StatusBar
    from matrix_editor.config import (
        MatrixThemeColors,
        MatrixWindowConfig,
        MAX_DISPLAY_SIZE,
        MATRIX_SIZES,
        MAX_EDITABLE_SIZE,
        SUPPORTED_IMAGE_FORMATS,
        EXPORT_FORMATS,
    )
    from matrix_editor.filter_panel import FilterPanel, QuickFilterBar
    from matrix_editor.image_utils import (
        MatrixImageRenderer,
        ImageToMatrixConverter,
        MatrixImageGenerator,
    )
    from matrix_editor.noise_dialog import NoiseGeneratorDialog


class MatrixEditor:
    """
    Main application class for the matrix editor.

    This class manages the UI and coordinates matrix manipulation and display.
    """

    def __init__(
        self,
        root: tk.Tk,
        config: Optional[MatrixWindowConfig] = None,
        theme: Optional[MatrixThemeColors] = None
    ):
        """
        Initialize the matrix editor application.

        Args:
            root: The root Tkinter window.
            config: Optional window configuration.
            theme: Optional theme colors.
        """
        self.root = root
        self.config = config or MatrixWindowConfig()
        self.theme_colors = theme or MatrixThemeColors()

        # Initialize managers
        self._theme_manager = ThemeManager(self.theme_colors)
        self._renderer = MatrixImageRenderer(MAX_DISPLAY_SIZE)

        # State
        self._initializing = True
        self._matrix: Optional[VGMatrix2D] = None
        self._current_pil_image: Optional[Image.Image] = None  # Current PIL image
        self._image_viewer: Optional[ZoomableImageViewer] = None
        self._cell_editor: Optional[MatrixCellEditor] = None
        self._undo_stack: list = []
        self._redo_stack: list = []
        self._current_file: Optional[str] = None  # Track current file path

        # Initialize UI variables
        self._init_variables()

        # Configure window
        self._configure_window()

        # Apply theme
        self._theme_manager.apply(ttk.Style())

        # Build UI
        self._build_ui()

        # Create initial matrix
        self._create_matrix(int(self.matrix_rows.get()), int(self.matrix_cols.get()))

        # Mark initialization complete
        self._initializing = False

        # Update display
        self._update_display()

    def _init_variables(self) -> None:
        """Initialize Tkinter variables."""
        self.matrix_rows = tk.StringVar(value="1024")
        self.matrix_cols = tk.StringVar(value="1024")
        self.default_value = tk.DoubleVar(value=0.5)
        self.normalize_display = tk.BooleanVar(value=True)
        self.show_transparency = tk.BooleanVar(value=True)
        self.edit_row = tk.IntVar(value=0)
        self.edit_col = tk.IntVar(value=0)
        self.edit_value = tk.StringVar(value="0.5")

    def _configure_window(self) -> None:
        """Configure the main window."""
        self.root.title(self.config.title)
        self.root.geometry(f"{self.config.width}x{self.config.height}")
        self.root.minsize(self.config.min_width, self.config.min_height)
        self.root.configure(bg=self.theme_colors.background)

        # Try to maximize
        try:
            self.root.state('zoomed')
        except tk.TclError:
            self.root.attributes('-zoomed', True)

    def _build_ui(self) -> None:
        """Build the main UI layout."""
        self._build_menu()
        self._build_header()
        self._build_main_content()
        self._build_status_bar()

    def _build_menu(self) -> None:
        """Build the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Matrix...", command=self._show_new_matrix_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Open Matrix...", command=self._open_matrix, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Matrix...", command=self._save_matrix, accelerator="Ctrl+S")
        file_menu.add_command(label="Save Matrix As...", command=self._save_matrix_as, accelerator="Ctrl+Shift+S")
        file_menu.add_separator()
        file_menu.add_command(label="Import Image...", command=self._import_image)
        file_menu.add_command(label="Export Image...", command=self._export_image)
        file_menu.add_command(label="Export as NumPy...", command=self._export_numpy)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        # Edit menu
        edit_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Edit", menu=edit_menu)
        edit_menu.add_command(label="Undo", command=self._undo, accelerator="Ctrl+Z")
        edit_menu.add_command(label="Redo", command=self._redo, accelerator="Ctrl+Y")
        edit_menu.add_separator()
        edit_menu.add_command(label="Fill with Value...", command=self._fill_with_value)
        edit_menu.add_command(label="Fill Random", command=self._fill_random)
        edit_menu.add_command(label="Clear (Set all to None)", command=self._clear_matrix)

        # Matrix menu
        matrix_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Matrix", menu=matrix_menu)
        matrix_menu.add_command(label="Generate from Noise...", command=self._show_noise_generator)
        matrix_menu.add_separator()
        matrix_menu.add_command(label="Normalize (0-1)", command=self._normalize_matrix)
        matrix_menu.add_command(label="Clip Values (0-1)", command=self._clip_matrix)
        matrix_menu.add_command(label="Invert Values", command=self._invert_matrix)
        matrix_menu.add_command(label="Scale...", command=self._show_scale_dialog)
        matrix_menu.add_separator()
        matrix_menu.add_command(label="Resize...", command=self._show_resize_dialog)

        # Bind keyboard shortcuts
        self.root.bind("<Control-z>", lambda e: self._undo())
        self.root.bind("<Control-y>", lambda e: self._redo())
        self.root.bind("<Control-o>", lambda e: self._open_matrix())
        self.root.bind("<Control-s>", lambda e: self._save_matrix())
        self.root.bind("<Control-Shift-S>", lambda e: self._save_matrix_as())

    def _build_header(self) -> None:
        """Build the header section."""
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(
            header_frame,
            text="VGMatrix2D Editor",
            style="Header.TLabel"
        ).pack(side=tk.LEFT)

        ttk.Label(
            header_frame,
            text="Create, Edit & Filter Matrices",
            style="Muted.TLabel"
        ).pack(side=tk.RIGHT)

    def _build_main_content(self) -> None:
        """Build the main content area."""
        # Use PanedWindow for resizable panels
        self._paned = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            sashwidth=6,
            sashrelief=tk.RAISED,
            bg=self.theme_colors.card
        )
        self._paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Left panel - Controls
        left_frame = ttk.Frame(self._paned)
        self._build_controls_panel(left_frame)
        self._paned.add(left_frame, minsize=300, width=420)

        # Right panel - Image display
        right_frame = ttk.Frame(self._paned)
        self._build_image_panel(right_frame)
        self._paned.add(right_frame, minsize=400)

    def _build_controls_panel(self, parent: ttk.Frame) -> None:
        """Build the controls panel."""
        # Scrollable frame
        scroll_frame = ScrollableFrame(parent, bg_color=self.theme_colors.background)
        scroll_frame.pack(fill=tk.BOTH, expand=True)

        # Build control sections
        self._build_matrix_controls(scroll_frame.scrollable_frame)
        self._build_value_editor(scroll_frame.scrollable_frame)
        self._build_filter_controls(scroll_frame.scrollable_frame)
        self._build_display_controls(scroll_frame.scrollable_frame)

    def _build_matrix_controls(self, parent: ttk.Frame) -> None:
        """Build matrix dimension and creation controls."""
        card = Card(parent, "Matrix Settings")
        card.pack(fill=tk.X, pady=5)

        # Size controls
        size_frame = ttk.Frame(card.content, style="Card.TFrame")
        size_frame.pack(fill=tk.X, pady=5)

        ttk.Label(size_frame, text="Size:", style="Card.TLabel").pack(side=tk.LEFT)

        # Rows
        rows_combo = ttk.Combobox(
            size_frame,
            textvariable=self.matrix_rows,
            values=[str(s) for s in MATRIX_SIZES],
            width=6,
            state="readonly"
        )
        rows_combo.pack(side=tk.LEFT, padx=5)
        rows_combo.bind("<<ComboboxSelected>>", self._on_size_change)

        ttk.Label(size_frame, text="x", style="Card.TLabel").pack(side=tk.LEFT)

        # Columns
        cols_combo = ttk.Combobox(
            size_frame,
            textvariable=self.matrix_cols,
            values=[str(s) for s in MATRIX_SIZES],
            width=6,
            state="readonly"
        )
        cols_combo.pack(side=tk.LEFT, padx=5)
        cols_combo.bind("<<ComboboxSelected>>", self._on_size_change)

        # Default value
        default_frame = ttk.Frame(card.content, style="Card.TFrame")
        default_frame.pack(fill=tk.X, pady=5)

        ttk.Label(default_frame, text="Default Value:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Entry(
            default_frame,
            textvariable=self.default_value,
            width=10
        ).pack(side=tk.RIGHT)

        # Create button
        ttk.Button(
            card.content,
            text="Create New Matrix",
            command=self._on_create_matrix
        ).pack(fill=tk.X, pady=5)

        # Quick fill buttons
        fill_frame = ttk.Frame(card.content, style="Card.TFrame")
        fill_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            fill_frame,
            text="Random",
            command=self._fill_random,
            width=8
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            fill_frame,
            text="Gradient",
            command=self._fill_gradient,
            width=8
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            fill_frame,
            text="Clear",
            command=self._clear_matrix,
            width=8
        ).pack(side=tk.LEFT, padx=2)

        # Noise generator button
        ttk.Button(
            card.content,
            text="🌊 Generate from Noise...",
            command=self._show_noise_generator
        ).pack(fill=tk.X, pady=5)

    def _build_value_editor(self, parent: ttk.Frame) -> None:
        """Build single value editor controls."""
        card = Card(parent, "Edit Value", collapsible=True)
        card.pack(fill=tk.X, pady=5)

        # Position
        pos_frame = ttk.Frame(card.content, style="Card.TFrame")
        pos_frame.pack(fill=tk.X, pady=2)

        ttk.Label(pos_frame, text="Row:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(
            pos_frame,
            textvariable=self.edit_row,
            from_=0,
            to=9999,
            width=6
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(pos_frame, text="Col:", style="Card.TLabel").pack(side=tk.LEFT)
        ttk.Spinbox(
            pos_frame,
            textvariable=self.edit_col,
            from_=0,
            to=9999,
            width=6
        ).pack(side=tk.LEFT, padx=5)

        # Value
        value_frame = ttk.Frame(card.content, style="Card.TFrame")
        value_frame.pack(fill=tk.X, pady=2)

        ttk.Label(value_frame, text="Value:", style="Card.TLabel").pack(side=tk.LEFT)
        value_entry = ttk.Entry(
            value_frame,
            textvariable=self.edit_value,
            width=12
        )
        value_entry.pack(side=tk.LEFT, padx=5)
        value_entry.bind("<Return>", lambda e: self._set_single_value())

        ttk.Button(
            value_frame,
            text="Set",
            command=self._set_single_value,
            width=6
        ).pack(side=tk.LEFT)

        ttk.Button(
            value_frame,
            text="Get",
            command=self._get_single_value,
            width=6
        ).pack(side=tk.LEFT, padx=2)

    def _build_filter_controls(self, parent: ttk.Frame) -> None:
        """Build filter controls."""
        card = Card(parent, "Filters")
        card.pack(fill=tk.X, pady=5)

        # Quick filter bar
        quick_bar = QuickFilterBar(
            card.content,
            on_apply_filter=self._apply_filter
        )
        quick_bar.pack(fill=tk.X, pady=5)

        # Full filter panel
        self._filter_panel = FilterPanel(
            card.content,
            on_apply_filter=self._apply_filter,
            theme=self.theme_colors
        )
        self._filter_panel.pack(fill=tk.X, pady=5)

    def _build_display_controls(self, parent: ttk.Frame) -> None:
        """Build display option controls."""
        card = Card(parent, "Display Options", collapsible=True)
        card.pack(fill=tk.X, pady=5)

        # Normalize checkbox
        ttk.Checkbutton(
            card.content,
            text="Normalize display values",
            variable=self.normalize_display,
            command=self._update_display
        ).pack(anchor=tk.W, pady=2)

        # Transparency checkbox
        ttk.Checkbutton(
            card.content,
            text="Show transparency (None values)",
            variable=self.show_transparency,
            command=self._update_display
        ).pack(anchor=tk.W, pady=2)

    def _build_image_panel(self, parent: ttk.Frame) -> None:
        """Build the image display panel with zoom support."""
        # Image container with border (parent is now the pane frame)
        image_container = ttk.Frame(parent, style="Card.TFrame")
        image_container.pack(fill=tk.BOTH, expand=True)

        # Info bar
        info_frame = ttk.Frame(image_container, style="Card.TFrame")
        info_frame.pack(fill=tk.X, padx=10, pady=5)

        self._info_label = ttk.Label(
            info_frame,
            text="Matrix: -",
            style="Card.TLabel"
        )
        self._info_label.pack(side=tk.LEFT)

        self._stats_label = ttk.Label(
            info_frame,
            text="",
            style="Muted.TLabel"
        )
        self._stats_label.pack(side=tk.RIGHT)

        # Zoomable image viewer
        self._image_viewer = ZoomableImageViewer(
            image_container,
            bg_color=self.theme_colors.card,
            zoom_min=0.1,
            zoom_max=10.0,
            zoom_step=0.15,
            on_click=self._on_viewer_click,
            on_hover=self._on_viewer_hover,
            on_zoom_change=self._on_zoom_change,
        )
        self._image_viewer.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Zoom controls bar
        zoom_controls = ttk.Frame(image_container, style="Card.TFrame")
        zoom_controls.pack(fill=tk.X, padx=10, pady=(0, 5))

        ttk.Button(
            zoom_controls,
            text="↶ Undo",
            command=self._undo,
            width=8
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            zoom_controls,
            text="↷ Redo",
            command=self._redo,
            width=8
        ).pack(side=tk.LEFT, padx=2)

        ttk.Separator(zoom_controls, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(
            zoom_controls,
            text="Fit",
            command=self._fit_to_view,
            width=6
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            zoom_controls,
            text="100%",
            command=self._reset_zoom,
            width=6
        ).pack(side=tk.LEFT, padx=2)

        ttk.Label(
            zoom_controls,
            text="Scroll: Zoom | Middle-click drag: Pan",
            style="Muted.TLabel"
        ).pack(side=tk.RIGHT)

    def _build_status_bar(self) -> None:
        """Build the status bar."""
        self._status_bar = StatusBar(self.root)
        self._status_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)

    # =========================================================================
    # Matrix Operations
    # =========================================================================

    def _create_matrix(self, rows: int, cols: int, default: Optional[float] = None) -> None:
        """Create a new matrix with given dimensions."""
        if default is None:
            default = self.default_value.get()

        self._save_undo_state()
        self._matrix = VGMatrix2D((rows, cols), default)
        self._update_display()
        self._status_bar.set_message(f"Created {rows}x{cols} matrix")

    def _on_create_matrix(self) -> None:
        """Handle create matrix button click."""
        try:
            rows = int(self.matrix_rows.get())
            cols = int(self.matrix_cols.get())
            self._create_matrix(rows, cols)
        except ValueError:
            messagebox.showerror("Error", "Invalid matrix dimensions")

    def _on_size_change(self, event=None) -> None:
        """Handle size combobox change - resize matrix preserving existing values."""
        if self._initializing:
            return

        try:
            new_rows = int(self.matrix_rows.get())
            new_cols = int(self.matrix_cols.get())

            if self._matrix is None:
                # No matrix yet, create new one
                self._create_matrix(new_rows, new_cols)
                return

            old_rows, old_cols = self._matrix.shape

            # If same size, do nothing
            if new_rows == old_rows and new_cols == old_cols:
                return

            self._save_undo_state()

            # Create new matrix with default value
            try:
                default = float(self.default_value.get())
            except (ValueError, tk.TclError):
                default = 0.0

            new_matrix = VGMatrix2D((new_rows, new_cols), default)

            # Copy existing values that fit in the new size
            copy_rows = min(old_rows, new_rows)
            copy_cols = min(old_cols, new_cols)

            for r in range(copy_rows):
                for c in range(copy_cols):
                    value = self._matrix.get_value_at(r, c)
                    new_matrix.set_value_at(r, c, value)

            self._matrix = new_matrix
            self._update_display()
            self._status_bar.set_message(f"Resized to {new_rows}x{new_cols}")

        except ValueError:
            pass  # Ignore invalid values

    def _fill_random(self) -> None:
        """Fill matrix with random values."""
        if self._matrix is None:
            return

        self._save_undo_state()
        np.random.seed()
        random_data = np.random.random(self._matrix.shape)
        self._matrix = VGMatrix2D.from_numpy(random_data)
        self._update_display()
        self._status_bar.set_message("Filled with random values")

    def _fill_gradient(self) -> None:
        """Fill matrix with a gradient pattern."""
        if self._matrix is None:
            return

        self._save_undo_state()
        rows, cols = self._matrix.shape

        # Create gradient from top-left to bottom-right
        for r in range(rows):
            for c in range(cols):
                value = (r + c) / (rows + cols - 2) if rows + cols > 2 else 0.5
                self._matrix.set_value_at(r, c, value)

        self._update_display()
        self._status_bar.set_message("Filled with gradient")

    def _fill_with_value(self) -> None:
        """Show dialog to fill with a specific value."""
        if self._matrix is None:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Fill with Value")
        dialog.geometry("250x100")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text="Value (or 'None'):").pack(pady=5)
        value_var = tk.StringVar(value="0.5")
        entry = ttk.Entry(dialog, textvariable=value_var)
        entry.pack(pady=5)
        entry.focus_set()

        def apply():
            text = value_var.get().strip().lower()
            self._save_undo_state()
            if text in ('none', 'null', 'nan', ''):
                self._matrix.fill(None)
            else:
                try:
                    self._matrix.fill(float(text))
                except ValueError:
                    messagebox.showerror("Error", "Invalid value")
                    return
            self._update_display()
            dialog.destroy()

        ttk.Button(dialog, text="Apply", command=apply).pack(pady=5)
        entry.bind("<Return>", lambda e: apply())

    def _clear_matrix(self) -> None:
        """Clear all values (set to None)."""
        if self._matrix is None:
            return

        self._save_undo_state()
        self._matrix.fill(None)
        self._update_display()
        self._status_bar.set_message("Matrix cleared")

    def _normalize_matrix(self) -> None:
        """Normalize matrix values to [0, 1]."""
        if self._matrix is None:
            return

        self._save_undo_state()
        self._matrix = self._matrix.normalize(0.0, 1.0)
        self._update_display()
        self._status_bar.set_message("Matrix normalized")

    def _clip_matrix(self) -> None:
        """Clip matrix values to [0, 1]."""
        if self._matrix is None:
            return

        self._save_undo_state()
        self._matrix = self._matrix.clip(0.0, 1.0)
        self._update_display()
        self._status_bar.set_message("Values clipped to [0, 1]")

    def _invert_matrix(self) -> None:
        """Invert matrix values (1 - value)."""
        if self._matrix is None:
            return

        self._save_undo_state()
        # Invert: new_value = 1 - old_value
        self._matrix = 1.0 - self._matrix
        self._update_display()
        self._status_bar.set_message("Values inverted")

    def _show_scale_dialog(self) -> None:
        """Show dialog to scale matrix values by a scalar."""
        if self._matrix is None:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Scale Matrix")
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(
            dialog,
            text="Multiply all values by a scalar.\nResult will be clipped to [0, 1]."
        ).grid(row=0, column=0, columnspan=2, pady=10, padx=10)

        ttk.Label(dialog, text="Scale factor:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
        scale_var = tk.DoubleVar(value=1.0)
        scale_entry = ttk.Entry(dialog, textvariable=scale_var, width=10)
        scale_entry.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        scale_entry.select_range(0, tk.END)
        scale_entry.focus()

        def apply_scale():
            try:
                factor = scale_var.get()
                self._save_undo_state()
                # Scale and clip to [0, 1]
                self._matrix = (self._matrix * factor).clip(0.0, 1.0)
                self._update_display()
                self._status_bar.set_message(f"Scaled by {factor} and clipped to [0, 1]")
                dialog.destroy()
            except (ValueError, tk.TclError) as e:
                messagebox.showerror("Error", f"Invalid scale factor: {e}")

        ttk.Button(dialog, text="Apply", command=apply_scale).grid(
            row=2, column=0, columnspan=2, pady=15
        )

        # Allow Enter key to apply
        dialog.bind("<Return>", lambda e: apply_scale())

    def _set_single_value(self) -> None:
        """Set a single value at the specified position."""
        if self._matrix is None:
            return

        try:
            row = self.edit_row.get()
            col = self.edit_col.get()
            text = self.edit_value.get().strip().lower()

            if row < 0 or row >= self._matrix.rows or col < 0 or col >= self._matrix.cols:
                messagebox.showerror("Error", "Position out of bounds")
                return

            self._save_undo_state()

            if text in ('none', 'null', 'nan', ''):
                self._matrix.set_value_at(row, col, None)
            else:
                self._matrix.set_value_at(row, col, float(text))

            self._update_display()
            self._status_bar.set_message(f"Set value at ({row}, {col})")
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    def _get_single_value(self) -> None:
        """Get the value at the specified position."""
        if self._matrix is None:
            return

        try:
            row = self.edit_row.get()
            col = self.edit_col.get()

            if row < 0 or row >= self._matrix.rows or col < 0 or col >= self._matrix.cols:
                messagebox.showerror("Error", "Position out of bounds")
                return

            value = self._matrix.get_value_at(row, col)
            if value is None:
                self.edit_value.set("None")
            else:
                self.edit_value.set(f"{value:.6f}")
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Error", f"Invalid input: {e}")

    # =========================================================================
    # Filter Operations
    # =========================================================================

    def _apply_filter(self, filter_name: str, params: Dict[str, Any]) -> None:
        """Apply a filter to the matrix."""
        if self._matrix is None:
            return

        try:
            self._save_undo_state()

            # Get the filter method
            filter_method = getattr(MatrixFilters, filter_name, None)
            if filter_method is None:
                messagebox.showerror("Error", f"Filter '{filter_name}' not found")
                return

            # Create the kernel
            kernel = filter_method(**params)

            # Handle tuple return (e.g., sobel_combined)
            if isinstance(kernel, tuple):
                # For combined filters, apply both and compute magnitude
                result_h = self._matrix.convolve(kernel[0])
                result_v = self._matrix.convolve(kernel[1])
                # Compute magnitude
                data_h = result_h.data
                data_v = result_v.data
                magnitude = np.sqrt(data_h**2 + data_v**2)
                self._matrix = VGMatrix2D.from_numpy(magnitude)
            else:
                # Apply single kernel
                self._matrix = self._matrix.convolve(kernel)

            self._update_display()
            self._status_bar.set_message(f"Applied filter: {filter_name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to apply filter: {e}")

    # =========================================================================
    # Import/Export
    # =========================================================================

    def _import_image(self) -> None:
        """Import an image and convert to matrix."""
        filepath = filedialog.askopenfilename(
            title="Import Image",
            filetypes=SUPPORTED_IMAGE_FORMATS
        )

        if not filepath:
            return

        try:
            # Ask about resizing
            result = messagebox.askyesnocancel(
                "Resize?",
                "Do you want to resize the image to match current matrix size?\n\n"
                "Yes: Resize to current matrix dimensions\n"
                "No: Use original image dimensions\n"
                "Cancel: Abort import"
            )

            if result is None:
                return

            if result and self._matrix is not None:
                # Resize to current matrix size
                target_size = (self._matrix.cols, self._matrix.rows)  # PIL uses (width, height)
            else:
                target_size = None

            self._save_undo_state()
            self._matrix = ImageToMatrixConverter.load_image_as_matrix(filepath, target_size)

            # Update size controls
            self.matrix_rows.set(str(self._matrix.rows))
            self.matrix_cols.set(str(self._matrix.cols))

            self._update_display()
            self._status_bar.set_message(f"Imported: {Path(filepath).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import image: {e}")

    def _export_image(self) -> None:
        """Export matrix as an image."""
        if self._matrix is None:
            messagebox.showwarning("Warning", "No matrix to export")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export Image",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")]
        )

        if not filepath:
            return

        try:
            image = self._renderer.get_pil_image(
                self._matrix,
                normalize=self.normalize_display.get(),
                show_transparency=self.show_transparency.get()
            )
            image.save(filepath)
            self._status_bar.set_message(f"Exported: {Path(filepath).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export image: {e}")

    def _export_numpy(self) -> None:
        """Export matrix as NumPy array."""
        if self._matrix is None:
            messagebox.showwarning("Warning", "No matrix to export")
            return

        filepath = filedialog.asksaveasfilename(
            title="Export NumPy Array",
            defaultextension=".npy",
            filetypes=[("NumPy files", "*.npy")]
        )

        if not filepath:
            return

        try:
            np.save(filepath, self._matrix.to_numpy())
            self._status_bar.set_message(f"Exported: {Path(filepath).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {e}")

    def _open_matrix(self) -> None:
        """Open a matrix from a file."""
        filepath = filedialog.askopenfilename(
            title="Open Matrix",
            filetypes=[
                ("VGMatrix files", "*.vgm *.vgmatrix"),
                ("JSON files", "*.json"),
                ("NumPy files", "*.npy"),
                ("All files", "*.*"),
            ]
        )

        if not filepath:
            return

        try:
            self._save_undo_state()
            self._matrix = VGMatrix2D.load(filepath)
            self._current_file = filepath

            # Update size controls
            self.matrix_rows.set(str(self._matrix.rows))
            self.matrix_cols.set(str(self._matrix.cols))

            self._update_display()
            self._update_window_title()
            self._status_bar.set_message(f"Opened: {Path(filepath).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open matrix: {e}")

    def _save_matrix(self) -> None:
        """Save the matrix to current file or prompt for new file."""
        if self._matrix is None:
            messagebox.showwarning("Warning", "No matrix to save")
            return

        if self._current_file:
            self._save_matrix_to_file(self._current_file)
        else:
            self._save_matrix_as()

    def _save_matrix_as(self) -> None:
        """Save the matrix to a new file."""
        if self._matrix is None:
            messagebox.showwarning("Warning", "No matrix to save")
            return

        filepath = filedialog.asksaveasfilename(
            title="Save Matrix As",
            defaultextension=".vgm",
            filetypes=[
                ("VGMatrix binary", "*.vgm"),
                ("JSON format", "*.json"),
                ("NumPy array", "*.npy"),
            ]
        )

        if not filepath:
            return

        self._save_matrix_to_file(filepath)
        self._current_file = filepath
        self._update_window_title()

    def _save_matrix_to_file(self, filepath: str) -> None:
        """Save matrix to the specified file."""
        try:
            self._matrix.save(filepath)
            self._status_bar.set_message(f"Saved: {Path(filepath).name}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save matrix: {e}")

    def _update_window_title(self) -> None:
        """Update the window title to show current file."""
        base_title = self.config.title
        if self._current_file:
            filename = Path(self._current_file).name
            self.root.title(f"{base_title} - {filename}")
        else:
            self.root.title(base_title)

    # =========================================================================
    # Undo/Redo
    # =========================================================================

    def _save_undo_state(self) -> None:
        """Save current state for undo."""
        if self._matrix is not None:
            self._undo_stack.append(self._matrix.copy())
            self._redo_stack.clear()
            # Limit stack size
            if len(self._undo_stack) > 50:
                self._undo_stack.pop(0)

    def _undo(self) -> None:
        """Undo last operation."""
        if not self._undo_stack:
            self._status_bar.set_message("Nothing to undo")
            return

        if self._matrix is not None:
            self._redo_stack.append(self._matrix.copy())

        self._matrix = self._undo_stack.pop()
        self._update_display()
        self._status_bar.set_message("Undone")

    def _redo(self) -> None:
        """Redo last undone operation."""
        if not self._redo_stack:
            self._status_bar.set_message("Nothing to redo")
            return

        if self._matrix is not None:
            self._undo_stack.append(self._matrix.copy())

        self._matrix = self._redo_stack.pop()
        self._update_display()
        self._status_bar.set_message("Redone")

    # =========================================================================
    # Display
    # =========================================================================

    def _update_display(self) -> None:
        """Update the image display."""
        if self._matrix is None:
            return

        try:
            # Get PIL image from matrix (not PhotoImage)
            self._current_pil_image = self._renderer.get_pil_image(
                self._matrix,
                normalize=self.normalize_display.get(),
                show_transparency=self.show_transparency.get()
            )

            # Update the zoomable viewer
            self._image_viewer.set_image(self._current_pil_image, reset_zoom=False)

            # Update info labels
            self._info_label.configure(
                text=f"Matrix: {self._matrix.rows} x {self._matrix.cols}"
            )

            # Calculate stats
            min_val = self._matrix.min()
            max_val = self._matrix.max()
            mean_val = self._matrix.mean()
            assigned = self._matrix.count_assigned()
            total = self._matrix.size

            stats_parts = []
            if min_val is not None:
                stats_parts.append(f"Min: {min_val:.3f}")
            if max_val is not None:
                stats_parts.append(f"Max: {max_val:.3f}")
            if mean_val is not None:
                stats_parts.append(f"Mean: {mean_val:.3f}")
            stats_parts.append(f"Assigned: {assigned}/{total}")

            self._stats_label.configure(text=" | ".join(stats_parts))
        except Exception as e:
            self._status_bar.set_message(f"Display error: {e}")

    def _on_viewer_click(self, image_x: int, image_y: int) -> None:
        """Handle click on image viewer for value inspection."""
        if self._matrix is None:
            return

        # image_x is column, image_y is row
        row = image_y
        col = image_x

        # Clamp to valid range
        row = max(0, min(row, self._matrix.rows - 1))
        col = max(0, min(col, self._matrix.cols - 1))

        # Update edit controls
        self.edit_row.set(row)
        self.edit_col.set(col)

        # Get and display value
        value = self._matrix.get_value_at(row, col)
        if value is None:
            self.edit_value.set("None")
            self._status_bar.set_message(f"({row}, {col}): None")
        else:
            self.edit_value.set(f"{value:.6f}")
            self._status_bar.set_message(f"({row}, {col}): {value:.6f}")

    def _on_viewer_hover(self, image_x: int, image_y: int) -> None:
        """Handle mouse hover on image viewer."""
        if self._matrix is None:
            return

        # image_x is column, image_y is row
        row = image_y
        col = image_x

        # Clamp
        row = max(0, min(row, self._matrix.rows - 1))
        col = max(0, min(col, self._matrix.cols - 1))

        # Update status bar with hover info
        value = self._matrix.get_value_at(row, col)
        if value is None:
            self._status_bar.set_info(f"Pos: ({row}, {col}) = None")
        else:
            self._status_bar.set_info(f"Pos: ({row}, {col}) = {value:.4f}")

    def _on_zoom_change(self, zoom_level: float) -> None:
        """Handle zoom level change."""
        pass  # Zoom label is updated by the viewer itself

    def _fit_to_view(self) -> None:
        """Fit image to the visible area."""
        self._image_viewer.fit_to_view()

    def _reset_zoom(self) -> None:
        """Reset zoom to 100%."""
        self._image_viewer.reset_zoom()

    # =========================================================================
    # Dialogs
    # =========================================================================

    def _show_new_matrix_dialog(self) -> None:
        """Show dialog to create a new matrix."""
        dialog = tk.Toplevel(self.root)
        dialog.title("New Matrix")
        dialog.geometry("300x200")
        dialog.transient(self.root)
        dialog.grab_set()

        # Rows
        ttk.Label(dialog, text="Rows:").grid(row=0, column=0, padx=10, pady=10)
        rows_var = tk.StringVar(value="64")
        rows_combo = ttk.Combobox(
            dialog,
            textvariable=rows_var,
            values=[str(s) for s in MATRIX_SIZES]
        )
        rows_combo.grid(row=0, column=1, padx=10, pady=10)

        # Columns
        ttk.Label(dialog, text="Columns:").grid(row=1, column=0, padx=10, pady=10)
        cols_var = tk.StringVar(value="64")
        cols_combo = ttk.Combobox(
            dialog,
            textvariable=cols_var,
            values=[str(s) for s in MATRIX_SIZES]
        )
        cols_combo.grid(row=1, column=1, padx=10, pady=10)

        # Default value
        ttk.Label(dialog, text="Default Value:").grid(row=2, column=0, padx=10, pady=10)
        default_var = tk.StringVar(value="0.5")
        ttk.Entry(dialog, textvariable=default_var).grid(row=2, column=1, padx=10, pady=10)

        def create():
            try:
                rows = int(rows_var.get())
                cols = int(cols_var.get())
                text = default_var.get().strip().lower()

                if text in ('none', 'null', 'nan', ''):
                    default = None
                else:
                    default = float(text)

                self.matrix_rows.set(str(rows))
                self.matrix_cols.set(str(cols))
                self._create_matrix(rows, cols, default)
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {e}")

        ttk.Button(dialog, text="Create", command=create).grid(
            row=3, column=0, columnspan=2, pady=20
        )

    def _show_noise_generator(self) -> None:
        """Show dialog to generate matrix from noise."""
        if self._matrix is None:
            # Create a default matrix first
            self._create_matrix(64, 64)

        def on_generate(new_matrix: VGMatrix2D):
            self._save_undo_state()
            self._matrix = new_matrix
            self.matrix_rows.set(str(new_matrix.rows))
            self.matrix_cols.set(str(new_matrix.cols))
            self._update_display()
            self._status_bar.set_message("Generated matrix from noise")

        NoiseGeneratorDialog(
            self.root,
            target_shape=self._matrix.shape,
            on_generate=on_generate,
            theme_bg=self.theme_colors.background,
            theme_card=self.theme_colors.card,
        )

    def _show_resize_dialog(self) -> None:
        """Show dialog to resize the matrix."""
        if self._matrix is None:
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Resize Matrix")
        dialog.geometry("300x180")
        dialog.transient(self.root)
        dialog.grab_set()

        # Current size info
        ttk.Label(
            dialog,
            text=f"Current size: {self._matrix.rows} x {self._matrix.cols}"
        ).grid(row=0, column=0, columnspan=2, pady=10)

        # New rows
        ttk.Label(dialog, text="New Rows:").grid(row=1, column=0, padx=10, pady=5)
        rows_var = tk.StringVar(value=str(self._matrix.rows))
        ttk.Entry(dialog, textvariable=rows_var).grid(row=1, column=1, padx=10, pady=5)

        # New columns
        ttk.Label(dialog, text="New Columns:").grid(row=2, column=0, padx=10, pady=5)
        cols_var = tk.StringVar(value=str(self._matrix.cols))
        ttk.Entry(dialog, textvariable=cols_var).grid(row=2, column=1, padx=10, pady=5)

        def resize():
            try:
                new_rows = int(rows_var.get())
                new_cols = int(cols_var.get())

                if new_rows <= 0 or new_cols <= 0:
                    raise ValueError("Dimensions must be positive")

                self._save_undo_state()
                self._matrix.resize((new_rows, new_cols))
                self.matrix_rows.set(str(new_rows))
                self.matrix_cols.set(str(new_cols))
                self._update_display()
                self._status_bar.set_message(f"Resized to {new_rows}x{new_cols}")
                dialog.destroy()
            except ValueError as e:
                messagebox.showerror("Error", f"Invalid input: {e}")

        ttk.Button(dialog, text="Resize", command=resize).grid(
            row=3, column=0, columnspan=2, pady=20
        )


def main():
    """Main entry point for the Matrix Editor application."""
    root = tk.Tk()
    app = MatrixEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()

