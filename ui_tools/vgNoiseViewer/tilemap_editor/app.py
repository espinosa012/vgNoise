"""
Tilemap Editor - Main application.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional
from tilemap import VGTileMap
try:
    from .config import (
        TilemapThemeColors,
        TilemapWindowConfig,
        DEFAULT_MAP_WIDTH,
        DEFAULT_MAP_HEIGHT,
        DEFAULT_TILE_SIZE,
        TILESET_PANEL_WIDTH,
    )
    from .tileset_panel import TilesetPanel
    from .tilemap_canvas import TilemapCanvas
except ImportError:
    # Direct execution fallback
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from config import (
        TilemapThemeColors,
        TilemapWindowConfig,
        DEFAULT_MAP_WIDTH,
        DEFAULT_MAP_HEIGHT,
        DEFAULT_TILE_SIZE,
        TILESET_PANEL_WIDTH,
    )
    from tileset_panel import TilesetPanel
    from tilemap_canvas import TilemapCanvas
class TilemapEditor:
    """Main tilemap editor application."""
    def __init__(
        self,
        root: tk.Tk,
        config: Optional[TilemapWindowConfig] = None,
        theme: Optional[TilemapThemeColors] = None
    ):
        """
        Initialize the tilemap editor.
        Args:
            root: The root Tkinter window.
            config: Optional window configuration.
            theme: Optional theme colors.
        """
        self.root = root
        self.config = config or TilemapWindowConfig()
        self.theme = theme or TilemapThemeColors()
        # State
        self.tilemap: Optional[VGTileMap] = None
        self._setup_window()
        self._setup_ui()
        self._create_default_tilemap()
    def _setup_window(self):
        """Setup window properties."""
        self.root.title(self.config.title)
        self.root.geometry(f"{self.config.width}x{self.config.height}")
        self.root.minsize(self.config.min_width, self.config.min_height)
        # Configure colors
        self.root.configure(bg=self.theme.background)
    def _setup_ui(self):
        """Setup the user interface."""
        # Menu bar
        self._create_menu()
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True)
        # Left panel - Tilesets
        left_panel = ttk.Frame(main_container, width=TILESET_PANEL_WIDTH)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=5, pady=5)
        left_panel.pack_propagate(False)
        self.tileset_panel = TilesetPanel(
            left_panel,
            on_tile_selected=self._on_tile_selected
        )
        self.tileset_panel.pack(fill=tk.BOTH, expand=True)
        # Right panel - Tilemap canvas and controls
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Toolbar
        toolbar = self._create_toolbar(right_panel)
        toolbar.pack(fill=tk.X, pady=(0, 5))
        # Tilemap canvas
        self.tilemap_canvas = TilemapCanvas(right_panel)
        self.tilemap_canvas.pack(fill=tk.BOTH, expand=True)
        # Status bar
        self.status_bar = ttk.Label(
            self.root,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    def _create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New Tilemap...", command=self._new_tilemap)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
    def _create_toolbar(self, parent):
        """Create the toolbar."""
        toolbar = ttk.Frame(parent)
        ttk.Label(toolbar, text="Tilemap:").pack(side=tk.LEFT, padx=5)
        # Map size info
        self.map_info_label = ttk.Label(toolbar, text="20x15")
        self.map_info_label.pack(side=tk.LEFT, padx=5)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )
        ttk.Button(
            toolbar,
            text="Clear",
            command=self._clear_tilemap
        ).pack(side=tk.LEFT, padx=2)

        # Layer controls
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(
            side=tk.LEFT, fill=tk.Y, padx=10
        )

        ttk.Label(toolbar, text="Layer:").pack(side=tk.LEFT, padx=5)

        self.layer_var = tk.IntVar(value=0)
        self.layer_spinbox = ttk.Spinbox(
            toolbar,
            from_=0,
            to=0,
            textvariable=self.layer_var,
            width=5,
            command=self._on_layer_changed
        )
        self.layer_spinbox.pack(side=tk.LEFT, padx=2)
        self.layer_spinbox.bind('<Return>', lambda e: self._on_layer_changed())

        ttk.Button(
            toolbar,
            text="+",
            command=self._add_layer,
            width=3
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            toolbar,
            text="-",
            command=self._remove_layer,
            width=3
        ).pack(side=tk.LEFT, padx=2)

        return toolbar

    def _create_default_tilemap(self):
        """Create a default empty tilemap."""
        self.tilemap = VGTileMap(
            DEFAULT_MAP_WIDTH,
            DEFAULT_MAP_HEIGHT,
            DEFAULT_TILE_SIZE,
            DEFAULT_TILE_SIZE
        )
        self.tilemap_canvas.set_tilemap(self.tilemap)
        self._update_map_info()
        self._update_layer_controls()
    def _new_tilemap(self):
        """Create a new tilemap with user-specified dimensions."""
        dialog = NewTilemapDialog(self.root)
        if dialog.result:
            width, height, tile_size = dialog.result
            self.tilemap = VGTileMap(width, height, tile_size, tile_size)
            self.tilemap_canvas.set_tilemap(self.tilemap)
            self._update_map_info()
            self._update_layer_controls()
            self.layer_var.set(0)
            self.status_bar.config(text=f"Created new tilemap: {width}x{height}")
    def _clear_tilemap(self):
        """Clear the current tilemap."""
        if not self.tilemap:
            return
        if messagebox.askyesno("Clear Tilemap", "Clear all tiles?"):
            for y in range(self.tilemap.height):
                for x in range(self.tilemap.width):
                    self.tilemap.set_tile(x, y, 0)
            self.tilemap_canvas.render()
            self.status_bar.config(text="Tilemap cleared")

    def _on_layer_changed(self):
        """Handle layer selection change."""
        if not self.tilemap:
            return

        layer = self.layer_var.get()
        if 0 <= layer < self.tilemap.num_layers:
            self.tilemap_canvas.set_current_layer(layer)
            self.status_bar.config(text=f"Active layer: {layer}")

    def _add_layer(self):
        """Add a new layer to the tilemap."""
        if not self.tilemap:
            return

        new_layer_idx = self.tilemap.add_layer()
        self._update_layer_controls()
        self.layer_var.set(new_layer_idx)
        self._on_layer_changed()
        self.tilemap_canvas.render()
        self.status_bar.config(text=f"Added layer {new_layer_idx}")

    def _remove_layer(self):
        """Remove the current layer."""
        if not self.tilemap or self.tilemap.num_layers <= 1:
            messagebox.showwarning("Cannot Remove", "Cannot remove the last layer")
            return

        current_layer = self.layer_var.get()
        if messagebox.askyesno("Remove Layer", f"Remove layer {current_layer}?"):
            if self.tilemap.remove_layer(current_layer):
                # Adjust current layer if needed
                if current_layer >= self.tilemap.num_layers:
                    current_layer = self.tilemap.num_layers - 1

                self._update_layer_controls()
                self.layer_var.set(current_layer)
                self._on_layer_changed()
                self.tilemap_canvas.render()
                self.status_bar.config(text=f"Removed layer")

    def _update_layer_controls(self):
        """Update layer controls to reflect current tilemap state."""
        if self.tilemap:
            self.layer_spinbox.config(to=self.tilemap.num_layers - 1)

    def _on_tile_selected(self, tileset, tile_id):
        """Handle tile selection from tileset panel."""
        # Register tileset if not already registered
        tileset_id = id(tileset)
        # Get the image from tileset panel
        idx = self.tileset_panel.current_tileset_idx
        if idx is not None:
            image = self.tileset_panel.tileset_images[idx]
            self.tilemap_canvas.register_tileset(tileset_id, tileset, image)
        # Set current tile
        self.tilemap_canvas.set_current_tile(tileset_id, tile_id)
        self.status_bar.config(text=f"Selected tile {tile_id} from tileset")
    def _update_map_info(self):
        """Update the map info label."""
        if self.tilemap:
            self.map_info_label.config(
                text=f"{self.tilemap.width}x{self.tilemap.height} "
                     f"({self.tilemap.tile_width}x{self.tilemap.tile_height} tiles) "
                     f"[{self.tilemap.num_layers} layers]"
            )
    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About Tilemap Editor",
            "Tilemap Editor v1.0\n\n"
            "A simple tilemap editor for vgMath.\n\n"
            "Features:\n"
            "- Load tileset images\n"
            "- Create and edit tilemaps\n"
            "- Paint tiles with mouse"
        )
class NewTilemapDialog(tk.Toplevel):
    """Dialog for creating a new tilemap."""
    def __init__(self, parent):
        super().__init__(parent)
        self.title("New Tilemap")
        self.result = None
        # Setup UI
        ttk.Label(self, text="Create New Tilemap").pack(padx=20, pady=10)
        # Inputs
        inputs_frame = ttk.Frame(self)
        inputs_frame.pack(padx=20, pady=10)
        ttk.Label(inputs_frame, text="Width (tiles):").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.width_var = tk.IntVar(value=DEFAULT_MAP_WIDTH)
        ttk.Spinbox(
            inputs_frame,
            from_=5,
            to=100,
            textvariable=self.width_var,
            width=10
        ).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(inputs_frame, text="Height (tiles):").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.height_var = tk.IntVar(value=DEFAULT_MAP_HEIGHT)
        ttk.Spinbox(
            inputs_frame,
            from_=5,
            to=100,
            textvariable=self.height_var,
            width=10
        ).grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(inputs_frame, text="Tile Size (px):").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5
        )
        self.tile_size_var = tk.IntVar(value=DEFAULT_TILE_SIZE)
        ttk.Spinbox(
            inputs_frame,
            from_=8,
            to=128,
            textvariable=self.tile_size_var,
            width=10
        ).grid(row=2, column=1, padx=5, pady=5)
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(padx=20, pady=10)
        ttk.Button(btn_frame, text="Create", command=self._on_ok).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(
            side=tk.LEFT, padx=5
        )
        # Center and modal
        self.transient(parent)
        self.grab_set()
        self.wait_window()
    def _on_ok(self):
        """Accept values."""
        self.result = (
            self.width_var.get(),
            self.height_var.get(),
            self.tile_size_var.get()
        )
        self.destroy()
