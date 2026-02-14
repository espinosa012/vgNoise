"""
Tileset management panel for the tilemap editor.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, List, Callable
from PIL import Image, ImageTk
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
from tilemap import TileSet
class TilesetPanel(ttk.Frame):
    """Panel for managing and displaying tilesets."""
    def __init__(
        self,
        parent,
        on_tile_selected: Optional[Callable[[TileSet, int], None]] = None,
        **kwargs
    ):
        """
        Initialize the tileset panel.
        Args:
            parent: Parent widget.
            on_tile_selected: Callback when a tile is selected (tileset, tile_id).
        """
        super().__init__(parent, **kwargs)
        self.on_tile_selected = on_tile_selected
        self.tilesets: List[TileSet] = []
        self.current_tileset_idx: Optional[int] = None
        self.selected_tile_id: Optional[int] = None
        self.tileset_images: List[Image.Image] = []  # Store loaded images
        self.zoom_level: float = 1.0  # Current zoom level
        self.selection_rect_id: Optional[int] = None  # Canvas rectangle for selection

        self._setup_ui()
    def _setup_ui(self):
        """Setup the UI components."""
        # Header
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(header, text="Tilesets", font=("TkDefaultFont", 10, "bold")).pack(
            side=tk.LEFT
        )
        ttk.Button(
            header,
            text="Remove",
            command=self._remove_tileset,
            width=8
        ).pack(side=tk.RIGHT, padx=2)
        ttk.Button(
            header,
            text="Add",
            command=self._add_tileset,
            width=8
        ).pack(side=tk.RIGHT, padx=2)
        # Tileset selector
        selector_frame = ttk.Frame(self)
        selector_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(selector_frame, text="Current:").pack(side=tk.LEFT, padx=(0, 5))
        self.tileset_combo = ttk.Combobox(
            selector_frame,
            state="readonly",
            width=20
        )
        self.tileset_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.tileset_combo.bind("<<ComboboxSelected>>", self._on_tileset_changed)
        # Tile size info
        self.info_label = ttk.Label(self, text="No tileset loaded")
        self.info_label.pack(fill=tk.X, padx=5, pady=5)

        # Tile size editor
        tile_size_frame = ttk.Frame(self)
        tile_size_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(tile_size_frame, text="Tile Size:").pack(side=tk.LEFT, padx=(0, 5))

        ttk.Label(tile_size_frame, text="W:").pack(side=tk.LEFT, padx=(0, 2))
        self.tile_width_var = tk.StringVar(value="16")
        self.tile_width_entry = ttk.Entry(
            tile_size_frame,
            textvariable=self.tile_width_var,
            width=6
        )
        self.tile_width_entry.pack(side=tk.LEFT, padx=2)
        self.tile_width_entry.bind('<Return>', self._on_tile_size_changed)
        self.tile_width_entry.bind('<FocusOut>', self._on_tile_size_changed)

        ttk.Label(tile_size_frame, text="H:").pack(side=tk.LEFT, padx=(5, 2))
        self.tile_height_var = tk.StringVar(value="16")
        self.tile_height_entry = ttk.Entry(
            tile_size_frame,
            textvariable=self.tile_height_var,
            width=6
        )
        self.tile_height_entry.pack(side=tk.LEFT, padx=2)
        self.tile_height_entry.bind('<Return>', self._on_tile_size_changed)
        self.tile_height_entry.bind('<FocusOut>', self._on_tile_size_changed)

        ttk.Button(
            tile_size_frame,
            text="Apply",
            command=self._on_tile_size_changed,
            width=6
        ).pack(side=tk.LEFT, padx=5)

        # Zoom controls
        zoom_frame = ttk.Frame(self)
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(zoom_frame, text="Zoom:").pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            zoom_frame,
            text="-",
            command=self._zoom_out,
            width=3
        ).pack(side=tk.LEFT, padx=2)

        self.zoom_label = ttk.Label(zoom_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=2)

        ttk.Button(
            zoom_frame,
            text="+",
            command=self._zoom_in,
            width=3
        ).pack(side=tk.LEFT, padx=2)

        ttk.Button(
            zoom_frame,
            text="Reset",
            command=self._zoom_reset,
            width=6
        ).pack(side=tk.LEFT, padx=2)

        # Tile picker canvas
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        # Add scrollbars
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.tile_canvas = tk.Canvas(
            canvas_frame,
            bg="#2d2d2d",
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set,
            highlightthickness=0
        )
        self.tile_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scroll.config(command=self.tile_canvas.yview)
        h_scroll.config(command=self.tile_canvas.xview)

        self.tile_canvas.bind("<Button-1>", self._on_canvas_click)
        self.tile_canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows/Mac
        self.tile_canvas.bind("<Button-4>", self._on_mouse_wheel)    # Linux scroll up
        self.tile_canvas.bind("<Button-5>", self._on_mouse_wheel)    # Linux scroll down

        # Selected tile info
        self.selected_label = ttk.Label(self, text="No tile selected")
        self.selected_label.pack(fill=tk.X, padx=5, pady=5)
    def _add_tileset(self):
        """Add a new tileset from an image file."""
        filepath = filedialog.askopenfilename(
            title="Select Tileset Image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.gif"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )

        if not filepath:
            return

        # Use default tile size 16x16
        tile_width = 16
        tile_height = 16

        try:
            # Load image first to get dimensions
            image = Image.open(filepath)

            # Create tileset with default size
            tileset = TileSet(tile_width=tile_width, tile_height=tile_height)

            # Try to load, if dimensions don't match, we'll handle it gracefully
            try:
                tileset.load_from_image(filepath)
            except ValueError:
                # Image dimensions not divisible by tile size
                # Still add it, user can adjust tile size later
                messagebox.showwarning(
                    "Dimension Mismatch",
                    f"Image dimensions ({image.width}x{image.height}) are not divisible by tile size (16x16).\n"
                    f"Please adjust the tile size using the controls below."
                )
                # Set manual grid size based on image dimensions
                tileset.columns = image.width // tile_width
                tileset.rows = image.height // tile_height
                tileset.image_path = filepath

            # Add to list
            self.tilesets.append(tileset)
            self.tileset_images.append(image)

            # Update combo
            name = Path(filepath).stem
            self.tileset_combo["values"] = [
                *self.tileset_combo["values"],
                f"{name} ({tileset.columns}x{tileset.rows})"
            ]

            # Select the new tileset
            self.tileset_combo.current(len(self.tilesets) - 1)
            self._on_tileset_changed()

            # Update tile size entries
            self.tile_width_var.set(str(tile_width))
            self.tile_height_var.set(str(tile_height))

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load tileset:\n{str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load tileset:\n{str(e)}")

    def _remove_tileset(self):
        """Remove the currently selected tileset."""
        if self.current_tileset_idx is None:
            messagebox.showwarning("No Tileset", "No tileset selected to remove")
            return

        if not messagebox.askyesno("Remove Tileset", "Are you sure you want to remove this tileset?"):
            return

        try:
            # Remove from lists
            self.tilesets.pop(self.current_tileset_idx)
            self.tileset_images.pop(self.current_tileset_idx)

            # Update combo
            values = list(self.tileset_combo["values"])
            values.pop(self.current_tileset_idx)
            self.tileset_combo["values"] = values

            # Clear selection
            self.current_tileset_idx = None
            self.selected_tile_id = None

            # Select first tileset if available
            if len(self.tilesets) > 0:
                self.tileset_combo.current(0)
                self._on_tileset_changed()
            else:
                self.tileset_combo.set("")
                self.tile_canvas.delete("all")
                self.info_label.config(text="No tileset loaded")
                self.selected_label.config(text="No tile selected")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to remove tileset:\n{str(e)}")

    def _on_tile_size_changed(self, event=None):
        """Handle tile size change from entry fields."""
        if self.current_tileset_idx is None:
            messagebox.showwarning("No Tileset", "No tileset selected")
            return

        try:
            # Get new tile size from entries
            new_width = int(self.tile_width_var.get())
            new_height = int(self.tile_height_var.get())

            # Validate
            if new_width <= 0 or new_height <= 0:
                messagebox.showerror("Invalid Size", "Tile size must be greater than 0")
                return

            if new_width > 512 or new_height > 512:
                messagebox.showerror("Invalid Size", "Tile size must be 512 or less")
                return

            # Get current tileset and image
            tileset = self.tilesets[self.current_tileset_idx]
            image = self.tileset_images[self.current_tileset_idx]

            # Check if dimensions are divisible
            if image.width % new_width != 0:
                if not messagebox.askyesno(
                    "Dimension Warning",
                    f"Image width ({image.width}) is not divisible by tile width ({new_width}).\n"
                    f"Some tiles may be cut off. Continue anyway?"
                ):
                    return

            if image.height % new_height != 0:
                if not messagebox.askyesno(
                    "Dimension Warning",
                    f"Image height ({image.height}) is not divisible by tile height ({new_height}).\n"
                    f"Some tiles may be cut off. Continue anyway?"
                ):
                    return

            # Update tileset
            tileset.tile_width = new_width
            tileset.tile_height = new_height
            tileset.columns = image.width // new_width
            tileset.rows = image.height // new_height

            # Update combo text
            values = list(self.tileset_combo["values"])
            name = values[self.current_tileset_idx].split(" (")[0]  # Get original name
            values[self.current_tileset_idx] = f"{name} ({tileset.columns}x{tileset.rows})"
            self.tileset_combo["values"] = values
            self.tileset_combo.current(self.current_tileset_idx)

            # Redisplay
            self._display_tileset()


        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for tile size")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update tile size:\n{str(e)}")

    def _on_tileset_changed(self, event=None):
        """Handle tileset selection change."""
        idx = self.tileset_combo.current()
        if idx < 0:
            return
        self.current_tileset_idx = idx
        self._display_tileset()
    def _display_tileset(self):
        """Display the current tileset in the canvas."""
        if self.current_tileset_idx is None:
            return

        tileset = self.tilesets[self.current_tileset_idx]
        image = self.tileset_images[self.current_tileset_idx]

        # Update info
        self.info_label.config(
            text=f"Tile size: {tileset.tile_width}x{tileset.tile_height} | "
                 f"Grid: {tileset.columns}x{tileset.rows}"
        )

        # Clear canvas
        self.tile_canvas.delete("all")

        # Apply zoom to image
        if self.zoom_level != 1.0:
            new_width = int(image.width * self.zoom_level)
            new_height = int(image.height * self.zoom_level)
            zoomed_image = image.resize((new_width, new_height), Image.Resampling.NEAREST)
        else:
            zoomed_image = image

        # Display tileset image
        self.tileset_photo = ImageTk.PhotoImage(zoomed_image)
        self.tile_canvas.create_image(0, 0, anchor=tk.NW, image=self.tileset_photo)

        # Draw grid with zoom
        tile_w = int(tileset.tile_width * self.zoom_level)
        tile_h = int(tileset.tile_height * self.zoom_level)

        for row in range(tileset.rows + 1):
            y = row * tile_h
            self.tile_canvas.create_line(
                0, y, zoomed_image.width, y,
                fill="#444444", width=1, tags="grid"
            )

        for col in range(tileset.columns + 1):
            x = col * tile_w
            self.tile_canvas.create_line(
                x, 0, x, zoomed_image.height,
                fill="#444444", width=1, tags="grid"
            )

        # Update scroll region
        self.tile_canvas.config(scrollregion=(0, 0, zoomed_image.width, zoomed_image.height))

        # Redraw selection if a tile is selected
        if self.selected_tile_id is not None:
            self._draw_selection()
    def _on_canvas_click(self, event):
        """Handle click on the tileset canvas."""
        if self.current_tileset_idx is None:
            return

        tileset = self.tilesets[self.current_tileset_idx]

        # Get canvas coordinates
        x = self.tile_canvas.canvasx(event.x)
        y = self.tile_canvas.canvasy(event.y)

        # Calculate tile position with zoom
        tile_w = tileset.tile_width * self.zoom_level
        tile_h = tileset.tile_height * self.zoom_level

        col = int(x // tile_w)
        row = int(y // tile_h)

        if 0 <= col < tileset.columns and 0 <= row < tileset.rows:
            tile_id = row * tileset.columns + col
            self.selected_tile_id = tile_id

            # Update label
            self.selected_label.config(
                text=f"Selected: Tile {tile_id} (col {col}, row {row})"
            )

            # Draw selection highlight
            self._draw_selection()

            # Callback
            if self.on_tile_selected:
                self.on_tile_selected(tileset, tile_id)

    def _draw_selection(self):
        """Draw a highlight rectangle around the selected tile."""
        if self.current_tileset_idx is None or self.selected_tile_id is None:
            return

        # Remove previous selection rectangle
        self.tile_canvas.delete("selection")

        tileset = self.tilesets[self.current_tileset_idx]

        # Calculate tile position
        col = self.selected_tile_id % tileset.columns
        row = self.selected_tile_id // tileset.columns

        # Calculate rectangle with zoom
        tile_w = tileset.tile_width * self.zoom_level
        tile_h = tileset.tile_height * self.zoom_level

        x1 = col * tile_w
        y1 = row * tile_h
        x2 = x1 + tile_w
        y2 = y1 + tile_h

        # Draw selection rectangle with bright color
        self.tile_canvas.create_rectangle(
            x1, y1, x2, y2,
            outline="#ffaa00",
            width=3,
            tags="selection"
        )

    def _zoom_in(self):
        """Zoom in the tileset view."""
        if self.zoom_level < 4.0:
            self.zoom_level *= 1.25
            self._update_zoom()

    def _zoom_out(self):
        """Zoom out the tileset view."""
        if self.zoom_level > 0.25:
            self.zoom_level /= 1.25
            self._update_zoom()

    def _zoom_reset(self):
        """Reset zoom to 100%."""
        self.zoom_level = 1.0
        self._update_zoom()

    def _on_mouse_wheel(self, event):
        """Handle mouse wheel for zooming."""
        if self.current_tileset_idx is None:
            return

        # Determine zoom direction
        if event.num == 4 or event.delta > 0:  # Scroll up
            self._zoom_in()
        elif event.num == 5 or event.delta < 0:  # Scroll down
            self._zoom_out()

    def _update_zoom(self):
        """Update the display after zoom change."""
        # Update zoom label
        self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")

        # Redisplay tileset
        self._display_tileset()

    def get_current_selection(self):
        """Get current tileset and tile selection."""
        if self.current_tileset_idx is None or self.selected_tile_id is None:
            return None, None
        return self.tilesets[self.current_tileset_idx], self.selected_tile_id
class TileSizeDialog(tk.Toplevel):
    """Dialog for entering tile size when loading a tileset."""
    def __init__(self, parent, filepath):
        super().__init__(parent)
        self.title("Tile Size")
        self.result = None
        # Get image size
        with Image.open(filepath) as img:
            self.image_width, self.image_height = img.size
        # Setup UI
        ttk.Label(
            self,
            text=f"Image size: {self.image_width}x{self.image_height}\n"
                 "Enter tile dimensions:"
        ).pack(padx=20, pady=10)
        # Tile size inputs
        inputs_frame = ttk.Frame(self)
        inputs_frame.pack(padx=20, pady=10)
        ttk.Label(inputs_frame, text="Tile Width:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.width_var = tk.IntVar(value=32)
        ttk.Spinbox(
            inputs_frame,
            from_=8,
            to=256,
            textvariable=self.width_var,
            width=10
        ).grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(inputs_frame, text="Tile Height:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.height_var = tk.IntVar(value=32)
        ttk.Spinbox(
            inputs_frame,
            from_=8,
            to=256,
            textvariable=self.height_var,
            width=10
        ).grid(row=1, column=1, padx=5, pady=5)
        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(padx=20, pady=10)
        ttk.Button(btn_frame, text="OK", command=self._on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        # Center and modal
        self.transient(parent)
        self.grab_set()
        self.wait_window()
    def _on_ok(self):
        """Validate and accept."""
        tile_width = self.width_var.get()
        tile_height = self.height_var.get()
        # Validate
        if self.image_width % tile_width != 0:
            messagebox.showerror(
                "Error",
                f"Image width ({self.image_width}) is not divisible by tile width ({tile_width})"
            )
            return
        if self.image_height % tile_height != 0:
            messagebox.showerror(
                "Error",
                f"Image height ({self.image_height}) is not divisible by tile height ({tile_height})"
            )
            return
        self.result = (tile_width, tile_height)
        self.destroy()
