"""
Zoomable Image Viewer - A reusable component for displaying images with zoom and pan.

This module provides a ZoomableImageViewer widget that can be used in any
Tkinter application to display images with zoom (mouse wheel) and pan
(middle mouse button drag) functionality.
"""

import tkinter as tk
from tkinter import ttk
from typing import Optional, Callable, Tuple
from PIL import Image, ImageTk
import numpy as np


class ZoomableImageViewer(ttk.Frame):
    """
    A reusable image viewer with zoom and pan functionality.

    Features:
    - Mouse wheel zoom (centered on cursor position)
    - Middle mouse button pan
    - Configurable zoom limits
    - Click callback for pixel inspection
    - Hover callback for real-time position feedback

    Example:
        >>> viewer = ZoomableImageViewer(parent, on_click=handle_click)
        >>> viewer.pack(fill=tk.BOTH, expand=True)
        >>> viewer.set_image(pil_image)
    """

    def __init__(
        self,
        parent: tk.Widget,
        bg_color: str = "#2d2d2d",
        zoom_min: float = 0.1,
        zoom_max: float = 10.0,
        zoom_step: float = 0.1,
        on_click: Optional[Callable[[int, int], None]] = None,
        on_hover: Optional[Callable[[int, int], None]] = None,
        on_zoom_change: Optional[Callable[[float], None]] = None,
        **kwargs
    ):
        """
        Initialize the zoomable image viewer.

        Args:
            parent: Parent widget.
            bg_color: Background color of the canvas.
            zoom_min: Minimum zoom level (default 0.1 = 10%).
            zoom_max: Maximum zoom level (default 10.0 = 1000%).
            zoom_step: Zoom increment per scroll step.
            on_click: Callback when image is clicked. Receives (image_x, image_y).
            on_hover: Callback when mouse moves over image. Receives (image_x, image_y).
            on_zoom_change: Callback when zoom level changes. Receives new zoom level.
        """
        super().__init__(parent, **kwargs)

        self._bg_color = bg_color
        self._zoom_min = zoom_min
        self._zoom_max = zoom_max
        self._zoom_step = zoom_step
        self._on_click = on_click
        self._on_hover = on_hover
        self._on_zoom_change = on_zoom_change

        # State
        self._zoom_level: float = 1.0
        self._original_image: Optional[Image.Image] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._pan_start_x: int = 0
        self._pan_start_y: int = 0

        self._build_ui()
        self._bind_events()

    def _build_ui(self) -> None:
        """Build the viewer UI."""
        # Main container
        self._container = ttk.Frame(self)
        self._container.pack(fill=tk.BOTH, expand=True)

        # Scrollbars
        h_scrollbar = ttk.Scrollbar(self._container, orient=tk.HORIZONTAL)
        v_scrollbar = ttk.Scrollbar(self._container, orient=tk.VERTICAL)

        # Canvas
        self._canvas = tk.Canvas(
            self._container,
            bg=self._bg_color,
            highlightthickness=0,
            xscrollcommand=h_scrollbar.set,
            yscrollcommand=v_scrollbar.set
        )

        # Configure scrollbars
        h_scrollbar.config(command=self._canvas.xview)
        v_scrollbar.config(command=self._canvas.yview)

        # Grid layout
        self._canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        self._container.grid_rowconfigure(0, weight=1)
        self._container.grid_columnconfigure(0, weight=1)

        # Zoom indicator
        self._zoom_label = ttk.Label(
            self,
            text="Zoom: 100%",
            style="Muted.TLabel"
        )
        self._zoom_label.pack(side=tk.BOTTOM, pady=2)

    def _bind_events(self) -> None:
        """Bind mouse events."""
        # Zoom with mouse wheel
        self._canvas.bind("<MouseWheel>", self._on_mouse_wheel)  # Windows/macOS
        self._canvas.bind("<Button-4>", self._on_mouse_wheel)    # Linux scroll up
        self._canvas.bind("<Button-5>", self._on_mouse_wheel)    # Linux scroll down

        # Pan with middle mouse button
        self._canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self._canvas.bind("<B2-Motion>", self._on_pan_move)
        self._canvas.bind("<ButtonRelease-2>", self._on_pan_end)

        # Click for inspection
        self._canvas.bind("<Button-1>", self._on_left_click)

        # Hover for real-time feedback
        self._canvas.bind("<Motion>", self._on_mouse_move)

    def set_image(self, image: Optional[Image.Image], reset_zoom: bool = False) -> None:
        """
        Set the image to display.

        Args:
            image: PIL Image to display, or None to clear.
            reset_zoom: If True, reset zoom to 100%.
        """
        self._original_image = image

        if reset_zoom:
            self._zoom_level = 1.0

        self._render_image()

    def set_zoom(self, zoom_level: float) -> None:
        """
        Set the zoom level.

        Args:
            zoom_level: New zoom level (1.0 = 100%).
        """
        self._zoom_level = max(self._zoom_min, min(self._zoom_max, zoom_level))
        self._render_image()
        self._notify_zoom_change()

    def get_zoom(self) -> float:
        """Get the current zoom level."""
        return self._zoom_level

    def fit_to_view(self) -> None:
        """Fit the image to the visible area."""
        if self._original_image is None:
            return

        # Get canvas size
        self._canvas.update_idletasks()
        canvas_width = self._canvas.winfo_width()
        canvas_height = self._canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            return

        # Calculate zoom to fit
        img_width, img_height = self._original_image.size
        zoom_x = canvas_width / img_width
        zoom_y = canvas_height / img_height

        self._zoom_level = min(zoom_x, zoom_y, 1.0)  # Don't zoom above 100% for fit
        self._zoom_level = max(self._zoom_min, self._zoom_level)

        self._render_image()
        self._notify_zoom_change()

    def reset_zoom(self) -> None:
        """Reset zoom to 100%."""
        self._zoom_level = 1.0
        self._render_image()
        self._notify_zoom_change()

    def _render_image(self) -> None:
        """Render the image with current zoom level."""
        self._canvas.delete("all")

        if self._original_image is None:
            self._photo_image = None
            self._canvas.config(scrollregion=(0, 0, 1, 1))
            return

        # Calculate zoomed size
        original_width, original_height = self._original_image.size
        zoomed_width = int(original_width * self._zoom_level)
        zoomed_height = int(original_height * self._zoom_level)

        # Ensure minimum size
        zoomed_width = max(1, zoomed_width)
        zoomed_height = max(1, zoomed_height)

        # Resize image
        if self._zoom_level != 1.0:
            resample = Image.Resampling.NEAREST if self._zoom_level > 1.0 else Image.Resampling.LANCZOS
            display_image = self._original_image.resize((zoomed_width, zoomed_height), resample)
        else:
            display_image = self._original_image

        # Convert to PhotoImage
        self._photo_image = ImageTk.PhotoImage(display_image)

        # Draw on canvas
        self._canvas.create_image(0, 0, anchor=tk.NW, image=self._photo_image)

        # Update scroll region
        self._canvas.config(scrollregion=(0, 0, zoomed_width, zoomed_height))

        # Update zoom label
        zoom_percent = int(self._zoom_level * 100)
        self._zoom_label.config(text=f"Zoom: {zoom_percent}%")

    def _on_mouse_wheel(self, event) -> None:
        """Handle mouse wheel for zoom."""
        # Get scroll direction
        if event.num == 4 or (hasattr(event, 'delta') and event.delta > 0):
            delta = self._zoom_step
        elif event.num == 5 or (hasattr(event, 'delta') and event.delta < 0):
            delta = -self._zoom_step
        else:
            return

        # Calculate new zoom level
        new_zoom = self._zoom_level + delta
        new_zoom = max(self._zoom_min, min(self._zoom_max, new_zoom))

        if new_zoom == self._zoom_level:
            return

        if self._original_image is None:
            self._zoom_level = new_zoom
            self._render_image()
            self._notify_zoom_change()
            return

        # Get mouse position on canvas
        canvas_x = self._canvas.canvasx(event.x)
        canvas_y = self._canvas.canvasy(event.y)

        # Calculate relative position
        old_width, old_height = self._original_image.size
        old_width *= self._zoom_level
        old_height *= self._zoom_level

        rel_x = canvas_x / old_width if old_width > 0 else 0.5
        rel_y = canvas_y / old_height if old_height > 0 else 0.5

        # Update zoom level
        self._zoom_level = new_zoom
        self._render_image()
        self._notify_zoom_change()

        # Calculate new scroll position to keep the same point under cursor
        new_width = self._original_image.size[0] * self._zoom_level
        new_height = self._original_image.size[1] * self._zoom_level

        new_canvas_x = rel_x * new_width - event.x
        new_canvas_y = rel_y * new_height - event.y

        # Scroll to maintain position
        if new_width > 0:
            self._canvas.xview_moveto(new_canvas_x / new_width)
        if new_height > 0:
            self._canvas.yview_moveto(new_canvas_y / new_height)

    def _on_pan_start(self, event) -> None:
        """Start panning."""
        self._pan_start_x = event.x
        self._pan_start_y = event.y
        self._canvas.config(cursor="fleur")

    def _on_pan_move(self, event) -> None:
        """Handle panning motion."""
        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y

        self._canvas.xview_scroll(int(-dx), "units")
        self._canvas.yview_scroll(int(-dy), "units")

        self._pan_start_x = event.x
        self._pan_start_y = event.y

    def _on_pan_end(self, event) -> None:
        """End panning."""
        self._canvas.config(cursor="")

    def _on_left_click(self, event) -> None:
        """Handle left click on image."""
        if self._on_click is None or self._original_image is None:
            return

        # Convert canvas coords to image coords
        image_x, image_y = self._canvas_to_image_coords(event.x, event.y)

        if image_x is not None and image_y is not None:
            self._on_click(image_x, image_y)

    def _on_mouse_move(self, event) -> None:
        """Handle mouse movement over image."""
        if self._on_hover is None or self._original_image is None:
            return

        # Convert canvas coords to image coords
        image_x, image_y = self._canvas_to_image_coords(event.x, event.y)

        if image_x is not None and image_y is not None:
            self._on_hover(image_x, image_y)

    def _canvas_to_image_coords(self, canvas_x: int, canvas_y: int) -> Tuple[Optional[int], Optional[int]]:
        """
        Convert canvas coordinates to image coordinates.

        Returns:
            Tuple of (image_x, image_y), or (None, None) if outside image.
        """
        if self._original_image is None:
            return None, None

        # Get position on canvas (accounting for scroll)
        x = self._canvas.canvasx(canvas_x)
        y = self._canvas.canvasy(canvas_y)

        # Convert to image coordinates
        image_x = int(x / self._zoom_level)
        image_y = int(y / self._zoom_level)

        # Check bounds
        img_width, img_height = self._original_image.size
        if 0 <= image_x < img_width and 0 <= image_y < img_height:
            return image_x, image_y

        return None, None

    def _notify_zoom_change(self) -> None:
        """Notify about zoom level change."""
        if self._on_zoom_change:
            self._on_zoom_change(self._zoom_level)

    def get_canvas(self) -> tk.Canvas:
        """Get the underlying canvas widget (for advanced customization)."""
        return self._canvas

