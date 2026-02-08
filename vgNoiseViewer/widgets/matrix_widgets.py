"""
Custom widgets for Matrix Editor App.

This module contains reusable custom widget classes specific to matrix editing.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List, Any, Tuple
from dataclasses import dataclass

# Local constants to avoid circular imports
EDIT_CELL_SIZE = 40
MAX_EDITABLE_SIZE = 16


@dataclass
class MatrixThemeColors:
    """Theme colors for matrix editor (local copy to avoid circular import)."""
    background: str = "#1e1e1e"
    foreground: str = "#ffffff"
    card: str = "#2d2d2d"
    accent: str = "#4a9eff"
    muted: str = "#808080"
    success: str = "#4caf50"
    warning: str = "#ff9800"
    error: str = "#f44336"


class MatrixCellEditor(ttk.Frame):
    """
    Widget for directly editing individual matrix cells.
    Only used for small matrices (up to MAX_EDITABLE_SIZE x MAX_EDITABLE_SIZE).
    """

    def __init__(
        self,
        parent: tk.Widget,
        rows: int,
        cols: int,
        on_cell_change: Optional[Callable[[int, int, Optional[float]], None]] = None,
        theme: Optional[MatrixThemeColors] = None,
        **kwargs
    ):
        """
        Initialize the cell editor.

        Args:
            parent: Parent widget.
            rows: Number of rows.
            cols: Number of columns.
            on_cell_change: Callback when a cell value changes (row, col, new_value).
            theme: Optional theme colors.
        """
        super().__init__(parent, **kwargs)

        self.rows = min(rows, MAX_EDITABLE_SIZE)
        self.cols = min(cols, MAX_EDITABLE_SIZE)
        self.on_cell_change = on_cell_change
        self.theme = theme or MatrixThemeColors()

        self._cells: List[List[ttk.Entry]] = []
        self._cell_vars: List[List[tk.StringVar]] = []
        self._updating = False

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the cell grid UI."""
        # Create a canvas with scrollbars for the grid
        canvas_frame = ttk.Frame(self)
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(
            canvas_frame,
            bg=self.theme.background,
            highlightthickness=0
        )

        # Scrollbars
        v_scroll = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self._canvas.yview)
        h_scroll = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self._canvas.xview)

        self._canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # Grid layout
        self._canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        canvas_frame.grid_rowconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(0, weight=1)

        # Inner frame for cells
        self._inner_frame = ttk.Frame(self._canvas)
        self._canvas.create_window((0, 0), window=self._inner_frame, anchor="nw")

        # Create cell entries
        for r in range(self.rows):
            row_cells = []
            row_vars = []
            for c in range(self.cols):
                var = tk.StringVar(value="0.0")
                entry = ttk.Entry(
                    self._inner_frame,
                    textvariable=var,
                    width=6,
                    justify='center'
                )
                entry.grid(row=r, column=c, padx=1, pady=1)
                entry.bind("<Return>", lambda e, row=r, col=c: self._on_cell_submit(row, col))
                entry.bind("<FocusOut>", lambda e, row=r, col=c: self._on_cell_submit(row, col))
                row_cells.append(entry)
                row_vars.append(var)
            self._cells.append(row_cells)
            self._cell_vars.append(row_vars)

        # Update scroll region
        self._inner_frame.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_cell_submit(self, row: int, col: int) -> None:
        """Handle cell value submission."""
        if self._updating:
            return

        var = self._cell_vars[row][col]
        text = var.get().strip()

        if text.lower() in ('', 'none', 'null', 'nan'):
            value = None
        else:
            try:
                value = float(text)
            except ValueError:
                # Reset to previous value or 0
                var.set("0.0")
                return

        if self.on_cell_change:
            self.on_cell_change(row, col, value)

    def set_cell_value(self, row: int, col: int, value: Optional[float]) -> None:
        """Set a cell's display value."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self._updating = True
            if value is None:
                self._cell_vars[row][col].set("None")
            else:
                self._cell_vars[row][col].set(f"{value:.3f}")
            self._updating = False

    def get_cell_value(self, row: int, col: int) -> Optional[float]:
        """Get a cell's current value."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            text = self._cell_vars[row][col].get().strip()
            if text.lower() in ('', 'none', 'null', 'nan'):
                return None
            try:
                return float(text)
            except ValueError:
                return None
        return None

    def update_from_matrix(self, matrix: "VGMatrix2D") -> None:
        """Update all cells from a VGMatrix2D."""
        self._updating = True
        for r in range(min(self.rows, matrix.rows)):
            for c in range(min(self.cols, matrix.cols)):
                value = matrix.get_value_at(r, c)
                if value is None:
                    self._cell_vars[r][c].set("None")
                else:
                    self._cell_vars[r][c].set(f"{value:.3f}")
        self._updating = False


class FilterParameterWidget(ttk.Frame):
    """Widget for a single filter parameter input."""

    def __init__(
        self,
        parent: tk.Widget,
        name: str,
        label: str,
        param_type: str,
        default: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        step: Optional[float] = None,
        choices: Optional[List[str]] = None,
        on_change: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        """
        Initialize the parameter widget.

        Args:
            parent: Parent widget.
            name: Parameter name.
            label: Display label.
            param_type: Type of parameter ("int", "float", "choice", "bool").
            default: Default value.
            min_value: Minimum value (for numeric types).
            max_value: Maximum value (for numeric types).
            step: Step size (for numeric types).
            choices: List of choices (for choice type).
            on_change: Callback when value changes.
        """
        super().__init__(parent, style="Card.TFrame", **kwargs)

        self.name = name
        self.param_type = param_type
        self.on_change = on_change
        self.min_value = min_value
        self.max_value = max_value
        self.step = step or 1.0

        # Create variable based on type
        if param_type == "int":
            self.variable = tk.IntVar(value=int(default) if default is not None else 0)
        elif param_type == "float":
            self.variable = tk.DoubleVar(value=float(default) if default is not None else 0.0)
        elif param_type == "bool":
            self.variable = tk.BooleanVar(value=bool(default) if default is not None else False)
        else:  # choice or string
            self.variable = tk.StringVar(value=str(default) if default is not None else "")

        self._build_ui(label, choices)

    def _build_ui(self, label: str, choices: Optional[List[str]]) -> None:
        """Build the widget UI based on parameter type."""
        # Label
        ttk.Label(
            self,
            text=label,
            style="Card.TLabel",
            width=15
        ).pack(side=tk.LEFT, padx=(0, 5))

        # Input widget
        if self.param_type == "bool":
            widget = ttk.Checkbutton(
                self,
                variable=self.variable,
                command=self._notify_change
            )
        elif self.param_type == "choice" and choices:
            widget = ttk.Combobox(
                self,
                textvariable=self.variable,
                values=choices,
                state="readonly",
                width=12
            )
            widget.bind("<<ComboboxSelected>>", lambda e: self._notify_change())
        else:
            # Numeric entry with validation
            widget = ttk.Entry(
                self,
                textvariable=self.variable,
                width=10,
                justify='center'
            )
            widget.bind("<Return>", lambda e: self._validate_and_notify())
            widget.bind("<FocusOut>", lambda e: self._validate_and_notify())

        widget.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def _validate_and_notify(self) -> None:
        """Validate numeric input and notify of change."""
        try:
            if self.param_type == "int":
                value = int(float(self.variable.get()))
            else:
                value = float(self.variable.get())

            # Clamp to range
            if self.min_value is not None:
                value = max(self.min_value, value)
            if self.max_value is not None:
                value = min(self.max_value, value)

            self.variable.set(value)
            self._notify_change()
        except (ValueError, tk.TclError):
            pass

    def _notify_change(self) -> None:
        """Notify that value has changed."""
        if self.on_change:
            self.on_change()

    def get_value(self) -> Any:
        """Get the current parameter value."""
        return self.variable.get()


class ScrollableFrame(ttk.Frame):
    """A scrollable frame container."""

    def __init__(self, parent: tk.Widget, bg_color: str = "#1e1e1e", **kwargs):
        """
        Initialize the scrollable frame.

        Args:
            parent: Parent widget.
            bg_color: Background color.
        """
        super().__init__(parent, **kwargs)

        # Create canvas
        self._canvas = tk.Canvas(self, bg=bg_color, highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._canvas.yview)

        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        # Pack widgets
        self._scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Create frame inside canvas
        self.scrollable_frame = ttk.Frame(self._canvas)
        self._window = self._canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Bind events
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Enable mouse wheel scrolling
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self._canvas.bind_all("<Button-4>", self._on_mousewheel)
        self._canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_frame_configure(self, event=None) -> None:
        """Update scroll region when frame size changes."""
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event=None) -> None:
        """Update frame width when canvas size changes."""
        self._canvas.itemconfig(self._window, width=event.width)

    def _on_mousewheel(self, event) -> None:
        """Handle mouse wheel scrolling."""
        if event.num == 4 or event.delta > 0:
            self._canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self._canvas.yview_scroll(1, "units")


class Card(ttk.Frame):
    """A card-style container with title."""

    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        collapsible: bool = False,
        **kwargs
    ):
        """
        Initialize the card.

        Args:
            parent: Parent widget.
            title: Card title.
            collapsible: Whether the card can be collapsed.
        """
        super().__init__(parent, style="Card.TFrame", **kwargs)

        self._collapsed = False
        self._collapsible = collapsible

        # Title frame
        title_frame = ttk.Frame(self, style="Card.TFrame")
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))

        if collapsible:
            self._collapse_btn = ttk.Label(
                title_frame,
                text="▼",
                style="Card.TLabel",
                cursor="hand2"
            )
            self._collapse_btn.pack(side=tk.LEFT, padx=(0, 5))
            self._collapse_btn.bind("<Button-1>", self._toggle_collapse)

        ttk.Label(
            title_frame,
            text=title,
            style="Section.TLabel"
        ).pack(side=tk.LEFT)

        # Content frame
        self.content = ttk.Frame(self, style="Card.TFrame")
        self.content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _toggle_collapse(self, event=None) -> None:
        """Toggle card collapsed state."""
        if not self._collapsible:
            return

        self._collapsed = not self._collapsed

        if self._collapsed:
            self.content.pack_forget()
            self._collapse_btn.configure(text="▶")
        else:
            self.content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            self._collapse_btn.configure(text="▼")


class StatusBar(ttk.Frame):
    """Status bar for displaying information."""

    def __init__(self, parent: tk.Widget, **kwargs):
        """Initialize the status bar."""
        super().__init__(parent, **kwargs)

        self._message_var = tk.StringVar(value="Ready")
        self._info_var = tk.StringVar(value="")

        # Message label
        ttk.Label(
            self,
            textvariable=self._message_var,
            style="Muted.TLabel"
        ).pack(side=tk.LEFT, padx=5)

        # Info label (right side)
        ttk.Label(
            self,
            textvariable=self._info_var,
            style="Muted.TLabel"
        ).pack(side=tk.RIGHT, padx=5)

    def set_message(self, message: str) -> None:
        """Set the status message."""
        self._message_var.set(message)

    def set_info(self, info: str) -> None:
        """Set the info text."""
        self._info_var.set(info)

