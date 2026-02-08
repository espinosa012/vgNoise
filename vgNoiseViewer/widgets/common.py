"""
Custom widgets for vgNoise Viewer.

This module contains reusable custom widget classes.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List, Any
import sys
from pathlib import Path

# Handle both package and direct execution imports
try:
    from ..core.config import ParameterConfig, ThemeColors
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import ParameterConfig, ThemeColors


# Re-export ParameterConfig for convenience
__all__ = [
    'StepperControl',
    'LabeledCombobox',
    'LabeledSpinbox',
    'Card',
    'ScrollableFrame',
    'ParameterConfig',
]


class StepperControl(ttk.Frame):
    """A numeric input control with +/- buttons and direct entry."""

    def __init__(
        self,
        parent: tk.Widget,
        config: ParameterConfig,
        variable: tk.Variable,
        on_change: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        """
        Initialize the stepper control.

        Args:
            parent: Parent widget.
            config: Parameter configuration.
            variable: Tkinter variable to bind to.
            on_change: Callback when value changes.
        """
        super().__init__(parent, style="Card.TFrame", **kwargs)

        self.config = config
        self.variable = variable
        self.on_change = on_change
        self._updating = False

        self._value_var = tk.StringVar()
        self._build_ui()
        self._bind_events()
        self._update_display()

    def _build_ui(self) -> None:
        """Build the widget UI."""
        # Label
        ttk.Label(
            self,
            text=self.config.label,
            style="Card.TLabel",
            width=18
        ).pack(side=tk.LEFT)

        # Control frame
        control_frame = ttk.Frame(self, style="Card.TFrame")
        control_frame.pack(side=tk.RIGHT)

        # Minus button
        self._minus_btn = ttk.Button(
            control_frame,
            text="−",
            width=3,
            command=self._decrease
        )
        self._minus_btn.pack(side=tk.LEFT, padx=(0, 2))

        # Value entry
        value_frame = ttk.Frame(control_frame, style="Card.TFrame")
        value_frame.pack(side=tk.LEFT, padx=2)

        self._entry = ttk.Entry(
            value_frame,
            textvariable=self._value_var,
            width=8,
            justify='center'
        )
        self._entry.pack()

        # Plus button
        self._plus_btn = ttk.Button(
            control_frame,
            text="+",
            width=3,
            command=self._increase
        )
        self._plus_btn.pack(side=tk.LEFT, padx=(2, 0))

    def _bind_events(self) -> None:
        """Bind event handlers."""
        self.variable.trace_add("write", self._on_variable_change)
        self._entry.bind("<Return>", self._on_entry_submit)
        self._entry.bind("<FocusOut>", self._on_entry_submit)

    def _decrease(self) -> None:
        """Decrease the value by step."""
        if self._updating:
            return
        current = self.variable.get()
        new_val = self.config.validate(current - self.config.step)
        new_val = self.config.round_to_step(new_val)
        self.variable.set(new_val)
        self._notify_change()

    def _increase(self) -> None:
        """Increase the value by step."""
        if self._updating:
            return
        current = self.variable.get()
        new_val = self.config.validate(current + self.config.step)
        new_val = self.config.round_to_step(new_val)
        self.variable.set(new_val)
        self._notify_change()

    def _update_display(self, *args) -> None:
        """Update the display value."""
        self._value_var.set(self.config.format_str.format(self.variable.get()))

    def _on_variable_change(self, *args) -> None:
        """Handle variable change."""
        self._update_display()

    def _on_entry_submit(self, event=None) -> None:
        """Handle entry submission."""
        if self._updating:
            return
        try:
            new_val = float(self._value_var.get())
            new_val = self.config.validate(new_val)
            new_val = self.config.round_to_step(new_val)
            self._updating = True
            self.variable.set(new_val)
            self._updating = False
            self._notify_change()
        except ValueError:
            self._update_display()

    def _notify_change(self) -> None:
        """Notify that value has changed."""
        if self.on_change and not self._updating:
            self.on_change()


class LabeledCombobox(ttk.Frame):
    """A combobox with a label."""

    def __init__(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.Variable,
        values: List[Any],
        on_change: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        """
        Initialize the labeled combobox.

        Args:
            parent: Parent widget.
            label: Label text.
            variable: Tkinter variable to bind to.
            values: List of values for the combobox.
            on_change: Callback when selection changes.
        """
        super().__init__(parent, style="Card.TFrame", **kwargs)

        self.variable = variable
        self.on_change = on_change

        # Label
        ttk.Label(
            self,
            text=label,
            style="Card.TLabel",
            width=18
        ).pack(side=tk.LEFT)

        # Combobox
        self._combo = ttk.Combobox(
            self,
            textvariable=variable,
            values=values,
            state="readonly",
            width=15
        )
        self._combo.pack(side=tk.RIGHT)
        self._combo.bind("<<ComboboxSelected>>", self._on_select)

    def _on_select(self, event=None) -> None:
        """Handle selection change."""
        if self.on_change:
            self.on_change()


class LabeledSpinbox(ttk.Frame):
    """A spinbox with a label."""

    def __init__(
        self,
        parent: tk.Widget,
        config: ParameterConfig,
        variable: tk.Variable,
        on_change: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        """
        Initialize the labeled spinbox.

        Args:
            parent: Parent widget.
            config: Parameter configuration.
            variable: Tkinter variable to bind to.
            on_change: Callback when value changes.
        """
        super().__init__(parent, style="Card.TFrame", **kwargs)

        self.config = config
        self.variable = variable
        self.on_change = on_change

        # Label
        ttk.Label(
            self,
            text=config.label,
            style="Card.TLabel",
            width=18
        ).pack(side=tk.LEFT)

        # Spinbox
        self._spinbox = ttk.Spinbox(
            self,
            from_=config.min_value,
            to=config.max_value,
            textvariable=variable,
            width=15,
            increment=config.step
        )
        self._spinbox.pack(side=tk.RIGHT)
        self._spinbox.bind("<Return>", self._on_submit)
        self._spinbox.bind("<FocusOut>", self._on_submit)

    def _on_submit(self, event=None) -> None:
        """Handle submission."""
        if self.on_change:
            self.on_change()


class Card(ttk.Frame):
    """A card-style container with title."""

    def __init__(
        self,
        parent: tk.Widget,
        title: str,
        **kwargs
    ):
        """
        Initialize the card.

        Args:
            parent: Parent widget.
            title: Card title.
        """
        super().__init__(parent, style="Card.TFrame", padding=10, **kwargs)

        self.title = title
        self.content_frame: Optional[ttk.Frame] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the card UI."""
        # Title
        ttk.Label(
            self,
            text=self.title,
            style="Section.TLabel"
        ).pack(anchor=tk.W, pady=(0, 10))

        # Content frame
        self.content_frame = ttk.Frame(self, style="Card.TFrame")
        self.content_frame.pack(fill=tk.BOTH, expand=True)


class ScrollableFrame(ttk.Frame):
    """A scrollable frame container."""

    def __init__(
        self,
        parent: tk.Widget,
        bg_color: str = "#1e1e1e",
        **kwargs
    ):
        """
        Initialize the scrollable frame.

        Args:
            parent: Parent widget.
            bg_color: Background color for the canvas.
        """
        super().__init__(parent, **kwargs)

        self.bg_color = bg_color
        self.scrollable_frame: Optional[ttk.Frame] = None
        self._canvas: Optional[tk.Canvas] = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the scrollable frame UI."""
        # Canvas
        self._canvas = tk.Canvas(
            self,
            bg=self.bg_color,
            highlightthickness=0
        )

        # Scrollbar
        scrollbar = ttk.Scrollbar(
            self,
            orient=tk.VERTICAL,
            command=self._canvas.yview
        )

        # Scrollable frame
        self.scrollable_frame = ttk.Frame(self._canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        )

        self._canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self._canvas.configure(yscrollcommand=scrollbar.set)

        # Enable mousewheel scrolling
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # Pack
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def _on_mousewheel(self, event) -> None:
        """Handle mousewheel scrolling."""
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
