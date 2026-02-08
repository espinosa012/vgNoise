"""
Theme management for vgNoise Viewer.

This module handles styling and theming for the Tkinter application.
"""

from tkinter import ttk
from typing import Optional

# Handle both package and direct execution imports
try:
    from .config import ThemeColors
except ImportError:
    from config import ThemeColors


class ThemeManager:
    """Manages application theming and styles."""

    def __init__(self, colors: Optional[ThemeColors] = None):
        """
        Initialize the theme manager.

        Args:
            colors: Optional custom color scheme. Uses default dark theme if not provided.
        """
        self.colors = colors or ThemeColors()
        self._style: Optional[ttk.Style] = None

    def apply(self, style: ttk.Style) -> None:
        """
        Apply the theme to the given style object.

        Args:
            style: The ttk.Style object to configure.
        """
        self._style = style
        style.theme_use('clam')

        self._configure_base_styles()
        self._configure_frame_styles()
        self._configure_label_styles()
        self._configure_input_styles()
        self._configure_button_styles()

    def _configure_base_styles(self) -> None:
        """Configure base styles."""
        self._style.configure(
            ".",
            background=self.colors.background,
            foreground=self.colors.foreground,
            fieldbackground=self.colors.card
        )

    def _configure_frame_styles(self) -> None:
        """Configure frame styles."""
        self._style.configure("TFrame", background=self.colors.background)
        self._style.configure("Card.TFrame", background=self.colors.card)

    def _configure_label_styles(self) -> None:
        """Configure label styles."""
        self._style.configure(
            "TLabel",
            background=self.colors.background,
            foreground=self.colors.foreground
        )
        self._style.configure(
            "Card.TLabel",
            background=self.colors.card,
            foreground=self.colors.foreground
        )
        self._style.configure(
            "Header.TLabel",
            background=self.colors.background,
            foreground=self.colors.foreground,
            font=('Segoe UI', 16, 'bold')
        )
        self._style.configure(
            "Section.TLabel",
            background=self.colors.card,
            foreground=self.colors.foreground,
            font=('Segoe UI', 11, 'bold')
        )
        self._style.configure(
            "Muted.TLabel",
            background=self.colors.background,
            foreground=self.colors.muted
        )

    def _configure_input_styles(self) -> None:
        """Configure input widget styles."""
        self._style.configure(
            "TScale",
            background=self.colors.card,
            troughcolor=self.colors.background
        )
        self._style.configure(
            "TCombobox",
            fieldbackground=self.colors.card,
            background=self.colors.card
        )
        self._style.configure(
            "TSpinbox",
            fieldbackground=self.colors.card,
            background=self.colors.card
        )
        self._style.configure(
            "TEntry",
            fieldbackground=self.colors.card,
            background=self.colors.card
        )

    def _configure_button_styles(self) -> None:
        """Configure button styles."""
        self._style.configure(
            "TButton",
            background=self.colors.card,
            foreground=self.colors.foreground
        )
        self._style.configure(
            "Accent.TButton",
            background=self.colors.accent,
            foreground=self.colors.foreground
        )
        self._style.map(
            "Accent.TButton",
            background=[('active', self.colors.accent_hover)]
        )
