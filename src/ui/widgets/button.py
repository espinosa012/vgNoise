"""
Button widget for interactive user input.
"""

from typing import Optional, Tuple

import pygame

from ..widget import Widget


class Button(Widget):
    """
    An interactive button widget.

    Features:
    - Visual feedback for hover, pressed, and disabled states
    - Configurable colors and border radius
    - Text label with alignment
    - Click callback support

    Example:
        button = Button(
            x=10, y=10,
            width=120, height=40,
            text="Click Me",
            on_click=lambda btn: print("Clicked!")
        )
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 100,
        height: int = 32,
        text: str = "",
        font_size: Optional[int] = None,
        text_color: Optional[Tuple[int, int, int]] = None,
        bg_color: Optional[Tuple[int, int, int]] = None,
        hover_color: Optional[Tuple[int, int, int]] = None,
        pressed_color: Optional[Tuple[int, int, int]] = None,
        disabled_color: Optional[Tuple[int, int, int]] = None,
        border_radius: int = 4,
        border_width: int = 0,
        border_color: Optional[Tuple[int, int, int]] = None,
        parent: Optional[Widget] = None
    ):
        """
        Initialize a Button.

        Args:
            x: X position.
            y: Y position.
            width: Button width.
            height: Button height.
            text: Button label text.
            font_size: Font size (default 16 if None).
            text_color: Text color (default white if None).
            bg_color: Background color (default blue if None).
            hover_color: Hover state color (default light blue if None).
            pressed_color: Pressed state color (default dark blue if None).
            disabled_color: Disabled state color (default gray if None).
            border_radius: Corner radius in pixels.
            border_width: Border width in pixels (0 = no border).
            border_color: Border color (default white if None).
            parent: Parent widget.
        """
        super().__init__(x, y, width, height, parent)

        self._text = text
        self._font_size = font_size
        self._text_color = text_color
        self._bg_color = bg_color
        self._hover_color = hover_color
        self._pressed_color = pressed_color
        self._disabled_color = disabled_color
        self._border_radius = border_radius
        self._border_width = border_width
        self._border_color = border_color

        # Cached rendered text (invalidated when text, size, or color state changes)
        self._rendered_text: Optional[pygame.Surface] = None
        self._needs_render = True
        self._last_render_color: Optional[tuple] = None

    @property
    def text(self) -> str:
        """Get the button text."""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Set the button text."""
        if self._text != value:
            self._text = value
            self._needs_render = True
            self._last_render_color = None

    def _get_current_bg_color(self) -> Tuple[int, int, int]:
        """Get the background color based on current state."""
        if not self.enabled:
            return self._disabled_color or (120, 120, 120)

        if self.pressed:
            return self._pressed_color or (40, 100, 200)

        if self.hovered:
            return self._hover_color or (100, 160, 255)

        return self._bg_color or (66, 135, 245)

    def _get_text_color(self) -> Tuple[int, int, int]:
        """Get the text color based on current state."""
        if not self.enabled:
            return (100, 100, 100)

        return self._text_color or (255, 255, 255)

    def _get_font(self) -> pygame.font.Font:
        """Get the pygame font object."""
        size = self._font_size or 16
        return pygame.font.Font(None, size)

    def _render_text(self) -> None:
        """Render the text to a surface, only when text or color has changed."""
        if not self._text:
            self._rendered_text = None
            self._last_render_color = None
            return

        color = self._get_text_color()
        if not self._needs_render and color == self._last_render_color:
            return  # Nothing changed — reuse the cached surface

        font = self._get_font()
        self._rendered_text = font.render(self._text, True, color)
        self._needs_render = False
        self._last_render_color = color

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the button."""
        if not self.visible:
            return

        abs_rect = self.absolute_rect

        # Draw background
        bg_color = self._get_current_bg_color()

        if self._border_radius > 0:
            pygame.draw.rect(
                surface,
                bg_color,
                abs_rect,
                border_radius=self._border_radius
            )
        else:
            pygame.draw.rect(surface, bg_color, abs_rect)

        # Draw border
        if self._border_width > 0:
            border_color = self._border_color or (255, 255, 255)

            if self.focused:
                border_color = (66, 135, 245)  # Blue for focused

            if self._border_radius > 0:
                pygame.draw.rect(
                    surface,
                    border_color,
                    abs_rect,
                    width=self._border_width,
                    border_radius=self._border_radius
                )
            else:
                pygame.draw.rect(
                    surface,
                    border_color,
                    abs_rect,
                    width=self._border_width
                )

        # Draw text (re-renders only when text or color state has changed)
        self._render_text()

        if self._rendered_text:
            # Center text in button
            text_rect = self._rendered_text.get_rect()
            text_rect.center = abs_rect.center
            surface.blit(self._rendered_text, text_rect)

        # Draw children
        self.draw_children(surface)

