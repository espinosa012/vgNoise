"""
Checkbox widget for boolean input.
"""

from typing import Optional, Tuple, Callable

import pygame

from ..widget import Widget


class Checkbox(Widget):
    """
    A checkbox widget for boolean input.

    Keyboard: Space/Enter toggles when focused. Tab passes through to UIManager.

    Features:
    - Check/uncheck toggle
    - Optional label text
    - Visual feedback for hover and disabled states
    - Change callback

    Example:
        checkbox = Checkbox(
            x=10, y=10,
            text="Enable feature",
            checked=True,
            on_change=lambda cb: print(f"Checked: {cb.checked}")
        )
    """

    _focusable = True

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        text: str = "",
        checked: bool = False,
        box_size: int = 18,
        spacing: int = 8,
        check_color: Optional[Tuple[int, int, int]] = None,
        box_color: Optional[Tuple[int, int, int]] = None,
        text_color: Optional[Tuple[int, int, int]] = None,
        font_size: Optional[int] = None,
        parent: Optional[Widget] = None,
    ):
        """
        Initialize a Checkbox.

        Args:
            x: X position.
            y: Y position.
            text: Label text displayed next to checkbox.
            checked: Initial checked state.
            box_size: Size of the checkbox square.
            spacing: Space between checkbox and label.
            check_color: Color of the checkmark (uses theme primary if None).
            box_color: Color of the checkbox box (uses theme surface if None).
            text_color: Color of the label text (uses theme text if None).
            font_size: Font size for label (uses theme default if None).
            parent: Parent widget.
            parent: Parent widget.
        """
        # Calculate initial size based on text
        height = box_size
        width = box_size

        super().__init__(x, y, width, height, parent)

        self._text = text
        self._checked = checked
        self._box_size = box_size
        self._spacing = spacing
        self._check_color = check_color
        self._box_color = box_color
        self._text_color = text_color
        self._font_size = font_size

        # Change callback
        self._on_change: Optional[Callable[[Checkbox], None]] = None

        # Cached text surface
        self._rendered_text: Optional[pygame.Surface] = None
        self._needs_render = True

        # Update size to include text
        self._update_size()

    @property
    def checked(self) -> bool:
        """Get the checked state."""
        return self._checked

    @checked.setter
    def checked(self, value: bool) -> None:
        """Set the checked state."""
        if self._checked != value:
            self._checked = value
            if self._on_change:
                self._on_change(self)

    @property
    def text(self) -> str:
        """Get the label text."""
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        """Set the label text."""
        if self._text != value:
            self._text = value
            self._needs_render = True
            self._update_size()

    def on_change(self, callback: Callable[['Checkbox'], None]) -> 'Checkbox':
        """
        Set the change callback.

        Args:
            callback: Function to call when checked state changes.

        Returns:
            Self for method chaining.
        """
        self._on_change = callback
        return self

    def toggle(self) -> None:
        """Toggle the checked state."""
        self.checked = not self._checked

    def _get_font(self) -> pygame.font.Font:
        """Get the pygame font object."""
        size = self._font_size or 16
        return pygame.font.Font(None, size)

    def _update_size(self) -> None:
        """Update widget size based on text."""
        if self._text:
            font = self._get_font()
            text_width, text_height = font.size(self._text)
            self._rect.width = self._box_size + self._spacing + text_width
            self._rect.height = max(self._box_size, text_height)
        else:
            self._rect.width = self._box_size
            self._rect.height = self._box_size

    def _render_text(self) -> None:
        """Render the label text."""
        if not self._text:
            self._rendered_text = None
            return

        font = self._get_font()

        # Get text color
        if not self.enabled:
            color = (100, 100, 100)
        elif self._text_color:
            color = self._text_color
        else:
            color = (255, 255, 255)

        self._rendered_text = font.render(self._text, True, color)
        self._needs_render = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Toggle with Space/Enter when focused; Tab falls through."""
        if not self._state.visible or not self._state.enabled:
            return False

        if self.focused and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                return False
            if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_KP_ENTER):
                self.toggle()
                return True
            if event.key == pygame.K_ESCAPE:
                self.blur()
                return True

        return super().handle_event(event)

    def _handle_mouse_up(self, event: pygame.event.Event) -> bool:
        """Handle mouse button up - toggle on click."""
        if event.button != 1:
            return False

        if self._state.pressed:
            self._state.pressed = False
            if self.contains_point(event.pos[0], event.pos[1]):
                self.toggle()
                return True

        return False

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the checkbox."""
        if not self.visible:
            return

        abs_x, abs_y = self.get_absolute_position()

        # Calculate box position (vertically centered)
        box_y = abs_y + (self.height - self._box_size) // 2
        box_rect = pygame.Rect(abs_x, box_y, self._box_size, self._box_size)

        # Get colors
        if not self.enabled:
            box_color = (60, 60, 60)
            border_color = (100, 100, 100)
        elif self.focused:
            box_color = self._box_color or (50, 50, 50)
            border_color = (66, 135, 245)  # Blue focus ring
        elif self.hovered:
            box_color = (70, 70, 70)
            border_color = (100, 160, 255)
        else:
            box_color = self._box_color or (50, 50, 50)
            border_color = (80, 80, 80)

        # Draw box background
        pygame.draw.rect(surface, box_color, box_rect, border_radius=3)

        # Draw box border
        pygame.draw.rect(surface, border_color, box_rect, width=2, border_radius=3)

        # Draw checkmark if checked
        if self._checked:
            check_color = self._check_color or (66, 135, 245)

            # Draw a checkmark using lines
            padding = self._box_size // 4
            points = [
                (box_rect.x + padding, box_rect.y + self._box_size // 2),
                (box_rect.x + self._box_size // 2 - 1, box_rect.y + self._box_size - padding - 2),
                (box_rect.x + self._box_size - padding, box_rect.y + padding)
            ]
            pygame.draw.lines(surface, check_color, False, points, width=2)

        # Draw label text
        if self._text:
            if self._needs_render:
                self._render_text()

            if self._rendered_text:
                text_x = abs_x + self._box_size + self._spacing
                text_y = abs_y + (self.height - self._rendered_text.get_height()) // 2
                surface.blit(self._rendered_text, (text_x, text_y))

        # Draw children
        self.draw_children(surface)

