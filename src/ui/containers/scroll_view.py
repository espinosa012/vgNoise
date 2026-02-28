"""
ScrollView container for scrollable content.
"""

from typing import Optional, Tuple

import pygame

from .container import Container
from ..widget import Widget


class ScrollView(Container):
    """
    A scrollable container widget.

    Allows content larger than the visible area to be scrolled.

    Features:
    - Vertical and horizontal scrolling
    - Mouse wheel support
    - Draggable scrollbar thumb
    - Clipping of content to visible area
    - Correct hit-testing for children at any scroll position

    How it works:
        Children are positioned in *content space* (starting at 0,0).
        ScrollView overrides ``_get_scroll_offset_for_children`` so that the
        engine's absolute-position calculation automatically shifts each
        child by ``(padding - scroll_x, padding - scroll_y)``.  This means
        ``contains_point``, ``absolute_rect`` and all event handling work
        correctly without any temporary position mutation.

    Example:
        scroll = ScrollView(
            x=10, y=10,
            width=200, height=300,
            show_scrollbar=True
        )
        for i in range(20):
            scroll.add_child(Label(x=0, y=i*30, text=f"Item {i+1}"))
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 200,
        height: int = 200,
        content_width: int = 0,   # 0 = same as viewport width
        content_height: int = 0,  # 0 = auto-calculate from children
        scroll_x: float = 0,
        scroll_y: float = 0,
        scroll_speed: int = 20,
        show_scrollbar: bool = True,
        scrollbar_width: int = 10,
        bg_color: Optional[Tuple[int, int, int, int]] = None,
        scrollbar_color: Optional[Tuple[int, int, int]] = None,
        scrollbar_track_color: Optional[Tuple[int, int, int]] = None,
        border_radius: int = 0,
        border_width: int = 0,
        border_color: Optional[Tuple[int, int, int]] = None,
        padding: int = 0,
        parent: Optional[Widget] = None,
    ):
        """
        Initialize a ScrollView.

        Args:
            x: X position.
            y: Y position.
            width: Viewport width.
            height: Viewport height.
            content_width: Total content width (0 = viewport width).
            content_height: Total content height (0 = auto from children).
            scroll_x: Initial horizontal scroll offset.
            scroll_y: Initial vertical scroll offset.
            scroll_speed: Pixels to scroll per mouse wheel tick.
            show_scrollbar: Whether to show the vertical scrollbar.
            scrollbar_width: Width of scrollbar in pixels.
            bg_color: Background color.
            scrollbar_color: Scrollbar thumb color.
            scrollbar_track_color: Scrollbar track color.
            border_radius: Corner radius.
            border_width: Border width.
            border_color: Border color.
            padding: Internal padding.
            parent: Parent widget.
        """
        # auto_size is always False for ScrollView (it has a fixed viewport size)
        super().__init__(
            x, y, width, height,
            bg_color, border_radius, border_width, border_color,
            padding, False, False, parent
        )

        self._content_width = content_width
        self._content_height = content_height
        self._scroll_x = float(scroll_x)
        self._scroll_y = float(scroll_y)
        self._scroll_speed = scroll_speed
        self._show_scrollbar = show_scrollbar
        self._scrollbar_width = scrollbar_width
        self._scrollbar_color = scrollbar_color
        self._scrollbar_track_color = scrollbar_track_color

        # Scrollbar drag state
        self._dragging_scrollbar = False
        self._drag_start_y = 0
        self._scroll_start_y = 0.0

    # -------------------------------------------------------------------------
    # Scroll offset — the core of the correct-position architecture
    # -------------------------------------------------------------------------

    def _get_scroll_offset_for_children(self) -> tuple[int, int]:
        """
        Shift children's absolute positions by the current scroll offset.

        This is called by Widget.get_absolute_position for every direct child,
        so absolute_rect, contains_point, and draw are all automatically
        correct without any temporary position mutation.
        """
        return self._padding - int(self._scroll_x), self._padding - int(self._scroll_y)

    # -------------------------------------------------------------------------
    # Scroll properties
    # -------------------------------------------------------------------------

    @property
    def scroll_x(self) -> float:
        """Horizontal scroll offset in pixels."""
        return self._scroll_x

    @scroll_x.setter
    def scroll_x(self, value: float) -> None:
        max_scroll = max(0.0, self.actual_content_width - self.viewport_width)
        self._scroll_x = max(0.0, min(float(value), max_scroll))

    @property
    def scroll_y(self) -> float:
        """Vertical scroll offset in pixels."""
        return self._scroll_y

    @scroll_y.setter
    def scroll_y(self, value: float) -> None:
        max_scroll = max(0.0, self.actual_content_height - self.viewport_height)
        self._scroll_y = max(0.0, min(float(value), max_scroll))

    # -------------------------------------------------------------------------
    # Viewport / content geometry
    # -------------------------------------------------------------------------

    @property
    def viewport_width(self) -> int:
        """Visible area width (may be reduced by scrollbar)."""
        w = self.content_width
        if self._show_scrollbar and self._needs_vertical_scroll():
            w -= self._scrollbar_width
        return max(0, w)

    @property
    def viewport_height(self) -> int:
        """Visible area height (may be reduced by horizontal scrollbar)."""
        h = self.content_height
        if self._show_scrollbar and self._needs_horizontal_scroll():
            h -= self._scrollbar_width
        return max(0, h)

    @property
    def actual_content_width(self) -> int:
        """Total content width (explicit or calculated from children)."""
        if self._content_width > 0:
            return self._content_width
        max_x = 0
        for child in self._children:
            max_x = max(max_x, child.x + child.width)
        return max_x

    @property
    def actual_content_height(self) -> int:
        """Total content height (explicit or calculated from children)."""
        if self._content_height > 0:
            return self._content_height
        max_y = 0
        for child in self._children:
            max_y = max(max_y, child.y + child.height)
        return max_y

    def _needs_horizontal_scroll(self) -> bool:
        # Compare against content_width (not viewport_width) to avoid circular recursion.
        return self.actual_content_width > self.content_width

    def _needs_vertical_scroll(self) -> bool:
        # Compare against content_height (not viewport_height) to avoid circular recursion.
        return self.actual_content_height > self.content_height

    # -------------------------------------------------------------------------
    # Scroll helpers
    # -------------------------------------------------------------------------

    def scroll_to(self, x: Optional[float] = None, y: Optional[float] = None) -> None:
        """Scroll to a specific position. Pass None to leave an axis unchanged."""
        if x is not None:
            self.scroll_x = x
        if y is not None:
            self.scroll_y = y

    def scroll_to_top(self) -> None:
        """Scroll to the top."""
        self.scroll_y = 0.0

    def scroll_to_bottom(self) -> None:
        """Scroll to the bottom."""
        self.scroll_y = max(0.0, self.actual_content_height - self.viewport_height)

    def scroll_to_widget(self, widget: Widget) -> None:
        """Scroll to make a child widget fully visible."""
        if widget not in self._children:
            return
        if widget.y < self._scroll_y:
            self.scroll_y = float(widget.y)
        elif widget.y + widget.height > self._scroll_y + self.viewport_height:
            self.scroll_y = float(widget.y + widget.height - self.viewport_height)

    # -------------------------------------------------------------------------
    # Scrollbar geometry
    # -------------------------------------------------------------------------

    def _get_scrollbar_track_rect(self) -> pygame.Rect:
        """Absolute rect of the vertical scrollbar track."""
        abs_rect = self.absolute_rect
        return pygame.Rect(
            abs_rect.right - self._padding - self._scrollbar_width,
            abs_rect.y + self._padding,
            self._scrollbar_width,
            self.content_height
        )

    def _get_scrollbar_thumb_rect(self) -> Optional[pygame.Rect]:
        """Absolute rect of the vertical scrollbar thumb, or None."""
        if not self._needs_vertical_scroll():
            return None

        content_h = self.actual_content_height
        viewport_h = self.viewport_height
        track_h = self.content_height

        thumb_h = max(20, int(track_h * (viewport_h / content_h)))
        max_scroll = content_h - viewport_h
        ratio = self._scroll_y / max_scroll if max_scroll > 0 else 0.0
        thumb_y = int(ratio * (track_h - thumb_h))

        track = self._get_scrollbar_track_rect()
        return pygame.Rect(track.x, track.y + thumb_y, self._scrollbar_width, thumb_h)

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self._state.visible or not self._state.enabled:
            return False

        # --- Scrollbar drag ---
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self._show_scrollbar and self._needs_vertical_scroll():
                thumb = self._get_scrollbar_thumb_rect()
                if thumb and thumb.collidepoint(event.pos):
                    self._dragging_scrollbar = True
                    self._drag_start_y = event.pos[1]
                    self._scroll_start_y = self._scroll_y
                    return True

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._dragging_scrollbar:
                self._dragging_scrollbar = False
                return True

        elif event.type == pygame.MOUSEMOTION:
            if self._dragging_scrollbar:
                delta_y = event.pos[1] - self._drag_start_y
                track_h = self.content_height
                content_h = self.actual_content_height
                viewport_h = self.viewport_height
                thumb_h = max(20, int(track_h * (viewport_h / content_h)))
                scrollable_track = track_h - thumb_h
                if scrollable_track > 0:
                    scroll_range = content_h - viewport_h
                    self.scroll_y = self._scroll_start_y + delta_y * (scroll_range / scrollable_track)
                return True

        # --- Mouse wheel ---
        if event.type == pygame.MOUSEWHEEL:
            if self.contains_point(*pygame.mouse.get_pos()):
                self.scroll_y = self._scroll_y - event.y * self._scroll_speed
                return True

        # --- Dispatch to children ---
        # Only forward mouse positional events when the pointer is inside the
        # viewport, so that clipped (invisible) children don't receive them.
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION):
            abs_rect = self.absolute_rect
            viewport_rect = pygame.Rect(
                abs_rect.x + self._padding,
                abs_rect.y + self._padding,
                self.viewport_width,
                self.viewport_height
            )
            if not viewport_rect.collidepoint(event.pos):
                # Still update hover state for the container itself
                self._handle_mouse_motion(event) if event.type == pygame.MOUSEMOTION else None
                return False

        for child in reversed(self._children):
            if child.handle_event(event):
                return True

        # Fall back to container-level events (hover, etc.)
        return super().handle_event(event)

    # -------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the scroll view with clipped children."""
        if not self.visible:
            return

        abs_rect = self.absolute_rect

        # Background
        if self._bg_color:
            if len(self._bg_color) == 4 and self._bg_color[3] < 255:
                temp = pygame.Surface((abs_rect.width, abs_rect.height), pygame.SRCALPHA)
                if self._border_radius > 0:
                    pygame.draw.rect(
                        temp, self._bg_color,
                        pygame.Rect(0, 0, abs_rect.width, abs_rect.height),
                        border_radius=self._border_radius
                    )
                else:
                    temp.fill(self._bg_color)
                surface.blit(temp, (abs_rect.x, abs_rect.y))
            else:
                if self._border_radius > 0:
                    pygame.draw.rect(surface, self._bg_color[:3], abs_rect,
                                     border_radius=self._border_radius)
                else:
                    pygame.draw.rect(surface, self._bg_color[:3], abs_rect)

        # Clip to viewport and draw children.
        # Because _get_scroll_offset_for_children is overridden, every child's
        # absolute_rect already reflects the current scroll position — no
        # temporary position mutation needed.
        viewport_rect = pygame.Rect(
            abs_rect.x + self._padding,
            abs_rect.y + self._padding,
            self.viewport_width,
            self.viewport_height
        )
        old_clip = surface.get_clip()
        surface.set_clip(viewport_rect.clip(old_clip))

        for child in self._children:
            if child.visible:
                # Fast cull: skip children entirely outside the viewport
                if viewport_rect.colliderect(child.absolute_rect):
                    child.draw(surface)

        surface.set_clip(old_clip)

        # Vertical scrollbar
        if self._show_scrollbar and self._needs_vertical_scroll():
            track = self._get_scrollbar_track_rect()
            track_color = self._scrollbar_track_color or (40, 40, 40)
            pygame.draw.rect(surface, track_color, track,
                             border_radius=self._scrollbar_width // 2)

            thumb = self._get_scrollbar_thumb_rect()
            if thumb:
                thumb_color = self._scrollbar_color or (100, 100, 100)
                if self._dragging_scrollbar:
                    thumb_color = tuple(min(255, c + 40) for c in thumb_color)
                pygame.draw.rect(surface, thumb_color, thumb,
                                 border_radius=self._scrollbar_width // 2)

        # Border
        if self._border_width > 0:
            border_color = self._border_color or (80, 80, 80)
            pygame.draw.rect(surface, border_color, abs_rect,
                             width=self._border_width,
                             border_radius=self._border_radius)