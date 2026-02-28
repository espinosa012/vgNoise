"""
Base Widget class for the UI framework.

This module provides the fundamental building block for all UI components.
"""

from __future__ import annotations

import pygame
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Callable


@dataclass
class WidgetState:
    """
    Represents the current state of a widget.

    Attributes:
        visible: Whether the widget is visible.
        enabled: Whether the widget can receive input.
        focused: Whether the widget has keyboard focus.
        hovered: Whether the mouse is over the widget.
        pressed: Whether the widget is being pressed.
    """
    visible: bool = True
    enabled: bool = True
    focused: bool = False
    hovered: bool = False
    pressed: bool = False


class Widget(ABC):
    """
    Abstract base class for all UI widgets.

    Provides common functionality:
    - Position and size management with pygame.Rect
    - Parent-child hierarchy
    - State management (visible, enabled, focused, hovered, pressed)
    - Event handling interface
    - Theme support

    Subclasses must implement:
    - draw(): Render the widget
    - update(): Update widget logic
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 100,
        height: int = 30,
        parent: Optional[Widget] = None
    ):
        """
        Initialize a widget.

        Args:
            x: X position relative to parent (or screen if no parent).
            y: Y position relative to parent (or screen if no parent).
            width: Widget width in pixels.
            height: Widget height in pixels.
            parent: Parent widget (optional).
        """
        self._rect = pygame.Rect(x, y, width, height)
        self._state = WidgetState()
        self._parent: Optional[Widget] = None
        self._children: List[Widget] = []

        # Event callbacks
        self._on_click: Optional[Callable[[Widget], None]] = None
        self._on_hover_enter: Optional[Callable[[Widget], None]] = None
        self._on_hover_exit: Optional[Callable[[Widget], None]] = None
        self._on_focus: Optional[Callable[[Widget], None]] = None
        self._on_blur: Optional[Callable[[Widget], None]] = None

        # Set parent (this also adds self to parent's children)
        if parent:
            self.parent = parent

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def x(self) -> int:
        """X position relative to parent."""
        return self._rect.x

    @x.setter
    def x(self, value: int) -> None:
        self._rect.x = value

    @property
    def y(self) -> int:
        """Y position relative to parent."""
        return self._rect.y

    @y.setter
    def y(self, value: int) -> None:
        self._rect.y = value

    @property
    def width(self) -> int:
        """Widget width."""
        return self._rect.width

    @width.setter
    def width(self, value: int) -> None:
        self._rect.width = max(0, value)

    @property
    def height(self) -> int:
        """Widget height."""
        return self._rect.height

    @height.setter
    def height(self, value: int) -> None:
        self._rect.height = max(0, value)

    @property
    def size(self) -> tuple[int, int]:
        """Widget size as (width, height)."""
        return self._rect.width, self._rect.height

    @size.setter
    def size(self, value: tuple[int, int]) -> None:
        self._rect.width = max(0, value[0])
        self._rect.height = max(0, value[1])

    @property
    def position(self) -> tuple[int, int]:
        """Widget position as (x, y) relative to parent."""
        return self._rect.x, self._rect.y

    @position.setter
    def position(self, value: tuple[int, int]) -> None:
        self._rect.x = value[0]
        self._rect.y = value[1]

    @property
    def rect(self) -> pygame.Rect:
        """Local rect (position relative to parent)."""
        return self._rect.copy()

    @property
    def absolute_rect(self) -> pygame.Rect:
        """
        Absolute rect in screen coordinates.

        Calculates position by traversing up the parent hierarchy.
        """
        abs_x, abs_y = self.get_absolute_position()
        return pygame.Rect(abs_x, abs_y, self._rect.width, self._rect.height)

    @property
    def state(self) -> WidgetState:
        """Current widget state."""
        return self._state

    @property
    def visible(self) -> bool:
        """Whether the widget is visible."""
        return self._state.visible

    @visible.setter
    def visible(self, value: bool) -> None:
        self._state.visible = value

    @property
    def enabled(self) -> bool:
        """Whether the widget can receive input."""
        return self._state.enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._state.enabled = value

    @property
    def focused(self) -> bool:
        """Whether the widget has focus."""
        return self._state.focused

    @property
    def hovered(self) -> bool:
        """Whether the mouse is over the widget."""
        return self._state.hovered

    @property
    def pressed(self) -> bool:
        """Whether the widget is being pressed."""
        return self._state.pressed

    @property
    def parent(self) -> Optional[Widget]:
        """Parent widget."""
        return self._parent

    @parent.setter
    def parent(self, value: Optional[Widget]) -> None:
        # Remove from old parent
        if self._parent:
            self._parent._children.remove(self)

        self._parent = value

        # Add to new parent
        if value and self not in value._children:
            value._children.append(self)

    @property
    def children(self) -> List[Widget]:
        """List of child widgets (read-only copy)."""
        return self._children.copy()

    # -------------------------------------------------------------------------
    # Position Methods
    # -------------------------------------------------------------------------

    def get_absolute_position(self) -> tuple[int, int]:
        """
        Get the absolute position in screen coordinates.

        Traverses the parent hierarchy, accounting for any scroll offsets
        applied by parent containers (e.g. ScrollView).

        Returns:
            Tuple of (x, y) in screen coordinates.
        """
        if self._parent:
            parent_x, parent_y = self._parent.get_absolute_position()
            off_x, off_y = self._parent._get_scroll_offset_for_children()
            return self._rect.x + parent_x + off_x, self._rect.y + parent_y + off_y
        return self._rect.x, self._rect.y

    def _get_scroll_offset_for_children(self) -> tuple[int, int]:
        """
        Return the offset applied to direct children's absolute positions.

        Override in scrollable containers (e.g. ScrollView) to shift children
        by the current scroll amount so that hit-testing and drawing are
        always consistent.
        """
        return 0, 0

    def contains_point(self, x: int, y: int) -> bool:
        """
        Check if a point (in screen coordinates) is inside the widget.

        Args:
            x: X coordinate in screen space.
            y: Y coordinate in screen space.

        Returns:
            True if the point is inside the widget.
        """
        return self.absolute_rect.collidepoint(x, y)

    # -------------------------------------------------------------------------
    # Child Management
    # -------------------------------------------------------------------------

    def add_child(self, child: Widget) -> None:
        """
        Add a child widget.

        Args:
            child: Widget to add as child.
        """
        child.parent = self

    def remove_child(self, child: Widget) -> None:
        """
        Remove a child widget.

        Args:
            child: Widget to remove.
        """
        if child in self._children:
            child.parent = None

    def clear_children(self) -> None:
        """Remove all child widgets."""
        for child in self._children.copy():
            child.parent = None

    def get_child_at(self, x: int, y: int) -> Optional[Widget]:
        """
        Get the topmost visible child at a screen position.

        Args:
            x: X coordinate in screen space.
            y: Y coordinate in screen space.

        Returns:
            The child widget at the position, or None.
        """
        # Iterate in reverse to get topmost widget first
        for child in reversed(self._children):
            if child.visible and child.contains_point(x, y):
                # Check if any of the child's children are at this position
                grandchild = child.get_child_at(x, y)
                if grandchild:
                    return grandchild
                return child
        return None

    # -------------------------------------------------------------------------
    # Event Callbacks
    # -------------------------------------------------------------------------

    def on_click(self, callback: Callable[[Widget], None]) -> Widget:
        """
        Set the click callback.

        Args:
            callback: Function to call when clicked.

        Returns:
            Self for method chaining.
        """
        self._on_click = callback
        return self

    def on_hover_enter(self, callback: Callable[[Widget], None]) -> Widget:
        """
        Set the hover enter callback.

        Args:
            callback: Function to call when mouse enters widget.

        Returns:
            Self for method chaining.
        """
        self._on_hover_enter = callback
        return self

    def on_hover_exit(self, callback: Callable[[Widget], None]) -> Widget:
        """
        Set the hover exit callback.

        Args:
            callback: Function to call when mouse exits widget.

        Returns:
            Self for method chaining.
        """
        self._on_hover_exit = callback
        return self

    def on_focus(self, callback: Callable[[Widget], None]) -> Widget:
        """
        Set the focus callback.

        Args:
            callback: Function to call when widget gains focus.

        Returns:
            Self for method chaining.
        """
        self._on_focus = callback
        return self

    def on_blur(self, callback: Callable[[Widget], None]) -> Widget:
        """
        Set the blur callback.

        Args:
            callback: Function to call when widget loses focus.

        Returns:
            Self for method chaining.
        """
        self._on_blur = callback
        return self

    # -------------------------------------------------------------------------
    # Focus Management
    # -------------------------------------------------------------------------

    def focus(self) -> None:
        """Give focus to this widget."""
        if not self._state.enabled:
            return

        # Remove focus from siblings
        if self._parent:
            for sibling in self._parent._children:
                if sibling is not self and sibling._state.focused:
                    sibling.blur()

        if not self._state.focused:
            self._state.focused = True
            if self._on_focus:
                self._on_focus(self)

    def blur(self) -> None:
        """Remove focus from this widget."""
        if self._state.focused:
            self._state.focused = False
            if self._on_blur:
                self._on_blur(self)

    # -------------------------------------------------------------------------
    # Event Handling
    # -------------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        """
        Handle a pygame event.

        This method processes events for this widget and its children.
        Override this method in subclasses to handle specific events.

        Args:
            event: Pygame event to handle.

        Returns:
            True if the event was consumed, False otherwise.
        """
        if not self._state.visible or not self._state.enabled:
            return False

        # First, let children handle the event (topmost first)
        for child in reversed(self._children):
            if child.handle_event(event):
                return True

        # Handle mouse events
        if event.type == pygame.MOUSEMOTION:
            return self._handle_mouse_motion(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            return self._handle_mouse_down(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            return self._handle_mouse_up(event)

        return False

    def _handle_mouse_motion(self, event: pygame.event.Event) -> bool:
        """Handle mouse motion events."""
        was_hovered = self._state.hovered
        is_hovered = self.contains_point(event.pos[0], event.pos[1])

        if is_hovered and not was_hovered:
            self._state.hovered = True
            if self._on_hover_enter:
                self._on_hover_enter(self)
        elif not is_hovered and was_hovered:
            self._state.hovered = False
            self._state.pressed = False
            if self._on_hover_exit:
                self._on_hover_exit(self)

        return False  # Don't consume motion events

    def _handle_mouse_down(self, event: pygame.event.Event) -> bool:
        """Handle mouse button down events."""
        if event.button != 1:  # Only handle left click
            return False

        if self.contains_point(event.pos[0], event.pos[1]):
            self._state.pressed = True
            self.focus()
            return True

        return False

    def _handle_mouse_up(self, event: pygame.event.Event) -> bool:
        """Handle mouse button up events."""
        if event.button != 1:
            return False

        if self._state.pressed:
            self._state.pressed = False
            if self.contains_point(event.pos[0], event.pos[1]):
                if self._on_click:
                    self._on_click(self)
                return True

        return False

    # -------------------------------------------------------------------------
    # Abstract Methods
    # -------------------------------------------------------------------------

    @abstractmethod
    def draw(self, surface: pygame.Surface) -> None:
        """
        Draw the widget to a surface.

        Args:
            surface: Pygame surface to draw on.
        """
        pass

    def update(self, dt: float) -> None:
        """
        Update the widget.

        Args:
            dt: Delta time since last update in seconds.
        """
        # Update children
        for child in self._children:
            child.update(dt)

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def draw_children(self, surface: pygame.Surface) -> None:
        """
        Draw all visible children.

        Args:
            surface: Pygame surface to draw on.
        """
        for child in self._children:
            if child.visible:
                child.draw(surface)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(x={self.x}, y={self.y}, size={self.size})"

