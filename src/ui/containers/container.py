"""
Base Container widget for holding and organizing other widgets.
"""

from typing import Optional, Tuple, Union, List

import pygame

from ..widget import Widget


class Container(Widget):
    """
    A base container widget that can hold other widgets.

    Extends Panel functionality with layout capabilities.
    Children can be positioned manually or using auto-layout
    in subclasses (VBox, HBox, Grid).

    Features:
    - Optional background and border
    - Padding for internal spacing
    - Base class for layout containers

    Example:
        container = Container(
            x=10, y=10,
            width=300, height=200,
            padding=10
        )
        container.add_child(label)
        container.add_child(button)
    """

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 200,
        height: int = 100,
        bg_color: Optional[Union[Tuple[int, int, int], Tuple[int, int, int, int]]] = None,
        border_radius: int = 0,
        border_width: int = 0,
        border_color: Optional[Tuple[int, int, int]] = None,
        padding: int = 0,
        auto_size: bool = False,
        clip_children: bool = False,
        parent: Optional[Widget] = None
    ):
        """
        Initialize a Container.

        Args:
            x: X position.
            y: Y position.
            width: Container width.
            height: Container height.
            bg_color: Background color as RGB or RGBA tuple.
            border_radius: Corner radius in pixels.
            border_width: Border width in pixels.
            border_color: Border color.
            padding: Internal padding.
            auto_size: If True, resize to fit children.
            clip_children: If True, children are clipped to the content area.
            parent: Parent widget.
        """
        super().__init__(x, y, width, height, parent)

        self._bg_color = bg_color
        self._border_radius = border_radius
        self._border_width = border_width
        self._border_color = border_color
        self._padding = padding
        self._auto_size = auto_size
        self._clip_children = clip_children

    @property
    def bg_color(self) -> Optional[Tuple]:
        """Get the background color."""
        return self._bg_color

    @bg_color.setter
    def bg_color(self, value: Optional[Tuple]) -> None:
        """Set the background color."""
        self._bg_color = value

    @property
    def padding(self) -> int:
        """Get the padding."""
        return self._padding

    @padding.setter
    def padding(self, value: int) -> None:
        """Set the padding."""
        self._padding = max(0, value)
        self._layout_children()

    @property
    def clip_children(self) -> bool:
        """Whether children are clipped to the content area."""
        return self._clip_children

    @clip_children.setter
    def clip_children(self, value: bool) -> None:
        """Enable or disable clipping of children to the content area."""
        self._clip_children = value

    @property
    def content_width(self) -> int:
        """Get the content area width (excluding padding)."""
        return self.width - (self._padding * 2)

    @property
    def content_height(self) -> int:
        """Get the content area height (excluding padding)."""
        return self.height - (self._padding * 2)

    @property
    def content_rect(self) -> pygame.Rect:
        """Get the content rect (excluding padding) in local coordinates."""
        return pygame.Rect(
            self._padding,
            self._padding,
            self.content_width,
            self.content_height
        )

    def add_child(self, child: Widget) -> None:
        """
        Add a child widget and trigger layout.

        Args:
            child: Widget to add.
        """
        super().add_child(child)
        self._layout_children()
        if self._auto_size:
            self._fit_to_children()

    def remove_child(self, child: Widget) -> None:
        """
        Remove a child widget and trigger layout.

        Args:
            child: Widget to remove.
        """
        super().remove_child(child)
        self._layout_children()
        if self._auto_size:
            self._fit_to_children()

    def _layout_children(self) -> None:
        """
        Layout children. Override in subclasses for auto-layout.

        Base implementation does nothing (manual positioning).
        """
        pass

    def _fit_to_children(self) -> None:
        """
        Resize container to fit all children.

        Called when auto_size is True after adding/removing children.
        """
        if not self._children:
            return

        max_x = 0
        max_y = 0

        for child in self._children:
            child_right = child.x + child.width
            child_bottom = child.y + child.height
            max_x = max(max_x, child_right)
            max_y = max(max_y, child_bottom)

        self._rect.width = max_x + self._padding
        self._rect.height = max_y + self._padding

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the container."""
        if not self.visible:
            return

        abs_rect = self.absolute_rect

        # Draw background
        if self._bg_color:
            if len(self._bg_color) == 4 and self._bg_color[3] < 255:
                # Alpha blending
                temp_surface = pygame.Surface(
                    (abs_rect.width, abs_rect.height),
                    pygame.SRCALPHA
                )
                if self._border_radius > 0:
                    pygame.draw.rect(
                        temp_surface,
                        self._bg_color,
                        pygame.Rect(0, 0, abs_rect.width, abs_rect.height),
                        border_radius=self._border_radius
                    )
                else:
                    temp_surface.fill(self._bg_color)
                surface.blit(temp_surface, (abs_rect.x, abs_rect.y))
            else:
                if self._border_radius > 0:
                    pygame.draw.rect(
                        surface,
                        self._bg_color[:3],
                        abs_rect,
                        border_radius=self._border_radius
                    )
                else:
                    pygame.draw.rect(surface, self._bg_color[:3], abs_rect)

        # Draw border
        if self._border_width > 0:
            border_color = self._border_color or (80, 80, 80)

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

        # Draw children (with optional clipping to content area)
        if self._clip_children:
            clip_rect = pygame.Rect(
                abs_rect.x + self._padding,
                abs_rect.y + self._padding,
                self.content_width,
                self.content_height
            )
            old_clip = surface.get_clip()
            surface.set_clip(clip_rect.clip(old_clip))
            self.draw_children(surface)
            surface.set_clip(old_clip)
        else:
            self.draw_children(surface)

