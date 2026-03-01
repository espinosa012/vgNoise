"""
Image widget for displaying Pygame surfaces or loaded images.
"""

from typing import Optional, Tuple, Union
from pathlib import Path

import pygame

from ..widget import Widget


class ImageWidget(Widget):
    """
    A widget for displaying images or Pygame surfaces.

    Features:
    - Load images from file paths
    - Display existing Pygame surfaces
    - Optional scaling modes (fit, fill, stretch)
    - Border and background support

    Example:
        # From file
        img = ImageWidget(
            x=10, y=10,
            image_path="images/icon.png"
        )

        # From surface
        img = ImageWidget(
            x=10, y=10,
            surface=my_surface,
            scale_mode='fit'
        )
    """

    # Scaling modes
    SCALE_NONE = 'none'       # No scaling, original size
    SCALE_STRETCH = 'stretch' # Stretch to fill widget
    SCALE_FIT = 'fit'         # Fit inside, maintain aspect ratio
    SCALE_FILL = 'fill'       # Fill widget, maintain aspect ratio, may crop

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 100,
        height: int = 100,
        image_path: Optional[Union[str, Path]] = None,
        surface: Optional[pygame.Surface] = None,
        scale_mode: str = SCALE_NONE,
        bg_color: Optional[Tuple[int, int, int]] = None,
        border_width: int = 0,
        border_color: Optional[Tuple[int, int, int]] = None,
        parent: Optional[Widget] = None
    ):
        """
        Initialize an ImageWidget.

        Args:
            x: X position.
            y: Y position.
            width: Widget width.
            height: Widget height.
            image_path: Path to image file to load.
            surface: Pygame surface to display.
            scale_mode: How to scale the image ('none', 'stretch', 'fit', 'fill').
            bg_color: Background color (shown if image doesn't fill widget).
            border_width: Border width in pixels.
            border_color: Border color.
            parent: Parent widget.
        """
        super().__init__(x, y, width, height, parent)

        self._original_surface: Optional[pygame.Surface] = None
        self._scaled_surface: Optional[pygame.Surface] = None
        self._scale_mode = scale_mode
        self._bg_color = bg_color
        self._border_width = border_width
        self._border_color = border_color
        self._image_rect: Optional[pygame.Rect] = None

        # Load image if path provided
        if image_path:
            self.load_image(image_path)
        elif surface:
            self.surface = surface

    @property
    def surface(self) -> Optional[pygame.Surface]:
        """Get the original surface."""
        return self._original_surface

    @surface.setter
    def surface(self, value: Optional[pygame.Surface]) -> None:
        """Set the surface to display."""
        self._original_surface = value
        self._update_scaled_surface()

    @property
    def scale_mode(self) -> str:
        """Get the scale mode."""
        return self._scale_mode

    @scale_mode.setter
    def scale_mode(self, value: str) -> None:
        """Set the scale mode."""
        if value not in (self.SCALE_NONE, self.SCALE_STRETCH, self.SCALE_FIT, self.SCALE_FILL):
            raise ValueError(f"Invalid scale mode: {value}")
        self._scale_mode = value
        self._update_scaled_surface()

    def load_image(self, path: Union[str, Path]) -> bool:
        """
        Load an image from a file path.

        Args:
            path: Path to the image file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            surface = pygame.image.load(str(path))
            self._original_surface = surface.convert_alpha()
            self._update_scaled_surface()
            return True
        except pygame.error as e:
            print(f"Failed to load image '{path}': {e}")
            self._original_surface = None
            self._scaled_surface = None
            return False

    def _update_scaled_surface(self) -> None:
        """Update the scaled surface based on current scale mode."""
        if not self._original_surface:
            self._scaled_surface = None
            self._image_rect = None
            return

        orig_w = self._original_surface.get_width()
        orig_h = self._original_surface.get_height()

        if self._scale_mode == self.SCALE_NONE:
            self._scaled_surface = self._original_surface
            # Center the image
            self._image_rect = pygame.Rect(
                (self.width - orig_w) // 2,
                (self.height - orig_h) // 2,
                orig_w,
                orig_h
            )

        elif self._scale_mode == self.SCALE_STRETCH:
            self._scaled_surface = pygame.transform.scale(
                self._original_surface,
                (self.width, self.height)
            )
            self._image_rect = pygame.Rect(0, 0, self.width, self.height)

        elif self._scale_mode == self.SCALE_FIT:
            # Calculate scale to fit while maintaining aspect ratio
            scale = min(self.width / orig_w, self.height / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            self._scaled_surface = pygame.transform.scale(
                self._original_surface,
                (new_w, new_h)
            )
            # Center the scaled image
            self._image_rect = pygame.Rect(
                (self.width - new_w) // 2,
                (self.height - new_h) // 2,
                new_w,
                new_h
            )

        elif self._scale_mode == self.SCALE_FILL:
            # Calculate scale to fill while maintaining aspect ratio
            scale = max(self.width / orig_w, self.height / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            self._scaled_surface = pygame.transform.scale(
                self._original_surface,
                (new_w, new_h)
            )
            # Center the scaled image (may be cropped)
            self._image_rect = pygame.Rect(
                (self.width - new_w) // 2,
                (self.height - new_h) // 2,
                new_w,
                new_h
            )

    @Widget.width.setter
    def width(self, value: int) -> None:
        """Override to update scaled surface on resize."""
        Widget.width.fset(self, value)
        self._update_scaled_surface()

    @Widget.height.setter
    def height(self, value: int) -> None:
        """Override to update scaled surface on resize."""
        Widget.height.fset(self, value)
        self._update_scaled_surface()

    def draw(self, surface: pygame.Surface) -> None:
        """Draw the image widget."""
        if not self.visible:
            return

        abs_rect = self.absolute_rect

        # Draw background
        if self._bg_color:
            pygame.draw.rect(surface, self._bg_color, abs_rect)

        # Draw image
        if self._scaled_surface and self._image_rect:
            # For FILL mode, we need to clip
            if self._scale_mode == self.SCALE_FILL:
                # Create a subsurface for clipping
                clip_rect = surface.get_clip()
                surface.set_clip(abs_rect.clip(clip_rect))

                surface.blit(
                    self._scaled_surface,
                    (abs_rect.x + self._image_rect.x, abs_rect.y + self._image_rect.y)
                )

                surface.set_clip(clip_rect)
            else:
                surface.blit(
                    self._scaled_surface,
                    (abs_rect.x + self._image_rect.x, abs_rect.y + self._image_rect.y)
                )

        # Draw border
        if self._border_width > 0:
            border_color = self._border_color or (80, 80, 80)

            pygame.draw.rect(
                surface,
                border_color,
                abs_rect,
                width=self._border_width
            )

        # Draw children
        self.draw_children(surface)

