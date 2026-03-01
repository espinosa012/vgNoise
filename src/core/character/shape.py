"""
Character visual representation using pygame primitives.

Provides a base class (CharacterShape) and concrete implementations
for drawing characters without external image assets. Designed to be
extended as the visual system grows in complexity.
"""

from abc import ABC, abstractmethod
import pygame

from core.color.color import Color, Colors


class CharacterShape(ABC):
    """
    Abstract base for all primitive-based character visuals.

    Each subclass describes *how* to draw a character using pygame
    primitives. BaseCharacter holds one CharacterShape instance and
    delegates rendering to it, so swapping visuals only requires
    replacing the shape — no changes to the character class.
    """

    @abstractmethod
    def draw(self, surface: "pygame.Surface", x: float, y: float) -> None:
        """
        Draw this shape onto surface at pixel position (x, y).

        Args:
            surface: Target pygame.Surface.
            x:       Pixel x coordinate (top-left anchor).
            y:       Pixel y coordinate (top-left anchor).
        """
        ...

    @property
    @abstractmethod
    def width(self) -> int:
        """Bounding-box width in pixels."""
        ...

    @property
    @abstractmethod
    def height(self) -> int:
        """Bounding-box height in pixels."""
        ...


class RectShape(CharacterShape):
    """
    Character visual represented by a filled rectangle with an
    optional contrasting border.

    Attributes:
        color:         Fill color of the rectangle.
        border_color:  Color of the border. None disables the border.
        border_width:  Thickness of the border in pixels (default 0 = no border).
    """

    def __init__(
        self,
        width: int,
        height: int,
        color: Color = Colors.WHITE,
        border_color: Color = None,
        border_width: int = 0,
    ) -> None:
        """
        Args:
            width:        Rectangle width in pixels.
            height:       Rectangle height in pixels.
            color:        Fill color (default: white).
            border_color: Border color. If None and border_width > 0,
                          a dark version of the fill color is used.
            border_width: Border thickness in pixels (0 = no border).
        """
        self._width = width
        self._height = height
        self.color = color
        self.border_width = border_width

        if border_color is None and border_width > 0:
            # Auto-derive a darker border from the fill color
            self.border_color: Color = Color(
                max(0, color.r - 60),
                max(0, color.g - 60),
                max(0, color.b - 60),
                color.a,
            )
        else:
            self.border_color = border_color

    # ------------------------------------------------------------------
    # CharacterShape interface
    # ------------------------------------------------------------------

    def draw(self, surface: "pygame.Surface", x: float, y: float) -> None:
        """Draw a filled rectangle, then the border on top if configured."""
        rect = pygame.Rect(int(x), int(y), self._width, self._height)

        # Fill
        pygame.draw.rect(surface, self.color.to_rgba(), rect)

        # Border
        if self.border_width > 0 and self.border_color is not None:
            pygame.draw.rect(surface, self.border_color.to_rgba(), rect, self.border_width)

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        """Change the rectangle dimensions."""
        self._width = width
        self._height = height

    def __repr__(self) -> str:
        return (
            f"RectShape(width={self._width}, height={self._height}, "
            f"color={self.color}, border_width={self.border_width})"
        )

