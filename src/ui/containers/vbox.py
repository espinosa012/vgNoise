"""
VBox container for vertical layout.
"""

from typing import Optional, Tuple

from .container import Container
from ..widget import Widget


class VBox(Container):
    """
    Vertical box layout container.

    Arranges children vertically from top to bottom with
    configurable spacing, horizontal alignment, and content justification.

    Features:
    - Automatic vertical arrangement
    - Configurable spacing between items
    - Horizontal alignment (left, center, right, stretch)
    - Content justification along the main axis (start, center, end,
      space_between, space_around) — only when auto_size=False
    - Optional auto-sizing to fit children

    Example:
        vbox = VBox(
            x=10, y=10,
            width=200, height=300,
            spacing=10,
            align='center',
            justify='space_between',
            auto_size=False
        )
        vbox.add_child(Label(text="Title"))
        vbox.add_child(Button(text="Option 1"))
        vbox.add_child(Button(text="Option 2"))
    """

    # Horizontal alignment constants
    ALIGN_LEFT = 'left'
    ALIGN_CENTER = 'center'
    ALIGN_RIGHT = 'right'
    ALIGN_STRETCH = 'stretch'

    # Vertical justification constants (main axis)
    JUSTIFY_START = 'start'
    JUSTIFY_CENTER = 'center'
    JUSTIFY_END = 'end'
    JUSTIFY_SPACE_BETWEEN = 'space_between'
    JUSTIFY_SPACE_AROUND = 'space_around'

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 200,
        height: int = 100,
        spacing: int = 8,
        align: str = ALIGN_LEFT,
        justify: str = JUSTIFY_START,
        bg_color: Optional[Tuple[int, int, int, int]] = None,
        border_radius: int = 0,
        border_width: int = 0,
        border_color: Optional[Tuple[int, int, int]] = None,
        padding: int = 0,
        auto_size: bool = True,
        clip_children: bool = False,
        parent: Optional[Widget] = None
    ):
        """
        Initialize a VBox.

        Args:
            x: X position.
            y: Y position.
            width: Container width.
            height: Container height (ignored if auto_size is True).
            spacing: Vertical spacing between children (used for START/CENTER/END justify).
            align: Horizontal alignment ('left', 'center', 'right', 'stretch').
            justify: How to distribute children along the vertical axis.
                     Only effective when auto_size=False.
                     'start'         — pack from top.
                     'center'        — center the group.
                     'end'           — pack to the bottom.
                     'space_between' — equal gaps between items.
                     'space_around'  — equal gaps around each item.
            bg_color: Background color.
            border_radius: Corner radius.
            border_width: Border width.
            border_color: Border color.
            padding: Internal padding.
            auto_size: If True, auto-size height to fit children (justify is ignored).
            clip_children: If True, clip children to the content area.
            parent: Parent widget.
        """
        super().__init__(
            x, y, width, height,
            bg_color, border_radius, border_width, border_color,
            padding, auto_size, clip_children, parent
        )

        self._spacing = spacing
        self._align = align
        self._justify = justify

    @property
    def spacing(self) -> int:
        """Get the spacing between children."""
        return self._spacing

    @spacing.setter
    def spacing(self, value: int) -> None:
        """Set the spacing between children."""
        self._spacing = max(0, value)
        self._layout_children()

    @property
    def align(self) -> str:
        """Get the horizontal alignment."""
        return self._align

    @align.setter
    def align(self, value: str) -> None:
        """Set the horizontal alignment."""
        self._align = value
        self._layout_children()

    @property
    def justify(self) -> str:
        """Get the vertical justification."""
        return self._justify

    @justify.setter
    def justify(self, value: str) -> None:
        """Set the vertical justification."""
        self._justify = value
        self._layout_children()

    def _apply_horizontal_align(self, child: Widget, content_width: int) -> None:
        """Apply horizontal alignment to a single child."""
        if self._align == self.ALIGN_CENTER:
            child.x = self._padding + (content_width - child.width) // 2
        elif self._align == self.ALIGN_RIGHT:
            child.x = self._padding + content_width - child.width
        elif self._align == self.ALIGN_STRETCH:
            child.x = self._padding
            child.width = content_width
        else:  # LEFT
            child.x = self._padding

    def _layout_children(self) -> None:
        """Arrange children vertically."""
        if not self._children:
            return

        content_width = self.content_width

        if self._auto_size:
            # Pack tightly from the top and auto-resize height.
            current_y = self._padding
            for child in self._children:
                child.y = current_y
                self._apply_horizontal_align(child, content_width)
                current_y += child.height + self._spacing

            total_height = current_y - self._spacing + self._padding
            self._rect.height = max(self._padding * 2, total_height)
            return

        # Fixed height — apply justification along the main axis.
        content_height = self.content_height
        total_child_height = sum(c.height for c in self._children)
        n = len(self._children)
        remaining = content_height - total_child_height

        if self._justify == self.JUSTIFY_CENTER:
            start_y = self._padding + remaining // 2
            gap = self._spacing
        elif self._justify == self.JUSTIFY_END:
            start_y = self._padding + remaining - (n - 1) * self._spacing
            gap = self._spacing
        elif self._justify == self.JUSTIFY_SPACE_BETWEEN:
            start_y = self._padding
            gap = remaining // (n - 1) if n > 1 else 0
        elif self._justify == self.JUSTIFY_SPACE_AROUND:
            per_side = remaining // n if n > 0 else 0
            start_y = self._padding + per_side // 2
            gap = per_side
        else:  # START
            start_y = self._padding
            gap = self._spacing

        current_y = start_y
        for child in self._children:
            child.y = current_y
            self._apply_horizontal_align(child, content_width)
            current_y += child.height + gap