"""
HBox container for horizontal layout.
"""

from typing import Optional, Tuple

from .container import Container
from ..widget import Widget


class HBox(Container):
    """
    Horizontal box layout container.

    Arranges children horizontally from left to right with
    configurable spacing, vertical alignment, and content justification.

    Features:
    - Automatic horizontal arrangement
    - Configurable spacing between items
    - Vertical alignment (top, center, bottom, stretch)
    - Content justification along the main axis (start, center, end,
      space_between, space_around) — only when auto_size=False
    - Optional auto-sizing to fit children

    Example:
        hbox = HBox(
            x=10, y=10,
            width=300, height=40,
            spacing=10,
            align='center',
            justify='space_between',
            auto_size=False
        )
        hbox.add_child(Button(text="Save"))
        hbox.add_child(Button(text="Cancel"))
        hbox.add_child(Button(text="Help"))
    """

    # Vertical alignment constants
    ALIGN_TOP = 'top'
    ALIGN_CENTER = 'center'
    ALIGN_BOTTOM = 'bottom'
    ALIGN_STRETCH = 'stretch'

    # Horizontal justification constants (main axis)
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
        height: int = 40,
        spacing: int = 8,
        align: str = ALIGN_CENTER,
        justify: str = JUSTIFY_START,
        bg_color: Optional[Tuple[int, int, int, int]] = None,
        border_radius: int = 0,
        border_width: int = 0,
        border_color: Optional[Tuple[int, int, int]] = None,
        padding: int = 0,
        auto_size: bool = True,
        clip_children: bool = False,
        parent: Optional[Widget] = None,
    ):
        """
        Initialize an HBox.

        Args:
            x: X position.
            y: Y position.
            width: Container width (ignored if auto_size is True).
            height: Container height.
            spacing: Horizontal spacing between children (used for START/CENTER/END justify).
            align: Vertical alignment ('top', 'center', 'bottom', 'stretch').
            justify: How to distribute children along the horizontal axis.
                     Only effective when auto_size=False.
                     'start'         — pack from left.
                     'center'        — center the group.
                     'end'           — pack to the right.
                     'space_between' — equal gaps between items.
                     'space_around'  — equal gaps around each item.
            bg_color: Background color.
            border_radius: Corner radius.
            border_width: Border width.
            border_color: Border color.
            padding: Internal padding.
            auto_size: If True, auto-size width to fit children (justify is ignored).
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
        """Get the vertical alignment."""
        return self._align

    @align.setter
    def align(self, value: str) -> None:
        """Set the vertical alignment."""
        self._align = value
        self._layout_children()

    @property
    def justify(self) -> str:
        """Get the horizontal justification."""
        return self._justify

    @justify.setter
    def justify(self, value: str) -> None:
        """Set the horizontal justification."""
        self._justify = value
        self._layout_children()

    def _apply_vertical_align(self, child: Widget, content_height: int) -> None:
        """Apply vertical alignment to a single child."""
        if self._align == self.ALIGN_CENTER:
            child.y = self._padding + (content_height - child.height) // 2
        elif self._align == self.ALIGN_BOTTOM:
            child.y = self._padding + content_height - child.height
        elif self._align == self.ALIGN_STRETCH:
            child.y = self._padding
            child.height = content_height
        else:  # TOP
            child.y = self._padding

    def _layout_children(self) -> None:
        """Arrange children horizontally."""
        if not self._children:
            return

        content_height = self.content_height

        if self._auto_size:
            # Pack tightly from the left and auto-resize width.
            current_x = self._padding
            for child in self._children:
                child.x = current_x
                self._apply_vertical_align(child, content_height)
                current_x += child.width + self._spacing

            total_width = current_x - self._spacing + self._padding
            self._rect.width = max(self._padding * 2, total_width)
            return

        # Fixed width — apply justification along the main axis.
        content_width = self.content_width
        total_child_width = sum(c.width for c in self._children)
        n = len(self._children)
        remaining = content_width - total_child_width

        if self._justify == self.JUSTIFY_CENTER:
            start_x = self._padding + remaining // 2
            gap = self._spacing
        elif self._justify == self.JUSTIFY_END:
            start_x = self._padding + remaining - (n - 1) * self._spacing
            gap = self._spacing
        elif self._justify == self.JUSTIFY_SPACE_BETWEEN:
            start_x = self._padding
            gap = remaining // (n - 1) if n > 1 else 0
        elif self._justify == self.JUSTIFY_SPACE_AROUND:
            per_side = remaining // n if n > 0 else 0
            start_x = self._padding + per_side // 2
            gap = per_side
        else:  # START
            start_x = self._padding
            gap = self._spacing

        current_x = start_x
        for child in self._children:
            child.x = current_x
            self._apply_vertical_align(child, content_height)
            current_x += child.width + gap

