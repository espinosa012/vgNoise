"""
Grid container for grid-based layout.
"""

from typing import Optional, Tuple

from .container import Container
from ..widget import Widget


class Grid(Container):
    """
    Grid layout container.

    Arranges children in a grid with configurable rows, columns,
    and spacing.
    """

    ALIGN_START = 'start'
    ALIGN_CENTER = 'center'
    ALIGN_END = 'end'
    ALIGN_STRETCH = 'stretch'

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 200,
        height: int = 200,
        columns: int = 3,
        rows: int = 0,
        spacing_x: int = 8,
        spacing_y: int = 8,
        cell_width: int = 0,
        cell_height: int = 0,
        cell_align_x: str = ALIGN_CENTER,
        cell_align_y: str = ALIGN_CENTER,
        bg_color: Optional[Tuple[int, int, int, int]] = None,
        border_radius: int = 0,
        border_width: int = 0,
        border_color: Optional[Tuple[int, int, int]] = None,
        padding: int = 0,
        auto_size: bool = False,
        clip_children: bool = False,
        parent: Optional[Widget] = None,
    ):
        super().__init__(
            x, y, width, height,
            bg_color, border_radius, border_width, border_color,
            padding, auto_size, clip_children, parent
        )

        self._columns = max(1, columns)
        self._rows = rows
        self._spacing_x = spacing_x
        self._spacing_y = spacing_y
        self._cell_width = cell_width
        self._cell_height = cell_height
        self._cell_align_x = cell_align_x
        self._cell_align_y = cell_align_y

    @property
    def columns(self) -> int:
        return self._columns

    @columns.setter
    def columns(self, value: int) -> None:
        self._columns = max(1, value)
        self._layout_children()

    @property
    def rows(self) -> int:
        if self._rows > 0:
            return self._rows
        return (len(self._children) + self._columns - 1) // self._columns

    def _get_cell_size(self) -> Tuple[int, int]:
        if self._cell_width > 0 and self._cell_height > 0:
            return self._cell_width, self._cell_height

        content_w = self.content_width
        content_h = self.content_height

        total_spacing_x = (self._columns - 1) * self._spacing_x
        total_spacing_y = (self.rows - 1) * self._spacing_y if self.rows > 0 else 0

        cell_w = self._cell_width if self._cell_width > 0 else \
                 (content_w - total_spacing_x) // self._columns

        cell_h = self._cell_height if self._cell_height > 0 else \
                 (content_h - total_spacing_y) // max(1, self.rows)

        return max(1, cell_w), max(1, cell_h)

    def _layout_children(self) -> None:
        if not self._children:
            return

        cell_w, cell_h = self._get_cell_size()

        for idx, child in enumerate(self._children):
            row = idx // self._columns
            col = idx % self._columns

            if self._rows > 0 and row >= self._rows:
                child.visible = False
                continue

            child.visible = True

            cell_x = self._padding + col * (cell_w + self._spacing_x)
            cell_y = self._padding + row * (cell_h + self._spacing_y)

            if self._cell_align_x == self.ALIGN_STRETCH:
                child.x = cell_x
                child.width = cell_w
            elif self._cell_align_x == self.ALIGN_CENTER:
                child.x = cell_x + (cell_w - child.width) // 2
            elif self._cell_align_x == self.ALIGN_END:
                child.x = cell_x + cell_w - child.width
            else:
                child.x = cell_x

            if self._cell_align_y == self.ALIGN_STRETCH:
                child.y = cell_y
                child.height = cell_h
            elif self._cell_align_y == self.ALIGN_CENTER:
                child.y = cell_y + (cell_h - child.height) // 2
            elif self._cell_align_y == self.ALIGN_END:
                child.y = cell_y + cell_h - child.height
            else:
                child.y = cell_y

