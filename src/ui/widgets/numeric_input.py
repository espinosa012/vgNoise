"""
NumericInput widget: a text field flanked by − and + buttons for numeric entry.
"""

from __future__ import annotations

from typing import Optional, Tuple, Callable

import pygame

from ..widget import Widget
from .button import Button
from .text_input import TextInput


class NumericInput(Widget):
    """
    A numeric input widget.

    Displays a text field with − and + buttons on each side.
    The buttons decrement / increment the value by *step*.
    Direct text editing is also supported; the value is validated
    (clamped to [min_value, max_value]) when the field loses focus or
    Enter is pressed.

    Example::

        ni = NumericInput(
            width=120, height=26,
            value=0.5, min_value=0.0, max_value=1.0, step=0.05,
        )
        ni.on_change(lambda w: print(w.value))
    """

    # Not focusable itself — the inner TextInput is the natural tab stop.
    _focusable = False

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 120,
        height: int = 26,
        value: float = 0.0,
        min_value: float = float("-inf"),
        max_value: float = float("inf"),
        step: float = 1.0,
        decimals: int = -1,
        font_size: int = 16,
        bg_color: Optional[Tuple[int, int, int]] = None,
        text_color: Optional[Tuple[int, int, int]] = None,
        border_color: Optional[Tuple[int, int, int]] = None,
        focus_border_color: Optional[Tuple[int, int, int]] = None,
        border_radius: int = 4,
        parent: Optional[Widget] = None,
    ):
        """
        Args:
            value: Initial value.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            step: Amount to increment / decrement per button press.
            decimals: Decimal places to display. -1 = auto-detect from step.
        """
        super().__init__(x, y, width, height, parent)

        self._min = min_value
        self._max = max_value
        self._step = step
        self._decimals = decimals if decimals >= 0 else self._auto_decimals(step)
        self._on_change_cb: Optional[Callable[["NumericInput"], None]] = None

        btn_w = height
        inp_w = max(20, width - 2 * btn_w)

        _btn_kw = dict(
            bg_color=(60, 60, 80),
            hover_color=(80, 80, 110),
            pressed_color=(40, 40, 60),
            border_radius=border_radius,
            font_size=font_size + 2,
        )

        self._btn_dec = Button(x=0, y=0, width=btn_w, height=height, text="-", parent=self, **_btn_kw)

        self._inp = TextInput(
            x=btn_w, y=0, width=inp_w, height=height,
            text=self._fmt(self._clamp(value)),
            font_size=font_size,
            bg_color=bg_color or (50, 50, 65),
            text_color=text_color or (255, 255, 255),
            border_color=border_color or (100, 100, 130),
            focus_border_color=focus_border_color or (100, 160, 255),
            border_radius=border_radius,
            parent=self,
        )

        self._btn_inc = Button(x=btn_w + inp_w, y=0, width=btn_w, height=height, text="+", parent=self, **_btn_kw)

        self._btn_dec.on_click(lambda _: self._decrement())
        self._btn_inc.on_click(lambda _: self._increment())
        self._inp.on_blur(lambda _: self._validate())
        self._inp.on_submit(lambda _: self._validate())

    # -------------------------------------------------------------------------
    # Layout
    # -------------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._rect.width

    @width.setter
    def width(self, value: int) -> None:
        self._rect.width = max(0, value)
        self._relayout()

    @property
    def height(self) -> int:
        return self._rect.height

    @height.setter
    def height(self, value: int) -> None:
        self._rect.height = max(0, value)
        self._relayout()

    def _relayout(self) -> None:
        if not hasattr(self, "_btn_dec"):
            return
        h = self._rect.height
        btn_w = h
        inp_w = max(20, self._rect.width - 2 * btn_w)

        self._btn_dec.x = 0
        self._btn_dec.width = btn_w
        self._btn_dec.height = h

        self._inp.x = btn_w
        self._inp.width = inp_w
        self._inp.height = h

        self._btn_inc.x = btn_w + inp_w
        self._btn_inc.width = btn_w
        self._btn_inc.height = h

    # -------------------------------------------------------------------------
    # Value helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _auto_decimals(step: float) -> int:
        """Infer decimal places to display from *step*."""
        if isinstance(step, int) or (isinstance(step, float) and step == int(step)):
            return 0
        s = format(step, "g")
        if "." in s:
            return len(s.split(".")[1])
        if "e-" in s:
            return int(s.split("e-")[1])
        return 0

    def _fmt(self, v: float) -> str:
        if self._decimals == 0:
            return str(int(round(v)))
        return f"{v:.{self._decimals}f}"

    def _clamp(self, v: float) -> float:
        return max(self._min, min(self._max, v))

    def _parse(self) -> float:
        try:
            return float(self._inp.text)
        except ValueError:
            return self._clamp(0.0)

    # -------------------------------------------------------------------------
    # Button actions
    # -------------------------------------------------------------------------

    def _increment(self) -> None:
        v = self._clamp(self._parse() + self._step)
        self._inp.text = self._fmt(v)
        if self._on_change_cb:
            self._on_change_cb(self)

    def _decrement(self) -> None:
        v = self._clamp(self._parse() - self._step)
        self._inp.text = self._fmt(v)
        if self._on_change_cb:
            self._on_change_cb(self)

    def _validate(self) -> None:
        """Clamp and reformat on focus loss or Enter."""
        v = self._clamp(self._parse())
        formatted = self._fmt(v)
        if self._inp.text != formatted:
            self._inp.text = formatted
        if self._on_change_cb:
            self._on_change_cb(self)

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @property
    def value(self) -> float:
        return self._parse()

    @value.setter
    def value(self, v: float) -> None:
        self._inp.text = self._fmt(self._clamp(v))

    @property
    def text(self) -> str:
        """Raw text of the inner field (compatible with scene config helpers)."""
        return self._inp.text

    @text.setter
    def text(self, v: str) -> None:
        self._inp.text = v

    def on_change(self, callback: Callable[["NumericInput"], None]) -> "NumericInput":
        self._on_change_cb = callback
        return self

    # -------------------------------------------------------------------------
    # Widget overrides
    # -------------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return
        self.draw_children(surface)

    def update(self, dt: float) -> None:
        super().update(dt)
