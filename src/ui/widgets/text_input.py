"""
TextInput widget for text entry.
"""

import platform
import subprocess
from typing import Optional, Tuple, Callable

import pygame

from ..widget import Widget

_PLATFORM = platform.system()


def _clipboard_get() -> str:
    """Return plain text from the system clipboard, or '' on failure."""
    try:
        if _PLATFORM == "Darwin":
            text = subprocess.run(["pbpaste"], capture_output=True).stdout.decode("utf-8", errors="replace")
        elif _PLATFORM == "Windows":
            text = subprocess.run(
                ["powershell", "-noprofile", "-command", "Get-Clipboard"],
                capture_output=True,
            ).stdout.decode("utf-8", errors="replace")
        else:
            try:
                text = subprocess.run(
                    ["xclip", "-selection", "clipboard", "-o"], capture_output=True
                ).stdout.decode("utf-8", errors="replace")
            except FileNotFoundError:
                text = subprocess.run(
                    ["xsel", "--clipboard", "--output"], capture_output=True
                ).stdout.decode("utf-8", errors="replace")
        return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")[0]
    except Exception:
        return ""


def _clipboard_set(text: str) -> None:
    """Copy *text* to the system clipboard, silently ignoring errors."""
    try:
        if _PLATFORM == "Darwin":
            subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
        elif _PLATFORM == "Windows":
            subprocess.run(["clip"], input=text.encode("utf-16-le"), check=True)
        else:
            try:
                subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode("utf-8"), check=True)
            except FileNotFoundError:
                subprocess.run(["xsel", "--clipboard", "--input"], input=text.encode("utf-8"), check=True)
    except Exception:
        pass


class TextInput(Widget):
    """
    A single-line text input widget.

    Features:
    - Text editing with cursor and selection
    - Ctrl/Cmd+A to select all
    - Shift+arrows to extend selection
    - Ctrl+arrows to jump by word
    - Placeholder text
    - Character limit
    - Visual feedback for focus and hover states
    - Change and submit callbacks
    """

    _focusable = True

    def __init__(
        self,
        x: int = 0,
        y: int = 0,
        width: int = 200,
        height: int = 32,
        text: str = "",
        placeholder: str = "",
        max_length: int = 0,
        font_size: Optional[int] = None,
        text_color: Optional[Tuple[int, int, int]] = None,
        placeholder_color: Optional[Tuple[int, int, int]] = None,
        bg_color: Optional[Tuple[int, int, int]] = None,
        border_color: Optional[Tuple[int, int, int]] = None,
        focus_border_color: Optional[Tuple[int, int, int]] = None,
        border_width: int = 1,
        border_radius: int = 4,
        padding: int = 8,
        parent: Optional[Widget] = None,
    ):
        super().__init__(x, y, width, height, parent)

        self._text = text
        self._placeholder = placeholder
        self._max_length = max_length
        self._font_size = font_size
        self._text_color = text_color
        self._placeholder_color = placeholder_color
        self._bg_color = bg_color
        self._border_color = border_color
        self._focus_border_color = focus_border_color
        self._border_width = border_width
        self._border_radius = border_radius
        self._padding = padding

        # Cursor state
        self._cursor_pos = len(text)
        self._cursor_visible = True
        self._cursor_timer = 0.0
        self._cursor_blink_rate = 0.5

        # Selection: anchor is the fixed end; cursor_pos is the moving end.
        # -1 means no selection.
        self._sel_anchor: int = -1

        # Scroll offset for long text
        self._scroll_offset = 0

        # Undo history: list of (text, cursor_pos) snapshots
        self._undo_stack: List[Tuple[str, int]] = []

        # Callbacks
        self._on_change: Optional[Callable[[TextInput], None]] = None
        self._on_submit: Optional[Callable[[TextInput], None]] = None

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, value: str) -> None:
        if self._max_length > 0:
            value = value[:self._max_length]
        if self._text != value:
            self._text = value
            self._cursor_pos = min(self._cursor_pos, len(value))
            self._sel_anchor = -1
            self._update_scroll()
            if self._on_change:
                self._on_change(self)

    @property
    def placeholder(self) -> str:
        return self._placeholder

    @placeholder.setter
    def placeholder(self, value: str) -> None:
        self._placeholder = value

    def on_change(self, callback: Callable[['TextInput'], None]) -> 'TextInput':
        self._on_change = callback
        return self

    def on_submit(self, callback: Callable[['TextInput'], None]) -> 'TextInput':
        self._on_submit = callback
        return self

    # -------------------------------------------------------------------------
    # Undo helpers
    # -------------------------------------------------------------------------

    def _push_undo(self) -> None:
        """Save current (text, cursor_pos) to the undo stack (max 100 entries)."""
        state = (self._text, self._cursor_pos)
        if not self._undo_stack or self._undo_stack[-1] != state:
            self._undo_stack.append(state)
            if len(self._undo_stack) > 100:
                del self._undo_stack[0]

    # -------------------------------------------------------------------------
    # Selection helpers
    # -------------------------------------------------------------------------

    def _has_selection(self) -> bool:
        return self._sel_anchor != -1 and self._sel_anchor != self._cursor_pos

    def _selection_range(self) -> Tuple[int, int]:
        """Return (start, end) of selection, start <= end."""
        return (min(self._sel_anchor, self._cursor_pos),
                max(self._sel_anchor, self._cursor_pos))

    def _delete_selection(self) -> None:
        start, end = self._selection_range()
        self._text = self._text[:start] + self._text[end:]
        self._cursor_pos = start
        self._sel_anchor = -1
        self._update_scroll()
        if self._on_change:
            self._on_change(self)

    # -------------------------------------------------------------------------
    # Word-boundary helpers (space-delimited for UI inputs)
    # -------------------------------------------------------------------------

    def _find_word_start(self, pos: int) -> int:
        """Position of the start of the word to the left of pos."""
        text = self._text
        p = pos
        while p > 0 and text[p - 1] == ' ':
            p -= 1
        while p > 0 and text[p - 1] != ' ':
            p -= 1
        return p

    def _find_word_end(self, pos: int) -> int:
        """Position of the end of the word to the right of pos."""
        text = self._text
        n = len(text)
        p = pos
        while p < n and text[p] == ' ':
            p += 1
        while p < n and text[p] != ' ':
            p += 1
        return p

    # -------------------------------------------------------------------------
    # Internal editing helpers
    # -------------------------------------------------------------------------

    def _get_font(self) -> pygame.font.Font:
        size = self._font_size or 16
        return pygame.font.Font(None, size)

    def _update_scroll(self) -> None:
        """Keep cursor visible inside the content area."""
        font = self._get_font()
        content_width = self.width - (self._padding * 2)
        cursor_x = font.size(self._text[:self._cursor_pos])[0]
        if cursor_x - self._scroll_offset > content_width:
            self._scroll_offset = cursor_x - content_width + 10
        elif cursor_x - self._scroll_offset < 0:
            self._scroll_offset = max(0, cursor_x - 10)

    def _insert_text(self, text: str) -> None:
        """Insert text at cursor (replaces selection if any)."""
        if self._has_selection():
            self._delete_selection()
        new_text = self._text[:self._cursor_pos] + text + self._text[self._cursor_pos:]
        if self._max_length > 0:
            new_text = new_text[:self._max_length]
        self._text = new_text
        self._cursor_pos = min(self._cursor_pos + len(text), len(self._text))
        self._update_scroll()
        if self._on_change:
            self._on_change(self)

    def _delete_char(self, forward: bool = False) -> None:
        """Delete a character (or the selection if one exists)."""
        if self._has_selection():
            self._delete_selection()
            return
        if forward:
            if self._cursor_pos < len(self._text):
                self._text = self._text[:self._cursor_pos] + self._text[self._cursor_pos + 1:]
                if self._on_change:
                    self._on_change(self)
        else:
            if self._cursor_pos > 0:
                self._text = self._text[:self._cursor_pos - 1] + self._text[self._cursor_pos:]
                self._cursor_pos -= 1
                self._update_scroll()
                if self._on_change:
                    self._on_change(self)

    # -------------------------------------------------------------------------
    # Focus overrides
    # -------------------------------------------------------------------------

    def focus(self) -> None:
        super().focus()
        self._cursor_visible = True
        self._cursor_timer = 0

    def blur(self) -> None:
        super().blur()
        self._sel_anchor = -1

    # -------------------------------------------------------------------------
    # Event handling
    # -------------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self._state.visible or not self._state.enabled:
            return False

        for child in reversed(self._children):
            if child.handle_event(event):
                return True

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.contains_point(event.pos[0], event.pos[1]):
                self._state.pressed = True
                self.focus()
                shift = bool(pygame.key.get_mods() & pygame.KMOD_SHIFT)
                self._set_cursor_from_mouse(event.pos[0], extend_selection=shift)
                return True
            else:
                if self.focused:
                    self.blur()

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self._state.pressed = False

        elif event.type == pygame.MOUSEMOTION:
            was_hovered = self._state.hovered
            is_hovered = self.contains_point(event.pos[0], event.pos[1])
            if is_hovered != was_hovered:
                self._state.hovered = is_hovered

        if self.focused and event.type == pygame.KEYDOWN:
            return self._handle_key(event)

        return False

    def _set_cursor_from_mouse(self, mouse_x: int, extend_selection: bool = False) -> None:
        """Set cursor from mouse position; optionally extend the selection."""
        abs_x, _ = self.get_absolute_position()
        local_x = mouse_x - abs_x - self._padding + self._scroll_offset

        font = self._get_font()
        best_pos = 0
        best_dist = abs(local_x)
        for i in range(1, len(self._text) + 1):
            char_x = font.size(self._text[:i])[0]
            dist = abs(local_x - char_x)
            if dist < best_dist:
                best_dist = dist
                best_pos = i

        if extend_selection:
            if self._sel_anchor == -1:
                self._sel_anchor = self._cursor_pos
        else:
            self._sel_anchor = -1

        self._cursor_pos = best_pos

    def _handle_key(self, event: pygame.event.Event) -> bool:
        """Handle keyboard input with full selection support."""
        ctrl = bool(event.mod & (pygame.KMOD_CTRL | pygame.KMOD_META))
        shift = bool(event.mod & pygame.KMOD_SHIFT)

        # Let Tab fall through to UIManager's navigation
        if event.key == pygame.K_TAB:
            return False

        # ── Select all ────────────────────────────────────────────────────────
        if event.key == pygame.K_a and ctrl:
            self._sel_anchor = 0
            self._cursor_pos = len(self._text)
            self._update_scroll()
            return True

        # ── Copy ──────────────────────────────────────────────────────────────
        if event.key == pygame.K_c and ctrl:
            if self._has_selection():
                start, end = self._selection_range()
                _clipboard_set(self._text[start:end])
            return True

        # ── Cut ───────────────────────────────────────────────────────────────
        if event.key == pygame.K_x and ctrl:
            if self._has_selection():
                start, end = self._selection_range()
                _clipboard_set(self._text[start:end])
                self._push_undo()
                self._delete_selection()
            return True

        # ── Paste ─────────────────────────────────────────────────────────────
        if event.key == pygame.K_v and ctrl:
            text = _clipboard_get()
            if text:
                self._push_undo()
                self._insert_text(text)
            return True

        # ── Undo ──────────────────────────────────────────────────────────────
        if event.key == pygame.K_z and ctrl:
            if self._undo_stack:
                text, pos = self._undo_stack.pop()
                self._text = text
                self._cursor_pos = min(pos, len(text))
                self._sel_anchor = -1
                self._update_scroll()
                if self._on_change:
                    self._on_change(self)
            return True

        # ── Cursor movement ───────────────────────────────────────────────────
        if event.key == pygame.K_LEFT:
            if ctrl:
                new_pos = self._find_word_start(self._cursor_pos)
            elif not shift and self._has_selection():
                new_pos = self._selection_range()[0]   # jump to selection start
            else:
                new_pos = max(0, self._cursor_pos - 1)
            if shift:
                if self._sel_anchor == -1:
                    self._sel_anchor = self._cursor_pos
            else:
                self._sel_anchor = -1
            self._cursor_pos = new_pos
            self._update_scroll()
            return True

        if event.key == pygame.K_RIGHT:
            if ctrl:
                new_pos = self._find_word_end(self._cursor_pos)
            elif not shift and self._has_selection():
                new_pos = self._selection_range()[1]   # jump to selection end
            else:
                new_pos = min(len(self._text), self._cursor_pos + 1)
            if shift:
                if self._sel_anchor == -1:
                    self._sel_anchor = self._cursor_pos
            else:
                self._sel_anchor = -1
            self._cursor_pos = new_pos
            self._update_scroll()
            return True

        if event.key == pygame.K_HOME:
            if shift:
                if self._sel_anchor == -1:
                    self._sel_anchor = self._cursor_pos
            else:
                self._sel_anchor = -1
            self._cursor_pos = 0
            self._update_scroll()
            return True

        if event.key == pygame.K_END:
            if shift:
                if self._sel_anchor == -1:
                    self._sel_anchor = self._cursor_pos
            else:
                self._sel_anchor = -1
            self._cursor_pos = len(self._text)
            self._update_scroll()
            return True

        # ── Deletion ─────────────────────────────────────────────────────────
        if event.key == pygame.K_BACKSPACE:
            self._push_undo()
            if ctrl and not self._has_selection():
                # Delete word to the left
                new_pos = self._find_word_start(self._cursor_pos)
                self._text = self._text[:new_pos] + self._text[self._cursor_pos:]
                self._cursor_pos = new_pos
                self._update_scroll()
                if self._on_change:
                    self._on_change(self)
            else:
                self._delete_char(forward=False)
            return True

        if event.key == pygame.K_DELETE:
            self._push_undo()
            if ctrl and not self._has_selection():
                # Delete word to the right
                end = self._find_word_end(self._cursor_pos)
                self._text = self._text[:self._cursor_pos] + self._text[end:]
                if self._on_change:
                    self._on_change(self)
            else:
                self._delete_char(forward=True)
            return True

        # ── Submit / Escape ───────────────────────────────────────────────────
        if event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
            if self._on_submit:
                self._on_submit(self)
            return True

        if event.key == pygame.K_ESCAPE:
            if self._has_selection():
                self._sel_anchor = -1
            else:
                self.blur()
            return True

        # ── Printable character ───────────────────────────────────────────────
        if event.unicode and event.unicode.isprintable():
            self._push_undo()
            self._insert_text(event.unicode)
            return True

        return False

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------

    def update(self, dt: float) -> None:
        super().update(dt)
        if self.focused:
            self._cursor_timer += dt
            if self._cursor_timer >= self._cursor_blink_rate:
                self._cursor_timer = 0
                self._cursor_visible = not self._cursor_visible
        else:
            self._cursor_visible = False

    # -------------------------------------------------------------------------
    # Draw
    # -------------------------------------------------------------------------

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return

        abs_rect = self.absolute_rect
        bg_color = self._bg_color or (50, 50, 50)

        pygame.draw.rect(surface, bg_color, abs_rect, border_radius=self._border_radius)

        if self._border_width > 0:
            if self.focused:
                border_color = self._focus_border_color or (66, 135, 245)
            else:
                border_color = self._border_color or (80, 80, 80)
            pygame.draw.rect(
                surface, border_color, abs_rect,
                width=self._border_width, border_radius=self._border_radius
            )

        content_rect = pygame.Rect(
            abs_rect.x + self._padding,
            abs_rect.y,
            abs_rect.width - (self._padding * 2),
            abs_rect.height,
        )

        old_clip = surface.get_clip()
        surface.set_clip(content_rect.clip(old_clip))

        font = self._get_font()

        # Selection highlight (drawn before text so text renders on top)
        if self.focused and self._has_selection():
            start, end = self._selection_range()
            sx = font.size(self._text[:start])[0] - self._scroll_offset
            ex = font.size(self._text[:end])[0] - self._scroll_offset
            sel_rect = pygame.Rect(
                content_rect.x + sx,
                abs_rect.y + self._padding,
                ex - sx,
                abs_rect.height - self._padding * 2,
            )
            # Clip selection rect to visible content area
            sel_rect.x = max(sel_rect.x, content_rect.x)
            sel_rect.right = min(sel_rect.right, content_rect.right)
            if sel_rect.width > 0:
                pygame.draw.rect(surface, (70, 120, 200), sel_rect)

        # Text or placeholder
        if self._text:
            text_color = self._text_color or (255, 255, 255)
            text_surface = font.render(self._text, True, text_color)
            text_y = abs_rect.y + (abs_rect.height - text_surface.get_height()) // 2
            surface.blit(text_surface, (content_rect.x - self._scroll_offset, text_y))
        elif self._placeholder and not self.focused:
            ph_color = self._placeholder_color or (100, 100, 100)
            ph_surface = font.render(self._placeholder, True, ph_color)
            text_y = abs_rect.y + (abs_rect.height - ph_surface.get_height()) // 2
            surface.blit(ph_surface, (content_rect.x, text_y))

        # Cursor (not shown when there is a selection)
        if self.focused and self._cursor_visible and not self._has_selection():
            cursor_x = font.size(self._text[:self._cursor_pos])[0] - self._scroll_offset
            cursor_color = self._text_color or (255, 255, 255)
            cursor_rect = pygame.Rect(
                content_rect.x + cursor_x,
                abs_rect.y + self._padding,
                2,
                abs_rect.height - (self._padding * 2),
            )
            pygame.draw.rect(surface, cursor_color, cursor_rect)

        surface.set_clip(old_clip)
        self.draw_children(surface)
