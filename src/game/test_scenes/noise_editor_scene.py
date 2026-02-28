"""
Noise Editor Scene - Interactive noise parameter editor with live preview.

Displays a grayscale tilemap generated from a noise and provides UI controls
to tweak **every** noise parameter.  Pressing *Actualizar* regenerates the
preview with the current settings.

Sections:
  • General      – seed, noise_type, frequency, offset_x, offset_y
  • Fractal      – fractal_type, octaves, lacunarity, persistence,
                   weighted_strength, ping_pong_strength
  • Cellular     – cellular_distance_function, cellular_return_type,
                   cellular_jitter
  • Domain Warp  – enabled (checkbox), type, amplitude, frequency,
                   fractal_type, fractal_octaves, fractal_lacunarity,
                   fractal_gain
  • Preview size – width, height
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pygame

from src.core.tilemap.tilemap import TileMap
from src.core.tilemap.tileset import TileSet
from src.core.camera.camera import Camera
from src.ui import (
    UIManager, Label, Button, TextInput, Checkbox, Slider,
    VBox, HBox, ScrollView, Dropdown, NumericInput,
)
from .base_scene import BaseScene

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
_Matrix2D = None
_NoiseGenerator2D = None


def _get_matrix2d_class():
    global _Matrix2D
    if _Matrix2D is None:
        from src.virigir_math_utilities.matrix.matrix2d import Matrix2D
        _Matrix2D = Matrix2D
    return _Matrix2D


def _get_noise_generator_class():
    global _NoiseGenerator2D
    if _NoiseGenerator2D is None:
        from src.virigir_math_utilities.noise.generators.noise2d import NoiseGenerator2D
        _NoiseGenerator2D = NoiseGenerator2D
    return _NoiseGenerator2D


# ---------------------------------------------------------------------------
# Enum definitions (kept as plain lists so we don't need to import enums
# at module level — avoids the circular-import issue).
# ---------------------------------------------------------------------------
NOISE_TYPES = ["PERLIN", "SIMPLEX", "SIMPLEX_SMOOTH", "CELLULAR", "VALUE_CUBIC", "VALUE"]
FRACTAL_TYPES = ["NONE", "FBM", "RIDGED", "PING_PONG"]
CELLULAR_DIST_FUNCS = ["EUCLIDEAN", "EUCLIDEAN_SQUARED", "MANHATTAN", "HYBRID"]
CELLULAR_RETURN_TYPES = [
    "CELL_VALUE", "DISTANCE", "DISTANCE_2",
    "DISTANCE_2_ADD", "DISTANCE_2_SUB", "DISTANCE_2_MUL", "DISTANCE_2_DIV",
]
DOMAIN_WARP_TYPES = ["SIMPLEX", "SIMPLEX_REDUCED", "BASIC_GRID"]
DOMAIN_WARP_FRACTAL_TYPES = ["NONE", "PROGRESSIVE", "INDEPENDENT"]

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------
GRAYSCALE_STEPS = 32
TILE_SIZE = 4
CAMERA_SPEED = 500
ZOOM_SPEED = 1.5

PANEL_WIDTH = 400        # default panel width in pixels
_SCROLL_PADDING = 10     # ScrollView padding (applied on all sides)
_SCROLLBAR_W = 10        # width of the vertical scrollbar
# Usable content width for the default panel width (kept for default args):
PANEL_CONTENT_W = PANEL_WIDTH - 2 * _SCROLL_PADDING - _SCROLLBAR_W  # = 370

ROW_LABEL_W = 128        # fixed width for the label column in a _row
ROW_SPACING = 8          # horizontal gap between label and widget

# Divider constants
_PANEL_MIN_W = 150       # minimum panel width
_DIVIDER_W = 4           # visible divider bar width in pixels
_DIVIDER_HIT_W = 10      # wider hit area for easier grabbing

PANEL_BG = (30, 30, 45, 240)
SECTION_BG = (40, 40, 58, 200)
LABEL_COLOR = (190, 190, 210)
TITLE_COLOR = (220, 220, 255)
INPUT_BG = (50, 50, 65)
INPUT_BORDER = (100, 100, 130)
FOCUS_BORDER = (100, 160, 255)

CONFIG_PATH = Path(__file__).parent.parent.parent.parent / "configs" / "config.json"


# ═══════════════════════════════════════════════════════════════════════════
# Helper: build a labelled row  (Label + widget)  inside a VBox
# ═══════════════════════════════════════════════════════════════════════════

def _lbl(text: str, font_size: int = 16) -> Label:
    return Label(text=text, font_size=font_size, color=LABEL_COLOR, auto_size=True)


def _section_title(text: str) -> Label:
    return Label(text=text, font_size=20, color=TITLE_COLOR, auto_size=True)


def _section_spacer() -> Label:
    """Small vertical gap between sections."""
    return Label(text="", font_size=4, auto_size=True)


def _numeric_input(
    default: float,
    *,
    min_value: float = float("-inf"),
    max_value: float = float("inf"),
    step: float = 1.0,
) -> NumericInput:
    return NumericInput(
        width=80, height=26,
        value=default, min_value=min_value, max_value=max_value, step=step,
        font_size=16,
        bg_color=INPUT_BG, text_color=(255, 255, 255),
        border_color=INPUT_BORDER, focus_border_color=FOCUS_BORDER,
        border_radius=4,
    )


def _dropdown(options: List[str], index: int = 0) -> Dropdown:
    return Dropdown(
        width=180, height=26, options=options, selected_index=index,
        font_size=16, bg_color=INPUT_BG, text_color=(255, 255, 255),
        border_color=INPUT_BORDER, selected_color=(50, 130, 80),
        hover_color=(70, 70, 90), max_visible=6,
        border_radius=4,
    )


def _row(label_text: str, widget, label_w: int = ROW_LABEL_W,
         content_w: int = PANEL_CONTENT_W) -> HBox:
    """Fixed-width label on the left; widget stretches to fill the remainder.

    The row is exactly content_w wide so it always fits the panel
    viewport without overflowing or being clipped.
    """
    widget_w = max(50, content_w - label_w - ROW_SPACING)
    widget.width = widget_w

    row = HBox(
        width=content_w, height=widget.height,
        spacing=ROW_SPACING, align='center', auto_size=False,
    )
    lbl = _lbl(label_text)
    lbl.width = label_w
    row.add_child(lbl)
    row.add_child(widget)
    return row


# ═══════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════

class NoiseEditorScene(BaseScene):
    """Interactive noise editor with live tilemap preview."""

    def __init__(self):
        super().__init__(
            name="Noise Editor",
            description="Edit noise parameters and preview the result as a grayscale tilemap",
        )
        self.running = True

        # UI
        self._ui: Optional[UIManager] = None
        self._scroll_view: Optional[ScrollView] = None

        # Viewer
        self._tilemap: Optional[TileMap] = None
        self._tileset: Optional[TileSet] = None
        self._camera: Optional[Camera] = None
        self._matrix = None  # Matrix2D, persists across updates
        self._map_w = 0
        self._map_h = 0

        # Controls – will be populated in _build_ui
        self._ctrl: Dict[str, Any] = {}

        # Panel width (user-resizable)
        self._panel_width: int = PANEL_WIDTH

        # Divider drag state
        self._dragging_divider: bool = False
        self._divider_start_x: int = 0
        self._divider_start_panel_w: int = 0

    # -- lifecycle -----------------------------------------------------------

    def on_enter(self):
        screen = pygame.display.get_surface()
        sw, sh = screen.get_size()
        self._build_ui(sw, sh)
        # Generate an initial preview
        self._on_update()

    def on_exit(self):
        self._tilemap = None
        self._tileset = None
        self._camera = None
        self._matrix = None

    def on_resize(self, sw: int, sh: int) -> None:
        """Handle window resize: update UIManager, panel height and camera."""
        if self._ui:
            self._ui.resize(sw, sh)
        if self._scroll_view:
            self._scroll_view.height = sh
        if self._camera:
            self._camera.width = sw
            self._camera.height = sh

    # -- UI construction -----------------------------------------------------

    def _panel_content_w(self) -> int:
        """Usable width inside the scroll view for the current panel width."""
        return max(50, self._panel_width - 2 * _SCROLL_PADDING - _SCROLLBAR_W)

    def _build_ui(self, sw: int, sh: int):
        self._ui = UIManager(sw, sh)
        content_w = self._panel_content_w()

        # ── Left panel (scrollable controls) ────────────────────────────────
        panel_vbox = VBox(
            x=0, y=0, width=content_w,
            spacing=8, align=VBox.ALIGN_LEFT,
            auto_size=True, padding=0,
        )

        self._build_general_section(panel_vbox, content_w)
        self._build_fractal_section(panel_vbox, content_w)
        self._build_cellular_section(panel_vbox, content_w)
        self._build_domain_warp_section(panel_vbox, content_w)
        self._build_preview_section(panel_vbox, content_w)
        self._build_buttons(panel_vbox, content_w)

        scroll = ScrollView(
            x=0, y=0, width=self._panel_width, height=sh,
            content_height=0,   # auto-calculate from children
            bg_color=PANEL_BG,
            scroll_speed=30,
            show_scrollbar=True,
            scrollbar_color=(110, 110, 140),
            scrollbar_track_color=(45, 45, 60),
            padding=_SCROLL_PADDING,
        )
        scroll.add_child(panel_vbox)
        self._ui.add(scroll)
        self._scroll_view = scroll

    def _rebuild_panel(self) -> None:
        """Rebuild the panel UI with the current panel width, preserving control values."""
        # Capture current state before clearing
        try:
            cfg = self._read_config()
        except Exception:
            cfg = {}
        pw_text = self._ctrl["preview_w"].text if "preview_w" in self._ctrl else "256"
        ph_text = self._ctrl["preview_h"].text if "preview_h" in self._ctrl else "256"
        info_text = self._info_label.text if hasattr(self, "_info_label") else ""

        screen = pygame.display.get_surface()
        sw, sh = screen.get_size()
        self._ui.clear()
        self._build_ui(sw, sh)

        # Restore state
        if cfg:
            self._write_config(cfg)
        if "preview_w" in self._ctrl:
            self._ctrl["preview_w"].text = pw_text
        if "preview_h" in self._ctrl:
            self._ctrl["preview_h"].text = ph_text
        if hasattr(self, "_info_label"):
            self._info_label.text = info_text

    # ── Section builders ────────────────────────────────────────────────────

    def _build_general_section(self, parent: VBox, content_w: int):
        parent.add_child(_section_title("─── General ───"))

        inp_seed = _numeric_input(0, min_value=0, max_value=2**31 - 1, step=1)
        self._ctrl["seed"] = inp_seed
        parent.add_child(_row("Seed", inp_seed, content_w=content_w))

        dd_noise = _dropdown(NOISE_TYPES, 0)
        self._ctrl["noise_type"] = dd_noise
        parent.add_child(_row("Noise type", dd_noise, content_w=content_w))

        inp_freq = _numeric_input(0.01, min_value=0.0001, max_value=10.0, step=0.001)
        self._ctrl["frequency"] = inp_freq
        parent.add_child(_row("Frequency", inp_freq, content_w=content_w))

        inp_ox = _numeric_input(0, min_value=-10000, max_value=10000, step=10)
        self._ctrl["offset_x"] = inp_ox
        parent.add_child(_row("Offset X", inp_ox, content_w=content_w))

        inp_oy = _numeric_input(0, min_value=-10000, max_value=10000, step=10)
        self._ctrl["offset_y"] = inp_oy
        parent.add_child(_row("Offset Y", inp_oy, content_w=content_w))

    def _build_fractal_section(self, parent: VBox, content_w: int):
        parent.add_child(_section_spacer())
        parent.add_child(_section_title("─── Fractal ───"))

        dd_frac = _dropdown(FRACTAL_TYPES, 1)
        self._ctrl["fractal_type"] = dd_frac
        parent.add_child(_row("Fractal type", dd_frac, content_w=content_w))

        inp_oct = _numeric_input(5, min_value=1, max_value=16, step=1)
        self._ctrl["octaves"] = inp_oct
        parent.add_child(_row("Octaves", inp_oct, content_w=content_w))

        inp_lac = _numeric_input(2.0, min_value=0.1, max_value=8.0, step=0.1)
        self._ctrl["lacunarity"] = inp_lac
        parent.add_child(_row("Lacunarity", inp_lac, content_w=content_w))

        inp_per = _numeric_input(0.5, min_value=0.0, max_value=1.0, step=0.05)
        self._ctrl["persistence"] = inp_per
        parent.add_child(_row("Persistence", inp_per, content_w=content_w))

        inp_ws = _numeric_input(0.0, min_value=-1.0, max_value=2.0, step=0.05)
        self._ctrl["weighted_strength"] = inp_ws
        parent.add_child(_row("Weighted str.", inp_ws, content_w=content_w))

        inp_pp = _numeric_input(2.0, min_value=0.0, max_value=10.0, step=0.1)
        self._ctrl["ping_pong_strength"] = inp_pp
        parent.add_child(_row("Ping-pong str.", inp_pp, content_w=content_w))

    def _build_cellular_section(self, parent: VBox, content_w: int):
        parent.add_child(_section_spacer())
        parent.add_child(_section_title("─── Cellular ───"))

        dd_dist = _dropdown(CELLULAR_DIST_FUNCS, 1)
        self._ctrl["cellular_distance_function"] = dd_dist
        parent.add_child(_row("Distance func.", dd_dist, content_w=content_w))

        dd_ret = _dropdown(CELLULAR_RETURN_TYPES, 1)
        self._ctrl["cellular_return_type"] = dd_ret
        parent.add_child(_row("Return type", dd_ret, content_w=content_w))

        inp_jit = _numeric_input(1.0, min_value=0.0, max_value=2.0, step=0.05)
        self._ctrl["cellular_jitter"] = inp_jit
        parent.add_child(_row("Jitter", inp_jit, content_w=content_w))

    def _build_domain_warp_section(self, parent: VBox, content_w: int):
        parent.add_child(_section_spacer())
        parent.add_child(_section_title("─── Domain Warp ───"))

        cb_dw = Checkbox(
            text="Enabled", checked=False, box_size=18,
            text_color=LABEL_COLOR, font_size=16,
        )
        self._ctrl["domain_warp_enabled"] = cb_dw
        parent.add_child(cb_dw)

        dd_dwt = _dropdown(DOMAIN_WARP_TYPES, 0)
        self._ctrl["domain_warp_type"] = dd_dwt
        parent.add_child(_row("Warp type", dd_dwt, content_w=content_w))

        inp_amp = _numeric_input(30.0, min_value=0.0, max_value=500.0, step=1.0)
        self._ctrl["domain_warp_amplitude"] = inp_amp
        parent.add_child(_row("Amplitude", inp_amp, content_w=content_w))

        inp_dwf = _numeric_input(0.05, min_value=0.0001, max_value=1.0, step=0.005)
        self._ctrl["domain_warp_frequency"] = inp_dwf
        parent.add_child(_row("Frequency", inp_dwf, content_w=content_w))

        dd_dwft = _dropdown(DOMAIN_WARP_FRACTAL_TYPES, 0)
        self._ctrl["domain_warp_fractal_type"] = dd_dwft
        parent.add_child(_row("Fractal type", dd_dwft, content_w=content_w))

        inp_dwo = _numeric_input(5, min_value=1, max_value=16, step=1)
        self._ctrl["domain_warp_fractal_octaves"] = inp_dwo
        parent.add_child(_row("Octaves", inp_dwo, content_w=content_w))

        inp_dwl = _numeric_input(2.0, min_value=0.1, max_value=8.0, step=0.1)
        self._ctrl["domain_warp_fractal_lacunarity"] = inp_dwl
        parent.add_child(_row("Lacunarity", inp_dwl, content_w=content_w))

        inp_dwg = _numeric_input(0.5, min_value=0.0, max_value=2.0, step=0.05)
        self._ctrl["domain_warp_fractal_gain"] = inp_dwg
        parent.add_child(_row("Gain", inp_dwg, content_w=content_w))

    def _build_preview_section(self, parent: VBox, content_w: int):
        parent.add_child(_section_spacer())
        parent.add_child(_section_title("─── Dimensiones ───"))

        inp_w = _numeric_input(256, min_value=1, max_value=4096, step=16)
        self._ctrl["preview_w"] = inp_w
        parent.add_child(_row("Ancho (X)", inp_w, content_w=content_w))

        inp_h = _numeric_input(256, min_value=1, max_value=4096, step=16)
        self._ctrl["preview_h"] = inp_h
        parent.add_child(_row("Alto (Y)", inp_h, content_w=content_w))

    def _build_buttons(self, parent: VBox, content_w: int):
        parent.add_child(_section_spacer())

        # Preset dropdown (if config.json has presets)
        noise_names = self._load_noise_names()
        if noise_names:
            parent.add_child(_section_title("─── Preset ───"))
            dd_presets = _dropdown(noise_names, 0)
            self._ctrl["preset_dropdown"] = dd_presets
            parent.add_child(_row("Preset", dd_presets, content_w=content_w))

        parent.add_child(_section_spacer())

        # Action buttons — centered in the panel
        btn_row = HBox(
            width=content_w, height=36,
            spacing=12, align='center', justify='center', auto_size=False,
        )

        btn_update = Button(
            width=160, height=36, text="Actualizar", font_size=18,
            bg_color=(50, 130, 80), hover_color=(70, 160, 100),
            pressed_color=(40, 100, 65), border_radius=6,
        )
        btn_update.on_click(lambda _b: self._on_update())
        btn_row.add_child(btn_update)

        btn_load = Button(
            width=160, height=36, text="Cargar JSON", font_size=18,
            bg_color=(60, 90, 160), hover_color=(80, 115, 190),
            pressed_color=(45, 70, 130), border_radius=6,
        )
        btn_load.on_click(lambda _b: self._on_load_json())
        btn_row.add_child(btn_load)

        parent.add_child(btn_row)

        parent.add_child(_section_spacer())

        # Info / error label
        self._info_label = Label(text="", font_size=16, color=(255, 200, 80), auto_size=True)
        parent.add_child(self._info_label)

    # -- Read controls -------------------------------------------------------

    def _read_config(self) -> Dict[str, Any]:
        """Build a noise config dict from the current UI control values."""
        c = self._ctrl

        def _int(key: str, default: int = 0) -> int:
            try:
                return int(c[key].text)
            except (ValueError, AttributeError):
                return default

        def _float(key: str, default: float = 0.0) -> float:
            try:
                return float(c[key].text)
            except (ValueError, AttributeError):
                return default

        cfg: Dict[str, Any] = {
            # General
            "seed": _int("seed", 0),
            "noise_type": c["noise_type"].selected_index,
            "frequency": _float("frequency", 0.01),
            "offset_x": _int("offset_x", 0),
            "offset_y": _int("offset_y", 0),
            # Fractal
            "fractal_type": c["fractal_type"].selected_index,
            "octaves": _int("octaves", 5),
            "lacunarity": _float("lacunarity", 2.0),
            "persistence": _float("persistence", 0.5),
            "weighted_strength": _float("weighted_strength", 0.0),
            "ping_pong_strength": _float("ping_pong_strength", 2.0),
            # Cellular
            "cellular_distance_function": c["cellular_distance_function"].selected_index,
            "cellular_return_type": c["cellular_return_type"].selected_index,
            "cellular_jitter": _float("cellular_jitter", 1.0),
            # Domain warp
            "domain_warp_enabled": 1 if c["domain_warp_enabled"].checked else 0,
            "domain_warp_type": c["domain_warp_type"].selected_index,
            "domain_warp_amplitude": _float("domain_warp_amplitude", 30.0),
            "domain_warp_frequency": _float("domain_warp_frequency", 0.05),
            "domain_warp_fractal_type": c["domain_warp_fractal_type"].selected_index,
            "domain_warp_fractal_octaves": _int("domain_warp_fractal_octaves", 5),
            "domain_warp_fractal_lacunarity": _float("domain_warp_fractal_lacunarity", 2.0),
            "domain_warp_fractal_gain": _float("domain_warp_fractal_gain", 0.5),
        }
        return cfg

    def _write_config(self, cfg: Dict[str, Any]):
        """Populate UI controls from a config dict."""
        c = self._ctrl

        def _set_text(key: str, value):
            if key in c:
                c[key].text = str(value)

        def _set_dropdown_by_name(key: str, value, options: List[str]):
            """Set dropdown by enum name or int value."""
            if key not in c:
                return
            if isinstance(value, int):
                c[key].selected_index = value
            elif isinstance(value, str):
                upper = value.upper()
                for i, opt in enumerate(options):
                    if opt == upper:
                        c[key].selected_index = i
                        return
                c[key].selected_index = 0

        # General
        _set_text("seed", cfg.get("seed", 0))
        _set_dropdown_by_name("noise_type", cfg.get("noise_type", 0), NOISE_TYPES)
        _set_text("frequency", cfg.get("frequency", 0.01))
        _set_text("offset_x", cfg.get("offset_x", 0))
        _set_text("offset_y", cfg.get("offset_y", 0))

        # Fractal
        _set_dropdown_by_name("fractal_type", cfg.get("fractal_type", 1), FRACTAL_TYPES)
        _set_text("octaves", cfg.get("octaves", 5))
        _set_text("lacunarity", cfg.get("lacunarity", 2.0))
        _set_text("persistence", cfg.get("persistence", 0.5))
        _set_text("weighted_strength", cfg.get("weighted_strength", 0.0))
        _set_text("ping_pong_strength", cfg.get("ping_pong_strength", 2.0))

        # Cellular
        _set_dropdown_by_name(
            "cellular_distance_function",
            cfg.get("cellular_distance_function", 0),
            CELLULAR_DIST_FUNCS,
        )
        _set_dropdown_by_name(
            "cellular_return_type",
            cfg.get("cellular_return_type", 1),
            CELLULAR_RETURN_TYPES,
        )
        _set_text("cellular_jitter", cfg.get("cellular_jitter", 1.0))

        # Domain warp
        if "domain_warp_enabled" in c:
            c["domain_warp_enabled"].checked = bool(cfg.get("domain_warp_enabled", 0))
        _set_dropdown_by_name(
            "domain_warp_type",
            cfg.get("domain_warp_type", 0),
            DOMAIN_WARP_TYPES,
        )
        _set_text("domain_warp_amplitude", cfg.get("domain_warp_amplitude", 30.0))
        _set_text("domain_warp_frequency", cfg.get("domain_warp_frequency", 0.05))
        _set_dropdown_by_name(
            "domain_warp_fractal_type",
            cfg.get("domain_warp_fractal_type", 0),
            DOMAIN_WARP_FRACTAL_TYPES,
        )
        _set_text("domain_warp_fractal_octaves", cfg.get("domain_warp_fractal_octaves", 5))
        _set_text("domain_warp_fractal_lacunarity", cfg.get("domain_warp_fractal_lacunarity", 2.0))
        _set_text("domain_warp_fractal_gain", cfg.get("domain_warp_fractal_gain", 0.5))

    # -- JSON helpers --------------------------------------------------------

    @staticmethod
    def _load_noise_names() -> List[str]:
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return list(data.get("noise", {}).keys())
        except Exception:
            return []

    @staticmethod
    def _load_noise_config(name: str) -> Optional[Dict[str, Any]]:
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("noise", {}).get(name)
        except Exception:
            return None

    def _on_load_json(self):
        """Load the preset selected in the preset dropdown."""
        dd = self._ctrl.get("preset_dropdown")
        if dd is None:
            self._info_label.text = "No hay presets disponibles"
            return
        name = dd.selected_text
        cfg = self._load_noise_config(name)
        if cfg is None:
            self._info_label.text = f"No se pudo cargar '{name}'"
            return
        self._write_config(cfg)
        self._info_label.text = f"Preset '{name}' cargado"
        self._on_update()

    # -- Generation ----------------------------------------------------------

    def _on_update(self):
        """Read controls, build noise, generate matrix, refresh tilemap."""
        cfg = self._read_config()

        # Dimensions
        try:
            pw = max(1, min(4096, int(self._ctrl["preview_w"].text)))
            ph = max(1, min(4096, int(self._ctrl["preview_h"].text)))
        except ValueError:
            self._info_label.text = "Tamaño inválido"
            return

        NoiseGenerator2D = _get_noise_generator_class()
        Matrix2D = _get_matrix2d_class()

        try:
            noise = NoiseGenerator2D.from_dict(cfg)
        except Exception as e:
            self._info_label.text = f"Error: {e}"
            print(f"[NoiseEditor] {e}")
            return

        # Decide whether to reuse the existing matrix or regenerate
        if self._matrix is not None:
            old_rows, old_cols = self._matrix._shape  # (ph, pw)
            new_rows, new_cols = ph, pw

            if new_rows > old_rows or new_cols > old_cols:
                # Size grew: resize in-place and fill only the new cells
                try:
                    self._matrix.resize((new_rows, new_cols), default_value=None)
                    # New rows strip: rows [old_rows, new_rows) × all cols
                    if new_rows > old_rows:
                        self._matrix.fill_values_from_noise_region(
                            noise, 0, new_cols, old_rows, new_rows
                        )
                    # New cols in old rows: rows [0, old_rows) × cols [old_cols, new_cols)
                    if new_cols > old_cols:
                        self._matrix.fill_values_from_noise_region(
                            noise, old_cols, new_cols, 0, min(old_rows, new_rows)
                        )
                except Exception as e:
                    self._info_label.text = f"Error: {e}"
                    print(f"[NoiseEditor] {e}")
                    return
            else:
                # Same size or smaller: trim and regenerate
                try:
                    self._matrix = Matrix2D.create_from_noise(noise, ph, pw)
                except Exception as e:
                    self._info_label.text = f"Error: {e}"
                    print(f"[NoiseEditor] {e}")
                    return
        else:
            try:
                self._matrix = Matrix2D.create_from_noise(noise, ph, pw)
            except Exception as e:
                self._info_label.text = f"Error: {e}"
                print(f"[NoiseEditor] {e}")
                return

        self._map_w = pw
        self._map_h = ph

        # Tileset (lazily created once)
        if self._tileset is None:
            self._tileset = TileSet.generate_grayscale_tileset(
                nsteps=GRAYSCALE_STEPS,
                tile_size=(TILE_SIZE, TILE_SIZE),
                columns=GRAYSCALE_STEPS,
                white_to_black=True,
            )

        # Tilemap (recreated whenever dimensions change)
        self._tilemap = TileMap(
            width=pw, height=ph,
            tile_size=(TILE_SIZE, TILE_SIZE),
        )
        self._tilemap.tileset = self._tileset

        tile_ids = np.clip(
            (self._matrix._data * (GRAYSCALE_STEPS - 1)).astype(int),
            0, GRAYSCALE_STEPS - 1,
        )
        for r in range(ph):
            for c in range(pw):
                self._tilemap.set_tile(c, r, int(tile_ids[r, c]))

        # Camera: create only on first call — preserve position/zoom afterwards
        if self._camera is None:
            screen = pygame.display.get_surface()
            sw, sh = screen.get_size()
            world_w = pw * TILE_SIZE
            world_h = ph * TILE_SIZE
            cam_x = max(0.0, (world_w - (sw - self._panel_width)) / 2)
            cam_y = max(0.0, (world_h - sh) / 2)
            self._camera = Camera(
                x=cam_x, y=cam_y,
                width=sw, height=sh,
                zoom=1.0, min_zoom=0.05, max_zoom=20.0,
            )

        self._info_label.text = f"Generado {ph}×{pw}"
        print(f"[NoiseEditor] Generated {ph}×{pw}")

    # -- events --------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.running = False
            return

        # ── Divider drag ──────────────────────────────────────────────────
        screen = pygame.display.get_surface()
        sw = screen.get_width()
        divider_x = self._panel_width

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            mx = event.pos[0]
            if abs(mx - divider_x) <= _DIVIDER_HIT_W // 2:
                self._dragging_divider = True
                self._divider_start_x = mx
                self._divider_start_panel_w = self._panel_width
                return  # consume — don't pass to UI

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._dragging_divider:
                self._dragging_divider = False
                self._rebuild_panel()
                return

        elif event.type == pygame.MOUSEMOTION:
            if self._dragging_divider:
                dx = event.pos[0] - self._divider_start_x
                new_w = self._divider_start_panel_w + dx
                new_w = max(_PANEL_MIN_W, min(sw - 200, new_w))
                self._panel_width = new_w
                # Live feedback: just resize the scroll view boundary
                if self._scroll_view:
                    self._scroll_view.width = new_w
                return

        if self._ui:
            self._ui.handle_event(event)

    # -- update --------------------------------------------------------------

    def update(self, dt: float) -> None:
        if self._ui:
            self._ui.update(dt)

        # Cursor: show horizontal-resize when hovering the divider
        mx = pygame.mouse.get_pos()[0]
        if self._dragging_divider or abs(mx - self._panel_width) <= _DIVIDER_HIT_W // 2:
            pygame.mouse.set_cursor(
                getattr(pygame, "SYSTEM_CURSOR_SIZEWE", pygame.SYSTEM_CURSOR_ARROW)
            )
        else:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_ARROW)

        if not self._camera:
            return

        keys = pygame.key.get_pressed()

        # Zoom
        if keys[pygame.K_e]:
            self._camera.zoom *= ZOOM_SPEED ** dt
        elif keys[pygame.K_q]:
            self._camera.zoom /= ZOOM_SPEED ** dt

        zoom = self._camera.zoom
        visible_w = self._camera.width / zoom
        visible_h = self._camera.height / zoom
        world_w = self._map_w * TILE_SIZE
        world_h = self._map_h * TILE_SIZE
        self._camera.set_bounds(
            min_x=0, max_x=max(0.0, world_w - visible_w),
            min_y=0, max_y=max(0.0, world_h - visible_h),
        )

        speed = CAMERA_SPEED / zoom
        dx = dy = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx -= speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx += speed * dt
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            dy -= speed * dt
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy += speed * dt
        if dx or dy:
            self._camera.move(dx, dy)

    # -- draw ----------------------------------------------------------------

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill((20, 20, 30))

        sw, sh = screen.get_size()

        # Tilemap (offset to the right of the panel)
        if self._tilemap and self._camera and self._tileset:
            # Clip drawing area to the right of the panel
            clip_rect = pygame.Rect(self._panel_width, 0, sw - self._panel_width, sh)
            old_clip = screen.get_clip()
            screen.set_clip(clip_rect)

            sub = screen.subsurface(clip_rect)
            self._tilemap.draw(sub, self._camera, self._tileset)

            screen.set_clip(old_clip)

            # HUD over the tilemap area
            self._draw_hud(screen)

        # UI panel (drawn last so it's on top)
        if self._ui:
            self._ui.draw(screen)

        # Divider bar
        divider_color = (150, 150, 200) if self._dragging_divider else (70, 70, 90)
        pygame.draw.rect(
            screen, divider_color,
            pygame.Rect(self._panel_width - _DIVIDER_W // 2, 0, _DIVIDER_W, sh)
        )

    def _draw_hud(self, screen: pygame.Surface) -> None:
        small = pygame.font.Font(None, 18)
        cam = self._camera
        info = (
            f"Map: {self._map_h}×{self._map_w}  "
            f"Cam: ({int(cam.x)},{int(cam.y)})  "
            f"Zoom: {cam.zoom:.2f}x"
        )
        surf = small.render(info, True, (200, 200, 200))
        screen.blit(surf, (self._panel_width + 8, 8))

        hint = "WASD/Arrows: Move | Q/E: Zoom | ESC: Quit"
        hint_surf = small.render(hint, True, (130, 130, 130))
        rect = hint_surf.get_rect(bottomleft=(self._panel_width + 8, screen.get_height() - 8))
        screen.blit(hint_surf, rect)