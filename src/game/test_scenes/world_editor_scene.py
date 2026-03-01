"""
World Editor Scene – visualise and edit VGWorld generation parameters.

Layout
──────
Left panel (resizable via divider)
  ┌──────────────────────────────────┐
  │ [Parámetros] [Noises] [Matrices] │  ← TabBar
  ├──────────────────────────────────┤
  │  (scrollable content per tab)    │
  └──────────────────────────────────┘
Right area: grayscale tilemap viewer (Camera + TileMap)

Tabs
────
• Parámetros – all WorldParameterName values as editable NumericInputs.
              "Aplicar" syncs them to the VGWorld object.
• Noises     – sub-tabs per noise from config.json; shows key fields and a
              "Previsualizar" button that renders the noise in the tilemap.
• Matrices   – list of WorldMatrixName entries; each row has a "Generar"
              button that generates a preview-size matrix and shows it.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pygame

from src.core.tilemap.tilemap import TileMap
from src.core.tilemap.tileset import TileSet
from src.core.camera.camera import Camera
from src.ui import (
    UIManager, Label, Button, Checkbox, Dropdown,
    VBox, HBox, ScrollView, NumericInput, TabBar, TextInput,
)
from .base_scene import BaseScene

# ── Lazy imports ─────────────────────────────────────────────────────────────
_Matrix2D = None
_NoiseGenerator2D = None
_VGWorld = None
_WorldParameterName = None
_WorldNoiseName = None
_WorldMatrixName = None


def _get_Matrix2D():
    global _Matrix2D
    if _Matrix2D is None:
        from virigir_math_utilities.matrix.matrix2d import Matrix2D
        _Matrix2D = Matrix2D
    return _Matrix2D


def _get_NoiseGenerator2D():
    global _NoiseGenerator2D
    if _NoiseGenerator2D is None:
        from virigir_math_utilities.noise.generators.noise2d import NoiseGenerator2D
        _NoiseGenerator2D = NoiseGenerator2D
    return _NoiseGenerator2D


def _get_world_classes():
    global _VGWorld, _WorldParameterName, _WorldNoiseName, _WorldMatrixName
    if _VGWorld is None:
        from vgworld.world.world import (
            VGWorld, WorldParameterName, WorldNoiseName, WorldMatrixName,
        )
        _VGWorld = VGWorld
        _WorldParameterName = WorldParameterName
        _WorldNoiseName = WorldNoiseName
        _WorldMatrixName = WorldMatrixName
    return _VGWorld, _WorldParameterName, _WorldNoiseName, _WorldMatrixName


# ── Config paths ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_JSON  = _ROOT / "configs" / "config.json"
CONFIG_TOML  = _ROOT / "configs" / "world_configs.toml"

# ── Visual constants ──────────────────────────────────────────────────────────
PANEL_WIDTH    = 420
TAB_H          = 36
SCROLL_PAD     = 10
SCROLLBAR_W    = 12
ROW_LABEL_W    = 152
ROW_SPACING    = 8
GRAYSCALE_STEPS = 32
TILE_SIZE       = 4
CAMERA_SPEED    = 500
ZOOM_SPEED      = 1.5
_PANEL_MIN_W    = 180
_DIVIDER_W      = 4
_DIVIDER_HIT_W  = 10
BOTTOM_BAR_H    = 46   # fixed bottom bar height (holds "Previsualizar")

PANEL_BG      = (28, 28, 42, 245)
SECTION_BG    = (38, 38, 55, 200)
LABEL_COLOR   = (185, 185, 210)
TITLE_COLOR   = (215, 215, 255)
VALUE_COLOR   = (130, 200, 130)
INPUT_BG      = (48, 48, 62)
INPUT_BORDER  = (95, 95, 128)
FOCUS_BORDER  = (95, 155, 255)
BTN_GEN_BG    = (48, 125, 78)
BTN_GEN_HV    = (68, 155, 98)
BTN_PRV_BG    = (55, 85, 155)
BTN_PRV_HV    = (75, 110, 185)
BTN_APL_BG    = (130, 90, 40)
BTN_APL_HV    = (165, 115, 60)

# ── Noise types ───────────────────────────────────────────────────────────────
_NOISE_TYPE_NAMES  = ["PERLIN", "SIMPLEX", "SIMPLEX_SMOOTH", "CELLULAR", "VALUE_CUBIC", "VALUE"]
_FRACTAL_TYPE_NAMES = ["NONE", "FBM", "RIDGED", "PING_PONG"]

# Editable noise fields: (label, config_key, widget_type, *args)
# widget_type: 'int' | 'float' | 'dropdown' | 'bool'
_NOISE_FIELD_CONFIG = [
    ("Seed",            "seed",                       'int',      0,      2**31-1, 1),
    ("Noise Type",      "noise_type",                 'dropdown', _NOISE_TYPE_NAMES, None, None),
    ("Frequency",       "frequency",                  'float',    0.0001, 10.0,    0.001),
    ("Offset X",        "offset_x",                   'int',      -10000, 10000,   10),
    ("Offset Y",        "offset_y",                   'int',      -10000, 10000,   10),
    ("Fractal Type",    "fractal_type",               'dropdown', _FRACTAL_TYPE_NAMES, None, None),
    ("Octaves",         "octaves",                    'int',      1,      16,      1),
    ("Lacunarity",      "lacunarity",                 'float',    0.1,    8.0,     0.01),
    ("Persistence",     "persistence",                'float',    0.0,    1.0,     0.01),
    ("Weighted Str.",   "weighted_strength",          'float',    -1.0,   2.0,     0.01),
    ("Domain Warp",     "domain_warp_enabled",        'bool',     None,   None,    None),
    ("DW Amplitude",    "domain_warp_amplitude",      'float',    0.0,    500.0,   0.01),
    ("DW Frequency",    "domain_warp_frequency",      'float',    0.0001, 1.0,     0.01),
    ("DW Octaves",      "domain_warp_fractal_octaves",'int',      1,      16,      1),
]

# WorldParameterName display config: (label, is_int, min, max, step)
_PARAM_CONFIG: Dict[str, tuple] = {
    "global_seed":              ("Global Seed",           True,  0,      2**31-1, 1),
    "world_size_x":             ("World Size X",          True,  1,      8192,    16),
    "world_size_y":             ("World Size Y",          True,  1,      8192,    16),
    "equator_latitude":         ("Equator Latitude",      True,  -90,    90,      1),
    "min_continental_height":   ("Min. Continental H.",   False, 0.0,    1.0,     0.001),
    "peaks_and_valleys_scale":  ("PV Scale",              False, 0.0,    10.0,    0.01),
    "continental_scale":        ("Continental Scale",     False, 0.0,    10.0,    0.01),
    "sea_scale":                ("Sea Scale",             False, 0.0,    10.0,    0.01),
    "sea_elevation_threshold":  ("Sea Threshold",         False, 0.0,    1.0,     0.001),
    "island_scale":             ("Island Scale",          False, 0.0,    1.0,     0.01),
    "volcanic_island_scale":    ("Volcanic Island Scale", False, 0.0,    1.0,     0.01),
    "island_threshold":         ("Island Threshold",      False, 0.0,    1.0,     0.01),
    "out_to_sea_factor":        ("Out to Sea Factor",     False, 0.0,    1.0,     0.01),
}

# WorldMatrixName → (display label, noise key to generate from, binarize?)
_MATRIX_CONFIG: Dict[str, tuple] = {
    "continental_elevation": ("Continental Elevation", "base_elevation",  False),
    "elevation":             ("Elevation",             "base_elevation",  False),
    "is_volcanic_land":      ("Is Volcanic Land",      "volcanic_noise",  True),
    "is_continent":          ("Is Continent",          "base_elevation",  True),
    "latitude":              ("Latitude",              None,              False),
    "river":                 ("River",                 None,              False),
    "river_birth_positions": ("River Birth Positions", None,              False),
    "river_flow":            ("River Flow",            None,              False),
    "temperature":           ("Temperature",           None,              False),
}


# ── Helper widget factories ───────────────────────────────────────────────────

def _lbl(text: str, font_size: int = 16, color=LABEL_COLOR) -> Label:
    return Label(text=text, font_size=font_size, color=color, auto_size=True)


def _title(text: str) -> Label:
    return Label(text=text, font_size=19, color=TITLE_COLOR, auto_size=True)


def _spacer() -> Label:
    return Label(text="", font_size=6, auto_size=True)


def _numeric(value, *, min_value, max_value, step, w=90, decimals: int = -1) -> NumericInput:
    return NumericInput(
        width=w, height=26, value=value,
        min_value=min_value, max_value=max_value, step=step,
        decimals=decimals,
        font_size=16, bg_color=INPUT_BG, text_color=(255, 255, 255),
        border_color=INPUT_BORDER, focus_border_color=FOCUS_BORDER,
        border_radius=4,
    )


def _btn(text, bg, hv, w=120, h=30) -> Button:
    return Button(
        width=w, height=h, text=text, font_size=16,
        bg_color=bg, hover_color=hv,
        pressed_color=tuple(max(0, c - 25) for c in bg),
        border_radius=5,
    )


def _row(label_text: str, widget, content_w: int) -> HBox:
    widget_w = max(50, content_w - ROW_LABEL_W - ROW_SPACING)
    widget.width = widget_w
    row = HBox(width=content_w, height=widget.height,
                spacing=ROW_SPACING, align='center', auto_size=False)
    lbl = _lbl(label_text)
    lbl.width = ROW_LABEL_W
    row.add_child(lbl)
    row.add_child(widget)
    return row


def _dropdown_small(options: List[str], index: int = 0, w: int = 150) -> Dropdown:
    return Dropdown(
        width=w, height=26, options=options, selected_index=index,
        font_size=15, bg_color=INPUT_BG, text_color=(255, 255, 255),
        border_color=INPUT_BORDER, selected_color=(45, 115, 70),
        hover_color=(60, 60, 80), max_visible=6,
        border_radius=4,
    )


def _text_input(text: str = "", w: int = 150) -> TextInput:
    return TextInput(
        width=w, height=26, text=text,
        font_size=16, text_color=(255, 255, 255),
        bg_color=INPUT_BG, border_color=INPUT_BORDER,
        focus_border_color=FOCUS_BORDER, border_radius=4,
    )


def _make_noise_widget(field_type: str, config_value, arg0, arg1, arg2):
    """Create the right widget for a noise field."""
    if field_type == 'int':
        val = int(config_value) if config_value is not None else int(arg0)
        return _numeric(val, min_value=arg0, max_value=arg1, step=arg2)
    elif field_type == 'float':
        val = float(config_value) if config_value is not None else float(arg0)
        return _numeric(val, min_value=arg0, max_value=arg1, step=arg2, decimals=5)
    elif field_type == 'dropdown':
        options = arg0  # arg0 is the list
        if isinstance(config_value, int):
            idx = max(0, min(config_value, len(options) - 1))
        elif isinstance(config_value, str):
            upper = config_value.upper()
            idx = options.index(upper) if upper in options else 0
        else:
            idx = 0
        return _dropdown_small(options, idx)
    elif field_type == 'bool':
        return Checkbox(checked=bool(config_value), box_size=18,
                        text_color=LABEL_COLOR, font_size=16)
    return Label(text=str(config_value), auto_size=True)


def _read_noise_widget(widget, field_type: str):
    """Read the current value from an editable noise widget."""
    if field_type == 'int':
        try:
            return int(float(widget.text))
        except (ValueError, AttributeError):
            return 0
    elif field_type == 'float':
        try:
            return float(widget.text)
        except (ValueError, AttributeError):
            return 0.0
    elif field_type == 'dropdown':
        return widget.selected_index
    elif field_type == 'bool':
        return 1 if widget.checked else 0
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Scene
# ═══════════════════════════════════════════════════════════════════════════════

class WorldEditorScene(BaseScene):
    """Editor that exposes VGWorld parameters, noise preview, and matrix generation."""

    TABS = ["Parámetros", "Noises", "Matrices"]
    TAB_PARAMS   = 0
    TAB_NOISES   = 1
    TAB_MATRICES = 2

    def __init__(self):
        super().__init__(
            name="World Editor",
            description="Edit VGWorld parameters and visualise generation results",
        )
        self.running = True

        # World data
        self._world = None               # VGWorld instance (or None on failure)
        self._noise_configs: Dict[str, Dict[str, Any]] = {}   # from config.json
        self._noise_names: List[str] = []
        self._param_configs: Dict[str, Dict[str, Any]] = {}   # from config.json ["parameters"]
        self._param_config_names: List[str] = []

        # UI
        self._ui: Optional[UIManager] = None
        self._main_tab_bar: Optional[TabBar] = None
        self._scroll_view: Optional[ScrollView] = None
        self._active_tab: int = 0

        # Per-tab VBoxes (built lazily, cached)
        self._tab_vboxes: List[Optional[VBox]] = [None, None, None]

        # Params tab controls
        self._param_controls: Dict[str, NumericInput] = {}
        self._param_status: Optional[Label] = None
        self._active_param_cfg_idx: int = 0
        self._param_cfg_dropdown: Optional[Dropdown] = None
        self._param_cfg_name_input: Optional[TextInput] = None

        # Noises tab state
        self._noise_dropdown: Optional[Dropdown] = None
        self._noise_inner_scroll: Optional[ScrollView] = None
        self._noise_name_input: Optional[TextInput] = None
        self._active_noise_idx: int = 0
        self._noise_tab_vboxes: Dict[str, VBox] = {}   # noise_name → VBox (cached)
        self._noise_controls: Dict[str, Dict[str, tuple]] = {}  # noise_name → {key: (widget, type)}

        # Matrices/noise preview size (persisted across panel rebuilds)
        self._saved_pw: int = 1024
        self._saved_ph: int = 1024
        self._preview_w: Optional[NumericInput] = None
        self._preview_h: Optional[NumericInput] = None
        self._matrix_status: Dict[str, Label] = {}
        self._matrix_generated: Dict[str, bool] = {}
        self._generated_matrices: Dict[str, Any] = {}   # mat_key → Matrix2D
        self._matrix_view_buttons: Dict[str, Button] = {}

        # Fixed bottom bar
        self._bottom_bar = None
        self._preview_btn: Optional[Button] = None

        # Viewer
        self._tilemap: Optional[TileMap] = None
        self._tileset: Optional[TileSet] = None
        self._camera: Optional[Camera] = None
        self._current_matrix = None   # Matrix2D shown in viewer
        self._viewer_label: str = ""
        self._map_w: int = 0
        self._map_h: int = 0

        # General status
        self._status_bar: Optional[Label] = None

        # Divider
        self._panel_width: int = PANEL_WIDTH
        self._dragging_divider: bool = False
        self._divider_start_x: int = 0
        self._divider_start_panel_w: int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_enter(self) -> None:
        self._load_data()
        screen = pygame.display.get_surface()
        sw, sh = screen.get_size()
        self._build_ui(sw, sh)

    def on_exit(self) -> None:
        self._tilemap = None
        self._tileset = None
        self._camera = None
        self._current_matrix = None

    def on_resize(self, sw: int, sh: int) -> None:
        if self._ui:
            self._ui.resize(sw, sh)
        if self._main_tab_bar:
            self._main_tab_bar.width = self._panel_width
        if self._scroll_view:
            self._scroll_view.height = sh - TAB_H - BOTTOM_BAR_H
        if self._bottom_bar:
            self._bottom_bar.y = sh - BOTTOM_BAR_H
            self._bottom_bar.width = self._panel_width
        if self._status_bar:
            self._status_bar.x = self._panel_width + 8
            self._status_bar.y = sh - 20
        if self._camera:
            self._camera.width = sw
            self._camera.height = sh

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self) -> None:
        # Noise + parameter configs from config.json
        try:
            with open(CONFIG_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._noise_configs = data.get("noise", {})
            self._noise_names = list(self._noise_configs.keys())
            self._param_configs = data.get("parameters", {})
            self._param_config_names = list(self._param_configs.keys())
        except Exception as e:
            print(f"[WorldEditor] config.json load error: {e}")

        # VGWorld
        try:
            VGWorld, *_ = _get_world_classes()
            self._world = VGWorld("default_parameters")
            print("[WorldEditor] VGWorld loaded.")
        except Exception as e:
            import traceback
            print(f"[WorldEditor] VGWorld unavailable: {e}")
            traceback.print_exc()
            self._world = None

    # ── UI construction ───────────────────────────────────────────────────────

    def _panel_content_w(self) -> int:
        return max(60, self._panel_width - 2 * SCROLL_PAD - SCROLLBAR_W)

    def _build_ui(self, sw: int, sh: int) -> None:
        self._ui = UIManager(sw, sh)
        pw = self._panel_width

        # Main TabBar
        self._main_tab_bar = TabBar(
            x=0, y=0, width=pw, height=TAB_H,
            tabs=self.TABS,
            selected_index=self._active_tab,
            font_size=17,
        )
        self._main_tab_bar.on_tab_change(self._on_main_tab_change)
        self._ui.add(self._main_tab_bar)

        # ScrollView for tab content (leaves room for the fixed bottom bar)
        self._scroll_view = ScrollView(
            x=0, y=TAB_H, width=pw, height=sh - TAB_H - BOTTOM_BAR_H,
            bg_color=PANEL_BG,
            scroll_speed=30,
            show_scrollbar=True,
            scrollbar_color=(105, 105, 138),
            scrollbar_track_color=(42, 42, 58),
            padding=SCROLL_PAD,
        )
        self._ui.add(self._scroll_view)

        # ── Fixed bottom bar: size controls + "Previsualizar" ────────────────
        bar_y = sh - BOTTOM_BAR_H
        self._bottom_bar = HBox(
            x=0, y=bar_y, width=pw, height=BOTTOM_BAR_H,
            spacing=8, align='center', justify='center', auto_size=False,
            bg_color=(32, 32, 48, 255),
        )
        self._bottom_bar.add_child(_lbl("X:", font_size=15))
        self._preview_w = _numeric(self._saved_pw, min_value=16, max_value=2048, step=16, w=100)
        self._bottom_bar.add_child(self._preview_w)
        self._bottom_bar.add_child(_lbl("Y:", font_size=15))
        self._preview_h = _numeric(self._saved_ph, min_value=16, max_value=2048, step=16, w=100)
        self._bottom_bar.add_child(self._preview_h)
        self._preview_btn = _btn("Previsualizar", BTN_PRV_BG, BTN_PRV_HV, w=140, h=34)
        self._preview_btn.on_click(lambda _: self._preview_selected_noise())
        self._bottom_bar.add_child(self._preview_btn)
        self._ui.add(self._bottom_bar)

        # Status label below the bottom bar (right side of screen)
        self._status_bar = Label(
            text="", font_size=15, color=(255, 200, 80), auto_size=True,
        )
        self._status_bar.x = self._panel_width + 8
        self._status_bar.y = sh - 20
        self._ui.add(self._status_bar)

        # Show initial tab
        self._show_tab(self._active_tab)

    def _on_main_tab_change(self, idx: int) -> None:
        self._active_tab = idx
        self._show_tab(idx)

    def _show_tab(self, idx: int) -> None:
        """Swap the ScrollView content to the selected tab."""
        if self._scroll_view is None:
            return

        # Build tab content lazily
        if self._tab_vboxes[idx] is None:
            cw = self._panel_content_w()
            if idx == self.TAB_PARAMS:
                self._tab_vboxes[idx] = self._build_params_tab(cw)
            elif idx == self.TAB_NOISES:
                self._tab_vboxes[idx] = self._build_noises_tab(cw)
            elif idx == self.TAB_MATRICES:
                self._tab_vboxes[idx] = self._build_matrices_tab(cw)

        self._scroll_view.clear_children()
        vbox = self._tab_vboxes[idx]
        if vbox:
            self._scroll_view.add_child(vbox)
        # Reset scroll position
        self._scroll_view.scroll_y = 0

    # ═══════════════════════ TAB: PARÁMETROS ══════════════════════════════════

    def _build_params_tab(self, cw: int) -> VBox:
        vbox = VBox(width=cw, spacing=7, align=VBox.ALIGN_LEFT, auto_size=True, padding=0)

        vbox.add_child(_title("── Parámetros del Mundo ──"))
        vbox.add_child(_spacer())

        # Dropdown to select the active parameter config
        if self._param_config_names:
            sel_row = HBox(width=cw, height=30, spacing=8, align='center', auto_size=False)
            sel_lbl = _lbl("Config:", font_size=15)
            sel_lbl.width = 52
            sel_row.add_child(sel_lbl)
            self._param_cfg_dropdown = _dropdown_small(
                self._param_config_names,
                self._active_param_cfg_idx,
                w=max(60, cw - 60),
            )
            self._param_cfg_dropdown.on_change(self._on_param_cfg_change)
            sel_row.add_child(self._param_cfg_dropdown)
            vbox.add_child(sel_row)

            # Name row: editable config name + save button
            name_row = HBox(width=cw, height=30, spacing=6, align='center', auto_size=False)
            name_lbl = _lbl("Nombre:", font_size=15)
            name_lbl.width = 68
            name_row.add_child(name_lbl)
            active_name = self._param_config_names[self._active_param_cfg_idx]
            self._param_cfg_name_input = _text_input(text=active_name, w=max(60, cw - 160))
            name_row.add_child(self._param_cfg_name_input)
            btn_save = _btn("Guardar", BTN_APL_BG, BTN_APL_HV, w=80, h=26)
            btn_save.on_click(lambda _: self._save_param_config())
            name_row.add_child(btn_save)
            vbox.add_child(name_row)
            vbox.add_child(_spacer())

        # Load current values from the active JSON config (fallback: world, then TOML)
        raw_params = self._get_active_param_values()

        self._param_controls = {}
        for key, (label, is_int, mn, mx, step) in _PARAM_CONFIG.items():
            current = raw_params.get(key, mn)
            inp = _numeric(current, min_value=mn, max_value=mx, step=step)
            self._param_controls[key] = inp
            vbox.add_child(_row(label, inp, cw))

        vbox.add_child(_spacer())

        # Apply button
        btn_row = HBox(width=cw, height=32, spacing=12, align='center', justify='center', auto_size=False)
        btn_apply = _btn("Aplicar Cambios", BTN_APL_BG, BTN_APL_HV, w=160)
        btn_apply.on_click(lambda _: self._apply_params())
        btn_row.add_child(btn_apply)
        vbox.add_child(btn_row)

        vbox.add_child(_spacer())
        self._param_status = Label(text="", font_size=15, color=(255, 200, 80), auto_size=True)
        vbox.add_child(self._param_status)

        return vbox

    def _get_active_param_values(self) -> Dict[str, Any]:
        """Return the param dict for the currently selected config."""
        if self._param_config_names:
            name = self._param_config_names[self._active_param_cfg_idx]
            return dict(self._param_configs.get(name, {}))
        # fallback: read from running world
        if self._world:
            _, WPN, *_ = _get_world_classes()
            result = {}
            for key in _PARAM_CONFIG:
                try:
                    result[key] = self._world.parameters.get(WPN[key], 0)
                except Exception:
                    pass
            return result
        return {}

    def _apply_params(self) -> None:
        if self._world is None:
            self._set_status("VGWorld no disponible")
            return
        try:
            _, WPN, *_ = _get_world_classes()
            for key, inp in self._param_controls.items():
                try:
                    enum_key = WPN[key]
                    _, is_int, *_ = _PARAM_CONFIG[key]
                    raw = inp.text
                    self._world.parameters[enum_key] = int(float(raw)) if is_int else float(raw)
                except Exception:
                    pass
            self._set_status("Parámetros aplicados al mundo.")
            if self._param_status:
                self._param_status.text = "✓ Cambios aplicados"
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_param_cfg_change(self, idx: int, _text: str = "") -> None:
        self._active_param_cfg_idx = idx
        if self._param_cfg_name_input and idx < len(self._param_config_names):
            self._param_cfg_name_input.text = self._param_config_names[idx]
        # Update control values in-place (no rebuild needed)
        raw = self._get_active_param_values()
        for key, inp in self._param_controls.items():
            _, is_int, mn, *_ = _PARAM_CONFIG[key]
            val = raw.get(key, mn)
            inp.text = str(int(val) if is_int else val)

    def _save_param_config(self) -> None:
        if self._param_cfg_name_input is None:
            return
        save_name = self._param_cfg_name_input.text.strip()
        if not save_name:
            self._set_status("El nombre no puede estar vacío")
            return
        cfg = {}
        for key, inp in self._param_controls.items():
            _, is_int, *_ = _PARAM_CONFIG[key]
            try:
                cfg[key] = int(float(inp.text)) if is_int else float(inp.text)
            except ValueError:
                pass
        try:
            with open(CONFIG_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "parameters" not in data:
                data["parameters"] = {}
            is_new = save_name not in data["parameters"]
            data["parameters"][save_name] = cfg
            with open(CONFIG_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            self._param_configs[save_name] = cfg
            if is_new:
                self._param_config_names.append(save_name)
                self._active_param_cfg_idx = len(self._param_config_names) - 1
                # Rebuild params tab to reflect the new dropdown entry
                self._tab_vboxes[self.TAB_PARAMS] = None
                self._param_cfg_dropdown = None
                self._param_cfg_name_input = None
                if self._active_tab == self.TAB_PARAMS:
                    self._show_tab(self.TAB_PARAMS)
            self._set_status(f"{'Creada' if is_new else 'Actualizada'}: {save_name}")
            if self._param_status:
                self._param_status.text = f"✓ {'Creada' if is_new else 'Actualizada'}: {save_name}"
        except Exception as e:
            self._set_status(f"Error al guardar: {e}")
            print(f"[WorldEditor] _save_param_config: {e}")

    # ═══════════════════════ TAB: NOISES ══════════════════════════════════════

    def _build_noises_tab(self, cw: int) -> VBox:
        outer = VBox(width=cw, spacing=8, align=VBox.ALIGN_LEFT, auto_size=True, padding=0)

        if not self._noise_names:
            outer.add_child(_lbl("No hay noises en config.json", color=(200, 100, 100)))
            return outer

        outer.add_child(_title("── Visualización de Noises ──"))
        outer.add_child(_spacer())

        # Dropdown to select the active noise
        selector_row = HBox(width=cw, height=30, spacing=8, align='center', auto_size=False)
        sel_lbl = _lbl("Noise:", font_size=15)
        sel_lbl.width = 52
        selector_row.add_child(sel_lbl)
        self._noise_dropdown = _dropdown_small(
            self._noise_names, self._active_noise_idx, w=max(60, cw - 60),
        )
        self._noise_dropdown.on_change(self._on_noise_dropdown_change)
        selector_row.add_child(self._noise_dropdown)
        outer.add_child(selector_row)

        # Name row: editable key name + Randomize + save button
        name_row = HBox(width=cw, height=30, spacing=6, align='center', auto_size=False)
        name_lbl = _lbl("Nombre:", font_size=15)
        name_lbl.width = 68
        name_row.add_child(name_lbl)
        self._noise_name_input = _text_input(
            text=self._noise_names[self._active_noise_idx] if self._noise_names else "",
            w=max(60, cw - 230),
        )
        name_row.add_child(self._noise_name_input)
        btn_rand = _btn("Rand", (90, 60, 120), (110, 80, 145), w=52, h=26)
        btn_rand.on_click(lambda _: self._randomize_noise_seed())
        name_row.add_child(btn_rand)
        btn_save = _btn("Guardar", BTN_APL_BG, BTN_APL_HV, w=80, h=26)
        btn_save.on_click(lambda _: self._save_noise_to_config())
        name_row.add_child(btn_save)
        outer.add_child(name_row)

        # Inner scroll for editable noise fields (fills remaining space in the tab)
        inner_h = 420
        self._noise_inner_scroll = ScrollView(
            x=0, y=0, width=cw, height=inner_h,
            bg_color=(33, 33, 48, 220),
            scroll_speed=25,
            show_scrollbar=True,
            scrollbar_color=(95, 95, 128),
            scrollbar_track_color=(40, 40, 55),
            padding=SCROLL_PAD,
        )
        inner_cw = max(60, cw - 2 * SCROLL_PAD - SCROLLBAR_W)
        self._refresh_noise_view(self._active_noise_idx, inner_cw)
        outer.add_child(self._noise_inner_scroll)

        return outer

    def _on_noise_dropdown_change(self, idx: int, _text: str = "") -> None:
        self._active_noise_idx = idx
        if self._noise_name_input and idx < len(self._noise_names):
            self._noise_name_input.text = self._noise_names[idx]
        if self._noise_inner_scroll is None or not self._noise_names:
            return
        inner_cw = self._panel_content_w() - 2 * SCROLL_PAD - SCROLLBAR_W
        self._refresh_noise_view(idx, max(60, inner_cw))

    def _randomize_noise_seed(self) -> None:
        """Set a random seed on the active noise's seed control."""
        import random
        name = self._noise_names[self._active_noise_idx] if self._noise_names else None
        if name is None:
            return
        controls = self._noise_controls.get(name, {})
        seed_widget, _ = controls.get("seed", (None, None))
        if seed_widget is not None:
            seed_widget.text = str(random.randint(0, 9_999_999))

    def _refresh_noise_view(self, idx: int, inner_cw: int) -> None:
        """Swap _noise_inner_scroll content to the noise at idx, building lazily."""
        if self._noise_inner_scroll is None:
            return
        name = self._noise_names[idx] if idx < len(self._noise_names) else ""
        if name and name not in self._noise_tab_vboxes:
            self._noise_tab_vboxes[name] = self._build_noise_field_vbox(name, inner_cw)
        self._noise_inner_scroll.clear_children()
        if name in self._noise_tab_vboxes:
            self._noise_inner_scroll.add_child(self._noise_tab_vboxes[name])
        self._noise_inner_scroll.scroll_y = 0

    def _build_noise_field_vbox(self, name: str, inner_cw: int) -> VBox:
        """Build a VBox of editable widgets for one noise config, caching controls."""
        cfg = self._noise_configs.get(name, {})
        fields: Dict[str, tuple] = {}

        vbox = VBox(width=inner_cw, spacing=6, align=VBox.ALIGN_LEFT, auto_size=True, padding=0)
        vbox.add_child(_lbl(name, font_size=18, color=TITLE_COLOR))
        vbox.add_child(_spacer())

        for display_label, key, wtype, arg0, arg1, arg2 in _NOISE_FIELD_CONFIG:
            config_value = cfg.get(key)
            if config_value is None and wtype not in ('bool',):
                continue
            widget = _make_noise_widget(wtype, config_value, arg0, arg1, arg2)
            fields[key] = (widget, wtype)
            vbox.add_child(_row(display_label, widget, inner_cw))

        self._noise_controls[name] = fields
        return vbox

    def _preview_selected_noise(self) -> None:
        if not self._noise_names:
            self._set_status("No hay noises disponibles")
            return
        name = self._noise_names[self._active_noise_idx]
        # Build config from editable controls (if controls exist), else use raw config
        controls = self._noise_controls.get(name, {})
        cfg = dict(self._noise_configs.get(name, {}))  # start from saved config
        for key, (widget, wtype) in controls.items():
            cfg[key] = _read_noise_widget(widget, wtype)
        if not cfg:
            self._set_status(f"Noise '{name}' no encontrado")
            return
        pw, ph = self._get_preview_size()
        try:
            NoiseGenerator2D = _get_NoiseGenerator2D()
            Matrix2D = _get_Matrix2D()
            noise = NoiseGenerator2D.from_dict(cfg)
            matrix = Matrix2D.create_from_noise(noise, ph, pw)
            self._show_matrix(matrix, f"Noise: {name}", pw, ph)
            self._set_status(f"Previsualizado: {name}  ({ph}×{pw})")
        except Exception as e:
            self._set_status(f"Error: {e}")
            print(f"[WorldEditor] {e}")

    def _save_noise_to_config(self) -> None:
        """Save the currently edited noise to config.json under the given name."""
        if self._noise_name_input is None or not self._noise_names:
            self._set_status("No hay noise activo")
            return
        save_name = self._noise_name_input.text.strip()
        if not save_name:
            self._set_status("El nombre no puede estar vacío")
            return

        # Collect UI values for the currently selected noise
        current_name = self._noise_names[self._active_noise_idx] if self._active_noise_idx < len(self._noise_names) else ""
        cfg = dict(self._noise_configs.get(current_name, {}))
        for key, (widget, wtype) in self._noise_controls.get(current_name, {}).items():
            cfg[key] = _read_noise_widget(widget, wtype)

        try:
            with open(CONFIG_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if "noise" not in data:
                data["noise"] = {}
            data["noise"][save_name] = cfg
            with open(CONFIG_JSON, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            is_new = save_name not in self._noise_names
            self._noise_configs[save_name] = cfg
            if is_new:
                self._noise_names.append(save_name)
                self._active_noise_idx = len(self._noise_names) - 1
                # Rebuild noises tab to show the new sub-tab
                self._tab_vboxes[self.TAB_NOISES] = None
                self._noise_tab_vboxes = {}
                self._noise_controls = {}
                self._noise_dropdown = None
                self._noise_inner_scroll = None
                self._noise_name_input = None
                if self._active_tab == self.TAB_NOISES:
                    self._show_tab(self.TAB_NOISES)

            self._set_status(f"{'Creado' if is_new else 'Actualizado'}: {save_name}")
        except Exception as e:
            self._set_status(f"Error al guardar: {e}")
            print(f"[WorldEditor] _save_noise_to_config: {e}")

    # ═══════════════════════ TAB: MATRICES ════════════════════════════════════

    def _build_matrices_tab(self, cw: int) -> VBox:
        vbox = VBox(width=cw, spacing=8, align=VBox.ALIGN_LEFT, auto_size=True, padding=0)

        vbox.add_child(_title("── Matrices del Mundo ──"))
        vbox.add_child(_spacer())

        # Build the full matrix list from WorldMatrixName enum (auto-includes new entries).
        # _MATRIX_CONFIG provides optional metadata (noise key, binarize); unknown entries
        # fall back to (member.value, None, False).
        try:
            _, _, _, WMN = _get_world_classes()
            all_matrix_items = [
                (m.name, *_MATRIX_CONFIG.get(m.name, (m.value, None, False)))
                for m in WMN
            ]
        except Exception:
            # Fallback: use only the static config
            all_matrix_items = [
                (k, lbl, nk, bz) for k, (lbl, nk, bz) in _MATRIX_CONFIG.items()
            ]

        VER_W = 52
        NAME_W = max(60, cw - VER_W - 8)

        for mat_key, mat_label, _noise_key, _do_binarize in all_matrix_items:
            row = HBox(width=cw, height=30, spacing=8, align='center', auto_size=False)

            name_lbl = _lbl(mat_label)
            name_lbl.width = NAME_W
            row.add_child(name_lbl)

            btn_ver = _btn("Ver", BTN_PRV_BG, BTN_PRV_HV, w=VER_W, h=26)
            def _make_ver(mk=mat_key, ml=mat_label):
                return lambda _: self._view_world_matrix(mk, ml)
            btn_ver.on_click(_make_ver())
            row.add_child(btn_ver)

            vbox.add_child(row)

        return vbox

    def _view_world_matrix(self, mat_key: str, mat_label: str) -> None:
        """Render a world matrix (from self._world.matrix) in the tilemap viewer."""
        if self._world is None:
            self._set_status("No hay mundo generado")
            return
        try:
            _, _, _, WMN = _get_world_classes()
            enum_member = WMN[mat_key]
            matrix = self._world.matrix.get(enum_member)
            if matrix is None:
                self._set_status(f"Matriz '{mat_label}' no disponible")
                return
            h, w = matrix._data.shape
            self._show_matrix(matrix, f"Matriz: {mat_label}", w, h)
            self._set_status(f"Visualizando: {mat_label}  ({h}×{w})")
        except Exception as e:
            self._set_status(f"Error: {e}")
            print(f"[WorldEditor] _view_world_matrix: {e}")

    def _generate_matrix(self, mat_key: str, noise_key: str, binarize: bool) -> None:
        cfg = self._noise_configs.get(noise_key)
        if cfg is None:
            self._set_status(f"Noise '{noise_key}' no encontrado en config.json")
            return
        pw, ph = self._get_preview_size()
        try:
            NoiseGenerator2D = _get_NoiseGenerator2D()
            Matrix2D = _get_Matrix2D()
            noise = NoiseGenerator2D.from_dict(cfg)
            matrix = Matrix2D.create_from_noise(noise, ph, pw)
            if binarize and self._world:
                _, WPN, *_ = _get_world_classes()
                threshold = self._world.parameters.get(
                    WPN["island_threshold"], 0.5
                )
                matrix.binarize(threshold)
            elif binarize:
                matrix.binarize(0.5)
            self._matrix_generated[mat_key] = True
            self._generated_matrices[mat_key] = matrix
            btn_ver = self._matrix_view_buttons.get(mat_key)
            if btn_ver:
                btn_ver.enabled = True
            lbl = self._matrix_status.get(mat_key)
            if lbl:
                lbl.text = "✓ Listo"
                lbl.color = (100, 200, 100)
            mat_label = _MATRIX_CONFIG.get(mat_key, (mat_key,))[0]
            self._show_matrix(matrix, f"Matriz: {mat_label}", pw, ph)
            self._set_status(f"Generada: {mat_label}  ({ph}×{pw})")
        except Exception as e:
            self._set_status(f"Error: {e}")
            print(f"[WorldEditor] _generate_matrix: {e}")

    # ── Viewer helpers ────────────────────────────────────────────────────────

    def _get_preview_size(self):
        pw = ph = 256
        try:
            if self._preview_w:
                pw = max(16, min(2048, int(self._preview_w.text)))
            if self._preview_h:
                ph = max(16, min(2048, int(self._preview_h.text)))
        except ValueError:
            pass
        return pw, ph

    def _show_matrix(self, matrix, label: str, map_w: int, map_h: int) -> None:
        """Render a Matrix2D into the tilemap viewer."""
        self._current_matrix = matrix
        self._viewer_label = label
        self._map_w = map_w
        self._map_h = map_h

        # Tileset (shared, created once)
        if self._tileset is None:
            self._tileset = TileSet.generate_grayscale_tileset(
                nsteps=GRAYSCALE_STEPS,
                tile_size=(TILE_SIZE, TILE_SIZE),
                columns=GRAYSCALE_STEPS,
                white_to_black=True,
            )

        # Tilemap
        self._tilemap = TileMap(
            width=map_w, height=map_h,
            tile_size=(TILE_SIZE, TILE_SIZE),
        )
        self._tilemap.tileset = self._tileset

        tile_ids = np.clip(
            (matrix._data * (GRAYSCALE_STEPS - 1)).astype(int),
            0, GRAYSCALE_STEPS - 1,
        )
        for r in range(map_h):
            for c in range(map_w):
                self._tilemap.set_tile(c, r, int(tile_ids[r, c]))

        # Camera: create once, then preserve position/zoom
        screen = pygame.display.get_surface()
        sw, sh = screen.get_size()
        if self._camera is None:
            world_w = map_w * TILE_SIZE
            world_h = map_h * TILE_SIZE
            vp_w = sw - self._panel_width
            self._camera = Camera(
                x=max(0.0, (world_w - vp_w) / 2),
                y=max(0.0, (world_h - sh) / 2),
                width=sw, height=sh,
                zoom=1.0, min_zoom=0.05, max_zoom=20.0,
            )

    def _set_status(self, text: str) -> None:
        if self._status_bar:
            self._status_bar.text = text

    # ── Events ────────────────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.running = False
            return

        # Divider drag
        screen = pygame.display.get_surface()
        sw = screen.get_width()
        divx = self._panel_width

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if abs(event.pos[0] - divx) <= _DIVIDER_HIT_W // 2:
                self._dragging_divider = True
                self._divider_start_x = event.pos[0]
                self._divider_start_panel_w = self._panel_width
                return

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._dragging_divider:
                self._dragging_divider = False
                self._rebuild_panel()
                return

        elif event.type == pygame.MOUSEMOTION:
            if self._dragging_divider:
                dx = event.pos[0] - self._divider_start_x
                nw = max(_PANEL_MIN_W, min(sw - 200, self._divider_start_panel_w + dx))
                self._panel_width = nw
                if self._scroll_view:
                    self._scroll_view.width = nw
                if self._main_tab_bar:
                    self._main_tab_bar.width = nw
                return

        if self._ui:
            self._ui.handle_event(event)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        if self._ui:
            self._ui.update(dt)

        # Cursor
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
        world_w = self._map_w * TILE_SIZE
        world_h = self._map_h * TILE_SIZE
        vp_w = self._camera.width / zoom
        vp_h = self._camera.height / zoom
        self._camera.set_bounds(
            min_x=0, max_x=max(0.0, world_w - vp_w),
            min_y=0, max_y=max(0.0, world_h - vp_h),
        )

        speed = CAMERA_SPEED / zoom
        dx = dy = 0.0
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]: dx -= speed * dt
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: dx += speed * dt
        if keys[pygame.K_UP]    or keys[pygame.K_w]: dy -= speed * dt
        if keys[pygame.K_DOWN]  or keys[pygame.K_s]: dy += speed * dt
        if dx or dy:
            self._camera.move(dx, dy)

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, screen: pygame.Surface) -> None:
        screen.fill((18, 18, 28))
        sw, sh = screen.get_size()

        # Tilemap viewer (right of panel)
        if self._tilemap and self._camera and self._tileset:
            clip = pygame.Rect(self._panel_width, 0, sw - self._panel_width, sh)
            old_clip = screen.get_clip()
            screen.set_clip(clip)
            sub = screen.subsurface(clip)
            self._tilemap.draw(sub, self._camera, self._tileset)
            screen.set_clip(old_clip)
            self._draw_hud(screen)
        else:
            # Placeholder when nothing is generated
            if sw > self._panel_width + 40:
                font = pygame.font.Font(None, 22)
                txt = font.render(
                    "Previsualiza un noise o genera una matriz →",
                    True, (70, 70, 95),
                )
                screen.blit(txt, (self._panel_width + 20, sh // 2 - 10))

        # UI panel (on top)
        if self._ui:
            self._ui.draw(screen)

        # Divider
        div_color = (145, 145, 195) if self._dragging_divider else (65, 65, 88)
        pygame.draw.rect(
            screen, div_color,
            pygame.Rect(self._panel_width - _DIVIDER_W // 2, 0, _DIVIDER_W, sh),
        )

    def _draw_hud(self, screen: pygame.Surface) -> None:
        font = pygame.font.Font(None, 18)
        cam = self._camera
        info = (
            f"{self._viewer_label}  |  "
            f"Map {self._map_h}×{self._map_w}  "
            f"Cam ({int(cam.x)},{int(cam.y)})  "
            f"Zoom {cam.zoom:.2f}×"
        )
        surf = font.render(info, True, (195, 195, 195))
        screen.blit(surf, (self._panel_width + 8, 6))

        # Cell value under cursor
        mx, my = pygame.mouse.get_pos()
        if mx > self._panel_width and self._current_matrix is not None:
            zoom = cam.zoom
            world_x = (mx - self._panel_width) / zoom + cam.x
            world_y = my / zoom + cam.y
            tile_x = int(world_x / TILE_SIZE)
            tile_y = int(world_y / TILE_SIZE)
            if 0 <= tile_x < self._map_w and 0 <= tile_y < self._map_h:
                value = self._current_matrix._data[tile_y, tile_x]
                cell_text = f"Celda ({tile_x},{tile_y}) = {value:.4f}"
                cell_surf = font.render(cell_text, True, (255, 220, 100))
                screen.blit(cell_surf, (self._panel_width + 8, 24))

        hint = "WASD/Flechas: mover | Q/E: zoom | ESC: salir"
        h_surf = font.render(hint, True, (110, 110, 130))
        screen.blit(h_surf, (self._panel_width + 8, screen.get_height() - 20))

    # ── Panel rebuild (after divider drag) ───────────────────────────────────

    def _rebuild_panel(self) -> None:
        # Persist preview size across rebuild
        self._saved_pw, self._saved_ph = self._get_preview_size()
        # Invalidate cached tab vboxes (width changed)
        self._tab_vboxes = [None, None, None]
        self._noise_tab_vboxes = {}
        self._noise_controls = {}
        self._noise_dropdown = None
        self._noise_inner_scroll = None
        self._noise_name_input = None
        self._param_cfg_dropdown = None
        self._param_cfg_name_input = None
        screen = pygame.display.get_surface()
        sw, sh = screen.get_size()
        self._ui.clear()
        self._build_ui(sw, sh)


# ── Utility ───────────────────────────────────────────────────────────────────

def _pascal_to_snake(name: str) -> str:
    """Convert 'PascalCase' → 'snake_case'."""
    import re
    s1 = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
