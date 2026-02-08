# vgNoiseViewer

Visual tools for noise generation and matrix manipulation using vgMath.

## Structure

```
vgNoiseViewer/
├── core/                    # Shared configuration and utilities
│   ├── config.py           # Theme colors, window config
│   ├── theme.py            # Theme management
│   └── image_viewer.py     # Zoomable image viewer widget
├── widgets/                 # Reusable UI components
│   ├── common.py           # StepperControl, LabeledCombobox, etc.
│   └── matrix_widgets.py   # Card, StatusBar, FilterParameterWidget
├── noise_viewer/           # Noise visualization application
│   ├── app.py              # Main NoiseViewer class
│   ├── factory.py          # Noise generator factory
│   └── image_utils.py      # Noise rendering utilities
├── matrix_editor/          # Matrix editing application
│   ├── app.py              # Main MatrixEditor class
│   ├── config.py           # Matrix-specific configuration
│   ├── filter_panel.py     # Filter discovery and UI
│   ├── image_utils.py      # Matrix rendering utilities
│   └── noise_dialog.py     # Noise generation dialog
├── tests/                  # Unit tests
│   ├── test_noise_viewer.py
│   └── test_matrix_editor.py
├── presets/                # Preset files
│   └── noise_preset.noise.json
├── run_noise_viewer.py     # Launch Noise Viewer
└── run_matrix_editor.py    # Launch Matrix Editor
```

## Running the Applications

### Noise Viewer
```bash
cd vgNoiseViewer
python run_noise_viewer.py
```

### Matrix Editor
```bash
cd vgNoiseViewer
python run_matrix_editor.py
```

## Features

- Real-time noise visualization
- Support for Perlin and OpenSimplex noise algorithms
- All Godot FastNoiseLite fractal types (FBM, Ridged, Ping-Pong)
- Dark theme UI with custom stepper controls
- High-performance Numba JIT acceleration

## Architecture

The viewer is built with a modular architecture:

```
vgNoiseViewer/
├── app.py              # Main application class (NoiseViewer)
├── config.py           # Configuration dataclasses and constants
├── theme.py            # ThemeManager for UI styling
├── widgets.py          # Reusable UI components (StepperControl, Card, etc.)
├── noise_factory.py    # NoiseGeneratorFactory and NoiseParameters
├── image_utils.py      # ImageGenerator and NoiseImageRenderer
├── test_app.py         # Comprehensive test suite (67 tests)
└── __init__.py         # Package exports
```

### Key Components

- **NoiseViewer**: Main application class managing UI and coordination
- **ThemeManager**: Handles dark theme styling for ttk widgets
- **NoiseGeneratorFactory**: Factory pattern for creating noise generators
- **NoiseParameters**: Data container for noise generation parameters
- **NoiseImageRenderer**: Converts noise data to displayable images
- **StepperControl**: Custom widget with +/- buttons for numeric input
- **Card**: Styled container widget for parameter groups

## Installation

```bash
pip install -r requirements.txt
```

Note: Tkinter comes pre-installed with Python on most systems.

## Usage

```bash
python app.py
```

## Running Tests

```bash
python -m pytest test_app.py -v
```

## Parameters

### Basic Parameters
- **Seed**: Random seed for reproducible results
- **Noise Type**: PERLIN or SIMPLEX algorithm
- **Frequency**: Base frequency (detail level)
- **Offset X/Y**: Domain offset for panning

### Fractal Parameters
- **Fractal Type**: NONE, FBM, RIDGED, or PING_PONG
- **Octaves**: Number of noise layers (1-9)
- **Lacunarity**: Frequency multiplier per octave
- **Persistence**: Amplitude multiplier per octave
- **Weighted Strength**: Octave weighting based on previous value
- **Ping Pong Strength**: Strength of ping-pong effect

### Image Settings
- **Image Size**: 128, 256, 512, or 1024 pixels

## Requirements

- Python 3.10+
- tkinter
- PIL/Pillow
- numpy
- numba
- vgmath (parent package)
