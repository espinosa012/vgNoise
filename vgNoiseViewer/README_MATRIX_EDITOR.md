# VGMatrix2D Editor

A visual matrix editor and filter application for manipulating VGMatrix2D matrices.

## Features

### Matrix Creation & Manipulation
- Create matrices of various sizes (8x8 to 1024x1024)
- Set custom default values or create empty matrices (None values)
- Fill with random values or gradient patterns
- **Generate from Noise**: Create matrices using procedural noise algorithms
- Edit individual cell values by clicking on the image or entering coordinates
- Resize existing matrices

### Noise Generator
Access via Menu: **Matrix → Generate from Noise...** or the "🌊 Generate from Noise..." button.

Features:
- All noise types: Perlin, Simplex, Value, Cellular, etc.
- Full fractal controls: octaves, lacunarity, persistence
- Cellular noise options: distance functions, return types
- Real-time preview as you adjust parameters
- Randomize seed or all parameters

### File Operations (Save/Load)
- **Open Matrix** (Ctrl+O): Load a previously saved matrix
- **Save Matrix** (Ctrl+S): Save the current matrix
- **Save Matrix As** (Ctrl+Shift+S): Save with a new filename

Supported formats:
- **VGMatrix Binary (.vgm)**: Efficient compressed binary format (recommended)
- **JSON (.json)**: Human-readable format, suitable for version control
- **NumPy (.npy)**: Compatible with NumPy, but loses None/unassigned information

### Image Import/Export
- **Import images**: Load PNG or JPEG images and convert them to grayscale matrices
  - Option to resize to match current matrix dimensions
  - Automatic grayscale conversion
- **Export images**: Save matrices as PNG or JPEG
  - Supports transparency visualization for None values
- **Export NumPy**: Save matrix data as .npy files

### Filtering
The application automatically discovers all filters available in `MatrixFilters` class:

#### Blur Filters
- Box Blur
- Gaussian Blur
- Motion Blur (horizontal, vertical, diagonal)

#### Sharpen Filters
- Sharpen
- Unsharp Mask

#### Edge Detection
- Sobel (horizontal, vertical)
- Prewitt (horizontal, vertical)
- Laplacian (4-connectivity, 8-connectivity)
- Ridge Detection

#### Effects
- Emboss (8 directions)
- High-pass Filter
- Low-pass Filter

#### Quick Filter Bar
Common filters are available as one-click buttons for rapid prototyping.

### Display Options
- **Normalize**: Automatically scales values for display
- **Show Transparency**: Displays None values as a checkerboard pattern

### Image Viewer with Zoom & Pan
- **Mouse wheel**: Zoom in/out (centered on cursor position)
- **Middle mouse button drag**: Pan the image
- **Fit button**: Fit image to visible area
- **100% button**: Reset to original size (100% zoom)
- **Click on image**: Select pixel for value inspection
- **Hover**: Real-time position and value display in status bar

### Undo/Redo
Full undo/redo support with up to 50 states (Ctrl+Z / Ctrl+Y).

## Usage

### Running the Application

```bash
cd vgNoiseViewer
python3 matrix_app.py
```

Or from the project root:
```bash
python3 -m vgNoiseViewer.matrix_app
```

### Basic Workflow

1. **Create a matrix**: Choose dimensions and default value, click "Create New Matrix"
2. **Edit values**: Click on the image to select a pixel, or enter coordinates manually
3. **Apply filters**: Select a filter category and filter, adjust parameters, click "Apply Filter"
4. **Export**: Save as image (PNG/JPEG) or NumPy array

### Keyboard Shortcuts

- `Ctrl+Z`: Undo
- `Ctrl+Y`: Redo

## Module Structure

```
vgNoiseViewer/
├── matrix_app.py           # Main application
├── matrix_app_config.py    # Configuration and constants
├── matrix_image_utils.py   # Image conversion utilities
├── matrix_widgets.py       # Custom UI widgets
├── matrix_filter_panel.py  # Dynamic filter panel
└── README_MATRIX_EDITOR.md # This file
```

## Adding New Filters

The filter panel automatically discovers new filters from `MatrixFilters` class. To add a new filter:

1. Add a static method to `MatrixFilters` in `src/vgmath/matrix/filters.py`
2. Use standard parameter names (`size`, `sigma`, `strength`, `direction`, etc.)
3. The filter will appear in the UI automatically on next launch

Example:
```python
@staticmethod
def my_custom_filter(size: int = 3, strength: float = 1.0) -> NDArray[np.float64]:
    """My custom filter description."""
    # Create and return kernel
    kernel = np.zeros((size, size), dtype=np.float64)
    # ... filter implementation ...
    return kernel
```

## Technical Details

### Transparency Handling
- None values in VGMatrix2D are displayed as transparent
- A checkerboard pattern is rendered behind transparent areas for visibility
- When exporting to PNG, transparency is preserved
- When exporting to JPEG (no alpha support), None areas appear as black

### Value Range
- Matrix values are typically in [0, 1] range
- Normalization option scales display to full range
- Clipping function constrains values to [0, 1]
- Filters may produce values outside [0, 1]; use normalize/clip as needed

## Serialization API

VGMatrix2D provides multiple serialization methods for different use cases:

### Binary Format (Most Efficient)
```python
# Save to bytes (compressed by default)
data = matrix.to_bytes(compressed=True)

# Load from bytes
restored = VGMatrix2D.from_bytes(data)
```

### JSON Format (Human Readable)
```python
# Save to JSON string
json_str = matrix.to_json(indent=2)

# Load from JSON string
restored = VGMatrix2D.from_json(json_str)

# Dictionary format for custom serialization
d = matrix.to_dict()
restored = VGMatrix2D.from_dict(d)
```

### File Operations
```python
# Save to file (format auto-detected from extension)
matrix.save("mymatrix.vgm")   # Binary format
matrix.save("mymatrix.json")  # JSON format
matrix.save("mymatrix.npy")   # NumPy format

# Load from file
loaded = VGMatrix2D.load("mymatrix.vgm")

# Force specific format
matrix.save("data.bin", format="binary")
loaded = VGMatrix2D.load("data.bin", format="binary")
```

### Pickle Support
```python
import pickle

# Serialize with pickle
pickled = pickle.dumps(matrix)
restored = pickle.loads(pickled)
```

### Format Comparison

| Format | Size | Speed | Preserves Mask | Human Readable |
|--------|------|-------|----------------|----------------|
| Binary (.vgm) | Smallest | Fastest | ✓ | ✗ |
| JSON (.json) | Larger | Medium | ✓ | ✓ |
| NumPy (.npy) | Medium | Fast | ✗ (uses NaN) | ✗ |
| Pickle | Large | Fast | ✓ | ✗ |

