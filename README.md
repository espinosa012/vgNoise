# vgMath

A math library for Python including procedural noise generation, matrix operations, and more.

## Features

- **Multiple noise algorithms**: Perlin, OpenSimplex, Simplex Smooth, Cellular (Worley/Voronoi), Value, Value Cubic
- **Fractal noise support**: FBM, Ridged, Ping-Pong with configurable octaves
- **Matrix operations**: VGMatrix2D with convolution, filters, serialization
- **High performance**: Numba JIT compilation for parallel processing
- **Godot compatible**: Noise parameter names and behavior match FastNoiseLite
- **Easy to use**: Simple API for 2D noise generation and matrix manipulation

## Installation

```bash
pip install vgmath
```

Or install from source:

```bash
git clone https://github.com/virigir/vgNoise.git
cd vgNoise
pip install -e .
```

## Quick Start

```python
from vgmath import PerlinNoise2D, FractalType

# Create a noise generator
noise = PerlinNoise2D(
    frequency=0.01,
    fractal_type=FractalType.FBM,
    octaves=5,
    seed=42
)

# Generate a 512x512 noise map
region = noise.generate_region([(0, 512, 512), (0, 512, 512)])

# Get a single value at a specific position
value = noise.get_value_at((100.5, 200.3))
```

## Noise Algorithms

### Perlin Noise (`PerlinNoise2D`)

**Classic gradient noise algorithm** invented by Ken Perlin in 1983.

#### How it works:
1. **Grid setup**: The input coordinates are mapped to a regular integer grid
2. **Gradient assignment**: Each grid vertex is assigned a pseudo-random gradient vector using a permutation table
3. **Distance vectors**: For each sample point, calculate vectors from each surrounding grid vertex to the sample point
4. **Dot products**: Compute the dot product between each gradient vector and its corresponding distance vector
5. **Interpolation**: Blend the four corner values using a smooth quintic interpolation function (6t⁵ - 15t⁴ + 10t³)

#### Characteristics:
- Produces smooth, natural-looking noise
- Zero values at integer lattice points
- Continuous first derivative (smooth transitions)
- Good for terrain, clouds, and organic textures

#### Parameters:
- `frequency`: Controls the scale/zoom of the noise pattern
- `octaves`: Number of noise layers to combine (1-9)
- `lacunarity`: Frequency multiplier between octaves (default: 2.0)
- `persistence`: Amplitude multiplier between octaves (default: 0.5)

---

### OpenSimplex Noise (`OpenSimplexNoise2D`)

**Patent-free alternative to Simplex noise** that produces visually similar results without the legal concerns.

#### How it works:
1. **Coordinate skewing**: Input coordinates are transformed using a skew matrix to map the square grid to a simplex (triangular) grid
2. **Simplex identification**: Determine which simplex (triangle in 2D) contains the input point
3. **Vertex contributions**: For each vertex of the simplex, calculate:
   - The unskewed position of the vertex
   - The distance from the input point to the vertex
   - A radial attenuation factor: `(2 - dx² - dy²)⁴` (falls off smoothly to zero)
4. **Gradient evaluation**: At each contributing vertex, compute the dot product of a pseudo-random gradient with the distance vector
5. **Summation**: Sum all vertex contributions, weighted by their attenuation factors

#### Characteristics:
- No visible axis-aligned artifacts (unlike Perlin)
- Computationally efficient (fewer vertices to evaluate than Perlin in higher dimensions)
- Smooth, isotropic appearance
- Better suited for flowing, organic patterns

#### Parameters:
Same as Perlin, plus:
- `weighted_strength`: Controls how previous octave values affect subsequent octave amplitudes
- `ping_pong_strength`: Intensity of the ping-pong fractal effect

---

### Cellular Noise (`CellularNoise2D`)

**Also known as Worley noise or Voronoi noise**, creates cell-like patterns based on distances to feature points.

#### How it works:
1. **Cell grid**: The space is divided into a regular grid of cells
2. **Feature points**: Each cell contains one randomly-placed feature point (controlled by jitter parameter)
3. **Neighborhood search**: For each sample point, search the 3x3 neighborhood of cells to find the closest feature points
4. **Distance calculation**: Compute distances using the selected distance function:
   - **Euclidean**: `√(dx² + dy²)` - circular cells
   - **Euclidean Squared**: `dx² + dy²` - faster, same shape
   - **Manhattan**: `|dx| + |dy|` - diamond-shaped cells
   - **Hybrid**: `|dx| + |dy| + (dx² + dy²)` - rounded diamonds
5. **Return value**: Based on the selected return type:
   - **DISTANCE**: Distance to closest point (creates cell interiors)
   - **DISTANCE_2**: Distance to second-closest point
   - **DISTANCE_2_SUB**: `distance2 - distance1` (creates cell edges/Voronoi borders)
   - **DISTANCE_2_ADD**: `(distance1 + distance2) / 2`
   - **DISTANCE_2_MUL**: `distance1 * distance2`
   - **DISTANCE_2_DIV**: `distance1 / distance2`
   - **CELL_VALUE**: Random value based on closest cell (flat colored regions)

#### Characteristics:
- Creates organic cell-like patterns
- Excellent for stone textures, biological cells, cracked surfaces
- DISTANCE_2_SUB produces clean Voronoi edges
- CELL_VALUE creates a mosaic/stained glass effect

#### Parameters:
- `distance_function`: The metric used to calculate distances
- `return_type`: What value to return (affects the visual pattern)
- `jitter`: Randomness of feature point placement (0.0 = grid centers, 1.0 = fully random within cell)

---

### Value Cubic Noise (`ValueCubicNoise2D`)

**Smooth value noise using cubic interpolation** for higher quality results than standard value noise.

#### How it works:
1. **Lattice values**: Each integer grid point is assigned a pseudo-random value in the range [-1, 1] using a hash function
2. **4x4 sampling**: For each sample point, gather values from a 4x4 grid of surrounding lattice points
3. **Catmull-Rom interpolation**: Apply cubic interpolation in both X and Y directions using the Catmull-Rom spline formula:
   ```
   p = (d - c) - (a - b)
   q = (a - b) - p
   r = c - a  
   s = b
   result = p*t³ + q*t² + r*t + s
   ```
   Where a, b, c, d are four consecutive values and t is the fractional position [0, 1]
4. **Bicubic blending**: First interpolate the four rows along X, then interpolate the four results along Y

#### Characteristics:
- Smoother than standard value noise (which uses bilinear interpolation)
- No gradient discontinuities
- Slightly softer appearance than Perlin noise
- Good for smooth, rolling terrain

#### Parameters:
Same as Perlin noise (frequency, octaves, lacunarity, persistence, etc.)

---

### Value Noise (`ValueNoise2D`)

**Simple lattice-based noise using bilinear interpolation**, the fastest noise algorithm available.

#### How it works:
1. **Lattice values**: Each integer grid point is assigned a pseudo-random value in range [-1, 1] using a hash function
2. **2x2 sampling**: For each sample point, get values from the four surrounding lattice points
3. **Quintic fade**: Apply smooth fade function to fractional coordinates: `6t⁵ - 15t⁴ + 10t³`
4. **Bilinear interpolation**: Blend the four corner values using the faded coordinates

#### Characteristics:
- Fastest noise algorithm (only 4 hash lookups per sample)
- Slightly blocky appearance compared to gradient noise
- Good for simple textures where speed is priority
- Can show grid alignment artifacts at low frequencies

#### Parameters:
Same as Perlin noise (frequency, octaves, lacunarity, persistence, etc.)

---

### Simplex Smooth Noise (`SimplexSmoothNoise2D`)

**Enhanced simplex noise with smoother falloff** for higher quality results with minimal artifacts.

#### How it works:
1. **Coordinate skewing**: Transform coordinates to simplex (triangular) grid space using skew factor `(√3 - 1) / 2`
2. **Simplex selection**: Determine which triangle contains the sample point based on relative position
3. **Vertex contributions**: For each of the 3 simplex vertices:
   - Calculate distance vector from vertex to sample point
   - Apply **extended falloff radius** of 0.6 (vs 0.5 in standard simplex) for smoother blending
   - Compute `(0.6 - dx² - dy²)⁴` attenuation (smoother than standard)
4. **Gradient evaluation**: Use 12-direction gradient table for better isotropy
5. **Summation**: Sum all vertex contributions with their attenuation weights

#### Characteristics:
- Smoother than standard OpenSimplex due to larger falloff radius
- Better isotropy with 12 gradient directions (vs 8 in standard simplex)
- No visible grid artifacts
- Ideal for organic, flowing patterns

#### Parameters:
Same as Perlin noise (frequency, octaves, lacunarity, persistence, etc.)

---

## Fractal Types

All noise generators support fractal layering to add detail at multiple scales:

### NONE
Single octave, no fractal combination. Returns raw noise values.

### FBM (Fractal Brownian Motion)
The classic fractal noise combination:
```
result = Σ (noise(frequency * lacunarity^i) * persistence^i)
```
Each octave adds finer detail at reduced amplitude.

### RIDGED
Creates sharp ridges by taking the absolute value and inverting:
```
noise = 1.0 - |raw_noise|
```
Excellent for mountain ridges, lightning, veins.

### PING_PONG
Creates terraced/banded effects by folding the noise values:
```
value = ping_pong(raw_noise * strength)
```
Useful for topographic map effects, stylized terrain.

## Performance

All noise generation uses Numba JIT compilation with parallel processing. Typical performance on a modern CPU:

| Noise Type     | 512x512 (5 octaves) |
|----------------|---------------------|
| Value          | ~11 ms              |
| Perlin         | ~11 ms              |
| OpenSimplex    | ~16 ms              |
| Simplex Smooth | ~23 ms              |
| Value Cubic    | ~25 ms              |
| Cellular       | ~32 ms              |

*Note: First call includes JIT compilation overhead (~1-2 seconds)*

## Author

Adrián R. Espinosa

## License

MIT
