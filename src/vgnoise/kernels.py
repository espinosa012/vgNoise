"""
Numba JIT-compiled noise generation kernels.

This module contains optimized noise generation functions using Numba
for maximum performance through JIT compilation and parallelization.
"""

import numpy as np
from numba import njit, prange
from numpy.typing import NDArray


@njit(fastmath=True)
def _fade(t: float) -> float:
    """Smoothstep fade function (quintic)."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@njit(parallel=True, fastmath=True, cache=True)
def perlin_fbm_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    perm: NDArray[np.int32],
    grads: NDArray[np.float64],
    octaves: int,
    lacunarity: float,
    persistence: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """
    Generate 2D Perlin FBM noise using parallel JIT compilation.

    Args:
        x: Flattened array of X coordinates (with frequency applied).
        y: Flattened array of Y coordinates (with frequency applied).
        perm: Permutation table (512 elements).
        grads: Gradient vectors (8x2).
        octaves: Number of octaves.
        lacunarity: Frequency multiplier per octave.
        persistence: Amplitude multiplier per octave.
        fractal_bounding: Normalization factor.

    Returns:
        Array of noise values normalized to [0, 1].
    """
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for _ in range(octaves):
            # Get integer and fractional parts
            floor_x = np.floor(xi)
            floor_y = np.floor(yi)
            xf = xi - floor_x
            yf = yi - floor_y

            ix = int(floor_x) & 255
            iy = int(floor_y) & 255
            ix1 = (ix + 1) & 255
            iy1 = (iy + 1) & 255

            # Fade curves
            u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
            v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

            # Hash coordinates
            h00 = perm[perm[ix] + iy] & 7
            h10 = perm[perm[ix1] + iy] & 7
            h01 = perm[perm[ix] + iy1] & 7
            h11 = perm[perm[ix1] + iy1] & 7

            # Gradient dot products
            n00 = grads[h00, 0] * xf + grads[h00, 1] * yf
            n10 = grads[h10, 0] * (xf - 1.0) + grads[h10, 1] * yf
            n01 = grads[h01, 0] * xf + grads[h01, 1] * (yf - 1.0)
            n11 = grads[h11, 0] * (xf - 1.0) + grads[h11, 1] * (yf - 1.0)

            # Bilinear interpolation
            nx0 = n00 + u * (n10 - n00)
            nx1 = n01 + u * (n11 - n01)
            noise = nx0 + v * (nx1 - nx0)

            total += noise * amp
            amp *= persistence
            xi *= lacunarity
            yi *= lacunarity

        result[i] = total * fractal_bounding * 0.5 + 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def perlin_fbm_2d_weighted(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    perm: NDArray[np.int32],
    grads: NDArray[np.float64],
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """
    Generate 2D Perlin FBM noise with weighted strength.
    """
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for _ in range(octaves):
            floor_x = np.floor(xi)
            floor_y = np.floor(yi)
            xf = xi - floor_x
            yf = yi - floor_y

            ix = int(floor_x) & 255
            iy = int(floor_y) & 255
            ix1 = (ix + 1) & 255
            iy1 = (iy + 1) & 255

            u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
            v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

            h00 = perm[perm[ix] + iy] & 7
            h10 = perm[perm[ix1] + iy] & 7
            h01 = perm[perm[ix] + iy1] & 7
            h11 = perm[perm[ix1] + iy1] & 7

            n00 = grads[h00, 0] * xf + grads[h00, 1] * yf
            n10 = grads[h10, 0] * (xf - 1.0) + grads[h10, 1] * yf
            n01 = grads[h01, 0] * xf + grads[h01, 1] * (yf - 1.0)
            n11 = grads[h11, 0] * (xf - 1.0) + grads[h11, 1] * (yf - 1.0)

            nx0 = n00 + u * (n10 - n00)
            nx1 = n01 + u * (n11 - n01)
            noise = nx0 + v * (nx1 - nx0)

            total += noise * amp
            # Apply weighted strength
            amp *= (1.0 - weighted_strength + weighted_strength * (noise + 1.0) * 0.5) * persistence
            xi *= lacunarity
            yi *= lacunarity

        result[i] = total * fractal_bounding * 0.5 + 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def perlin_ridged_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    perm: NDArray[np.int32],
    grads: NDArray[np.float64],
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """
    Generate 2D ridged multifractal noise.
    """
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for _ in range(octaves):
            floor_x = np.floor(xi)
            floor_y = np.floor(yi)
            xf = xi - floor_x
            yf = yi - floor_y

            ix = int(floor_x) & 255
            iy = int(floor_y) & 255
            ix1 = (ix + 1) & 255
            iy1 = (iy + 1) & 255

            u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
            v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

            h00 = perm[perm[ix] + iy] & 7
            h10 = perm[perm[ix1] + iy] & 7
            h01 = perm[perm[ix] + iy1] & 7
            h11 = perm[perm[ix1] + iy1] & 7

            n00 = grads[h00, 0] * xf + grads[h00, 1] * yf
            n10 = grads[h10, 0] * (xf - 1.0) + grads[h10, 1] * yf
            n01 = grads[h01, 0] * xf + grads[h01, 1] * (yf - 1.0)
            n11 = grads[h11, 0] * (xf - 1.0) + grads[h11, 1] * (yf - 1.0)

            nx0 = n00 + u * (n10 - n00)
            nx1 = n01 + u * (n11 - n01)
            noise_raw = nx0 + v * (nx1 - nx0)

            # Ridged: abs and invert
            noise = 1.0 - abs(noise_raw)
            total += noise * amp

            # Apply weighted strength
            if weighted_strength > 0:
                amp *= (1.0 - weighted_strength + weighted_strength * noise) * persistence
            else:
                amp *= persistence

            xi *= lacunarity
            yi *= lacunarity

        result[i] = total * fractal_bounding

    return result


@njit(fastmath=True)
def _ping_pong(t: float) -> float:
    """Ping-pong function for terraced effects."""
    t = t - int(t * 0.5) * 2
    if t < 1.0:
        return t
    return 2.0 - t


@njit(parallel=True, fastmath=True, cache=True)
def perlin_pingpong_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    perm: NDArray[np.int32],
    grads: NDArray[np.float64],
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    ping_pong_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """
    Generate 2D ping-pong fractal noise.
    """
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for _ in range(octaves):
            floor_x = np.floor(xi)
            floor_y = np.floor(yi)
            xf = xi - floor_x
            yf = yi - floor_y

            ix = int(floor_x) & 255
            iy = int(floor_y) & 255
            ix1 = (ix + 1) & 255
            iy1 = (iy + 1) & 255

            u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
            v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

            h00 = perm[perm[ix] + iy] & 7
            h10 = perm[perm[ix1] + iy] & 7
            h01 = perm[perm[ix] + iy1] & 7
            h11 = perm[perm[ix1] + iy1] & 7

            n00 = grads[h00, 0] * xf + grads[h00, 1] * yf
            n10 = grads[h10, 0] * (xf - 1.0) + grads[h10, 1] * yf
            n01 = grads[h01, 0] * xf + grads[h01, 1] * (yf - 1.0)
            n11 = grads[h11, 0] * (xf - 1.0) + grads[h11, 1] * (yf - 1.0)

            nx0 = n00 + u * (n10 - n00)
            nx1 = n01 + u * (n11 - n01)
            noise_raw = nx0 + v * (nx1 - nx0)

            # Ping-pong effect
            pp_val = (noise_raw + 1.0) * ping_pong_strength
            pp_val = pp_val - int(pp_val * 0.5) * 2
            if pp_val >= 1.0:
                pp_val = 2.0 - pp_val
            noise = pp_val

            total += (noise - 0.5) * 2.0 * amp

            # Apply weighted strength
            if weighted_strength > 0:
                amp *= (1.0 - weighted_strength + weighted_strength * noise) * persistence
            else:
                amp *= persistence

            xi *= lacunarity
            yi *= lacunarity

        result[i] = total * fractal_bounding * 0.5 + 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def perlin_single_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    perm: NDArray[np.int32],
    grads: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
    Generate single octave 2D Perlin noise (no fractal).
    """
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]

        floor_x = np.floor(xi)
        floor_y = np.floor(yi)
        xf = xi - floor_x
        yf = yi - floor_y

        ix = int(floor_x) & 255
        iy = int(floor_y) & 255
        ix1 = (ix + 1) & 255
        iy1 = (iy + 1) & 255

        u = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
        v = yf * yf * yf * (yf * (yf * 6.0 - 15.0) + 10.0)

        h00 = perm[perm[ix] + iy] & 7
        h10 = perm[perm[ix1] + iy] & 7
        h01 = perm[perm[ix] + iy1] & 7
        h11 = perm[perm[ix1] + iy1] & 7

        n00 = grads[h00, 0] * xf + grads[h00, 1] * yf
        n10 = grads[h10, 0] * (xf - 1.0) + grads[h10, 1] * yf
        n01 = grads[h01, 0] * xf + grads[h01, 1] * (yf - 1.0)
        n11 = grads[h11, 0] * (xf - 1.0) + grads[h11, 1] * (yf - 1.0)

        nx0 = n00 + u * (n10 - n00)
        nx1 = n01 + u * (n11 - n01)
        noise = nx0 + v * (nx1 - nx0)

        result[i] = (noise + 1.0) * 0.5

    return result


# =============================================================================
# OpenSimplex 2D Kernels
# =============================================================================

# OpenSimplex 2D constants
STRETCH_2D = -0.211324865405187  # (1 / sqrt(2 + 1) - 1) / 2
SQUISH_2D = 0.366025403784439    # (sqrt(2 + 1) - 1) / 2
NORM_2D = 47.0


@njit(fastmath=True, cache=True)
def _opensimplex_extrapolate(
    perm: NDArray[np.int16],
    perm_grad_index: NDArray[np.int16],
    gradients: NDArray[np.float64],
    xsb: int,
    ysb: int,
    dx: float,
    dy: float
) -> float:
    """Calculate gradient contribution for a single point."""
    index = perm_grad_index[(perm[xsb & 255] + ysb) & 255]
    return gradients[index] * dx + gradients[index + 1] * dy


@njit(fastmath=True, cache=True)
def _opensimplex_sample(
    perm: NDArray[np.int16],
    perm_grad_index: NDArray[np.int16],
    gradients: NDArray[np.float64],
    x: float,
    y: float
) -> float:
    """Sample single OpenSimplex 2D noise value."""
    # Skew input
    s = (x + y) * STRETCH_2D
    xs = x + s
    ys = y + s

    # Floor to get base vertex
    xsb = int(np.floor(xs))
    ysb = int(np.floor(ys))

    # Unskew
    t = (xsb + ysb) * SQUISH_2D
    xb = xsb + t
    yb = ysb + t

    # Delta from base
    dx0 = x - xb
    dy0 = y - yb

    value = 0.0

    # Contribution (0, 0)
    attn0 = 2.0 - dx0 * dx0 - dy0 * dy0
    if attn0 > 0:
        attn0 *= attn0
        value += attn0 * attn0 * _opensimplex_extrapolate(
            perm, perm_grad_index, gradients, xsb, ysb, dx0, dy0
        )

    # Contribution (1, 0)
    dx1 = dx0 - 1.0 - SQUISH_2D
    dy1 = dy0 - SQUISH_2D
    attn1 = 2.0 - dx1 * dx1 - dy1 * dy1
    if attn1 > 0:
        attn1 *= attn1
        value += attn1 * attn1 * _opensimplex_extrapolate(
            perm, perm_grad_index, gradients, xsb + 1, ysb, dx1, dy1
        )

    # Contribution (0, 1)
    dx2 = dx0 - SQUISH_2D
    dy2 = dy0 - 1.0 - SQUISH_2D
    attn2 = 2.0 - dx2 * dx2 - dy2 * dy2
    if attn2 > 0:
        attn2 *= attn2
        value += attn2 * attn2 * _opensimplex_extrapolate(
            perm, perm_grad_index, gradients, xsb, ysb + 1, dx2, dy2
        )

    # Contribution (1, 1)
    dx3 = dx0 - 1.0 - 2.0 * SQUISH_2D
    dy3 = dy0 - 1.0 - 2.0 * SQUISH_2D
    attn3 = 2.0 - dx3 * dx3 - dy3 * dy3
    if attn3 > 0:
        attn3 *= attn3
        value += attn3 * attn3 * _opensimplex_extrapolate(
            perm, perm_grad_index, gradients, xsb + 1, ysb + 1, dx3, dy3
        )

    return value / NORM_2D


@njit(parallel=True, fastmath=True, cache=True)
def opensimplex_fbm_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    perm: NDArray[np.int16],
    perm_grad_index: NDArray[np.int16],
    gradients: NDArray[np.float64],
    octaves: int,
    lacunarity: float,
    persistence: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D OpenSimplex FBM noise using parallel JIT compilation."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for _ in range(octaves):
            noise = _opensimplex_sample(perm, perm_grad_index, gradients, xi, yi)
            total += noise * amp
            amp *= persistence
            xi *= lacunarity
            yi *= lacunarity

        result[i] = total * fractal_bounding * 0.5 + 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def opensimplex_fbm_2d_weighted(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    perm: NDArray[np.int16],
    perm_grad_index: NDArray[np.int16],
    gradients: NDArray[np.float64],
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D OpenSimplex FBM noise with weighted strength."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for _ in range(octaves):
            noise = _opensimplex_sample(perm, perm_grad_index, gradients, xi, yi)
            total += noise * amp
            amp *= (1.0 - weighted_strength + weighted_strength * (noise + 1.0) * 0.5) * persistence
            xi *= lacunarity
            yi *= lacunarity

        result[i] = total * fractal_bounding * 0.5 + 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def opensimplex_ridged_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    perm: NDArray[np.int16],
    perm_grad_index: NDArray[np.int16],
    gradients: NDArray[np.float64],
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D OpenSimplex ridged multifractal noise."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for _ in range(octaves):
            noise_raw = _opensimplex_sample(perm, perm_grad_index, gradients, xi, yi)
            noise = 1.0 - abs(noise_raw)
            total += noise * amp

            if weighted_strength > 0:
                amp *= (1.0 - weighted_strength + weighted_strength * noise) * persistence
            else:
                amp *= persistence

            xi *= lacunarity
            yi *= lacunarity

        result[i] = total * fractal_bounding

    return result


@njit(parallel=True, fastmath=True, cache=True)
def opensimplex_pingpong_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    perm: NDArray[np.int16],
    perm_grad_index: NDArray[np.int16],
    gradients: NDArray[np.float64],
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    ping_pong_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D OpenSimplex ping-pong fractal noise."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for _ in range(octaves):
            noise_raw = _opensimplex_sample(perm, perm_grad_index, gradients, xi, yi)

            # Ping-pong
            pp_val = (noise_raw + 1.0) * ping_pong_strength
            pp_val = pp_val - int(pp_val * 0.5) * 2
            if pp_val >= 1.0:
                pp_val = 2.0 - pp_val
            noise = pp_val

            total += (noise - 0.5) * 2.0 * amp

            if weighted_strength > 0:
                amp *= (1.0 - weighted_strength + weighted_strength * noise) * persistence
            else:
                amp *= persistence

            xi *= lacunarity
            yi *= lacunarity

        result[i] = total * fractal_bounding * 0.5 + 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def opensimplex_single_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    perm: NDArray[np.int16],
    perm_grad_index: NDArray[np.int16],
    gradients: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Generate single octave 2D OpenSimplex noise (no fractal)."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        noise = _opensimplex_sample(perm, perm_grad_index, gradients, x[i], y[i])
        result[i] = (noise + 1.0) * 0.5

    return result


# =============================================================================
# Cellular (Worley/Voronoi) 2D Kernels
# =============================================================================

# Cellular distance function constants
CELLULAR_EUCLIDEAN = 0
CELLULAR_EUCLIDEAN_SQUARED = 1
CELLULAR_MANHATTAN = 2
CELLULAR_HYBRID = 3

# Cellular return type constants
CELLULAR_CELL_VALUE = 0
CELLULAR_DISTANCE = 1
CELLULAR_DISTANCE_2 = 2
CELLULAR_DISTANCE_2_ADD = 3
CELLULAR_DISTANCE_2_SUB = 4
CELLULAR_DISTANCE_2_MUL = 5
CELLULAR_DISTANCE_2_DIV = 6


@njit(fastmath=True, cache=True)
def _cellular_hash_2d(seed: int, x: int, y: int) -> int:
    """Hash function for cellular noise feature points."""
    # FNV-1a inspired hash
    h = seed
    h ^= x
    h *= 0x27d4eb2d
    h ^= y
    h *= 0x27d4eb2d
    return h


@njit(fastmath=True, cache=True)
def _cellular_hash_to_float(hash_val: int) -> float:
    """Convert hash to float in range [0, 1)."""
    return (hash_val & 0x7fffffff) / 2147483648.0


@njit(fastmath=True, cache=True)
def _cellular_distance(
    dx: float,
    dy: float,
    distance_func: int
) -> float:
    """Calculate distance based on distance function type."""
    if distance_func == CELLULAR_EUCLIDEAN:
        return np.sqrt(dx * dx + dy * dy)
    elif distance_func == CELLULAR_EUCLIDEAN_SQUARED:
        return dx * dx + dy * dy
    elif distance_func == CELLULAR_MANHATTAN:
        return abs(dx) + abs(dy)
    else:  # CELLULAR_HYBRID
        return abs(dx) + abs(dy) + (dx * dx + dy * dy)


@njit(fastmath=True, cache=True)
def _cellular_sample(
    x: float,
    y: float,
    seed: int,
    distance_func: int,
    return_type: int,
    jitter: float
) -> float:
    """
    Sample cellular noise at a single point.

    Returns value in range approximately [-1, 1] for most return types.
    """
    # Get cell coordinates
    xr = int(np.floor(x))
    yr = int(np.floor(y))

    # Initialize distances
    distance0 = 1e10  # Closest
    distance1 = 1e10  # Second closest
    closest_hash = 0

    # Search 3x3 neighborhood
    for xi in range(-1, 2):
        for yi in range(-1, 2):
            # Cell coordinates
            cx = xr + xi
            cy = yr + yi

            # Hash for this cell
            h = _cellular_hash_2d(seed, cx, cy)

            # Feature point position within cell (jittered)
            fx = cx + _cellular_hash_to_float(h) * jitter
            h = _cellular_hash_2d(seed, h, cy)
            fy = cy + _cellular_hash_to_float(h) * jitter

            # Distance to feature point
            dx = x - fx
            dy = y - fy
            d = _cellular_distance(dx, dy, distance_func)

            # Update closest distances
            if d < distance0:
                distance1 = distance0
                distance0 = d
                closest_hash = h
            elif d < distance1:
                distance1 = d

    # Calculate return value based on return type
    if return_type == CELLULAR_CELL_VALUE:
        return _cellular_hash_to_float(closest_hash) * 2.0 - 1.0
    elif return_type == CELLULAR_DISTANCE:
        return distance0
    elif return_type == CELLULAR_DISTANCE_2:
        return distance1
    elif return_type == CELLULAR_DISTANCE_2_ADD:
        return (distance0 + distance1) * 0.5
    elif return_type == CELLULAR_DISTANCE_2_SUB:
        return distance1 - distance0
    elif return_type == CELLULAR_DISTANCE_2_MUL:
        return distance0 * distance1
    else:  # CELLULAR_DISTANCE_2_DIV
        if distance1 > 1e-10:
            return distance0 / distance1
        return 0.0


@njit(parallel=True, fastmath=True, cache=True)
def cellular_single_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    distance_func: int,
    return_type: int,
    jitter: float
) -> NDArray[np.float64]:
    """Generate single 2D cellular noise (no fractal)."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        val = _cellular_sample(x[i], y[i], seed, distance_func, return_type, jitter)
        # Normalize to [0, 1] range
        if return_type == CELLULAR_CELL_VALUE:
            result[i] = (val + 1.0) * 0.5
        elif return_type == CELLULAR_DISTANCE_2_SUB:
            result[i] = min(1.0, max(0.0, val))
        elif return_type in (CELLULAR_DISTANCE, CELLULAR_DISTANCE_2):
            result[i] = min(1.0, max(0.0, val))
        elif return_type == CELLULAR_DISTANCE_2_ADD:
            result[i] = min(1.0, max(0.0, val))
        elif return_type == CELLULAR_DISTANCE_2_MUL:
            result[i] = min(1.0, max(0.0, val))
        elif return_type == CELLULAR_DISTANCE_2_DIV:
            result[i] = min(1.0, max(0.0, val))
        else:
            result[i] = min(1.0, max(0.0, (val + 1.0) * 0.5))

    return result


@njit(parallel=True, fastmath=True, cache=True)
def cellular_fbm_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    distance_func: int,
    return_type: int,
    jitter: float,
    octaves: int,
    lacunarity: float,
    persistence: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D cellular FBM noise."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for octave in range(octaves):
            # Use different seed per octave for variation
            octave_seed = seed + octave * 1337
            val = _cellular_sample(xi, yi, octave_seed, distance_func, return_type, jitter)

            # Convert to [-1, 1] range for accumulation
            if return_type == CELLULAR_CELL_VALUE:
                noise = val
            else:
                noise = val * 2.0 - 1.0

            total += noise * amp
            amp *= persistence
            xi *= lacunarity
            yi *= lacunarity

        # Normalize and clamp to [0, 1]
        result[i] = min(1.0, max(0.0, total * fractal_bounding * 0.5 + 0.5))

    return result


@njit(parallel=True, fastmath=True, cache=True)
def cellular_fbm_2d_weighted(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    distance_func: int,
    return_type: int,
    jitter: float,
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D cellular FBM noise with weighted strength."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for octave in range(octaves):
            octave_seed = seed + octave * 1337
            val = _cellular_sample(xi, yi, octave_seed, distance_func, return_type, jitter)

            if return_type == CELLULAR_CELL_VALUE:
                noise = val
            else:
                noise = val * 2.0 - 1.0

            total += noise * amp
            amp *= (1.0 - weighted_strength + weighted_strength * (noise + 1.0) * 0.5) * persistence
            xi *= lacunarity
            yi *= lacunarity

        result[i] = min(1.0, max(0.0, total * fractal_bounding * 0.5 + 0.5))

    return result


@njit(parallel=True, fastmath=True, cache=True)
def cellular_ridged_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    distance_func: int,
    return_type: int,
    jitter: float,
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D cellular ridged multifractal noise."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for octave in range(octaves):
            octave_seed = seed + octave * 1337
            val = _cellular_sample(xi, yi, octave_seed, distance_func, return_type, jitter)

            if return_type == CELLULAR_CELL_VALUE:
                noise_raw = val
            else:
                noise_raw = val * 2.0 - 1.0

            # Ridged: absolute value inverted
            noise = 1.0 - abs(noise_raw)
            total += noise * amp

            if weighted_strength > 0:
                amp *= (1.0 - weighted_strength + weighted_strength * noise) * persistence
            else:
                amp *= persistence

            xi *= lacunarity
            yi *= lacunarity

        result[i] = min(1.0, max(0.0, total * fractal_bounding))

    return result


@njit(parallel=True, fastmath=True, cache=True)
def cellular_pingpong_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    distance_func: int,
    return_type: int,
    jitter: float,
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    ping_pong_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D cellular ping-pong fractal noise."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for octave in range(octaves):
            octave_seed = seed + octave * 1337
            val = _cellular_sample(xi, yi, octave_seed, distance_func, return_type, jitter)

            if return_type == CELLULAR_CELL_VALUE:
                noise_raw = val
            else:
                noise_raw = val * 2.0 - 1.0

            # Ping-pong effect
            pp_val = (noise_raw + 1.0) * ping_pong_strength
            pp_val = pp_val - int(pp_val * 0.5) * 2
            if pp_val >= 1.0:
                pp_val = 2.0 - pp_val
            noise = pp_val

            total += (noise - 0.5) * 2.0 * amp

            if weighted_strength > 0:
                amp *= (1.0 - weighted_strength + weighted_strength * noise) * persistence
            else:
                amp *= persistence

            xi *= lacunarity
            yi *= lacunarity

        result[i] = min(1.0, max(0.0, total * fractal_bounding * 0.5 + 0.5))

    return result


# =============================================================================
# Value Cubic 2D Kernels
# =============================================================================

@njit(fastmath=True, cache=True)
def _value_cubic_hash(seed: int, x: int, y: int) -> float:
    """
    Hash function that returns a value in range [-1, 1].
    Uses a simple but effective hash based on bit manipulation.
    """
    h = seed
    h ^= x * 0x27d4eb2d
    h ^= y * 0x1b873593
    h ^= h >> 15
    h *= 0x85ebca6b
    h ^= h >> 13
    h *= 0xc2b2ae35
    h ^= h >> 16
    # Convert to [-1, 1] range
    return ((h & 0x7fffffff) / 1073741824.0) - 1.0


@njit(fastmath=True, cache=True)
def _cubic_interpolate(a: float, b: float, c: float, d: float, t: float) -> float:
    """
    Catmull-Rom cubic interpolation.

    Args:
        a: Value at position -1
        b: Value at position 0
        c: Value at position 1
        d: Value at position 2
        t: Interpolation factor [0, 1]

    Returns:
        Interpolated value
    """
    p = (d - c) - (a - b)
    q = (a - b) - p
    r = c - a
    s = b
    return p * t * t * t + q * t * t + r * t + s


@njit(fastmath=True, cache=True)
def _value_cubic_sample(x: float, y: float, seed: int) -> float:
    """
    Sample Value Cubic noise at a single point using bicubic interpolation.

    Uses a 4x4 grid of values with Catmull-Rom interpolation for smooth results.
    """
    # Get integer coordinates
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))

    # Fractional parts
    xs = x - x1
    ys = y - y1

    # Get values for 4x4 grid
    # We need positions: (x1-1, y1-1) to (x1+2, y1+2)

    # Row -1 (y1-1)
    v00 = _value_cubic_hash(seed, x1 - 1, y1 - 1)
    v10 = _value_cubic_hash(seed, x1, y1 - 1)
    v20 = _value_cubic_hash(seed, x1 + 1, y1 - 1)
    v30 = _value_cubic_hash(seed, x1 + 2, y1 - 1)

    # Row 0 (y1)
    v01 = _value_cubic_hash(seed, x1 - 1, y1)
    v11 = _value_cubic_hash(seed, x1, y1)
    v21 = _value_cubic_hash(seed, x1 + 1, y1)
    v31 = _value_cubic_hash(seed, x1 + 2, y1)

    # Row 1 (y1+1)
    v02 = _value_cubic_hash(seed, x1 - 1, y1 + 1)
    v12 = _value_cubic_hash(seed, x1, y1 + 1)
    v22 = _value_cubic_hash(seed, x1 + 1, y1 + 1)
    v32 = _value_cubic_hash(seed, x1 + 2, y1 + 1)

    # Row 2 (y1+2)
    v03 = _value_cubic_hash(seed, x1 - 1, y1 + 2)
    v13 = _value_cubic_hash(seed, x1, y1 + 2)
    v23 = _value_cubic_hash(seed, x1 + 1, y1 + 2)
    v33 = _value_cubic_hash(seed, x1 + 2, y1 + 2)

    # Interpolate along x for each row
    row0 = _cubic_interpolate(v00, v10, v20, v30, xs)
    row1 = _cubic_interpolate(v01, v11, v21, v31, xs)
    row2 = _cubic_interpolate(v02, v12, v22, v32, xs)
    row3 = _cubic_interpolate(v03, v13, v23, v33, xs)

    # Interpolate along y
    return _cubic_interpolate(row0, row1, row2, row3, ys)


@njit(parallel=True, fastmath=True, cache=True)
def value_cubic_single_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int
) -> NDArray[np.float64]:
    """Generate single octave 2D Value Cubic noise (no fractal)."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        noise = _value_cubic_sample(x[i], y[i], seed)
        result[i] = (noise + 1.0) * 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def value_cubic_fbm_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    octaves: int,
    lacunarity: float,
    persistence: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D Value Cubic FBM noise."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for octave in range(octaves):
            octave_seed = seed + octave * 1337
            noise = _value_cubic_sample(xi, yi, octave_seed)
            total += noise * amp
            amp *= persistence
            xi *= lacunarity
            yi *= lacunarity

        result[i] = total * fractal_bounding * 0.5 + 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def value_cubic_fbm_2d_weighted(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D Value Cubic FBM noise with weighted strength."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for octave in range(octaves):
            octave_seed = seed + octave * 1337
            noise = _value_cubic_sample(xi, yi, octave_seed)
            total += noise * amp
            amp *= (1.0 - weighted_strength + weighted_strength * (noise + 1.0) * 0.5) * persistence
            xi *= lacunarity
            yi *= lacunarity

        result[i] = total * fractal_bounding * 0.5 + 0.5

    return result


@njit(parallel=True, fastmath=True, cache=True)
def value_cubic_ridged_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D Value Cubic ridged multifractal noise."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for octave in range(octaves):
            octave_seed = seed + octave * 1337
            noise_raw = _value_cubic_sample(xi, yi, octave_seed)

            # Ridged: absolute value inverted
            noise = 1.0 - abs(noise_raw)
            total += noise * amp

            if weighted_strength > 0:
                amp *= (1.0 - weighted_strength + weighted_strength * noise) * persistence
            else:
                amp *= persistence

            xi *= lacunarity
            yi *= lacunarity

        result[i] = min(1.0, max(0.0, total * fractal_bounding))

    return result


@njit(parallel=True, fastmath=True, cache=True)
def value_cubic_pingpong_2d(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    seed: int,
    octaves: int,
    lacunarity: float,
    persistence: float,
    weighted_strength: float,
    ping_pong_strength: float,
    fractal_bounding: float
) -> NDArray[np.float64]:
    """Generate 2D Value Cubic ping-pong fractal noise."""
    n = len(x)
    result = np.zeros(n, dtype=np.float64)

    for i in prange(n):
        xi = x[i]
        yi = y[i]
        total = 0.0
        amp = 1.0

        for octave in range(octaves):
            octave_seed = seed + octave * 1337
            noise_raw = _value_cubic_sample(xi, yi, octave_seed)

            # Ping-pong effect
            pp_val = (noise_raw + 1.0) * ping_pong_strength
            pp_val = pp_val - int(pp_val * 0.5) * 2
            if pp_val >= 1.0:
                pp_val = 2.0 - pp_val
            noise = pp_val

            total += (noise - 0.5) * 2.0 * amp

            if weighted_strength > 0:
                amp *= (1.0 - weighted_strength + weighted_strength * noise) * persistence
            else:
                amp *= persistence

            xi *= lacunarity
            yi *= lacunarity

        result[i] = min(1.0, max(0.0, total * fractal_bounding * 0.5 + 0.5))

    return result

