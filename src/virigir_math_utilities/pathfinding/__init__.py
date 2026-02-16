"""
Pathfinding algorithms module.

This module provides various pathfinding algorithms including A* and its variants,
along with different heuristic functions for estimating distances.

Main Components:
    - astar: Core A* pathfinding algorithm
    - astar_grid_2d: Convenience function for 2D grid pathfinding
    - astar_with_callbacks: A* with visualization/debugging hooks
    - Manhattan: Manhattan distance heuristic
    - Heuristic: Base class for custom heuristics
    - PathResult: Result object containing path and statistics

Quick Start:
    >>> from virigir_math_utilities.pathfinding import astar_grid_2d, Manhattan
    >>>
    >>> def is_walkable(pos):
    ...     x, y = pos
    ...     # Define your walkability logic here
    ...     return True
    >>>
    >>> result = astar_grid_2d((0, 0), (10, 10), is_walkable, Manhattan())
    >>> if result.found:
    ...     print(f"Path found with {result.path_length} steps!")
"""

# Core algorithm functions
from .astar import (
    astar,
    astar_grid_2d,
    astar_with_callbacks,
)

# Heuristic classes
from .heuristics import (
    Heuristic,
    Manhattan,
    Euclidean,
    Chebyshev,
    Octile,
    Zero,
)

# Node and result classes
from .node import (
    PriorityNode,
    PathResult,
    reconstruct_path,
)

# Type definitions
from .heuristics import Position, Position2D, Position3D


__all__ = [
    # Algorithm functions
    'astar',
    'astar_grid_2d',
    'astar_with_callbacks',

    # Heuristics
    'Heuristic',
    'Manhattan',
    'Euclidean',
    'Chebyshev',
    'Octile',
    'Zero',

    # Data structures
    'PriorityNode',
    'PathResult',
    'reconstruct_path',

    # Types
    'Position',
    'Position2D',
    'Position3D',
]

__version__ = '0.1.0'

