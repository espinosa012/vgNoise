"""
Heuristic functions for A* pathfinding algorithm.

This module provides various heuristic functions used to estimate the cost
from a given position to the goal in pathfinding algorithms.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union


# Type aliases for clarity
Position2D = Tuple[int, int]
Position3D = Tuple[int, int, int]
Position = Union[Position2D, Position3D]


class Heuristic(ABC):
    """
    Abstract base class for heuristic functions.

    A heuristic function estimates the cost from a position to the goal.
    For A* to guarantee the shortest path, the heuristic must be admissible
    (never overestimate the actual cost).
    """

    @abstractmethod
    def calculate(self, from_pos: Position, to_pos: Position) -> float:
        """
        Calculate the estimated cost from one position to another.

        Args:
            from_pos: Starting position (tuple of coordinates)
            to_pos: Goal position (tuple of coordinates)

        Returns:
            Estimated cost as a float

        Raises:
            ValueError: If positions have incompatible dimensions
        """
        pass

    def __call__(self, from_pos: Position, to_pos: Position) -> float:
        """Allow the heuristic to be called as a function."""
        return self.calculate(from_pos, to_pos)


class Manhattan(Heuristic):
    """
    Manhattan distance heuristic (L1 norm).

    Calculates the sum of absolute differences of coordinates.
    Best suited for grid-based movement where only cardinal directions
    (up, down, left, right) are allowed.

    Formula (2D): |x1 - x2| + |y1 - y2|
    Formula (3D): |x1 - x2| + |y1 - y2| + |z1 - z2|

    Examples:
        >>> heuristic = Manhattan()
        >>> heuristic.calculate((0, 0), (3, 4))
        7.0
        >>> heuristic.calculate((1, 2, 3), (4, 6, 3))
        7.0
    """

    def __init__(self, weight: float = 1.0):
        """
        Initialize Manhattan heuristic.

        Args:
            weight: Multiplier for the heuristic value. Default is 1.0.
                   Values > 1.0 make A* faster but less optimal (weighted A*).
                   Values < 1.0 make A* explore more nodes.
        """
        if weight <= 0:
            raise ValueError("Weight must be positive")
        self.weight = weight

    def calculate(self, from_pos: Position, to_pos: Position) -> float:
        """
        Calculate Manhattan distance between two positions.

        Args:
            from_pos: Starting position (tuple of coordinates)
            to_pos: Goal position (tuple of coordinates)

        Returns:
            Manhattan distance multiplied by weight

        Raises:
            ValueError: If positions have different dimensions
        """
        if len(from_pos) != len(to_pos):
            raise ValueError(
                f"Position dimensions must match: "
                f"got {len(from_pos)} and {len(to_pos)}"
            )

        distance = sum(abs(a - b) for a, b in zip(from_pos, to_pos))
        return float(distance) * self.weight


# Placeholder classes for future implementation
class Euclidean(Heuristic):
    """
    Euclidean distance heuristic (L2 norm).

    Calculates the straight-line distance between two points.
    Best suited for free movement in any direction.

    Formula (2D): sqrt((x1-x2)² + (y1-y2)²)

    Note: Not yet implemented. Will be added in future versions.
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        raise NotImplementedError("Euclidean heuristic will be implemented in the future")

    def calculate(self, from_pos: Position, to_pos: Position) -> float:
        raise NotImplementedError("Euclidean heuristic will be implemented in the future")


class Chebyshev(Heuristic):
    """
    Chebyshev distance heuristic (L∞ norm).

    Calculates the maximum absolute difference of coordinates.
    Best suited for grid-based movement where diagonal moves are allowed
    and have the same cost as cardinal moves.

    Formula (2D): max(|x1-x2|, |y1-y2|)

    Note: Not yet implemented. Will be added in future versions.
    """

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        raise NotImplementedError("Chebyshev heuristic will be implemented in the future")

    def calculate(self, from_pos: Position, to_pos: Position) -> float:
        raise NotImplementedError("Chebyshev heuristic will be implemented in the future")


class Octile(Heuristic):
    """
    Octile distance heuristic (diagonal distance).

    Combines Manhattan and diagonal movement costs.
    Best suited for 8-directional grid movement where diagonal moves
    cost more than cardinal moves (typically sqrt(2)).

    Formula (2D): D * (dx + dy) + (D2 - 2*D) * min(dx, dy)
    where D is the cardinal cost and D2 is the diagonal cost

    Note: Not yet implemented. Will be added in future versions.
    """

    def __init__(self, weight: float = 1.0, diagonal_cost: float = 1.414213562):
        self.weight = weight
        self.diagonal_cost = diagonal_cost
        raise NotImplementedError("Octile heuristic will be implemented in the future")

    def calculate(self, from_pos: Position, to_pos: Position) -> float:
        raise NotImplementedError("Octile heuristic will be implemented in the future")


class Zero(Heuristic):
    """
    Zero heuristic (always returns 0).

    When used with A*, this degrades the algorithm to Dijkstra's algorithm,
    which guarantees the shortest path but explores more nodes.
    Useful for testing and comparison purposes.
    """

    def calculate(self, from_pos: Position, to_pos: Position) -> float:
        """Always returns 0, turning A* into Dijkstra's algorithm."""
        return 0.0

