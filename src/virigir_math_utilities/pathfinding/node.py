"""
Node data structures for A* pathfinding.

This module provides the node structures used in the pathfinding algorithms,
including priority queue wrappers and path reconstruction utilities.
"""

from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Hashable


# Generic type for node data (must be hashable for use in dictionaries)
T = TypeVar('T', bound=Hashable)


@dataclass(order=True)
class PriorityNode(Generic[T]):
    """
    Node wrapper for use in priority queues (heaps).

    This class wraps a node with its priority (f-score) for efficient
    retrieval of the lowest-cost node in A*'s open set.

    Attributes:
        priority: The f-score (g + h) used for ordering in the priority queue
        node: The actual node data (position, state, etc.)
        g_cost: The actual cost from start to this node
        parent: The parent node in the path (for reconstruction)

    Examples:
        >>> node1 = PriorityNode(5.0, (0, 0), 3.0)
        >>> node2 = PriorityNode(7.0, (1, 1), 5.0)
        >>> node1 < node2
        True
    """

    # Primary comparison field
    priority: float

    # Data fields (excluded from comparison)
    node: T = field(compare=False)
    g_cost: float = field(compare=False)
    parent: Optional[T] = field(default=None, compare=False)

    def __repr__(self) -> str:
        return (f"PriorityNode(priority={self.priority:.2f}, "
                f"node={self.node}, g_cost={self.g_cost:.2f})")


@dataclass
class PathResult(Generic[T]):
    """
    Result of a pathfinding operation.

    Contains the path found (if any) along with useful statistics
    about the search process.

    Attributes:
        path: List of nodes from start to goal, or None if no path found
        cost: Total cost of the path, or None if no path found
        nodes_explored: Number of nodes explored during the search
        path_length: Number of steps in the path (0 if no path)
    """

    path: Optional[list[T]]
    cost: Optional[float]
    nodes_explored: int

    @property
    def path_length(self) -> int:
        """Get the length of the path (number of steps)."""
        return len(self.path) if self.path else 0

    @property
    def found(self) -> bool:
        """Check if a path was found."""
        return self.path is not None

    def __bool__(self) -> bool:
        """Allow boolean evaluation (True if path found)."""
        return self.found

    def __repr__(self) -> str:
        if self.found:
            return (f"PathResult(path_length={self.path_length}, "
                    f"cost={self.cost:.2f}, nodes_explored={self.nodes_explored})")
        return f"PathResult(no path found, nodes_explored={self.nodes_explored})"


def reconstruct_path(came_from: dict[T, T], current: T) -> list[T]:
    """
    Reconstruct the path from start to goal using the came_from dictionary.

    This function traces back from the goal node to the start node using
    the parent relationships stored in came_from during the A* search.

    Args:
        came_from: Dictionary mapping each node to its parent
        current: The goal node to trace back from

    Returns:
        List of nodes from start to goal (inclusive)

    Examples:
        >>> came_from = {(1,0): (0,0), (2,0): (1,0), (2,1): (2,0)}
        >>> reconstruct_path(came_from, (2,1))
        [(0, 0), (1, 0), (2, 0), (2, 1)]
    """
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    return list(reversed(path))

