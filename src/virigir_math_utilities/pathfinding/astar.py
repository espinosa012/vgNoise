"""
A* pathfinding algorithm implementation.

This module provides a flexible and efficient implementation of the A* algorithm
for finding the shortest path between two points in a graph or grid.
"""

from heapq import heappush, heappop
from typing import Callable, TypeVar, Hashable, Optional, Set
from .node import PriorityNode, PathResult, reconstruct_path
from .heuristics import Heuristic


# Generic type for nodes (must be hashable)
T = TypeVar('T', bound=Hashable)


def astar(
    start: T,
    goal: T,
    neighbors_fn: Callable[[T], list[T]],
    cost_fn: Callable[[T, T], float],
    heuristic: Heuristic,
    max_iterations: Optional[int] = None
) -> PathResult[T]:
    """
    Find the shortest path from start to goal using the A* algorithm.

    A* is an informed search algorithm that uses a heuristic function to
    efficiently find the shortest path. It guarantees the optimal solution
    if the heuristic is admissible (never overestimates the actual cost).

    Args:
        start: The starting node/position
        goal: The goal node/position
        neighbors_fn: Function that returns list of neighbor nodes for a given node
        cost_fn: Function that returns the cost of moving from one node to another
        heuristic: Heuristic object to estimate cost from node to goal
        max_iterations: Maximum number of iterations before giving up (None for unlimited)

    Returns:
        PathResult object containing the path, cost, and statistics

    Examples:
        >>> from pathfinding.heuristics import Manhattan
        >>>
        >>> def get_neighbors(pos):
        ...     x, y = pos
        ...     return [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]
        >>>
        >>> def cost(from_pos, to_pos):
        ...     return 1.0
        >>>
        >>> result = astar((0,0), (3,4), get_neighbors, cost, Manhattan())
        >>> print(result.path)
        [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]

    Time Complexity: O(b^d) where b is branching factor and d is depth
    Space Complexity: O(b^d) for storing nodes in open and closed sets
    """
    # Initialize the open set (priority queue)
    open_set: list[PriorityNode[T]] = []
    heappush(open_set, PriorityNode(0.0, start, 0.0, None))

    # Track which nodes we've visited
    closed_set: Set[T] = set()

    # Store the best path to each node
    came_from: dict[T, T] = {}

    # Store the best known cost to reach each node
    g_score: dict[T, float] = {start: 0.0}

    # Track iterations
    iterations = 0
    nodes_explored = 0

    while open_set:
        # Check iteration limit
        if max_iterations is not None and iterations >= max_iterations:
            return PathResult(None, None, nodes_explored)

        iterations += 1

        # Get the node with lowest f-score
        current_wrapper = heappop(open_set)
        current = current_wrapper.node

        # Skip if we've already processed this node
        if current in closed_set:
            continue

        # Mark as explored
        closed_set.add(current)
        nodes_explored += 1

        # Check if we've reached the goal
        if current == goal:
            path = reconstruct_path(came_from, current)
            return PathResult(path, g_score[current], nodes_explored)

        # Explore neighbors
        for neighbor in neighbors_fn(current):
            # Skip already processed nodes
            if neighbor in closed_set:
                continue

            # Calculate tentative g-score
            tentative_g = g_score[current] + cost_fn(current, neighbor)

            # If this is a better path to the neighbor, update it
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                # Calculate f-score (g + h)
                h_score = heuristic.calculate(neighbor, goal)
                f_score = tentative_g + h_score

                # Add to open set
                heappush(open_set, PriorityNode(f_score, neighbor, tentative_g, current))

    # No path found
    return PathResult(None, None, nodes_explored)


def astar_grid_2d(
    start: tuple[int, int],
    goal: tuple[int, int],
    is_walkable_fn: Callable[[tuple[int, int]], bool],
    heuristic: Heuristic,
    allow_diagonal: bool = False,
    diagonal_cost: float = 1.414213562,
    max_iterations: Optional[int] = None
) -> PathResult[tuple[int, int]]:
    """
    Convenience function for A* pathfinding on a 2D grid.

    This is a specialized version of astar() optimized for grid-based pathfinding.
    It automatically generates neighbors and calculates costs based on grid movement.

    Args:
        start: Starting position (x, y)
        goal: Goal position (x, y)
        is_walkable_fn: Function that returns True if a position is walkable
        heuristic: Heuristic object to estimate cost from position to goal
        allow_diagonal: Whether diagonal movement is allowed (default: False)
        diagonal_cost: Cost of diagonal movement (default: sqrt(2) ≈ 1.414)
        max_iterations: Maximum iterations before giving up (None for unlimited)

    Returns:
        PathResult object containing the path, cost, and statistics

    Examples:
        >>> from pathfinding.heuristics import Manhattan
        >>>
        >>> # Define a simple grid (1 = walkable, 0 = blocked)
        >>> grid = [[1, 1, 0], [1, 1, 1], [0, 1, 1]]
        >>>
        >>> def is_walkable(pos):
        ...     x, y = pos
        ...     return 0 <= x < 3 and 0 <= y < 3 and grid[y][x] == 1
        >>>
        >>> result = astar_grid_2d((0,0), (2,2), is_walkable, Manhattan())
        >>> print(f"Path length: {result.path_length}")
        Path length: 5
    """
    def get_neighbors(pos: tuple[int, int]) -> list[tuple[int, int]]:
        """Get valid neighbors for a grid position."""
        x, y = pos
        neighbors = []

        # Cardinal directions
        cardinal = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        for neighbor in cardinal:
            if is_walkable_fn(neighbor):
                neighbors.append(neighbor)

        # Diagonal directions
        if allow_diagonal:
            diagonal = [(x+1, y+1), (x+1, y-1), (x-1, y+1), (x-1, y-1)]
            for neighbor in diagonal:
                if is_walkable_fn(neighbor):
                    neighbors.append(neighbor)

        return neighbors

    def cost(from_pos: tuple[int, int], to_pos: tuple[int, int]) -> float:
        """Calculate movement cost between two positions."""
        # Check if it's a diagonal move
        if allow_diagonal and abs(from_pos[0] - to_pos[0]) + abs(from_pos[1] - to_pos[1]) == 2:
            return diagonal_cost
        return 1.0

    return astar(start, goal, get_neighbors, cost, heuristic, max_iterations)


def astar_with_callbacks(
    start: T,
    goal: T,
    neighbors_fn: Callable[[T], list[T]],
    cost_fn: Callable[[T, T], float],
    heuristic: Heuristic,
    on_node_explored: Optional[Callable[[T], None]] = None,
    on_node_added: Optional[Callable[[T, float], None]] = None,
    max_iterations: Optional[int] = None
) -> PathResult[T]:
    """
    A* algorithm with callback hooks for visualization or debugging.

    This version of A* allows you to register callbacks that are invoked
    during the search process, useful for visualization, debugging, or
    progress tracking.

    Args:
        start: The starting node/position
        goal: The goal node/position
        neighbors_fn: Function that returns list of neighbor nodes
        cost_fn: Function that returns the cost of moving between nodes
        heuristic: Heuristic object to estimate cost to goal
        on_node_explored: Callback invoked when a node is explored
        on_node_added: Callback invoked when a node is added to open set (node, f_score)
        max_iterations: Maximum iterations before giving up

    Returns:
        PathResult object containing the path, cost, and statistics
    """
    open_set: list[PriorityNode[T]] = []
    heappush(open_set, PriorityNode(0.0, start, 0.0, None))

    closed_set: Set[T] = set()
    came_from: dict[T, T] = {}
    g_score: dict[T, float] = {start: 0.0}

    iterations = 0
    nodes_explored = 0

    while open_set:
        if max_iterations is not None and iterations >= max_iterations:
            return PathResult(None, None, nodes_explored)

        iterations += 1
        current_wrapper = heappop(open_set)
        current = current_wrapper.node

        if current in closed_set:
            continue

        closed_set.add(current)
        nodes_explored += 1

        # Invoke exploration callback
        if on_node_explored:
            on_node_explored(current)

        if current == goal:
            path = reconstruct_path(came_from, current)
            return PathResult(path, g_score[current], nodes_explored)

        for neighbor in neighbors_fn(current):
            if neighbor in closed_set:
                continue

            tentative_g = g_score[current] + cost_fn(current, neighbor)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                h_score = heuristic.calculate(neighbor, goal)
                f_score = tentative_g + h_score

                heappush(open_set, PriorityNode(f_score, neighbor, tentative_g, current))

                # Invoke added callback
                if on_node_added:
                    on_node_added(neighbor, f_score)

    return PathResult(None, None, nodes_explored)

