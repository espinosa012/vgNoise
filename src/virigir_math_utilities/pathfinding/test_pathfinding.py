"""
Tests for the pathfinding module.

This file contains unit tests and examples for the A* pathfinding algorithm
and its heuristics, particularly focusing on the Manhattan heuristic.
"""

import pytest
from virigir_math_utilities.pathfinding import (
    astar,
    astar_grid_2d,
    Manhattan,
    PathResult,
)


class TestManhattanHeuristic:
    """Tests for the Manhattan distance heuristic."""

    def test_2d_manhattan_distance(self):
        """Test Manhattan distance calculation in 2D."""
        heuristic = Manhattan()

        # Test basic distance
        assert heuristic.calculate((0, 0), (3, 4)) == 7.0
        assert heuristic.calculate((1, 1), (4, 5)) == 7.0

        # Test same position
        assert heuristic.calculate((5, 5), (5, 5)) == 0.0

        # Test negative coordinates
        assert heuristic.calculate((-1, -1), (2, 2)) == 6.0

    def test_3d_manhattan_distance(self):
        """Test Manhattan distance calculation in 3D."""
        heuristic = Manhattan()

        # Test 3D distance
        assert heuristic.calculate((0, 0, 0), (3, 4, 5)) == 12.0
        assert heuristic.calculate((1, 2, 3), (4, 6, 3)) == 7.0

    def test_weighted_manhattan(self):
        """Test weighted Manhattan heuristic."""
        heuristic = Manhattan(weight=2.0)

        # Distance should be doubled
        assert heuristic.calculate((0, 0), (3, 4)) == 14.0

    def test_invalid_weight(self):
        """Test that invalid weights raise an error."""
        with pytest.raises(ValueError):
            Manhattan(weight=0)

        with pytest.raises(ValueError):
            Manhattan(weight=-1.0)

    def test_dimension_mismatch(self):
        """Test that mismatched dimensions raise an error."""
        heuristic = Manhattan()

        with pytest.raises(ValueError):
            heuristic.calculate((0, 0), (1, 2, 3))

    def test_callable(self):
        """Test that heuristic can be called as a function."""
        heuristic = Manhattan()

        # Should work as a callable
        assert heuristic((0, 0), (3, 4)) == 7.0


class TestAStarBasic:
    """Basic tests for the A* algorithm."""

    def test_simple_path(self):
        """Test finding a simple path in a grid."""
        def get_neighbors(pos):
            x, y = pos
            return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        def cost(from_pos, to_pos):
            return 1.0

        heuristic = Manhattan()
        result = astar((0, 0), (3, 0), get_neighbors, cost, heuristic)

        assert result.found
        assert result.path_length == 4
        assert result.path == [(0, 0), (1, 0), (2, 0), (3, 0)]
        assert result.cost == 3.0

    def test_no_path(self):
        """Test when no path exists."""
        def get_neighbors(pos):
            # Dead end - no neighbors
            return []

        def cost(from_pos, to_pos):
            return 1.0

        heuristic = Manhattan()
        result = astar((0, 0), (3, 0), get_neighbors, cost, heuristic)

        assert not result.found
        assert result.path is None
        assert result.cost is None
        assert result.nodes_explored > 0

    def test_max_iterations(self):
        """Test that max_iterations limits the search."""
        def get_neighbors(pos):
            x, y = pos
            return [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

        def cost(from_pos, to_pos):
            return 1.0

        heuristic = Manhattan()
        result = astar((0, 0), (100, 100), get_neighbors, cost, heuristic, max_iterations=10)

        # Should give up before finding the path
        assert not result.found


class TestAStarGrid2D:
    """Tests for the 2D grid convenience function."""

    def test_simple_grid_path(self):
        """Test pathfinding on a simple grid."""
        # Simple 5x5 grid, all walkable
        def is_walkable(pos):
            x, y = pos
            return 0 <= x < 5 and 0 <= y < 5

        heuristic = Manhattan()
        result = astar_grid_2d((0, 0), (4, 4), is_walkable, heuristic)

        assert result.found
        assert result.path_length == 9  # 8 moves + 1 for start position
        assert result.cost == 8.0

    def test_grid_with_obstacles(self):
        """Test pathfinding with obstacles."""
        # Grid with a wall
        obstacles = {(1, 1), (1, 2), (1, 3)}

        def is_walkable(pos):
            x, y = pos
            return 0 <= x < 5 and 0 <= y < 5 and pos not in obstacles

        heuristic = Manhattan()
        result = astar_grid_2d((0, 2), (2, 2), is_walkable, heuristic)

        assert result.found
        # Should go around the wall
        assert (1, 2) not in result.path

    def test_diagonal_movement(self):
        """Test pathfinding with diagonal movement allowed."""
        def is_walkable(pos):
            x, y = pos
            return 0 <= x < 5 and 0 <= y < 5

        heuristic = Manhattan()
        result = astar_grid_2d((0, 0), (2, 2), is_walkable, heuristic, allow_diagonal=True)

        assert result.found
        # With diagonals, should be shorter
        assert result.path_length <= 3  # Could be diagonal moves

    def test_blocked_goal(self):
        """Test when goal is blocked."""
        def is_walkable(pos):
            # Goal is not walkable
            return pos != (5, 5)

        heuristic = Manhattan()
        result = astar_grid_2d((0, 0), (5, 5), is_walkable, heuristic, max_iterations=100)

        assert not result.found


class TestPathResult:
    """Tests for PathResult class."""

    def test_path_result_with_path(self):
        """Test PathResult when a path is found."""
        path = [(0, 0), (1, 0), (2, 0)]
        result = PathResult(path, 2.0, 5)

        assert result.found
        assert bool(result)  # Should be truthy
        assert result.path_length == 3
        assert result.cost == 2.0
        assert result.nodes_explored == 5

    def test_path_result_no_path(self):
        """Test PathResult when no path is found."""
        result = PathResult(None, None, 10)

        assert not result.found
        assert not bool(result)  # Should be falsy
        assert result.path_length == 0
        assert result.cost is None
        assert result.nodes_explored == 10

    def test_path_result_repr(self):
        """Test string representation of PathResult."""
        result_with_path = PathResult([(0, 0), (1, 0)], 1.0, 3)
        assert "path_length=2" in repr(result_with_path)
        assert "cost=1.00" in repr(result_with_path)

        result_no_path = PathResult(None, None, 5)
        assert "no path found" in repr(result_no_path)


def example_maze_pathfinding():
    """
    Example: Finding a path through a maze.

    This demonstrates how to use the pathfinding module for a practical
    maze-solving problem.
    """
    # Define a maze (0 = walkable, 1 = wall)
    maze = [
        [0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
    ]

    def is_walkable(pos):
        x, y = pos
        if 0 <= x < len(maze[0]) and 0 <= y < len(maze):
            return maze[y][x] == 0
        return False

    heuristic = Manhattan()
    result = astar_grid_2d((0, 0), (4, 4), is_walkable, heuristic)

    if result.found:
        print(f"Path found with {result.path_length} steps!")
        print(f"Total cost: {result.cost}")
        print(f"Nodes explored: {result.nodes_explored}")
        print(f"Path: {result.path}")
        return result.path
    else:
        print("No path found!")
        return None


def example_weighted_heuristic():
    """
    Example: Using a weighted heuristic for faster (but suboptimal) pathfinding.

    This demonstrates how weighted A* can trade optimality for speed.
    """
    def is_walkable(pos):
        x, y = pos
        return 0 <= x < 100 and 0 <= y < 100

    # Standard A* (optimal)
    heuristic_normal = Manhattan(weight=1.0)
    result_normal = astar_grid_2d((0, 0), (99, 99), is_walkable, heuristic_normal)

    # Weighted A* (faster, less optimal)
    heuristic_weighted = Manhattan(weight=2.0)
    result_weighted = astar_grid_2d((0, 0), (99, 99), is_walkable, heuristic_weighted)

    print(f"Normal A*: explored {result_normal.nodes_explored} nodes")
    print(f"Weighted A*: explored {result_weighted.nodes_explored} nodes")

    return result_normal, result_weighted


if __name__ == "__main__":
    print("Running pathfinding examples...")
    print("\n" + "="*50)
    print("Example 1: Maze Pathfinding")
    print("="*50)
    example_maze_pathfinding()

    print("\n" + "="*50)
    print("Example 2: Weighted Heuristic Comparison")
    print("="*50)
    example_weighted_heuristic()

