#!/usr/bin/env python3
"""Simple test for pathfinding"""

from heuristics import Manhattan
from astar import astar_grid_2d

# Test Manhattan
h = Manhattan()
dist = h.calculate((0, 0), (3, 4))
print(f"Manhattan distance: {dist}")

# Test pathfinding
def walkable(pos):
    x, y = pos
    return 0 <= x < 5 and 0 <= y < 5

result = astar_grid_2d((0, 0), (4, 4), walkable, h)
print(f"Path found: {result.found}")
print(f"Length: {result.path_length}")
print(f"Cost: {result.cost}")

