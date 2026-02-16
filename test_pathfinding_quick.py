#!/usr/bin/env python3
"""
Quick test script for the pathfinding module.
"""

import sys
sys.path.insert(0, '/home/deck/Documents/virigir/vgNoise/src')

from virigir_math_utilities.pathfinding.heuristics import Manhattan
from virigir_math_utilities.pathfinding.astar import astar_grid_2d

# Test Manhattan heuristic
heuristic = Manhattan()
print('Testing Manhattan heuristic:')
print(f'  Distance (0,0) to (3,4): {heuristic.calculate((0, 0), (3, 4))}')
print(f'  Distance (1,2,3) to (4,6,3): {heuristic.calculate((1, 2, 3), (4, 6, 3))}')
print(f'  Same position (5,5) to (5,5): {heuristic.calculate((5, 5), (5, 5))}')

# Test weighted heuristic
weighted = Manhattan(weight=2.0)
print(f'\nWeighted Manhattan (weight=2.0):')
print(f'  Distance (0,0) to (3,4): {weighted.calculate((0, 0), (3, 4))}')

# Test simple pathfinding
def is_walkable(pos):
    x, y = pos
    return 0 <= x < 5 and 0 <= y < 5

print('\n' + '='*50)
print('Testing A* pathfinding on 5x5 grid:')
print('='*50)
result = astar_grid_2d((0, 0), (4, 4), is_walkable, heuristic)
print(f'  Path found: {result.found}')
print(f'  Path length: {result.path_length}')
print(f'  Cost: {result.cost}')
print(f'  Nodes explored: {result.nodes_explored}')
print(f'  Path: {result.path}')

# Test with obstacles
print('\n' + '='*50)
print('Testing A* pathfinding with obstacles:')
print('='*50)
obstacles = {(1, 1), (1, 2), (1, 3)}

def is_walkable_with_obstacles(pos):
    x, y = pos
    return 0 <= x < 5 and 0 <= y < 5 and pos not in obstacles

result2 = astar_grid_2d((0, 2), (2, 2), is_walkable_with_obstacles, heuristic)
print(f'  Path found: {result2.found}')
print(f'  Path length: {result2.path_length}')
print(f'  Cost: {result2.cost}')
print(f'  Path: {result2.path}')

# Test with diagonal movement
print('\n' + '='*50)
print('Testing A* pathfinding with diagonal movement:')
print('='*50)
result3 = astar_grid_2d((0, 0), (2, 2), is_walkable, heuristic, allow_diagonal=True)
print(f'  Path found: {result3.found}')
print(f'  Path length: {result3.path_length}')
print(f'  Cost: {result3.cost:.2f}')
print(f'  Path: {result3.path}')

print('\n✓ All tests passed!')

