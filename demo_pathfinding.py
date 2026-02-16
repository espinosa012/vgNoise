#!/usr/bin/env python3
"""
Demostración del módulo de pathfinding.

Este script muestra cómo usar el algoritmo A* con la heurística Manhattan
para pathfinding en un grid 2D con obstáculos.

Para ejecutar:
    cd /home/deck/Documents/virigir/vgNoise
    python3 demo_pathfinding.py
"""

import sys
import os

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Ahora podemos importar desde el módulo pathfinding directamente
# sin pasar por el __init__.py problemático
from virigir_math_utilities.pathfinding.heuristics import Manhattan
from virigir_math_utilities.pathfinding.astar import astar_grid_2d


def print_grid_with_path(width, height, obstacles, path, start, goal):
    """Imprime el grid con el path encontrado."""
    print("\nGrid:")
    print("  S = Start, G = Goal, # = Obstacle, * = Path, . = Empty")
    print()

    for y in range(height):
        print(f" {y:2d} ", end="")
        for x in range(width):
            pos = (x, y)
            if pos == start:
                print(" S ", end="")
            elif pos == goal:
                print(" G ", end="")
            elif pos in obstacles:
                print(" # ", end="")
            elif path and pos in path:
                print(" * ", end="")
            else:
                print(" . ", end="")
        print()

    print("    ", end="")
    for x in range(width):
        print(f" {x:1d} ", end="")
    print("\n")


def demo_basic_pathfinding():
    """Demostración básica de pathfinding."""
    print("=" * 60)
    print("DEMO 1: Pathfinding Básico")
    print("=" * 60)

    # Configurar grid simple
    width, height = 10, 10
    start = (0, 0)
    goal = (9, 9)

    def is_walkable(pos):
        x, y = pos
        return 0 <= x < width and 0 <= y < height

    # Ejecutar A*
    heuristic = Manhattan()
    result = astar_grid_2d(start, goal, is_walkable, heuristic)

    # Mostrar resultados
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"\nResultado:")
    print(f"  ✓ Path encontrado: {result.found}")
    print(f"  ✓ Longitud del path: {result.path_length} pasos")
    print(f"  ✓ Costo total: {result.cost}")
    print(f"  ✓ Nodos explorados: {result.nodes_explored}")

    print_grid_with_path(width, height, set(), result.path, start, goal)


def demo_pathfinding_with_obstacles():
    """Demostración de pathfinding con obstáculos."""
    print("=" * 60)
    print("DEMO 2: Pathfinding con Obstáculos")
    print("=" * 60)

    # Configurar grid con obstáculos
    width, height = 10, 10
    start = (0, 5)
    goal = (9, 5)

    # Crear un muro vertical
    obstacles = {(5, y) for y in range(2, 8)}

    def is_walkable(pos):
        x, y = pos
        return (0 <= x < width and
                0 <= y < height and
                pos not in obstacles)

    # Ejecutar A*
    heuristic = Manhattan()
    result = astar_grid_2d(start, goal, is_walkable, heuristic)

    # Mostrar resultados
    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Obstáculos: {len(obstacles)} celdas bloqueadas")
    print(f"\nResultado:")
    print(f"  ✓ Path encontrado: {result.found}")
    print(f"  ✓ Longitud del path: {result.path_length} pasos")
    print(f"  ✓ Costo total: {result.cost}")
    print(f"  ✓ Nodos explorados: {result.nodes_explored}")

    print_grid_with_path(width, height, obstacles, result.path, start, goal)


def demo_diagonal_movement():
    """Demostración de pathfinding con movimiento diagonal."""
    print("=" * 60)
    print("DEMO 3: Pathfinding con Movimiento Diagonal")
    print("=" * 60)

    width, height = 8, 8
    start = (0, 0)
    goal = (7, 7)

    def is_walkable(pos):
        x, y = pos
        return 0 <= x < width and 0 <= y < height

    # Sin diagonal
    heuristic = Manhattan()
    result_no_diag = astar_grid_2d(start, goal, is_walkable, heuristic,
                                   allow_diagonal=False)

    # Con diagonal
    result_with_diag = astar_grid_2d(start, goal, is_walkable, heuristic,
                                     allow_diagonal=True,
                                     diagonal_cost=1.414213562)

    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"\nSin movimiento diagonal:")
    print(f"  Longitud: {result_no_diag.path_length} pasos")
    print(f"  Costo: {result_no_diag.cost:.2f}")
    print(f"  Nodos explorados: {result_no_diag.nodes_explored}")

    print(f"\nCon movimiento diagonal:")
    print(f"  Longitud: {result_with_diag.path_length} pasos")
    print(f"  Costo: {result_with_diag.cost:.2f}")
    print(f"  Nodos explorados: {result_with_diag.nodes_explored}")

    print_grid_with_path(width, height, set(), result_with_diag.path, start, goal)


def demo_weighted_astar():
    """Demostración de A* ponderado."""
    print("=" * 60)
    print("DEMO 4: Weighted A* (Trade-off Velocidad vs Optimalidad)")
    print("=" * 60)

    width, height = 20, 20
    start = (0, 0)
    goal = (19, 19)

    # Crear algunos obstáculos aleatorios
    obstacles = {(5, y) for y in range(5, 15)}
    obstacles.update({(15, y) for y in range(5, 15)})
    obstacles.update({(x, 10) for x in range(5, 15)})

    def is_walkable(pos):
        x, y = pos
        return (0 <= x < width and
                0 <= y < height and
                pos not in obstacles)

    # A* normal
    heuristic_normal = Manhattan(weight=1.0)
    result_normal = astar_grid_2d(start, goal, is_walkable, heuristic_normal)

    # A* ponderado
    heuristic_weighted = Manhattan(weight=2.0)
    result_weighted = astar_grid_2d(start, goal, is_walkable, heuristic_weighted)

    print(f"Start: {start}")
    print(f"Goal: {goal}")
    print(f"Obstáculos: {len(obstacles)} celdas bloqueadas")

    print(f"\nA* Normal (weight=1.0):")
    print(f"  Costo: {result_normal.cost:.2f}")
    print(f"  Nodos explorados: {result_normal.nodes_explored}")

    print(f"\nWeighted A* (weight=2.0):")
    print(f"  Costo: {result_weighted.cost:.2f}")
    print(f"  Nodos explorados: {result_weighted.nodes_explored}")
    print(f"  Reducción de nodos: {result_normal.nodes_explored - result_weighted.nodes_explored} "
          f"({100 * (1 - result_weighted.nodes_explored / result_normal.nodes_explored):.1f}%)")


def demo_manhattan_heuristic():
    """Demostración de la heurística Manhattan."""
    print("=" * 60)
    print("DEMO 5: Heurística Manhattan")
    print("=" * 60)

    heuristic = Manhattan()

    print("Distancias Manhattan en 2D:")
    test_cases_2d = [
        ((0, 0), (3, 4)),
        ((1, 1), (4, 5)),
        ((5, 5), (5, 5)),
        ((-1, -1), (2, 2)),
    ]

    for start, goal in test_cases_2d:
        dist = heuristic.calculate(start, goal)
        print(f"  {start} -> {goal}: {dist}")

    print("\nDistancias Manhattan en 3D:")
    test_cases_3d = [
        ((0, 0, 0), (3, 4, 5)),
        ((1, 2, 3), (4, 6, 3)),
    ]

    for start, goal in test_cases_3d:
        dist = heuristic.calculate(start, goal)
        print(f"  {start} -> {goal}: {dist}")

    print("\nHeurística ponderada (weight=2.0):")
    weighted = Manhattan(weight=2.0)
    for start, goal in test_cases_2d[:2]:
        dist = weighted.calculate(start, goal)
        print(f"  {start} -> {goal}: {dist}")


def main():
    """Ejecuta todas las demostraciones."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "DEMOSTRACIÓN DE PATHFINDING A*" + " " * 17 + "║")
    print("║" + " " * 15 + "Heurística Manhattan" + " " * 22 + "║")
    print("╚" + "═" * 58 + "╝")
    print("\n")

    try:
        demo_manhattan_heuristic()

        demo_basic_pathfinding()

        demo_pathfinding_with_obstacles()

        demo_diagonal_movement()

        demo_weighted_astar()

        print("\n" + "=" * 60)
        print("✓ Todas las demostraciones completadas exitosamente!")
        print("=" * 60)
        print("\nVer README.md para más información sobre el uso del módulo.")

    except Exception as e:
        print(f"\n✗ Error en la demostración: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

