#!/usr/bin/env python3
"""
Ejemplos rápidos de uso del módulo de pathfinding.

Este archivo contiene snippets listos para copiar y pegar.
"""

# =============================================================================
# EJEMPLO 1: Pathfinding Básico en Grid
# =============================================================================

def ejemplo_basico():
    """Ejemplo más simple de pathfinding en un grid."""
    from virigir_math_utilities.pathfinding import astar_grid_2d, Manhattan

    # Definir walkability (cualquier celda entre 0,0 y 9,9 es caminable)
    def is_walkable(pos):
        x, y = pos
        return 0 <= x < 10 and 0 <= y < 10

    # Encontrar path
    result = astar_grid_2d(
        start=(0, 0),
        goal=(9, 9),
        is_walkable_fn=is_walkable,
        heuristic=Manhattan()
    )

    # Usar resultado
    if result.found:
        print(f"✓ Path encontrado con {result.path_length} pasos")
        print(f"  Costo: {result.cost}")
        print(f"  Path: {result.path}")


# =============================================================================
# EJEMPLO 2: Grid con Obstáculos
# =============================================================================

def ejemplo_con_obstaculos():
    """Pathfinding evitando obstáculos."""
    from virigir_math_utilities.pathfinding import astar_grid_2d, Manhattan

    # Definir obstáculos
    obstacles = {(2, 2), (2, 3), (2, 4), (3, 2), (4, 2)}

    def is_walkable(pos):
        x, y = pos
        return (0 <= x < 10 and
                0 <= y < 10 and
                pos not in obstacles)

    result = astar_grid_2d((0, 0), (5, 5), is_walkable, Manhattan())

    if result.found:
        print(f"✓ Path encontrado (evitando {len(obstacles)} obstáculos)")


# =============================================================================
# EJEMPLO 3: Movimiento Diagonal
# =============================================================================

def ejemplo_diagonal():
    """Pathfinding con movimiento diagonal permitido."""
    from virigir_math_utilities.pathfinding import astar_grid_2d, Manhattan

    def is_walkable(pos):
        x, y = pos
        return 0 <= x < 10 and 0 <= y < 10

    # Con diagonal (más rápido, costo diagonal = sqrt(2))
    result = astar_grid_2d(
        start=(0, 0),
        goal=(7, 7),
        is_walkable_fn=is_walkable,
        heuristic=Manhattan(),
        allow_diagonal=True,
        diagonal_cost=1.414213562
    )

    print(f"✓ Con diagonal: {result.path_length} pasos, costo {result.cost:.2f}")


# =============================================================================
# EJEMPLO 4: Weighted A* (Más Rápido)
# =============================================================================

def ejemplo_weighted():
    """Weighted A* sacrifica optimalidad por velocidad."""
    from virigir_math_utilities.pathfinding import astar_grid_2d, Manhattan

    def is_walkable(pos):
        x, y = pos
        return 0 <= x < 50 and 0 <= y < 50

    # A* normal
    result_normal = astar_grid_2d(
        (0, 0), (49, 49), is_walkable, Manhattan(weight=1.0)
    )

    # Weighted A* (explora menos nodos)
    result_weighted = astar_grid_2d(
        (0, 0), (49, 49), is_walkable, Manhattan(weight=2.0)
    )

    print(f"Normal: {result_normal.nodes_explored} nodos")
    print(f"Weighted: {result_weighted.nodes_explored} nodos")
    print(f"Reducción: {100*(1-result_weighted.nodes_explored/result_normal.nodes_explored):.1f}%")


# =============================================================================
# EJEMPLO 5: Pathfinding Genérico (No Grid)
# =============================================================================

def ejemplo_grafo_generico():
    """A* en un grafo arbitrario (no grid)."""
    from virigir_math_utilities.pathfinding import astar, Manhattan

    # Grafo de ejemplo (nodos conectados)
    graph = {
        'A': [('B', 1), ('C', 4)],
        'B': [('A', 1), ('C', 2), ('D', 5)],
        'C': [('A', 4), ('B', 2), ('D', 1)],
        'D': [('B', 5), ('C', 1)]
    }

    # Posiciones para la heurística
    positions = {
        'A': (0, 0),
        'B': (1, 0),
        'C': (2, 0),
        'D': (3, 0)
    }

    def get_neighbors(node):
        return [neighbor for neighbor, cost in graph[node]]

    def cost(from_node, to_node):
        for neighbor, c in graph[from_node]:
            if neighbor == to_node:
                return c
        return float('inf')

    class GraphHeuristic:
        def calculate(self, from_node, to_node):
            return abs(positions[from_node][0] - positions[to_node][0])

    result = astar('A', 'D', get_neighbors, cost, GraphHeuristic())

    if result.found:
        print(f"✓ Path en grafo: {result.path}")


# =============================================================================
# EJEMPLO 6: Con Callbacks para Visualización
# =============================================================================

def ejemplo_con_callbacks():
    """A* con callbacks para tracking o visualización."""
    from virigir_math_utilities.pathfinding import astar_with_callbacks, Manhattan

    explored = []
    added = []

    def on_explored(node):
        explored.append(node)

    def on_added(node, f_score):
        added.append((node, f_score))

    def get_neighbors(pos):
        x, y = pos
        return [(x+1,y), (x-1,y), (x,y+1), (x,y-1)]

    def cost(from_pos, to_pos):
        return 1.0

    result = astar_with_callbacks(
        start=(0, 0),
        goal=(3, 3),
        neighbors_fn=get_neighbors,
        cost_fn=cost,
        heuristic=Manhattan(),
        on_node_explored=on_explored,
        on_node_added=on_added
    )

    print(f"✓ Nodos explorados: {len(explored)}")
    print(f"  Nodos agregados: {len(added)}")


# =============================================================================
# EJEMPLO 7: Límite de Iteraciones
# =============================================================================

def ejemplo_con_limite():
    """Limitar búsqueda para prevenir timeouts."""
    from virigir_math_utilities.pathfinding import astar_grid_2d, Manhattan

    def is_walkable(pos):
        x, y = pos
        return 0 <= x < 1000 and 0 <= y < 1000

    # Limitar a 100 iteraciones
    result = astar_grid_2d(
        start=(0, 0),
        goal=(999, 999),
        is_walkable_fn=is_walkable,
        heuristic=Manhattan(),
        max_iterations=100
    )

    if not result.found:
        print(f"✗ Búsqueda interrumpida tras explorar {result.nodes_explored} nodos")


# =============================================================================
# EJEMPLO 8: Usando el PathResult
# =============================================================================

def ejemplo_path_result():
    """Trabajar con el objeto PathResult."""
    from virigir_math_utilities.pathfinding import astar_grid_2d, Manhattan

    def is_walkable(pos):
        x, y = pos
        return 0 <= x < 10 and 0 <= y < 10

    result = astar_grid_2d((0, 0), (5, 5), is_walkable, Manhattan())

    # Verificar si se encontró path
    if result.found:  # o: if result:
        print(f"✓ Path encontrado!")

        # Acceder a propiedades
        print(f"  Longitud: {result.path_length} pasos")
        print(f"  Costo: {result.cost}")
        print(f"  Nodos explorados: {result.nodes_explored}")

        # Iterar sobre el path
        for i, pos in enumerate(result.path):
            print(f"    Paso {i}: {pos}")

        # Obtener primer y último paso
        start_pos = result.path[0]
        end_pos = result.path[-1]
        print(f"  De {start_pos} a {end_pos}")
    else:
        print(f"✗ No se encontró path")
        print(f"  Explorados: {result.nodes_explored} nodos")


# =============================================================================
# EJEMPLO 9: Heurística Manhattan en Diferentes Dimensiones
# =============================================================================

def ejemplo_manhattan_dimensiones():
    """Usar Manhattan en 2D y 3D."""
    from virigir_math_utilities.pathfinding.heuristics import Manhattan

    h = Manhattan()

    # 2D
    dist_2d = h.calculate((0, 0), (3, 4))
    print(f"Manhattan 2D: {dist_2d}")  # 7.0

    # 3D
    dist_3d = h.calculate((0, 0, 0), (3, 4, 5))
    print(f"Manhattan 3D: {dist_3d}")  # 12.0

    # Weighted
    hw = Manhattan(weight=2.0)
    dist_weighted = hw.calculate((0, 0), (3, 4))
    print(f"Weighted Manhattan: {dist_weighted}")  # 14.0

    # Como callable
    dist = h((0, 0), (5, 5))
    print(f"Como callable: {dist}")  # 10.0


# =============================================================================
# EJEMPLO 10: Maze Solver Completo
# =============================================================================

def ejemplo_maze_completo():
    """Resolver un maze completo con visualización."""
    from virigir_math_utilities.pathfinding import astar_grid_2d, Manhattan

    # Definir maze (0=walkable, 1=wall)
    maze = [
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ]

    height = len(maze)
    width = len(maze[0])

    def is_walkable(pos):
        x, y = pos
        if 0 <= x < width and 0 <= y < height:
            return maze[y][x] == 0
        return False

    # Resolver
    start = (0, 0)
    goal = (6, 4)
    result = astar_grid_2d(start, goal, is_walkable, Manhattan())

    # Visualizar
    if result.found:
        print("Maze resuelto:")
        for y in range(height):
            for x in range(width):
                pos = (x, y)
                if pos == start:
                    print("S", end=" ")
                elif pos == goal:
                    print("G", end=" ")
                elif pos in result.path:
                    print("*", end=" ")
                elif maze[y][x] == 1:
                    print("#", end=" ")
                else:
                    print(".", end=" ")
            print()


# =============================================================================
# Ejecutar todos los ejemplos
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EJEMPLOS DE USO - PATHFINDING A*")
    print("="*70)

    ejemplos = [
        ("Básico", ejemplo_basico),
        ("Con Obstáculos", ejemplo_con_obstaculos),
        ("Movimiento Diagonal", ejemplo_diagonal),
        ("Weighted A*", ejemplo_weighted),
        ("Grafo Genérico", ejemplo_grafo_generico),
        ("Con Callbacks", ejemplo_con_callbacks),
        ("Límite de Iteraciones", ejemplo_con_limite),
        ("PathResult", ejemplo_path_result),
        ("Manhattan Dimensiones", ejemplo_manhattan_dimensiones),
        ("Maze Solver", ejemplo_maze_completo),
    ]

    for nombre, funcion in ejemplos:
        print(f"\n{'─'*70}")
        print(f"EJEMPLO: {nombre}")
        print(f"{'─'*70}")
        try:
            funcion()
        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\n{'='*70}")
    print("Todos los ejemplos ejecutados!")
    print(f"{'='*70}")

