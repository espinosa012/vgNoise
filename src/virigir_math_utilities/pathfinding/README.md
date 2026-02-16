# Pathfinding Module

Implementación flexible y eficiente del algoritmo A* para pathfinding en Python.

## Características

- ✅ **Algoritmo A* genérico**: Funciona con cualquier tipo de grafo o estructura de datos
- ✅ **Función especializada para grids 2D**: `astar_grid_2d()` para uso simplificado
- ✅ **Heurística Manhattan**: Implementada y probada (2D y 3D)
- ✅ **Sistema extensible de heurísticas**: Preparado para agregar Euclidean, Chebyshev, Octile
- ✅ **Callbacks para visualización**: `astar_with_callbacks()` para debugging
- ✅ **Movimiento diagonal opcional**: Soporta 4 u 8 direcciones
- ✅ **Control de iteraciones**: Límite configurable para prevenir bucles infinitos
- ✅ **Resultados detallados**: PathResult con path, costo y estadísticas

## Estructura del Módulo

```
pathfinding/
├── __init__.py          # API pública del módulo
├── astar.py             # Implementación del algoritmo A*
├── heuristics.py        # Funciones heurísticas
├── node.py              # Estructuras de datos (PriorityNode, PathResult)
└── test_pathfinding.py  # Tests y ejemplos
```

## Uso Básico

### Pathfinding en Grid 2D

```python
from virigir_math_utilities.pathfinding import astar_grid_2d, Manhattan

# Definir función de walkability
def is_walkable(pos):
    x, y = pos
    # Tu lógica aquí (ej: revisar si la celda está libre)
    return 0 <= x < 10 and 0 <= y < 10

# Crear heurística
heuristic = Manhattan()

# Encontrar path
result = astar_grid_2d(
    start=(0, 0),
    goal=(9, 9),
    is_walkable_fn=is_walkable,
    heuristic=heuristic
)

# Usar el resultado
if result.found:
    print(f"Path encontrado con {result.path_length} pasos")
    print(f"Costo total: {result.cost}")
    print(f"Nodos explorados: {result.nodes_explored}")
    for step in result.path:
        print(f"  -> {step}")
else:
    print("No se encontró path")
```

### Pathfinding con Movimiento Diagonal

```python
result = astar_grid_2d(
    start=(0, 0),
    goal=(5, 5),
    is_walkable_fn=is_walkable,
    heuristic=Manhattan(),
    allow_diagonal=True,
    diagonal_cost=1.414213562  # sqrt(2)
)
```

### Pathfinding Genérico (Grafo Arbitrario)

```python
from virigir_math_utilities.pathfinding import astar, Manhattan

# Definir función de vecinos
def get_neighbors(node):
    # Retorna lista de nodos vecinos
    return [...]

# Definir función de costo
def cost(from_node, to_node):
    # Retorna el costo de moverse entre nodos
    return 1.0

# Ejecutar A*
result = astar(
    start=start_node,
    goal=goal_node,
    neighbors_fn=get_neighbors,
    cost_fn=cost,
    heuristic=Manhattan()
)
```

## Heurísticas Disponibles

### Manhattan (Implementada) ✅

Distancia Manhattan (L1 norm). Ideal para grids con movimiento en 4 direcciones.

```python
from virigir_math_utilities.pathfinding import Manhattan

# Heurística estándar
heuristic = Manhattan()
distance = heuristic.calculate((0, 0), (3, 4))  # 7.0

# Heurística ponderada (Weighted A*)
weighted = Manhattan(weight=2.0)
distance = weighted.calculate((0, 0), (3, 4))  # 14.0

# También funciona en 3D
distance_3d = heuristic.calculate((0, 0, 0), (3, 4, 5))  # 12.0
```

**Fórmula:**
- 2D: `|x1 - x2| + |y1 - y2|`
- 3D: `|x1 - x2| + |y1 - y2| + |z1 - z2|`

### Euclidean (Por implementar)

Distancia euclidiana (L2 norm). Ideal para movimiento libre.

**Fórmula:** `sqrt((x1-x2)² + (y1-y2)²)`

### Chebyshev (Por implementar)

Distancia Chebyshev (L∞ norm). Ideal para grids con movimiento diagonal sin costo extra.

**Fórmula:** `max(|x1-x2|, |y1-y2|)`

### Octile (Por implementar)

Distancia octile (diagonal distance). Ideal para grids con movimiento diagonal costoso.

**Fórmula:** `D * (dx + dy) + (D2 - 2*D) * min(dx, dy)`

### Zero

Heurística cero (convierte A* en Dijkstra).

```python
from virigir_math_utilities.pathfinding import Zero

heuristic = Zero()
# Siempre retorna 0, garantiza el path óptimo pero explora más nodos
```

## Ejemplos Avanzados

### Pathfinding con Obstáculos

```python
# Grid con obstáculos
obstacles = {(1, 1), (1, 2), (1, 3), (2, 2)}

def is_walkable(pos):
    x, y = pos
    return (0 <= x < 10 and 
            0 <= y < 10 and 
            pos not in obstacles)

result = astar_grid_2d((0, 0), (5, 5), is_walkable, Manhattan())
```

### Pathfinding con Límite de Iteraciones

```python
# Limitar búsqueda para prevenir timeouts
result = astar_grid_2d(
    start=(0, 0),
    goal=(100, 100),
    is_walkable_fn=is_walkable,
    heuristic=Manhattan(),
    max_iterations=1000  # Máximo 1000 iteraciones
)

if not result.found:
    print("Búsqueda interrumpida por límite de iteraciones")
```

### Pathfinding con Callbacks (Visualización)

```python
from virigir_math_utilities.pathfinding import astar_with_callbacks, Manhattan

explored_nodes = []
added_nodes = []

def on_explored(node):
    explored_nodes.append(node)
    print(f"Explorando: {node}")

def on_added(node, f_score):
    added_nodes.append((node, f_score))

result = astar_with_callbacks(
    start=(0, 0),
    goal=(5, 5),
    neighbors_fn=get_neighbors,
    cost_fn=cost,
    heuristic=Manhattan(),
    on_node_explored=on_explored,
    on_node_added=on_added
)
```

### Weighted A* (Más Rápido, Menos Óptimo)

```python
# A* ponderado explora menos nodos pero puede no encontrar el path óptimo
weighted_heuristic = Manhattan(weight=2.0)

result = astar_grid_2d(
    start=(0, 0),
    goal=(50, 50),
    is_walkable_fn=is_walkable,
    heuristic=weighted_heuristic
)

print(f"Nodos explorados con weighted A*: {result.nodes_explored}")
```

## API Reference

### PathResult

Objeto que contiene el resultado de una búsqueda de pathfinding.

**Atributos:**
- `path: Optional[list[T]]` - Lista de nodos desde start hasta goal (None si no se encontró)
- `cost: Optional[float]` - Costo total del path (None si no se encontró)
- `nodes_explored: int` - Número de nodos explorados durante la búsqueda
- `path_length: int` - Longitud del path (0 si no se encontró)
- `found: bool` - True si se encontró un path

**Ejemplo:**
```python
result = astar_grid_2d(start, goal, is_walkable, Manhattan())

if result.found:
    print(f"Path: {result.path}")
    print(f"Cost: {result.cost}")
    print(f"Length: {result.path_length}")
    print(f"Explored: {result.nodes_explored}")
```

## Complejidad

- **Temporal:** O(b^d) donde b es el branching factor y d la profundidad
- **Espacial:** O(b^d) para almacenar nodos en open y closed sets

## Próximas Implementaciones

1. **Heurísticas adicionales:**
   - [ ] Euclidean distance
   - [ ] Chebyshev distance
   - [ ] Octile distance

2. **Variantes de A*:**
   - [ ] Bidirectional A* (búsqueda desde ambos extremos)
   - [ ] IDA* (Iterative Deepening A*)
   - [ ] Theta* (any-angle pathfinding)
   - [ ] Jump Point Search (JPS) para grids

3. **Optimizaciones:**
   - [ ] Path smoothing
   - [ ] Hierarchical pathfinding
   - [ ] Goal bounding

4. **Utilidades:**
   - [ ] Generador de cost maps
   - [ ] Validador de paths
   - [ ] Visualización de búsqueda

## Contribuir

Para agregar nuevas heurísticas, extender la clase `Heuristic`:

```python
from virigir_math_utilities.pathfinding.heuristics import Heuristic

class MiHeuristica(Heuristic):
    def calculate(self, from_pos, to_pos):
        # Tu implementación aquí
        return estimated_cost
```

## Licencia

[Tu licencia aquí]

