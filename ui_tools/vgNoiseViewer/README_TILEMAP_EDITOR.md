# Tilemap Editor
Editor visual de tilemaps para vgMath.
## Características
- **Gestión de Tilesets**: Carga múltiples tilesets desde imágenes PNG/JPG
- **Edición Visual**: Pinta tiles directamente con el ratón
- **Múltiples Mapas**: Crea tilemaps de diferentes tamaños
- **Interfaz Simple**: UI limpia y modular
## Requisitos
```bash
pip install Pillow
```
## Ejecutar
```bash
cd ui_tools/vgNoiseViewer
python3 run_tilemap_editor.py
```
## Uso
### 1. Cargar un Tileset
1. Click en **"Add"** en el panel de Tilesets
2. Selecciona una imagen PNG o JPG
3. Especifica el tamaño de cada tile (ancho y alto en píxeles)
4. El tileset se cargará mostrando todos los tiles disponibles
### 2. Seleccionar un Tile
- Click en cualquier tile del tileset cargado
- El tile seleccionado se muestra en la parte inferior del panel
### 3. Pintar en el Tilemap
- Con un tile seleccionado, click en el canvas del tilemap
- Click y arrastra para pintar múltiples tiles
- Los tiles se renderizan automáticamente desde el tileset
### 4. Crear Nuevo Mapa
1. Click en **"New Map"** en la toolbar
2. Especifica:
   - Ancho y alto del mapa (en tiles)
   - Tamaño de cada tile (en píxeles)
3. Se creará un nuevo tilemap vacío
### 5. Limpiar Mapa
- Click en **"Clear"** para borrar todos los tiles del mapa actual
## Estructura de la UI
```
┌─────────────────────────────────────────────┐
│ Menu: File | Help                           │
├──────────────┬──────────────────────────────┤
│              │ Toolbar: New Map | Clear     │
│  Tilesets    ├──────────────────────────────┤
│              │                              │
│  [Add]       │                              │
│              │                              │
│  Current:    │      Tilemap Canvas          │
│  [Selector]  │      (Click to paint)        │
│              │                              │
│  Info        │                              │
│              │                              │
│  ┌────────┐  │                              │
│  │        │  │                              │
│  │Tileset │  │                              │
│  │ Grid   │  │                              │
│  │        │  │                              │
│  └────────┘  │                              │
│              │                              │
│  Selected:   │                              │
│  Tile ID     │                              │
│              │                              │
├──────────────┴──────────────────────────────┤
│ Status Bar                                  │
└─────────────────────────────────────────────┘
```
## Componentes
### Módulos
- **`app.py`**: Aplicación principal y coordinación
- **`tileset_panel.py`**: Panel de gestión de tilesets
- **`tilemap_canvas.py`**: Canvas de edición del tilemap
- **`config.py`**: Configuración y constantes
### Clases Principales
#### `TilemapEditor`
Aplicación principal que coordina todos los componentes.
#### `TilesetPanel`
Panel lateral que gestiona:
- Lista de tilesets cargados
- Selector de tileset actual
- Visualización del tileset con grid
- Selección de tiles
#### `TilemapCanvas`
Canvas principal que:
- Renderiza el tilemap completo
- Gestiona el pintado con ratón
- Actualiza tiles en tiempo real
## Formato de Tilesets
Los tilesets deben ser imágenes donde:
- Las dimensiones son múltiplos exactos del tamaño de tile
- Los tiles están organizados en un grid regular
- No hay spacing ni margin entre tiles
Ejemplo: Una imagen de 256x128 con tiles de 32x32 = Grid de 8x4 tiles (32 tiles totales)
## Ejemplos de Uso
### Crear un Mapa Simple

```python
from tilemap import TileMap, TileSet

# Crear tileset
tileset = TileSet(tile_width=32, tile_height=32)
tileset.load_from_image("dungeon_tiles.png")
# Crear mapa
tilemap = TileMap(20, 15, 32, 32)
# Colocar tiles
tilemap.set_tile(0, 0, 1)  # Pared
tilemap.set_tile(1, 0, 0)  # Suelo
```
## Atajos de Teclado
Actualmente no implementados, pero planificados:
- `Ctrl+N`: Nuevo mapa
- `Ctrl+Z`: Deshacer
- `Delete`: Borrar tile seleccionado
## Limitaciones Actuales
- Solo soporta un tileset activo a la vez para renderizado
- No hay sistema de capas
- No hay guardado/carga de mapas
- No hay herramienta de relleno (bucket fill)
- No hay selección rectangular
## Próximas Funcionalidades
- Guardado/carga de tilemaps en formato JSON
- Múltiples capas
- Herramienta de relleno
- Selección y copia de regiones
- Exportación a imagen PNG
- Zoom in/out del canvas
## Notas Técnicas
- Usa PIL/Pillow para manejo de imágenes
- Renderizado optimizado: solo actualiza tiles modificados durante el pintado
- Arquitectura modular permite fácil extensión
## Soporte
Para más información sobre las clases de tilemap, consulta:
- `src/tilemap/tilemap.py` - VGTileMap
- `src/tilemap/tileset.py` - TileSet  
- `src/tilemap/tile.py` - Tile
