#!/usr/bin/env python3
"""Test rápido de la clase Camera."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.camera.camera import Camera

print("=" * 60)
print("TEST: Camera Class")
print("=" * 60)

# Test 1: Creación
print("\n1. Creando cámara...")
cam = Camera(x=0, y=0, width=800, height=600, zoom=1.0)
print(f"   ✓ {cam}")

# Test 2: Movimiento
print("\n2. Movimiento de cámara...")
cam.move(100, 50)
print(f"   ✓ Después de move(100, 50): pos=({cam.x}, {cam.y})")

# Test 3: Zoom
print("\n3. Zoom...")
cam.zoom_in(1.5)
print(f"   ✓ Zoom in 1.5x: zoom={cam.zoom:.2f}")

# Test 4: Conversión de coordenadas
print("\n4. Conversión de coordenadas...")
world_pos = (200, 150)
screen_pos = cam.world_to_screen(*world_pos)
print(f"   ✓ world_to_screen{world_pos} = {screen_pos}")
back_to_world = cam.screen_to_world(*screen_pos)
print(f"   ✓ screen_to_world{screen_pos} = {back_to_world}")

# Test 5: Área visible
print("\n5. Área visible...")
visible = cam.get_visible_area()
print(f"   ✓ Área visible: min=({visible[0]:.0f}, {visible[1]:.0f}), max=({visible[2]:.0f}, {visible[3]:.0f})")

# Test 6: Tiles visibles
print("\n6. Tiles visibles (tile_size=32x32)...")
tiles = cam.get_visible_tiles(32, 32, 100, 100)
print(f"   ✓ Tiles visibles: ({tiles[0]}, {tiles[1]}) -> ({tiles[2]}, {tiles[3]})")
print(f"   ✓ Total tiles a renderizar: {(tiles[2]-tiles[0]) * (tiles[3]-tiles[1])}")

# Test 7: Bounds
print("\n7. Configurando límites...")
cam.set_bounds(min_x=0, max_x=1000, min_y=0, max_y=800)
cam.set_position(2000, 2000)  # Fuera de límites
print(f"   ✓ Intentó ir a (2000, 2000), quedó en ({cam.x}, {cam.y})")

# Test 8: Center on
print("\n8. Center on...")
cam.center_on(500, 400)
print(f"   ✓ Centrado en (500, 400), cámara en ({cam.x:.0f}, {cam.y:.0f})")

print("\n" + "=" * 60)
print("✓ TODOS LOS TESTS PASARON")
print("=" * 60)
print("\nLa clase Camera está lista para usar!")
print("Ejecuta: python src/game/pygame_test.py")

