#!/usr/bin/env python3
"""
Script básico de prueba de Pygame.

Este script demuestra las funcionalidades básicas de pygame:
- Crear una ventana
- Loop de juego básico
- Manejo de eventos (cerrar ventana, teclas)
- Dibujar formas y colores
- Control de FPS
- Uso de la clase Color de vgNoise
"""

import sys
import pygame

# Importar nuestras clases
from core.color.color import Color
from core.tilemap.tilemap import TileMap
from core.tilemap.tileset import TileSet
from core.camera.camera import Camera


# Constantes
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
FPS = 60

# Colores usando nuestra clase Color
BLACK = Color(0, 0, 0)
WHITE = Color(255, 255, 255)
RED = Color(255, 0, 0)
GREEN = Color(0, 255, 0)
BLUE = Color(0, 0, 255)
YELLOW = Color(255, 255, 0)
CYAN = Color(0, 255, 255)
MAGENTA = Color(255, 0, 255)


class BaseGameApp:
    """Aplicación de prueba básica de Pygame."""

    def __init__(self):
        """Inicializa Pygame y crea la ventana."""
        # Inicializar Pygame
        self.tileset = None
        self.tilemap = None
        pygame.init()

        # Crear ventana
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("TileMap Render Test - vgNoise")

        # Reloj para controlar FPS
        self.clock = pygame.time.Clock()

        # Estado
        self.running = True
        self.frame_count = 0

        # Posición del cuadrado que se mueve
        self.square_x = 100
        self.square_y = 100
        self.square_speed = 5

        # Crear cámara
        self.camera = Camera(
            x=0, y=0,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT,
            zoom=1.0
        )
        self.camera_speed = 10  # Aumentar velocidad para que sea más visible

        # Crear tilemap de prueba con tileset grayscale
        self.setup_tilemap()

        # Configurar límites de cámara basados en el tilemap
        self.camera.set_bounds_from_tilemap(
            self.tilemap.width,
            self.tilemap.height,
            self.tilemap.tile_width,
            self.tilemap.tile_height
        )

        print("✓ Pygame inicializado correctamente")
        print(f"✓ Ventana creada: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        print(f"✓ Cámara: {self.camera}")

    def setup_tilemap(self):
        """Configura el tilemap de prueba con tileset grayscale."""
        # Generar tileset grayscale de 32 colores
        print("Generando tileset grayscale de 32 colores...")
        self.tileset = TileSet.generate_grayscale_tileset(
            nsteps=32,
            tile_width=32,
            tile_height=32,
            columns=8,
            output_path="grayscale_tileset_32.png"
        )
        print(f"✓ Tileset generado: {self.tileset}")

        # Crear tilemap de 50x40 tiles (más grande que la ventana para permitir movimiento)
        self.tilemap = TileMap(
            width=50,
            height=40,
            tile_width=32,
            tile_height=32,
            num_layers=1
        )

        # Registrar tileset en el tilemap
        self.tilemap.add_tileset(0, self.tileset)

        # Llenar el tilemap con un patrón de prueba
        for y in range(self.tilemap.height):
            for x in range(self.tilemap.width):
                # Crear un patrón interesante usando los tiles
                # Patrón de tablero de ajedrez con gradientes
                if (x + y) % 2 == 0:
                    tile_id = (x + y) % 32
                else:
                    tile_id = (31 - ((x + y) % 32))

                # Correcto: set_tile(x, y, tile_id, tileset_id, layer)
                self.tilemap.set_tile(x, y, tile_id, tileset_id=0, layer=0)

        print(f"✓ Tilemap creado: {self.tilemap.width}x{self.tilemap.height} tiles")

    def render_tilemap(self):
        """
        Renderiza el tilemap en la pantalla usando la cámara.
        """
        if not self.tilemap or not self.tileset:
            return

        start_tile_x, start_tile_y, end_tile_x, end_tile_y = self.camera.get_visible_tiles(
            self.tilemap.tile_width,
            self.tilemap.tile_height,
            self.tilemap.width,
            self.tilemap.height
        )

        # Renderizar cada capa
        for layer_idx in range(self.tilemap.num_layers):
            layer = self.tilemap.layers[layer_idx]

            # Renderizar solo los tiles visibles
            for tile_y in range(start_tile_y, end_tile_y):
                for tile_x in range(start_tile_x, end_tile_x):
                    cell = layer.get_tile(tile_x, tile_y)
                    if cell and not cell.is_empty:
                        tileset_id = cell.tileset_id
                        tile_id = cell.tile_id

                        # Obtener tileset
                        tileset = self.tilemap.tilesets.get(tileset_id)
                        if tileset and tileset.surface:
                            # Obtener surface del tile
                            tile_surface = tileset.get_tile_surface(tile_id)
                            if tile_surface:
                                # Calcular posición en mundo
                                world_x = tile_x * self.tilemap.tile_width
                                world_y = tile_y * self.tilemap.tile_height

                                # Convertir a coordenadas de pantalla usando la cámara
                                screen_x, screen_y = self.camera.world_to_screen(world_x, world_y)

                                # Aplicar zoom al tile si es necesario
                                if self.camera.zoom != 1.0:
                                    # Escalar el tile según el zoom
                                    scaled_width = int(self.tilemap.tile_width * self.camera.zoom)
                                    scaled_height = int(self.tilemap.tile_height * self.camera.zoom)
                                    scaled_tile = pygame.transform.scale(
                                        tile_surface,
                                        (scaled_width, scaled_height)
                                    )
                                    self.screen.blit(scaled_tile, (int(screen_x), int(screen_y)))
                                else:
                                    self.screen.blit(tile_surface, (int(screen_x), int(screen_y)))

    def handle_events(self):
        """Maneja eventos de teclado y ratón."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                print("✓ Ventana cerrada por el usuario")

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    print("✓ ESC presionado - cerrando")
                elif event.key == pygame.K_SPACE:
                    print(f"✓ Cámara: {self.camera}")
                elif event.key == pygame.K_r:
                    # Reset camera
                    self.camera.set_position(0, 0)
                    self.camera.zoom = 1.0
                    print("✓ Cámara reseteada")

            elif event.type == pygame.MOUSEWHEEL:
                # Zoom con rueda del ratón
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if event.y > 0:  # Scroll up = zoom in
                    self.camera.zoom_at_point(mouse_x, mouse_y, 1.1)
                elif event.y < 0:  # Scroll down = zoom out
                    self.camera.zoom_at_point(mouse_x, mouse_y, 0.9)

        # Movimiento continuo con teclas (cámara)
        keys = pygame.key.get_pressed()

        # Movimiento de cámara con flechas o WASD
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.camera.move(-self.camera_speed, 0)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.camera.move(self.camera_speed, 0)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.camera.move(0, -self.camera_speed)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.camera.move(0, self.camera_speed)

        # Zoom con + y -
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            self.camera.zoom_in(1.02)
        if keys[pygame.K_MINUS]:
            self.camera.zoom_out(1.02)

    def update(self):
        """Actualiza la lógica del juego."""
        self.frame_count += 1

        # Actualizar cámara (para smooth movement)
        self.camera.update()

    def draw(self):
        """Dibuja toda la info en la pantalla."""
        # Limpiar pantalla con color negro
        self.screen.fill(BLACK.to_rgb())

        # Renderizar tilemap con cámara
        self.render_tilemap()

        # Dibujar título
        font_large = pygame.font.Font(None, 48)
        title = font_large.render("TileMap + Camera Test", True, WHITE.to_rgb())
        self.screen.blit(title, (WINDOW_WIDTH // 2 - title.get_width() // 2, 10))

        # Dibujar info
        font_small = pygame.font.Font(None, 18)
        active_chunks = len(self.tilemap.get_active_chunks(layer=0))
        info_texts = [
            f"TileMap: {self.tilemap.width}x{self.tilemap.height} tiles",
            f"TileSet: {self.tileset.columns}x{self.tileset.rows} (32 colors)",
            f"Chunk size: {self.tilemap.chunk_size}x{self.tilemap.chunk_size}",
            f"Active chunks: {active_chunks}",
            f"Camera: ({self.camera.x:.0f}, {self.camera.y:.0f})",
            f"Zoom: {self.camera.zoom:.2f}x",
            f"FPS: {int(self.clock.get_fps())}",
            "",
            "Controls:",
            "WASD/Arrows: Move camera",
            "Mouse wheel: Zoom",
            "+/- : Zoom in/out",
            "R: Reset camera",
            "ESC: Exit",
        ]

        y_offset = WINDOW_HEIGHT - 270
        for text in info_texts:
            surface = font_small.render(text, True, CYAN.to_rgb())
            self.screen.blit(surface, (10, y_offset))
            y_offset += 20

        # Actualizar pantalla
        pygame.display.flip()

    def run(self):
        """Loop principal del juego."""
        print("\n" + "=" * 60)
        print("TILEMAP + CAMERA TEST - LOOP INICIADO")
        print("=" * 60)
        print("Controles:")
        print("  - WASD o Flechas: Mover cámara")
        print("  - Rueda del ratón: Zoom (centrado en cursor)")
        print("  - +/- : Zoom in/out")
        print("  - R: Reset cámara")
        print("  - SPACE: Info de cámara en consola")
        print("  - ESC o X: Cerrar ventana")
        print("=" * 60 + "\n")

        while self.running:
            # Manejar eventos
            self.handle_events()

            # Actualizar lógica
            self.update()

            # Dibujar
            self.draw()

            # Controlar FPS
            self.clock.tick(FPS)

        # Cerrar Pygame
        pygame.quit()
        print("\n✓ Pygame cerrado correctamente")

        # Limpiar archivo de tileset temporal
        self._cleanup()

    def _cleanup(self):
        """Limpia archivos temporales."""
        import os
        from pathlib import Path

        if hasattr(self, 'tileset') and self.tileset.image_path:
            tileset_path = Path(self.tileset.image_path)
            if tileset_path.exists():
                try:
                    os.remove(tileset_path)
                    print(f"✓ Archivo temporal eliminado: {tileset_path.name}")
                except:
                    pass

    def __del__(self):
        """Asegura que Pygame se cierre correctamente."""
        try:
            pygame.quit()
        except:
            pass


def main():
    """Función principal."""
    print("=" * 60)
    print("PYGAME TEST - vgNoise Project")
    print("=" * 60)
    print(f"Pygame version: {pygame.version.ver}")
    print(f"SDL version: {'.'.join(map(str, pygame.get_sdl_version()))}")
    print("=" * 60)
    print()

    try:
        # Crear y ejecutar la aplicación
        app = BaseGameApp()
        app.run()

        print("\n" + "=" * 60)
        print("✓ TEST COMPLETADO EXITOSAMENTE")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

