#!/usr/bin/env python3
"""
Script básico de prueba de Pygame con tilemap grande y sistema de chunks.

Este script demuestra:
- Tilemap grande de 2024x2024 tiles
- Sistema de chunks de 256x256 tiles
- Renderizado eficiente basado en chunks visibles
- Cámara con movimiento y zoom
"""

import sys
from pathlib import Path

from game.test_scenes import NoiseEditorScene

# Agregar el directorio src al path para imports
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pygame

# Importar escena
from test_scenes.matrix_viewer_scene import MatrixViewerScene


# Constantes
WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 700
FPS = 60


class BaseGameApp:
    """Aplicación de prueba de Pygame con tilemap grande y chunks."""

    def __init__(self):
        """Inicializa Pygame y crea la ventana."""
        # Inicializar Pygame
        pygame.init()

        # Crear ventana
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Matrix Viewer - vgEngine")

        # Reloj para controlar FPS
        self.clock = pygame.time.Clock()

        # Estado
        self.running = True
        self.frame_count = 0


        # Crear y configurar la escena
        self.scene = NoiseEditorScene()
        self.scene.on_enter()

        print("✓ Pygame inicializado correctamente")
        print(f"✓ Ventana creada: {WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        print(f"✓ Escena cargada: MatrixViewerScene")

    def handle_events(self):
        """Maneja eventos de teclado y ratón."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                self.scene.running = False
                print("✓ Ventana cerrada por el usuario")

            # Pasar eventos a la escena
            self.scene.handle_event(event)

            # Si la escena indica que quiere cerrar, cerramos
            if not  self.scene.running:
                self.running = False

    def update(self):
        """Actualiza la lógica del juego."""
        self.frame_count += 1

        # Calcular delta time
        dt = self.clock.get_time() / 1000.0

        # Actualizar escena
        self.scene.update(dt)

    def draw(self):
        """Dibuja la escena en la pantalla."""
        # Limpiar pantalla y dibujar escena
        self.scene.draw(self.screen)

        # Dibujar FPS
        font = pygame.font.Font(None, 20)
        fps_text = font.render(f"FPS: {int(self.clock.get_fps())}", True, (100, 255, 100))
        self.screen.blit(fps_text, (WINDOW_WIDTH - 80, 10))

        # Actualizar pantalla
        pygame.display.flip()

    def run(self):
        """Loop principal del juego."""
        print("\n" + "=" * 60)
        print("MATRIX VIEWER")
        print("=" * 60)
        print("Enter matrix dimensions and click Generate.")
        print()
        print("Controles (menú):")
        print("  - Tab: Cambiar campo")
        print("  - Enter/Click Generate: Generar matriz")
        print("  - ESC: Salir")
        print()
        print("Controles (visor):")
        print("  - WASD o Flechas: Mover cámara")
        print("  - Q/E: Zoom out/in")
        print("  - ESC: Volver al menú")
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

        # Cleanup de la escena
        self.scene.on_exit()

        # Cerrar Pygame
        pygame.quit()
        print("\n✓ Pygame cerrado correctamente")

    def __del__(self):
        """Asegura que Pygame se cierre correctamente."""
        try:
            pygame.quit()
        except:
            pass


def main():
    """Función principal."""
    print("=" * 60)
    print("PYGAME TEST - Random Terrain Map - vgEngine")
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

