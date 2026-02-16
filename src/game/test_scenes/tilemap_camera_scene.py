"""
Tilemap + Camera test scene.
"""

import pygame
from pathlib import Path

from core.color.color import Color
from core.tilemap.tilemap import TileMap
from core.tilemap.tileset import TileSet
from core.camera.camera import Camera
from .base_scene import BaseScene


class TilemapCameraScene(BaseScene):
    """
    Scene that demonstrates tilemap rendering with camera movement and zoom.
    """

    def __init__(self):
        super().__init__(
            name="Tilemap + Camera",
            description="Navigate tilemap with WASD, zoom with mouse wheel"
        )
        self.tilemap = None
        self.tileset = None
        self.camera = None
        self.camera_speed = 10

        # Colors
        self.bg_color = Color(0, 0, 0)
        self.text_color = Color(0, 255, 255)

    def setup(self, screen_width: int, screen_height: int) -> None:
        """Setup tilemap and camera."""
        super().setup(screen_width, screen_height)

        print("Generating grayscale tileset (32 colors)...")
        self.tileset = TileSet.generate_grayscale_tileset(
            nsteps=32,
            tile_width=32,
            tile_height=32,
            columns=8,
            output_path="grayscale_tileset_32.png"
        )
        print(f"✓ Tileset generated: {self.tileset}")

        # Create tilemap (larger than viewport)
        self.tilemap = TileMap(
            width=50,
            height=40,
            tile_width=32,
            tile_height=32,
            num_layers=1
        )

        self.tilemap.add_tileset(0, self.tileset)

        # Fill with pattern
        for y in range(self.tilemap.height):
            for x in range(self.tilemap.width):
                if (x + y) % 2 == 0:
                    tile_id = (x + y) % 32
                else:
                    tile_id = (31 - ((x + y) % 32))
                self.tilemap.set_tile(x, y, tile_id, tileset_id=0, layer=0)

        print(f"✓ Tilemap created: {self.tilemap.width}x{self.tilemap.height} tiles")

        # Setup camera
        self.camera = Camera(
            x=0, y=0,
            width=screen_width,
            height=screen_height,
            zoom=1.0
        )

        self.camera.set_bounds_from_tilemap(
            self.tilemap.width,
            self.tilemap.height,
            self.tilemap.tile_width,
            self.tilemap.tile_height
        )

        print(f"✓ Camera: {self.camera}")

    def cleanup(self) -> None:
        """Cleanup tileset file."""
        super().cleanup()

        if self.tileset and self.tileset.image_path:
            tileset_path = Path(self.tileset.image_path)
            if tileset_path.exists():
                try:
                    tileset_path.unlink()
                    print(f"✓ Tileset file deleted: {tileset_path.name}")
                except:
                    pass

    def handle_events(self, events: list) -> None:
        """Handle events."""
        for event in events:
            if event.type == pygame.MOUSEWHEEL:
                # Zoom with mouse wheel
                mouse_x, mouse_y = pygame.mouse.get_pos()
                if event.y > 0:  # Scroll up = zoom in
                    self.camera.zoom_at_point(mouse_x, mouse_y, 1.1)
                elif event.y < 0:  # Scroll down = zoom out
                    self.camera.zoom_at_point(mouse_x, mouse_y, 0.9)

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset camera
                    self.camera.set_position(0, 0)
                    self.camera.zoom = 1.0
                    print("✓ Camera reset")
                elif event.key == pygame.K_SPACE:
                    print(f"✓ Camera: {self.camera}")

    def handle_keys(self, keys) -> None:
        """Handle continuous key presses."""
        # Camera movement with WASD or arrows
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.camera.move(-self.camera_speed, 0)
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.camera.move(self.camera_speed, 0)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.camera.move(0, -self.camera_speed)
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.camera.move(0, self.camera_speed)

        # Zoom with + and -
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            self.camera.zoom_in(1.02)
        if keys[pygame.K_MINUS]:
            self.camera.zoom_out(1.02)

    def update(self, dt: float) -> None:
        """Update camera."""
        if self.camera:
            self.camera.update()

    def draw(self, screen: pygame.Surface) -> None:
        """Draw tilemap."""
        # Clear screen
        screen.fill(self.bg_color.to_rgb())

        if not self.tilemap or not self.tileset or not self.camera:
            return

        # Get visible tiles
        start_tile_x, start_tile_y, end_tile_x, end_tile_y = self.camera.get_visible_tiles(
            self.tilemap.tile_width,
            self.tilemap.tile_height,
            self.tilemap.width,
            self.tilemap.height
        )

        # Render visible tiles
        for layer_idx in range(self.tilemap.num_layers):
            layer = self.tilemap.layers[layer_idx]

            for tile_y in range(start_tile_y, end_tile_y):
                for tile_x in range(start_tile_x, end_tile_x):
                    cell = layer.get_tile(tile_x, tile_y)
                    if cell and not cell.is_empty:
                        tileset = self.tilemap.tilesets.get(cell.tileset_id)
                        if tileset and tileset.surface:
                            tile_surface = tileset.get_tile_surface(cell.tile_id)
                            if tile_surface:
                                # World to screen coordinates
                                world_x = tile_x * self.tilemap.tile_width
                                world_y = tile_y * self.tilemap.tile_height
                                screen_x, screen_y = self.camera.world_to_screen(world_x, world_y)

                                # Apply zoom if needed
                                if self.camera.zoom != 1.0:
                                    scaled_width = int(self.tilemap.tile_width * self.camera.zoom)
                                    scaled_height = int(self.tilemap.tile_height * self.camera.zoom)
                                    scaled_tile = pygame.transform.scale(tile_surface, (scaled_width, scaled_height))
                                    screen.blit(scaled_tile, (int(screen_x), int(screen_y)))
                                else:
                                    screen.blit(tile_surface, (int(screen_x), int(screen_y)))

    def get_info_text(self) -> list:
        """Get info text."""
        if not self.tilemap or not self.camera:
            return super().get_info_text()

        active_chunks = len(self.tilemap.get_active_chunks(layer=0))

        return [
            f"Scene: {self.name}",
            f"TileMap: {self.tilemap.width}x{self.tilemap.height} tiles",
            f"Chunk size: {self.tilemap.chunk_size}x{self.tilemap.chunk_size}",
            f"Active chunks: {active_chunks}",
            f"Camera: ({self.camera.x:.0f}, {self.camera.y:.0f})",
            f"Zoom: {self.camera.zoom:.2f}x",
            "",
            "Controls:",
            "WASD/Arrows: Move camera",
            "Mouse wheel: Zoom",
            "+/- : Zoom in/out",
            "R: Reset camera",
            "1-9: Change scene",
            "ESC: Exit",
        ]

