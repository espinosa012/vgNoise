"""
VGTileMap class for managing tilemaps with chunk support.
"""

from typing import Tuple, List, Optional, Dict

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    pygame = None  # type: ignore[assignment]
    HAS_PYGAME = False

from .mapcell import MapCell
from .tileset import TileSet
from core.camera.camera import Camera


class TileMapChunk:
    """
    Represents a chunk of tiles in a tilemap layer.

    Attributes:
        chunk_x: X coordinate of the chunk in chunk units.
        chunk_y: Y coordinate of the chunk in chunk units.
        chunk_size: Size of the chunk (width and height in tiles).
        data: 2D list of MapCell objects for this chunk.
        dirty: Whether the chunk surface needs to be re-rendered.
    """

    def __init__(self, chunk_x: int, chunk_y: int, chunk_size: int) -> None:
        """
        Initialize a tilemap chunk.

        Args:
            chunk_x: X coordinate of the chunk in chunk units.
            chunk_y: Y coordinate of the chunk in chunk units.
            chunk_size: Size of the chunk (width and height in tiles).
        """
        self.chunk_x = chunk_x
        self.chunk_y = chunk_y
        self.chunk_size = chunk_size
        self.data: List[List[MapCell]] = [
            [MapCell() for _ in range(chunk_size)] for _ in range(chunk_size)
        ]
        self.dirty: bool = True
        self._surface: Optional['pygame.Surface'] = None

    def get_tile(self, local_x: int, local_y: int) -> Optional[MapCell]:
        """
        Get tile at local position within the chunk.

        Args:
            local_x: X coordinate within chunk (0 to chunk_size-1).
            local_y: Y coordinate within chunk (0 to chunk_size-1).

        Returns:
            MapCell object or None if out of bounds.
        """
        if 0 <= local_x < self.chunk_size and 0 <= local_y < self.chunk_size:
            return self.data[local_y][local_x]
        return None

    def set_tile(self, local_x: int, local_y: int, tileset_id: int, tile_id: int) -> None:
        """
        Set tile at local position within the chunk.

        Args:
            local_x: X coordinate within chunk.
            local_y: Y coordinate within chunk.
            tileset_id: The tileset ID.
            tile_id: The tile ID.
        """
        if 0 <= local_x < self.chunk_size and 0 <= local_y < self.chunk_size:
            if tile_id < 0:
                self.data[local_y][local_x].clear()
            else:
                self.data[local_y][local_x].set(tileset_id, tile_id)
            self.dirty = True
            self._surface = None

    def render_surface(self, tileset: TileSet, tile_w: int, tile_h: int) -> Optional['pygame.Surface']:
        """
        Pre-render this chunk to a single surface. Returns cached surface if not dirty.

        Args:
            tileset: TileSet to use for rendering.
            tile_w: Tile width in pixels.
            tile_h: Tile height in pixels.

        Returns:
            A pygame Surface with all tiles composited, or None if pygame is unavailable.
        """
        if not HAS_PYGAME:
            return None

        if not self.dirty and self._surface is not None:
            return self._surface

        size_px = self.chunk_size * tile_w, self.chunk_size * tile_h
        surf = pygame.Surface(size_px, pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))

        for ly in range(self.chunk_size):
            row = self.data[ly]
            for lx in range(self.chunk_size):
                cell = row[lx]
                if not cell.is_empty:
                    tile_surf = tileset.get_tile_surface(cell.tile_id)
                    if tile_surf:
                        surf.blit(tile_surf, (lx * tile_w, ly * tile_h))

        self._surface = surf
        self.dirty = False
        return surf

    def is_empty(self) -> bool:
        """Check if all cells in the chunk are empty."""
        for row in self.data:
            for cell in row:
                if not cell.is_empty:
                    return False
        return True

    def __repr__(self) -> str:
        return f"TileMapChunk(pos=({self.chunk_x}, {self.chunk_y}), size={self.chunk_size})"


class TileMapLayer:
    """
    Represents a single layer in a tilemap with chunk-based storage.

    Attributes:
        width: Width of the layer in tiles.
        height: Height of the layer in tiles.
        chunk_size: Size of each chunk in tiles (default: 16).
        chunks: Dictionary of chunks, keyed by (chunk_x, chunk_y).
    """

    def __init__(self, width: int, height: int, chunk_size: int = 16) -> None:
        """
        Initialize a tilemap layer with chunk support.

        Args:
            width: Width of the layer in tiles.
            height: Height of the layer in tiles.
            chunk_size: Size of each chunk in tiles (default: 16).
        """
        self.width: int = width
        self.height: int = height
        self.chunk_size: int = chunk_size
        self.chunks: Dict[Tuple[int, int], TileMapChunk] = {}

    def _get_chunk_coords(self, x: int, y: int) -> Tuple[int, int, int, int]:
        """
        Get chunk coordinates and local coordinates for a tile position.

        Args:
            x: X coordinate in tile units.
            y: Y coordinate in tile units.

        Returns:
            Tuple of (chunk_x, chunk_y, local_x, local_y).
        """
        chunk_x = x // self.chunk_size
        chunk_y = y // self.chunk_size
        local_x = x % self.chunk_size
        local_y = y % self.chunk_size
        return chunk_x, chunk_y, local_x, local_y

    def _get_or_create_chunk(self, chunk_x: int, chunk_y: int) -> TileMapChunk:
        """
        Get existing chunk or create a new one.

        Args:
            chunk_x: X coordinate of the chunk.
            chunk_y: Y coordinate of the chunk.

        Returns:
            TileMapChunk object.
        """
        chunk_key = (chunk_x, chunk_y)
        if chunk_key not in self.chunks:
            self.chunks[chunk_key] = TileMapChunk(chunk_x, chunk_y, self.chunk_size)
        return self.chunks[chunk_key]

    def get_tile(self, x: int, y: int) -> Optional[MapCell]:
        """
        Get the MapCell at the specified position.

        Args:
            x: X coordinate in tile units.
            y: Y coordinate in tile units.

        Returns:
            MapCell object or None if out of bounds.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return None

        chunk_x, chunk_y, local_x, local_y = self._get_chunk_coords(x, y)
        chunk_key = (chunk_x, chunk_y)

        # Return empty cell if chunk doesn't exist
        if chunk_key not in self.chunks:
            return MapCell()

        return self.chunks[chunk_key].get_tile(local_x, local_y)

    def set_tile(self, x: int, y: int, tileset_id: int, tile_id: int) -> None:
        """
        Set the tile at the specified position.

        Args:
            x: X coordinate in tile units.
            y: Y coordinate in tile units.
            tileset_id: The tileset ID.
            tile_id: The tile ID.
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            return

        chunk_x, chunk_y, local_x, local_y = self._get_chunk_coords(x, y)
        chunk = self._get_or_create_chunk(chunk_x, chunk_y)
        chunk.set_tile(local_x, local_y, tileset_id, tile_id)

        # Remove chunk if it becomes empty (memory optimization)
        if chunk.is_empty():
            chunk_key = (chunk_x, chunk_y)
            if chunk_key in self.chunks:
                del self.chunks[chunk_key]

    def clear(self) -> None:
        """Clear all tiles in the layer by removing all chunks."""
        self.chunks.clear()

    def get_active_chunks(self) -> List[Tuple[int, int]]:
        """
        Get list of coordinates of all active (non-empty) chunks.

        Returns:
            List of (chunk_x, chunk_y) tuples.
        """
        return list(self.chunks.keys())

    def get_chunk(self, chunk_x: int, chunk_y: int) -> Optional[TileMapChunk]:
        """
        Get a specific chunk.

        Args:
            chunk_x: X coordinate of the chunk.
            chunk_y: Y coordinate of the chunk.

        Returns:
            TileMapChunk or None if chunk doesn't exist.
        """
        return self.chunks.get((chunk_x, chunk_y))

    def __repr__(self) -> str:
        """String representation of the layer."""
        return f"TileMapLayer(width={self.width}, height={self.height}, chunk_size={self.chunk_size}, chunks={len(self.chunks)})"


class TileMap:
    """
    Simple tilemap class for managing tile grids with layer and chunk support.

    Attributes:
        width: Width of the tilemap in tiles.
        height: Height of the tilemap in tiles.
        tile_size: Size of each tile as (width, height) in pixels.
        chunk_size: Size of each chunk in tiles.
        layers: List of TileMapLayer objects.
        tilesets: Dictionary of TileSet objects indexed by tileset_id.
    """

    def __init__(
        self,
        width: int,
        height: int,
        tile_size: Tuple[int, int] = (32, 32),
        num_layers: int = 1,
        chunk_size: int = 16
    ) -> None:
        """
        Initialize the tilemap with chunk support.

        Args:
            width: Width of the tilemap in tiles.
            height: Height of the tilemap in tiles.
            tile_size: Size of each tile as (width, height) in pixels (default: (32, 32)).
            num_layers: Number of layers (default: 1).
            chunk_size: Size of each chunk in tiles (default: 16).
        """
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.chunk_size = chunk_size
        self.layers: List[TileMapLayer] = []
        self.tilesets: Dict[int, TileSet] = {}  # tileset_id -> TileSet

        # Create initial layers with chunk support
        for _ in range(num_layers):
            self.layers.append(TileMapLayer(width, height, chunk_size))

    @property
    def tile_width(self) -> int:
        """Get tile width for backward compatibility."""
        return self.tile_size[0]

    @property
    def tile_height(self) -> int:
        """Get tile height for backward compatibility."""
        return self.tile_size[1]


    @property
    def num_layers(self) -> int:
        """Get the number of layers."""
        return len(self.layers)

    # map cells
    def get_tile(self, x: int, y: int, layer: int = 0) -> Optional[MapCell]:
        """
        Get the MapCell at the specified position and layer.

        Args:
            x: X coordinate in tile units.
            y: Y coordinate in tile units.
            layer: Layer index (default: 0).

        Returns:
            MapCell object or None if out of bounds/invalid layer.
        """
        if 0 <= layer < len(self.layers):
            return self.layers[layer].get_tile(x, y)
        return None

    def set_tile(self, x: int, y: int, tile_id: int, tileset_id: int = 0, layer: int = 0) -> None:
        """
        Set the tile at the specified position and layer.

        Args:
            x: X coordinate in tile units.
            y: Y coordinate in tile units.
            tile_id: The tile ID to set.
            tileset_id: The tileset ID (default: 0).
            layer: Layer index (default: 0).
        """
        if 0 <= layer < len(self.layers):
            self.layers[layer].set_tile(x, y, tileset_id, tile_id)

    # layers
    def add_layer(self) -> int:
        """
        Add a new empty layer on top.

        Returns:
            Index of the new layer.
        """
        self.layers.append(TileMapLayer(self.width, self.height, self.chunk_size))
        return len(self.layers) - 1

    def remove_layer(self, layer: int) -> bool:
        """
        Remove a layer by index.

        Args:
            layer: Layer index to remove.

        Returns:
            True if removed successfully, False otherwise.
        """
        if 0 <= layer < len(self.layers) and len(self.layers) > 1:
            self.layers.pop(layer)
            return True
        return False

    def clear_layer(self, layer: int = 0) -> None:
        """
        Clear all tiles in a layer (removes all chunks).

        Args:
            layer: Layer index to clear (default: 0).
        """
        if 0 <= layer < len(self.layers):
            self.layers[layer].clear()

    def get_active_chunks(self, layer: int = 0) -> List[Tuple[int, int]]:
        """
        Get list of active chunk coordinates for a specific layer.

        Args:
            layer: Layer index (default: 0).

        Returns:
            List of (chunk_x, chunk_y) tuples.
        """
        if 0 <= layer < len(self.layers):
            return self.layers[layer].get_active_chunks()
        return []

    def get_chunk(self, chunk_x: int, chunk_y: int, layer: int = 0) -> Optional[TileMapChunk]:
        """
        Get a specific chunk from a layer.

        Args:
            chunk_x: X coordinate of the chunk.
            chunk_y: Y coordinate of the chunk.
            layer: Layer index (default: 0).

        Returns:
            TileMapChunk or None if not found.
        """
        if 0 <= layer < len(self.layers):
            return self.layers[layer].get_chunk(chunk_x, chunk_y)
        return None

    def get_chunk_size(self) -> int:
        """
        Get the chunk size used by this tilemap.

        Returns:
            Chunk size in tiles.
        """
        return self.chunk_size

    def get_chunks_in_area(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        layer: int = 0
    ) -> List[Tuple[int, int]]:
        """
        Get all chunks that intersect with a given tile area.

        Args:
            start_x: Start X coordinate in tiles.
            start_y: Start Y coordinate in tiles.
            end_x: End X coordinate in tiles.
            end_y: End Y coordinate in tiles.
            layer: Layer index (default: 0).

        Returns:
            List of (chunk_x, chunk_y) tuples that intersect the area.
        """
        if not (0 <= layer < len(self.layers)):
            return []

        # Calculate chunk bounds
        start_chunk_x = start_x // self.chunk_size
        start_chunk_y = start_y // self.chunk_size
        end_chunk_x = end_x // self.chunk_size
        end_chunk_y = end_y // self.chunk_size

        # Get active chunks in the area
        active_chunks = self.layers[layer].get_active_chunks()
        chunks_in_area = []

        for chunk_x, chunk_y in active_chunks:
            if (start_chunk_x <= chunk_x <= end_chunk_x and
                start_chunk_y <= chunk_y <= end_chunk_y):
                chunks_in_area.append((chunk_x, chunk_y))

        return chunks_in_area

    # tilesets
    def add_tileset(self, tileset_id: int, tileset: TileSet) -> None:
        """
        Register a tileset with the tilemap.

        Args:
            tileset_id: Unique identifier for the tileset.
            tileset: TileSet object to register.
        """
        self.tilesets[tileset_id] = tileset

    def remove_tileset(self, tileset_id: int) -> bool:
        """
        Remove a tileset from the tilemap.

        Args:
            tileset_id: ID of the tileset to remove.

        Returns:
            True if removed successfully, False if not found.
        """
        if tileset_id in self.tilesets:
            del self.tilesets[tileset_id]
            return True
        return False

    def get_tileset(self, tileset_id: int) -> Optional[TileSet]:
        """
        Get a tileset by its ID.

        Args:
            tileset_id: ID of the tileset.

        Returns:
            TileSet object or None if not found.
        """
        return self.tilesets.get(tileset_id)

    def has_tileset(self, tileset_id: int) -> bool:
        """
        Check if a tileset is registered.

        Args:
            tileset_id: ID of the tileset.

        Returns:
            True if the tileset exists, False otherwise.
        """
        return tileset_id in self.tilesets

    def clear_tilesets(self) -> None:
        """Clear all registered tilesets."""
        self.tilesets.clear()

    def get_tileset_ids(self) -> List[int]:
        """
        Get a list of all registered tileset IDs.

        Returns:
            List of tileset IDs.
        """
        return list(self.tilesets.keys())

    # size
    def get_size(self) -> Tuple[int, int]:
        """
        Get the size of the tilemap in tiles.

        Returns:
            Tuple of (width, height) in tiles.
        """
        return self.width, self.height

    def get_pixel_size(self) -> Tuple[int, int]:
        """
        Get the size of the tilemap in pixels.

        Returns:
            Tuple of (width, height) in pixels.
        """
        return self.width * self.tile_width, self.height * self.tile_height

    # -------------------------------------------------------------------------
    # Coordinate conversion
    # -------------------------------------------------------------------------

    def tile_to_pixel(self, tile_x: int, tile_y: int) -> Tuple[int, int]:
        """
        Convert grid (tile) coordinates to global pixel coordinates.

        The returned point corresponds to the top-left corner of the tile.

        Args:
            tile_x: X coordinate in tile units.
            tile_y: Y coordinate in tile units.

        Returns:
            Tuple (pixel_x, pixel_y) of the top-left corner of the tile.
        """
        return tile_x * self.tile_width, tile_y * self.tile_height

    def pixel_to_tile(self, pixel_x: int, pixel_y: int) -> Tuple[int, int]:
        """
        Convert global pixel coordinates to grid (tile) coordinates.

        Uses integer division, so any pixel inside a tile maps to that
        tile's grid position. The tile does not need to have any cell
        defined at that position.

        Args:
            pixel_x: X coordinate in pixels.
            pixel_y: Y coordinate in pixels.

        Returns:
            Tuple (tile_x, tile_y) of the grid cell that contains the point.
        """
        return pixel_x // self.tile_width, pixel_y // self.tile_height

    # -------------------------------------------------------------------------
    # Rendering helpers
    # -------------------------------------------------------------------------

    def draw(
        self,
        surface: 'pygame.Surface',
        camera: Camera,
        tileset: Optional[TileSet] = None,
        layer: int = 0,
    ) -> None:
        """
        Efficiently render visible chunks to *surface* using chunk-based rendering.

        Each chunk is pre-rendered into a single surface that is cached until
        a tile inside it changes (dirty flag).  Only visible chunks are drawn,
        and scaling (zoom) is performed once per visible chunk instead of once
        per tile.

        Args:
            surface: Target pygame surface (usually the screen).
            camera: Camera instance for coordinate conversion and visibility.
            tileset: TileSet to use.  Falls back to ``self.tilesets.get(0)`` or
                     the legacy ``self.tileset`` attribute if not provided.
            layer: Layer index to render (default 0).
        """
        if not HAS_PYGAME:
            return

        # Resolve tileset
        if tileset is None:
            tileset = self.tilesets.get(0) or getattr(self, 'tileset', None)
        if tileset is None or not (0 <= layer < len(self.layers)):
            return

        tile_w = self.tile_width
        tile_h = self.tile_height
        cs = self.chunk_size
        zoom = camera.zoom

        chunk_world_w = cs * tile_w
        chunk_world_h = cs * tile_h
        needs_scale = (zoom != 1.0)

        # Determine visible chunk range from the camera
        min_x, min_y, max_x, max_y = camera.get_visible_area()
        start_cx = max(0, int(min_x // chunk_world_w))
        start_cy = max(0, int(min_y // chunk_world_h))
        end_cx = int(max_x // chunk_world_w) + 1
        end_cy = int(max_y // chunk_world_h) + 1

        lyr = self.layers[layer]

        for cy in range(start_cy, end_cy + 1):
            # Compute Y screen bounds for this row of chunks
            sy_top = round(camera.world_to_screen(0, cy * chunk_world_h)[1])
            sy_bot = round(camera.world_to_screen(0, (cy + 1) * chunk_world_h)[1])
            dest_h = sy_bot - sy_top
            if dest_h < 1:
                continue

            for cx in range(start_cx, end_cx + 1):
                chunk = lyr.chunks.get((cx, cy))
                if chunk is None:
                    continue

                # Pre-render the chunk at native resolution (cached if clean)
                base_surf = chunk.render_surface(tileset, tile_w, tile_h)
                if base_surf is None:
                    continue

                # Compute X screen bounds for this column of chunks
                sx_left = round(camera.world_to_screen(cx * chunk_world_w, 0)[0])
                sx_right = round(camera.world_to_screen((cx + 1) * chunk_world_w, 0)[0])
                dest_w = sx_right - sx_left
                if dest_w < 1:
                    continue

                if needs_scale:
                    draw_surf = pygame.transform.scale(base_surf, (dest_w, dest_h))
                else:
                    draw_surf = base_surf

                surface.blit(draw_surf, (sx_left, sy_top))

    def __repr__(self) -> str:
        """String representation of the tilemap."""
        total_chunks = sum(len(layer.chunks) for layer in self.layers)
        return (f"TileMap(width={self.width}, height={self.height}, "
                f"tile_size={self.tile_size[0]}x{self.tile_size[1]}, "
                f"layers={len(self.layers)}, chunk_size={self.chunk_size}, "
                f"active_chunks={total_chunks})")

