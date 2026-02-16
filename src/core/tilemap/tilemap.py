"""
VGTileMap class for managing tilemaps with chunk support.
"""

from typing import Tuple, List, Optional, Dict
from .mapcell import MapCell
from .tileset import TileSet


class TileMapChunk:
    """
    Represents a chunk of tiles in a tilemap layer.

    Attributes:
        chunk_x: X coordinate of the chunk in chunk units.
        chunk_y: Y coordinate of the chunk in chunk units.
        chunk_size: Size of the chunk (width and height in tiles).
        data: 2D list of MapCell objects for this chunk.
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

    def __repr__(self) -> str:
        """String representation of the tilemap."""
        total_chunks = sum(len(layer.chunks) for layer in self.layers)
        return (f"TileMap(width={self.width}, height={self.height}, "
                f"tile_size={self.tile_size[0]}x{self.tile_size[1]}, "
                f"layers={len(self.layers)}, chunk_size={self.chunk_size}, "
                f"active_chunks={total_chunks})")

