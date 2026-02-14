"""
VGTileMap class for managing tilemaps.
"""

from typing import Tuple, List, Optional


class TileMapLayer:
    """
    Represents a single layer in a tilemap.

    Attributes:
        width: Width of the layer in tiles.
        height: Height of the layer in tiles.
        data: 2D list where each cell is (tileset_id, tile_id) or None.
    """

    def __init__(self, width: int, height: int) -> None:
        """
        Initialize a tilemap layer.

        Args:
            width: Width of the layer in tiles.
            height: Height of the layer in tiles.
        """
        self.width = width
        self.height = height
        # Each cell stores (tileset_id, tile_id) or None
        self.data: List[List[Optional[Tuple[int, int]]]] = [
            [None for _ in range(width)] for _ in range(height)
        ]

    def get_tile(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """
        Get the tile at the specified position.

        Args:
            x: X coordinate in tile units.
            y: Y coordinate in tile units.

        Returns:
            Tuple of (tileset_id, tile_id) or None if empty/out of bounds.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y][x]
        return None

    def set_tile(self, x: int, y: int, tileset_id: int, tile_id: int) -> None:
        """
        Set the tile at the specified position.

        Args:
            x: X coordinate in tile units.
            y: Y coordinate in tile units.
            tileset_id: The tileset ID.
            tile_id: The tile ID.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            if tile_id < 0:
                # Negative tile_id means clear the cell
                self.data[y][x] = None
            else:
                self.data[y][x] = (tileset_id, tile_id)

    def clear(self) -> None:
        """Clear all tiles in the layer (set to None)."""
        self.data = [[None for _ in range(self.width)] for _ in range(self.height)]

    def __repr__(self) -> str:
        """String representation of the layer."""
        return f"TileMapLayer(width={self.width}, height={self.height})"


class TileMap:
    """
    Simple tilemap class for managing tile grids with layer support.

    Attributes:
        width: Width of the tilemap in tiles.
        height: Height of the tilemap in tiles.
        tile_width: Width of each tile in pixels.
        tile_height: Height of each tile in pixels.
        layers: List of 2D lists of tile IDs (one per layer).
    """

    def __init__(
        self,
        width: int,
        height: int,
        tile_width: int = 32,
        tile_height: int = 32,
        num_layers: int = 1
    ) -> None:
        """
        Initialize the tilemap.

        Args:
            width: Width of the tilemap in tiles.
            height: Height of the tilemap in tiles.
            tile_width: Width of each tile in pixels (default: 32).
            tile_height: Height of each tile in pixels (default: 32).
            num_layers: Number of layers (default: 1).
        """
        self.width = width
        self.height = height
        self.tile_width = tile_width
        self.tile_height = tile_height
        # Each layer is now a TileMapLayer object
        self.layers: List[TileMapLayer] = []

        # Create initial layers
        for _ in range(num_layers):
            self.layers.append(TileMapLayer(width, height))

    @property
    def data(self) -> List[List[Optional[Tuple[int, int]]]]:
        """
        Get the first layer data (for backward compatibility).

        Returns:
            2D list of tiles from layer 0.
        """
        return self.layers[0].data if self.layers else []

    @property
    def num_layers(self) -> int:
        """Get the number of layers."""
        return len(self.layers)

    def get_tile(self, x: int, y: int, layer: int = 0) -> Optional[Tuple[int, int]]:
        """
        Get the tile at the specified position and layer.

        Args:
            x: X coordinate in tile units.
            y: Y coordinate in tile units.
            layer: Layer index (default: 0).

        Returns:
            Tuple of (tileset_id, tile_id) or None if empty/out of bounds.
        """
        if 0 <= layer < len(self.layers):
            return self.layers[layer].get_tile(x, y)
        return None

    def get_tile_id(self, x: int, y: int, layer: int = 0) -> int:
        """
        Get the tile ID at the specified position (for backward compatibility).

        Args:
            x: X coordinate in tile units.
            y: Y coordinate in tile units.
            layer: Layer index (default: 0).

        Returns:
            The tile ID at the position, or -1 if out of bounds or empty.
        """
        tile = self.get_tile(x, y, layer)
        if tile is not None:
            return tile[1]  # Return tile_id from (tileset_id, tile_id)
        return -1

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
        self.layers.append(TileMapLayer(self.width, self.height))
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
        Clear all tiles in a layer (set to None/empty).

        Args:
            layer: Layer index to clear (default: 0).
        """
        if 0 <= layer < len(self.layers):
            self.layers[layer].clear()

    def get_layer(self, layer: int) -> List[List[Optional[Tuple[int, int]]]]:
        """
        Get a reference to a specific layer data.

        Args:
            layer: Layer index.

        Returns:
            2D list of tiles (tileset_id, tile_id) or None for the layer, or empty list if invalid.
        """
        if 0 <= layer < len(self.layers):
            return self.layers[layer].data
        return []

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
        return (f"VGTileMap(width={self.width}, height={self.height}, "
                f"tile_size=({self.tile_width}x{self.tile_height}), layers={len(self.layers)})")

