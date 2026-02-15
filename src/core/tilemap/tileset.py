"""
TileSet class for managing tile collections for tilemaps.

Uses pygame for image handling and rendering. Compatible with pygame-based
game engines while maintaining compatibility with other systems through
the image_path attribute.
"""

from typing import Optional, Tuple, List
from pathlib import Path

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

# Import Color class for tileset generation
try:
    from ..color.color import Color
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False


class TileSet:
    """
    Simple tileset class for managing tile collections.

    Attributes:
        tile_width: Width of each tile in pixels.
        tile_height: Height of each tile in pixels.
        columns: Number of columns in the tileset.
        rows: Number of rows in the tileset.
        image_path: Path to the tileset image file.
        surface: Pygame surface containing the tileset image.
    """

    def __init__(self, tile_width: int = 32, tile_height: int = 32) -> None:
        """
        Initialize a tileset.

        Args:
            tile_width: Width of each tile in pixels (default: 32).
            tile_height: Height of each tile in pixels (default: 32).
        """
        self.tile_width = tile_width
        self.tile_height = tile_height
        self.columns = 0
        self.rows = 0
        self.image_path: Optional[str] = None
        self.surface: Optional['pygame.Surface'] = None

    def set_grid_size(self, columns: int, rows: int) -> None:
        """
        Set the grid size of the tileset.

        Args:
            columns: Number of columns.
            rows: Number of rows.
        """
        self.columns = columns
        self.rows = rows

    def load_from_image(self, image_path: str) -> None:
        """
        Load tileset from a PNG image file and calculate grid automatically.

        Args:
            image_path: Path to the PNG image file.

        Raises:
            ImportError: If pygame is not installed.
            FileNotFoundError: If the image file does not exist.
            ValueError: If the image dimensions are not divisible by tile size.
        """
        if not HAS_PYGAME:
            raise ImportError("pygame is required to load images. Install with: pip install pygame")

        # Initialize pygame if not already initialized
        if not pygame.get_init():
            pygame.init()

        # Resolve and validate path
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Load image with pygame
        self.surface = pygame.image.load(str(path)).convert_alpha()
        image_width, image_height = self.surface.get_size()

        # Calculate grid size
        if image_width % self.tile_width != 0:
            raise ValueError(
                f"Image width ({image_width}) is not divisible by tile_width ({self.tile_width})"
            )
        if image_height % self.tile_height != 0:
            raise ValueError(
                f"Image height ({image_height}) is not divisible by tile_height ({self.tile_height})"
            )

        self.columns = image_width // self.tile_width
        self.rows = image_height // self.tile_height
        self.image_path = str(path.resolve())

    def get_tile_rect(self, tile_id: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Get the rectangle (x, y, width, height) for a tile in the tileset image.

        Args:
            tile_id: The tile ID.

        Returns:
            Tuple of (x, y, width, height) or None if tile_id is invalid.
        """
        if self.columns == 0 or tile_id < 0:
            return None

        col = tile_id % self.columns
        row = tile_id // self.columns

        x = col * self.tile_width
        y = row * self.tile_height

        return x, y, self.tile_width, self.tile_height

    def get_tile_surface(self, tile_id: int) -> Optional['pygame.Surface']:
        """
        Get a pygame Surface for a specific tile.

        Args:
            tile_id: The tile ID.

        Returns:
            Pygame Surface containing the tile or None if invalid.
        """
        if self.surface is None:
            return None

        rect = self.get_tile_rect(tile_id)
        if rect is None:
            return None

        x, y, w, h = rect
        return self.surface.subsurface((x, y, w, h))

    def __repr__(self) -> str:
        """String representation of the tileset."""
        return f"TileSet(tile_size={self.tile_width}x{self.tile_height}, grid={self.columns}x{self.rows})"

    @staticmethod
    def generate_tileset_from_colors(
        colors: List['Color'],
        tile_width: int = 32,
        tile_height: int = 32,
        columns: int = None,
        output_path: str = None
    ) -> 'TileSet':
        """
        Generate a tileset image from a list of flat colors.

        Args:
            colors: List of Color objects to create tiles from.
            tile_width: Width of each tile in pixels (default: 32).
            tile_height: Height of each tile in pixels (default: 32).
            columns: Number of columns in the tileset. If None, creates a square-ish grid.
            output_path: Path to save the generated tileset image. If None, a temporary file is created.

        Returns:
            TileSet object with the generated tileset loaded.

        Raises:
            ImportError: If pygame or Color class is not available.
            ValueError: If colors list is empty.
        """
        if not HAS_PYGAME:
            raise ImportError("pygame is required to generate images. Install with: pip install pygame")

        if not HAS_COLOR:
            raise ImportError("Color class is required for tileset generation")

        if not colors:
            raise ValueError("Colors list cannot be empty")

        # Initialize pygame if not already initialized
        if not pygame.get_init():
            pygame.init()

        num_tiles = len(colors)

        # Calculate grid dimensions
        if columns is None:
            # Create a roughly square grid
            columns = int(num_tiles ** 0.5) + (1 if num_tiles ** 0.5 % 1 > 0 else 0)

        rows = (num_tiles + columns - 1) // columns  # Ceiling division

        # Create surface
        image_width = columns * tile_width
        image_height = rows * tile_height

        surface = pygame.Surface((image_width, image_height), pygame.SRCALPHA)
        surface.fill((0, 0, 0, 0))  # Transparent background

        # Fill tiles with colors
        for idx, color in enumerate(colors):
            col = idx % columns
            row = idx // columns

            x = col * tile_width
            y = row * tile_height

            # Create a rect with the flat color
            rgba = color.to_rgba()
            pygame.draw.rect(surface, rgba, (x, y, tile_width, tile_height))

        # Save image
        if output_path is None:
            from tempfile import NamedTemporaryFile
            with NamedTemporaryFile(mode='w', suffix='.png', delete=False) as f:
                output_path = f.name

        path = Path(output_path)
        pygame.image.save(surface, str(path))

        # Create and load tileset
        tileset = TileSet(tile_width, tile_height)
        tileset.load_from_image(str(path))

        return tileset

    @staticmethod
    def generate_color_scale_tileset(
        init_color: 'Color',
        final_color: 'Color',
        nsteps: int,
        tile_width: int = 32,
        tile_height: int = 32,
        columns: int = None,
        output_path: str = None
    ) -> 'TileSet':
        """
        Generate a tileset from a color scale using Color.get_color_scale method.

        This is an example method that demonstrates how to use Color.get_color_scale
        to generate a tileset with a gradient of colors.

        Args:
            init_color: Starting color of the scale.
            final_color: Ending color of the scale.
            nsteps: Number of steps in the color scale (must be >= 2).
            tile_width: Width of each tile in pixels (default: 32).
            tile_height: Height of each tile in pixels (default: 32).
            columns: Number of columns in the tileset. If None, creates a square-ish grid.
            output_path: Path to save the generated tileset image. If None, a temporary file is created.

        Returns:
            TileSet object with the generated color scale tileset loaded.

        Raises:
            ImportError: If pygame or Color class is not available.
            ValueError: If nsteps < 2.

        Example:
            >>> from core.color.color import Color, Colors
            >>> from core.tilemap.tileset import TileSet
            >>> # Create a red to blue gradient tileset with 16 steps
            >>> tileset = TileSet.generate_color_scale_tileset(
            ...     Colors.RED,
            ...     Colors.BLUE,
            ...     nsteps=16,
            ...     tile_width=32,
            ...     tile_height=32,
            ...     columns=8,
            ...     output_path="gradient_tileset.png"
            ... )
        """
        if not HAS_COLOR:
            raise ImportError("Color class is required for tileset generation")

        # Use the Color.get_color_scale method to generate the color list
        colors = Color.get_color_scale(init_color, final_color, nsteps)

        # Generate tileset from the colors
        return TileSet.generate_tileset_from_colors(
            colors,
            tile_width=tile_width,
            tile_height=tile_height,
            columns=columns,
            output_path=output_path
        )

    @staticmethod
    def generate_grayscale_tileset(
        nsteps: int = 16,
        tile_width: int = 32,
        tile_height: int = 32,
        columns: int = None,
        output_path: str = None
    ) -> 'TileSet':
        """
        Generate a grayscale tileset from black (0,0,0) to white (255,255,255).

        This method creates a grayscale gradient with the specified number of steps,
        useful for creating grayscale palettes, shadow/light gradients, or testing purposes.

        Args:
            nsteps: Number of grayscale steps (must be >= 2). Common values: 8, 16, 32, 64, 256.
            tile_width: Width of each tile in pixels (default: 32).
            tile_height: Height of each tile in pixels (default: 32).
            columns: Number of columns in the tileset. If None, creates a square-ish grid.
            output_path: Path to save the generated tileset image. If None, a temporary file is created.

        Returns:
            TileSet object with the generated grayscale tileset loaded.

        Raises:
            ImportError: If pygame or Color class is not available.
            ValueError: If nsteps < 2.

        Example:
            >>> from core.tilemap.tileset import TileSet
            >>> # Create a 16-step grayscale tileset
            >>> tileset = TileSet.generate_grayscale_tileset(
            ...     nsteps=16,
            ...     tile_width=32,
            ...     tile_height=32,
            ...     columns=8,
            ...     output_path="grayscale_16.png"
            ... )
            >>>
            >>> # Create a 256-step grayscale tileset
            >>> tileset = TileSet.generate_grayscale_tileset(
            ...     nsteps=256,
            ...     tile_width=16,
            ...     tile_height=16,
            ...     columns=16,
            ...     output_path="grayscale_256.png"
            ... )
        """
        if not HAS_COLOR:
            raise ImportError("Color class is required for tileset generation")

        # Create black and white colors
        black = Color(0, 0, 0)
        white = Color(255, 255, 255)

        # Use the Color.get_color_scale method to generate the grayscale list
        colors = Color.get_color_scale(black, white, nsteps)

        # Generate tileset from the grayscale colors
        return TileSet.generate_tileset_from_colors(
            colors,
            tile_width=tile_width,
            tile_height=tile_height,
            columns=columns,
            output_path=output_path
        )

