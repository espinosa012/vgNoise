"""Camera class for 2D tilemap rendering with zoom and movement support."""

from typing import Tuple, Optional


class Camera:
    """2D Camera for tilemap rendering with smooth movement and zoom."""

    def __init__(self, x: float = 0.0, y: float = 0.0, width: int = 800, height: int = 600,
                 zoom: float = 1.0, min_zoom: float = 0.25, max_zoom: float = 4.0) -> None:
        """
        Initialize a 2D camera for tilemap rendering.

        Args:
            x: Initial x position of the camera in world units (default: 0.0).
            y: Initial y position of the camera in world units (default: 0.0).
            width: Width of the viewport in pixels (default: 800).
            height: Height of the viewport in pixels (default: 600).
            zoom: Initial zoom level (default: 1.0). Values > 1.0 zoom in, < 1.0 zoom out.
            min_zoom: Minimum allowed zoom level (default: 0.25).
            max_zoom: Maximum allowed zoom level (default: 4.0).
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self._zoom = zoom
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self._target_x: Optional[float] = None
        self._target_y: Optional[float] = None
        self._smoothing = 0.1
        self._min_x: Optional[float] = None
        self._max_x: Optional[float] = None
        self._min_y: Optional[float] = None
        self._max_y: Optional[float] = None

    @property
    def zoom(self) -> float:
        """
        Get the current zoom level.

        Returns:
            Current zoom level as a float.
        """
        return self._zoom

    @zoom.setter
    def zoom(self, value: float) -> None:
        """
        Set the zoom level, clamped between min_zoom and max_zoom.

        Args:
            value: Desired zoom level. Will be clamped to [min_zoom, max_zoom].
        """
        self._zoom = max(self.min_zoom, min(self.max_zoom, value))

    def move(self, dx: float, dy: float) -> None:
        """
        Move the camera by the specified delta.

        Args:
            dx: Delta x to move the camera (in world units).
            dy: Delta y to move the camera (in world units).
        """
        self.x += dx
        self.y += dy
        self._apply_bounds()

    def set_position(self, x: float, y: float) -> None:
        """
        Set the camera position directly.

        Args:
            x: New x position (in world units).
            y: New y position (in world units).
        """
        self.x = x
        self.y = y
        self._apply_bounds()

    def set_target(self, x: float, y: float) -> None:
        """
        Set a target position for smooth camera movement.

        The camera will smoothly move towards this target when update() is called.

        Args:
            x: Target x position (in world units).
            y: Target y position (in world units).
        """
        self._target_x = x
        self._target_y = y

    def set_smoothing(self, smoothing: float) -> None:
        """
        Set the smoothing factor for camera movement.

        Args:
            smoothing: Smoothing factor (0.0 to 1.0). Higher values = faster movement.
                      0.0 = no movement, 1.0 = instant movement.
        """
        self._smoothing = max(0.0, min(1.0, smoothing))

    def update(self) -> None:
        """
        Update the camera position for smooth movement.

        This should be called every frame if using smooth camera movement via set_target().
        The camera will gradually move towards the target position based on the smoothing factor.
        """
        if self._target_x is not None and self._target_y is not None:
            self.x += (self._target_x - self.x) * self._smoothing
            self.y += (self._target_y - self.y) * self._smoothing
            distance_sq = (self._target_x - self.x) ** 2 + (self._target_y - self.y) ** 2
            if distance_sq < 1.0:
                self.x = self._target_x
                self.y = self._target_y
                self._target_x = None
                self._target_y = None
            self._apply_bounds()

    def zoom_in(self, factor: float = 1.1) -> None:
        """
        Zoom in by the specified factor.

        Args:
            factor: Zoom factor (default: 1.1). Values > 1.0 zoom in.
        """
        self.zoom *= factor

    def zoom_out(self, factor: float = 1.1) -> None:
        """
        Zoom out by the specified factor.

        Args:
            factor: Zoom factor (default: 1.1). Values > 1.0 zoom out.
        """
        self.zoom /= factor

    def zoom_at_point(self, screen_x: int, screen_y: int, factor: float) -> None:
        """
        Zoom in/out while keeping a specific screen point stationary.

        This is useful for mouse wheel zoom where you want to zoom towards the cursor position.

        Args:
            screen_x: Screen x coordinate to zoom towards.
            screen_y: Screen y coordinate to zoom towards.
            factor: Zoom factor. Values > 1.0 zoom in, < 1.0 zoom out.
        """
        world_x, world_y = self.screen_to_world(screen_x, screen_y)
        old_zoom = self._zoom
        self.zoom *= factor
        if self._zoom != old_zoom:
            zoom_ratio = self._zoom / old_zoom
            self.x += world_x * (1 - zoom_ratio)
            self.y += world_y * (1 - zoom_ratio)
            self._apply_bounds()

    def set_bounds(self, min_x: Optional[float] = None, max_x: Optional[float] = None,
                   min_y: Optional[float] = None, max_y: Optional[float] = None) -> None:
        """
        Set camera movement boundaries.

        The camera position will be clamped to these bounds.

        Args:
            min_x: Minimum x position (None = no limit).
            max_x: Maximum x position (None = no limit).
            min_y: Minimum y position (None = no limit).
            max_y: Maximum y position (None = no limit).
        """
        self._min_x = min_x
        self._max_x = max_x
        self._min_y = min_y
        self._max_y = max_y
        self._apply_bounds()

    def set_bounds_from_tilemap(self, tilemap_width: int, tilemap_height: int,
                                tile_width: int, tile_height: int) -> None:
        """
        Set camera boundaries based on tilemap dimensions.

        This ensures the camera cannot move outside the tilemap area.

        Args:
            tilemap_width: Width of the tilemap in tiles.
            tilemap_height: Height of the tilemap in tiles.
            tile_width: Width of each tile in pixels.
            tile_height: Height of each tile in pixels.
        """
        world_width = tilemap_width * tile_width
        world_height = tilemap_height * tile_height
        self.set_bounds(min_x=0.0, max_x=max(0.0, world_width - self.width / self._zoom),
                       min_y=0.0, max_y=max(0.0, world_height - self.height / self._zoom))

    def _apply_bounds(self) -> None:
        """
        Apply camera position boundaries (internal method).

        Clamps the camera position to the set bounds.
        """
        if self._min_x is not None:
            self.x = max(self._min_x, self.x)
        if self._max_x is not None:
            self.x = min(self._max_x, self.x)
        if self._min_y is not None:
            self.y = max(self._min_y, self.y)
        if self._max_y is not None:
            self.y = min(self._max_y, self.y)

    def center_on(self, world_x: float, world_y: float) -> None:
        """
        Center the camera on a specific world position instantly.

        Args:
            world_x: World x coordinate to center on.
            world_y: World y coordinate to center on.
        """
        self.x = world_x - (self.width / 2) / self._zoom
        self.y = world_y - (self.height / 2) / self._zoom
        self._apply_bounds()

    def center_on_smooth(self, world_x: float, world_y: float) -> None:
        """
        Center the camera on a specific world position with smooth movement.

        The camera will smoothly move to center on the target position.

        Args:
            world_x: World x coordinate to center on.
            world_y: World y coordinate to center on.
        """
        target_x = world_x - (self.width / 2) / self._zoom
        target_y = world_y - (self.height / 2) / self._zoom
        self.set_target(target_x, target_y)

    def world_to_screen(self, world_x: float, world_y: float) -> Tuple[float, float]:
        """
        Convert world coordinates to screen coordinates.

        Args:
            world_x: World x coordinate.
            world_y: World y coordinate.

        Returns:
            Tuple of (screen_x, screen_y) coordinates.
        """
        screen_x = (world_x - self.x) * self._zoom
        screen_y = (world_y - self.y) * self._zoom
        return screen_x, screen_y

    def screen_to_world(self, screen_x: float, screen_y: float) -> Tuple[float, float]:
        """
        Convert screen coordinates to world coordinates.

        Args:
            screen_x: Screen x coordinate.
            screen_y: Screen y coordinate.

        Returns:
            Tuple of (world_x, world_y) coordinates.
        """
        world_x = screen_x / self._zoom + self.x
        world_y = screen_y / self._zoom + self.y
        return world_x, world_y

    def get_visible_area(self) -> Tuple[float, float, float, float]:
        """
        Get the visible area in world coordinates.

        Returns:
            Tuple of (min_x, min_y, max_x, max_y) in world units.
        """
        min_x = self.x
        min_y = self.y
        max_x = self.x + self.width / self._zoom
        max_y = self.y + self.height / self._zoom
        return min_x, min_y, max_x, max_y

    def get_visible_tiles(self, tile_width: int, tile_height: int,
                         tilemap_width: int = None, tilemap_height: int = None) -> Tuple[int, int, int, int]:
        """
        Get the range of tiles visible on screen.

        This is useful for culling and only rendering visible tiles.

        Args:
            tile_width: Width of each tile in pixels.
            tile_height: Height of each tile in pixels.
            tilemap_width: Total width of the tilemap in tiles (optional, for clamping).
            tilemap_height: Total height of the tilemap in tiles (optional, for clamping).

        Returns:
            Tuple of (start_tile_x, start_tile_y, end_tile_x, end_tile_y).
            These are tile indices, not pixel coordinates.
        """
        min_x, min_y, max_x, max_y = self.get_visible_area()
        start_tile_x = max(0, int(min_x / tile_width) - 1)
        start_tile_y = max(0, int(min_y / tile_height) - 1)
        end_tile_x = int(max_x / tile_width) + 2
        end_tile_y = int(max_y / tile_height) + 2
        if tilemap_width is not None:
            end_tile_x = min(end_tile_x, tilemap_width)
        if tilemap_height is not None:
            end_tile_y = min(end_tile_y, tilemap_height)
        return start_tile_x, start_tile_y, end_tile_x, end_tile_y

    def is_visible(self, world_x: float, world_y: float, margin: float = 0) -> bool:
        """
        Check if a point is visible on screen.

        Args:
            world_x: World x coordinate of the point.
            world_y: World y coordinate of the point.
            margin: Extra margin around the screen in pixels (default: 0).

        Returns:
            True if the point is visible, False otherwise.
        """
        min_x, min_y, max_x, max_y = self.get_visible_area()
        margin_world = margin / self._zoom
        return (min_x - margin_world <= world_x <= max_x + margin_world and
                min_y - margin_world <= world_y <= max_y + margin_world)

    def is_rect_visible(self, world_x: float, world_y: float, width: float, height: float) -> bool:
        """
        Check if a rectangle is visible on screen.

        Args:
            world_x: World x coordinate of the rectangle's top-left corner.
            world_y: World y coordinate of the rectangle's top-left corner.
            width: Width of the rectangle in world units.
            height: Height of the rectangle in world units.

        Returns:
            True if any part of the rectangle is visible, False otherwise.
        """
        cam_min_x, cam_min_y, cam_max_x, cam_max_y = self.get_visible_area()
        return not (world_x + width < cam_min_x or world_x > cam_max_x or
                   world_y + height < cam_min_y or world_y > cam_max_y)

    def resize_viewport(self, width: int, height: int) -> None:
        """
        Resize the camera viewport.

        This should be called when the window is resized.

        Args:
            width: New viewport width in pixels.
            height: New viewport height in pixels.
        """
        self.width = width
        self.height = height

    def __repr__(self) -> str:
        return f"Camera(pos=({self.x:.1f}, {self.y:.1f}), viewport={self.width}x{self.height}, zoom={self._zoom:.2f})"
