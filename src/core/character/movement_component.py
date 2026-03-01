"""
MovementComponent: handles grid-based movement and pathfinding for characters.
"""

from typing import Callable, List, Tuple

from virigir_math_utilities.pathfinding import astar_grid_2d, Manhattan, PathResult

GridPos = Tuple[int, int]


class MovementComponent:
    """
    Handles grid-based movement for a character using A* pathfinding.

    The component receives an is_walkable callable so that neither the
    pathfinding algorithm nor this component needs to know anything about
    the tilemap structure or world rules.

    Attributes:
        move_speed: Steps (cells) per second.
    """

    def __init__(
        self,
        is_walkable_fn: Callable[[GridPos], bool],
        move_speed: float = 4.0,
    ) -> None:
        """
        Initialize the movement component.

        Args:
            is_walkable_fn: Callable that receives a (x, y) tuple and returns
                            True if that cell can be walked on.
            move_speed:     Cells per second at which the character advances
                            along the path.
        """
        self._is_walkable = is_walkable_fn
        self.move_speed = move_speed

        self._path: List[GridPos] = []
        self._is_moving: bool = False
        self._move_timer: float = 0.0

    # ------------------------------------------------------------------
    # Pathfinding
    # ------------------------------------------------------------------

    def request_move_to(self, origin: GridPos, destination: GridPos) -> bool:
        """
        Calculate an A* path from origin to destination and begin movement.

        Args:
            origin:      Current grid position of the character.
            destination: Target grid position.

        Returns:
            True if a valid path was found and movement has started,
            False if no path exists.
        """
        result: PathResult = astar_grid_2d(
            start=origin,
            goal=destination,
            is_walkable_fn=self._is_walkable,
            heuristic=Manhattan(),
        )

        if not result.found:
            return False

        # Drop the origin cell — the character is already there.
        self._path = list(result.path[1:])
        self._is_moving = bool(self._path)
        self._move_timer = 0.0
        return True

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, delta_time: float, on_step: Callable[[GridPos], None]) -> None:
        """
        Advance movement along the cached path.

        Should be called every frame from BaseCharacter.update().

        Args:
            delta_time: Seconds elapsed since the last frame.
            on_step:    Callback invoked with the new GridPos every time
                        the character moves one cell forward.
        """
        if not self._is_moving or not self._path:
            return

        self._move_timer += delta_time

        step_duration = 1.0 / self.move_speed
        while self._move_timer >= step_duration and self._path:
            self._move_timer -= step_duration
            next_pos: GridPos = self._path.pop(0)
            on_step(next_pos)

        if not self._path:
            self._is_moving = False

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_moving(self) -> bool:
        """True while the character is travelling along a path."""
        return self._is_moving

    @property
    def remaining_path(self) -> List[GridPos]:
        """A copy of the pending steps in the current path."""
        return list(self._path)

    def stop(self) -> None:
        """Interrupt movement immediately, discarding the current path."""
        self._path.clear()
        self._is_moving = False
        self._move_timer = 0.0

