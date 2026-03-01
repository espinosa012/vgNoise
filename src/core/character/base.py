from typing import Optional, Tuple
import pygame

from core.base.game_object import GameObject
from core.character.movement_component import MovementComponent
from core.character.shape import CharacterShape, RectShape
from core.color.color import Colors

GridPos = Tuple[int, int]


class BaseCharacter(GameObject):
    """
    Base class for all character entities (player, enemies, NPCs).
    Inherits from GameObject and adds character-specific functionality.

    Grid-based movement is handled via a MovementComponent that uses A*
    pathfinding. Subclasses should override _cell_is_walkable() to define
    their own walkability rules; the world reference is passed in so that
    the character can query it without coupling to the tilemap directly.

    Visual representation is handled by a CharacterShape instance that
    draws the character using pygame primitives. Replace self.shape at
    any time to change the character's appearance.
    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        name: str = None,
        world=None,
        grid_pos: GridPos = (0, 0),
        move_speed: float = 4.0,
        shape: Optional[CharacterShape] = None,
    ):
        """
        Initialize a character.

        Args:
            x:          Initial pixel x position.
            y:          Initial pixel y position.
            name:       Optional name for the character.
            world:      VGWorld instance used for walkability queries.
            grid_pos:   Initial position in grid (cell) coordinates.
            move_speed: Grid-based movement speed in cells per second.
            shape:      Visual representation built from pygame primitives.
                        Defaults to a 32×32 white RectShape if None.
        """
        super().__init__(x, y, name)

        # Character-specific properties
        self.health: float = 100.0
        self.max_health: float = 100.0
        self.speed: float = 100.0  # pixels per second (free movement)

        # Free movement velocity (legacy / non-grid movement)
        self.velocity_x: float = 0.0
        self.velocity_y: float = 0.0

        # State
        self.facing_direction: str = "right"  # "left", "right", "up", "down"
        self.is_moving: bool = False

        # ------------------------------------------------------------------
        # Grid-based movement
        # ------------------------------------------------------------------
        self.world = world
        self.grid_x: int = grid_pos[0]
        self.grid_y: int = grid_pos[1]

        self._movement: Optional[MovementComponent] = None
        if world is not None:
            self._movement = MovementComponent(
                is_walkable_fn=self._cell_is_walkable,
                move_speed=move_speed,
            )

        # ------------------------------------------------------------------
        # Visual shape
        # ------------------------------------------------------------------
        # Holds the primitive-based visual of this character.
        # Replace or mutate self.shape at any time to change appearance.
        self.shape: CharacterShape = shape if shape is not None else RectShape(
            width=32,
            height=32,
            color=Colors.WHITE,
            border_width=1,
        )

    # ------------------------------------------------------------------
    # Grid position
    # ------------------------------------------------------------------

    @property
    def grid_position(self) -> GridPos:
        """Current position in grid (cell) coordinates."""
        return self.grid_x, self.grid_y

    # ------------------------------------------------------------------
    # Walkability — override in subclasses
    # ------------------------------------------------------------------

    def _cell_is_walkable(self, pos: GridPos) -> bool:
        """
        Decide whether the character can walk on the given cell.

        Delegates to world.is_walkable by default. Override in subclasses
        to add character-specific restrictions (e.g. blocking water, lava…).

        Args:
            pos: (x, y) grid position to test.

        Returns:
            True if the cell is passable for this character.
        """
        if self.world is None:
            return True
        return self.world.is_walkable(pos[0], pos[1])

    # ------------------------------------------------------------------
    # Grid-based movement API
    # ------------------------------------------------------------------

    def move_to(self, destination: GridPos) -> bool:
        """
        Request grid-based movement to destination using A* pathfinding.

        Args:
            destination: Target (x, y) grid position.

        Returns:
            True if a valid path was found and movement has started.
        """
        if self._movement is None:
            return False
        return self._movement.request_move_to(self.grid_position, destination)

    def stop_grid_movement(self) -> None:
        """Interrupt grid-based movement immediately."""
        if self._movement is not None:
            self._movement.stop()

    def on_move(self, new_pos: GridPos) -> None:
        """
        Hook called after each grid step.

        Override in subclasses to trigger animations, sounds, etc.

        Args:
            new_pos: The new (x, y) grid position after the step.
        """
        pass

    def _on_step(self, new_pos: GridPos) -> None:
        """Internal callback wired to MovementComponent.update()."""
        self.grid_x, self.grid_y = new_pos
        self.on_move(new_pos)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, delta_time: float):
        """
        Update character state.

        Args:
            delta_time: Time elapsed since last frame in seconds.
        """
        # Free (pixel-space) movement
        if self.velocity_x != 0 or self.velocity_y != 0:
            self.translate(self.velocity_x * delta_time, self.velocity_y * delta_time)
            self.is_moving = True
        else:
            self.is_moving = False

        # Grid-based movement
        if self._movement is not None:
            self._movement.update(delta_time, on_step=self._on_step)
            if self._movement.is_moving:
                self.is_moving = True

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, surface: "pygame.Surface") -> None:
        """
        Draw the character on a pygame surface.

        Delegates to self.shape.draw(), which renders the character using
        pygame primitives. Override in subclasses for additional elements
        (health bars, name tags, debug info…).

        Args:
            surface: The pygame.Surface to draw onto.
        """
        self.shape.draw(surface, self.x, self.y)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def take_damage(self, amount: float):
        """
        Apply damage to the character.

        Args:
            amount: Damage amount.
        """
        self.health = max(0.0, self.health - amount)
        if self.health <= 0:
            self.on_death()

    def heal(self, amount: float):
        """
        Heal the character.

        Args:
            amount: Heal amount.
        """
        self.health = min(self.max_health, self.health + amount)

    # ------------------------------------------------------------------
    # Free movement
    # ------------------------------------------------------------------

    def move(self, direction_x: float, direction_y: float):
        """
        Set movement velocity based on direction.

        Args:
            direction_x: Horizontal direction (-1 to 1).
            direction_y: Vertical direction (-1 to 1).
        """
        self.velocity_x = direction_x * self.speed
        self.velocity_y = direction_y * self.speed

        if direction_x > 0:
            self.facing_direction = "right"
        elif direction_x < 0:
            self.facing_direction = "left"
        elif direction_y > 0:
            self.facing_direction = "down"
        elif direction_y < 0:
            self.facing_direction = "up"

    def stop(self):
        """Stop free movement."""
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.is_moving = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_death(self):
        """
        Called when character health reaches 0.
        Override for custom death behavior.
        """
        self.destroy()

    @property
    def is_alive(self) -> bool:
        """Check if character is alive."""
        return self.health > 0 and not self.destroyed

