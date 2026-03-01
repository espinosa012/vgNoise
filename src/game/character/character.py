"""
Base character class for all game characters.
"""

from typing import Optional
from core.base.game_object import GameObject


class BaseCharacter(GameObject):
    """
    Base class for all character entities in the game.

    Provides basic character functionality that all character types
    (player, enemies, NPCs) will inherit from.

    Attributes:
        health: Current health points
        max_health: Maximum health points
        speed: Movement speed in pixels per second
        velocity_x: Current horizontal velocity
        velocity_y: Current vertical velocity
        is_moving: Whether the character is currently moving
    """

    def __init__(
        self,
        x: float = 0,
        y: float = 0,
        name: Optional[str] = None
    ):
        """
        Initialize a BaseCharacter.

        Args:
            x: Initial x position
            y: Initial y position
            name: Optional name for the character
        """
        super().__init__(x, y, name)

        # Health
        self.health: float = 100.0
        self.max_health: float = 100.0

        # Movement
        self.speed: float = 100.0
        self.velocity_x: float = 0.0
        self.velocity_y: float = 0.0

        # State
        self.is_moving: bool = False

    # ==================== Properties ====================

    @property
    def is_alive(self) -> bool:
        """Check if the character is alive."""
        return self.health > 0

    @property
    def health_percentage(self) -> float:
        """Get health as a percentage (0.0 to 1.0)."""
        if self.max_health <= 0:
            return 0.0
        return self.health / self.max_health

    # ==================== Lifecycle Methods ====================

    def update(self, delta_time: float):
        """
        Update character state each frame.

        Args:
            delta_time: Time elapsed since last frame in seconds
        """
        self._apply_velocity(delta_time)

    def render(self, renderer):
        """
        Render the character.
        Override this in subclasses for specific rendering.

        Args:
            renderer: Renderer object to draw with
        """
        pass

    # ==================== Movement Methods ====================

    def _apply_velocity(self, delta_time: float):
        """
        Apply current velocity to position.

        Args:
            delta_time: Time elapsed since last frame
        """
        if self.velocity_x != 0 or self.velocity_y != 0:
            self.translate(
                self.velocity_x * delta_time,
                self.velocity_y * delta_time
            )
            self.is_moving = True
        else:
            self.is_moving = False

    def set_velocity(self, vx: float, vy: float):
        """
        Set the character's velocity directly.

        Args:
            vx: Horizontal velocity
            vy: Vertical velocity
        """
        self.velocity_x = vx
        self.velocity_y = vy

    def stop(self):
        """Stop all character movement."""
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.is_moving = False

    # ==================== Health Methods ====================

    def take_damage(self, amount: float):
        """
        Apply damage to the character.

        Args:
            amount: Amount of damage to apply
        """
        self.health = max(0.0, self.health - amount)
        if not self.is_alive:
            self.on_death()

    def heal(self, amount: float):
        """
        Heal the character.

        Args:
            amount: Amount to heal
        """
        self.health = min(self.max_health, self.health + amount)

    def on_death(self):
        """
        Called when the character dies (health reaches 0).
        Override in subclasses for custom death behavior.
        """
        pass

