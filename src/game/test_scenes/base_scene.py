"""
Base scene class for test scenes.
"""

from typing import Optional
import pygame


class BaseScene:
    """
    Base class for all test scenes.

    Each scene should implement the following methods:
    - setup(): Initialize the scene
    - handle_events(events): Handle pygame events
    - update(dt): Update scene logic
    - draw(screen): Draw the scene
    """

    def __init__(self, name: str, description: str = ""):
        """
        Initialize base scene.

        Args:
            name: Scene name.
            description: Scene description.
        """
        self.name = name
        self.description = description
        self.active = False

    def setup(self, screen_width: int, screen_height: int) -> None:
        """
        Setup/initialize the scene.

        Args:
            screen_width: Screen width in pixels.
            screen_height: Screen height in pixels.
        """
        self.active = True
        print(f"✓ Scene '{self.name}' initialized")

    def cleanup(self) -> None:
        """Cleanup scene resources."""
        self.active = False

    def handle_events(self, events: list) -> None:
        """
        Handle pygame events.

        Args:
            events: List of pygame events.
        """
        pass

    def handle_keys(self, keys) -> None:
        """
        Handle continuous key presses.

        Args:
            keys: pygame.key.get_pressed() result.
        """
        pass

    def update(self, dt: float) -> None:
        """
        Update scene logic.

        Args:
            dt: Delta time in seconds.
        """
        pass

    def draw(self, screen: pygame.Surface) -> None:
        """
        Draw the scene.

        Args:
            screen: Pygame surface to draw on.
        """
        pass

    def get_info_text(self) -> list:
        """
        Get info text to display on screen.

        Returns:
            List of strings to display.
        """
        return [
            f"Scene: {self.name}",
            self.description
        ]

    def __repr__(self) -> str:
        return f"Scene('{self.name}')"

