"""
Character package for game entities.
Contains base character class and controller system.
"""

from .character import BaseCharacter
from .controller.base_controller import BaseCharacterController

__all__ = [
    'BaseCharacter',
    'BaseCharacterController',
]

