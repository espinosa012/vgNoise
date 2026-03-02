"""
Core game framework module.
Provides base classes for building games with Python.
"""

from core.base.game_object import GameObject
from .character.base import BaseCharacter
from .item.item import BaseItem, ItemType, ItemState
from .health.entity_health import EntityHealth

__all__ = [
    'GameObject',
    'BaseCharacter',
    'BaseItem',
    'ItemType',
    'ItemState',
    'EntityHealth',
]

