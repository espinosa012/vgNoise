"""
Core game framework module.
Provides base classes for building games with Python.
"""

from core.base.game_object import GameObject
from .character.base import BaseCharacter
from .item.item import BaseItem, ItemType, ItemState
from .entity_stats.entity_health import EntityHealth
from .entity_stats.entity_hunger import EntityHunger
from .entity_stats.entity_stamina import EntityStamina
from .entity_stats.entity_stat import EntityStat

__all__ = [
    'GameObject',
    'BaseCharacter',
    'BaseItem',
    'ItemType',
    'ItemState',
    'EntityStat',
    'EntityHealth',
    'EntityHunger',
    'EntityStamina',
]

