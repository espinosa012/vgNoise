"""
Test scenes package.

Contains example scenes for testing different features.
"""

from .base_scene import BaseScene
from .tilemap_camera_scene import TilemapCameraScene

# Registry of all available scenes
AVAILABLE_SCENES = [
    TilemapCameraScene,
]


def get_scene_by_index(index: int) -> BaseScene:
    """
    Get a scene by its index.

    Args:
        index: Scene index (0-based).

    Returns:
        Scene instance or None if invalid index.
    """
    if 0 <= index < len(AVAILABLE_SCENES):
        return AVAILABLE_SCENES[index]()
    return None


def get_scene_list() -> list:
    """
    Get list of available scene names.

    Returns:
        List of scene names.
    """
    scenes = []
    for scene_class in AVAILABLE_SCENES:
        scene = scene_class()
        scenes.append(f"{scene.name}: {scene.description}")
    return scenes


__all__ = [
    "BaseScene",
    "TilemapCameraScene",
    "AVAILABLE_SCENES",
    "get_scene_by_index",
    "get_scene_list",
]

