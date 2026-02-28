"""
UI Framework for Pygame.

A complete object-oriented UI component system for building game interfaces.

Main Components:
- Widget: Base class for all UI elements
- UIManager: Central manager for handling UI hierarchy and events

Widgets:
- Label: Text display
- Button: Interactive button
- Panel: Container panel
- ImageWidget: Image display
- Checkbox: Boolean input
- Slider: Numeric range input
- TextInput: Text entry

Containers:
- Container: Base container
- VBox: Vertical layout
- HBox: Horizontal layout
- Grid: Grid layout
- ScrollView: Scrollable container

Usage Example:
    from src.ui import UIManager, Button, Label, VBox

    # Initialize UI manager
    ui = UIManager(screen_width, screen_height)

    # Create UI components
    vbox = VBox(x=10, y=10, spacing=10)
    vbox.add_child(Label(text="Hello, World!"))
    vbox.add_child(Button(text="Click Me").on_click(lambda btn: print("Clicked!")))

    ui.add(vbox)

    # In game loop:
    for event in pygame.event.get():
        ui.handle_event(event)

    ui.update(dt)
    ui.draw(screen)
"""

from .widget import Widget, WidgetState
from .manager import UIManager

# Import widgets
from .widgets import (
    Label,
    Button,
    Panel,
    ImageWidget,
    Checkbox,
    Slider,
    TextInput,
    Dropdown,
    NumericInput,
)

# Import containers
from .containers import (
    Container,
    VBox,
    HBox,
    Grid,
    ScrollView
)

__all__ = [
    # Core
    'Widget',
    'WidgetState',
    'UIManager',

    # Widgets
    'Label',
    'Button',
    'Panel',
    'ImageWidget',
    'Checkbox',
    'Slider',
    'TextInput',
    'Dropdown',
    'NumericInput',

    # Containers
    'Container',
    'VBox',
    'HBox',
    'Grid',
    'ScrollView',
]

__version__ = '1.0.0'

