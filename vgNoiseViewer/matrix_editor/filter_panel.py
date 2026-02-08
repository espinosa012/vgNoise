"""
Filter Panel for Matrix Editor App.

This module provides a dynamic filter panel that automatically discovers
and presents all available filters from MatrixFilters class.
"""

import tkinter as tk
from tkinter import ttk
from typing import Callable, Optional, List, Dict, Any, Tuple
import inspect
import sys
from pathlib import Path

# Add parent directory to path to import vgmath
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vgmath.matrix import VGMatrix2D
from vgmath.matrix.filters import MatrixFilters, BlurType, EdgeDetectionType

# Handle both package and direct execution imports
try:
    from ..widgets.matrix_widgets import FilterParameterWidget, Card, ScrollableFrame
    from .config import MatrixThemeColors, FilterParameterConfig
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from widgets.matrix_widgets import FilterParameterWidget, Card, ScrollableFrame
    from matrix_editor.config import MatrixThemeColors, FilterParameterConfig


class FilterInfo:
    """Information about a discovered filter."""

    def __init__(
        self,
        name: str,
        display_name: str,
        method: Callable,
        parameters: List[FilterParameterConfig],
        category: str = "Other"
    ):
        self.name = name
        self.display_name = display_name
        self.method = method
        self.parameters = parameters
        self.category = category


class FilterDiscovery:
    """
    Discovers available filters from MatrixFilters class automatically.

    This allows the UI to update automatically when new filters are added
    to the MatrixFilters class.
    """

    # Map parameter names to their configurations
    PARAM_CONFIGS = {
        'size': FilterParameterConfig(
            name='size', label='Kernel Size', param_type='int',
            default=3, min_value=3, max_value=15, step=2,
            choices=['3', '5', '7', '9', '11', '13', '15']
        ),
        'sigma': FilterParameterConfig(
            name='sigma', label='Sigma', param_type='float',
            default=1.0, min_value=0.1, max_value=10.0, step=0.1
        ),
        'strength': FilterParameterConfig(
            name='strength', label='Strength', param_type='float',
            default=1.0, min_value=0.1, max_value=5.0, step=0.1
        ),
        'amount': FilterParameterConfig(
            name='amount', label='Amount', param_type='float',
            default=1.0, min_value=0.1, max_value=5.0, step=0.1
        ),
        'direction': FilterParameterConfig(
            name='direction', label='Direction', param_type='choice',
            default='horizontal',
            choices=['horizontal', 'vertical', 'diagonal',
                    'north', 'south', 'east', 'west',
                    'northeast', 'northwest', 'southeast', 'southwest']
        ),
        'normalize': FilterParameterConfig(
            name='normalize', label='Normalize', param_type='bool',
            default=False
        ),
    }

    # Category mapping based on method name patterns
    CATEGORY_PATTERNS = {
        'blur': 'Blur',
        'gaussian': 'Blur',
        'motion': 'Blur',
        'box': 'Blur',
        'sharpen': 'Sharpen',
        'unsharp': 'Sharpen',
        'sobel': 'Edge Detection',
        'prewitt': 'Edge Detection',
        'laplacian': 'Edge Detection',
        'edge': 'Edge Detection',
        'ridge': 'Edge Detection',
        'emboss': 'Effects',
        'high_pass': 'Frequency',
        'low_pass': 'Frequency',
        'identity': 'Utility',
        'dilate': 'Morphological',
        'erode': 'Morphological',
        'custom': 'Custom',
        'separable': 'Custom',
    }

    # Methods to exclude from filter discovery
    EXCLUDED_METHODS = {'blur', 'edge_detection', 'custom', 'separable', 'sobel_combined'}

    @classmethod
    def discover_filters(cls) -> Dict[str, List[FilterInfo]]:
        """
        Discover all available filters from MatrixFilters class.

        Returns:
            Dictionary mapping categories to lists of FilterInfo.
        """
        filters_by_category: Dict[str, List[FilterInfo]] = {}

        # Get all static methods from MatrixFilters
        for name, method in inspect.getmembers(MatrixFilters, predicate=inspect.isfunction):
            # Skip private methods and excluded methods
            if name.startswith('_') or name in cls.EXCLUDED_METHODS:
                continue

            # Get method signature
            try:
                sig = inspect.signature(method)
            except (ValueError, TypeError):
                continue

            # Extract parameters
            params = []
            for param_name, param in sig.parameters.items():
                if param_name in cls.PARAM_CONFIGS:
                    param_config = cls.PARAM_CONFIGS[param_name]
                    # Use default from signature if available
                    if param.default != inspect.Parameter.empty:
                        param_config = FilterParameterConfig(
                            name=param_config.name,
                            label=param_config.label,
                            param_type=param_config.param_type,
                            default=param.default,
                            min_value=param_config.min_value,
                            max_value=param_config.max_value,
                            step=param_config.step,
                            choices=param_config.choices
                        )
                    params.append(param_config)

            # Determine category
            category = 'Other'
            for pattern, cat in cls.CATEGORY_PATTERNS.items():
                if pattern in name.lower():
                    category = cat
                    break

            # Create display name
            display_name = name.replace('_', ' ').title()

            # Create filter info
            filter_info = FilterInfo(
                name=name,
                display_name=display_name,
                method=method,
                parameters=params,
                category=category
            )

            # Add to category
            if category not in filters_by_category:
                filters_by_category[category] = []
            filters_by_category[category].append(filter_info)

        # Sort filters within each category
        for category in filters_by_category:
            filters_by_category[category].sort(key=lambda f: f.display_name)

        return filters_by_category


class FilterPanel(ttk.Frame):
    """
    Dynamic filter panel that discovers and presents available filters.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_apply_filter: Callable[[str, Dict[str, Any]], None],
        theme: Optional[MatrixThemeColors] = None,
        **kwargs
    ):
        """
        Initialize the filter panel.

        Args:
            parent: Parent widget.
            on_apply_filter: Callback when a filter should be applied.
                            Receives (filter_name, parameters_dict).
            theme: Optional theme colors.
        """
        super().__init__(parent, **kwargs)

        self.on_apply_filter = on_apply_filter
        self.theme = theme or MatrixThemeColors()

        self._filters = FilterDiscovery.discover_filters()
        self._current_filter: Optional[FilterInfo] = None
        self._param_widgets: Dict[str, FilterParameterWidget] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the filter panel UI."""
        # Filter selection
        selection_frame = ttk.Frame(self)
        selection_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(
            selection_frame,
            text="Filter:",
            style="Card.TLabel"
        ).pack(side=tk.LEFT, padx=(0, 5))

        # Category dropdown
        self._category_var = tk.StringVar()
        categories = list(self._filters.keys())
        if categories:
            self._category_var.set(categories[0])

        self._category_combo = ttk.Combobox(
            selection_frame,
            textvariable=self._category_var,
            values=categories,
            state="readonly",
            width=12
        )
        self._category_combo.pack(side=tk.LEFT, padx=5)
        self._category_combo.bind("<<ComboboxSelected>>", self._on_category_change)

        # Filter dropdown
        self._filter_var = tk.StringVar()
        self._filter_combo = ttk.Combobox(
            selection_frame,
            textvariable=self._filter_var,
            state="readonly",
            width=18
        )
        self._filter_combo.pack(side=tk.LEFT, padx=5)
        self._filter_combo.bind("<<ComboboxSelected>>", self._on_filter_change)

        # Parameters frame
        self._params_frame = ttk.Frame(self)
        self._params_frame.pack(fill=tk.X, padx=5, pady=5)

        # Apply button
        self._apply_btn = ttk.Button(
            self,
            text="Apply Filter",
            command=self._apply_filter
        )
        self._apply_btn.pack(pady=10)

        # Initialize with first category
        if categories:
            self._update_filter_list()

    def _on_category_change(self, event=None) -> None:
        """Handle category selection change."""
        self._update_filter_list()

    def _update_filter_list(self) -> None:
        """Update the filter dropdown based on selected category."""
        category = self._category_var.get()
        if category in self._filters:
            filter_names = [f.display_name for f in self._filters[category]]
            self._filter_combo['values'] = filter_names
            if filter_names:
                self._filter_var.set(filter_names[0])
                self._on_filter_change()

    def _on_filter_change(self, event=None) -> None:
        """Handle filter selection change."""
        category = self._category_var.get()
        filter_name = self._filter_var.get()

        # Find the filter
        if category in self._filters:
            for filter_info in self._filters[category]:
                if filter_info.display_name == filter_name:
                    self._current_filter = filter_info
                    self._update_parameters()
                    return

    def _update_parameters(self) -> None:
        """Update parameter widgets for current filter."""
        # Clear existing parameter widgets
        for widget in self._params_frame.winfo_children():
            widget.destroy()
        self._param_widgets.clear()

        if not self._current_filter:
            return

        # Create parameter widgets
        for param_config in self._current_filter.parameters:
            widget = FilterParameterWidget(
                self._params_frame,
                name=param_config.name,
                label=param_config.label,
                param_type=param_config.param_type,
                default=param_config.default,
                min_value=param_config.min_value,
                max_value=param_config.max_value,
                step=param_config.step,
                choices=param_config.choices
            )
            widget.pack(fill=tk.X, pady=2)
            self._param_widgets[param_config.name] = widget

    def _apply_filter(self) -> None:
        """Apply the currently selected filter."""
        if not self._current_filter:
            return

        # Collect parameter values
        params = {}
        for name, widget in self._param_widgets.items():
            params[name] = widget.get_value()

        # Call the callback
        self.on_apply_filter(self._current_filter.name, params)

    def get_available_filters(self) -> Dict[str, List[str]]:
        """Get all available filters by category."""
        return {
            cat: [f.display_name for f in filters]
            for cat, filters in self._filters.items()
        }


class QuickFilterBar(ttk.Frame):
    """
    Quick access bar for commonly used filters.
    """

    def __init__(
        self,
        parent: tk.Widget,
        on_apply_filter: Callable[[str, Dict[str, Any]], None],
        **kwargs
    ):
        """
        Initialize the quick filter bar.

        Args:
            parent: Parent widget.
            on_apply_filter: Callback when a filter should be applied.
        """
        super().__init__(parent, **kwargs)

        self.on_apply_filter = on_apply_filter

        self._build_ui()

    def _build_ui(self) -> None:
        """Build the quick filter bar UI."""
        # Common filters as buttons
        quick_filters = [
            ("Blur", "gaussian_blur", {'size': 3, 'sigma': 1.0}),
            ("Sharpen", "sharpen", {'strength': 1.0}),
            ("Edge H", "sobel_horizontal", {}),
            ("Edge V", "sobel_vertical", {}),
            ("Emboss", "emboss", {'direction': 'southeast', 'strength': 1.0}),
            ("Laplacian", "laplacian", {}),
        ]

        for label, filter_name, params in quick_filters:
            btn = ttk.Button(
                self,
                text=label,
                command=lambda n=filter_name, p=params: self.on_apply_filter(n, p),
                width=8
            )
            btn.pack(side=tk.LEFT, padx=2)

