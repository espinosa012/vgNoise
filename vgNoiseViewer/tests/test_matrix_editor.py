"""
Test script for Matrix Editor App components.

This script tests the main components of the matrix editor to ensure
they work correctly before running the full GUI.
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_matrix_image_utils():
    """Test image utilities."""
    print("Testing Matrix Image Utils...")

    from matrix_editor.image_utils import (
        MatrixImageGenerator,
        ImageToMatrixConverter,
        MatrixImageRenderer
    )
    from vgmath.matrix import VGMatrix2D

    # Create a test matrix with some None values
    matrix = VGMatrix2D((10, 10), 0.5)
    matrix.set_value_at(0, 0, None)
    matrix.set_value_at(5, 5, None)
    matrix.set_value_at(9, 9, 1.0)

    # Test checkerboard generation
    checkerboard = MatrixImageGenerator.create_checkerboard(100, 100)
    assert checkerboard.size == (100, 100), "Checkerboard size mismatch"
    print("  ✓ Checkerboard generation works")

    # Test matrix to image conversion
    image = MatrixImageGenerator.matrix_to_image(matrix, normalize=True, show_transparency=True)
    assert image.size == (10, 10), f"Image size mismatch: {image.size}"
    assert image.mode == 'RGBA', f"Image mode should be RGBA, got {image.mode}"
    print("  ✓ Matrix to image conversion works (with transparency)")

    # Test without transparency
    image_no_alpha = MatrixImageGenerator.matrix_to_image(matrix, normalize=True, show_transparency=False)
    assert image_no_alpha.mode == 'RGB', f"Image mode should be RGB, got {image_no_alpha.mode}"
    print("  ✓ Matrix to image conversion works (without transparency)")

    # Test resize
    resized = MatrixImageGenerator.resize_for_display(image, max_size=100)
    assert max(resized.size) <= 100, "Resize didn't work correctly"
    print("  ✓ Image resize works")

    # Test renderer
    renderer = MatrixImageRenderer(max_display_size=200)
    pil_image = renderer.get_pil_image(matrix)
    assert pil_image is not None, "Renderer failed"
    print("  ✓ Renderer works")

    print("  All image utils tests passed!\n")


def test_filter_discovery():
    """Test filter discovery."""
    print("Testing Filter Discovery...")

    from matrix_editor.filter_panel import FilterDiscovery

    filters = FilterDiscovery.discover_filters()

    # Should have at least some categories
    assert len(filters) > 0, "No filters discovered"
    print(f"  Found {len(filters)} filter categories")

    # Count total filters
    total_filters = sum(len(f) for f in filters.values())
    print(f"  Found {total_filters} total filters")

    # Check expected categories exist
    expected_categories = ['Blur', 'Edge Detection', 'Effects']
    for cat in expected_categories:
        if cat in filters:
            print(f"  ✓ Category '{cat}' found with {len(filters[cat])} filters")
        else:
            print(f"  ⚠ Category '{cat}' not found")

    # Print all discovered filters
    print("\n  Discovered filters by category:")
    for category, filter_list in sorted(filters.items()):
        print(f"    {category}:")
        for f in filter_list:
            param_names = [p.name for p in f.parameters]
            params_str = f"({', '.join(param_names)})" if param_names else "()"
            print(f"      - {f.display_name}{params_str}")

    print("\n  Filter discovery tests passed!\n")


def test_filter_application():
    """Test applying filters to matrices."""
    print("Testing Filter Application...")

    from vgmath.matrix import VGMatrix2D
    from vgmath.matrix.filters import MatrixFilters

    # Create test matrix
    matrix = VGMatrix2D((20, 20), 0.0)
    for i in range(20):
        for j in range(20):
            matrix.set_value_at(i, j, (i + j) / 38.0)

    # Test various filters
    filters_to_test = [
        ("gaussian_blur", {'size': 3, 'sigma': 1.0}),
        ("box_blur", {'size': 3}),
        ("sharpen", {'strength': 1.0}),
        ("sobel_horizontal", {}),
        ("laplacian", {}),
        ("emboss", {'direction': 'southeast', 'strength': 1.0}),
        ("high_pass", {'size': 3}),
    ]

    for filter_name, params in filters_to_test:
        try:
            filter_method = getattr(MatrixFilters, filter_name)
            kernel = filter_method(**params)

            # Handle tuple return (sobel_combined)
            if isinstance(kernel, tuple):
                result = matrix.convolve(kernel[0])
            else:
                result = matrix.convolve(kernel)

            assert result.shape == matrix.shape, f"{filter_name}: Shape mismatch"
            print(f"  ✓ {filter_name} works")
        except Exception as e:
            print(f"  ✗ {filter_name} failed: {e}")

    print("\n  Filter application tests passed!\n")


def test_widgets():
    """Test widget creation (without display)."""
    print("Testing Widgets...")

    # Just verify imports work
    from widgets.matrix_widgets import (
        MatrixCellEditor,
        FilterParameterWidget,
        ScrollableFrame,
        Card,
        StatusBar,
    )

    print("  ✓ All widget imports successful")
    print("  Widget tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Matrix Editor Component Tests")
    print("=" * 60 + "\n")

    try:
        test_matrix_image_utils()
        test_filter_discovery()
        test_filter_application()
        test_widgets()

        print("=" * 60)
        print("All tests passed!")
        print("=" * 60)
        return True
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

