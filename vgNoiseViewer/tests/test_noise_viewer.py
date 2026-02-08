"""
Tests for vgNoise Viewer application.

This module contains comprehensive tests for the NoiseViewer application,
testing all UI components, noise generation, and user interactions.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
from tkinter import ttk

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from PIL import Image

from vgmath import PerlinNoise2D, NoiseType, FractalType


class TestNoiseViewerInitialization(unittest.TestCase):
    """Tests for NoiseViewer initialization."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()  # Hide window during tests

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_window_title(self):
        """Test that window title is set correctly."""
        self.assertEqual(self.root.title(), "vgNoise Viewer")

    def test_default_seed_value(self):
        """Test default seed value."""
        self.assertEqual(self.viewer.seed.get(), 0)

    def test_default_noise_type(self):
        """Test default noise type."""
        self.assertEqual(self.viewer.noise_type.get(), NoiseType.PERLIN.name)

    def test_default_frequency(self):
        """Test default frequency value."""
        self.assertAlmostEqual(self.viewer.frequency.get(), 0.01, places=3)

    def test_default_offset_values(self):
        """Test default offset values."""
        self.assertEqual(self.viewer.offset_x.get(), 0.0)
        self.assertEqual(self.viewer.offset_y.get(), 0.0)

    def test_default_fractal_type(self):
        """Test default fractal type."""
        self.assertEqual(self.viewer.fractal_type.get(), FractalType.FBM.name)

    def test_default_octaves(self):
        """Test default octaves value."""
        self.assertEqual(self.viewer.octaves.get(), 5)

    def test_default_lacunarity(self):
        """Test default lacunarity value."""
        self.assertAlmostEqual(self.viewer.lacunarity.get(), 2.0, places=1)

    def test_default_persistence(self):
        """Test default persistence value."""
        self.assertAlmostEqual(self.viewer.persistence.get(), 0.5, places=2)

    def test_default_weighted_strength(self):
        """Test default weighted strength value."""
        self.assertEqual(self.viewer.weighted_strength.get(), 0.0)

    def test_default_ping_pong_strength(self):
        """Test default ping pong strength value."""
        self.assertAlmostEqual(self.viewer.ping_pong_strength.get(), 2.0, places=1)

    def test_default_image_size(self):
        """Test default image size."""
        self.assertEqual(self.viewer.image_size.get(), 512)

    def test_image_canvas_created(self):
        """Test that image canvas is created."""
        self.assertIsNotNone(self.viewer._image_canvas)

    def test_photo_image_created(self):
        """Test that photo image is created after initialization."""
        self.assertIsNotNone(self.viewer.photo_image)

    def test_initialization_flag(self):
        """Test that initialization flag is False after setup."""
        self.assertFalse(self.viewer._initializing)


class TestNoiseGeneratorCreation(unittest.TestCase):
    """Tests for noise generator creation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_create_generator_returns_perlin_noise(self):
        """Test that _create_generator returns a PerlinNoise2D instance."""
        generator = self.viewer._create_generator()
        self.assertIsInstance(generator, PerlinNoise2D)

    def test_generator_uses_current_frequency(self):
        """Test that generator uses the current frequency value."""
        self.viewer.frequency.set(0.05)
        generator = self.viewer._create_generator()
        self.assertAlmostEqual(generator.frequency, 0.05, places=3)

    def test_generator_uses_current_offset(self):
        """Test that generator uses the current offset values."""
        self.viewer.offset_x.set(100.0)
        self.viewer.offset_y.set(200.0)
        generator = self.viewer._create_generator()
        self.assertEqual(generator.offset, (100.0, 200.0))

    def test_generator_uses_current_fractal_type(self):
        """Test that generator uses the current fractal type."""
        self.viewer.fractal_type.set(FractalType.RIDGED.name)
        generator = self.viewer._create_generator()
        self.assertEqual(generator.fractal_type, FractalType.RIDGED)

    def test_generator_uses_current_octaves(self):
        """Test that generator uses the current octaves value."""
        self.viewer.octaves.set(3)
        generator = self.viewer._create_generator()
        self.assertEqual(generator.octaves, 3)

    def test_generator_uses_current_lacunarity(self):
        """Test that generator uses the current lacunarity value."""
        self.viewer.lacunarity.set(3.0)
        generator = self.viewer._create_generator()
        self.assertAlmostEqual(generator.lacunarity, 3.0, places=1)

    def test_generator_uses_current_persistence(self):
        """Test that generator uses the current persistence value."""
        self.viewer.persistence.set(0.7)
        generator = self.viewer._create_generator()
        self.assertAlmostEqual(generator.persistence, 0.7, places=2)

    def test_generator_uses_current_seed(self):
        """Test that generator uses the current seed value."""
        self.viewer.seed.set(12345)
        generator = self.viewer._create_generator()
        self.assertEqual(generator.seed, 12345)

    def test_generator_uses_weighted_strength(self):
        """Test that generator uses the weighted strength value."""
        self.viewer.weighted_strength.set(0.5)
        generator = self.viewer._create_generator()
        self.assertAlmostEqual(generator.weighted_strength, 0.5, places=2)

    def test_generator_uses_ping_pong_strength(self):
        """Test that generator uses the ping pong strength value."""
        self.viewer.ping_pong_strength.set(3.0)
        generator = self.viewer._create_generator()
        self.assertAlmostEqual(generator.ping_pong_strength, 3.0, places=1)


class TestImageGeneration(unittest.TestCase):
    """Tests for image generation functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_update_image_creates_photo_image(self):
        """Test that update_image creates a PhotoImage."""
        self.viewer.update_image()
        self.assertIsNotNone(self.viewer.photo_image)

    def test_update_image_during_initialization_does_nothing(self):
        """Test that update_image does nothing during initialization."""
        self.viewer._initializing = True
        old_image = self.viewer.photo_image
        self.viewer.update_image()
        # Image should not change during initialization
        self.assertEqual(self.viewer.photo_image, old_image)

    def test_update_image_with_different_sizes(self):
        """Test that update_image works with different image sizes."""
        for size in [128, 256, 512]:
            with self.subTest(size=size):
                self.viewer.image_size.set(size)
                self.viewer.update_image()
                self.assertIsNotNone(self.viewer.photo_image)

    def test_update_image_with_all_fractal_types(self):
        """Test that update_image works with all fractal types."""
        for ftype in FractalType:
            with self.subTest(fractal_type=ftype.name):
                self.viewer.fractal_type.set(ftype.name)
                self.viewer.update_image()
                self.assertIsNotNone(self.viewer.photo_image)


class TestRandomSeed(unittest.TestCase):
    """Tests for random seed functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_randomize_seed_changes_seed(self):
        """Test that _randomize_seed changes the seed value."""
        original_seed = self.viewer.seed.get()
        # Run multiple times to ensure it changes (very unlikely to be same)
        different = False
        for _ in range(10):
            self.viewer._randomize_seed()
            if self.viewer.seed.get() != original_seed:
                different = True
                break
        self.assertTrue(different)

    def test_randomize_seed_generates_non_negative(self):
        """Test that _randomize_seed generates non-negative seeds."""
        for _ in range(20):
            self.viewer._randomize_seed()
            self.assertGreaterEqual(self.viewer.seed.get(), 0)

    def test_randomize_seed_generates_within_range(self):
        """Test that _randomize_seed generates seeds within valid range."""
        for _ in range(20):
            self.viewer._randomize_seed()
            seed = self.viewer.seed.get()
            self.assertGreaterEqual(seed, 0)
            self.assertLess(seed, 1000000)


class TestParameterValidation(unittest.TestCase):
    """Tests for parameter validation and bounds checking."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_frequency_minimum_bound(self):
        """Test frequency minimum bound in generator."""
        self.viewer.frequency.set(0.001)
        generator = self.viewer._create_generator()
        self.assertGreaterEqual(generator.frequency, 0.001)

    def test_octaves_clamped_minimum(self):
        """Test that octaves is clamped to minimum of 1."""
        self.viewer.octaves.set(0)
        generator = self.viewer._create_generator()
        self.assertGreaterEqual(generator.octaves, 1)

    def test_octaves_clamped_maximum(self):
        """Test that octaves is clamped to maximum of 9."""
        self.viewer.octaves.set(15)
        generator = self.viewer._create_generator()
        self.assertLessEqual(generator.octaves, 9)

    def test_persistence_bounds(self):
        """Test persistence value bounds."""
        self.viewer.persistence.set(0.5)
        generator = self.viewer._create_generator()
        self.assertGreaterEqual(generator.persistence, 0.0)
        self.assertLessEqual(generator.persistence, 1.0)

    def test_weighted_strength_clamped(self):
        """Test that weighted strength is clamped to 0-1 range."""
        self.viewer.weighted_strength.set(0.5)
        generator = self.viewer._create_generator()
        self.assertGreaterEqual(generator.weighted_strength, 0.0)
        self.assertLessEqual(generator.weighted_strength, 1.0)


class TestNoiseOutput(unittest.TestCase):
    """Tests for noise output values."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_noise_values_in_valid_range(self):
        """Test that generated noise values are in [0, 1] range."""
        generator = self.viewer._create_generator()
        region = generator.generate_region([(0, 100, 50), (0, 100, 50)])

        self.assertGreaterEqual(region.min(), 0.0)
        self.assertLessEqual(region.max(), 1.0)

    def test_noise_reproducibility_with_same_seed(self):
        """Test that same seed produces same noise."""
        self.viewer.seed.set(42)
        generator1 = self.viewer._create_generator()
        region1 = generator1.generate_region([(0, 50, 25), (0, 50, 25)])

        generator2 = self.viewer._create_generator()
        region2 = generator2.generate_region([(0, 50, 25), (0, 50, 25)])

        np.testing.assert_array_almost_equal(region1, region2)

    def test_different_seeds_produce_different_noise(self):
        """Test that different seeds produce different noise."""
        self.viewer.seed.set(42)
        generator1 = self.viewer._create_generator()
        region1 = generator1.generate_region([(0, 50, 25), (0, 50, 25)])

        self.viewer.seed.set(123)
        generator2 = self.viewer._create_generator()
        region2 = generator2.generate_region([(0, 50, 25), (0, 50, 25)])

        # Arrays should be different
        self.assertFalse(np.allclose(region1, region2))

    def test_noise_output_shape(self):
        """Test that noise output has correct shape."""
        generator = self.viewer._create_generator()

        for size in [64, 128, 256]:
            with self.subTest(size=size):
                region = generator.generate_region([(0, size, size), (0, size, size)])
                self.assertEqual(region.shape, (size, size))


class TestFractalTypes(unittest.TestCase):
    """Tests for different fractal types."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_fbm_produces_valid_output(self):
        """Test FBM fractal type produces valid output."""
        self.viewer.fractal_type.set(FractalType.FBM.name)
        generator = self.viewer._create_generator()
        region = generator.generate_region([(0, 50, 25), (0, 50, 25)])

        self.assertEqual(region.shape, (25, 25))
        self.assertGreaterEqual(region.min(), 0.0)
        self.assertLessEqual(region.max(), 1.0)

    def test_ridged_produces_valid_output(self):
        """Test RIDGED fractal type produces valid output."""
        self.viewer.fractal_type.set(FractalType.RIDGED.name)
        generator = self.viewer._create_generator()
        region = generator.generate_region([(0, 50, 25), (0, 50, 25)])

        self.assertEqual(region.shape, (25, 25))
        self.assertGreaterEqual(region.min(), 0.0)
        self.assertLessEqual(region.max(), 1.0)

    def test_ping_pong_produces_valid_output(self):
        """Test PING_PONG fractal type produces valid output."""
        self.viewer.fractal_type.set(FractalType.PING_PONG.name)
        generator = self.viewer._create_generator()
        region = generator.generate_region([(0, 50, 25), (0, 50, 25)])

        self.assertEqual(region.shape, (25, 25))
        self.assertGreaterEqual(region.min(), 0.0)
        self.assertLessEqual(region.max(), 1.0)

    def test_none_produces_valid_output(self):
        """Test NONE fractal type (single octave) produces valid output."""
        self.viewer.fractal_type.set(FractalType.NONE.name)
        generator = self.viewer._create_generator()
        region = generator.generate_region([(0, 50, 25), (0, 50, 25)])

        self.assertEqual(region.shape, (25, 25))
        self.assertGreaterEqual(region.min(), 0.0)
        self.assertLessEqual(region.max(), 1.0)

    def test_different_fractals_produce_different_output(self):
        """Test that different fractal types produce different output."""
        self.viewer.seed.set(42)
        results = {}

        for ftype in [FractalType.FBM, FractalType.RIDGED, FractalType.PING_PONG]:
            self.viewer.fractal_type.set(ftype.name)
            generator = self.viewer._create_generator()
            region = generator.generate_region([(0, 50, 25), (0, 50, 25)])
            results[ftype.name] = region

        # All should be different from each other
        self.assertFalse(np.allclose(results['FBM'], results['RIDGED']))
        self.assertFalse(np.allclose(results['FBM'], results['PING_PONG']))
        self.assertFalse(np.allclose(results['RIDGED'], results['PING_PONG']))


class TestUIComponents(unittest.TestCase):
    """Tests for UI component existence and configuration."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_dark_theme_colors_set(self):
        """Test that dark theme colors are configured."""
        self.assertEqual(self.viewer.theme_colors.background, "#1e1e1e")
        self.assertEqual(self.viewer.theme_colors.foreground, "#ffffff")
        self.assertEqual(self.viewer.theme_colors.card, "#2d2d2d")
        self.assertEqual(self.viewer.theme_colors.accent, "#4a9eff")

    def test_root_background_color(self):
        """Test that root window has correct background color."""
        bg = self.root.cget('bg')
        self.assertEqual(bg, "#1e1e1e")

    def test_minimum_window_size(self):
        """Test that minimum window size is set."""
        min_width = self.root.minsize()[0]
        min_height = self.root.minsize()[1]
        self.assertEqual(min_width, 800)
        self.assertEqual(min_height, 600)


class TestStepperControls(unittest.TestCase):
    """Tests for stepper control functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_frequency_step_value(self):
        """Test frequency stepper step value."""
        # Frequency should step by 0.001
        initial = self.viewer.frequency.get()
        self.viewer.frequency.set(initial + 0.001)
        self.assertAlmostEqual(
            self.viewer.frequency.get() - initial,
            0.001,
            places=4
        )

    def test_octaves_step_value(self):
        """Test octaves stepper step value."""
        # Octaves should step by 1
        initial = self.viewer.octaves.get()
        self.viewer.octaves.set(initial + 1)
        self.assertEqual(self.viewer.octaves.get() - initial, 1)

    def test_lacunarity_step_value(self):
        """Test lacunarity stepper step value."""
        # Lacunarity should step by 0.1
        initial = self.viewer.lacunarity.get()
        self.viewer.lacunarity.set(initial + 0.1)
        self.assertAlmostEqual(
            self.viewer.lacunarity.get() - initial,
            0.1,
            places=2
        )

    def test_persistence_step_value(self):
        """Test persistence stepper step value."""
        # Persistence should step by 0.05
        initial = self.viewer.persistence.get()
        self.viewer.persistence.set(initial + 0.05)
        self.assertAlmostEqual(
            self.viewer.persistence.get() - initial,
            0.05,
            places=3
        )


class TestNegativeSeedHandling(unittest.TestCase):
    """Tests for negative seed handling."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_negative_seed_does_not_crash(self):
        """Test that negative seed values don't crash the application."""
        self.viewer.seed.set(-12345)
        try:
            generator = self.viewer._create_generator()
            region = generator.generate_region([(0, 50, 25), (0, 50, 25)])
            self.assertIsNotNone(region)
        except Exception as e:
            self.fail(f"Negative seed caused exception: {e}")

    def test_negative_seed_produces_valid_output(self):
        """Test that negative seed produces valid noise output."""
        self.viewer.seed.set(-42)
        generator = self.viewer._create_generator()
        region = generator.generate_region([(0, 50, 25), (0, 50, 25)])

        self.assertEqual(region.shape, (25, 25))
        self.assertGreaterEqual(region.min(), 0.0)
        self.assertLessEqual(region.max(), 1.0)

    def test_negative_seed_equivalent_to_absolute_value(self):
        """Test that negative seed produces same result as its absolute value."""
        self.viewer.seed.set(-42)
        generator1 = self.viewer._create_generator()
        region1 = generator1.generate_region([(0, 50, 25), (0, 50, 25)])

        self.viewer.seed.set(42)
        generator2 = self.viewer._create_generator()
        region2 = generator2.generate_region([(0, 50, 25), (0, 50, 25)])

        np.testing.assert_array_almost_equal(region1, region2)


class TestNoiseTypeChange(unittest.TestCase):
    """Tests for noise type switching functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_change_noise_type_to_simplex(self):
        """Test changing noise type from PERLIN to SIMPLEX."""
        self.viewer.noise_type.set('SIMPLEX')
        try:
            self.viewer.update_image()
            self.assertIsNotNone(self.viewer.photo_image)
        except Exception as e:
            self.fail(f"Changing to SIMPLEX failed: {e}")

    def test_change_noise_type_to_perlin(self):
        """Test changing noise type from SIMPLEX to PERLIN."""
        self.viewer.noise_type.set('SIMPLEX')
        self.viewer.update_image()
        self.viewer.noise_type.set('PERLIN')
        try:
            self.viewer.update_image()
            self.assertIsNotNone(self.viewer.photo_image)
        except Exception as e:
            self.fail(f"Changing to PERLIN failed: {e}")

    def test_noise_type_produces_different_output(self):
        """Test that different noise types produce different output."""
        self.viewer.seed.set(42)

        # Get PERLIN output
        self.viewer.noise_type.set('PERLIN')
        gen_perlin = self.viewer._create_generator()
        region_perlin = gen_perlin.generate_region([(0, 50, 25), (0, 50, 25)])

        # Get SIMPLEX output
        self.viewer.noise_type.set('SIMPLEX')
        gen_simplex = self.viewer._create_generator()
        region_simplex = gen_simplex.generate_region([(0, 50, 25), (0, 50, 25)])

        # They should be different
        self.assertFalse(np.allclose(region_perlin, region_simplex))

    def test_simplex_generator_created_correctly(self):
        """Test that SIMPLEX creates OpenSimplexNoise2D generator."""
        from vgmath import OpenSimplexNoise2D

        self.viewer.noise_type.set('SIMPLEX')
        generator = self.viewer._create_generator()
        self.assertIsInstance(generator, OpenSimplexNoise2D)

    def test_perlin_generator_created_correctly(self):
        """Test that PERLIN creates PerlinNoise2D generator."""
        self.viewer.noise_type.set('PERLIN')
        generator = self.viewer._create_generator()
        self.assertIsInstance(generator, PerlinNoise2D)

    def test_multiple_noise_type_switches(self):
        """Test multiple rapid switches between noise types."""
        for _ in range(5):
            self.viewer.noise_type.set('PERLIN')
            self.viewer.update_image()
            self.viewer.noise_type.set('SIMPLEX')
            self.viewer.update_image()

        # Should complete without error
        self.assertIsNotNone(self.viewer.photo_image)

    def test_simplex_uses_all_godot_parameters(self):
        """Test that SIMPLEX generator uses all Godot parameters."""
        from vgmath import OpenSimplexNoise2D

        self.viewer.noise_type.set('SIMPLEX')
        self.viewer.frequency.set(0.05)
        self.viewer.offset_x.set(100.0)
        self.viewer.offset_y.set(200.0)
        self.viewer.fractal_type.set(FractalType.RIDGED.name)
        self.viewer.octaves.set(3)
        self.viewer.lacunarity.set(3.0)
        self.viewer.persistence.set(0.7)
        self.viewer.weighted_strength.set(0.5)
        self.viewer.ping_pong_strength.set(3.0)
        self.viewer.seed.set(42)

        generator = self.viewer._create_generator()

        self.assertIsInstance(generator, OpenSimplexNoise2D)
        self.assertAlmostEqual(generator.frequency, 0.05, places=3)
        self.assertEqual(generator.offset, (100.0, 200.0))
        self.assertEqual(generator.fractal_type, FractalType.RIDGED)
        self.assertEqual(generator.octaves, 3)
        self.assertAlmostEqual(generator.lacunarity, 3.0, places=1)
        self.assertAlmostEqual(generator.persistence, 0.7, places=2)
        self.assertAlmostEqual(generator.weighted_strength, 0.5, places=2)
        self.assertAlmostEqual(generator.ping_pong_strength, 3.0, places=1)

    def test_simplex_with_all_fractal_types(self):
        """Test SIMPLEX works with all fractal types."""
        self.viewer.noise_type.set('SIMPLEX')

        for ftype in FractalType:
            with self.subTest(fractal_type=ftype.name):
                self.viewer.fractal_type.set(ftype.name)
                self.viewer.update_image()
                self.assertIsNotNone(self.viewer.photo_image)

    def test_simplex_different_fractals_produce_different_output(self):
        """Test that different fractal types with SIMPLEX produce different output."""
        self.viewer.noise_type.set('SIMPLEX')
        self.viewer.seed.set(42)
        results = {}

        for ftype in [FractalType.FBM, FractalType.RIDGED, FractalType.PING_PONG]:
            self.viewer.fractal_type.set(ftype.name)
            generator = self.viewer._create_generator()
            region = generator.generate_region([(0, 50, 25), (0, 50, 25)])
            results[ftype.name] = region

        # All should be different from each other
        self.assertFalse(np.allclose(results['FBM'], results['RIDGED']))
        self.assertFalse(np.allclose(results['FBM'], results['PING_PONG']))
        self.assertFalse(np.allclose(results['RIDGED'], results['PING_PONG']))


class TestPerformance(unittest.TestCase):
    """Performance tests for noise generation."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        """Clean up after tests."""
        cls.root.destroy()

    def setUp(self):
        """Set up each test."""
        from noise_viewer.app import NoiseViewer
        self.viewer = NoiseViewer(self.root)

    def test_512x512_generation_under_threshold(self):
        """Test that 512x512 generation completes in reasonable time."""
        import time

        self.viewer.image_size.set(512)
        generator = self.viewer._create_generator()

        # Warmup
        _ = generator.generate_region([(0, 64, 64), (0, 64, 64)])

        start = time.perf_counter()
        _ = generator.generate_region([(0, 512, 512), (0, 512, 512)])
        elapsed = time.perf_counter() - start

        # Should complete in under 500ms (generous threshold)
        self.assertLess(elapsed, 0.5, f"512x512 generation took {elapsed:.3f}s")

    def test_multiple_updates_stable(self):
        """Test that multiple rapid updates don't cause issues."""
        for i in range(10):
            self.viewer.seed.set(i)
            self.viewer.update_image()

        # Should complete without error
        self.assertIsNotNone(self.viewer.photo_image)


def run_tests():
    """Run all tests and return results."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseViewerInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseGeneratorCreation))
    suite.addTests(loader.loadTestsFromTestCase(TestImageGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestRandomSeed))
    suite.addTests(loader.loadTestsFromTestCase(TestParameterValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseOutput))
    suite.addTests(loader.loadTestsFromTestCase(TestFractalTypes))
    suite.addTests(loader.loadTestsFromTestCase(TestUIComponents))
    suite.addTests(loader.loadTestsFromTestCase(TestStepperControls))
    suite.addTests(loader.loadTestsFromTestCase(TestNegativeSeedHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestNoiseTypeChange))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    run_tests()
