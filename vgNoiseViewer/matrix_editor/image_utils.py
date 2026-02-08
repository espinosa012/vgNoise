"""
Image utilities for Matrix Editor App.

This module handles matrix-to-image conversion with support for
transparency (None values) and grayscale image import.
"""

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk, ImageDraw
from typing import Tuple, Optional
import sys
from pathlib import Path

# Add parent directory to path to import vgmath
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vgmath.matrix import VGMatrix2D

# Handle both package and direct execution imports
try:
    from .config import MAX_DISPLAY_SIZE
except ImportError:
    from matrix_editor.config import MAX_DISPLAY_SIZE


class MatrixImageGenerator:
    """Generates images from VGMatrix2D with transparency support."""

    @staticmethod
    def create_checkerboard(width: int, height: int, cell_size: int = 8,
                           color1: str = "#404040", color2: str = "#303030") -> Image.Image:
        """
        Create a checkerboard pattern image for transparency background.

        Args:
            width: Image width.
            height: Image height.
            cell_size: Size of each checkerboard cell.
            color1: First checkerboard color.
            color2: Second checkerboard color.

        Returns:
            PIL Image with checkerboard pattern.
        """
        img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(img)

        # Convert hex colors to RGB tuples
        c1 = tuple(int(color1.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        c2 = tuple(int(color2.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        for y in range(0, height, cell_size):
            for x in range(0, width, cell_size):
                color = c1 if ((x // cell_size) + (y // cell_size)) % 2 == 0 else c2
                draw.rectangle([x, y, x + cell_size - 1, y + cell_size - 1], fill=color)

        return img

    @staticmethod
    def matrix_to_image(
        matrix: VGMatrix2D,
        normalize: bool = True,
        show_transparency: bool = True
    ) -> Image.Image:
        """
        Convert VGMatrix2D to a PIL Image with transparency support.

        Args:
            matrix: VGMatrix2D to convert.
            normalize: Whether to normalize values to [0, 1] range.
            show_transparency: Whether to show None values as transparent.

        Returns:
            PIL Image in RGBA mode (with transparency) or RGB mode.
        """
        data = matrix.data.copy()
        mask = matrix.mask

        if normalize and np.any(mask):
            # Normalize only assigned values
            min_val = matrix.min()
            max_val = matrix.max()
            if min_val is not None and max_val is not None and min_val != max_val:
                data = (data - min_val) / (max_val - min_val)
            elif min_val is not None:
                data = np.clip(data, 0, 1)

        # Clip to valid range
        data = np.clip(data, 0, 1)

        # Convert to 8-bit
        gray_data = (data * 255).astype(np.uint8)

        if show_transparency:
            # Create RGBA image
            height, width = matrix.shape

            # Create checkerboard background
            background = MatrixImageGenerator.create_checkerboard(width, height)
            background = background.convert('RGBA')

            # Create grayscale layer
            gray_img = Image.fromarray(gray_data, mode='L').convert('RGBA')

            # Create alpha channel from mask
            alpha = (mask * 255).astype(np.uint8)
            alpha_img = Image.fromarray(alpha, mode='L')

            # Apply alpha to grayscale
            r, g, b, _ = gray_img.split()
            gray_with_alpha = Image.merge('RGBA', (r, g, b, alpha_img))

            # Composite over checkerboard
            result = Image.alpha_composite(background, gray_with_alpha)
            return result
        else:
            # Simple grayscale, unassigned values shown as black
            gray_data[~mask] = 0
            return Image.fromarray(gray_data, mode='L').convert('RGB')

    @staticmethod
    def resize_for_display(
        image: Image.Image,
        max_size: int = MAX_DISPLAY_SIZE,
        resample: int = Image.Resampling.NEAREST
    ) -> Image.Image:
        """
        Resize image for display if necessary.

        Args:
            image: PIL Image to resize.
            max_size: Maximum dimension for display.
            resample: Resampling method (NEAREST preserves pixel look).

        Returns:
            Resized image or original if already within size.
        """
        width, height = image.size

        if width <= max_size and height <= max_size:
            # Scale up small images for better visibility
            scale = max_size / max(width, height)
            if scale > 1:
                new_width = int(width * scale)
                new_height = int(height * scale)
                return image.resize((new_width, new_height), resample)
            return image

        # Scale down large images
        scale = max_size / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)

        return image.resize((new_width, new_height), resample)

    @staticmethod
    def to_photo_image(image: Image.Image) -> ImageTk.PhotoImage:
        """
        Convert PIL Image to Tkinter PhotoImage.

        Args:
            image: PIL Image.

        Returns:
            Tkinter PhotoImage.
        """
        return ImageTk.PhotoImage(image)


class ImageToMatrixConverter:
    """Converts images to VGMatrix2D."""

    @staticmethod
    def load_image_as_matrix(
        filepath: str,
        target_size: Optional[Tuple[int, int]] = None
    ) -> VGMatrix2D:
        """
        Load an image file and convert it to a VGMatrix2D.

        Args:
            filepath: Path to the image file.
            target_size: Optional (width, height) to resize to. If None, uses original size.

        Returns:
            VGMatrix2D with grayscale values normalized to [0, 1].

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a valid image.
        """
        try:
            img = Image.open(filepath)
        except Exception as e:
            raise ValueError(f"Could not open image: {e}")

        # Resize if requested
        if target_size is not None:
            img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')

        # Convert to numpy array and normalize
        data = np.array(img, dtype=np.float64) / 255.0

        # Create VGMatrix2D
        return VGMatrix2D.from_numpy(data)

    @staticmethod
    def image_to_grayscale_array(image: Image.Image) -> NDArray[np.float64]:
        """
        Convert a PIL Image to a grayscale numpy array.

        Args:
            image: PIL Image in any mode.

        Returns:
            2D numpy array with values in [0, 1].
        """
        if image.mode != 'L':
            image = image.convert('L')

        return np.array(image, dtype=np.float64) / 255.0


class MatrixImageRenderer:
    """High-level renderer combining generation and display utilities."""

    def __init__(self, max_display_size: int = MAX_DISPLAY_SIZE):
        """
        Initialize the renderer.

        Args:
            max_display_size: Maximum display dimension.
        """
        self.max_display_size = max_display_size
        self._generator = MatrixImageGenerator()

    def render(
        self,
        matrix: VGMatrix2D,
        normalize: bool = True,
        show_transparency: bool = True
    ) -> ImageTk.PhotoImage:
        """
        Render a VGMatrix2D to a displayable PhotoImage.

        Args:
            matrix: Matrix to render.
            normalize: Whether to normalize values.
            show_transparency: Whether to show None values as transparent.

        Returns:
            Tkinter PhotoImage ready for display.
        """
        # Generate image
        image = self._generator.matrix_to_image(matrix, normalize, show_transparency)

        # Resize for display
        image = self._generator.resize_for_display(image, self.max_display_size)

        # Convert to PhotoImage
        return self._generator.to_photo_image(image)

    def get_pil_image(
        self,
        matrix: VGMatrix2D,
        normalize: bool = True,
        show_transparency: bool = True
    ) -> Image.Image:
        """
        Get a PIL Image from a VGMatrix2D (for saving).

        Args:
            matrix: Matrix to convert.
            normalize: Whether to normalize values.
            show_transparency: Whether to show transparency.

        Returns:
            PIL Image.
        """
        return self._generator.matrix_to_image(matrix, normalize, show_transparency)

