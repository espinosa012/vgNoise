"""
Image generation and display utilities for vgNoise Viewer.

This module handles noise-to-image conversion and display.
"""

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageTk
from typing import Tuple, Optional

# Handle both package and direct execution imports
try:
    from .config import MAX_DISPLAY_SIZE
except ImportError:
    from config import MAX_DISPLAY_SIZE


class ImageGenerator:
    """Generates images from noise data."""

    @staticmethod
    def noise_to_image(
        noise_data: NDArray[np.float64],
        normalize: bool = True
    ) -> Image.Image:
        """
        Convert noise data to a grayscale PIL Image.

        Args:
            noise_data: 2D array of noise values.
            normalize: Whether to normalize values to [0, 1] range.

        Returns:
            PIL Image in grayscale mode.
        """
        if normalize:
            data = np.clip(noise_data, 0, 1)
        else:
            data = noise_data

        # Convert to 8-bit grayscale
        image_data = (data * 255).astype(np.uint8)
        return Image.fromarray(image_data, mode='L')

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
            resample: Resampling method.

        Returns:
            Resized image or original if already within size.
        """
        width, height = image.size

        if width <= max_size and height <= max_size:
            return image

        # Calculate new size maintaining aspect ratio
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


class NoiseImageRenderer:
    """Renders noise to displayable images."""

    def __init__(
        self,
        max_display_size: int = MAX_DISPLAY_SIZE
    ):
        """
        Initialize the renderer.

        Args:
            max_display_size: Maximum display dimension.
        """
        self.max_display_size = max_display_size
        self._generator = ImageGenerator()
        self._current_image: Optional[Image.Image] = None
        self._current_photo: Optional[ImageTk.PhotoImage] = None

    def render(
        self,
        noise_data: NDArray[np.float64],
        display_size: Optional[int] = None
    ) -> ImageTk.PhotoImage:
        """
        Render noise data to a displayable PhotoImage.

        Args:
            noise_data: 2D array of noise values.
            display_size: Optional custom display size.

        Returns:
            Tkinter PhotoImage ready for display.
        """
        # Convert to image
        self._current_image = self._generator.noise_to_image(noise_data)

        # Resize for display
        max_size = display_size or self.max_display_size
        display_image = self._generator.resize_for_display(
            self._current_image,
            max_size
        )

        # Convert to PhotoImage
        self._current_photo = self._generator.to_photo_image(display_image)

        return self._current_photo

    def get_current_image(self) -> Optional[Image.Image]:
        """Get the current full-resolution image."""
        return self._current_image

    def save_current_image(self, path: str) -> bool:
        """
        Save the current image to file.

        Args:
            path: File path to save to.

        Returns:
            True if saved successfully, False otherwise.
        """
        if self._current_image is None:
            return False

        try:
            self._current_image.save(path)
            return True
        except Exception:
            return False
