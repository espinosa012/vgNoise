"""
Matrix Filters Library - Collection of convolution kernels and filters.

This module provides a variety of pre-defined filters for image/matrix processing,
including blur, sharpen, edge detection, emboss, and more.

All filters return NumPy arrays that can be used with VGMatrix2D.convolve() or
VGMatrix2D.apply_kernel().
"""

from enum import Enum
from typing import Tuple, Union
import numpy as np
from numpy.typing import NDArray


class BlurType(Enum):
    """Types of blur filters available."""
    BOX = "box"
    GAUSSIAN = "gaussian"
    MOTION_HORIZONTAL = "motion_horizontal"
    MOTION_VERTICAL = "motion_vertical"
    MOTION_DIAGONAL = "motion_diagonal"


class EdgeDetectionType(Enum):
    """Types of edge detection filters."""
    SOBEL_HORIZONTAL = "sobel_horizontal"
    SOBEL_VERTICAL = "sobel_vertical"
    SOBEL_COMBINED = "sobel_combined"
    PREWITT_HORIZONTAL = "prewitt_horizontal"
    PREWITT_VERTICAL = "prewitt_vertical"
    LAPLACIAN = "laplacian"
    LAPLACIAN_DIAGONAL = "laplacian_diagonal"


class MatrixFilters:
    """
    Collection of static methods for creating convolution kernels/filters.

    All methods return NumPy arrays suitable for use with matrix convolution.

    Example:
        >>> from vgmath.matrix.filters import MatrixFilters
        >>> kernel = MatrixFilters.gaussian_blur(5, sigma=1.0)
        >>> filtered_matrix = matrix.convolve(kernel)
    """

    # =========================================================================
    # Blur Filters
    # =========================================================================

    @staticmethod
    def box_blur(size: int = 3) -> NDArray[np.float64]:
        """
        Create a box blur (average) kernel.

        All values are equal, resulting in a simple averaging effect.

        Args:
            size: Kernel size (must be odd, default 3).

        Returns:
            Normalized box blur kernel.

        Raises:
            ValueError: If size is not a positive odd integer.
        """
        if size < 1 or size % 2 == 0:
            raise ValueError(f"Size must be a positive odd integer, got {size}")

        return np.ones((size, size), dtype=np.float64) / (size * size)

    @staticmethod
    def gaussian_blur(size: int = 3, sigma: float = 1.0) -> NDArray[np.float64]:
        """
        Create a Gaussian blur kernel.

        Produces a smoother blur than box blur, with more weight on center pixels.

        Args:
            size: Kernel size (must be odd, default 3).
            sigma: Standard deviation of the Gaussian distribution.
                   Higher values produce more blur.

        Returns:
            Normalized Gaussian blur kernel.

        Raises:
            ValueError: If size is not a positive odd integer or sigma <= 0.
        """
        if size < 1 or size % 2 == 0:
            raise ValueError(f"Size must be a positive odd integer, got {size}")
        if sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {sigma}")

        center = size // 2
        kernel = np.zeros((size, size), dtype=np.float64)

        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x*x + y*y) / (2 * sigma * sigma))

        # Normalize
        return kernel / np.sum(kernel)

    @staticmethod
    def motion_blur(size: int = 5, direction: str = "horizontal") -> NDArray[np.float64]:
        """
        Create a motion blur kernel.

        Simulates motion blur in a specific direction.

        Args:
            size: Kernel size (must be odd, default 5).
            direction: Blur direction - "horizontal", "vertical", or "diagonal".

        Returns:
            Normalized motion blur kernel.

        Raises:
            ValueError: If size is invalid or direction is unknown.
        """
        if size < 1 or size % 2 == 0:
            raise ValueError(f"Size must be a positive odd integer, got {size}")

        kernel = np.zeros((size, size), dtype=np.float64)

        if direction == "horizontal":
            kernel[size // 2, :] = 1.0
        elif direction == "vertical":
            kernel[:, size // 2] = 1.0
        elif direction == "diagonal":
            np.fill_diagonal(kernel, 1.0)
        else:
            raise ValueError(f"Unknown direction: {direction}. Use 'horizontal', 'vertical', or 'diagonal'")

        return kernel / np.sum(kernel)

    @staticmethod
    def blur(blur_type: BlurType = BlurType.GAUSSIAN, size: int = 3, **kwargs) -> NDArray[np.float64]:
        """
        Generic blur filter factory.

        Args:
            blur_type: Type of blur (BlurType enum).
            size: Kernel size (must be odd).
            **kwargs: Additional arguments for specific blur types (e.g., sigma for Gaussian).

        Returns:
            Blur kernel of the specified type.
        """
        if blur_type == BlurType.BOX:
            return MatrixFilters.box_blur(size)
        elif blur_type == BlurType.GAUSSIAN:
            sigma = kwargs.get("sigma", 1.0)
            return MatrixFilters.gaussian_blur(size, sigma)
        elif blur_type == BlurType.MOTION_HORIZONTAL:
            return MatrixFilters.motion_blur(size, "horizontal")
        elif blur_type == BlurType.MOTION_VERTICAL:
            return MatrixFilters.motion_blur(size, "vertical")
        elif blur_type == BlurType.MOTION_DIAGONAL:
            return MatrixFilters.motion_blur(size, "diagonal")
        else:
            raise ValueError(f"Unknown blur type: {blur_type}")

    # =========================================================================
    # Sharpen Filters
    # =========================================================================

    @staticmethod
    def sharpen(strength: float = 1.0) -> NDArray[np.float64]:
        """
        Create a sharpen kernel.

        Enhances edges and details in the matrix.

        Args:
            strength: Sharpening strength (1.0 = normal, higher = more sharp).

        Returns:
            Sharpen kernel.
        """
        center = 1.0 + 4.0 * strength
        edge = -strength

        return np.array([
            [0, edge, 0],
            [edge, center, edge],
            [0, edge, 0]
        ], dtype=np.float64)

    @staticmethod
    def unsharp_mask(size: int = 5, sigma: float = 1.0, amount: float = 1.0) -> NDArray[np.float64]:
        """
        Create an unsharp mask kernel.

        Sharpens by subtracting a blurred version from the original.

        Args:
            size: Kernel size (must be odd).
            sigma: Gaussian sigma for the blur component.
            amount: Sharpening amount (1.0 = normal).

        Returns:
            Unsharp mask kernel.
        """
        # Create identity kernel
        identity = np.zeros((size, size), dtype=np.float64)
        identity[size // 2, size // 2] = 1.0 + amount

        # Subtract scaled Gaussian blur
        gaussian = MatrixFilters.gaussian_blur(size, sigma)

        return identity - amount * gaussian

    # =========================================================================
    # Edge Detection Filters
    # =========================================================================

    @staticmethod
    def sobel_horizontal() -> NDArray[np.float64]:
        """
        Create a Sobel horizontal edge detection kernel.

        Detects vertical edges (horizontal gradients).

        Returns:
            Sobel horizontal kernel.
        """
        return np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float64)

    @staticmethod
    def sobel_vertical() -> NDArray[np.float64]:
        """
        Create a Sobel vertical edge detection kernel.

        Detects horizontal edges (vertical gradients).

        Returns:
            Sobel vertical kernel.
        """
        return np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float64)

    @staticmethod
    def sobel_combined() -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Get both Sobel kernels for combined edge detection.

        The combined edge magnitude is typically computed as:
        magnitude = sqrt(sobel_h^2 + sobel_v^2)

        Returns:
            Tuple of (horizontal kernel, vertical kernel).
        """
        return (MatrixFilters.sobel_horizontal(), MatrixFilters.sobel_vertical())

    @staticmethod
    def prewitt_horizontal() -> NDArray[np.float64]:
        """
        Create a Prewitt horizontal edge detection kernel.

        Similar to Sobel but with uniform weights.

        Returns:
            Prewitt horizontal kernel.
        """
        return np.array([
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ], dtype=np.float64)

    @staticmethod
    def prewitt_vertical() -> NDArray[np.float64]:
        """
        Create a Prewitt vertical edge detection kernel.

        Returns:
            Prewitt vertical kernel.
        """
        return np.array([
            [-1, -1, -1],
            [0, 0, 0],
            [1, 1, 1]
        ], dtype=np.float64)

    @staticmethod
    def laplacian() -> NDArray[np.float64]:
        """
        Create a Laplacian edge detection kernel (4-connectivity).

        Detects edges in all directions simultaneously.

        Returns:
            Laplacian kernel.
        """
        return np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float64)

    @staticmethod
    def laplacian_diagonal() -> NDArray[np.float64]:
        """
        Create a Laplacian edge detection kernel (8-connectivity).

        Includes diagonal neighbors for edge detection.

        Returns:
            Laplacian diagonal kernel.
        """
        return np.array([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], dtype=np.float64)

    @staticmethod
    def edge_detection(detection_type: EdgeDetectionType = EdgeDetectionType.SOBEL_HORIZONTAL) -> Union[NDArray[np.float64], Tuple[NDArray[np.float64], NDArray[np.float64]]]:
        """
        Generic edge detection filter factory.

        Args:
            detection_type: Type of edge detection (EdgeDetectionType enum).

        Returns:
            Edge detection kernel of the specified type.
            For SOBEL_COMBINED, returns a tuple of (horizontal, vertical) kernels.
        """
        if detection_type == EdgeDetectionType.SOBEL_HORIZONTAL:
            return MatrixFilters.sobel_horizontal()
        elif detection_type == EdgeDetectionType.SOBEL_VERTICAL:
            return MatrixFilters.sobel_vertical()
        elif detection_type == EdgeDetectionType.SOBEL_COMBINED:
            return MatrixFilters.sobel_combined()
        elif detection_type == EdgeDetectionType.PREWITT_HORIZONTAL:
            return MatrixFilters.prewitt_horizontal()
        elif detection_type == EdgeDetectionType.PREWITT_VERTICAL:
            return MatrixFilters.prewitt_vertical()
        elif detection_type == EdgeDetectionType.LAPLACIAN:
            return MatrixFilters.laplacian()
        elif detection_type == EdgeDetectionType.LAPLACIAN_DIAGONAL:
            return MatrixFilters.laplacian_diagonal()
        else:
            raise ValueError(f"Unknown edge detection type: {detection_type}")

    # =========================================================================
    # Emboss Filters
    # =========================================================================

    @staticmethod
    def emboss(direction: str = "southeast", strength: float = 1.0) -> NDArray[np.float64]:
        """
        Create an emboss kernel.

        Creates a 3D relief effect.

        Args:
            direction: Light direction - "north", "south", "east", "west",
                      "northeast", "northwest", "southeast", "southwest".
            strength: Emboss strength multiplier.

        Returns:
            Emboss kernel.
        """
        kernels = {
            "north": np.array([
                [0, 1, 0],
                [0, 0, 0],
                [0, -1, 0]
            ]),
            "south": np.array([
                [0, -1, 0],
                [0, 0, 0],
                [0, 1, 0]
            ]),
            "east": np.array([
                [0, 0, 0],
                [-1, 0, 1],
                [0, 0, 0]
            ]),
            "west": np.array([
                [0, 0, 0],
                [1, 0, -1],
                [0, 0, 0]
            ]),
            "northeast": np.array([
                [0, 0, 1],
                [0, 0, 0],
                [-1, 0, 0]
            ]),
            "northwest": np.array([
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, -1]
            ]),
            "southeast": np.array([
                [-1, 0, 0],
                [0, 0, 0],
                [0, 0, 1]
            ]),
            "southwest": np.array([
                [0, 0, -1],
                [0, 0, 0],
                [1, 0, 0]
            ])
        }

        if direction not in kernels:
            raise ValueError(
                f"Unknown direction: {direction}. "
                f"Use one of: {list(kernels.keys())}"
            )

        return kernels[direction].astype(np.float64) * strength

    # =========================================================================
    # Special Effect Filters
    # =========================================================================

    @staticmethod
    def ridge_detection() -> NDArray[np.float64]:
        """
        Create a ridge detection kernel.

        Highlights ridge-like structures in the matrix.

        Returns:
            Ridge detection kernel.
        """
        return np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=np.float64)

    @staticmethod
    def high_pass(size: int = 3) -> NDArray[np.float64]:
        """
        Create a high-pass filter kernel.

        Removes low-frequency components (smoothed areas), keeping edges.

        Args:
            size: Kernel size (must be odd).

        Returns:
            High-pass filter kernel.
        """
        if size < 1 or size % 2 == 0:
            raise ValueError(f"Size must be a positive odd integer, got {size}")

        # Start with negative box blur
        kernel = -np.ones((size, size), dtype=np.float64) / (size * size)

        # Add 1 to center (identity - low pass = high pass)
        center = size // 2
        kernel[center, center] += 1.0

        return kernel

    @staticmethod
    def low_pass(size: int = 3, sigma: float = 1.0) -> NDArray[np.float64]:
        """
        Create a low-pass filter kernel (alias for Gaussian blur).

        Removes high-frequency components (edges), keeping smooth areas.

        Args:
            size: Kernel size (must be odd).
            sigma: Gaussian sigma.

        Returns:
            Low-pass filter kernel (Gaussian blur).
        """
        return MatrixFilters.gaussian_blur(size, sigma)

    @staticmethod
    def identity() -> NDArray[np.float64]:
        """
        Create an identity kernel.

        Applying this kernel returns the original matrix unchanged.

        Returns:
            Identity kernel.
        """
        return np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ], dtype=np.float64)

    # =========================================================================
    # Morphological-like Filters
    # =========================================================================

    @staticmethod
    def dilate(size: int = 3) -> NDArray[np.float64]:
        """
        Create a dilation-like kernel.

        When used with max pooling or as a weight for finding local maxima.
        Note: For true morphological dilation, use specialized operations.

        Args:
            size: Kernel size (must be odd).

        Returns:
            Dilation kernel (all ones).
        """
        if size < 1 or size % 2 == 0:
            raise ValueError(f"Size must be a positive odd integer, got {size}")

        return np.ones((size, size), dtype=np.float64)

    @staticmethod
    def erode(size: int = 3) -> NDArray[np.float64]:
        """
        Create an erosion-like kernel.

        Similar to dilate but for erosion operations.

        Args:
            size: Kernel size (must be odd).

        Returns:
            Erosion kernel (all ones).
        """
        return MatrixFilters.dilate(size)

    # =========================================================================
    # Custom Kernel Creation
    # =========================================================================

    @staticmethod
    def custom(values: list, normalize: bool = False) -> NDArray[np.float64]:
        """
        Create a custom kernel from a 2D list of values.

        Args:
            values: 2D list of kernel values.
            normalize: If True, normalize kernel to sum to 1.

        Returns:
            Custom kernel as NumPy array.

        Example:
            >>> kernel = MatrixFilters.custom([
            ...     [1, 2, 1],
            ...     [2, 4, 2],
            ...     [1, 2, 1]
            ... ], normalize=True)
        """
        kernel = np.array(values, dtype=np.float64)

        if kernel.ndim != 2:
            raise ValueError("Kernel must be 2D")

        if normalize:
            total = np.sum(kernel)
            if total != 0:
                kernel = kernel / total

        return kernel

    @staticmethod
    def separable(row_kernel: list, col_kernel: list) -> NDArray[np.float64]:
        """
        Create a 2D kernel from separable 1D kernels.

        The resulting kernel is the outer product of row and column kernels.
        Useful for efficient convolution (can be applied as two 1D passes).

        Args:
            row_kernel: 1D list for row direction.
            col_kernel: 1D list for column direction.

        Returns:
            2D kernel as outer product of inputs.

        Example:
            >>> # Create Gaussian-like 5x5 kernel
            >>> kernel = MatrixFilters.separable(
            ...     [1, 4, 6, 4, 1],
            ...     [1, 4, 6, 4, 1]
            ... )
        """
        row = np.array(row_kernel, dtype=np.float64).reshape(-1, 1)
        col = np.array(col_kernel, dtype=np.float64).reshape(1, -1)

        kernel = row @ col

        # Normalize
        total = np.sum(kernel)
        if total != 0:
            kernel = kernel / total

        return kernel

