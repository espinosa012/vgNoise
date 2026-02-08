"""
VGMatrix2D - Efficient 2D matrix data structure for numerical operations.

This module provides a high-performance 2D matrix class optimized for
numerical operations like addition, subtraction, multiplication, and
convolution filtering.
"""

from typing import Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray


class VGMatrix2D:
    """
    Efficient 2D matrix data structure with support for unassigned values.

    Uses NumPy arrays internally for high-performance operations.
    Unassigned values are tracked using a boolean mask (more efficient
    than storing None values directly).

    Attributes:
        _data: Internal NumPy array storing the matrix values.
        _mask: Boolean mask where True indicates an assigned value.
        _shape: Tuple (rows, columns) representing matrix dimensions.

    Example:
        >>> matrix = VGMatrix2D((512, 512), 0.55)
        >>> matrix.get_value_at(0, 0)
        0.55
        >>> matrix.set_value_at(0, 0, None)
        >>> matrix.get_value_at(0, 0)
        None
    """

    def __init__(
        self,
        shape: Tuple[int, int],
        default_value: Optional[float] = 0.0
    ) -> None:
        """
        Initialize a 2D matrix with given shape and default value.

        Args:
            shape: Tuple (rows, columns) specifying matrix dimensions.
            default_value: Initial value for all cells. Use None for unassigned.

        Raises:
            ValueError: If shape dimensions are not positive integers.
        """
        if len(shape) != 2 or shape[0] <= 0 or shape[1] <= 0:
            raise ValueError(f"Shape must be a tuple of two positive integers, got {shape}")

        self._shape = shape

        if default_value is None:
            # All values unassigned
            self._data = np.zeros(shape, dtype=np.float64)
            self._mask = np.zeros(shape, dtype=np.bool_)
        else:
            # All values assigned with default
            self._data = np.full(shape, default_value, dtype=np.float64)
            self._mask = np.ones(shape, dtype=np.bool_)

    @property
    def shape(self) -> Tuple[int, int]:
        """Get the matrix shape (rows, columns)."""
        return self._shape

    @property
    def rows(self) -> int:
        """Get the number of rows."""
        return self._shape[0]

    @property
    def cols(self) -> int:
        """Get the number of columns."""
        return self._shape[1]

    @property
    def size(self) -> int:
        """Get the total number of elements."""
        return self._shape[0] * self._shape[1]

    @property
    def data(self) -> NDArray[np.float64]:
        """Get the internal data array (read-only view)."""
        return self._data.view()

    @property
    def mask(self) -> NDArray[np.bool_]:
        """Get the assignment mask (True = assigned)."""
        return self._mask.view()

    def _check_bounds(self, row: int, col: int) -> None:
        """Check if indices are within bounds."""
        if not (0 <= row < self._shape[0] and 0 <= col < self._shape[1]):
            raise IndexError(
                f"Index ({row}, {col}) out of bounds for matrix of shape {self._shape}"
            )

    def get_value_at(self, row: int, col: int) -> Optional[float]:
        """
        Get the value at the specified position.

        Args:
            row: Row index (0-based).
            col: Column index (0-based).

        Returns:
            The value at (row, col), or None if unassigned.

        Raises:
            IndexError: If indices are out of bounds.
        """
        self._check_bounds(row, col)

        if self._mask[row, col]:
            return float(self._data[row, col])
        return None

    def set_value_at(self, row: int, col: int, value: Optional[float]) -> None:
        """
        Set the value at the specified position.

        Args:
            row: Row index (0-based).
            col: Column index (0-based).
            value: The value to set, or None to mark as unassigned.

        Raises:
            IndexError: If indices are out of bounds.
        """
        self._check_bounds(row, col)

        if value is None:
            self._mask[row, col] = False
        else:
            self._data[row, col] = value
            self._mask[row, col] = True

    def is_assigned(self, row: int, col: int) -> bool:
        """Check if a position has an assigned value."""
        self._check_bounds(row, col)
        return bool(self._mask[row, col])

    def count_assigned(self) -> int:
        """Count the number of assigned values."""
        return int(np.sum(self._mask))

    def count_unassigned(self) -> int:
        """Count the number of unassigned values."""
        return self.size - self.count_assigned()

    def fill(self, value: Optional[float]) -> None:
        """Fill the entire matrix with a value."""
        if value is None:
            self._mask.fill(False)
        else:
            self._data.fill(value)
            self._mask.fill(True)

    def resize(self, new_shape: Tuple[int, int], default_value: Optional[float] = None) -> None:
        """
        Resize the matrix, preserving existing values where possible.

        Args:
            new_shape: New shape (rows, columns).
            default_value: Value for new cells (None for unassigned).
        """
        if len(new_shape) != 2 or new_shape[0] <= 0 or new_shape[1] <= 0:
            raise ValueError(f"Shape must be a tuple of two positive integers, got {new_shape}")

        old_shape = self._shape

        # Create new arrays
        if default_value is None:
            new_data = np.zeros(new_shape, dtype=np.float64)
            new_mask = np.zeros(new_shape, dtype=np.bool_)
        else:
            new_data = np.full(new_shape, default_value, dtype=np.float64)
            new_mask = np.ones(new_shape, dtype=np.bool_)

        # Copy existing data
        min_rows = min(old_shape[0], new_shape[0])
        min_cols = min(old_shape[1], new_shape[1])

        new_data[:min_rows, :min_cols] = self._data[:min_rows, :min_cols]
        new_mask[:min_rows, :min_cols] = self._mask[:min_rows, :min_cols]

        self._data = new_data
        self._mask = new_mask
        self._shape = new_shape

    def copy(self) -> "VGMatrix2D":
        """Create a deep copy of the matrix."""
        result = VGMatrix2D.__new__(VGMatrix2D)
        result._shape = self._shape
        result._data = self._data.copy()
        result._mask = self._mask.copy()
        return result

    # =========================================================================
    # Arithmetic Operations
    # =========================================================================

    def __add__(self, other: Union["VGMatrix2D", float, int]) -> "VGMatrix2D":
        """Add another matrix or scalar."""
        result = self.copy()
        result._add_inplace(other)
        return result

    def __radd__(self, other: Union[float, int]) -> "VGMatrix2D":
        """Right addition (scalar + matrix)."""
        return self.__add__(other)

    def __iadd__(self, other: Union["VGMatrix2D", float, int]) -> "VGMatrix2D":
        """In-place addition."""
        self._add_inplace(other)
        return self

    def _add_inplace(self, other: Union["VGMatrix2D", float, int]) -> None:
        """Internal in-place addition."""
        if isinstance(other, VGMatrix2D):
            if self._shape != other._shape:
                raise ValueError(
                    f"Matrix shapes don't match: {self._shape} vs {other._shape}"
                )
            # Only add where both are assigned
            both_assigned = self._mask & other._mask
            self._data[both_assigned] += other._data[both_assigned]
            # Result is unassigned where either operand is unassigned
            self._mask &= other._mask
        else:
            # Scalar addition
            self._data[self._mask] += other

    def __sub__(self, other: Union["VGMatrix2D", float, int]) -> "VGMatrix2D":
        """Subtract another matrix or scalar."""
        result = self.copy()
        result._sub_inplace(other)
        return result

    def __rsub__(self, other: Union[float, int]) -> "VGMatrix2D":
        """Right subtraction (scalar - matrix)."""
        result = self.copy()
        result._data = other - result._data
        return result

    def __isub__(self, other: Union["VGMatrix2D", float, int]) -> "VGMatrix2D":
        """In-place subtraction."""
        self._sub_inplace(other)
        return self

    def _sub_inplace(self, other: Union["VGMatrix2D", float, int]) -> None:
        """Internal in-place subtraction."""
        if isinstance(other, VGMatrix2D):
            if self._shape != other._shape:
                raise ValueError(
                    f"Matrix shapes don't match: {self._shape} vs {other._shape}"
                )
            both_assigned = self._mask & other._mask
            self._data[both_assigned] -= other._data[both_assigned]
            self._mask &= other._mask
        else:
            self._data[self._mask] -= other

    def __mul__(self, other: Union["VGMatrix2D", float, int]) -> "VGMatrix2D":
        """Element-wise multiplication or scalar multiplication."""
        result = self.copy()
        result._mul_inplace(other)
        return result

    def __rmul__(self, other: Union[float, int]) -> "VGMatrix2D":
        """Right multiplication (scalar * matrix)."""
        return self.__mul__(other)

    def __imul__(self, other: Union["VGMatrix2D", float, int]) -> "VGMatrix2D":
        """In-place multiplication."""
        self._mul_inplace(other)
        return self

    def _mul_inplace(self, other: Union["VGMatrix2D", float, int]) -> None:
        """Internal in-place multiplication."""
        if isinstance(other, VGMatrix2D):
            if self._shape != other._shape:
                raise ValueError(
                    f"Matrix shapes don't match: {self._shape} vs {other._shape}"
                )
            both_assigned = self._mask & other._mask
            self._data[both_assigned] *= other._data[both_assigned]
            self._mask &= other._mask
        else:
            self._data[self._mask] *= other

    def __truediv__(self, other: Union["VGMatrix2D", float, int]) -> "VGMatrix2D":
        """Element-wise division or scalar division."""
        result = self.copy()
        result._div_inplace(other)
        return result

    def __itruediv__(self, other: Union["VGMatrix2D", float, int]) -> "VGMatrix2D":
        """In-place division."""
        self._div_inplace(other)
        return self

    def _div_inplace(self, other: Union["VGMatrix2D", float, int]) -> None:
        """Internal in-place division."""
        if isinstance(other, VGMatrix2D):
            if self._shape != other._shape:
                raise ValueError(
                    f"Matrix shapes don't match: {self._shape} vs {other._shape}"
                )
            both_assigned = self._mask & other._mask
            # Avoid division by zero
            safe_divisor = np.where(other._data != 0, other._data, 1)
            self._data[both_assigned] /= safe_divisor[both_assigned]
            self._mask &= other._mask
        else:
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            self._data[self._mask] /= other

    def __neg__(self) -> "VGMatrix2D":
        """Negate the matrix."""
        result = self.copy()
        result._data = -result._data
        return result

    def matmul(self, other: "VGMatrix2D") -> "VGMatrix2D":
        """
        Matrix multiplication (dot product).

        Args:
            other: Matrix to multiply with. Must have shape (self.cols, n).

        Returns:
            Result matrix of shape (self.rows, other.cols).

        Raises:
            ValueError: If matrix dimensions are incompatible.
        """
        if self._shape[1] != other._shape[0]:
            raise ValueError(
                f"Matrix dimensions incompatible for multiplication: "
                f"{self._shape} @ {other._shape}"
            )

        # For matmul, treat unassigned as 0
        data_a = np.where(self._mask, self._data, 0)
        data_b = np.where(other._mask, other._data, 0)

        result_data = data_a @ data_b

        result = VGMatrix2D.__new__(VGMatrix2D)
        result._shape = (self._shape[0], other._shape[1])
        result._data = result_data
        result._mask = np.ones(result._shape, dtype=np.bool_)

        return result

    def __matmul__(self, other: "VGMatrix2D") -> "VGMatrix2D":
        """Matrix multiplication operator (@)."""
        return self.matmul(other)

    # =========================================================================
    # Convolution and Filtering
    # =========================================================================

    def convolve(
        self,
        kernel: Union["VGMatrix2D", NDArray[np.float64]],
        mode: str = "same",
        boundary: str = "fill",
        fill_value: float = 0.0
    ) -> "VGMatrix2D":
        """
        Apply a convolution kernel to the matrix.

        Args:
            kernel: Convolution kernel (VGMatrix2D or NumPy array).
            mode: Output size mode:
                - 'same': Output has same size as input (default).
                - 'full': Full convolution output.
                - 'valid': Only positions where kernel fully overlaps.
            boundary: Boundary handling:
                - 'fill': Pad with fill_value (default).
                - 'wrap': Wrap around edges.
                - 'symm': Symmetric boundary (reflect).
            fill_value: Value to use for 'fill' boundary mode.

        Returns:
            New VGMatrix2D with convolution result.
        """
        # Get kernel data
        if isinstance(kernel, VGMatrix2D):
            kernel_data = np.where(kernel._mask, kernel._data, 0)
        else:
            kernel_data = np.asarray(kernel, dtype=np.float64)

        # Get source data (treat unassigned as fill_value)
        source_data = np.where(self._mask, self._data, fill_value)

        # Flip kernel for convolution (vs correlation)
        kernel_flipped = kernel_data[::-1, ::-1]

        # Calculate padding based on mode
        kh, kw = kernel_flipped.shape
        pad_h, pad_w = kh // 2, kw // 2

        # Apply padding based on boundary mode
        if boundary == "fill":
            padded = np.pad(
                source_data,
                ((pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=fill_value
            )
        elif boundary == "wrap":
            padded = np.pad(
                source_data,
                ((pad_h, pad_h), (pad_w, pad_w)),
                mode="wrap"
            )
        elif boundary == "symm":
            padded = np.pad(
                source_data,
                ((pad_h, pad_h), (pad_w, pad_w)),
                mode="symmetric"
            )
        else:
            padded = np.pad(
                source_data,
                ((pad_h, pad_h), (pad_w, pad_w)),
                mode="constant",
                constant_values=fill_value
            )

        # Perform convolution using sliding window view
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, kernel_flipped.shape)
        result_data = np.einsum('ijkl,kl->ij', windows, kernel_flipped)

        # Handle output mode
        if mode == "same":
            pass  # Already correct size
        elif mode == "valid":
            # Crop to valid region
            vh, vw = self._shape[0] - kh + 1, self._shape[1] - kw + 1
            if vh > 0 and vw > 0:
                start_h = (result_data.shape[0] - vh) // 2
                start_w = (result_data.shape[1] - vw) // 2
                result_data = result_data[start_h:start_h+vh, start_w:start_w+vw]
            else:
                result_data = np.array([[]], dtype=np.float64)
        elif mode == "full":
            # Full convolution - recompute with more padding
            full_pad_h, full_pad_w = kh - 1, kw - 1
            if boundary == "fill":
                padded_full = np.pad(
                    source_data,
                    ((full_pad_h, full_pad_h), (full_pad_w, full_pad_w)),
                    mode="constant",
                    constant_values=fill_value
                )
            else:
                padded_full = np.pad(
                    source_data,
                    ((full_pad_h, full_pad_h), (full_pad_w, full_pad_w)),
                    mode="wrap" if boundary == "wrap" else "symmetric"
                )
            windows_full = sliding_window_view(padded_full, kernel_flipped.shape)
            result_data = np.einsum('ijkl,kl->ij', windows_full, kernel_flipped)

        # Create result matrix
        result = VGMatrix2D.__new__(VGMatrix2D)
        result._shape = result_data.shape
        result._data = result_data.astype(np.float64)
        result._mask = np.ones(result._shape, dtype=np.bool_)

        return result

    def apply_kernel(
        self,
        kernel: Union["VGMatrix2D", NDArray[np.float64]],
        normalize: bool = False
    ) -> "VGMatrix2D":
        """
        Apply a convolution kernel (alias for convolve with 'same' mode).

        Args:
            kernel: Convolution kernel.
            normalize: If True, normalize kernel to sum to 1.

        Returns:
            New VGMatrix2D with filtered result.
        """
        if isinstance(kernel, VGMatrix2D):
            kernel_data = np.where(kernel._mask, kernel._data, 0)
        else:
            kernel_data = np.asarray(kernel, dtype=np.float64)

        if normalize:
            kernel_sum = np.sum(kernel_data)
            if kernel_sum != 0:
                kernel_data = kernel_data / kernel_sum

        return self.convolve(kernel_data, mode="same")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def to_numpy(self, fill_unassigned: Optional[float] = None) -> NDArray[np.float64]:
        """
        Convert to NumPy array.

        Args:
            fill_unassigned: Value to use for unassigned positions.
                            If None, uses np.nan.

        Returns:
            NumPy array with matrix data.
        """
        if fill_unassigned is None:
            result = np.where(self._mask, self._data, np.nan)
        else:
            result = np.where(self._mask, self._data, fill_unassigned)
        return result.copy()

    @classmethod
    def from_numpy(cls, array: NDArray, mask: Optional[NDArray[np.bool_]] = None) -> "VGMatrix2D":
        """
        Create a VGMatrix2D from a NumPy array.

        Args:
            array: 2D NumPy array.
            mask: Optional boolean mask (True = assigned).
                 If None, all values are considered assigned.

        Returns:
            New VGMatrix2D instance.
        """
        array = np.asarray(array, dtype=np.float64)
        if array.ndim != 2:
            raise ValueError(f"Array must be 2D, got {array.ndim}D")

        result = cls.__new__(cls)
        result._shape = array.shape
        result._data = array.copy()

        if mask is None:
            result._mask = np.ones(array.shape, dtype=np.bool_)
        else:
            result._mask = np.asarray(mask, dtype=np.bool_).copy()

        return result

    def from_noise(
        self,
        noise: "NoiseGenerator",
        row_start: int,
        row_end: int,
        col_start: int,
        col_end: int
    ) -> None:
        """
        Fill the matrix with values from a noise region.

        The requested region is clipped to fit within the matrix dimensions.
        This method uses vectorized operations for efficient large-region sampling.

        Args:
            noise: Source noise generator with generate_region or get_value_at method.
            row_start: Starting row index in noise space (inclusive).
            row_end: Ending row index in noise space (exclusive).
            col_start: Starting column index in noise space (inclusive).
            col_end: Ending column index in noise space (exclusive).

        Note:
            The matrix is not resized. If the requested region is larger than
            the matrix, only the portion that fits will be filled.
        """
        # Validate region
        if row_end <= row_start or col_end <= col_start:
            return  # Empty region, nothing to do

        # Calculate effective region size (clipped to matrix dimensions)
        effective_rows = min(row_end - row_start, self._shape[0])
        effective_cols = min(col_end - col_start, self._shape[1])

        if effective_rows <= 0 or effective_cols <= 0:
            return

        # Adjust end indices based on matrix size
        effective_row_end = row_start + effective_rows
        effective_col_end = col_start + effective_cols

        # Use generate_region for efficient vectorized noise generation
        if hasattr(noise, 'generate_region'):
            # generate_region expects: [(x_start, x_end, num_points), (y_start, y_end, num_points)]
            # where coordinates are float and we want integer grid positions
            region = [
                (float(row_start), float(effective_row_end - 1), effective_rows),
                (float(col_start), float(effective_col_end - 1), effective_cols)
            ]
            noise_data = noise.generate_region(region)

            # Copy to matrix data (only the portion that fits)
            self._data[:effective_rows, :effective_cols] = noise_data
            self._mask[:effective_rows, :effective_cols] = True

        elif hasattr(noise, 'get_values_vectorized'):
            # Use vectorized method with meshgrid
            rows = np.arange(row_start, effective_row_end, dtype=np.float64)
            cols = np.arange(col_start, effective_col_end, dtype=np.float64)
            xx, yy = np.meshgrid(rows, cols, indexing='ij')

            noise_data = noise.get_values_vectorized(xx.flatten(), yy.flatten())
            noise_data = noise_data.reshape((effective_rows, effective_cols))

            self._data[:effective_rows, :effective_cols] = noise_data
            self._mask[:effective_rows, :effective_cols] = True

        else:
            # Fallback: use get_value_at (slower but always works)
            for r in range(effective_rows):
                for c in range(effective_cols):
                    value = noise.get_value_at((row_start + r, col_start + c))
                    self._data[r, c] = value
                    self._mask[r, c] = True

    def min(self, ignore_unassigned: bool = True) -> Optional[float]:
        """Get minimum value."""
        if ignore_unassigned:
            if not np.any(self._mask):
                return None
            return float(np.min(self._data[self._mask]))
        return float(np.min(self._data))

    def max(self, ignore_unassigned: bool = True) -> Optional[float]:
        """Get maximum value."""
        if ignore_unassigned:
            if not np.any(self._mask):
                return None
            return float(np.max(self._data[self._mask]))
        return float(np.max(self._data))

    def mean(self, ignore_unassigned: bool = True) -> Optional[float]:
        """Get mean value."""
        if ignore_unassigned:
            if not np.any(self._mask):
                return None
            return float(np.mean(self._data[self._mask]))
        return float(np.mean(self._data))

    def sum(self, ignore_unassigned: bool = True) -> float:
        """Get sum of values."""
        if ignore_unassigned:
            return float(np.sum(self._data[self._mask]))
        return float(np.sum(self._data))

    def clip(self, min_value: float, max_value: float) -> "VGMatrix2D":
        """Clip values to range [min_value, max_value]."""
        result = self.copy()
        np.clip(result._data, min_value, max_value, out=result._data)
        return result

    def normalize(self, new_min: float = 0.0, new_max: float = 1.0) -> "VGMatrix2D":
        """Normalize values to range [new_min, new_max]."""
        result = self.copy()

        if not np.any(self._mask):
            return result

        old_min = self.min()
        old_max = self.max()

        if old_min == old_max:
            result._data[self._mask] = (new_min + new_max) / 2
        else:
            result._data[self._mask] = (
                (self._data[self._mask] - old_min) / (old_max - old_min)
                * (new_max - new_min) + new_min
            )

        return result

    def __repr__(self) -> str:
        """String representation."""
        assigned = self.count_assigned()
        return (
            f"VGMatrix2D(shape={self._shape}, "
            f"assigned={assigned}/{self.size})"
        )

    def __str__(self) -> str:
        """Human-readable string representation."""
        return f"VGMatrix2D {self._shape[0]}x{self._shape[1]}"

    def __eq__(self, other: object) -> bool:
        """Check equality with another matrix."""
        if not isinstance(other, VGMatrix2D):
            return False
        if self._shape != other._shape:
            return False
        if not np.array_equal(self._mask, other._mask):
            return False
        # Compare only assigned values
        return np.allclose(
            self._data[self._mask],
            other._data[other._mask]
        )

    # =========================================================================
    # Filter Methods (using MatrixFilters)
    # =========================================================================

    def blur(
        self,
        blur_type: str = "gaussian",
        size: int = 3,
        sigma: float = 1.0
    ) -> "VGMatrix2D":
        """
        Apply a blur filter to the matrix.

        Args:
            blur_type: Type of blur - "box", "gaussian", "motion_horizontal",
                      "motion_vertical", "motion_diagonal".
            size: Kernel size (must be odd, default 3).
            sigma: Gaussian sigma (only used for gaussian blur).

        Returns:
            New blurred VGMatrix2D.

        Example:
            >>> blurred = matrix.blur("gaussian", size=5, sigma=1.5)
        """
        from .filters import MatrixFilters, BlurType

        blur_map = {
            "box": BlurType.BOX,
            "gaussian": BlurType.GAUSSIAN,
            "motion_horizontal": BlurType.MOTION_HORIZONTAL,
            "motion_vertical": BlurType.MOTION_VERTICAL,
            "motion_diagonal": BlurType.MOTION_DIAGONAL,
        }

        if blur_type not in blur_map:
            raise ValueError(f"Unknown blur type: {blur_type}. Use one of: {list(blur_map.keys())}")

        kernel = MatrixFilters.blur(blur_map[blur_type], size, sigma=sigma)
        return self.convolve(kernel)

    def sharpen(self, strength: float = 1.0) -> "VGMatrix2D":
        """
        Apply a sharpen filter to the matrix.

        Args:
            strength: Sharpening strength (1.0 = normal, higher = more sharp).

        Returns:
            New sharpened VGMatrix2D.
        """
        from .filters import MatrixFilters

        kernel = MatrixFilters.sharpen(strength)
        return self.convolve(kernel)

    def edge_detect(self, method: str = "sobel_horizontal") -> "VGMatrix2D":
        """
        Apply edge detection to the matrix.

        Args:
            method: Edge detection method - "sobel_horizontal", "sobel_vertical",
                   "prewitt_horizontal", "prewitt_vertical", "laplacian",
                   "laplacian_diagonal".

        Returns:
            New VGMatrix2D with detected edges.
        """
        from .filters import MatrixFilters, EdgeDetectionType

        edge_map = {
            "sobel_horizontal": EdgeDetectionType.SOBEL_HORIZONTAL,
            "sobel_vertical": EdgeDetectionType.SOBEL_VERTICAL,
            "prewitt_horizontal": EdgeDetectionType.PREWITT_HORIZONTAL,
            "prewitt_vertical": EdgeDetectionType.PREWITT_VERTICAL,
            "laplacian": EdgeDetectionType.LAPLACIAN,
            "laplacian_diagonal": EdgeDetectionType.LAPLACIAN_DIAGONAL,
        }

        if method not in edge_map:
            raise ValueError(f"Unknown edge detection method: {method}. Use one of: {list(edge_map.keys())}")

        kernel = MatrixFilters.edge_detection(edge_map[method])
        return self.convolve(kernel)

    def emboss(self, direction: str = "southeast", strength: float = 1.0) -> "VGMatrix2D":
        """
        Apply an emboss effect to the matrix.

        Args:
            direction: Light direction - "north", "south", "east", "west",
                      "northeast", "northwest", "southeast", "southwest".
            strength: Emboss strength multiplier.

        Returns:
            New embossed VGMatrix2D.
        """
        from .filters import MatrixFilters

        kernel = MatrixFilters.emboss(direction, strength)
        return self.convolve(kernel)

    def high_pass(self, size: int = 3) -> "VGMatrix2D":
        """
        Apply a high-pass filter to the matrix.

        Removes low-frequency components (smooth areas), keeping edges.

        Args:
            size: Kernel size (must be odd).

        Returns:
            New high-pass filtered VGMatrix2D.
        """
        from .filters import MatrixFilters

        kernel = MatrixFilters.high_pass(size)
        return self.convolve(kernel)

    def low_pass(self, size: int = 3, sigma: float = 1.0) -> "VGMatrix2D":
        """
        Apply a low-pass filter to the matrix.

        Removes high-frequency components (edges), keeping smooth areas.
        Equivalent to Gaussian blur.

        Args:
            size: Kernel size (must be odd).
            sigma: Gaussian sigma.

        Returns:
            New low-pass filtered VGMatrix2D.
        """
        from .filters import MatrixFilters

        kernel = MatrixFilters.low_pass(size, sigma)
        return self.convolve(kernel)

    def ridge_detect(self) -> "VGMatrix2D":
        """
        Apply ridge detection to the matrix.

        Highlights ridge-like structures.

        Returns:
            New VGMatrix2D with detected ridges.
        """
        from .filters import MatrixFilters

        kernel = MatrixFilters.ridge_detection()
        return self.convolve(kernel)

    def unsharp_mask(
        self,
        size: int = 5,
        sigma: float = 1.0,
        amount: float = 1.0
    ) -> "VGMatrix2D":
        """
        Apply unsharp mask sharpening.

        A more sophisticated sharpening that subtracts a blurred version.

        Args:
            size: Kernel size (must be odd).
            sigma: Gaussian sigma for blur component.
            amount: Sharpening amount (1.0 = normal).

        Returns:
            New sharpened VGMatrix2D.
        """
        from .filters import MatrixFilters

        kernel = MatrixFilters.unsharp_mask(size, sigma, amount)
        return self.convolve(kernel)

    # =========================================================================
    # Serialization / Deserialization
    # =========================================================================

    def to_bytes(self, compressed: bool = True) -> bytes:
        """
        Serialize the matrix to a binary format.

        This is the most efficient format for storage and transmission.
        The format includes a header with version, shape, and compression info,
        followed by the data and mask arrays.

        Args:
            compressed: If True, compress the data using zlib (default True).

        Returns:
            Bytes object containing the serialized matrix.

        Example:
            >>> matrix = VGMatrix2D((100, 100), 0.5)
            >>> data = matrix.to_bytes()
            >>> restored = VGMatrix2D.from_bytes(data)
        """
        import struct
        import zlib

        # Header format:
        # - Magic number (4 bytes): 'VGM2' to identify the format
        # - Version (1 byte): format version for future compatibility
        # - Flags (1 byte): bit 0 = compressed
        # - Rows (4 bytes, uint32)
        # - Cols (4 bytes, uint32)

        magic = b'VGM2'
        version = 1
        flags = 1 if compressed else 0

        header = struct.pack(
            '<4sBBII',
            magic,
            version,
            flags,
            self._shape[0],
            self._shape[1]
        )

        # Serialize data and mask
        data_bytes = self._data.tobytes()
        mask_bytes = np.packbits(self._mask).tobytes()  # Pack bools to bits for efficiency

        # Combine data and mask with length prefixes
        payload = struct.pack('<I', len(data_bytes)) + data_bytes
        payload += struct.pack('<I', len(mask_bytes)) + mask_bytes

        if compressed:
            payload = zlib.compress(payload, level=6)

        return header + payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "VGMatrix2D":
        """
        Deserialize a matrix from binary format.

        Args:
            data: Bytes object containing serialized matrix data.

        Returns:
            New VGMatrix2D instance with the deserialized data.

        Raises:
            ValueError: If the data is invalid or corrupted.

        Example:
            >>> data = matrix.to_bytes()
            >>> restored = VGMatrix2D.from_bytes(data)
            >>> assert matrix == restored
        """
        import struct
        import zlib

        if len(data) < 14:  # Minimum header size
            raise ValueError("Data too short to be a valid VGMatrix2D")

        # Parse header
        magic, version, flags, rows, cols = struct.unpack('<4sBBII', data[:14])

        if magic != b'VGM2':
            raise ValueError(f"Invalid magic number: {magic}. Expected 'VGM2'")

        if version > 1:
            raise ValueError(f"Unsupported format version: {version}")

        compressed = bool(flags & 1)
        payload = data[14:]

        if compressed:
            try:
                payload = zlib.decompress(payload)
            except zlib.error as e:
                raise ValueError(f"Failed to decompress data: {e}")

        # Parse payload
        offset = 0

        # Read data array
        data_len = struct.unpack('<I', payload[offset:offset + 4])[0]
        offset += 4
        data_bytes = payload[offset:offset + data_len]
        offset += data_len

        # Read mask array
        mask_len = struct.unpack('<I', payload[offset:offset + 4])[0]
        offset += 4
        mask_bytes = payload[offset:offset + mask_len]

        # Reconstruct arrays
        matrix_data = np.frombuffer(data_bytes, dtype=np.float64).reshape((rows, cols))
        mask_packed = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask_unpacked = np.unpackbits(mask_packed)[:rows * cols].reshape((rows, cols))

        # Create matrix
        result = cls.__new__(cls)
        result._shape = (rows, cols)
        result._data = matrix_data.copy()
        result._mask = mask_unpacked.astype(np.bool_)

        return result

    def to_dict(self) -> dict:
        """
        Serialize the matrix to a dictionary (JSON-compatible).

        This format is human-readable and suitable for JSON serialization.
        For large matrices, consider using to_bytes() instead.

        Returns:
            Dictionary containing the matrix data.

        Example:
            >>> matrix = VGMatrix2D((10, 10), 0.5)
            >>> d = matrix.to_dict()
            >>> import json
            >>> json_str = json.dumps(d)
        """
        import base64

        return {
            "type": "VGMatrix2D",
            "version": 1,
            "shape": list(self._shape),
            "data": base64.b64encode(self._data.tobytes()).decode('ascii'),
            "mask": base64.b64encode(np.packbits(self._mask).tobytes()).decode('ascii'),
            "dtype": str(self._data.dtype),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VGMatrix2D":
        """
        Deserialize a matrix from a dictionary.

        Args:
            d: Dictionary containing serialized matrix data.

        Returns:
            New VGMatrix2D instance.

        Raises:
            ValueError: If the dictionary is invalid.

        Example:
            >>> d = matrix.to_dict()
            >>> restored = VGMatrix2D.from_dict(d)
        """
        import base64

        if d.get("type") != "VGMatrix2D":
            raise ValueError(f"Invalid type: {d.get('type')}. Expected 'VGMatrix2D'")

        version = d.get("version", 1)
        if version > 1:
            raise ValueError(f"Unsupported version: {version}")

        shape = tuple(d["shape"])
        rows, cols = shape

        # Decode data
        data_bytes = base64.b64decode(d["data"])
        mask_bytes = base64.b64decode(d["mask"])

        matrix_data = np.frombuffer(data_bytes, dtype=np.float64).reshape(shape)
        mask_packed = np.frombuffer(mask_bytes, dtype=np.uint8)
        mask_unpacked = np.unpackbits(mask_packed)[:rows * cols].reshape(shape)

        # Create matrix
        result = cls.__new__(cls)
        result._shape = shape
        result._data = matrix_data.copy()
        result._mask = mask_unpacked.astype(np.bool_)

        return result

    def to_json(self, indent: Optional[int] = None) -> str:
        """
        Serialize the matrix to a JSON string.

        Args:
            indent: Indentation level for pretty printing. None for compact.

        Returns:
            JSON string representation of the matrix.

        Example:
            >>> json_str = matrix.to_json(indent=2)
            >>> restored = VGMatrix2D.from_json(json_str)
        """
        import json
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> "VGMatrix2D":
        """
        Deserialize a matrix from a JSON string.

        Args:
            json_str: JSON string containing serialized matrix data.

        Returns:
            New VGMatrix2D instance.

        Example:
            >>> json_str = matrix.to_json()
            >>> restored = VGMatrix2D.from_json(json_str)
        """
        import json
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: str, format: str = "auto") -> None:
        """
        Save the matrix to a file.

        Args:
            filepath: Path to the file to save.
            format: File format - "auto" (detect from extension), "binary", "json", or "npy".

        Raises:
            ValueError: If the format is unknown.

        Example:
            >>> matrix.save("mymatrix.vgm")  # Binary format
            >>> matrix.save("mymatrix.json")  # JSON format
            >>> matrix.save("mymatrix.npy")   # NumPy format (no mask)
        """
        from pathlib import Path

        path = Path(filepath)

        if format == "auto":
            ext = path.suffix.lower()
            if ext in ('.vgm', '.vgmatrix', '.bin'):
                format = "binary"
            elif ext == '.json':
                format = "json"
            elif ext == '.npy':
                format = "npy"
            else:
                format = "binary"  # Default

        if format == "binary":
            with open(filepath, 'wb') as f:
                f.write(self.to_bytes(compressed=True))
        elif format == "json":
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.to_json(indent=2))
        elif format == "npy":
            # NumPy format - saves data only, unassigned values become NaN
            np.save(filepath, self.to_numpy(fill_unassigned=np.nan))
        else:
            raise ValueError(f"Unknown format: {format}. Use 'binary', 'json', or 'npy'")

    @classmethod
    def load(cls, filepath: str, format: str = "auto") -> "VGMatrix2D":
        """
        Load a matrix from a file.

        Args:
            filepath: Path to the file to load.
            format: File format - "auto" (detect from extension), "binary", "json", or "npy".

        Returns:
            New VGMatrix2D instance with loaded data.

        Raises:
            ValueError: If the format is unknown or the file is invalid.
            FileNotFoundError: If the file doesn't exist.

        Example:
            >>> matrix = VGMatrix2D.load("mymatrix.vgm")
            >>> matrix = VGMatrix2D.load("mymatrix.json")
        """
        from pathlib import Path

        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        if format == "auto":
            ext = path.suffix.lower()
            if ext in ('.vgm', '.vgmatrix', '.bin'):
                format = "binary"
            elif ext == '.json':
                format = "json"
            elif ext == '.npy':
                format = "npy"
            else:
                # Try to detect by reading first bytes
                with open(filepath, 'rb') as f:
                    header = f.read(4)
                if header == b'VGM2':
                    format = "binary"
                else:
                    format = "json"  # Assume JSON

        if format == "binary":
            with open(filepath, 'rb') as f:
                return cls.from_bytes(f.read())
        elif format == "json":
            with open(filepath, 'r', encoding='utf-8') as f:
                return cls.from_json(f.read())
        elif format == "npy":
            # Load NumPy array, NaN values become unassigned
            data = np.load(filepath)
            mask = ~np.isnan(data)
            data = np.nan_to_num(data, nan=0.0)
            return cls.from_numpy(data, mask=mask)
        else:
            raise ValueError(f"Unknown format: {format}. Use 'binary', 'json', or 'npy'")

    def __getstate__(self) -> dict:
        """
        Support for pickle serialization.

        Returns:
            Dictionary containing the object state.
        """
        return {
            'shape': self._shape,
            'data': self._data,
            'mask': self._mask,
        }

    def __setstate__(self, state: dict) -> None:
        """
        Support for pickle deserialization.

        Args:
            state: Dictionary containing the object state.
        """
        self._shape = state['shape']
        self._data = state['data']
        self._mask = state['mask']


