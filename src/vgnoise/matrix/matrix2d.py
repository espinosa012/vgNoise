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
