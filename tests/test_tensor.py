"""
Unit tests for CodeSutra Tensor type (Phase 1: NumPy CPU backend)
"""

import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensor_value import TensorValue, HAS_NUMPY, HAS_TORCH

# Skip all tests if NumPy not available
pytestmark = pytest.mark.skipif(not HAS_NUMPY, reason="NumPy required for tensor tests")

import numpy as np


class TestTensorConstruction:
    """Test tensor creation and initialization."""
    
    def test_tensor_from_list_1d(self):
        """Create 1D tensor from list."""
        t = TensorValue([1, 2, 3], backend="numpy")
        assert t.shape == [3]
        assert t.dtype == "int64" or t.dtype == "int32"
    
    def test_tensor_from_list_2d(self):
        """Create 2D tensor from nested list."""
        t = TensorValue([[1, 2], [3, 4]], backend="numpy")
        assert t.shape == [2, 2]
    
    def test_tensor_from_numpy_array(self):
        """Create tensor from existing NumPy array."""
        arr = np.array([1, 2, 3])
        t = TensorValue(arr, backend="numpy")
        assert t.shape == [3]
    
    def test_tensor_dtype_float(self):
        """Test float tensor dtype."""
        t = TensorValue([1.0, 2.0, 3.0], backend="numpy")
        assert "float" in t.dtype
    
    def test_tensor_dtype_explicit(self):
        """Create tensor with explicit dtype."""
        t = TensorValue(np.array([1, 2, 3], dtype=np.float32), backend="numpy")
        assert "float32" in t.dtype or "float32" == t.dtype


class TestTensorProperties:
    """Test tensor introspection properties."""
    
    def test_shape_property(self):
        """Test .shape property."""
        t = TensorValue([[1, 2, 3], [4, 5, 6]], backend="numpy")
        assert t.shape == [2, 3]
    
    def test_ndim_property(self):
        """Test .ndim (number of dimensions)."""
        t_1d = TensorValue([1, 2, 3], backend="numpy")
        t_2d = TensorValue([[1, 2], [3, 4]], backend="numpy")
        t_3d = TensorValue([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], backend="numpy")
        assert t_1d.ndim == 1
        assert t_2d.ndim == 2
        assert t_3d.ndim == 3
    
    def test_dtype_property(self):
        """Test .dtype property."""
        t = TensorValue([1, 2, 3], backend="numpy")
        assert isinstance(t.dtype, str)
        assert len(t.dtype) > 0
    
    def test_device_property_cpu(self):
        """Test .device property returns 'cpu' for NumPy."""
        t = TensorValue([1, 2, 3], backend="numpy")
        assert t.device == "cpu"
    
    def test_size_method(self):
        """Test .size() method returns total elements."""
        t = TensorValue([[1, 2, 3], [4, 5, 6]], backend="numpy")
        assert t.size() == 6


class TestTensorConversion:
    """Test tensor conversion to/from Python types."""
    
    def test_to_list_1d(self):
        """Convert 1D tensor to list."""
        t = TensorValue([1, 2, 3], backend="numpy")
        lst = t.to_list()
        assert lst == [1, 2, 3]
        assert isinstance(lst, list)
    
    def test_to_list_2d(self):
        """Convert 2D tensor to nested list."""
        t = TensorValue([[1, 2], [3, 4]], backend="numpy")
        lst = t.to_list()
        assert lst == [[1, 2], [3, 4]]
        assert isinstance(lst, list)
    
    def test_roundtrip_list_to_tensor_to_list(self):
        """Test conversion roundtrip: list -> tensor -> list."""
        original = [[1, 2, 3], [4, 5, 6]]
        t = TensorValue(original, backend="numpy")
        recovered = t.to_list()
        assert recovered == original


class TestTensorArithmetic:
    """Test arithmetic operations on tensors."""
    
    def test_addition_tensors(self):
        """Add two tensors."""
        t1 = TensorValue([1, 2, 3], backend="numpy")
        t2 = TensorValue([4, 5, 6], backend="numpy")
        result = t1 + t2
        assert result.to_list() == [5, 7, 9]
    
    def test_subtraction_tensors(self):
        """Subtract two tensors."""
        t1 = TensorValue([5, 6, 7], backend="numpy")
        t2 = TensorValue([1, 2, 3], backend="numpy")
        result = t1 - t2
        assert result.to_list() == [4, 4, 4]
    
    def test_multiplication_tensors(self):
        """Multiply two tensors element-wise."""
        t1 = TensorValue([1, 2, 3], backend="numpy")
        t2 = TensorValue([2, 2, 2], backend="numpy")
        result = t1 * t2
        assert result.to_list() == [2, 4, 6]
    
    def test_division_tensors(self):
        """Divide two tensors element-wise."""
        t1 = TensorValue([4.0, 6.0, 8.0], backend="numpy")
        t2 = TensorValue([2.0, 2.0, 2.0], backend="numpy")
        result = t1 / t2
        expected = [2.0, 3.0, 4.0]
        assert np.allclose(result.to_list(), expected)
    
    def test_power_tensors(self):
        """Power operation on tensors."""
        t1 = TensorValue([2, 3, 4], backend="numpy")
        t2 = TensorValue([2, 2, 2], backend="numpy")
        result = t1 ** t2
        assert result.to_list() == [4, 9, 16]
    
    def test_addition_scalar(self):
        """Add scalar to tensor."""
        t = TensorValue([1, 2, 3], backend="numpy")
        result = t + 10
        assert result.to_list() == [11, 12, 13]
    
    def test_multiplication_scalar(self):
        """Multiply tensor by scalar."""
        t = TensorValue([1, 2, 3], backend="numpy")
        result = t * 5
        assert result.to_list() == [5, 10, 15]
    
    def test_scalar_on_left_side(self):
        """Test scalar operator when it's on the left side."""
        t = TensorValue([2, 4, 6], backend="numpy")
        result = 2 * t
        assert result.to_list() == [4, 8, 12]
    
    def test_addition_list(self):
        """Add list to tensor (list auto-converts)."""
        t = TensorValue([1, 2, 3], backend="numpy")
        result = t + [1, 1, 1]
        assert result.to_list() == [2, 3, 4]


class TestTensorComparison:
    """Test comparison operations on tensors."""
    
    def test_less_than(self):
        """Test < operator."""
        t1 = TensorValue([1, 2, 3], backend="numpy")
        t2 = TensorValue([2, 2, 2], backend="numpy")
        result = t1 < t2
        assert result.to_list() == [True, False, False]
    
    def test_equal(self):
        """Test == operator."""
        t1 = TensorValue([1, 2, 3], backend="numpy")
        t2 = TensorValue([1, 2, 3], backend="numpy")
        result = t1 == t2
        assert result.to_list() == [True, True, True]
    
    def test_not_equal(self):
        """Test != operator."""
        t1 = TensorValue([1, 2, 3], backend="numpy")
        t2 = TensorValue([1, 3, 3], backend="numpy")
        result = t1 != t2
        assert result.to_list() == [False, True, False]


class TestTensorIndexing:
    """Test indexing and element access."""
    
    def test_index_1d(self):
        """Index into 1D tensor."""
        t = TensorValue([10, 20, 30], backend="numpy")
        assert t[0] == 10
        assert t[1] == 20
        assert t[2] == 30
    
    def test_index_2d(self):
        """Index into 2D tensor."""
        t = TensorValue([[1, 2, 3], [4, 5, 6]], backend="numpy")
        assert t[0, 0] == 1
        assert t[0, 2] == 3
        assert t[1, 1] == 5
    
    def test_slice_returns_tensor(self):
        """Slicing should return a new tensor."""
        t = TensorValue([[1, 2, 3], [4, 5, 6]], backend="numpy")
        row = t[0]
        assert isinstance(row, TensorValue)
        assert row.shape == [3]


class TestTensorMethods:
    """Test tensor methods like reshape, flatten, transpose."""
    
    def test_reshape(self):
        """Test reshape method."""
        t = TensorValue([1, 2, 3, 4, 5, 6], backend="numpy")
        reshaped = t.reshape([2, 3])
        assert reshaped.shape == [2, 3]
        assert reshaped.to_list() == [[1, 2, 3], [4, 5, 6]]
    
    def test_flatten(self):
        """Test flatten method."""
        t = TensorValue([[1, 2, 3], [4, 5, 6]], backend="numpy")
        flat = t.flatten()
        assert flat.shape == [6]
        assert flat.to_list() == [1, 2, 3, 4, 5, 6]
    
    def test_transpose_2d(self):
        """Test transpose on 2D tensor."""
        t = TensorValue([[1, 2, 3], [4, 5, 6]], backend="numpy")
        transposed = t.transpose()
        assert transposed.shape == [3, 2]
        assert transposed.to_list() == [[1, 4], [2, 5], [3, 6]]
    
    def test_sum_all(self):
        """Test sum() without axis."""
        t = TensorValue([1, 2, 3, 4, 5], backend="numpy")
        result = t.sum()
        assert result == 15
    
    def test_sum_axis(self):
        """Test sum(axis=...) on 2D tensor."""
        t = TensorValue([[1, 2, 3], [4, 5, 6]], backend="numpy")
        result = t.sum(axis=0)  # sum columns
        assert isinstance(result, TensorValue)
        assert result.to_list() == [5, 7, 9]
    
    def test_mean_all(self):
        """Test mean() without axis."""
        t = TensorValue([1.0, 2.0, 3.0, 4.0, 5.0], backend="numpy")
        result = t.mean()
        assert abs(result - 3.0) < 1e-6
    
    def test_max_all(self):
        """Test max()."""
        t = TensorValue([1, 5, 3, 2, 4], backend="numpy")
        result = t.max()
        assert result == 5
    
    def test_min_all(self):
        """Test min()."""
        t = TensorValue([1, 5, 3, 2, 4], backend="numpy")
        result = t.min()
        assert result == 1


class TestTensorPrettyPrint:
    """Test string representation of tensors."""
    
    def test_str_small_tensor(self):
        """Small tensors should print their content."""
        t = TensorValue([1, 2, 3], backend="numpy")
        s = str(t)
        assert "1" in s and "2" in s and "3" in s
    
    def test_repr(self):
        """Test repr format."""
        t = TensorValue([1, 2, 3], backend="numpy")
        r = repr(t)
        assert "Tensor" in r


class TestTensorDevice:
    """Test device-related operations."""
    
    def test_cpu_method_returns_copy(self):
        """cpu() should return a copy."""
        t = TensorValue([1, 2, 3], backend="numpy")
        t_cpu = t.cpu()
        assert isinstance(t_cpu, TensorValue)
        assert t_cpu.to_list() == [1, 2, 3]
    
    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_gpu_raises_without_cuda(self):
        """gpu() should raise if CUDA not available."""
        t = TensorValue([1, 2, 3], backend="numpy")
        # This may raise depending on CUDA availability
        # For now, just ensure no silent failure
        try:
            t_gpu = t.gpu()
            # If it succeeds, we have CUDA
            assert t_gpu.device in ("cuda", "cpu")
        except RuntimeError as e:
            # Expected if no CUDA
            assert "CUDA" in str(e) or "gpu" in str(e).lower()


class TestTensorBroadcasting:
    """Test broadcasting behavior."""
    
    def test_broadcast_scalar_to_tensor(self):
        """Scalar should broadcast to tensor shape."""
        t = TensorValue([[1, 2], [3, 4]], backend="numpy")
        result = t + 10
        assert result.to_list() == [[11, 12], [13, 14]]
    
    def test_broadcast_different_shapes_compatible(self):
        """Compatible shapes should broadcast."""
        t1 = TensorValue([[1], [2], [3]], backend="numpy")  # [3, 1]
        t2 = TensorValue([10, 20], backend="numpy")  # [2]
        result = t1 + t2
        # Should broadcast to [3, 2]
        assert result.shape[0] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
