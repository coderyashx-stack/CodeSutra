"""
TensorValue — First-class Tensor type for CodeSutra.

Backed by NumPy (CPU) and optionally PyTorch (GPU).
Provides a high-level, user-friendly API for multi-dimensional arrays.
"""

import sys

# Try to import NumPy and PyTorch; graceful fallback if missing
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


class TensorValue:
    """
    Represents a multi-dimensional array (tensor) in CodeSutra.
    
    Backed by either NumPy (CPU) or PyTorch (GPU).
    Provides:
      - Introspection: .shape, .dtype, .device, .ndim
      - Conversion: .to_list(), and to/from Python objects
      - Operators: arithmetic, comparison, indexing
      - Methods: reshape, transpose, flatten, gpu(), cpu(), etc.
    """

    def __init__(self, obj, backend=None):
        """
        Initialize a TensorValue.

        Args:
            obj: Underlying Python object (numpy.ndarray, torch.Tensor, or list)
            backend: "numpy", "torch", or auto-detected
        """
        if isinstance(obj, TensorValue):
            # If already a TensorValue, extract the object
            obj = obj._obj
            if backend is None:
                backend = obj._backend

        # Auto-detect or validate backend
        if backend is None:
            if HAS_TORCH and isinstance(obj, torch.Tensor):
                backend = "torch"
            elif HAS_NUMPY and isinstance(obj, np.ndarray):
                backend = "numpy"
            elif isinstance(obj, list):
                # Default to numpy for lists
                if HAS_NUMPY:
                    backend = "numpy"
                else:
                    raise RuntimeError(
                        "NumPy is required for tensor operations. "
                        "Install with: pip install numpy"
                    )
            else:
                raise TypeError(
                    f"Cannot create tensor from {type(obj)}. "
                    f"Expected list, numpy.ndarray, or torch.Tensor"
                )
        
        # Convert to appropriate backend object if needed
        if backend == "numpy":
            if isinstance(obj, list):
                obj = np.array(obj)
            elif not HAS_NUMPY:
                raise RuntimeError(
                    "NumPy is required for tensor operations. "
                    "Install with: pip install numpy"
                )
            elif not isinstance(obj, np.ndarray):
                raise TypeError(f"Expected list or numpy.ndarray for numpy backend, got {type(obj)}")
        elif backend == "torch":
            if isinstance(obj, list):
                if not HAS_TORCH:
                    raise RuntimeError("PyTorch required for torch tensors")
                obj = torch.tensor(obj)
            elif not HAS_TORCH:
                raise RuntimeError("PyTorch required for torch backend")
            elif not isinstance(obj, torch.Tensor):
                raise TypeError(f"Expected list or torch.Tensor for torch backend, got {type(obj)}")
        else:
            raise ValueError(f"Unknown backend: {backend}")

        self._backend = backend
        self._obj = obj

    @property
    def shape(self):
        """Return the shape of the tensor as a list."""
        if self._backend == "numpy":
            return list(self._obj.shape)
        elif self._backend == "torch":
            return list(self._obj.shape)
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    @property
    def ndim(self):
        """Return the number of dimensions."""
        if self._backend == "numpy":
            return self._obj.ndim
        elif self._backend == "torch":
            return self._obj.ndim
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    @property
    def dtype(self):
        """Return the data type as a string."""
        if self._backend == "numpy":
            return str(self._obj.dtype)
        elif self._backend == "torch":
            # Convert torch dtype to string
            dtype_map = {
                torch.float32: "float32",
                torch.float64: "float64",
                torch.int32: "int32",
                torch.int64: "int64",
                torch.int16: "int16",
                torch.uint8: "uint8",
                torch.bool: "bool",
            }
            return dtype_map.get(self._obj.dtype, str(self._obj.dtype))
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    @property
    def device(self):
        """Return the device as a string ('cpu' or 'cuda')."""
        if self._backend == "numpy":
            return "cpu"
        elif self._backend == "torch":
            return str(self._obj.device).split(":")[0]  # "cpu" or "cuda"
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def size(self):
        """Return the total number of elements."""
        if self._backend == "numpy":
            return self._obj.size
        elif self._backend == "torch":
            return self._obj.numel()
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def to_list(self):
        """Convert tensor to nested Python lists."""
        if self._backend == "numpy":
            return self._obj.tolist()
        elif self._backend == "torch":
            return self._obj.detach().cpu().numpy().tolist()
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def cpu(self):
        """Move tensor to CPU. Returns a new TensorValue."""
        if self._backend == "numpy":
            return TensorValue(self._obj.copy(), backend="numpy")
        elif self._backend == "torch":
            cpu_obj = self._obj.detach().cpu()
            return TensorValue(cpu_obj, backend="torch")
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def gpu(self):
        """
        Move tensor to GPU. Returns a new TensorValue.
        
        Raises RuntimeError if CUDA is not available.
        """
        if self._backend == "torch":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available. Cannot move tensor to GPU. "
                    "Check torch installation: torch.cuda.is_available()"
                )
            gpu_obj = self._obj.to("cuda")
            return TensorValue(gpu_obj, backend="torch")
        elif self._backend == "numpy":
            # Convert NumPy to PyTorch, then move to GPU
            if not HAS_TORCH:
                raise RuntimeError(
                    "PyTorch is required for GPU operations. "
                    "Install with: pip install torch"
                )
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available. Cannot move tensor to GPU."
                )
            torch_obj = torch.from_numpy(self._obj).to("cuda")
            return TensorValue(torch_obj, backend="torch")
        else:
            raise RuntimeError(f"Unknown backend: {self._backend}")

    def _coerce_other(self, other):
        """
        Coerce 'other' to a compatible form for binary operations.
        
        Returns (self_obj, other_obj, backend)
        """
        if isinstance(other, TensorValue):
            # Both are tensors; ensure same backend
            if self._backend != other._backend:
                # Promote to torch if one is torch, both are not torch
                if self._backend == "torch":
                    other = other.gpu()
                elif other._backend == "torch":
                    self_new = self.gpu()
                    return self_new._obj, other._obj, "torch"
            return self._obj, other._obj, self._backend
        elif isinstance(other, (int, float)):
            # Scalar; compatible with any tensor
            if self._backend == "numpy":
                return self._obj, other, "numpy"
            elif self._backend == "torch":
                return self._obj, torch.tensor(other, dtype=self._obj.dtype, device=self._obj.device), "torch"
        elif isinstance(other, list):
            # Convert list to tensor, then recurse
            other_tensor = TensorValue(other, backend=self._backend)
            return self._coerce_other(other_tensor)
        else:
            raise TypeError(
                f"Cannot perform operation between Tensor and {type(other).__name__}"
            )

    def _binary_op(self, op_name, other):
        """
        Perform a binary operation, delegating to NumPy/PyTorch.
        
        Args:
            op_name: Name of operation ("add", "sub", "mul", "div", "pow", etc.)
            other: Operand (TensorValue, number, or list)
        
        Returns:
            New TensorValue with result
        """
        self_obj, other_obj, backend = self._coerce_other(other)

        # Map operation names to methods/functions
        ops = {
            "add": lambda a, b: a + b,
            "sub": lambda a, b: a - b,
            "mul": lambda a, b: a * b,
            "div": lambda a, b: a / b,
            "floordiv": lambda a, b: a // b,
            "pow": lambda a, b: a ** b,
            "mod": lambda a, b: a % b,
            "lt": lambda a, b: a < b,
            "le": lambda a, b: a <= b,
            "eq": lambda a, b: a == b,
            "ne": lambda a, b: a != b,
            "gt": lambda a, b: a > b,
            "ge": lambda a, b: a >= b,
        }

        if op_name not in ops:
            raise NotImplementedError(f"Operation {op_name} not implemented for tensors")

        try:
            result_obj = ops[op_name](self_obj, other_obj)
            return TensorValue(result_obj, backend=backend)
        except Exception as e:
            raise RuntimeError(f"Tensor operation {op_name} failed: {e}")

    def _unary_op(self, op_name):
        """
        Perform a unary operation.
        
        Args:
            op_name: Name of operation ("neg", "abs", etc.)
        
        Returns:
            New TensorValue with result
        """
        ops = {
            "neg": lambda a: -a,
            "abs": lambda a: np.abs(a) if self._backend == "numpy" else torch.abs(a),
            "not": lambda a: ~a if self._backend == "numpy" else ~a,
        }

        if op_name not in ops:
            raise NotImplementedError(f"Operation {op_name} not implemented for tensors")

        try:
            result_obj = ops[op_name](self._obj)
            return TensorValue(result_obj, backend=self._backend)
        except Exception as e:
            raise RuntimeError(f"Tensor operation {op_name} failed: {e}")

    def __add__(self, other):
        return self._binary_op("add", other)

    def __radd__(self, other):
        return self._binary_op("add", other)

    def __sub__(self, other):
        return self._binary_op("sub", other)

    def __rsub__(self, other):
        # other - self
        temp = TensorValue(other) if not isinstance(other, TensorValue) else other
        return temp._binary_op("sub", self)

    def __mul__(self, other):
        return self._binary_op("mul", other)

    def __rmul__(self, other):
        return self._binary_op("mul", other)

    def __truediv__(self, other):
        return self._binary_op("div", other)

    def __rtruediv__(self, other):
        temp = TensorValue(other) if not isinstance(other, TensorValue) else other
        return temp._binary_op("div", self)

    def __floordiv__(self, other):
        return self._binary_op("floordiv", other)

    def __rfloordiv__(self, other):
        temp = TensorValue(other) if not isinstance(other, TensorValue) else other
        return temp._binary_op("floordiv", self)

    def __pow__(self, other):
        return self._binary_op("pow", other)

    def __rpow__(self, other):
        temp = TensorValue(other) if not isinstance(other, TensorValue) else other
        return temp._binary_op("pow", self)

    def __mod__(self, other):
        return self._binary_op("mod", other)

    def __rmod__(self, other):
        temp = TensorValue(other) if not isinstance(other, TensorValue) else other
        return temp._binary_op("mod", self)

    def __lt__(self, other):
        return self._binary_op("lt", other)

    def __le__(self, other):
        return self._binary_op("le", other)

    def __eq__(self, other):
        return self._binary_op("eq", other)

    def __ne__(self, other):
        return self._binary_op("ne", other)

    def __gt__(self, other):
        return self._binary_op("gt", other)

    def __ge__(self, other):
        return self._binary_op("ge", other)

    def __neg__(self):
        return self._unary_op("neg")

    def __abs__(self):
        return self._unary_op("abs")

    def __invert__(self):
        return self._unary_op("not")

    def __getitem__(self, key):
        """Support indexing: t[0], t[0, 1], t[:, 1], etc."""
        try:
            result = self._obj[key]
            # If result is a scalar, unwrap it
            if isinstance(result, (np.ndarray, torch.Tensor)):
                if result.ndim == 0:
                    # Scalar — unwrap to Python native
                    if self._backend == "numpy":
                        return result.item()
                    elif self._backend == "torch":
                        return result.item()
                else:
                    return TensorValue(result, backend=self._backend)
            else:
                # Already a scalar (e.g., numpy scalar type)
                if hasattr(result, "item"):
                    return result.item()
                return result
        except Exception as e:
            raise RuntimeError(f"Tensor indexing failed: {e}")

    def reshape(self, shape):
        """Reshape tensor to given shape. Returns new TensorValue."""
        try:
            if self._backend == "numpy":
                new_obj = self._obj.reshape(shape)
            elif self._backend == "torch":
                new_obj = self._obj.reshape(shape)
            else:
                raise RuntimeError(f"Unknown backend: {self._backend}")
            return TensorValue(new_obj, backend=self._backend)
        except Exception as e:
            raise RuntimeError(f"Reshape failed: {e}")

    def flatten(self):
        """Flatten tensor to 1D. Returns new TensorValue."""
        try:
            if self._backend == "numpy":
                new_obj = self._obj.flatten()
            elif self._backend == "torch":
                new_obj = self._obj.flatten()
            else:
                raise RuntimeError(f"Unknown backend: {self._backend}")
            return TensorValue(new_obj, backend=self._backend)
        except Exception as e:
            raise RuntimeError(f"Flatten failed: {e}")

    def transpose(self, *axes):
        """
        Transpose tensor. If axes given, permute; otherwise swap last two dims.
        Returns new TensorValue.
        """
        try:
            if axes:
                if self._backend == "numpy":
                    new_obj = np.transpose(self._obj, axes)
                elif self._backend == "torch":
                    new_obj = torch.permute(self._obj, axes)
                else:
                    raise RuntimeError(f"Unknown backend: {self._backend}")
            else:
                # Default: reverse all dimensions
                if self._backend == "numpy":
                    new_obj = np.transpose(self._obj)
                elif self._backend == "torch":
                    new_obj = torch.transpose(self._obj, 0, 1) if self._obj.ndim == 2 else torch.permute(self._obj, tuple(range(self._obj.ndim)[::-1]))
                else:
                    raise RuntimeError(f"Unknown backend: {self._backend}")
            return TensorValue(new_obj, backend=self._backend)
        except Exception as e:
            raise RuntimeError(f"Transpose failed: {e}")

    def sum(self, axis=None, keepdims=False):
        """
        Sum tensor along axis. If axis is None, sum all elements.
        Returns new TensorValue or scalar (if result is 0-D).
        """
        try:
            if self._backend == "numpy":
                result = np.sum(self._obj, axis=axis, keepdims=keepdims)
            elif self._backend == "torch":
                if axis is None:
                    result = torch.sum(self._obj)
                else:
                    result = torch.sum(self._obj, dim=axis, keepdim=keepdims)
            else:
                raise RuntimeError(f"Unknown backend: {self._backend}")
            
            # If result is scalar, return as TensorValue but unwrap on __str__
            if hasattr(result, "ndim"):
                if result.ndim == 0:
                    # Unwrap scalar
                    if hasattr(result, "item"):
                        return result.item()
                    return result
            return TensorValue(result, backend=self._backend) if isinstance(result, (np.ndarray, torch.Tensor)) else result
        except Exception as e:
            raise RuntimeError(f"Sum failed: {e}")

    def mean(self, axis=None, keepdims=False):
        """
        Mean of tensor along axis. If axis is None, mean of all elements.
        Returns scalar or new TensorValue.
        """
        try:
            if self._backend == "numpy":
                result = np.mean(self._obj, axis=axis, keepdims=keepdims)
            elif self._backend == "torch":
                if axis is None:
                    result = torch.mean(self._obj.float())
                else:
                    result = torch.mean(self._obj.float(), dim=axis, keepdim=keepdims)
            else:
                raise RuntimeError(f"Unknown backend: {self._backend}")
            
            if hasattr(result, "ndim"):
                if result.ndim == 0:
                    if hasattr(result, "item"):
                        return result.item()
                    return result
            return TensorValue(result, backend=self._backend) if isinstance(result, (np.ndarray, torch.Tensor)) else result
        except Exception as e:
            raise RuntimeError(f"Mean failed: {e}")

    def max(self, axis=None, keepdims=False):
        """Max along axis. If axis is None, max of all elements. Returns scalar or TensorValue."""
        try:
            if self._backend == "numpy":
                result = np.max(self._obj, axis=axis, keepdims=keepdims)
            elif self._backend == "torch":
                if axis is None:
                    result = torch.max(self._obj)
                else:
                    result = torch.max(self._obj, dim=axis, keepdim=keepdims)[0]
            else:
                raise RuntimeError(f"Unknown backend: {self._backend}")
            
            if hasattr(result, "ndim"):
                if result.ndim == 0:
                    if hasattr(result, "item"):
                        return result.item()
                    return result
            return TensorValue(result, backend=self._backend) if isinstance(result, (np.ndarray, torch.Tensor)) else result
        except Exception as e:
            raise RuntimeError(f"Max failed: {e}")

    def min(self, axis=None, keepdims=False):
        """Min along axis. If axis is None, min of all elements. Returns scalar or TensorValue."""
        try:
            if self._backend == "numpy":
                result = np.min(self._obj, axis=axis, keepdims=keepdims)
            elif self._backend == "torch":
                if axis is None:
                    result = torch.min(self._obj)
                else:
                    result = torch.min(self._obj, dim=axis, keepdim=keepdims)[0]
            else:
                raise RuntimeError(f"Unknown backend: {self._backend}")
            
            if hasattr(result, "ndim"):
                if result.ndim == 0:
                    if hasattr(result, "item"):
                        return result.item()
                    return result
            return TensorValue(result, backend=self._backend) if isinstance(result, (np.ndarray, torch.Tensor)) else result
        except Exception as e:
            raise RuntimeError(f"Min failed: {e}")

    def __str__(self):
        """Pretty-print the tensor."""
        if self._backend == "numpy":
            arr = self._obj
        elif self._backend == "torch":
            arr = self._obj.detach().cpu().numpy()
        else:
            return f"<Tensor backend={self._backend}>"

        # Small tensors: show full content
        if arr.size < 100:
            content = str(arr)
            return content
        # Large tensors: show summary
        else:
            return f"<Tensor shape={list(arr.shape)} dtype={arr.dtype} device={self.device}>"

    def __repr__(self):
        return f"Tensor({self.__str__()})"
