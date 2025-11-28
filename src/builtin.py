"""
CodeSutra Standard Library - Built-in functions
"""

import math
import re
from typing import Any, List, Dict, Callable, Optional
from python_bridge import wrap


class TensorModule:
    """Callable module for tensor creation and factory functions."""
    
    def __call__(self, value: Any, dtype: Optional[str] = None, device: str = "cpu"):
        """Allow tensor(...) calls."""
        return BuiltinLibrary.tensor(value, dtype, device)
    
    def __getattr__(self, name: str):
        """Support tensor.zeros, tensor.ones, etc."""
        methods = {
            'zeros': BuiltinLibrary.tensor_zeros,
            'ones': BuiltinLibrary.tensor_ones,
            'arange': BuiltinLibrary.tensor_arange,
            'random': BuiltinLibrary.tensor_random,
        }
        if name in methods:
            return methods[name]
        raise AttributeError(f"Tensor has no attribute '{name}'")


class CodeSutraFunction:
    """Represents a callable function value"""
    
    def __init__(self, params: List[str], body, closure):
        self.params = params
        self.body = body
        self.closure = closure
    
    def __repr__(self):
        return f"<function with {len(self.params)} params>"


class BuiltinLibrary:
    """Standard library for CodeSutra"""
    
    @staticmethod
    def to_number(value: Any) -> float:
        """Convert value to number"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        elif isinstance(value, str):
            try:
                if '.' in value:
                    return float(value)
                return float(int(value))
            except:
                return float('nan')
        elif value is None:
            return 0.0
        else:
            return float('nan')
    
    @staticmethod
    def to_string(value: Any) -> str:
        """Convert value to string"""
        if isinstance(value, str):
            return value
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif value is None:
            return "nil"
        elif isinstance(value, list):
            elements = ', '.join(BuiltinLibrary.to_string(v) for v in value)
            return f"[{elements}]"
        elif isinstance(value, dict):
            pairs = ', '.join(f"{k}: {BuiltinLibrary.to_string(v)}" for k, v in value.items())
            return f"{{{pairs}}}"
        elif isinstance(value, CodeSutraFunction):
            return str(value)
        else:
            return str(value)
    
    @staticmethod
    def to_bool(value: Any) -> bool:
        """Convert value to boolean (truthiness)"""
        if value is None or value is False:
            return False
        if value == 0 or value == "" or value == []:
            return False
        return True
    
    @staticmethod
    def type_of(value: Any) -> str:
        """Get the type of a value"""
        if value is None:
            return "nil"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "dict"
        elif isinstance(value, CodeSutraFunction):
            return "function"
        else:
            return "object"
    
    @staticmethod
    def length(value: Any) -> int:
        """Get length of value"""
        if isinstance(value, (str, list)):
            return len(value)
        elif isinstance(value, dict):
            return len(value)
        elif value is None:
            return 0
        else:
            raise TypeError(f"length not defined for {BuiltinLibrary.type_of(value)}")
    
    @staticmethod
    def push(array: list, value: Any) -> list:
        """Add element to array"""
        array.append(value)
        return array
    
    @staticmethod
    def pop(array: list) -> Any:
        """Remove and return last element"""
        if not array:
            return None
        return array.pop()
    
    @staticmethod
    def shift(array: list) -> Any:
        """Remove and return first element"""
        if not array:
            return None
        return array.pop(0)
    
    @staticmethod
    def unshift(array: list, value: Any) -> list:
        """Add element to beginning of array"""
        array.insert(0, value)
        return array
    
    @staticmethod
    def join(array: list, separator: str = "") -> str:
        """Join array elements into string"""
        return separator.join(BuiltinLibrary.to_string(v) for v in array)
    
    @staticmethod
    def split(string: str, separator: str = "") -> list:
        """Split string into array"""
        if separator == "":
            return list(string)
        return string.split(separator)
    
    @staticmethod
    def range(start: float, end: float = None, step: float = 1) -> list:
        """Create range of numbers"""
        if end is None:
            end = start
            start = 0
        
        result = []
        if step > 0:
            current = start
            while current < end:
                result.append(int(current) if current == int(current) else current)
                current += step
        elif step < 0:
            current = start
            while current > end:
                result.append(int(current) if current == int(current) else current)
                current += step
        
        return result
    
    @staticmethod
    def reverse(array: list) -> list:
        """Reverse an array"""
        return list(reversed(array))
    
    @staticmethod
    def sort(array: list, key: Optional[Callable] = None) -> list:
        """Sort an array"""
        if key:
            return sorted(array, key=key)
        return sorted(array)
    
    @staticmethod
    def keys(dict_obj: dict) -> list:
        """Get keys from dictionary"""
        return list(dict_obj.keys())
    
    @staticmethod
    def values(dict_obj: dict) -> list:
        """Get values from dictionary"""
        return list(dict_obj.values())
    
    @staticmethod
    def has(dict_obj: dict, key: str) -> bool:
        """Check if dictionary has key"""
        return key in dict_obj
    
    # Math functions
    @staticmethod
    def sqrt(value: float) -> float:
        """Square root"""
        return math.sqrt(BuiltinLibrary.to_number(value))
    
    @staticmethod
    def pow(base: float, exponent: float) -> float:
        """Power function"""
        return math.pow(BuiltinLibrary.to_number(base), BuiltinLibrary.to_number(exponent))
    
    @staticmethod
    def abs(value: float) -> float:
        """Absolute value"""
        return abs(BuiltinLibrary.to_number(value))
    
    @staticmethod
    def floor(value: float) -> int:
        """Floor function"""
        return math.floor(BuiltinLibrary.to_number(value))
    
    @staticmethod
    def ceil(value: float) -> int:
        """Ceiling function"""
        return math.ceil(BuiltinLibrary.to_number(value))
    
    @staticmethod
    def round(value: float, decimals: int = 0) -> float:
        """Round function"""
        num = BuiltinLibrary.to_number(value)
        return round(num, int(BuiltinLibrary.to_number(decimals)))
    
    @staticmethod
    def min(*args) -> float:
        """Minimum value - works with tensors and numbers"""
        from tensor_value import TensorValue
        
        if not args:
            return None
        
        # If first arg is a tensor, call its min method
        if len(args) == 1 and isinstance(args[0], TensorValue):
            return args[0].min()
        
        # Otherwise work with numbers
        nums = [BuiltinLibrary.to_number(v) for v in args]
        return min(nums)
    
    @staticmethod
    def max(*args) -> float:
        """Maximum value - works with tensors and numbers"""
        from tensor_value import TensorValue
        
        if not args:
            return None
        
        # If first arg is a tensor, call its max method
        if len(args) == 1 and isinstance(args[0], TensorValue):
            return args[0].max()
        
        # Otherwise work with numbers
        nums = [BuiltinLibrary.to_number(v) for v in args]
        return max(nums)
    
    @staticmethod
    def sum(*args) -> float:
        """Sum values - works with tensors, lists, and numbers"""
        from tensor_value import TensorValue
        
        if not args:
            return 0
        
        # If single tensor, call its sum method
        if len(args) == 1 and isinstance(args[0], TensorValue):
            return args[0].sum()
        
        # If single list, sum its elements
        if len(args) == 1 and isinstance(args[0], list):
            return sum(BuiltinLibrary.to_number(v) for v in args[0])
        
        # Otherwise sum all arguments
        return sum(BuiltinLibrary.to_number(v) for v in args)
    
    @staticmethod
    def mean(*args) -> float:
        """Mean/average - works with tensors, lists, and numbers"""
        from tensor_value import TensorValue
        
        if not args:
            return 0
        
        # If single tensor, call its mean method
        if len(args) == 1 and isinstance(args[0], TensorValue):
            return args[0].mean()
        
        # If single list, calculate mean
        if len(args) == 1 and isinstance(args[0], list):
            if not args[0]:
                return 0
            return sum(BuiltinLibrary.to_number(v) for v in args[0]) / len(args[0])
        
        # Otherwise calculate mean of all arguments
        if not args:
            return 0
        return sum(BuiltinLibrary.to_number(v) for v in args) / len(args)
    
    @staticmethod
    def min_old(*args) -> float:
        """Minimum value"""
        if not args:
            return None
        nums = [BuiltinLibrary.to_number(v) for v in args]
        return min(nums)
    
    @staticmethod
    def max_old(*args) -> float:
        """Maximum value"""
        if not args:
            return None
        nums = [BuiltinLibrary.to_number(v) for v in args]
        return max(nums)
    
    @staticmethod
    def sin(value: float) -> float:
        """Sine function"""
        return math.sin(BuiltinLibrary.to_number(value))
    
    @staticmethod
    def cos(value: float) -> float:
        """Cosine function"""
        return math.cos(BuiltinLibrary.to_number(value))
    
    @staticmethod
    def tan(value: float) -> float:
        """Tangent function"""
        return math.tan(BuiltinLibrary.to_number(value))
    
    @staticmethod
    def log(value: float, base: float = math.e) -> float:
        """Logarithm function"""
        return math.log(BuiltinLibrary.to_number(value), BuiltinLibrary.to_number(base))
    
    @staticmethod
    def exp(value: float) -> float:
        """Exponential function"""
        return math.exp(BuiltinLibrary.to_number(value))
    
    @staticmethod
    def random(min_val: float = 0, max_val: float = 1) -> float:
        """Generate random number"""
        import random as py_random
        min_num = BuiltinLibrary.to_number(min_val)
        max_num = BuiltinLibrary.to_number(max_val)
        return py_random.uniform(min_num, max_num)
    
    # String functions
    @staticmethod
    def upper(string: str) -> str:
        """Convert to uppercase"""
        return BuiltinLibrary.to_string(string).upper()
    
    @staticmethod
    def lower(string: str) -> str:
        """Convert to lowercase"""
        return BuiltinLibrary.to_string(string).lower()
    
    @staticmethod
    def trim(string: str) -> str:
        """Trim whitespace"""
        return BuiltinLibrary.to_string(string).strip()
    
    @staticmethod
    def starts_with(string: str, prefix: str) -> bool:
        """Check if string starts with prefix"""
        return BuiltinLibrary.to_string(string).startswith(BuiltinLibrary.to_string(prefix))
    
    @staticmethod
    def ends_with(string: str, suffix: str) -> bool:
        """Check if string ends with suffix"""
        return BuiltinLibrary.to_string(string).endswith(BuiltinLibrary.to_string(suffix))
    
    @staticmethod
    def contains(string: str, substring: str) -> bool:
        """Check if string contains substring"""
        return BuiltinLibrary.to_string(substring) in BuiltinLibrary.to_string(string)
    
    @staticmethod
    def index_of(string: str, substring: str) -> int:
        """Find index of substring"""
        s = BuiltinLibrary.to_string(string)
        sub = BuiltinLibrary.to_string(substring)
        try:
            return s.index(sub)
        except ValueError:
            return -1
    
    @staticmethod
    def substring(string: str, start: int, end: int = None) -> str:
        """Extract substring"""
        s = BuiltinLibrary.to_string(string)
        start_idx = int(BuiltinLibrary.to_number(start))
        if end is None:
            return s[start_idx:]
        end_idx = int(BuiltinLibrary.to_number(end))
        return s[start_idx:end_idx]
    
    @staticmethod
    def replace(string: str, search: str, replacement: str) -> str:
        """Replace substring"""
        s = BuiltinLibrary.to_string(string)
        search_str = BuiltinLibrary.to_string(search)
        replace_str = BuiltinLibrary.to_string(replacement)
        return s.replace(search_str, replace_str)
    
    @staticmethod
    def char_at(string: str, index: int) -> str:
        """Get character at index"""
        s = BuiltinLibrary.to_string(string)
        idx = int(BuiltinLibrary.to_number(index))
        if 0 <= idx < len(s):
            return s[idx]
        return ""
    
    @staticmethod
    def repeat(string: str, count: int) -> str:
        """Repeat string"""
        s = BuiltinLibrary.to_string(string)
        c = int(BuiltinLibrary.to_number(count))
        return s * c
    
    @staticmethod
    def tensor(value: Any, dtype: Optional[str] = None, device: str = "cpu"):
        """
        Create a tensor from a list, or wrap a numpy/torch array.
        
        Args:
            value: A CodeSutra list or Python array-like
            dtype: Optional dtype string (e.g., "float32", "int64")
            device: "cpu" or "cuda"
        
        Returns:
            TensorValue
        """
        from tensor_value import TensorValue, HAS_NUMPY, HAS_TORCH, np, torch
        
        # Convert CodeSutra list or PyProxy to Python native
        if isinstance(value, list):
            value = value  # already a list
        else:
            # Try to unwrap PyProxy or other types
            value_unwrapped = __py_unwrap(value) if not isinstance(value, list) else value
            value = value_unwrapped
        
        # Determine backend
        backend = None
        if device.startswith("cuda"):
            if not HAS_TORCH:
                raise RuntimeError(
                    "PyTorch is required for GPU tensors. "
                    "Install with: pip install torch"
                )
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available on this system.")
            backend = "torch"
        else:
            if not HAS_NUMPY:
                raise RuntimeError(
                    "NumPy is required for tensors. "
                    "Install with: pip install numpy"
                )
            backend = "numpy"
        
        # Map dtype string to backend dtype
        if backend == "numpy":
            if dtype:
                value = np.array(value, dtype=dtype)
            else:
                value = np.array(value)
        elif backend == "torch":
            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "int32": torch.int32,
                "int64": torch.int64,
                "int16": torch.int16,
                "uint8": torch.uint8,
                "bool": torch.bool,
            }
            torch_dtype = dtype_map.get(dtype, None) if dtype else None
            value = torch.tensor(value, dtype=torch_dtype, device=device)
        
        return TensorValue(value, backend=backend)
    
    @staticmethod
    def tensor_zeros(shape, dtype: Optional[str] = None, device: str = "cpu"):
        """Create a tensor of zeros."""
        from tensor_value import TensorValue, HAS_NUMPY, HAS_TORCH, np, torch
        
        backend = "torch" if device.startswith("cuda") else "numpy"
        
        if backend == "numpy":
            if dtype:
                obj = np.zeros(shape, dtype=dtype)
            else:
                obj = np.zeros(shape)
        else:
            if not HAS_TORCH:
                raise RuntimeError("PyTorch required for GPU tensors")
            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "int32": torch.int32,
                "int64": torch.int64,
            }
            torch_dtype = dtype_map.get(dtype, torch.float32)
            obj = torch.zeros(shape, dtype=torch_dtype, device=device)
        
        return TensorValue(obj, backend=backend)
    
    @staticmethod
    def tensor_ones(shape, dtype: Optional[str] = None, device: str = "cpu"):
        """Create a tensor of ones."""
        from tensor_value import TensorValue, HAS_NUMPY, HAS_TORCH, np, torch
        
        backend = "torch" if device.startswith("cuda") else "numpy"
        
        if backend == "numpy":
            if dtype:
                obj = np.ones(shape, dtype=dtype)
            else:
                obj = np.ones(shape)
        else:
            if not HAS_TORCH:
                raise RuntimeError("PyTorch required for GPU tensors")
            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "int32": torch.int32,
                "int64": torch.int64,
            }
            torch_dtype = dtype_map.get(dtype, torch.float32)
            obj = torch.ones(shape, dtype=torch_dtype, device=device)
        
        return TensorValue(obj, backend=backend)
    
    @staticmethod
    def tensor_arange(start, end=None, step=1, dtype: Optional[str] = None, device: str = "cpu"):
        """Create a tensor with evenly spaced values."""
        from tensor_value import TensorValue, HAS_NUMPY, HAS_TORCH, np, torch
        
        if end is None:
            end = start
            start = 0
        
        backend = "torch" if device.startswith("cuda") else "numpy"
        
        if backend == "numpy":
            if dtype:
                obj = np.arange(start, end, step, dtype=dtype)
            else:
                obj = np.arange(start, end, step)
        else:
            if not HAS_TORCH:
                raise RuntimeError("PyTorch required for GPU tensors")
            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "int32": torch.int32,
                "int64": torch.int64,
            }
            torch_dtype = dtype_map.get(dtype, None)
            obj = torch.arange(start, end, step, dtype=torch_dtype, device=device)
        
        return TensorValue(obj, backend=backend)
    
    @staticmethod
    def tensor_random(shape, dtype: Optional[str] = None, device: str = "cpu"):
        """Create a tensor with random values in [0, 1)."""
        from tensor_value import TensorValue, HAS_NUMPY, HAS_TORCH, np, torch
        
        backend = "torch" if device.startswith("cuda") else "numpy"
        
        if backend == "numpy":
            obj = np.random.rand(*shape) if isinstance(shape, (list, tuple)) else np.random.rand(shape)
            if dtype:
                obj = obj.astype(dtype)
        else:
            if not HAS_TORCH:
                raise RuntimeError("PyTorch required for GPU tensors")
            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
            }
            torch_dtype = dtype_map.get(dtype, torch.float32)
            obj = torch.rand(shape, dtype=torch_dtype, device=device)
        
        return TensorValue(obj, backend=backend)


def print_function(*args) -> None:
    """Print function that outputs values"""
    output = ' '.join(BuiltinLibrary.to_string(arg) for arg in args)
    print(output)


def __py_as_ndarray(x: Any):
    """Convert a CodeSutra value (list/dict) to a numpy ndarray and wrap it."""
    try:
        import numpy as np
    except Exception as e:
        raise RuntimeError(f"numpy is not available: {e}")

    # Unwrap proxy-like objects
    if hasattr(x, '_obj'):
        x = getattr(x, '_obj')

    if isinstance(x, list):
        return wrap(np.array(x))
    elif isinstance(x, (int, float)):
        return wrap(np.array([x]))
    else:
        # If it's already an ndarray or similar, attempt to wrap it
        return wrap(x)


def __py_as_dataframe(x: Any):
    """Convert a CodeSutra value to a pandas DataFrame and wrap it."""
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError(f"pandas is not available: {e}")

    # Unwrap proxy-like objects
    if hasattr(x, '_obj'):
        x = getattr(x, '_obj')

    if isinstance(x, dict):
        return wrap(pd.DataFrame([x]))
    elif isinstance(x, list):
        return wrap(pd.DataFrame(x))
    else:
        return wrap(x)


def __py_unwrap(x: Any):
    """Unwrap a PyProxy to access the underlying Python object.

    If x is a PyProxy, returns the wrapped object.
    Otherwise, returns x as-is.
    """
    if hasattr(x, '_obj'):
        return getattr(x, '_obj')
    return x


def __py_is_proxy(x: Any) -> bool:
    """Check if an object is a PyProxy (wrapping a Python object)."""
    return hasattr(x, '_obj')


def __py_is_ndarray(x: Any) -> bool:
    """Check if an object is a NumPy ndarray (or a proxy wrapping one)."""
    try:
        import numpy as np
    except Exception:
        return False

    obj = x
    if hasattr(x, '_obj'):
        obj = getattr(x, '_obj')

    return isinstance(obj, np.ndarray)


def create_globals() -> Dict[str, Any]:
    """Create global environment with built-in functions"""
    return {
        # Output
        'print': print_function,
        
        # Type conversion
        'number': BuiltinLibrary.to_number,
        'string': BuiltinLibrary.to_string,
        'bool': BuiltinLibrary.to_bool,
        'type': BuiltinLibrary.type_of,
        
        # Array operations
        'length': BuiltinLibrary.length,
        'push': BuiltinLibrary.push,
        'pop': BuiltinLibrary.pop,
        'shift': BuiltinLibrary.shift,
        'unshift': BuiltinLibrary.unshift,
        'join': BuiltinLibrary.join,
        'reverse': BuiltinLibrary.reverse,
        'sort': BuiltinLibrary.sort,
        
        # String operations
        'split': BuiltinLibrary.split,
        'upper': BuiltinLibrary.upper,
        'lower': BuiltinLibrary.lower,
        'trim': BuiltinLibrary.trim,
        'starts_with': BuiltinLibrary.starts_with,
        'ends_with': BuiltinLibrary.ends_with,
        'contains': BuiltinLibrary.contains,
        'index_of': BuiltinLibrary.index_of,
        'substring': BuiltinLibrary.substring,
        'replace': BuiltinLibrary.replace,
        'char_at': BuiltinLibrary.char_at,
        'repeat': BuiltinLibrary.repeat,
        
        # Dictionary operations
        'keys': BuiltinLibrary.keys,
        'values': BuiltinLibrary.values,
        'has': BuiltinLibrary.has,
        
        # Math functions
        'sqrt': BuiltinLibrary.sqrt,
        'pow': BuiltinLibrary.pow,
        'abs': BuiltinLibrary.abs,
        'floor': BuiltinLibrary.floor,
        'ceil': BuiltinLibrary.ceil,
        'round': BuiltinLibrary.round,
        'min': BuiltinLibrary.min,
        'max': BuiltinLibrary.max,
        'sum': BuiltinLibrary.sum,
        'mean': BuiltinLibrary.mean,
        'sin': BuiltinLibrary.sin,
        'cos': BuiltinLibrary.cos,
        'tan': BuiltinLibrary.tan,
        'log': BuiltinLibrary.log,
        'exp': BuiltinLibrary.exp,
        'random': BuiltinLibrary.random,
        
        # Range
        'range': BuiltinLibrary.range,
        
        # Constants
        'PI': math.pi,
        'E': math.e,
        
        # Tensor type (callable module with factory methods)
        'tensor': TensorModule(),
        
        # Python interop helpers
        'py': {
            'as_ndarray': lambda x: __py_as_ndarray(x),
            'as_dataframe': lambda x: __py_as_dataframe(x),
            'unwrap': lambda x: __py_unwrap(x),
            'is_proxy': lambda x: __py_is_proxy(x),
            'is_ndarray': lambda x: __py_is_ndarray(x),
        }
    }
