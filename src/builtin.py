"""
CodeSutra Standard Library - Built-in functions
"""

import math
import re
from typing import Any, List, Dict, Callable, Optional
from python_bridge import wrap


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
        """Minimum value"""
        if not args:
            return None
        nums = [BuiltinLibrary.to_number(v) for v in args]
        return min(nums)
    
    @staticmethod
    def max(*args) -> float:
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
        # Python interop helpers
        'py': {
            'as_ndarray': lambda x: __py_as_ndarray(x),
            'as_dataframe': lambda x: __py_as_dataframe(x),
            'unwrap': lambda x: __py_unwrap(x),
        }
    }
