"""Lightweight Python object proxy to smooth CodeSutra <-> Python interop

This module provides `wrap(obj)` which returns a proxy for Python objects
so they behave more naturally when accessed from CodeSutra: attribute access,
calling, and indexing return wrapped objects. It also includes simple
conversion heuristics for NumPy and Pandas when calling functions.
"""
from typing import Any
import importlib


class PyProxy:
    def __init__(self, obj: Any):
        self._obj = obj

    def __getattr__(self, name: str) -> Any:
        try:
            attr = getattr(self._obj, name)
        except Exception as e:
            raise AttributeError(f"Attribute {name} not found on {type(self._obj)}: {e}")
        return wrap(attr)

    def __call__(self, *args, **kwargs) -> Any:
        # determine target module for heuristics
        module = getattr(self._obj, "__module__", "") or getattr(self._obj, "__name__", "")
        new_args = [self._convert_arg(a, module) for a in args]
        new_kwargs = {k: self._convert_arg(v, module) for k, v in kwargs.items()}
        result = self._obj(*new_args, **new_kwargs)
        return wrap(result)

    # Numeric/operator support: delegate to underlying object when possible
    def _unwrap(self, other):
        return other._obj if isinstance(other, PyProxy) else other

    def __add__(self, other):
        return wrap(self._obj + self._unwrap(other))

    def __radd__(self, other):
        return wrap(self._unwrap(other) + self._obj)

    def __sub__(self, other):
        return wrap(self._obj - self._unwrap(other))

    def __rsub__(self, other):
        return wrap(self._unwrap(other) - self._obj)

    def __mul__(self, other):
        return wrap(self._obj * self._unwrap(other))

    def __rmul__(self, other):
        return wrap(self._unwrap(other) * self._obj)

    def __truediv__(self, other):
        return wrap(self._obj / self._unwrap(other))

    def __rtruediv__(self, other):
        return wrap(self._unwrap(other) / self._obj)

    def __mod__(self, other):
        return wrap(self._obj % self._unwrap(other))

    def __rmod__(self, other):
        return wrap(self._unwrap(other) % self._obj)

    def __pow__(self, other):
        return wrap(self._obj ** self._unwrap(other))

    def __rpow__(self, other):
        return wrap(self._unwrap(other) ** self._obj)

    def __getitem__(self, key):
        try:
            val = self._obj[key]
        except Exception as e:
            raise
        return wrap(val)

    def _convert_arg(self, arg: Any, target_module: str) -> Any:
        # Unwrap proxies
        if isinstance(arg, PyProxy):
            arg = arg._obj

        # Heuristic: if calling into numpy and arg is a list, convert to ndarray
        if "numpy" in target_module:
            try:
                np = importlib.import_module("numpy")
                if isinstance(arg, list):
                    return np.array(arg)
            except Exception:
                pass

        # Heuristic: if calling into pandas and arg is list/dict, convert to DataFrame
        if "pandas" in target_module:
            try:
                pd = importlib.import_module("pandas")
                if isinstance(arg, dict):
                    return pd.DataFrame([arg])
                if isinstance(arg, list):
                    return pd.DataFrame(arg)
            except Exception:
                pass

        return arg

    def __repr__(self) -> str:
        # Use the same representation as __str__ for interactive displays
        return self.__str__()

    def __str__(self) -> str:
        # Pretty-print common Python numeric/data objects when small
        try:
            # NumPy arrays
            np = importlib.import_module("numpy")
            if isinstance(self._obj, getattr(np, 'ndarray')):
                # For small arrays, show contents
                try:
                    size = int(self._obj.size)
                except Exception:
                    size = 999999
                if size <= 20 and getattr(self._obj, 'ndim', 1) <= 2:
                    try:
                        return f"{np.array2string(self._obj)}"
                    except Exception:
                        return f"<PyProxy numpy.ndarray shape={getattr(self._obj, 'shape', None)}>"
                else:
                    return f"<PyProxy numpy.ndarray shape={getattr(self._obj, 'shape', None)}>"
        except Exception:
            pass

        try:
            # Pandas DataFrame/Series
            pd = importlib.import_module("pandas")
            if isinstance(self._obj, getattr(pd, 'DataFrame')):
                try:
                    rows = len(self._obj)
                except Exception:
                    rows = 999999
                if rows <= 10:
                    try:
                        return self._obj.to_string()
                    except Exception:
                        return f"<PyProxy pandas.DataFrame rows={rows}>"
                else:
                    return f"<PyProxy pandas.DataFrame rows={rows}>"
            if isinstance(self._obj, getattr(pd, 'Series')):
                try:
                    length = len(self._obj)
                except Exception:
                    length = 999999
                if length <= 20:
                    try:
                        return self._obj.to_string()
                    except Exception:
                        return f"<PyProxy pandas.Series len={length}>"
                else:
                    return f"<PyProxy pandas.Series len={length}>"
        except Exception:
            pass

        # Fallback: show a short repr with type
        try:
            return f"<PyProxy {type(self._obj).__module__}.{type(self._obj).__name__}>"
        except Exception:
            return f"<PyProxy {repr(self._obj)}>"


def wrap(obj: Any) -> Any:
    """Wrap Python objects into a proxy for CodeSutra. Leave primitives and
    CodeSutra-native types alone so they behave as before.
    Also unwraps numpy scalars to native Python numbers for better arithmetic.
    """
    # Try to unwrap numpy scalars to native Python numbers
    try:
        np = importlib.import_module("numpy")
        if isinstance(obj, np.generic):
            # numpy scalar (int64, float32, etc.) → unwrap to Python native
            return obj.item()
    except Exception:
        pass

    # Defer importing CodeSutra-specific types to avoid import cycles
    try:
        from builtin import CodeSutraFunction
    except Exception:
        CodeSutraFunction = None

    # Primitive/native types we keep as-is
    natives = (int, float, str, bool, list, dict, type(None))
    if isinstance(obj, natives):
        return obj

    if CodeSutraFunction and isinstance(obj, CodeSutraFunction):
        return obj

    return PyProxy(obj)
