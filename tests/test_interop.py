import importlib.util
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lexer import Lexer
from parser import Parser
from interpreter import Interpreter
from python_bridge import wrap


def py_unwrap(x):
    """Unwrap a PyProxy to get the underlying Python object."""
    if hasattr(x, '_obj'):
        return getattr(x, '_obj')
    return x


def run_code(code: str) -> Interpreter:
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()
    interp = Interpreter()
    interp.interpret(program)
    return interp


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def test_from_import_function_call_math():
    code = "from math import sqrt as s; r = s(9);"
    interp = run_code(code)
    r = interp.current_env.get('r')
    assert isinstance(r, (int, float))
    assert float(r) == 3.0


def test_numpy_array_and_indexing():
    if not has_module('numpy'):
        import pytest; pytest.skip('numpy not installed')

    code = "import numpy as np; arr = np.array([1,2,3]); v = arr[2];"
    interp = run_code(code)
    arr = interp.current_env.get('arr')
    v = interp.current_env.get('v')

    import numpy as np
    # arr should be a PyProxy wrapping ndarray; unwrap it
    arr_unwrapped = py_unwrap(arr)
    assert isinstance(arr_unwrapped, np.ndarray)
    assert arr_unwrapped.tolist() == [1, 2, 3]
    # v should now be a native Python number (unwrapped from numpy scalar)
    assert isinstance(v, (int, float))
    assert v == 3


def test_py_as_ndarray_helper():
    if not has_module('numpy'):
        import pytest; pytest.skip('numpy not installed')

    code = "x = py.as_ndarray([1,2,3]);"
    interp = run_code(code)
    x = interp.current_env.get('x')
    import numpy as np
    x_unwrapped = py_unwrap(x)
    assert isinstance(x_unwrapped, np.ndarray)
    assert x_unwrapped.tolist() == [1, 2, 3]


def test_pandas_dataframe_helper():
    if not has_module('pandas'):
        import pytest; pytest.skip('pandas not installed')

    code = "import pandas as pd; df = pd.DataFrame([{name: 'Alice', age: 30}]);"
    interp = run_code(code)
    df = interp.current_env.get('df')
    import pandas as pd
    df_unwrapped = py_unwrap(df)
    assert isinstance(df_unwrapped, pd.DataFrame)
    assert df_unwrapped.loc[0, 'age'] == 30


def test_py_unwrap_helper():
    """Test the py.unwrap() helper explicitly."""
    if not has_module('numpy'):
        import pytest; pytest.skip('numpy not installed')

    code = "import numpy as np; arr = np.array([5,6,7]); unwrapped = py.unwrap(arr);"
    interp = run_code(code)
    unwrapped = interp.current_env.get('unwrapped')
    import numpy as np
    # unwrap should have returned the raw ndarray
    assert isinstance(unwrapped, np.ndarray)
    assert unwrapped.tolist() == [5, 6, 7]
