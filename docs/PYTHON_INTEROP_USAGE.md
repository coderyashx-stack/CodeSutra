# Python Interoperability Guide

CodeSutra seamlessly integrates with the Python ecosystem, giving you instant access to NumPy, Pandas, Matplotlib, scikit-learn, and any other installed Python package. This guide shows you how to use Python libraries from CodeSutra scripts.

## Quick Start

### Import a Python Module

```
import numpy as np;
arr = np.array([1, 2, 3, 4, 5]);
print(arr);
```

Output:
```
[1 2 3 4 5]
```

### Import Specific Functions

```
from math import sqrt as s;
result = s(16);
print(result);
```

Output:
```
4.0
```

## Syntax

CodeSutra uses two import forms (similar to Python):

### `import module [as alias]`

Imports an entire module.

```
import numpy as np;
import pandas as pd;
```

After import, access module members with dot notation:
```
np.array([1, 2, 3])
np.sum([1, 2, 3])
pd.DataFrame([{name: "Alice", age: 30}])
```

### `from module import name [as alias], name2 [as alias2], ...`

Imports specific functions or objects.

```
from math import sqrt as s, sin, cos;
from numpy import array as arr;
```

After import, use the names directly:
```
s(9)     # sqrt(9)
sin(0)   # 0
cos(0)   # 1
arr([1, 2, 3])  # Create array directly
```

## Key Features

### 1. Automatic Type Conversion

When calling Python functions, CodeSutra automatically converts native lists and dicts to Python types when appropriate:

**NumPy Example:**
```
import numpy as np;
data = [1, 2, 3];
result = np.sum(data);    # data is auto-converted to np.ndarray
print(result);             # Output: 6
```

**Pandas Example:**
```
import pandas as pd;
data = [{name: "Alice", age: 30}, {name: "Bob", age: 25}];
df = pd.DataFrame(data);  # data is auto-converted to suitable format
print(df);
```

### 2. NumPy Scalars Automatically Unwrap

When you index into a NumPy array, the result is automatically converted to a native CodeSutra number:

```
import numpy as np;
arr = np.array([10, 20, 30]);
x = arr[1];           # Indexing returns native number 20
result = x + 5;       # Direct arithmetic: 20 + 5 = 25
print(result);        # Output: 25 (no proxy wrapping)
```

This makes arithmetic seamless and output clean.

### 3. Pretty Printing

Small NumPy arrays and Pandas DataFrames are displayed nicely:

```
import numpy as np;
arr = np.array([1, 2, 3, 4, 5]);
print(arr);           # Output: [1 2 3 4 5]

import pandas as pd;
df = pd.DataFrame([{name: "Alice", age: 30}, {name: "Bob", age: 25}]);
print(df);            # Output:
                      #      name  age
                      #  0  Alice   30
                      #  1    Bob   25
```

Large objects show summary info:
```
import numpy as np;
big_arr = np.ones((1000, 1000));
print(big_arr);       # Output: <PyProxy numpy.ndarray shape=(1000, 1000)>
```

## Standard Helpers

CodeSutra provides a `py` object with conversion and inspection helpers:

### `py.as_ndarray(x)`

Force convert a CodeSutra list or value to a NumPy ndarray:

```
import numpy as np;
data = [1, 2, 3];
arr = py.as_ndarray(data);  # Explicit conversion
print(type(arr));            # Works even if np.sum() doesn't auto-convert
```

### `py.as_dataframe(x)`

Force convert a CodeSutra list or dict to a Pandas DataFrame:

```
import pandas as pd;
row = {name: "Charlie", age: 35};
df = py.as_dataframe(row);
print(df);
```

### `py.unwrap(x)`

Extract the underlying Python object from a proxy. Useful when you need the raw Python object:

```
import numpy as np;
arr = np.array([1, 2, 3]);
raw_obj = py.unwrap(arr);
# raw_obj is the actual numpy.ndarray, not a proxy
```

### `py.is_proxy(x)`

Check if an object is a PyProxy (wrapping a Python object):

```
import numpy as np;
arr = np.array([1, 2, 3]);
print(py.is_proxy(arr));         # Output: true
print(py.is_proxy([1, 2, 3]));   # Output: false (native CodeSutra list)
```

### `py.is_ndarray(x)`

Check if an object is a NumPy ndarray (or proxy wrapping one):

```
import numpy as np;
arr = np.array([1, 2, 3]);
print(py.is_ndarray(arr));       # Output: true
print(py.is_ndarray([1, 2, 3])); # Output: false
```

## Examples

### NumPy: Array Operations

```
import numpy as np;

# Create array
arr = np.array([1, 2, 3, 4, 5]);
print("arr:", arr);

# NumPy functions
s = np.sum(arr);
print("sum:", s);

# Element-wise operations
arr2 = arr * 2;
print("doubled:", arr2);

# Indexing and slicing
print("first:", arr[0]);
print("last:", arr[-1]);
```

Output:
```
arr: [1 2 3 4 5]
sum: 15
doubled: [ 2  4  6  8 10]
first: 1
last: 5
```

### Pandas: Data Analysis

```
import pandas as pd;

# Create DataFrame from list of dicts
data = [
  {name: "Alice", age: 30, city: "NYC"},
  {name: "Bob", age: 25, city: "LA"},
  {name: "Charlie", age: 35, city: "Chicago"}
];
df = pd.DataFrame(data);
print("Full DataFrame:");
print(df);

# Access columns
ages = df['age'].tolist();
print("\nAges:", ages);

# Filter
young = df[df['age'] < 30];
print("\nYounger than 30:");
print(young);
```

Output:
```
Full DataFrame:
       name  age     city
0    Alice   30      NYC
1      Bob   25       LA
2  Charlie   35  Chicago

Ages: [30, 25, 35]

Younger than 30:
  name  age city
1  Bob   25   LA
```

### Matplotlib: Plotting

```
from matplotlib import pyplot as plt;

# Create some data
x = [0, 1, 2, 3, 4];
y = [0, 1, 4, 9, 16];  # y = x^2

# Plot
plt.plot(x, y);
plt.xlabel("x");
plt.ylabel("y");
plt.title("y = x^2");
plt.savefig("plot.png");
print("Plot saved to plot.png");
```

### scikit-learn: Machine Learning

```
from sklearn.datasets import load_iris;
from sklearn.ensemble import RandomForestClassifier;

# Load iris dataset
iris = load_iris();
X = iris.data;
y = iris.target;

# Train classifier
clf = RandomForestClassifier(n_estimators: 10);
clf.fit(X, y);

# Predict
prediction = clf.predict([[5.1, 3.5, 1.4, 0.2]]);
print("Prediction:", prediction);
```

## Best Practices

### 1. Check Module Availability

If a library might not be installed, wrap imports in error handling:

```
func safe_numpy() {
  try {
    import numpy as np;
    return np.array([1, 2, 3]);
  } catch (e) {
    print("NumPy not installed");
    return nil;
  }
}

arr = safe_numpy();
```

### 2. Use Type Helpers for Clarity

For complex conversions, use the explicit helpers:

```
import numpy as np;

# Instead of relying on auto-conversion:
data = [1, 2, 3];
result = np.sum(data);  # Works, but implicit

# Prefer explicit conversion for clarity:
data = [1, 2, 3];
arr = py.as_ndarray(data);
result = np.sum(arr);   # Explicit, easier to understand
```

### 3. Unwrap When Needed

If you need to pass a Python object to another function:

```
import numpy as np;

arr = np.array([1, 2, 3]);
raw = py.unwrap(arr);     # Get the actual ndarray
# Now raw can be passed to Python code that expects numpy.ndarray
```

### 4. Leverage NumPy Scalar Unwrapping

No need to explicitly convert array elements:

```
import numpy as np;

arr = np.array([10, 20, 30]);
x = arr[0];          # Automatically a native number
y = x + 5;           # Direct arithmetic works
print(y);            # Output: 15
```

## Limitations and Caveats

### 1. Not All Python Idioms Work

CodeSutra's syntax and semantics differ from Python. Some advanced Python features won't work:

- List comprehensions: use `for` loops instead.
- Dictionary comprehensions: build dicts manually.
- Generators: not supported; use arrays instead.
- Type hints: CodeSutra is dynamically typed.

### 2. Complex Object Conversions

While simple lists/dicts auto-convert, complex nested structures may need explicit handling:

```
import numpy as np;

# Simple: auto-converts
simple = [1, 2, 3];
arr = np.array(simple);  # Works

# Complex: may need explicit unwrapping or conversion
nested = [[1, 2], [3, 4]];
arr = np.array(nested);  # May or may not work as expected
# Use py.as_ndarray() if unsure:
arr = py.as_ndarray(nested);
```

### 3. Memory and Performance

Large Python objects (e.g., huge NumPy arrays, large DataFrames) are shared between CodeSutra and Python without copying. Modifications in one language affect the other:

```
import numpy as np;

arr = np.array([1, 2, 3]);
arr[0] = 999;     # Modifies the original ndarray
print(arr);       # Output: [999 2 3]
```

### 4. Error Messages

Python errors are wrapped in CodeSutra runtime errors. If something goes wrong, the original Python error is included:

```
import numpy as np;

try {
  bad = np.array("not a number");  # NumPy will raise an error
} catch (e) {
  print("Error:", e);
}
```

## Debugging

### Check Object Type

Use CodeSutra's `type()` function:

```
import numpy as np;

arr = np.array([1, 2, 3]);
print(type(arr));   # Output: object (generic, but it's an ndarray)
```

### Unwrap and Inspect

Use `py.unwrap()` to get the raw Python object and inspect it:

```
import numpy as np;

arr = np.array([1, 2, 3]);
raw = py.unwrap(arr);
print(raw);         # Shows the actual ndarray details
```

### Check if Proxy

Use `py.is_proxy()`:

```
import numpy as np;

arr = np.array([1, 2, 3]);
print(py.is_proxy(arr));          # true
print(py.is_proxy([1, 2, 3]));    # false
```

## Summary

| Feature | Example | Notes |
|---------|---------|-------|
| Import module | `import numpy as np;` | Access via dot notation: `np.array()` |
| Import name | `from math import sqrt;` | Use directly: `sqrt(9)` |
| Auto-convert lists | `np.sum([1,2,3])` | Lists → ndarray when calling NumPy |
| Auto-convert dicts | `pd.DataFrame([{a:1}])` | Dicts → DataFrame when calling Pandas |
| Unwrap scalars | `arr[0] + 5` | NumPy scalars → native numbers automatically |
| Pretty print | `print(arr)` | Small arrays/DataFrames shown nicely |
| Explicit convert | `py.as_ndarray(x)` | Force conversion when auto-convert doesn't work |
| Unwrap proxy | `py.unwrap(x)` | Get raw Python object |
| Check type | `py.is_proxy(x)`, `py.is_ndarray(x)` | Inspect object type |

---

For more information, see:
- `docs/PYTHON_INTEROP.md` — Technical overview
- `examples/numpy_example.codesutra` — NumPy example script
- `examples/pandas_example.codesutra` — Pandas example script
