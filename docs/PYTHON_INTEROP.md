# Python Interoperability (CodeSutra)

This document explains how CodeSutra integrates with the Python ecosystem.

## Syntax

- `import <module> [as <alias>];`
  - Example: `import numpy as np;`
- `from <module> import name [as alias] [, name2 [as alias2], ...];`
  - Example: `from math import sqrt as s;`

Semicolons are required at the end of import statements (CodeSutra's parser
expects them).

## What you get

- `import` and `from` load real Python modules and objects using `importlib`.
- Imported modules and objects are proxied through a lightweight wrapper so
  attribute access, calling, and indexing behave more naturally from CodeSutra.

## Example (NumPy)

```
import numpy as np;
arr = np.array([1,2,3]);
print(np.sum(arr));
```

The wrapper attempts simple conversions:
- When calling into `numpy` functions, if an argument is a native CodeSutra
  list, the wrapper will try to convert it to a `numpy.ndarray` using
  `numpy.array(...)`.
- When calling into `pandas`, lists/dicts may be converted to `DataFrame`.

These heuristics make common workflows convenient but are intentionally
lightweight and best-effort.

## Limitations and notes

- The wrapper will not magically convert every structure; complex conversions
  (e.g., nested custom types) may require explicit conversion in your CodeSutra
  program.
- If a Python package is not installed (e.g., `numpy` or `pandas`), importing
  will raise a runtime error from the interpreter with the original message.
- Returned Python objects are proxied. If you want raw Python objects inside
  CodeSutra (rare), you can access underlying attributes via the proxy.

## Examples shipped with the repo
- `examples/numpy_example.codesutra`
- `examples/pandas_example.codesutra`

## Next improvements (ideas)
- Add configuration toggles for automatic conversions (on/off).
- Provide explicit conversion helpers in the standard library (e.g., `py.as_ndarray()`)
- Add deeper wrappers for common libraries to expose idiomatic CodeSutra APIs.

***
For questions or to request support for a specific library, open an issue.
