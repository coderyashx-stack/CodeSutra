# CodeSutra Tensor Type — User Guide

**Version 1.0**  
**Status**: Production Ready (Phase 1)

---

## Quick Start

Create a tensor from a list and perform operations:

```codesutra
# Create tensors
t1 = tensor([1, 2, 3]);
t2 = tensor([4, 5, 6]);

# Arithmetic
result = t1 + t2;      # [5, 7, 9]
product = t1 * 2;      # [2, 4, 6]

# Introspection
print(t1.shape);       # [3]
print(t1.dtype);       # int64
print(t1.device);      # cpu

# Aggregation
sum_val = sum(t1);     # 6
mean_val = mean(t1);   # 2.0
max_val = max(t1);     # 3
min_val = min(t1);     # 1

# Convert back to list
lst = t1.to_list();    # [1, 2, 3]
```

---

## What is a Tensor?

A **tensor** is a multi-dimensional array of numbers. Think of it as:

- **1D tensor**: a list of numbers `[1, 2, 3]`
- **2D tensor**: a matrix `[[1, 2], [3, 4]]`
- **3D tensor**: a cube of numbers `[[[...], [...]], [[...], [...]]]`
- **ND tensor**: n-dimensional arrays, up to memory limits

Tensors in CodeSutra are backed by NumPy (for CPU) or PyTorch (for GPU), giving you high-performance linear algebra out of the box.

---

## Construction

### From Lists

The simplest way to create a tensor is from a CodeSutra list:

```codesutra
# 1D tensor
t = tensor([1, 2, 3]);

# 2D tensor (matrix)
m = tensor([[1, 2, 3], [4, 5, 6]]);

# 3D tensor
cube = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
```

### With Explicit Dtype

By default, CodeSutra infers the dtype from the list values. To be explicit:

```codesutra
# All integers
t_int = tensor([1, 2, 3]);                    # dtype: int64

# All floats
t_float = tensor([1.0, 2.0, 3.0]);           # dtype: float64

# Explicit dtype specification
t_float32 = tensor([1, 2, 3], dtype: "float32");
t_float64 = tensor([1, 2, 3], dtype: "float64");
t_int32 = tensor([1, 2, 3], dtype: "int32");
```

### Factory Functions

Create special tensors without listing values:

```codesutra
# Zeros and ones
zeros = tensor.zeros([3, 3]);      # 3x3 matrix of 0.0
ones = tensor.ones([2, 4]);        # 2x4 matrix of 1.0

# Range
range_t = tensor.arange(0, 10);    # [0, 1, 2, ..., 9]
range_t = tensor.arange(0, 10, 2); # [0, 2, 4, 6, 8]

# Random
rand = tensor.random([3, 3]);      # 3x3 random [0, 1)
```

---

## Properties & Introspection

Every tensor has read-only properties:

```codesutra
t = tensor([[1, 2, 3], [4, 5, 6]]);

# Shape and dimensions
t.shape;    # [2, 3] — shape as a list
t.ndim;     # 2 — number of dimensions
t.size();   # 6 — total elements

# Type information
t.dtype;    # "int64" or appropriate type string

# Device
t.device;   # "cpu" (GPU support coming in Phase 2)
```

---

## Operators

### Arithmetic (Element-wise)

```codesutra
t1 = tensor([1, 2, 3]);
t2 = tensor([4, 5, 6]);

add = t1 + t2;      # [5, 7, 9]
sub = t2 - t1;      # [3, 3, 3]
mul = t1 * t2;      # [4, 10, 18]
div = t2 / t1;      # [4.0, 2.5, 2.0]
pow = t1 ** 2;      # [1, 4, 9]
mod = t2 % 2;       # [0, 1, 0]
```

### Scalar Broadcasting

Scalars automatically broadcast to the tensor's shape:

```codesutra
t = tensor([1, 2, 3]);

t_plus_10 = t + 10;    # [11, 12, 13]
t_times_5 = t * 5;     # [5, 10, 15]
t_squared = t ** 2;    # [1, 4, 9]
```

### List Broadcasting

Lists are automatically converted to tensors:

```codesutra
t = tensor([1, 2, 3]);
result = t + [1, 1, 1];    # [2, 3, 4]
```

### Comparison (Returns Boolean Tensor)

```codesutra
t1 = tensor([1, 2, 3]);
t2 = tensor([2, 2, 2]);

mask = t1 < t2;   # [true, false, false]
mask = t1 == t2;  # [false, true, false]
mask = t1 != t2;  # [true, false, true]
```

---

## Indexing & Slicing

### Single-Dimension Indexing

```codesutra
t = tensor([10, 20, 30, 40, 50]);

val = t[0];        # 10 (single element, unwrapped to number)
val = t[2];        # 30
val = t[-1];       # 50 (last element)
```

### Multi-Dimensional Indexing

```codesutra
m = tensor([[1, 2, 3], [4, 5, 6]]);

row = m[0];        # [1, 2, 3] (returns tensor)
row = m[1];        # [4, 5, 6]
```

**Note**: CodeSutra's parser currently doesn't support comma-based multi-dimensional indexing like `m[0, 1]`. Use sequential indexing instead:

```codesutra
m = tensor([[1, 2, 3], [4, 5, 6]]);
val = m[0][1];     # 2 (first row, second element)
```

---

## Methods

### Shape Manipulation

```codesutra
t = tensor([1, 2, 3, 4, 5, 6]);

# Reshape to [2, 3]
reshaped = t.reshape([2, 3]);
# [[1, 2, 3],
#  [4, 5, 6]]

# Flatten to 1D
flat = reshaped.flatten();
# [1, 2, 3, 4, 5, 6]

# Transpose (reverse all dimensions)
m = tensor([[1, 2], [3, 4], [5, 6]]);
mt = m.transpose();
# [[1, 3, 5],
#  [2, 4, 6]]
```

### Reduction Operations

These aggregate values and return scalars (or tensors if axis specified):

```codesutra
t = tensor([1, 2, 3, 4, 5]);

# Aggregate all
total = sum(t);    # 15
avg = mean(t);     # 3.0
maximum = max(t);  # 5
minimum = min(t);  # 1

# 2D example
m = tensor([[1, 2, 3], [4, 5, 6]]);

# Sum along axis
col_sums = m.sum(axis: 0);  # [5, 7, 9] (sum down columns)
row_sums = m.sum(axis: 1);  # [6, 15] (sum across rows)

# Mean along axis
col_means = m.mean(axis: 0); # [2.5, 3.5, 4.5]
```

### Device Operations (Phase 1: CPU only)

```codesutra
t = tensor([1, 2, 3]);

# Move to CPU (returns a copy)
t_cpu = t.cpu();

# GPU support coming in Phase 2
# t.device         # "cpu" (CPU always available)
```

### Conversion

```codesutra
t = tensor([[1, 2], [3, 4]]);

# To list
lst = t.to_list();     # [[1, 2], [3, 4]]

# Access underlying Python object (for advanced users)
import py;
py_obj = py.unwrap(t); # numpy.ndarray or torch.Tensor
```

---

## Builtin Functions for Tensors

CodeSutra's builtin functions are tensor-aware:

```codesutra
t = tensor([1, 2, 3, 4, 5]);

# Works on tensors
s = sum(t);      # 15
m = mean(t);     # 3.0
mx = max(t);     # 5
mn = min(t);     # 1

# Also works on lists and numbers (backward compatible)
s = sum([1, 2, 3]);    # 6
s = sum(1, 2, 3);      # 6
```

---

## Examples

### Basic Arithmetic

```codesutra
# Create two tensors
x = tensor([1, 2, 3]);
y = tensor([10, 20, 30]);

# Compute
z = x + y;
print(z);  # [11, 22, 33]

# Element-wise operations
prod = x * y;  # [10, 40, 90]
div = y / x;   # [10.0, 10.0, 10.0]
```

### Working with Matrices

```codesutra
# Create a 3x3 matrix
m = tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

# Get properties
print(m.shape);     # [3, 3]
print(m.dtype);     # int64

# Flatten
flat = m.flatten();
print(flat.shape);  # [9]

# Reshape
m2 = m.reshape([1, 9]);
print(m2.shape);    # [1, 9]

# Transpose
mt = m.transpose();
print(mt.shape);    # [3, 3]
```

### Aggregation

```codesutra
data = tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]);

# Global statistics
print("Total:", sum(data));      # 45
print("Average:", mean(data));   # 5.0
print("Max:", max(data));        # 9
print("Min:", min(data));        # 1

# Column sums (sum down rows)
col_sums = data.sum(axis: 0);
print(col_sums);  # [12, 15, 18]

# Row sums (sum across columns)
row_sums = data.sum(axis: 1);
print(row_sums);  # [6, 15, 24]
```

### Creating Special Tensors

```codesutra
# Zeros
zeros_mat = tensor.zeros([3, 3]);
print(zeros_mat);
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]]

# Ones
ones_vec = tensor.ones([5]);
print(ones_vec);
# [1. 1. 1. 1. 1.]

# Range
seq = tensor.arange(1, 6);
print(seq);  # [1 2 3 4 5]

# Random
rand_mat = tensor.random([2, 2]);
print(rand_mat);
# [[0.xxx xxx]
#  [0.yyy yyy]]
```

---

## Tensor vs NumPy Arrays

If you use Python interop, you'll encounter NumPy arrays. Here's the difference:

| Feature | CodeSutra Tensor | NumPy Array |
|---------|------------------|-------------|
| **Creation** | `t = tensor([1,2,3])` | `import numpy as np; a = np.array([1,2,3])` |
| **Type** | `TensorValue` | `numpy.ndarray` |
| **Introspection** | `t.shape`, `t.dtype` | `a.shape`, `a.dtype` |
| **Operations** | `t + 5`, `t * other_t` | `a + 5`, `a * other_a` |
| **Conversion** | `t.to_list()` | `a.tolist()` |
| **To NumPy** | `import py; py.unwrap(t)` | Already NumPy |

**You can mix them freely** — CodeSutra tensors and NumPy arrays work together seamlessly:

```codesutra
import numpy as np;

# Create a NumPy array from Python
arr = np.array([1, 2, 3]);

# Wrap as CodeSutra tensor
t = tensor(arr);

# Use tensor operations
result = t * 2;

# Unwrap back to NumPy
arr_out = py.unwrap(result);
```

---

## Performance Notes

### CPU Performance (Phase 1)

- CodeSutra tensors use **NumPy** for CPU computation
- NumPy is heavily optimized (written in C) for numerical operations
- Most operations are **very fast** on modern CPUs

### Memory

- Tensors own their data (automatically freed by Python's garbage collector)
- Operations that don't change shape return **views** when possible (NumPy semantics)
- Use `.to_list()` to convert back to CodeSutra lists when needed

### GPU Support (Phase 2)

- Coming soon: `.gpu()` and `.cuda()` to move tensors to GPU
- Requires PyTorch and CUDA-enabled hardware
- See design spec for details

---

## Common Patterns

### Normalize a Vector

```codesutra
v = tensor([3, 4]);

# Magnitude
magnitude = sqrt(sum(v * v));
print(magnitude);  # 5.0

# Normalized (unit vector)
v_norm = v / magnitude;
print(v_norm);     # [0.6, 0.8]
```

### Element-wise Comparison

```codesutra
t = tensor([1, 5, 3, 2, 4]);
threshold = 3;

# Which elements are >= threshold?
mask = t >= threshold;  # [false, true, false, false, true]
print(mask);
```

### Broadcasting Rule (NumPy-style)

Tensors with different shapes can be combined if they're **compatible** for broadcasting:

```codesutra
# Shape [3] broadcasts with shape [1, 3]
t1 = tensor([1, 2, 3]);        # [3]
t2 = tensor([[10], [20]]);     # [2, 1]

# Compatible shapes; broadcasting produces [2, 3]
result = t1 + t2;
print(result.shape);   # [2, 3]
print(result);         # [[11, 12, 13], [21, 22, 23]]
```

---

## Troubleshooting

### Shape Mismatch Error

```
RuntimeError: Tensor operation failed: operands could not be broadcast together
```

**Cause**: Trying to combine tensors with incompatible shapes.

**Solution**: Check `t.shape` and ensure shapes are broadcastable:

```codesutra
t1 = tensor([1, 2, 3]);        # [3]
t2 = tensor([1, 2]);           # [2] — NOT broadcastable with [3]

# ❌ This fails:
result = t1 + t2;

# ✅ This works:
t2_padded = tensor([1, 2, 0]);
result = t1 + t2_padded;
```

### DType Mismatch

```codesutra
t_int = tensor([1, 2, 3]);       # int64
t_float = tensor([1.0, 2.0, 3.0]); # float64

# This works (NumPy auto-promotes)
result = t_int + t_float;
print(result.dtype);  # float64
```

### Cannot Call Tensor as Function

```
RuntimeError: Object is not callable
```

**Cause**: You assigned a tensor to a variable that shadows a function.

```codesutra
sum = tensor([1, 2, 3]);  # ❌ Shadows the sum() function

# ✅ Use different name
my_tensor = tensor([1, 2, 3]);
total = sum(my_tensor);
```

---

## API Reference

### Constructor

```
tensor(value, dtype=None, device="cpu") -> Tensor
```

Creates a tensor from a list or NumPy array.

**Args:**
- `value`: List, nested list, or NumPy array
- `dtype`: Optional dtype string ("int32", "int64", "float32", "float64", etc.)
- `device`: "cpu" (only option in Phase 1)

**Returns:** `Tensor`

---

### Factory Functions

```
tensor.zeros(shape, dtype=None, device="cpu") -> Tensor
tensor.ones(shape, dtype=None, device="cpu") -> Tensor
tensor.arange(start, end=None, step=1, dtype=None, device="cpu") -> Tensor
tensor.random(shape, dtype=None, device="cpu") -> Tensor
```

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `.shape` | list | Shape of tensor, e.g., `[2, 3]` |
| `.ndim` | int | Number of dimensions |
| `.dtype` | str | Data type, e.g., `"int64"` |
| `.device` | str | Device: `"cpu"` (GPU in Phase 2) |

---

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `.reshape(shape)` | Tensor | Reshape to new shape |
| `.flatten()` | Tensor | Flatten to 1D |
| `.transpose()` | Tensor | Transpose (reverse dims) |
| `.sum(axis=None, keepdims=False)` | scalar or Tensor | Sum along axis |
| `.mean(axis=None, keepdims=False)` | scalar or Tensor | Mean along axis |
| `.max(axis=None, keepdims=False)` | scalar or Tensor | Max along axis |
| `.min(axis=None, keepdims=False)` | scalar or Tensor | Min along axis |
| `.to_list()` | list | Convert to nested lists |
| `.cpu()` | Tensor | Copy to CPU (Phase 1: always CPU) |

---

### Builtin Functions (Tensor-aware)

```
sum(tensor or list or *args) -> scalar
mean(tensor or list or *args) -> scalar
max(tensor or list or *args) -> scalar
min(tensor or list or *args) -> scalar
```

---

## What's Next (Phase 2+)

- GPU support via PyTorch (`.gpu()`, `.cuda()`)
- `matmul()` for matrix multiplication
- Autograd / differentiation (`.backward()`)
- More reduction ops (variance, std, etc.)
- Advanced indexing and slicing
- Tensor comprehensions

---

## Contributing & Feedback

Found a bug or have a feature request? Open an issue on GitHub or reach out.

**CodeSutra Tensor is ready to power your AI-first scripts!** 🚀
