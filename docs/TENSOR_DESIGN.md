# CodeSutra Tensor Type — Design Specification

**Version 1.0**  
**Date**: November 28, 2025  
**Status**: Design Review

---

## Executive Summary

CodeSutra introduces **native tensor types** as first-class values in the language. Tensors are:

- **Multi-dimensional arrays** with shape, dtype, and device information
- **Backed by NumPy (CPU) and PyTorch (GPU)** for seamless performance
- **Integrated deeply** into CodeSutra's type system, operators, and standard library
- **Device-agnostic**: Users can move tensors between CPU and GPU transparently
- **Zero-copy interop** with Python's NumPy, PyTorch, and TensorFlow

This positions CodeSutra as **"A high-level, AI-first scripting language with native tensors and seamless Python interoperability."**

---

## 1. Tensor Type Semantics

### 1.1 Type System

```
CodeSutra Type Hierarchy:
  value
    ├── number (float, int)
    ├── string
    ├── bool
    ├── nil
    ├── list
    ├── dict
    ├── function
    └── tensor  ← NEW
```

**Key Property**: Tensors are **first-class values**, not wrappers.

### 1.2 Dynamic Shape and Dtype

**Shape** is dynamic and can be unknown at parse time:

```codesutra
t1 = tensor([1, 2, 3]);           # shape: [3]
t2 = tensor([[1, 2], [3, 4]]);    # shape: [2, 2]
t3 = tensor.random([10, 20, 30]); # shape: [10, 20, 30]
```

**Dtype** is inferred from construction but can be explicit:

```codesutra
t_int = tensor([1, 2, 3]);                    # dtype: int64
t_float = tensor([1.0, 2.0, 3.0]);           # dtype: float32
t_explicit = tensor([1, 2, 3], dtype: "float64");  # explicit dtype
```

**Philosophy**: Like NumPy and PyTorch, CodeSutra tensors are **dynamically typed** — shape and dtype are runtime properties, not static annotations.

### 1.3 Device Semantics

Every tensor has a **device**: `"cpu"` or `"cuda"`.

```codesutra
t = tensor([1, 2, 3]);           # device: "cpu" (default)
gpu_t = t.gpu();                 # device: "cuda"
cpu_t = gpu_t.cpu();             # device: "cpu"
```

**Implicit device propagation**:
```codesutra
t_cpu = tensor([1, 2, 3]);       # cpu
t_gpu = tensor([4, 5, 6]).gpu(); # gpu
result = t_cpu + t_gpu;          # ERROR: device mismatch (see section 2.5)
```

### 1.4 Mutability and Copy Semantics

**Default**: Tensors are **immutable** from the CodeSutra perspective.

Operations return **new tensors**, never modify in-place:

```codesutra
t = tensor([1, 2, 3]);
t2 = t * 2;       # Returns new tensor, t is unchanged
t3 = t + 10;      # Returns new tensor, t is unchanged
```

**Backend behavior**: 
- NumPy arrays are wrapped and copied on mutation (copy-on-write semantics)
- PyTorch tensors share gradients across operations (autograd-friendly)

This is transparent to the user. The language guarantees immutability; the backend handles efficiency.

### 1.5 Memory Model

- **Ownership**: CodeSutra tensors own their data (managed by Python's GC)
- **Lifetime**: Automatic via Python garbage collection
- **Zero-copy interop**: Tensors can be passed to/from NumPy/PyTorch without copying

### 1.6 Broadcasting

CodeSutra follows **NumPy's broadcasting rules** automatically:

```codesutra
t1 = tensor([1, 2, 3]);          # shape: [3]
t2 = tensor([[1], [2], [3]]);    # shape: [3, 1]
result = t1 + t2;                # shape: [3, 3] (broadcasted)
```

Broadcasting happens implicitly in all binary operations.

---

## 2. Operator Semantics

### 2.1 Arithmetic Operators

All standard operators work element-wise:

```codesutra
t1 = tensor([1, 2, 3]);
t2 = tensor([4, 5, 6]);

add = t1 + t2;        # [5, 7, 9]
sub = t1 - t2;        # [-3, -3, -3]
mul = t1 * t2;        # [4, 10, 18]
div = t2 / t1;        # [4.0, 2.5, 2.0]
pow = t1 ** 2;        # [1, 4, 9]
mod = t2 % 2;         # [0, 1, 0]
```

**Scalar broadcasting**:
```codesutra
t = tensor([1, 2, 3]);
result = t * 5;       # [5, 10, 15] (scalar broadcasts)
```

**Mixed types** (tensor + list):
```codesutra
t = tensor([1, 2, 3]);
result = t + [1, 1, 1];  # Converts list to tensor, then adds
```

### 2.2 Comparison Operators

Return boolean tensors:

```codesutra
t1 = tensor([1, 2, 3]);
t2 = tensor([2, 2, 2]);

mask = t1 < t2;       # tensor([true, false, false])
mask = t1 == t2;      # tensor([false, true, false])
```

### 2.3 Logical Operators (and, or, not)

Work on boolean tensors:

```codesutra
t1 = tensor([true, true, false]);
t2 = tensor([true, false, false]);

result = t1 and t2;   # tensor([true, false, false])
result = t1 or t2;    # tensor([true, true, false])
result = not t1;      # tensor([false, false, true])
```

### 2.4 Indexing and Slicing

```codesutra
t = tensor([[1, 2, 3], [4, 5, 6]]);

elem = t[0];          # tensor([1, 2, 3]) (row)
elem = t[0][1];       # 2 (scalar, unwrapped)
elem = t[1, 2];       # 6 (advanced indexing)

row = t[0];           # tensor([1, 2, 3])
col = t[:, 1];        # tensor([2, 5])
```

**Note**: Slicing returns tensors; single element access returns scalars (unwrapped like NumPy scalars).

### 2.5 Device Mismatch Handling

**Rule**: All operands in a binary operation must be on the same device.

```codesutra
t_cpu = tensor([1, 2, 3]);
t_gpu = tensor([4, 5, 6]).gpu();

result = t_cpu + t_gpu;  # ERROR: "Tensors on different devices (cpu + cuda)"
```

**Resolution**: Explicit device movement:
```codesutra
result = t_cpu.gpu() + t_gpu;     # OK: both on GPU
result = t_cpu + t_gpu.cpu();     # OK: both on CPU
```

**Rationale**: Forces explicit device awareness, preventing silent errors.

---

## 3. Tensor Construction

### 3.1 From Lists

**Automatic conversion**:
```codesutra
t1 = tensor([1, 2, 3]);
t2 = tensor([[1, 2], [3, 4]]);
t3 = tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]);
```

**With dtype**:
```codesutra
t = tensor([1, 2, 3], dtype: "float32");
t = tensor([1, 2, 3], dtype: "int32");
```

**With device**:
```codesutra
t = tensor([1, 2, 3], device: "cuda");
t = tensor([1, 2, 3], device: "cpu");  # default
```

### 3.2 Factory Functions

```codesutra
t_zeros = tensor.zeros([3, 3]);           # 3x3 zeros
t_ones = tensor.ones([2, 4]);             # 2x4 ones
t_eye = tensor.eye(3);                    # 3x3 identity
t_arange = tensor.arange(0, 10, 1);       # [0, 1, ..., 9]
t_linspace = tensor.linspace(0, 1, 10);   # 10 evenly spaced in [0, 1]
t_random = tensor.random([3, 3]);         # random [0, 1)
t_randn = tensor.randn([3, 3]);           # normal distribution
```

**With device**:
```codesutra
t = tensor.zeros([3, 3], device: "cuda");
t = tensor.random([10, 10], device: "cuda");
```

### 3.3 From Python Objects

Seamless conversion from NumPy/PyTorch:

```codesutra
import numpy as np;

arr = np.array([1, 2, 3]);
t = tensor(arr);           # Wraps NumPy array

import torch;

torch_t = torch.tensor([1, 2, 3]);
t = tensor(torch_t);       # Wraps PyTorch tensor
```

---

## 4. Tensor API

### 4.1 Properties

```codesutra
t = tensor([[1, 2, 3], [4, 5, 6]]);

t.shape;               # [2, 3]
t.dtype;               # "int64" (or appropriate dtype)
t.device;              # "cpu" or "cuda"
t.ndim;                # 2 (number of dimensions)
t.size;                # 6 (total number of elements)
```

### 4.2 Methods

#### Shape/View Operations

```codesutra
t = tensor([[1, 2], [3, 4], [5, 6]]);  # shape: [3, 2]

t_reshaped = t.reshape([2, 3]);        # [2, 3]
t_flat = t.reshape([-1]);              # [6] (flatten)
t_trans = t.transpose();                # [2, 3]
t_unsqueezed = t.unsqueeze(0);         # [1, 3, 2]
t_squeezed = t.squeeze();               # removes dims of size 1
```

#### Device Operations

```codesutra
t = tensor([1, 2, 3]);

t_gpu = t.gpu();       # Move to GPU
t_cpu = t_gpu.cpu();   # Move to CPU
t_pin = t.pin_memory(); # Pin to host memory (for GPU transfer)
```

#### Reduction Operations

```codesutra
t = tensor([[1, 2, 3], [4, 5, 6]]);

sum_all = t.sum();           # scalar: 21
sum_cols = t.sum(axis: 0);   # [5, 7, 9]
sum_rows = t.sum(axis: 1);   # [6, 15]

mean_all = t.mean();         # 3.5
max_val = t.max();           # 6
min_val = t.min();           # 1
```

#### Element-wise Operations

```codesutra
t = tensor([-2, -1, 0, 1, 2]);

t_abs = t.abs();       # [2, 1, 0, 1, 2]
t_sqrt = tensor([1, 4, 9]).sqrt();  # [1, 2, 3]
t_exp = t.exp();       # e^t for each element
t_log = tensor([1, 2.718, 7.389]).log();  # [0, 1, 2]
t_sin = t.sin();
t_cos = t.cos();
```

#### Activation Functions

```codesutra
t = tensor([-2, -1, 0, 1, 2]);

t_relu = t.relu();     # [0, 0, 0, 1, 2]
t_sigmoid = t.sigmoid();
t_tanh = t.tanh();
```

#### Type Conversion

```codesutra
t = tensor([1, 2, 3], dtype: "int32");

t_float = t.float();      # Convert to float32
t_double = t.double();    # Convert to float64
t_int = t.int();          # Convert to int32
t_long = t.long();        # Convert to int64

t_np = t.numpy();         # Convert to NumPy array
t_list = t.tolist();      # Convert to Python list
```

#### Indexing and Selection

```codesutra
t = tensor([[1, 2, 3], [4, 5, 6]]);

mask = t > 2;
selected = t[mask];        # Non-zero elements where mask is true
```

### 4.3 Matrix Operations

```codesutra
a = tensor([[1, 2], [3, 4]]);
b = tensor([[5, 6], [7, 8]]);

# Matrix multiplication
c = matmul(a, b);           # 2x2 @ 2x2 = 2x2

# Element-wise operations also work
c = a + b;
c = a * b;  # Element-wise multiplication, NOT matrix mult

# Transpose
a_t = a.transpose();

# Determinant
det = a.det();

# Inverse
a_inv = a.inv();
```

### 4.4 Stacking and Concatenation

```codesutra
t1 = tensor([1, 2, 3]);
t2 = tensor([4, 5, 6]);

concat = concat(t1, t2, axis: 0);  # [1, 2, 3, 4, 5, 6]
stacked = stack([t1, t2], axis: 0); # [[1, 2, 3], [4, 5, 6]]
```

---

## 5. Standard Library Functions

### 5.1 Tensor Creation (already covered above)

### 5.2 Linear Algebra

```codesutra
# Matrix multiplication
result = matmul(a, b);
result = a @ b;  # Shorthand (if @ is added to syntax)

# Dot product
dot = dot(a, b);

# Norm
norm = norm(t);        # Frobenius norm by default
norm_l1 = norm(t, p: 1);
norm_l2 = norm(t, p: 2);

# Eigendecomposition
eigenvalues, eigenvectors = eig(t);

# Singular value decomposition
u, s, v = svd(t);
```

### 5.3 Aggregation

```codesutra
all_true = all(mask);   # True if all elements are true
any_true = any(mask);   # True if any element is true
argmax = argmax(t, axis: 0);
argmin = argmin(t, axis: 0);
```

### 5.4 Sorting and Searching

```codesutra
sorted_t = sort(t);
sorted_t, indices = sort(t, return_indices: true);

unique_vals = unique(t);
```

---

## 6. Integration with Python

### 6.1 NumPy Integration

CodeSutra tensors seamlessly wrap NumPy arrays:

```codesutra
import numpy as np;

# Create NumPy array
arr = np.array([[1, 2, 3], [4, 5, 6]]);

# Wrap as tensor
t = tensor(arr);
print(t.shape);       # [2, 3]

# Operations return CodeSutra tensors
result = t * 2;       # CodeSutra tensor

# Convert back to NumPy
arr_back = t.numpy(); # NumPy array
```

**Zero-copy semantics**: When wrapping NumPy arrays, no copy is made unless the array is not C-contiguous.

### 6.2 PyTorch Integration

CodeSutra tensors seamlessly work with PyTorch:

```codesutra
import torch;

# Create PyTorch tensor
torch_t = torch.tensor([[1, 2, 3], [4, 5, 6]]);

# Wrap as CodeSutra tensor
t = tensor(torch_t);
print(t.device);      # "cuda" if torch_t is on GPU

# Operations preserve device and autograd graph
result = t * 2;       # Still a PyTorch tensor under the hood
```

**Autograd support**: Operations on PyTorch tensors preserve the computation graph for backpropagation.

### 6.3 Automatic Type Conversion

When calling Python functions expecting NumPy/PyTorch tensors, CodeSutra tensors auto-convert:

```codesutra
import numpy as np;

t = tensor([1, 2, 3]);
result = np.sum(t);        # Auto-converts to NumPy, calls np.sum
```

---

## 7. Type Inference and Coercion

### 7.1 List → Tensor Auto-Conversion

In most contexts, lists are automatically converted to tensors:

```codesutra
t1 = tensor([1, 2, 3]);
t2 = t1 + [1, 1, 1];      # [1, 1, 1] is auto-converted to tensor
```

### 7.2 Scalar → Tensor Broadcasting

Scalars broadcast to tensors:

```codesutra
t = tensor([1, 2, 3]);
result = t + 10;           # 10 broadcasts to [10, 10, 10]
result = 5 * t;            # 5 broadcasts
```

### 7.3 NumPy Scalar Unwrapping

When indexing a tensor, if the result is a single scalar, it's unwrapped to a native number:

```codesutra
t = tensor([[1, 2], [3, 4]]);
val = t[0, 0];             # 1 (native number, not a tensor)
result = val + 5;          # 6 (direct arithmetic)
```

---

## 8. Error Handling

### 8.1 Shape Mismatch

```codesutra
t1 = tensor([1, 2, 3]);           # [3]
t2 = tensor([[1, 2], [3, 4]]);    # [2, 2]

result = t1 + t2;  # ERROR: "Cannot broadcast shapes [3] and [2, 2]"
```

### 8.2 Device Mismatch

```codesutra
t_cpu = tensor([1, 2, 3]);
t_gpu = tensor([4, 5, 6]).gpu();

result = t_cpu + t_gpu;  # ERROR: "Tensors on different devices: cpu + cuda"
```

### 8.3 Invalid Operations

```codesutra
t = tensor([1, 2, 3]);
result = t.reshape([2, 2]);  # ERROR: "Cannot reshape [3] into [2, 2]"

t2 = tensor([[1, 2], [3, 4]]);
result = t2.inverse();  # ERROR: "Matrix must be square for inverse"
```

---

## 9. Pretty Printing

### 9.1 Small Tensors

Display in matrix notation:

```codesutra
t = tensor([[1, 2, 3], [4, 5, 6]]);
print(t);

# Output:
# [[1 2 3]
#  [4 5 6]]
```

### 9.2 Large Tensors

Show shape and dtype info:

```codesutra
t = tensor.random([1000, 1000]);
print(t);

# Output:
# <tensor shape=[1000, 1000] dtype=float32 device=cpu>
```

### 9.3 High-Dimensional Tensors

```codesutra
t = tensor.random([10, 20, 30, 40]);
print(t);

# Output:
# <tensor shape=[10, 20, 30, 40] dtype=float32 device=cpu>
```

---

## 10. Example Usage Patterns

### 10.1 Simple Arithmetic

```codesutra
t1 = tensor([1, 2, 3]);
t2 = tensor([4, 5, 6]);

sum = t1 + t2;
prod = t1 * t2;
print(sum);   # [5, 7, 9]
print(prod);  # [4, 10, 18]
```

### 10.2 Matrix Operations

```codesutra
a = tensor([[1, 2, 3], [4, 5, 6]]);
b = tensor([[7, 8], [9, 10], [11, 12]]);

# Matrix multiplication (2x3) @ (3x2) = 2x2
c = matmul(a, b);
print(c.shape);  # [2, 2]
```

### 10.3 GPU Computation

```codesutra
# Create on CPU
x = tensor.random([1000, 1000]);
y = tensor.random([1000, 1000]);

# Move to GPU
x_gpu = x.gpu();
y_gpu = y.gpu();

# Compute on GPU
z_gpu = matmul(x_gpu, y_gpu);

# Move back to CPU
z = z_gpu.cpu();
print(z.shape);  # [1000, 1000]
```

### 10.4 Machine Learning Workflow

```codesutra
import torch;
import numpy as np;

# Data
x = tensor.random([100, 10]);
y = tensor.random([100, 1]);

# Model parameters
w = tensor.randn([10, 1], device: "cuda");
b = tensor.zeros([1], device: "cuda");

# Move data to GPU
x_gpu = x.gpu();
y_gpu = y.gpu();

# Forward pass
logits = matmul(x_gpu, w) + b;
pred = logits.sigmoid();

# Loss (simple MSE)
loss = ((pred - y_gpu) ** 2).mean();
print(loss);
```

### 10.5 NumPy Interoperability

```codesutra
import numpy as np;

# Create NumPy array
arr = np.random.randn(3, 3);

# Convert to tensor
t = tensor(arr);

# Compute
result = t @ t;  # matrix multiply

# Convert back to NumPy
result_np = result.numpy();
print(result_np);
```

---

## 11. Open Design Questions (For User Decision)

### Q1: Should we support in-place operations?

**Option A (Immutable - recommended)**: 
- All ops return new tensors
- Simpler semantics, safer
- Slightly higher memory overhead

**Option B (Mutable)**:
- Support `.relu_()`, `.mul_()` style operations
- Requires careful lifetime management
- More memory-efficient but complex

**→ Recommendation**: Option A (immutable). Aligns with CodeSutra's overall design philosophy.

---

### Q2: Matrix multiplication operator syntax?

**Option A (function only)**:
```codesutra
result = matmul(a, b);
```

**Option B (add @ operator)**:
```codesutra
result = a @ b;
```

**→ Recommendation**: Start with Option A (function). Add @ operator later if syntax allows.

---

### Q3: Dtype naming convention?

**Option A (NumPy-style)**:
```codesutra
tensor([1, 2, 3], dtype: "int64");
tensor([1.0, 2.0], dtype: "float32");
```

**Option B (Python-style)**:
```codesutra
tensor([1, 2, 3], dtype: int);
tensor([1.0, 2.0], dtype: float);
```

**→ Recommendation**: Option A (string-based). More explicit, avoids type system changes.

---

### Q4: Automatic device inference?

**Option A (explicit device)**:
```codesutra
t = tensor([1, 2, 3], device: "cuda");
```

**Option B (detect from system)**:
```codesutra
t = tensor([1, 2, 3]);  # Auto-detects GPU if available
```

**→ Recommendation**: Option A (explicit). Forces user awareness, prevents silent device migration.

---

### Q5: Lazy evaluation / JIT compilation?

**Option A (eager evaluation - recommended)**:
- All operations execute immediately
- Debugging is straightforward
- No compilation overhead

**Option B (lazy evaluation)**:
- Build computation graph first
- Execute on demand (like JAX/TensorFlow)
- Requires significant interpreter changes

**→ Recommendation**: Option A (eager). Keep language simple, add JIT later.

---

## 12. Implementation Roadmap

### Phase 1 (MVP - Core Tensor Type)

- [ ] Tensor type as first-class value
- [ ] Construction from lists
- [ ] Basic arithmetic operators
- [ ] NumPy backend (CPU only)
- [ ] `.shape`, `.dtype`, `.device` properties
- [ ] `.numpy()`, `.tolist()` conversions
- [ ] Pretty printing for small tensors

**Estimated**: 3-4 hours

### Phase 2 (Device Support)

- [ ] GPU device detection
- [ ] `.gpu()` and `.cpu()` methods
- [ ] Device mismatch error handling
- [ ] PyTorch backend integration

**Estimated**: 2-3 hours

### Phase 3 (Methods & Operations)

- [ ] `.reshape()`, `.transpose()`, `.squeeze()`, `.unsqueeze()`
- [ ] `.sum()`, `.mean()`, `.max()`, `.min()`
- [ ] `.relu()`, `.sigmoid()`, `.tanh()`
- [ ] `matmul()`, `dot()`, `norm()`

**Estimated**: 3-4 hours

### Phase 4 (Advanced Features & Polish)

- [ ] Factory functions (`tensor.zeros()`, `tensor.random()`, etc.)
- [ ] Advanced indexing and slicing
- [ ] Broadcasting for mixed tensor/scalar operations
- [ ] Comprehensive error messages
- [ ] Full test suite

**Estimated**: 4-5 hours

### Phase 5 (Documentation & Examples)

- [ ] Tensor API documentation
- [ ] Usage guide with examples
- [ ] ML workflow examples

**Estimated**: 2-3 hours

**Total Estimated Time**: 14-19 hours of development

---

## 13. Why This Design Wins

### 13.1 For Users

- **Simple**: Python-like syntax, no type annotations required
- **Powerful**: Full NumPy/PyTorch capabilities under the hood
- **Fast**: GPU support for real ML workloads
- **Familiar**: Similar to PyTorch, NumPy, JAX

### 13.2 For the Language

- **Unique**: Native tensor support differentiates CodeSutra from Python
- **Strategic**: Opens door to AI/ML community
- **Extensible**: Foundation for future GPU optimization, JIT, etc.
- **Coherent**: Fits naturally with existing Python interop

### 13.3 For Outreach

**Before tensors**: "CodeSutra is a Python-compatible language with Python interop"
- Generic, not memorable

**After tensors**: "CodeSutra is a lightweight, high-level language designed for AI workflows with native tensor support and seamless Python interoperability"
- Clear differentiator, compelling narrative

---

## 14. Summary

CodeSutra tensors are:

| Property | Value |
|----------|-------|
| **Type** | First-class value type |
| **Shape** | Dynamic, runtime-determined |
| **Dtype** | Dynamic, with explicit control |
| **Device** | CPU (NumPy) or GPU (PyTorch) |
| **Semantics** | Immutable (copy-on-write) |
| **Broadcasting** | NumPy-style, implicit |
| **Interop** | Zero-copy with NumPy/PyTorch |
| **Error Handling** | Clear, descriptive messages |
| **Pretty Printing** | Matrix-style for small, summary for large |

This design gives CodeSutra a **real, unique identity** in the language ecosystem while maintaining simplicity and coherence.

---

## Appendix: Syntax Summary

```codesutra
# Construction
t = tensor([1, 2, 3]);
t = tensor([[1, 2], [3, 4]]);
t = tensor([1, 2, 3], dtype: "float32", device: "cuda");

# Factories
t = tensor.zeros([3, 3]);
t = tensor.ones([2, 4]);
t = tensor.random([10, 10]);

# Properties
shape = t.shape;
dtype = t.dtype;
device = t.device;

# Operations
t2 = t + 5;
t2 = t * other_tensor;
t2 = t.reshape([6]);
t2 = t.transpose();

# Device
t_gpu = t.gpu();
t_cpu = t_gpu.cpu();

# Conversion
arr = t.numpy();
lst = t.tolist();

# Matrix ops
result = matmul(a, b);
norm = norm(t);

# Reduction
sum_val = t.sum();
mean_val = t.mean();
```

---

**This design is ready for implementation and external communication.**
