# �� CodeSutra - A Beginner-Friendly High-Level Programming Language

CodeSutra is a Python/R-inspired programming language designed to be **extremely beginner-friendly** while maintaining powerful capabilities for data manipulation, mathematics, and automation.

## ✨ Key Features


- **🚀 Native Tensor Type**: First-class support for multi-dimensional arrays with NumPy (CPU) and PyTorch (GPU) backends
- **🐍 Python Interoperability**: Seamlessly use NumPy, Pandas, PyTorch, TensorFlow, and other Python libraries from CodeSutra
## 📋 Language Basics

### Hello World
```codesutra
print("Hello, World!")
```

### Variables & Types
```codesutra
name = "Alice"
age = 25
height = 5.8
is_student = true
numbers = [1, 2, 3, 4, 5]
person = {name: "Bob", age: 30}
```

### Functions
```codesutra
func greet(name) {
  return "Hello, " + name + "!"
}

result = greet("CodeSutra")
print(result)
```

### Control Flow
```codesutra
if age >= 18 {
  print("Adult")
} else {
  print("Minor")
}

for i in range(5) {
  print(i)
}
```

### Lists & Data
```codesutra
nums = [1, 2, 3, 4, 5]
doubled = map(func(x) { return x * 2 }, nums)
sum_val = reduce(func(a, b) { return a + b }, nums, 0)
```

### Math Operations
```codesutra
result = sqrt(16)        # 4
power = pow(2, 3)        # 8
rounded = round(3.14159, 2)  # 3.14
```

### 🎯 Native Tensors for AI/ML
```codesutra
# Create tensors from lists
t1 = tensor([1, 2, 3]);
t2 = tensor([4, 5, 6]);

# Arithmetic operations
result = t1 + t2;       # [5, 7, 9]
product = t1 * 2;       # [2, 4, 6]

# Tensor properties
print(t1.shape);        # [3]
print(t1.dtype);        # int64
print(t1.device);       # cpu

# Reduction operations
total = sum(t1);        # 6
average = mean(t1);     # 2.0

# Create special tensors
zeros_matrix = tensor.zeros([3, 3]);
ones_vector = tensor.ones([5]);
```

See [TENSOR.md](docs/TENSOR.md) for comprehensive tensor documentation.

## 🏗️ Project Structure

```
CodeSutra/
├── src/
│   ├── lexer.py          # Tokenization
│   ├── parser.py         # AST generation
│   ├── interpreter.py    # Execution engine
│   ├── builtin.py        # Standard library
│   └── main.py           # Entry point
├── examples/             # Example programs
├── docs/                 # Documentation
├── tests/                # Test suite
└── README.md
```

## 🔨 Implementation Components

### 1. Lexer (Tokenization)
- Converts source code into tokens
- Handles strings, numbers, operators, keywords
- Tracks line numbers for error reporting

### 2. Parser (AST Generation)
- Recursive descent parser
- Builds abstract syntax tree
- Handles expressions, statements, functions

### 3. Interpreter (Runtime)
- Tree-walking interpreter
- Manages variable scope and environments
- Handles function calls and control flow

### 4. Standard Library
- Math functions (sqrt, pow, sin, cos, etc.)
- String operations
- List/Array operations (map, filter, reduce)
- Data structure utilities
- Type conversion functions

## 🚀 Quick Start

```bash
# Run the REPL
python src/main.py

# Run a script
python src/main.py examples/hello.codesutra

# Run tests
python -m pytest tests/
```

## 📚 Documentation

- [Language Syntax Guide](docs/SYNTAX.md)
- [Standard Library Reference](docs/STDLIB.md)
- [Examples](examples/)
- [Architecture Guide](docs/ARCHITECTURE.md)

## 🎯 Roadmap

- [x] Lexer/Tokenizer
- [x] Parser & AST
- [x] Basic Interpreter
- [x] Control Flow (if/else, loops)
- [x] Functions
- [x] Built-in Data Types
- [x] Standard Library
- [ ] Module system
- [ ] Error handling improvements
- [ ] Performance optimizations
- [ ] Compiler to bytecode

## 💡 Design Philosophy

CodeSutra prioritizes:
1. **Readability**: Code should be self-explanatory
2. **Simplicity**: Minimal keywords, clear semantics
3. **Expressiveness**: Powerful operations with clean syntax
4. **Learner-Friendly**: Great for beginners, grows with skill
5. **Practical**: Real-world capabilities from day one

## �� Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## 📄 License

MIT License - See LICENSE file for details

---

**Happy Coding with CodeSutra!** 🎓
