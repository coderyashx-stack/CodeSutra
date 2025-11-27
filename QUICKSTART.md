# CodeSutra Quick Start Guide

## Installation

No installation needed! CodeSutra requires Python 3.7+.

## Running CodeSutra

### Interactive REPL (Read-Eval-Print Loop)

```bash
cd /workspaces/CodeSutra
python src/main.py
```

Then type CodeSutra code directly:

```
>>> x = 10
>>> y = 20
>>> print(x + y)
30
>>> func square(n) { return n * n; }
>>> square(5)
25
```

Type `help` for built-in help, or `exit` to quit.

### Run a Script File

```bash
python src/main.py path/to/script.codesutra
```

Example:
```bash
python src/main.py examples/hello.codesutra
python src/main.py examples/fibonacci.codesutra
python src/main.py examples/primes.codesutra
```

## Your First Program

Create a file `hello.codesutra`:

```codesutra
print("Hello, World!");
```

Run it:
```bash
python src/main.py hello.codesutra
```

## Basic Concepts

### Variables

```codesutra
name = "Alice";
age = 25;
height = 5.8;
```

### Data Types

```codesutra
# Numbers
x = 42;
y = 3.14;

# Strings
greeting = "Hello";

# Booleans
is_valid = true;

# Arrays
numbers = [1, 2, 3, 4, 5];

# Dictionaries
person = {name: "Bob", age: 30, city: "NYC"};

# Nil (null)
nothing = nil;
```

### Control Flow

```codesutra
# If/Else
if (age >= 18) {
    print("Adult");
} else {
    print("Minor");
}

# Loops
for (i in range(5)) {
    print(i);
}

# While
x = 0;
while (x < 5) {
    print(x);
    x = x + 1;
}
```

### Functions

```codesutra
# Function declaration
func add(a, b) {
    return a + b;
}

# Call function
result = add(3, 4);
print(result);  # 7

# Anonymous functions
double = func(x) { return x * 2; };
print(double(5));  # 10
```

### Arrays and Dictionaries

```codesutra
# Array operations
arr = [1, 2, 3];
length(arr);        # 3
push(arr, 4);       # Adds 4 to end
popped = pop(arr);  # Removes and returns 4
reversed = reverse([1, 2, 3]);  # [3, 2, 1]

# Dictionary operations
person = {name: "Alice", age: 30};
person.name;               # "Alice"
person["age"];             # 30
keys(person);              # ["name", "age"]
values(person);            # ["Alice", 30]
has(person, "name");       # true
```

### String Operations

```codesutra
text = "Hello";
upper(text);       # "HELLO"
lower(text);       # "hello"
length(text);      # 5
contains(text, "ell");   # true
split("a,b,c", ",");     # ["a", "b", "c"]
join(["a", "b"], "-");   # "a-b"
```

### Math Operations

```codesutra
sqrt(16);          # 4
pow(2, 3);         # 8
abs(-5);           # 5
floor(3.7);        # 3
ceil(3.2);         # 4
round(3.14159, 2); # 3.14

sin(PI/2);         # 1
cos(0);            # 1

min(1, 2, 3);      # 1
max(1, 2, 3);      # 3
```

## Example Programs

See the `examples/` directory for complete programs:

- `hello.codesutra` - Simple Hello World
- `loops.codesutra` - Loop examples
- `factorial.codesutra` - Recursive factorial
- `fibonacci.codesutra` - Fibonacci sequence
- `arrays.codesutra` - Array operations
- `strings.codesutra` - String manipulation
- `dicts.codesutra` - Dictionary operations
- `math.codesutra` - Math functions
- `functions.codesutra` - Function examples
- `grades.codesutra` - Grade calculator
- `primes.codesutra` - Prime number checker

## Documentation

- [Language Syntax Guide](docs/SYNTAX.md) - Complete syntax reference
- [Standard Library Reference](docs/STDLIB.md) - All built-in functions
- [Architecture Guide](docs/ARCHITECTURE.md) - How CodeSutra works

## Common Errors

### `NameError: Undefined variable 'x'`
You're trying to use a variable that hasn't been defined yet.

```codesutra
# Wrong
print(x);  # Error: x not defined

# Right
x = 10;
print(x);
```

### `SyntaxError`
You have a syntax error in your code. Check brackets, semicolons, etc.

```codesutra
# Wrong
func test() { print("missing closing brace"

# Right
func test() { print("ok"); }
```

### Division by Zero
You can't divide by zero.

```codesutra
# Wrong
result = 10 / 0;

# Right
if (divisor != 0) {
    result = 10 / divisor;
}
```

## Tips & Tricks

### Comments
```codesutra
# This is a comment
# Use them to explain your code
```

### Ternary Operator
```codesutra
status = age >= 18 ? "Adult" : "Minor";
```

### Early Return
```codesutra
func check(x) {
    if (x < 0) {
        return "negative";
    }
    return "positive";
}
```

### Compound Assignment
```codesutra
x = 10;
x += 5;   # Same as x = x + 5
x -= 3;   # Same as x = x - 3
x *= 2;   # Same as x = x * 2
x /= 2;   # Same as x = x / 2
```

## Need Help?

1. Check the [Syntax Guide](docs/SYNTAX.md)
2. Look at the [Standard Library](docs/STDLIB.md)
3. Run the examples in the `examples/` folder
4. Type `help` in the REPL

Happy coding! 🚀
