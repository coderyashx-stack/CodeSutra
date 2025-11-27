# CodeSutra Language Syntax Guide

## Table of Contents
1. [Variables](#variables)
2. [Data Types](#data-types)
3. [Operators](#operators)
4. [Control Flow](#control-flow)
5. [Functions](#functions)
6. [Advanced Features](#advanced-features)

## Variables

### Declaration
```codesutra
x = 10;                    # Implicit type
let name = "Alice";        # Explicit let
const PI = 3.14159;       # Constant (immutable)
```

### Naming Rules
- Must start with letter or underscore
- Can contain letters, numbers, underscores
- Case-sensitive
- No reserved word keywords

## Data Types

### Numbers
```codesutra
integer = 42;
float = 3.14;
negative = -17;
exponential = 1.5e-4;
```

### Strings
```codesutra
single = 'hello';
double = "world";
empty = "";
multiline = "line1\nline2";    # Escape sequences: \n, \t, \r, \\, \"
```

### Booleans
```codesutra
is_valid = true;
is_empty = false;
```

### Arrays
```codesutra
empty_array = [];
numbers = [1, 2, 3, 4, 5];
mixed = [1, "hello", true, nil];
nested = [[1, 2], [3, 4]];
```

### Dictionaries
```codesutra
empty_dict = {};
person = {
    name: "Bob",
    age: 30,
    city: "NYC"
};
```

### Nil
```codesutra
nothing = nil;
```

## Operators

### Arithmetic
```codesutra
a + b      # Addition
a - b      # Subtraction
a * b      # Multiplication
a / b      # Division
a % b      # Modulo
a ** b     # Exponentiation (power)
```

### Comparison
```codesutra
a == b     # Equal
a != b     # Not equal
a < b      # Less than
a <= b     # Less than or equal
a > b      # Greater than
a >= b     # Greater than or equal
```

### Logical
```codesutra
a and b    # Logical AND
a or b     # Logical OR
not a      # Logical NOT
```

### Assignment
```codesutra
x = 10;        # Direct assignment
x += 5;        # Add and assign
x -= 3;        # Subtract and assign
x *= 2;        # Multiply and assign
x /= 2;        # Divide and assign
```

### Other
```codesutra
condition ? true_val : false_val   # Ternary operator
obj[index]     # Array/dict access
obj.property   # Object property access
```

## Control Flow

### If Statement
```codesutra
if (condition) {
    # code
}

if (condition) {
    # code
} else {
    # code
}

if (condition1) {
    # code
} else if (condition2) {
    # code
} else {
    # code
}
```

### While Loop
```codesutra
while (condition) {
    # code
}
```

### For Loop
```codesutra
for (variable in iterable) {
    # code
}

# Examples:
for (i in range(10)) { print(i); }
for (item in [1, 2, 3]) { print(item); }
for (key in dict) { print(key); }
```

### Break and Continue
```codesutra
while (true) {
    if (condition) {
        break;      # Exit loop
    }
    if (skip_cond) {
        continue;   # Skip to next iteration
    }
}
```

## Functions

### Function Declaration
```codesutra
func add(a, b) {
    return a + b;
}

func greet(name) {
    print("Hello, " + name);
}

func no_args() {
    return 42;
}

func variadic(a, b, c) {
    return a + b + c;
}
```

### Function Calls
```codesutra
result = add(5, 3);
greet("Alice");
value = no_args();
```

### Anonymous Functions
```codesutra
double = func(x) { return x * 2; };
result = double(5);

# First-class functions
apply = func(f, x) { return f(x); };
apply(double, 10);
```

### Return Statement
```codesutra
func example() {
    if (condition) {
        return "early";
    }
    return "normal";
}
```

## Advanced Features

### String Operations
```codesutra
str = "Hello";
upper_str = upper(str);           # "HELLO"
lower_str = lower(str);           # "hello"
trimmed = trim("  text  ");       # "text"
parts = split("a,b,c", ",");     # ["a", "b", "c"]
joined = join(["a", "b"], "-");  # "a-b"
```

### Array Operations
```codesutra
arr = [1, 2, 3];
length(arr);                 # 3
push(arr, 4);               # arr now [1, 2, 3, 4]
popped = pop(arr);          # popped: 4, arr: [1, 2, 3]
reversed = reverse(arr);    # [3, 2, 1]
```

### Math Functions
```codesutra
sqrt(16)           # 4
pow(2, 3)          # 8
abs(-5)            # 5
floor(3.7)         # 3
ceil(3.2)          # 4
round(3.14159, 2)  # 3.14
sin(PI/2)          # 1
cos(0)             # 1
```

### Type Conversion
```codesutra
number("42")       # 42
string(42)         # "42"
bool(1)            # true
type(42)           # "number"
```

### Accessing Collections
```codesutra
# Array access
arr = [10, 20, 30];
arr[0]             # 10
arr[1]             # 20
arr[-1]            # 30 (negative indexing)

# String access
str = "Hello";
str[0]             # "H"
str[4]             # "o"

# Dictionary access
person = {name: "Alice", age: 30};
person["name"]     # "Alice"
person.name        # "Alice" (dot notation)
```

## Comments
```codesutra
# This is a single-line comment
# Multiple comment lines
```

## Operator Precedence (High to Low)
1. Primary (literals, identifiers, parentheses, brackets)
2. Postfix (function calls, array access, member access)
3. Unary (!, -)
4. Power (**)
5. Multiplicative (*, /, %)
6. Additive (+, -)
7. Comparison (<, <=, >, >=)
8. Equality (==, !=)
9. Logical AND (and)
10. Logical OR (or)
11. Ternary (? :)
12. Assignment (=, +=, -=, *=, /=)
