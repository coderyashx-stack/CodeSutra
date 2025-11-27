# CodeSutra Architecture Guide

## Overview

CodeSutra is built as a tree-walking interpreter with four main components:

1. **Lexer** - Tokenization
2. **Parser** - AST Generation
3. **Interpreter** - Execution Engine
4. **Built-in Library** - Standard Functions

## Architecture Diagram

```
Source Code
    ↓
[LEXER] → Tokens
    ↓
[PARSER] → AST (Abstract Syntax Tree)
    ↓
[INTERPRETER] → Results
    ↓
Output / Values
```

## Component Details

### 1. Lexer (`lexer.py`)

The lexer converts raw source code into a stream of tokens.

**Key Components:**
- `TokenType` enum - All possible token types
- `Token` dataclass - Represents a single token
- `Lexer` class - Performs tokenization

**Process:**
1. Reads characters one by one
2. Identifies token boundaries
3. Creates Token objects with type, lexeme, and literal value
4. Handles strings, numbers, identifiers, operators, keywords

**Tokens Include:**
- Literals: numbers, strings, booleans, nil
- Identifiers and keywords
- Operators: arithmetic, comparison, logical, assignment
- Delimiters: parentheses, braces, brackets, etc.

### 2. Parser (`parser.py`)

The parser builds an Abstract Syntax Tree (AST) from tokens.

**Key Components:**
- AST Node Classes - Represent code structure
- `Parser` class - Recursive descent parser

**AST Node Types:**

**Expressions:**
- `NumberExpr`, `StringExpr`, `BoolExpr`, `NilExpr`
- `IdentifierExpr`, `ArrayExpr`, `DictExpr`
- `BinaryOpExpr`, `UnaryOpExpr`
- `CallExpr`, `MemberExpr`, `IndexExpr`
- `AssignExpr`, `FuncExpr`, `TernaryExpr`

**Statements:**
- `ExprStmt`, `BlockStmt`
- `IfStmt`, `WhileStmt`, `ForStmt`
- `FuncDeclStmt`, `ReturnStmt`
- `BreakStmt`, `ContinueStmt`, `VarDeclStmt`
- `Program` - Root node

**Parsing Strategy:**
- Recursive descent with operator precedence
- Handles operator precedence correctly
- Reports syntax errors with location info

### 3. Interpreter (`interpreter.py`)

The interpreter executes the AST using the visitor pattern.

**Key Components:**
- `Environment` - Manages variable scope
- `Interpreter` - Visitor that executes nodes
- Exception classes for control flow

**Execution:**
1. Starts with global environment containing built-in functions
2. Creates new environments for function scopes
3. Executes statements and evaluates expressions
4. Maintains variable bindings through environments

**Key Features:**
- Automatic memory management (Python's GC)
- Scope management with environment chains
- Control flow exceptions (break, continue, return)
- Type coercion (implicit conversions)

**Environment Chain:**
```
Global Environment (built-ins)
    ↓
Function 1 Environment (parent: global)
    ↓
Function 2 Environment (parent: Function 1)
```

### 4. Built-in Library (`builtin.py`)

Provides standard functions and utilities.

**Categories:**
- Type conversion: `number()`, `string()`, `bool()`, `type()`
- Array operations: `push()`, `pop()`, `length()`, etc.
- String functions: `upper()`, `lower()`, `split()`, etc.
- Dictionary operations: `keys()`, `values()`, `has()`
- Math functions: `sqrt()`, `pow()`, `sin()`, `cos()`, etc.
- Range: `range()` for iteration

## Data Types

CodeSutra supports these data types:

```
Number      - int or float (internally all floats)
String      - UTF-8 text
Boolean     - true/false
Nil         - null value
Array       - [1, 2, 3]
Dictionary  - {key: value}
Function    - func or CodeSutraFunction
```

## Execution Flow Example

For code: `print(2 + 3);`

```
1. LEXING:
   Token(NUMBER, "2", 2)
   Token(PLUS, "+")
   Token(NUMBER, "3", 3)
   Token(LPAREN, "(")
   ... etc

2. PARSING:
   ExprStmt(
     CallExpr(
       IdentifierExpr("print"),
       [BinaryOpExpr(
         NumberExpr(2),
         Token(PLUS),
         NumberExpr(3)
       )]
     )
   )

3. INTERPRETING:
   - Evaluate CallExpr
   - Look up "print" in environment → built-in function
   - Evaluate arguments: 2 + 3 → 5
   - Call print(5) → outputs "5"
```

## Key Design Decisions

### 1. Tree-Walking Interpreter
- Simple to understand and implement
- Good for learning and prototyping
- Slower than compiled/bytecode approaches
- Easy to add features

### 2. Dynamic Typing
- Types checked at runtime
- Type coercion for operations
- Flexible but less type-safe

### 3. Automatic Garbage Collection
- Uses Python's GC
- No manual memory management needed
- Beginner-friendly

### 4. Visitor Pattern
- Clean separation of structure (AST) and logic (interpretation)
- Easy to add new operations
- Extensible for future features

## Scope and Environment Management

Variables are looked up through an environment chain:

```python
class Environment:
    def get(name):
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get(name)  # Look up chain
        raise NameError()
```

This enables:
- Local variable shadowing
- Closure support
- Clean separation of scopes

## Error Handling

Three main error types:

1. **SyntaxError** - Lexer or parser issues
2. **NameError** - Undefined variable
3. **RuntimeError** - Type mismatches, operations

## Future Enhancements

Potential improvements:
1. Bytecode compilation
2. Module/import system
3. Object-oriented features
4. Better error messages with stack traces
5. Debugging support
6. Performance optimizations
7. Jit compilation

## Performance Characteristics

Current (Tree-walking):
- Simple programs: Fast enough
- Recursive programs: Slow with deep recursion
- Loops: Acceptable but not optimized

Optimization opportunities:
- Bytecode compilation
- Caching compiled functions
- JIT compilation
- Constant folding
- Dead code elimination
