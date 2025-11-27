# Contributing to CodeSutra

Thank you for your interest in contributing to CodeSutra! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/coderyashx-stack/CodeSutra.git
   cd CodeSutra
   ```

2. **Understand the Structure**
   - `src/lexer.py` - Tokenization
   - `src/parser.py` - AST generation
   - `src/interpreter.py` - Execution engine
   - `src/builtin.py` - Standard library
   - `src/main.py` - Entry point
   - `examples/` - Example programs
   - `docs/` - Documentation
   - `tests/` - Unit tests

## Development Workflow

### Setting Up for Development

```bash
cd /workspaces/CodeSutra
# All dependencies are built-in Python
```

### Testing Your Changes

```bash
# Test a specific example
python src/main.py examples/hello.codesutra

# Run the REPL
python src/main.py
```

### Creating New Examples

1. Create a `.codesutra` file in `examples/`
2. Write your example code
3. Test it with: `python src/main.py examples/myfile.codesutra`

## Areas for Contribution

### 1. **Language Features**
- [ ] Module/import system
- [ ] Object-oriented features (classes, inheritance)
- [ ] Lambda expressions (shorter syntax for anonymous functions)
- [ ] Multiple assignment: `a, b = 1, 2`
- [ ] Slice syntax: `arr[1:3]`
- [ ] Try/catch error handling
- [ ] Regular expressions
- [ ] Destructuring

### 2. **Standard Library**
- [ ] File I/O functions
- [ ] More string methods (format, interpolation)
- [ ] Array/collection operations (map, filter with cleaner syntax)
- [ ] JSON parsing
- [ ] Date/time functions
- [ ] HTTP requests
- [ ] Data processing utilities

### 3. **Performance**
- [ ] Bytecode compilation
- [ ] JIT compilation
- [ ] Constant folding
- [ ] Dead code elimination
- [ ] Caching

### 4. **Developer Experience**
- [ ] Better error messages with stack traces
- [ ] Line numbers in error reports
- [ ] IDE support/LSP server
- [ ] Debugger
- [ ] Package manager
- [ ] Interactive documentation

### 5. **Testing**
- [ ] Unit tests for lexer
- [ ] Unit tests for parser
- [ ] Unit tests for interpreter
- [ ] Integration tests
- [ ] Example tests

## Code Style

### Python Code Style
- Use PEP 8 conventions
- Use type hints where possible
- Keep functions focused and small
- Document complex logic
- Use meaningful variable names

Example:
```python
def calculate_sum(numbers: List[int]) -> int:
    """Calculate the sum of a list of numbers."""
    total = 0
    for num in numbers:
        total += num
    return total
```

### CodeSutra Code Style
- Use clear variable names
- Comment complex logic
- Break large functions into smaller ones
- Use meaningful function names

Example:
```codesutra
# Calculate factorial using recursion
func factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}
```

## Submitting Changes

### Steps to Submit

1. **Fork the Repository** (if needed)

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Test thoroughly
   - Update documentation if needed

4. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add description of changes"
   ```

5. **Push to Your Branch**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Describe what you changed
   - Explain why you made the changes
   - Link to related issues if applicable

### Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain the changes and why
- **Tests**: Include tests for new features
- **Documentation**: Update docs if needed
- **Examples**: Include example code if applicable
- **No Breaking Changes**: Unless necessary and documented

### Example PR Description

```markdown
## Description
Adds support for slice syntax in arrays.

## Changes
- Modified parser to handle slice expressions
- Added visit_slice method to interpreter
- Updated standard library documentation

## Testing
- Added test_array_slicing.py
- Tested with examples/slicing.codesutra

## Related Issues
Closes #42
```

## Reporting Issues

### Bug Reports

Include:
- CodeSutra version (git commit or date)
- Python version
- Operating system
- Code that triggers the bug
- Expected vs actual behavior
- Error message/stack trace

Example:
```markdown
**Description**: Division by zero crashes interpreter

**Environment**:
- CodeSutra: latest main
- Python: 3.9.5
- OS: Ubuntu 20.04

**Steps to Reproduce**:
```codesutra
result = 10 / 0;
```

**Expected**: Error message
**Actual**: Crash with exception

**Error**:
```
ZeroDivisionError: division by zero
```
```

### Feature Requests

Include:
- Clear description of the feature
- Use cases and examples
- Why it's useful
- Any implementation ideas

Example:
```markdown
**Request**: Add string interpolation

**Use Case**: Make string formatting easier
```codesutra
name = "World"
print("Hello, ${name}!")
```

**Benefits**: Cleaner syntax than concatenation
```

## Documentation Guidelines

### Writing Documentation

- **Use Clear Language**: Avoid jargon
- **Include Examples**: Show how to use features
- **Cover Edge Cases**: Document special behavior
- **Keep it Accurate**: Update docs with code changes
- **Organize Logically**: Group related topics

### Documentation Files

- `README.md` - Overview and features
- `QUICKSTART.md` - Getting started guide
- `docs/SYNTAX.md` - Language syntax
- `docs/STDLIB.md` - Built-in functions
- `docs/ARCHITECTURE.md` - Internal design
- Code comments - Explain complex logic

## Development Tools

### Debugging

Use Python's built-in tools:
```python
import pdb
pdb.set_trace()  # Breakpoint
```

### Testing Code Snippets

In the REPL:
```bash
python src/main.py
>>> # Test your code here
```

### Running Specific Examples

```bash
python src/main.py examples/fibonacci.codesutra
```

## Release Process

1. Update version number
2. Update CHANGELOG
3. Run all tests
4. Create release notes
5. Tag release: `git tag v1.0.0`
6. Create GitHub release

## Code Review Process

- **Maintainers Review**: Check for quality, style, correctness
- **Automated Tests**: Must pass CI/CD
- **Documentation**: Must be updated
- **Feedback**: Constructive comments
- **Approval**: 1+ maintainer approval required

## Community

- **Discussions**: Ask questions, share ideas
- **Issues**: Report bugs and feature requests
- **PRs**: Contribute code improvements
- **Examples**: Share cool programs

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue
- Start a discussion
- Check existing documentation

Thank you for contributing to CodeSutra! 🎓
