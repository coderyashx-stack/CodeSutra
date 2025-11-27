# CodeSutra Standard Library Reference

## Type Conversion Functions

### `number(value)`
Converts a value to a number.
```codesutra
number("42")      # 42
number("3.14")    # 3.14
number(true)      # 1
number(false)     # 0
```

### `string(value)`
Converts a value to a string.
```codesutra
string(42)        # "42"
string(true)      # "true"
string([1,2,3])   # "[1, 2, 3]"
```

### `bool(value)`
Converts a value to a boolean (checks truthiness).
```codesutra
bool(1)           # true
bool(0)           # false
bool("")          # false
bool("text")      # true
```

### `type(value)`
Returns the type of a value as a string.
```codesutra
type(42)          # "number"
type("hi")        # "string"
type([])          # "array"
type(nil)         # "nil"
```

## Array Functions

### `length(array)`
Returns the number of elements in an array.
```codesutra
length([1, 2, 3])       # 3
length([])              # 0
```

### `push(array, value)`
Adds an element to the end of an array.
```codesutra
arr = [1, 2];
push(arr, 3);   # arr is now [1, 2, 3]
```

### `pop(array)`
Removes and returns the last element.
```codesutra
arr = [1, 2, 3];
popped = pop(arr);  # popped: 3, arr: [1, 2]
```

### `shift(array)`
Removes and returns the first element.
```codesutra
arr = [1, 2, 3];
shifted = shift(arr);  # shifted: 1, arr: [2, 3]
```

### `unshift(array, value)`
Adds an element to the beginning of an array.
```codesutra
arr = [2, 3];
unshift(arr, 1);  # arr is now [1, 2, 3]
```

### `join(array, separator)`
Joins array elements into a string.
```codesutra
join([1, 2, 3], ",")     # "1,2,3"
join(["a", "b"], "-")    # "a-b"
```

### `reverse(array)`
Returns a new reversed array.
```codesutra
reverse([1, 2, 3])   # [3, 2, 1]
```

### `sort(array)`
Returns a sorted array.
```codesutra
sort([3, 1, 2])      # [1, 2, 3]
```

## String Functions

### `length(string)`
Returns the number of characters in a string.
```codesutra
length("hello")    # 5
length("")         # 0
```

### `upper(string)`
Converts string to uppercase.
```codesutra
upper("hello")     # "HELLO"
```

### `lower(string)`
Converts string to lowercase.
```codesutra
lower("HELLO")     # "hello"
```

### `trim(string)`
Removes leading and trailing whitespace.
```codesutra
trim("  hello  ")   # "hello"
```

### `split(string, separator)`
Splits string into an array.
```codesutra
split("a,b,c", ",")     # ["a", "b", "c"]
split("hello", "")      # ["h", "e", "l", "l", "o"]
```

### `join(array, separator)`
Joins array elements with separator.
```codesutra
join(["a", "b", "c"], ",")    # "a,b,c"
```

### `starts_with(string, prefix)`
Checks if string starts with prefix.
```codesutra
starts_with("hello", "he")    # true
starts_with("hello", "lo")    # false
```

### `ends_with(string, suffix)`
Checks if string ends with suffix.
```codesutra
ends_with("hello", "lo")      # true
ends_with("hello", "he")      # false
```

### `contains(string, substring)`
Checks if string contains substring.
```codesutra
contains("hello", "ell")      # true
contains("hello", "xyz")      # false
```

### `index_of(string, substring)`
Returns the index of substring (-1 if not found).
```codesutra
index_of("hello", "ll")       # 2
index_of("hello", "x")        # -1
```

### `substring(string, start, end)`
Extracts substring.
```codesutra
substring("hello", 1, 4)      # "ell"
substring("hello", 0)         # "hello"
```

### `replace(string, search, replacement)`
Replaces all occurrences.
```codesutra
replace("hello", "l", "L")    # "heLLo"
```

### `char_at(string, index)`
Gets character at index.
```codesutra
char_at("hello", 0)           # "h"
char_at("hello", 4)           # "o"
```

### `repeat(string, count)`
Repeats string count times.
```codesutra
repeat("ab", 3)               # "ababab"
```

## Dictionary Functions

### `keys(dictionary)`
Returns array of dictionary keys.
```codesutra
keys({a: 1, b: 2})    # ["a", "b"]
```

### `values(dictionary)`
Returns array of dictionary values.
```codesutra
values({a: 1, b: 2})  # [1, 2]
```

### `has(dictionary, key)`
Checks if dictionary has a key.
```codesutra
has({a: 1, b: 2}, "a")    # true
has({a: 1, b: 2}, "c")    # false
```

## Math Functions

### `sqrt(number)`
Square root.
```codesutra
sqrt(16)     # 4
sqrt(2)      # 1.414...
```

### `pow(base, exponent)`
Power function (base^exponent).
```codesutra
pow(2, 3)    # 8
pow(5, 2)    # 25
```

### `abs(number)`
Absolute value.
```codesutra
abs(-5)      # 5
abs(3)       # 3
```

### `floor(number)`
Rounds down to nearest integer.
```codesutra
floor(3.7)   # 3
floor(3.2)   # 3
```

### `ceil(number)`
Rounds up to nearest integer.
```codesutra
ceil(3.2)    # 4
ceil(3.7)    # 4
```

### `round(number, decimals)`
Rounds to nearest value.
```codesutra
round(3.14159)         # 3
round(3.14159, 2)      # 3.14
round(3.14159, 4)      # 3.1416
```

### `min(numbers...)`
Returns minimum value.
```codesutra
min(3, 1, 4, 1, 5)     # 1
```

### `max(numbers...)`
Returns maximum value.
```codesutra
max(3, 1, 4, 1, 5)     # 5
```

### Trigonometric Functions

### `sin(radians)`, `cos(radians)`, `tan(radians)`
Trigonometric functions.
```codesutra
sin(0)         # 0
cos(0)         # 1
tan(0)         # 0
sin(PI/2)      # 1
```

### `log(number, base)`
Logarithm. Default base is e.
```codesutra
log(10)        # log base e of 10
log(100, 10)   # 2
```

### `exp(number)`
e raised to the power of number.
```codesutra
exp(1)         # 2.718... (e)
exp(0)         # 1
```

### `random(min, max)`
Returns random number between min and max.
```codesutra
random()           # Random between 0 and 1
random(1, 100)     # Random between 1 and 100
```

## Range Function

### `range(end)` or `range(start, end, step)`
Creates array of numbers.
```codesutra
range(5)           # [0, 1, 2, 3, 4]
range(1, 5)        # [1, 2, 3, 4]
range(0, 10, 2)    # [0, 2, 4, 6, 8]
```

## Constants

### `PI`
Mathematical constant π (3.14159...).
```codesutra
PI     # 3.14159265...
```

### `E`
Mathematical constant e (2.71828...).
```codesutra
E      # 2.71828182...
```

## Output Function

### `print(value)`
Prints a value to output.
```codesutra
print("Hello, World!");
print(42);
print([1, 2, 3]);
```
