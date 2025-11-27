#!/usr/bin/env python3
"""
CodeSutra - Main entry point
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexer import Lexer
from parser import Parser
from interpreter import Interpreter


def run_repl():
    """Run interactive REPL"""
    interpreter = Interpreter()
    
    print("=" * 60)
    print("🚀 Welcome to CodeSutra!")
    print("=" * 60)
    print("Type 'exit' to quit, 'help' for help")
    print()
    
    while True:
        try:
            prompt = ">>> "
            code = input(prompt)
            
            if code.lower() == 'exit':
                print("Goodbye!")
                break
            elif code.lower() == 'help':
                print_help()
                continue
            elif not code.strip():
                continue
            
            # Tokenize
            lexer = Lexer(code)
            tokens = lexer.tokenize()
            
            # Parse
            parser = Parser(tokens)
            program = parser.parse()
            
            # Interpret
            result = interpreter.interpret(program)
            
            # Print result if not None
            if result is not None:
                print(interpreter._to_string(result))
        
        except SyntaxError as e:
            print(f"❌ SyntaxError: {e}", file=sys.stderr)
        except NameError as e:
            print(f"❌ NameError: {e}", file=sys.stderr)
        except RuntimeError as e:
            print(f"❌ RuntimeError: {e}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)


def run_file(filename: str):
    """Run a CodeSutra file"""
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' not found", file=sys.stderr)
        sys.exit(1)
    
    with open(filename, 'r') as f:
        code = f.read()
    
    try:
        # Tokenize
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        
        # Parse
        parser = Parser(tokens)
        program = parser.parse()
        
        # Interpret
        interpreter = Interpreter()
        interpreter.interpret(program)
    
    except SyntaxError as e:
        print(f"❌ SyntaxError: {e}", file=sys.stderr)
        sys.exit(1)
    except NameError as e:
        print(f"❌ NameError: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"❌ RuntimeError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


def print_help():
    """Print help information"""
    print("""
╔═══════════════════════════════════════════════════════════════╗
║              CodeSutra Language Features                      ║
╚═══════════════════════════════════════════════════════════════╝

📚 BASICS:
  Variables:    x = 10; name = "Alice"; arr = [1, 2, 3]
  Functions:    func add(a, b) { return a + b; }
  Comments:     # This is a comment

🔄 CONTROL FLOW:
  If/Else:      if (x > 5) { print("big"); } else { print("small"); }
  Loops:        for (i in range(10)) { print(i); }
  While:        while (x < 100) { x = x + 1; }
  Break:        break;
  Continue:     continue;

📊 DATA TYPES:
  Numbers:      42, 3.14, -17
  Strings:      "hello", 'world'
  Booleans:     true, false
  Arrays:       [1, 2, 3, 4, 5]
  Dicts:        {name: "Bob", age: 30}
  Nil:          nil

🔧 OPERATORS:
  Arithmetic:   +, -, *, /, %, **
  Comparison:   ==, !=, <, <=, >, >=
  Logical:      and, or, not
  Ternary:      x > 5 ? "yes" : "no"
  Assignment:   =, +=, -=, *=, /=

📦 BUILT-IN FUNCTIONS:
  Math:         sqrt(), pow(), abs(), floor(), ceil(), round()
                sin(), cos(), tan(), log(), exp(), random()
                min(), max()
  
  String:       upper(), lower(), trim(), split(), join()
                starts_with(), ends_with(), contains()
                substring(), replace(), char_at(), repeat()
  
  Array:        length(), push(), pop(), shift(), unshift()
                reverse(), sort(), join()
  
  Dict:         keys(), values(), has()
  
  Type:         type(), number(), string(), bool()

🎯 EXAMPLES:
  print("Hello, World!");
  for (i in range(10)) { print(i * i); }
  func fibonacci(n) { return n <= 1 ? n : fibonacci(n-1) + fibonacci(n-2); }

Type 'exit' to quit the REPL.
    """)


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        # Run file
        run_file(sys.argv[1])
    else:
        # Run REPL
        run_repl()


if __name__ == '__main__':
    main()
