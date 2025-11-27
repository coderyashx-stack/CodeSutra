"""
CodeSutra Interpreter - Executes the AST
"""

from typing import Any, Dict, Optional, List
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parser import *
from builtin import CodeSutraFunction, create_globals
import importlib
from python_bridge import wrap


class BreakException(Exception):
    """Exception for break statement"""
    pass


class ContinueException(Exception):
    """Exception for continue statement"""
    pass


class ReturnException(Exception):
    """Exception for return statement"""
    def __init__(self, value):
        self.value = value


class Environment:
    """Represents a scope for variables"""
    
    def __init__(self, parent: Optional['Environment'] = None):
        self.parent = parent
        self.variables: Dict[str, Any] = {}
    
    def define(self, name: str, value: Any):
        """Define a variable in this environment"""
        self.variables[name] = value
    
    def get(self, name: str) -> Any:
        """Get a variable value"""
        if name in self.variables:
            return self.variables[name]
        if self.parent:
            return self.parent.get(name)
        raise NameError(f"Undefined variable '{name}'")
    
    def set(self, name: str, value: Any):
        """Set a variable value"""
        if name in self.variables:
            self.variables[name] = value
            return
        if self.parent:
            self.parent.set(name, value)
            return
        raise NameError(f"Undefined variable '{name}'")
    
    def exists(self, name: str) -> bool:
        """Check if variable exists"""
        if name in self.variables:
            return True
        if self.parent:
            return self.parent.exists(name)
        return False


class Interpreter:
    """Interprets and executes CodeSutra programs"""
    
    def __init__(self):
        self.global_env = Environment()
        self.current_env = self.global_env
        self._setup_globals()
    
    def _setup_globals(self):
        """Setup global functions and constants"""
        globals_dict = create_globals()
        for name, value in globals_dict.items():
            self.global_env.define(name, value)
    
    def interpret(self, program: Program) -> Any:
        """Interpret a program"""
        try:
            result = None
            for stmt in program.statements:
                result = stmt.accept(self)
            return result
        except ReturnException:
            raise RuntimeError("Cannot return outside of function")
    
    # Visitor methods for expressions
    def visit_number(self, node: NumberExpr) -> float:
        return node.value
    
    def visit_string(self, node: StringExpr) -> str:
        return node.value
    
    def visit_bool(self, node: BoolExpr) -> bool:
        return node.value
    
    def visit_nil(self, node: NilExpr) -> None:
        return None
    
    def visit_identifier(self, node: IdentifierExpr) -> Any:
        return self.current_env.get(node.name)
    
    def visit_array(self, node: ArrayExpr) -> list:
        return [elem.accept(self) for elem in node.elements]
    
    def visit_dict(self, node: DictExpr) -> dict:
        result = {}
        for key, value_expr in node.pairs:
            result[key] = value_expr.accept(self)
        return result
    
    def visit_binary_op(self, node: BinaryOpExpr) -> Any:
        left = node.left.accept(self)
        right = node.right.accept(self)
        op_type = node.op.type
        
        # Arithmetic
        if op_type == TokenType.PLUS:
            if isinstance(left, str) or isinstance(right, str):
                return self._to_string(left) + self._to_string(right)
            elif isinstance(left, list) and isinstance(right, list):
                return left + right
            else:
                # Try Python/native operation first (for PyProxy/numpy/etc.)
                try:
                    result = left + right
                    return result
                except Exception:
                    return self._to_number(left) + self._to_number(right)

        elif op_type == TokenType.MINUS:
            try:
                return left - right
            except Exception:
                return self._to_number(left) - self._to_number(right)
        elif op_type == TokenType.STAR:
            if isinstance(left, str) and isinstance(right, (int, float)):
                return left * int(self._to_number(right))
            elif isinstance(left, list) and isinstance(right, (int, float)):
                return left * int(self._to_number(right))
            else:
                try:
                    return left * right
                except Exception:
                    return self._to_number(left) * self._to_number(right)
        elif op_type == TokenType.SLASH:
            # Try Python division first
            try:
                return left / right
            except Exception:
                right_num = self._to_number(right)
                if right_num == 0:
                    raise RuntimeError("Division by zero")
                return self._to_number(left) / right_num
        elif op_type == TokenType.PERCENT:
            try:
                return left % right
            except Exception:
                return self._to_number(left) % self._to_number(right)
        elif op_type == TokenType.POWER:
            try:
                return left ** right
            except Exception:
                return self._to_number(left) ** self._to_number(right)
        
        # Comparison
        elif op_type == TokenType.EQ:
            return left == right
        elif op_type == TokenType.NE:
            return left != right
        elif op_type == TokenType.LT:
            return self._to_number(left) < self._to_number(right)
        elif op_type == TokenType.LE:
            return self._to_number(left) <= self._to_number(right)
        elif op_type == TokenType.GT:
            return self._to_number(left) > self._to_number(right)
        elif op_type == TokenType.GE:
            return self._to_number(left) >= self._to_number(right)
        
        # Logical
        elif op_type == TokenType.AND:
            return self._is_truthy(left) and self._is_truthy(right)
        elif op_type == TokenType.OR:
            return left if self._is_truthy(left) else right
        
        else:
            raise RuntimeError(f"Unknown binary operator: {op_type}")
    
    def visit_unary_op(self, node: UnaryOpExpr) -> Any:
        operand = node.operand.accept(self)
        op_type = node.op.type
        
        if op_type == TokenType.MINUS:
            return -self._to_number(operand)
        elif op_type == TokenType.NOT:
            return not self._is_truthy(operand)
        else:
            raise RuntimeError(f"Unknown unary operator: {op_type}")
    
    def visit_call(self, node: CallExpr) -> Any:
        func = node.func.accept(self)
        args = [arg.accept(self) for arg in node.args]
        
        if isinstance(func, CodeSutraFunction):
            return self._call_user_function(func, args)
        elif callable(func):
            return func(*args)
        else:
            raise RuntimeError(f"Object is not callable: {type(func)}")
    
    def _call_user_function(self, func: CodeSutraFunction, args: list) -> Any:
        """Call a user-defined function"""
        if len(args) != len(func.params):
            raise RuntimeError(f"Expected {len(func.params)} arguments, got {len(args)}")
        
        # Create new environment for function execution
        func_env = Environment(func.closure)
        for param, arg in zip(func.params, args):
            func_env.define(param, arg)
        
        # Execute function body
        prev_env = self.current_env
        self.current_env = func_env
        try:
            func.body.accept(self)
            return None
        except ReturnException as e:
            return e.value
        finally:
            self.current_env = prev_env
    
    def visit_member(self, node: MemberExpr) -> Any:
        obj = node.object.accept(self)
        # Dictionary property access (CodeSutra native)
        if isinstance(obj, dict):
            if node.property in obj:
                return obj[node.property]
            return None

        # For Python modules/objects and other Python values, try attribute access
        # This enables `import numpy as np; np.array` style usage.
        try:
            return getattr(obj, node.property)
        except Exception:
            # If attribute not present, return None for compatibility with previous behavior
            return None
    
    def visit_index(self, node: IndexExpr) -> Any:
        obj = node.object.accept(self)
        index = node.index.accept(self)
        
        if isinstance(obj, (list, str)):
            idx = int(self._to_number(index))
            if -len(obj) <= idx < len(obj):
                return obj[idx]
            return None
        elif isinstance(obj, dict):
            key = self._to_string(index)
            return obj.get(key)
        else:
            # Try Python-style indexing for other objects (e.g., numpy arrays)
            try:
                return obj[index]
            except Exception:
                raise RuntimeError(f"Cannot index {type(obj)}")
    
    def visit_assign(self, node: AssignExpr) -> Any:
        value = node.value.accept(self)
        
        if node.op.type == TokenType.ASSIGN:
            # Direct assignment
            if isinstance(node.target, IdentifierExpr):
                # Always define in current scope if assigning
                self.current_env.define(node.target.name, value)
            elif isinstance(node.target, IndexExpr):
                obj = node.target.object.accept(self)
                index = node.target.index.accept(self)
                if isinstance(obj, list):
                    idx = int(self._to_number(index))
                    if -len(obj) <= idx < len(obj):
                        obj[idx] = value
                elif isinstance(obj, dict):
                    key = self._to_string(index)
                    obj[key] = value
            elif isinstance(node.target, MemberExpr):
                obj = node.target.object.accept(self)
                if isinstance(obj, dict):
                    obj[node.target.property] = value
        else:
            # Compound assignment
            current = node.target.accept(self)
            if node.op.type == TokenType.PLUS_ASSIGN:
                value = self._to_number(current) + self._to_number(value)
            elif node.op.type == TokenType.MINUS_ASSIGN:
                value = self._to_number(current) - self._to_number(value)
            elif node.op.type == TokenType.STAR_ASSIGN:
                value = self._to_number(current) * self._to_number(value)
            elif node.op.type == TokenType.SLASH_ASSIGN:
                value = self._to_number(current) / self._to_number(value)
            
            if isinstance(node.target, IdentifierExpr):
                self.current_env.set(node.target.name, value)
        
        return value
    
    def visit_func(self, node: FuncExpr) -> CodeSutraFunction:
        return CodeSutraFunction(node.params, node.body, self.current_env)
    
    def visit_ternary(self, node: TernaryExpr) -> Any:
        condition = node.condition.accept(self)
        if self._is_truthy(condition):
            return node.true_expr.accept(self)
        else:
            return node.false_expr.accept(self)
    
    # Visitor methods for statements
    def visit_expr_stmt(self, node: ExprStmt) -> Any:
        return node.expr.accept(self)
    
    def visit_block(self, node: BlockStmt) -> Any:
        result = None
        for stmt in node.statements:
            result = stmt.accept(self)
        return result
    
    def visit_if(self, node: IfStmt) -> Any:
        condition = node.condition.accept(self)
        if self._is_truthy(condition):
            return node.then_branch.accept(self)
        elif node.else_branch:
            return node.else_branch.accept(self)
        return None
    
    def visit_while(self, node: WhileStmt) -> Any:
        result = None
        while self._is_truthy(node.condition.accept(self)):
            try:
                result = node.body.accept(self)
            except BreakException:
                break
            except ContinueException:
                continue
        return result
    
    def visit_for(self, node: ForStmt) -> Any:
        iterable = node.iterable.accept(self)
        result = None
        
        if not isinstance(iterable, (list, str, dict)):
            raise RuntimeError(f"Cannot iterate over {type(iterable)}")
        
        # Create new environment for loop variable
        loop_env = Environment(self.current_env)
        prev_env = self.current_env
        self.current_env = loop_env
        
        try:
            items = iterable if isinstance(iterable, list) else (list(iterable) if isinstance(iterable, str) else list(iterable.keys()))
            for item in items:
                self.current_env.define(node.var, item)
                try:
                    result = node.body.accept(self)
                except BreakException:
                    break
                except ContinueException:
                    continue
        finally:
            self.current_env = prev_env
        
        return result
    
    def visit_func_decl(self, node: FuncDeclStmt) -> Any:
        func = CodeSutraFunction(node.params, node.body, self.current_env)
        self.current_env.define(node.name, func)
        return None
    
    def visit_return(self, node: ReturnStmt) -> Any:
        if node.value:
            value = node.value.accept(self)
        else:
            value = None
        raise ReturnException(value)
    
    def visit_break(self, node: BreakStmt) -> Any:
        raise BreakException()
    
    def visit_continue(self, node: ContinueStmt) -> Any:
        raise ContinueException()
    
    def visit_var_decl(self, node: VarDeclStmt) -> Any:
        if node.value:
            value = node.value.accept(self)
        else:
            value = None
        self.current_env.define(node.name, value)
        return None

    def visit_import(self, node: ImportStmt) -> Any:
        """Handle 'import module [as alias]' statements by loading Python modules"""
        try:
            module = importlib.import_module(node.module)
        except Exception as e:
            raise RuntimeError(f"Failed to import module '{node.module}': {e}")

        name = node.alias if node.alias else node.module.split('.')[-1]
        self.current_env.define(name, wrap(module))
        return None

    def visit_from_import(self, node: FromImportStmt) -> Any:
        """Handle 'from module import name [as alias], ...' statements"""
        try:
            module = importlib.import_module(node.module)
        except Exception as e:
            raise RuntimeError(f"Failed to import module '{node.module}': {e}")

        for name, alias in node.names:
            try:
                obj = getattr(module, name)
            except AttributeError:
                # Try importing submodule (e.g., from pkg import submod)
                try:
                    obj = importlib.import_module(f"{node.module}.{name}")
                except Exception:
                    raise RuntimeError(f"Module '{node.module}' has no attribute or submodule '{name}'")

            varname = alias if alias else name
            self.current_env.define(varname, wrap(obj))

        return None
    
    def visit_program(self, node: Program) -> Any:
        return self.interpret(node)
    
    # Helper methods
    def _to_number(self, value: Any) -> float:
        """Convert value to number"""
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, bool):
            return 1.0 if value else 0.0
        elif isinstance(value, str):
            try:
                if '.' in value:
                    return float(value)
                return float(int(value))
            except:
                return float('nan')
        elif value is None:
            return 0.0
        else:
            raise TypeError(f"Cannot convert {type(value)} to number")
    
    def _to_string(self, value: Any) -> str:
        """Convert value to string"""
        if isinstance(value, str):
            return value
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif value is None:
            return "nil"
        elif isinstance(value, list):
            elements = ', '.join(self._to_string(v) for v in value)
            return f"[{elements}]"
        elif isinstance(value, dict):
            pairs = ', '.join(f"{k}: {self._to_string(v)}" for k, v in value.items())
            return f"{{{pairs}}}"
        else:
            return str(value)
    
    def _is_truthy(self, value: Any) -> bool:
        """Check truthiness of value"""
        if value is None or value is False:
            return False
        if value == 0 or value == "" or value == []:
            return False
        return True
