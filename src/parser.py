"""
CodeSutra AST (Abstract Syntax Tree) and Parser
"""

from dataclasses import dataclass
from typing import List, Optional, Union, Any
from abc import ABC, abstractmethod
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lexer import Token, TokenType, Lexer


# ============================================================================
# AST Node Classes
# ============================================================================

class ASTNode(ABC):
    """Base class for all AST nodes"""
    
    @abstractmethod
    def accept(self, visitor):
        pass


# Expressions
@dataclass
class NumberExpr(ASTNode):
    value: Union[int, float]
    
    def accept(self, visitor):
        return visitor.visit_number(self)


@dataclass
class StringExpr(ASTNode):
    value: str
    
    def accept(self, visitor):
        return visitor.visit_string(self)


@dataclass
class BoolExpr(ASTNode):
    value: bool
    
    def accept(self, visitor):
        return visitor.visit_bool(self)


@dataclass
class NilExpr(ASTNode):
    def accept(self, visitor):
        return visitor.visit_nil(self)


@dataclass
class IdentifierExpr(ASTNode):
    name: str
    
    def accept(self, visitor):
        return visitor.visit_identifier(self)


@dataclass
class ArrayExpr(ASTNode):
    elements: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_array(self)


@dataclass
class DictExpr(ASTNode):
    pairs: List[tuple]  # List of (key, value) tuples
    
    def accept(self, visitor):
        return visitor.visit_dict(self)


@dataclass
class BinaryOpExpr(ASTNode):
    left: ASTNode
    op: Token
    right: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_binary_op(self)


@dataclass
class UnaryOpExpr(ASTNode):
    op: Token
    operand: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_unary_op(self)


@dataclass
class CallExpr(ASTNode):
    func: ASTNode
    args: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_call(self)


@dataclass
class MemberExpr(ASTNode):
    object: ASTNode
    property: str
    
    def accept(self, visitor):
        return visitor.visit_member(self)


@dataclass
class IndexExpr(ASTNode):
    object: ASTNode
    index: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_index(self)


@dataclass
class AssignExpr(ASTNode):
    target: ASTNode
    value: ASTNode
    op: Token  # For +=, -=, etc.
    
    def accept(self, visitor):
        return visitor.visit_assign(self)


@dataclass
class FuncExpr(ASTNode):
    params: List[str]
    body: 'BlockStmt'
    
    def accept(self, visitor):
        return visitor.visit_func(self)


@dataclass
class TernaryExpr(ASTNode):
    condition: ASTNode
    true_expr: ASTNode
    false_expr: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_ternary(self)


# Statements
@dataclass
class ExprStmt(ASTNode):
    expr: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_expr_stmt(self)


@dataclass
class BlockStmt(ASTNode):
    statements: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_block(self)


@dataclass
class IfStmt(ASTNode):
    condition: ASTNode
    then_branch: ASTNode
    else_branch: Optional[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_if(self)


@dataclass
class WhileStmt(ASTNode):
    condition: ASTNode
    body: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_while(self)


@dataclass
class ForStmt(ASTNode):
    var: str
    iterable: ASTNode
    body: ASTNode
    
    def accept(self, visitor):
        return visitor.visit_for(self)


@dataclass
class FuncDeclStmt(ASTNode):
    name: str
    params: List[str]
    body: BlockStmt
    
    def accept(self, visitor):
        return visitor.visit_func_decl(self)


@dataclass
class ReturnStmt(ASTNode):
    value: Optional[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_return(self)


@dataclass
class BreakStmt(ASTNode):
    def accept(self, visitor):
        return visitor.visit_break(self)


@dataclass
class ContinueStmt(ASTNode):
    def accept(self, visitor):
        return visitor.visit_continue(self)


@dataclass
class VarDeclStmt(ASTNode):
    name: str
    value: Optional[ASTNode]
    is_const: bool = False
    
    def accept(self, visitor):
        return visitor.visit_var_decl(self)


@dataclass
class ImportStmt(ASTNode):
    module: str
    alias: Optional[str] = None

    def accept(self, visitor):
        return visitor.visit_import(self)


@dataclass
class FromImportStmt(ASTNode):
    module: str
    names: List[tuple]  # List of (name, alias) tuples

    def accept(self, visitor):
        return visitor.visit_from_import(self)


@dataclass
class Program(ASTNode):
    statements: List[ASTNode]
    
    def accept(self, visitor):
        return visitor.visit_program(self)


# ============================================================================
# Parser
# ============================================================================

class Parser:
    """Recursive descent parser for CodeSutra"""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
    
    def parse(self) -> Program:
        """Parse the entire program"""
        statements = []
        while not self._is_at_end():
            stmt = self._statement()
            if stmt:
                statements.append(stmt)
        return Program(statements)
    
    def _statement(self) -> Optional[ASTNode]:
        """Parse a statement"""
        if self._match(TokenType.IMPORT):
            return self._import_stmt()
        if self._match(TokenType.FROM):
            return self._from_import_stmt()
        if self._match(TokenType.FUNC):
            return self._func_decl()
        if self._match(TokenType.IF):
            return self._if_stmt()
        if self._match(TokenType.WHILE):
            return self._while_stmt()
        if self._match(TokenType.FOR):
            return self._for_stmt()
        if self._match(TokenType.RETURN):
            return self._return_stmt()
        if self._match(TokenType.BREAK):
            self._consume_semicolon()
            return BreakStmt()
        if self._match(TokenType.CONTINUE):
            self._consume_semicolon()
            return ContinueStmt()
        if self._match(TokenType.LBRACE):
            return self._block()
        if self._match(TokenType.LET, TokenType.CONST):
            is_const = self.tokens[self.position - 1].type == TokenType.CONST
            return self._var_decl(is_const)
        return self._expr_stmt()
    
    def _import_stmt(self) -> ImportStmt:
        """Parse an import statement: import module [as alias]"""
        module = self._consume(TokenType.IDENTIFIER, "Expected module name").lexeme

        # Handle dotted names like 'numpy.linalg'
        while self._match(TokenType.DOT):
            module += "." + self._consume(TokenType.IDENTIFIER, "Expected identifier after '.'").lexeme

        alias = None
        if self._match(TokenType.AS):
            alias = self._consume(TokenType.IDENTIFIER, "Expected alias name").lexeme

        self._consume_semicolon()
        return ImportStmt(module, alias)

    def _from_import_stmt(self) -> FromImportStmt:
        """Parse a from-import statement: from module import name [as alias], ..."""
        module = self._consume(TokenType.IDENTIFIER, "Expected module name").lexeme

        # Handle dotted names like 'numpy.linalg'
        while self._match(TokenType.DOT):
            module += "." + self._consume(TokenType.IDENTIFIER, "Expected identifier after '.'").lexeme

        self._consume(TokenType.IMPORT, "Expected 'import' after module name")

        names = []
        # Parse 'name [as alias], name2 [as alias2], ...'
        name = self._consume(TokenType.IDENTIFIER, "Expected name to import").lexeme
        alias = None
        if self._match(TokenType.AS):
            alias = self._consume(TokenType.IDENTIFIER, "Expected alias name").lexeme
        names.append((name, alias))

        while self._match(TokenType.COMMA):
            name = self._consume(TokenType.IDENTIFIER, "Expected name to import").lexeme
            alias = None
            if self._match(TokenType.AS):
                alias = self._consume(TokenType.IDENTIFIER, "Expected alias name").lexeme
            names.append((name, alias))

        self._consume_semicolon()
        return FromImportStmt(module, names)
    
    def _func_decl(self) -> FuncDeclStmt:
        """Parse a function declaration"""
        name = self._consume(TokenType.IDENTIFIER, "Expected function name").lexeme
        self._consume(TokenType.LPAREN, "Expected '(' after function name")
        
        params = []
        if not self._check(TokenType.RPAREN):
            params.append(self._consume(TokenType.IDENTIFIER, "Expected parameter name").lexeme)
            while self._match(TokenType.COMMA):
                params.append(self._consume(TokenType.IDENTIFIER, "Expected parameter name").lexeme)
        
        self._consume(TokenType.RPAREN, "Expected ')' after parameters")
        self._consume(TokenType.LBRACE, "Expected '{' before function body")
        body = self._block()
        
        return FuncDeclStmt(name, params, body)
    
    def _if_stmt(self) -> IfStmt:
        """Parse an if statement"""
        self._consume(TokenType.LPAREN, "Expected '(' after 'if'")
        condition = self._expression()
        self._consume(TokenType.RPAREN, "Expected ')' after if condition")
        
        then_branch = self._statement()
        else_branch = None
        
        if self._match(TokenType.ELSE):
            else_branch = self._statement()
        
        return IfStmt(condition, then_branch, else_branch)
    
    def _while_stmt(self) -> WhileStmt:
        """Parse a while loop"""
        self._consume(TokenType.LPAREN, "Expected '(' after 'while'")
        condition = self._expression()
        self._consume(TokenType.RPAREN, "Expected ')' after while condition")
        
        body = self._statement()
        return WhileStmt(condition, body)
    
    def _for_stmt(self) -> ForStmt:
        """Parse a for loop"""
        self._consume(TokenType.LPAREN, "Expected '(' after 'for'")
        var = self._consume(TokenType.IDENTIFIER, "Expected variable name").lexeme
        self._consume(TokenType.IN, "Expected 'in' in for loop")
        iterable = self._expression()
        self._consume(TokenType.RPAREN, "Expected ')' after for clause")
        
        body = self._statement()
        return ForStmt(var, iterable, body)
    
    def _return_stmt(self) -> ReturnStmt:
        """Parse a return statement"""
        value = None
        if not self._check(TokenType.SEMICOLON) and not self._check(TokenType.RBRACE) and not self._is_at_end():
            value = self._expression()
        self._consume_semicolon()
        return ReturnStmt(value)
    
    def _var_decl(self, is_const: bool = False) -> VarDeclStmt:
        """Parse a variable declaration"""
        name = self._consume(TokenType.IDENTIFIER, "Expected variable name").lexeme
        value = None
        
        if self._match(TokenType.ASSIGN):
            value = self._expression()
        
        self._consume_semicolon()
        return VarDeclStmt(name, value, is_const)
    
    def _block(self) -> BlockStmt:
        """Parse a block of statements"""
        statements = []
        
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            stmt = self._statement()
            if stmt:
                statements.append(stmt)
        
        self._consume(TokenType.RBRACE, "Expected '}' after block")
        return BlockStmt(statements)
    
    def _expr_stmt(self) -> ExprStmt:
        """Parse an expression statement"""
        expr = self._expression()
        self._consume_semicolon()
        return ExprStmt(expr)
    
    def _expression(self) -> ASTNode:
        """Parse an expression"""
        return self._assignment()
    
    def _assignment(self) -> ASTNode:
        """Parse an assignment or ternary expression"""
        expr = self._ternary()
        
        if self._match(TokenType.ASSIGN):
            op = self.tokens[self.position - 1]
            value = self._assignment()
            return AssignExpr(expr, value, op)
        elif self._match(TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN, 
                         TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN):
            op = self.tokens[self.position - 1]
            value = self._assignment()
            return AssignExpr(expr, value, op)
        
        return expr
    
    def _ternary(self) -> ASTNode:
        """Parse a ternary expression"""
        expr = self._logical_or()
        
        if self._match(TokenType.QUESTION):
            true_expr = self._expression()
            self._consume(TokenType.COLON, "Expected ':' in ternary expression")
            false_expr = self._ternary()
            return TernaryExpr(expr, true_expr, false_expr)
        
        return expr
    
    def _logical_or(self) -> ASTNode:
        """Parse a logical OR expression"""
        expr = self._logical_and()
        
        while self._match(TokenType.OR):
            op = self.tokens[self.position - 1]
            right = self._logical_and()
            expr = BinaryOpExpr(expr, op, right)
        
        return expr
    
    def _logical_and(self) -> ASTNode:
        """Parse a logical AND expression"""
        expr = self._equality()
        
        while self._match(TokenType.AND):
            op = self.tokens[self.position - 1]
            right = self._equality()
            expr = BinaryOpExpr(expr, op, right)
        
        return expr
    
    def _equality(self) -> ASTNode:
        """Parse equality expressions"""
        expr = self._comparison()
        
        while self._match(TokenType.EQ, TokenType.NE):
            op = self.tokens[self.position - 1]
            right = self._comparison()
            expr = BinaryOpExpr(expr, op, right)
        
        return expr
    
    def _comparison(self) -> ASTNode:
        """Parse comparison expressions"""
        expr = self._additive()
        
        while self._match(TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE):
            op = self.tokens[self.position - 1]
            right = self._additive()
            expr = BinaryOpExpr(expr, op, right)
        
        return expr
    
    def _additive(self) -> ASTNode:
        """Parse additive expressions"""
        expr = self._multiplicative()
        
        while self._match(TokenType.PLUS, TokenType.MINUS):
            op = self.tokens[self.position - 1]
            right = self._multiplicative()
            expr = BinaryOpExpr(expr, op, right)
        
        return expr
    
    def _multiplicative(self) -> ASTNode:
        """Parse multiplicative expressions"""
        expr = self._power()
        
        while self._match(TokenType.STAR, TokenType.SLASH, TokenType.PERCENT):
            op = self.tokens[self.position - 1]
            right = self._power()
            expr = BinaryOpExpr(expr, op, right)
        
        return expr
    
    def _power(self) -> ASTNode:
        """Parse power expressions"""
        expr = self._unary()
        
        if self._match(TokenType.POWER):
            op = self.tokens[self.position - 1]
            right = self._power()  # Right associative
            expr = BinaryOpExpr(expr, op, right)
        
        return expr
    
    def _unary(self) -> ASTNode:
        """Parse unary expressions"""
        if self._match(TokenType.NOT, TokenType.MINUS):
            op = self.tokens[self.position - 1]
            expr = self._unary()
            return UnaryOpExpr(op, expr)
        
        return self._postfix()
    
    def _postfix(self) -> ASTNode:
        """Parse postfix expressions"""
        expr = self._primary()
        
        while True:
            if self._match(TokenType.LPAREN):
                args = self._args()
                self._consume(TokenType.RPAREN, "Expected ')' after arguments")
                expr = CallExpr(expr, args)
            elif self._match(TokenType.LBRACKET):
                index = self._expression()
                self._consume(TokenType.RBRACKET, "Expected ']' after index")
                expr = IndexExpr(expr, index)
            elif self._match(TokenType.DOT):
                prop = self._consume(TokenType.IDENTIFIER, "Expected property name").lexeme
                expr = MemberExpr(expr, prop)
            else:
                break
        
        return expr
    
    def _args(self) -> List[ASTNode]:
        """Parse function arguments"""
        args = []
        if not self._check(TokenType.RPAREN):
            args.append(self._expression())
            while self._match(TokenType.COMMA):
                args.append(self._expression())
        return args
    
    def _primary(self) -> ASTNode:
        """Parse primary expressions"""
        if self._match(TokenType.NUMBER):
            return NumberExpr(self.tokens[self.position - 1].literal)
        
        if self._match(TokenType.STRING):
            return StringExpr(self.tokens[self.position - 1].literal)
        
        if self._match(TokenType.TRUE):
            return BoolExpr(True)
        
        if self._match(TokenType.FALSE):
            return BoolExpr(False)
        
        if self._match(TokenType.NIL):
            return NilExpr()
        
        if self._match(TokenType.IDENTIFIER):
            return IdentifierExpr(self.tokens[self.position - 1].lexeme)
        
        if self._match(TokenType.LPAREN):
            expr = self._expression()
            self._consume(TokenType.RPAREN, "Expected ')' after expression")
            return expr
        
        if self._match(TokenType.LBRACKET):
            elements = []
            if not self._check(TokenType.RBRACKET):
                elements.append(self._expression())
                while self._match(TokenType.COMMA):
                    elements.append(self._expression())
            self._consume(TokenType.RBRACKET, "Expected ']' after array elements")
            return ArrayExpr(elements)
        
        if self._match(TokenType.LBRACE):
            pairs = []
            if not self._check(TokenType.RBRACE):
                key = self._consume(TokenType.IDENTIFIER, "Expected key in dict").lexeme
                self._consume(TokenType.COLON, "Expected ':' after key")
                value = self._expression()
                pairs.append((key, value))
                while self._match(TokenType.COMMA):
                    key = self._consume(TokenType.IDENTIFIER, "Expected key in dict").lexeme
                    self._consume(TokenType.COLON, "Expected ':' after key")
                    value = self._expression()
                    pairs.append((key, value))
            self._consume(TokenType.RBRACE, "Expected '}' after dict")
            return DictExpr(pairs)
        
        if self._match(TokenType.FUNC):
            self._consume(TokenType.LPAREN, "Expected '(' after 'func'")
            params = []
            if not self._check(TokenType.RPAREN):
                params.append(self._consume(TokenType.IDENTIFIER, "Expected parameter name").lexeme)
                while self._match(TokenType.COMMA):
                    params.append(self._consume(TokenType.IDENTIFIER, "Expected parameter name").lexeme)
            self._consume(TokenType.RPAREN, "Expected ')' after parameters")
            self._consume(TokenType.LBRACE, "Expected '{' before function body")
            body = self._block()
            return FuncExpr(params, body)
        
        raise SyntaxError(f"Unexpected token: {self._peek()}")
    
    def _match(self, *types: TokenType) -> bool:
        """Check if current token matches any of the given types"""
        for token_type in types:
            if self._check(token_type):
                self._advance()
                return True
        return False
    
    def _check(self, token_type: TokenType) -> bool:
        """Check if current token is of given type"""
        if self._is_at_end():
            return False
        return self._peek().type == token_type
    
    def _advance(self) -> Token:
        """Consume and return current token"""
        if not self._is_at_end():
            self.position += 1
        return self.tokens[self.position - 1]
    
    def _is_at_end(self) -> bool:
        """Check if we're at the end of tokens"""
        return self._peek().type == TokenType.EOF
    
    def _peek(self) -> Token:
        """Return current token without advancing"""
        return self.tokens[self.position]
    
    def _consume(self, token_type: TokenType, message: str) -> Token:
        """Consume a token of given type or raise error"""
        if self._check(token_type):
            return self._advance()
        raise SyntaxError(f"{message}. Got {self._peek()}")
    
    def _consume_semicolon(self):
        """Consume optional semicolon"""
        self._match(TokenType.SEMICOLON)
