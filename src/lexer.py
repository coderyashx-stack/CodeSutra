"""
CodeSutra Lexer - Tokenizes source code into a stream of tokens
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Union


class TokenType(Enum):
    """All token types in the CodeSutra language"""
    # Literals
    NUMBER = auto()
    STRING = auto()
    TRUE = auto()
    FALSE = auto()
    NIL = auto()

    # Identifiers and keywords
    IDENTIFIER = auto()
    
    # Keywords
    FUNC = auto()
    RETURN = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    WHILE = auto()
    IN = auto()
    BREAK = auto()
    CONTINUE = auto()
    CLASS = auto()
    IMPORT = auto()
    FROM = auto()
    AS = auto()
    LET = auto()
    CONST = auto()
    AND = auto()
    OR = auto()
    NOT = auto()

    # Operators
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    POWER = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    ASSIGN = auto()
    PLUS_ASSIGN = auto()
    MINUS_ASSIGN = auto()
    STAR_ASSIGN = auto()
    SLASH_ASSIGN = auto()

    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACE = auto()
    RBRACE = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    DOT = auto()
    COLON = auto()
    SEMICOLON = auto()
    ARROW = auto()
    QUESTION = auto()

    # Special
    EOF = auto()
    NEWLINE = auto()


@dataclass
class Token:
    """Represents a token in the source code"""
    type: TokenType
    lexeme: str
    literal: Optional[Union[int, float, str, bool]] = None
    line: int = 1
    column: int = 1

    def __repr__(self):
        if self.literal is not None:
            return f"Token({self.type.name}, '{self.lexeme}', {self.literal}, {self.line}:{self.column})"
        return f"Token({self.type.name}, '{self.lexeme}', {self.line}:{self.column})"


class Lexer:
    """Tokenizes CodeSutra source code"""

    KEYWORDS = {
        'func': TokenType.FUNC,
        'return': TokenType.RETURN,
        'if': TokenType.IF,
        'else': TokenType.ELSE,
        'for': TokenType.FOR,
        'while': TokenType.WHILE,
        'in': TokenType.IN,
        'break': TokenType.BREAK,
        'continue': TokenType.CONTINUE,
        'class': TokenType.CLASS,
        'import': TokenType.IMPORT,
        'from': TokenType.FROM,
        'as': TokenType.AS,
        'let': TokenType.LET,
        'const': TokenType.CONST,
        'and': TokenType.AND,
        'or': TokenType.OR,
        'not': TokenType.NOT,
        'true': TokenType.TRUE,
        'false': TokenType.FALSE,
        'nil': TokenType.NIL,
    }

    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source code"""
        while not self._is_at_end():
            self._skip_whitespace_and_comments()
            if self._is_at_end():
                break

            start_col = self.column
            ch = self._peek()

            if ch == '\n':
                self._advance()
                self.line += 1
                self.column = 1
            elif ch.isdigit():
                self._scan_number()
            elif ch.isalpha() or ch == '_':
                self._scan_identifier()
            elif ch == '"' or ch == "'":
                self._scan_string()
            elif ch == '+':
                if self._peek_next() == '=':
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(TokenType.PLUS_ASSIGN, '+=', None, self.line, start_col))
                else:
                    self._advance()
                    self.tokens.append(Token(TokenType.PLUS, '+', None, self.line, start_col))
            elif ch == '-':
                if self._peek_next() == '=':
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(TokenType.MINUS_ASSIGN, '-=', None, self.line, start_col))
                elif self._peek_next() == '>':
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(TokenType.ARROW, '->', None, self.line, start_col))
                else:
                    self._advance()
                    self.tokens.append(Token(TokenType.MINUS, '-', None, self.line, start_col))
            elif ch == '*':
                if self._peek_next() == '*':
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(TokenType.POWER, '**', None, self.line, start_col))
                elif self._peek_next() == '=':
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(TokenType.STAR_ASSIGN, '*=', None, self.line, start_col))
                else:
                    self._advance()
                    self.tokens.append(Token(TokenType.STAR, '*', None, self.line, start_col))
            elif ch == '/':
                if self._peek_next() == '=':
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(TokenType.SLASH_ASSIGN, '/=', None, self.line, start_col))
                else:
                    self._advance()
                    self.tokens.append(Token(TokenType.SLASH, '/', None, self.line, start_col))
            elif ch == '%':
                self._advance()
                self.tokens.append(Token(TokenType.PERCENT, '%', None, self.line, start_col))
            elif ch == '=':
                if self._peek_next() == '=':
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(TokenType.EQ, '==', None, self.line, start_col))
                else:
                    self._advance()
                    self.tokens.append(Token(TokenType.ASSIGN, '=', None, self.line, start_col))
            elif ch == '!':
                if self._peek_next() == '=':
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(TokenType.NE, '!=', None, self.line, start_col))
                else:
                    self._advance()
                    self.tokens.append(Token(TokenType.NOT, '!', None, self.line, start_col))
            elif ch == '<':
                if self._peek_next() == '=':
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(TokenType.LE, '<=', None, self.line, start_col))
                else:
                    self._advance()
                    self.tokens.append(Token(TokenType.LT, '<', None, self.line, start_col))
            elif ch == '>':
                if self._peek_next() == '=':
                    self._advance()
                    self._advance()
                    self.tokens.append(Token(TokenType.GE, '>=', None, self.line, start_col))
                else:
                    self._advance()
                    self.tokens.append(Token(TokenType.GT, '>', None, self.line, start_col))
            elif ch == '(':
                self._advance()
                self.tokens.append(Token(TokenType.LPAREN, '(', None, self.line, start_col))
            elif ch == ')':
                self._advance()
                self.tokens.append(Token(TokenType.RPAREN, ')', None, self.line, start_col))
            elif ch == '{':
                self._advance()
                self.tokens.append(Token(TokenType.LBRACE, '{', None, self.line, start_col))
            elif ch == '}':
                self._advance()
                self.tokens.append(Token(TokenType.RBRACE, '}', None, self.line, start_col))
            elif ch == '[':
                self._advance()
                self.tokens.append(Token(TokenType.LBRACKET, '[', None, self.line, start_col))
            elif ch == ']':
                self._advance()
                self.tokens.append(Token(TokenType.RBRACKET, ']', None, self.line, start_col))
            elif ch == ',':
                self._advance()
                self.tokens.append(Token(TokenType.COMMA, ',', None, self.line, start_col))
            elif ch == '.':
                self._advance()
                self.tokens.append(Token(TokenType.DOT, '.', None, self.line, start_col))
            elif ch == ':':
                self._advance()
                self.tokens.append(Token(TokenType.COLON, ':', None, self.line, start_col))
            elif ch == ';':
                self._advance()
                self.tokens.append(Token(TokenType.SEMICOLON, ';', None, self.line, start_col))
            elif ch == '?':
                self._advance()
                self.tokens.append(Token(TokenType.QUESTION, '?', None, self.line, start_col))
            else:
                raise SyntaxError(f"Unexpected character '{ch}' at line {self.line}, column {self.column}")

        self.tokens.append(Token(TokenType.EOF, '', None, self.line, self.column))
        return self.tokens

    def _scan_number(self):
        """Scan a numeric literal"""
        start = self.position
        start_col = self.column

        while not self._is_at_end() and self._peek().isdigit():
            self._advance()

        # Handle decimal point
        if not self._is_at_end() and self._peek() == '.' and self._peek_next().isdigit():
            self._advance()
            while not self._is_at_end() and self._peek().isdigit():
                self._advance()
            value = float(self.source[start:self.position])
        else:
            value = int(self.source[start:self.position])

        lexeme = self.source[start:self.position]
        self.tokens.append(Token(TokenType.NUMBER, lexeme, value, self.line, start_col))

    def _scan_string(self):
        """Scan a string literal"""
        quote = self._peek()
        start_col = self.column
        self._advance()
        start = self.position
        value = ''

        while not self._is_at_end() and self._peek() != quote:
            if self._peek() == '\\' and self._peek_next() is not None:
                self._advance()
                next_ch = self._peek()
                if next_ch == 'n':
                    value += '\n'
                elif next_ch == 't':
                    value += '\t'
                elif next_ch == 'r':
                    value += '\r'
                elif next_ch == '\\':
                    value += '\\'
                elif next_ch == quote:
                    value += quote
                else:
                    value += next_ch
                self._advance()
            else:
                if self._peek() == '\n':
                    self.line += 1
                    self.column = 0
                value += self._peek()
                self._advance()

        if self._is_at_end():
            raise SyntaxError(f"Unterminated string at line {self.line}")

        self._advance()  # closing quote
        self.tokens.append(Token(TokenType.STRING, self.source[self.position - len(value) - 2:self.position], value, self.line, start_col))

    def _scan_identifier(self):
        """Scan an identifier or keyword"""
        start = self.position
        start_col = self.column

        while not self._is_at_end() and (self._peek().isalnum() or self._peek() == '_'):
            self._advance()

        lexeme = self.source[start:self.position]
        token_type = self.KEYWORDS.get(lexeme, TokenType.IDENTIFIER)
        
        if token_type in (TokenType.TRUE, TokenType.FALSE):
            literal = token_type == TokenType.TRUE
            self.tokens.append(Token(token_type, lexeme, literal, self.line, start_col))
        elif token_type == TokenType.NIL:
            self.tokens.append(Token(token_type, lexeme, None, self.line, start_col))
        else:
            self.tokens.append(Token(token_type, lexeme, None, self.line, start_col))

    def _skip_whitespace_and_comments(self):
        """Skip whitespace and comments"""
        while not self._is_at_end():
            ch = self._peek()
            if ch in (' ', '\t', '\r'):
                self._advance()
            elif ch == '#':
                # Line comment
                while not self._is_at_end() and self._peek() != '\n':
                    self._advance()
            else:
                break

    def _peek(self) -> Optional[str]:
        """Look at the current character without advancing"""
        if self._is_at_end():
            return None
        return self.source[self.position]

    def _peek_next(self) -> Optional[str]:
        """Look at the next character without advancing"""
        if self.position + 1 >= len(self.source):
            return None
        return self.source[self.position + 1]

    def _advance(self) -> Optional[str]:
        """Consume and return the current character"""
        if self._is_at_end():
            return None
        ch = self.source[self.position]
        self.position += 1
        self.column += 1
        return ch

    def _is_at_end(self) -> bool:
        """Check if we've reached the end of input"""
        return self.position >= len(self.source)
