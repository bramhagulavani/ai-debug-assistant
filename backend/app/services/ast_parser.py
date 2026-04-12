"""AST Parser Service for AI Debugging Assistant.

Parses Python and JavaScript source code to extract semantic context —
function names, parameters, imports, class names, and the code window
around an error line. This context is injected into the LLM prompt so
the model reasons about the user's specific code, not generic patterns.

Tree-sitter is used when available. If the grammar packages are missing,
the service automatically falls back to a regex-based parser so the
pipeline never breaks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from tree_sitter import Language, Parser
    import tree_sitter_python as tspython
    import tree_sitter_javascript as tsjavascript
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter packages not found — using regex fallback parser")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_IMPORTS = 10
MAX_FUNCTIONS = 8
ERROR_WINDOW_RADIUS = 10  # lines above and below the error line


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class FunctionInfo:
    """Represents a single function extracted from source code."""

    name: str
    start_line: int
    end_line: int
    parameters: list[str] = field(default_factory=list)

    def signature(self) -> str:
        """Returns a human-readable function signature string."""
        params = ", ".join(self.parameters)
        return f"{self.name}({params})"


@dataclass
class ParsedContext:
    """
    All semantic context extracted from a source file.
    Call to_prompt_string() to get the LLM-ready representation.
    """

    language: str
    imports: list[str] = field(default_factory=list)
    functions: list[FunctionInfo] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    error_function: Optional[FunctionInfo] = None
    error_window: list[str] = field(default_factory=list)

    def to_prompt_string(self) -> str:
        """
        Serialises context into a compact string for LLM prompt injection.
        Keeps it short — the model only needs structure, not full source.
        """
        parts: list[str] = [f"[Language] {self.language}"]

        if self.imports:
            parts.append(f"[Imports] {', '.join(self.imports[:MAX_IMPORTS])}")

        if self.class_names:
            parts.append(f"[Classes] {', '.join(self.class_names)}")

        if self.functions:
            sigs = [f.signature() for f in self.functions[:MAX_FUNCTIONS]]
            parts.append(f"[Functions] {', '.join(sigs)}")

        if self.error_function:
            ef = self.error_function
            parts.append(
                f"[Error in function] {ef.signature()} "
                f"lines {ef.start_line}–{ef.end_line}"
            )

        if self.error_window:
            window = "\n".join(self.error_window)
            parts.append(f"[Code window around error]\n{window}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, str] = {
    ".py":  "python",
    ".js":  "javascript",
    ".jsx": "javascript",
    ".ts":  "javascript",
    ".tsx": "javascript",
}

_SHEBANG_MAP: dict[str, str] = {
    "python": "python",
    "node":   "javascript",
}


def detect_language(code: str, filename: Optional[str] = None) -> str:
    """
    Detects the programming language of a source snippet.

    Priority order:
      1. File extension (most reliable)
      2. Shebang line
      3. Keyword heuristic
      4. Fallback: 'unknown'

    Args:
        code:     The source code string.
        filename: Optional filename — used for extension matching.

    Returns:
        'python', 'javascript', or 'unknown'
    """
    if filename:
        for ext, lang in _EXTENSION_MAP.items():
            if filename.endswith(ext):
                return lang

    lines = code.strip().splitlines()
    if lines and lines[0].startswith("#!"):
        for key, lang in _SHEBANG_MAP.items():
            if key in lines[0]:
                return lang

    # Keyword heuristic — count language-specific tokens
    py_score = sum(1 for kw in ["def ", "elif ", "print(", "import ", "__name__"]
                   if kw in code)
    js_score = sum(1 for kw in ["function ", "const ", "=>", "require(", "==="]
                   if kw in code)

    if py_score > js_score:
        return "python"
    if js_score > py_score:
        return "javascript"
    return "unknown"


# ---------------------------------------------------------------------------
# Shared helper — error window builder
# ---------------------------------------------------------------------------

def _build_error_window(lines: list[str], error_line: int) -> list[str]:
    """
    Returns a numbered slice of source lines centred on error_line.

    Args:
        lines:      All source lines (0-indexed list).
        error_line: 1-based line number of the error.

    Returns:
        List of strings in the format '  42 | <source line>'
    """
    lo = max(0, error_line - ERROR_WINDOW_RADIUS - 1)
    hi = min(len(lines), error_line + ERROR_WINDOW_RADIUS)
    return [
        f"{lo + i + 1:>4} | {line}"
        for i, line in enumerate(lines[lo:hi])
    ]


def _find_enclosing_function(
    functions: list[FunctionInfo], error_line: int
) -> Optional[FunctionInfo]:
    """
    Returns the innermost function whose range contains error_line.
    Works correctly even when end_line is unknown (regex fallback):
    picks the function whose start_line is closest to but not past error_line.

    Args:
        functions:  All functions extracted from the file.
        error_line: 1-based line number of the error.

    Returns:
        The best matching FunctionInfo, or None.
    """
    # First try: exact range match (tree-sitter knows end lines)
    exact = [
        f for f in functions
        if f.start_line <= error_line <= f.end_line
    ]
    if exact:
        # Prefer the innermost (highest start_line)
        return max(exact, key=lambda f: f.start_line)

    # Fallback: closest function whose start is at or before error_line
    # Used when end_line is unreliable (regex parser)
    candidates = [f for f in functions if f.start_line <= error_line]
    if candidates:
        return max(candidates, key=lambda f: f.start_line)

    return None


# ---------------------------------------------------------------------------
# Regex-based fallback parser
# ---------------------------------------------------------------------------

class _RegexParser:
    """
    Lightweight fallback parser using regular expressions.
    Shallower than tree-sitter but requires no compiled grammar packages.
    End lines are unknown so error_function matching uses start-line proximity.
    """

    _PY_IMPORT = re.compile(r"^\s*(?:import|from)\s+([\w.]+)", re.MULTILINE)
    _PY_CLASS  = re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)
    _PY_DEF    = re.compile(
        r"^[ \t]*def\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE
    )

    _JS_IMPORT = re.compile(
        r"""(?:import\s+.*?from\s+['"](.+?)['"]|require\s*\(\s*['"](.+?)['"]\s*\))""",
        re.MULTILINE,
    )
    _JS_CLASS  = re.compile(r"^\s*class\s+(\w+)", re.MULTILINE)
    # Covers: function foo(), const foo = () =>, const foo = async () =>
    _JS_FUNC   = re.compile(
        r"""(?:function\s+(\w+)\s*\(([^)]*)\)|"""
        r"""(?:const|let|var)\s+(\w+)\s*=\s*async\s*\(([^)]*)\)\s*=>|"""
        r"""(?:const|let|var)\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>)""",
        re.MULTILINE,
    )

    def parse(
        self, code: str, language: str, error_line: Optional[int]
    ) -> ParsedContext:
        """Parse source code with regex and return a ParsedContext."""
        ctx = ParsedContext(language=language)
        lines = code.splitlines()

        if language == "python":
            ctx.imports = [m.group(1) for m in self._PY_IMPORT.finditer(code)]
            ctx.class_names = [m.group(1) for m in self._PY_CLASS.finditer(code)]
            for m in self._PY_DEF.finditer(code):
                params = [
                    p.strip().split(":")[0].split("=")[0].strip()
                    for p in m.group(2).split(",")
                    if p.strip()
                ]
                start = code[: m.start()].count("\n") + 1
                ctx.functions.append(
                    FunctionInfo(name=m.group(1), start_line=start,
                                 end_line=start, parameters=params)
                )

        elif language == "javascript":
            ctx.imports = [
                m.group(1) or m.group(2)
                for m in self._JS_IMPORT.finditer(code)
            ]
            ctx.class_names = [m.group(1) for m in self._JS_CLASS.finditer(code)]
            for m in self._JS_FUNC.finditer(code):
                # Groups: (1,2) named fn | (3,4) async arrow | (5,6) regular arrow
                name = m.group(1) or m.group(3) or m.group(5) or "<anonymous>"
                raw  = m.group(2) or m.group(4) or m.group(6) or ""
                params = [p.strip() for p in raw.split(",") if p.strip()]
                start = code[: m.start()].count("\n") + 1
                ctx.functions.append(
                    FunctionInfo(name=name, start_line=start,
                                 end_line=start, parameters=params)
                )

        if error_line is not None:
            ctx.error_window = _build_error_window(lines, error_line)
            ctx.error_function = _find_enclosing_function(ctx.functions, error_line)

        return ctx


# ---------------------------------------------------------------------------
# Tree-sitter full AST parser
# ---------------------------------------------------------------------------

class _TreeSitterParser:
    """
    Full AST parser backed by tree-sitter grammar packages.
    Provides precise line ranges, correct parameter extraction,
    and deep nesting support.
    """

    def __init__(self) -> None:
        if not TREE_SITTER_AVAILABLE:
            raise RuntimeError("tree-sitter packages are not installed")

        # FIX: Parser(language) constructor — correct API for v0.21+
        py_lang = Language(tspython.language())
        js_lang = Language(tsjavascript.language())

        self._parsers: dict[str, Parser] = {
            "python":     Parser(py_lang),
            "javascript": Parser(js_lang),
        }

    @staticmethod
    def _text(node, src: bytes) -> str:
        """Extract raw UTF-8 text for a tree-sitter node."""
        return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    # --- Python ------------------------------------------------------------

    def _parse_python(
        self, tree, src: bytes, error_line: Optional[int]
    ) -> ParsedContext:
        ctx = ParsedContext(language="python")
        root = tree.root_node

        for node in root.children:
            if node.type in ("import_statement", "import_from_statement"):
                ctx.imports.append(self._text(node, src).strip())
            if node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    ctx.class_names.append(self._text(name_node, src))

        def walk(node):
            if node.type == "function_definition":
                name_node   = node.child_by_field_name("name")
                params_node = node.child_by_field_name("parameters")
                name = self._text(name_node, src) if name_node else "?"
                params: list[str] = []
                if params_node:
                    for child in params_node.children:
                        if child.type in (
                            "identifier", "typed_parameter", "default_parameter"
                        ):
                            raw = self._text(child, src)
                            params.append(
                                raw.split(":")[0].split("=")[0].strip()
                            )
                ctx.functions.append(FunctionInfo(
                    name=name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parameters=params,
                ))
            for child in node.children:
                walk(child)

        walk(root)

        if error_line is not None:
            lines = src.decode("utf-8", errors="replace").splitlines()
            ctx.error_window   = _build_error_window(lines, error_line)
            ctx.error_function = _find_enclosing_function(ctx.functions, error_line)

        return ctx

    # --- JavaScript --------------------------------------------------------

    def _parse_javascript(
        self, tree, src: bytes, error_line: Optional[int]
    ) -> ParsedContext:
        ctx = ParsedContext(language="javascript")
        root = tree.root_node

        def walk(node):
            if node.type == "import_statement":
                ctx.imports.append(self._text(node, src).strip())

            if node.type == "call_expression":
                fn_node = node.child_by_field_name("function")
                if fn_node and self._text(fn_node, src) == "require":
                    ctx.imports.append(self._text(node, src).strip())

            if node.type == "class_declaration":
                name_node = node.child_by_field_name("name")
                if name_node:
                    ctx.class_names.append(self._text(name_node, src))

            if node.type in (
                "function_declaration", "function_expression",
                "arrow_function", "method_definition"
            ):
                name_node   = node.child_by_field_name("name")
                params_node = node.child_by_field_name("parameters")
                name = self._text(name_node, src) if name_node else "<anonymous>"
                params: list[str] = []
                if params_node:
                    for child in params_node.children:
                        if child.type == "identifier":
                            params.append(self._text(child, src))
                ctx.functions.append(FunctionInfo(
                    name=name,
                    start_line=node.start_point[0] + 1,
                    end_line=node.end_point[0] + 1,
                    parameters=params,
                ))

            for child in node.children:
                walk(child)

        walk(root)

        if error_line is not None:
            lines = src.decode("utf-8", errors="replace").splitlines()
            ctx.error_window   = _build_error_window(lines, error_line)
            ctx.error_function = _find_enclosing_function(ctx.functions, error_line)

        return ctx

    # --- public ------------------------------------------------------------

    def parse(
        self, code: str, language: str, error_line: Optional[int]
    ) -> ParsedContext:
        """
        Parse source code via tree-sitter and return rich semantic context.

        Args:
            code:       Full source file or relevant snippet.
            language:   'python' or 'javascript'.
            error_line: 1-based line number where the error occurred.

        Returns:
            ParsedContext with all extracted semantic information.
        """
        parser = self._parsers.get(language)
        if not parser:
            raise ValueError(f"Unsupported language for tree-sitter: {language}")

        src  = code.encode("utf-8")
        tree = parser.parse(src)

        if language == "python":
            return self._parse_python(tree, src, error_line)
        return self._parse_javascript(tree, src, error_line)


# ---------------------------------------------------------------------------
# Public facade — always import this, never the internals
# ---------------------------------------------------------------------------

class ASTParserService:
    """
    Public entry point for all AST parsing in the application.

    Automatically selects tree-sitter when available and logs a warning
    when falling back to the regex parser. The rest of the pipeline
    never needs to know which backend is active.

    Usage:
        parser = ASTParserService()
        ctx    = parser.parse(code=src, error_line=42, filename="app.py")
        print(ctx.to_prompt_string())
    """

    def __init__(self) -> None:
        if TREE_SITTER_AVAILABLE:
            try:
                self._backend         = _TreeSitterParser()
                self._using_treesitter = True
            except Exception as e:
                # FIX: log the reason instead of silently swallowing it
                logger.warning(
                    "tree-sitter init failed (%s) — falling back to regex parser", e
                )
                self._backend         = _RegexParser()
                self._using_treesitter = False
        else:
            self._backend         = _RegexParser()
            self._using_treesitter = False

    @property
    def backend_name(self) -> str:
        """Returns which parser backend is active — useful for debugging."""
        return "tree-sitter" if self._using_treesitter else "regex-fallback"

    def parse(
        self,
        code: str,
        error_line: Optional[int] = None,
        filename: Optional[str] = None,
        language: Optional[str] = None,
    ) -> ParsedContext:
        """
        Parse source code and return structured semantic context.

        Args:
            code:       The source code to analyse.
            error_line: 1-based line number of the error (optional).
            filename:   Helps language detection via file extension.
            language:   Explicit language override ('python'/'javascript').

        Returns:
            ParsedContext — call .to_prompt_string() for the LLM-ready string.
        """
        if not code or not code.strip():
            return ParsedContext(language="unknown")

        lang = language or detect_language(code, filename)

        if lang == "unknown":
            ctx = ParsedContext(language="unknown")
            if error_line:
                ctx.error_window = _build_error_window(code.splitlines(), error_line)
            return ctx

        # Defensive: unsupported language treated as python rather than crashing
        # intentional default — see detect_language for supported values
        if lang not in ("python", "javascript"):
            lang = "python"

        return self._backend.parse(code, lang, error_line)


__all__ = [
    "ASTParserService",
    "ParsedContext",
    "FunctionInfo",
    "detect_language",
]