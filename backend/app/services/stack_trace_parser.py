"""Stack Trace Parser Service for AI Debugging Assistant.

Parses raw stack trace strings from Python, JavaScript, and Java into
structured data. The extracted information — file path, line number,
function name, error type, and error message — is used to:

  1. Pinpoint the exact error line for the AST parser.
  2. Build a clean, structured block for the LLM prompt.
  3. Highlight the error location in the frontend diff viewer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StackFrame:
    """Represents a single frame in a stack trace."""

    filename: str
    line_number: int
    function_name: str
    source_line: Optional[str] = None   # the actual code on that line

    def __str__(self) -> str:
        base = f"{self.filename}:{self.line_number} in {self.function_name}"
        if self.source_line:
            return f"{base} → {self.source_line.strip()}"
        return base


@dataclass
class ParsedStackTrace:
    """
    Fully structured representation of a stack trace.
    The bottom frame is always the direct error location.
    """

    language: str
    error_type: str
    error_message: str
    frames: list[StackFrame] = field(default_factory=list)
    raw: str = ""

    @property
    def _error_frame(self) -> Optional[StackFrame]:
        """
        Returns the frame closest to the actual crash point.
        Python: last frame (most recent call last).
        JavaScript/Java: first frame (top of the at-list).
        """
        if not self.frames:
            return None
        return self.frames[-1] if self.language == "python" else self.frames[0]

    @property
    def error_line(self) -> Optional[int]:
        frame = self._error_frame
        return frame.line_number if frame else None

    @property
    def error_filename(self) -> Optional[str]:
        frame = self._error_frame
        return frame.filename if frame else None

    @property
    def error_function(self) -> Optional[str]:
        frame = self._error_frame
        return frame.function_name if frame else None

    def to_prompt_string(self) -> str:
        """
        Serialises the parsed stack trace into a compact LLM-ready string.
        Only includes the bottom 3 frames — enough context, not noise.
        """
        parts: list[str] = [
            f"[Error type] {self.error_type}",
            f"[Error message] {self.error_message}",
        ]

        if self.error_filename:
            parts.append(f"[Error location] {self.error_filename} line {self.error_line}")

        if self.error_function:
            parts.append(f"[Error function] {self.error_function}")

        # Bottom 3 frames give the call chain without overwhelming the prompt
        if self.language == "python":
            relevant_frames = self.frames[-3:] if len(self.frames) > 3 else self.frames
        else:
            relevant_frames = self.frames[:3] if len(self.frames) > 3 else self.frames
        if relevant_frames:
            frame_lines = [f"  {i+1}. {frame}" for i, frame in enumerate(relevant_frames)]
            parts.append("[Call chain]\n" + "\n".join(frame_lines))

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Language-specific parsers
# ---------------------------------------------------------------------------

class _PythonTraceParser:
    """
    Parses Python tracebacks.

    Handles the standard format:
        Traceback (most recent call last):
          File "path/to/file.py", line 42, in function_name
            source code here
        ErrorType: error message
    """

    # Matches:  File "path/to/file.py", line 42, in some_function
    _FRAME_RE = re.compile(
        r'File "(?P<file>[^"]+)",\s+line\s+(?P<line>\d+),\s+in\s+(?P<func>\S+)'
    )

    # Matches the final error line: ErrorType: message
    _ERROR_RE = re.compile(
        r'^(?P<type>[\w.]+(?:Error|Exception|Warning|Interrupt|Exit|Break'
        r'|Stop|Fault|Miss|Bound|Flow|Overflow|Kill|Signal|Timeout'
        r'|NotFound|Denied|Invalid|Closed|Exists|Busy|Full|Empty'
        r'|Exhausted|Exceeded|Expired|Refused|Reset|Lost|Dropped'
        r'|Failed|Aborted|Cancelled|Skipped|Ignored|Suppressed'
        r'|Raised|Thrown|Caught|Handled|Unhandled|Uncaught|Bare'
        r'|Base|Runtime|System|OS|IO|EOFError|KeyboardInterrupt'
        r'|GeneratorExit|ArithmeticError|LookupError))'
        r':\s*(?P<msg>.+)$',
        re.MULTILINE,
    )

    # Simpler fallback for error line — catches anything like "SomeError: msg"
    _ERROR_FALLBACK_RE = re.compile(
        r'^(?P<type>\w+(?:Error|Exception|Warning|Interrupt))'
        r'(?::\s*(?P<msg>.*))?$',
        re.MULTILINE,
    )

    def parse(self, text: str) -> Optional[ParsedStackTrace]:
        """
        Attempt to parse text as a Python traceback.
        Returns None if the text does not look like a Python traceback.
        """
        if "Traceback (most recent call last)" not in text:
            return None

        result = ParsedStackTrace(
            language="python",
            error_type="UnknownError",
            error_message="",
            raw=text,
        )

        lines = text.splitlines()
        i = 0
        while i < len(lines):
            frame_match = self._FRAME_RE.search(lines[i])
            if frame_match:
                source_line = None
                # The line immediately after the File/line/in line is the source
                if i + 1 < len(lines) and not lines[i + 1].strip().startswith("File"):
                    source_line = lines[i + 1]
                result.frames.append(StackFrame(
                    filename=frame_match.group("file"),
                    line_number=int(frame_match.group("line")),
                    function_name=frame_match.group("func"),
                    source_line=source_line,
                ))
            i += 1

        # Find error type + message — try the last non-empty line first
        error_found = False
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            m = self._ERROR_FALLBACK_RE.match(line)
            if m:
                result.error_type    = m.group("type")
                result.error_message = (m.group("msg") or "").strip()
                error_found = True
                break

        if not error_found:
            result.error_type    = "UnknownError"
            result.error_message = text.strip().splitlines()[-1]

        return result


class _JavaScriptTraceParser:
    """
    Parses JavaScript / Node.js stack traces.

    Handles the standard V8 format:
        TypeError: Cannot read properties of undefined
            at functionName (file.js:42:10)
            at Object.<anonymous> (file.js:10:3)
    """

    # Matches:  at functionName (file.js:42:10)
    #       or: at file.js:42:10   (anonymous)
    _FRAME_RE = re.compile(
        r'at\s+(?:(?P<func>[^\s(]+)\s+\()?(?P<file>[^():]+\.(?:js|ts|jsx|tsx|mjs|cjs))'
        r':(?P<line>\d+):\d+\)?'
    )

    # Matches: TypeError: message   or   Error: message
    _ERROR_RE = re.compile(
        r'^(?P<type>[\w.]*(?:Error|Exception|Warning|TypeError|RangeError'
        r'|ReferenceError|SyntaxError|URIError|EvalError))'
        r':\s*(?P<msg>.+)$',
        re.MULTILINE,
    )

    def parse(self, text: str) -> Optional[ParsedStackTrace]:
        """
        Attempt to parse text as a JavaScript/Node.js stack trace.
        Returns None if no JS-style frames are found.
        """
        if not self._FRAME_RE.search(text):
            return None

        result = ParsedStackTrace(
            language="javascript",
            error_type="Error",
            error_message="",
            raw=text,
        )

        for m in self._FRAME_RE.finditer(text):
            result.frames.append(StackFrame(
                filename=m.group("file"),
                line_number=int(m.group("line")),
                function_name=m.group("func") or "<anonymous>",
            ))

        # Error type + message is always on the first non-empty line
        first_line = next(
            (l.strip() for l in text.splitlines() if l.strip()), ""
        )
        error_match = self._ERROR_RE.match(first_line)
        if error_match:
            result.error_type    = error_match.group("type")
            result.error_message = error_match.group("msg").strip()
        else:
            result.error_type    = "Error"
            result.error_message = first_line

        return result


class _JavaTraceParser:
    """
    Parses Java stack traces.

    Handles the standard JVM format:
        java.lang.NullPointerException: message
            at com.example.Class.method(File.java:42)
    """

    # Matches:  at com.example.ClassName.methodName(FileName.java:42)
    _FRAME_RE = re.compile(
        r'at\s+(?P<class>[\w.$]+)\.(?P<method>\w+)\((?P<file>[\w.]+):(?P<line>\d+)\)'
    )

    # Matches: java.lang.NullPointerException: message
    _ERROR_RE = re.compile(
        r'^(?P<type>[\w.]+(?:Exception|Error|Throwable|Fault))'
        r'(?::\s*(?P<msg>.+))?$',
        re.MULTILINE,
    )

    def parse(self, text: str) -> Optional[ParsedStackTrace]:
        """
        Attempt to parse text as a Java stack trace.
        Returns None if no Java-style frames are found.
        """
        if not self._FRAME_RE.search(text):
            return None

        result = ParsedStackTrace(
            language="java",
            error_type="Exception",
            error_message="",
            raw=text,
        )

        for m in self._FRAME_RE.finditer(text):
            result.frames.append(StackFrame(
                filename=m.group("file"),
                line_number=int(m.group("line")),
                function_name=f"{m.group('class')}.{m.group('method')}",
            ))

        first_line = next(
            (l.strip() for l in text.splitlines() if l.strip()), ""
        )
        error_match = self._ERROR_RE.match(first_line)
        if error_match:
            result.error_type    = error_match.group("type").split(".")[-1]
            result.error_message = (error_match.group("msg") or "").strip()
        else:
            result.error_type    = "Exception"
            result.error_message = first_line

        return result


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class StackTraceParser:
    """
    Public entry point for all stack trace parsing.

    Tries each language parser in order and returns the first successful
    result. Falls back to a minimal ParsedStackTrace when no parser matches
    so the pipeline always has something to work with.

    Usage:
        parser = StackTraceParser()
        result = parser.parse(raw_traceback_string)
        print(result.error_line)          # int — feed this to ASTParserService
        print(result.to_prompt_string())  # str — inject into LLM prompt
    """

    def __init__(self) -> None:
        # Order matters — Python check requires "Traceback" sentinel,
        # JS check requires a .js/.ts file extension, Java is last.
        self._parsers = [
            _PythonTraceParser(),
            _JavaScriptTraceParser(),
            _JavaTraceParser(),
        ]

    def parse(self, raw: str) -> ParsedStackTrace:
        """
        Parse a raw stack trace string into structured data.

        Args:
            raw: The full stack trace text as copied from a terminal or log.

        Returns:
            ParsedStackTrace with language, error type, message, and frames.
            Never raises — returns a minimal object if nothing matches.
        """
        if not raw or not raw.strip():
            return ParsedStackTrace(
                language="unknown",
                error_type="UnknownError",
                error_message="No stack trace provided",
                raw=raw or "",
            )

        for parser in self._parsers:
            result = parser.parse(raw)
            if result is not None and result.frames:
                return result

        # Nothing matched — return what we can from the raw text
        last_line = next(
            (l.strip() for l in reversed(raw.splitlines()) if l.strip()), raw.strip()
        )
        return ParsedStackTrace(
            language="unknown",
            error_type="UnknownError",
            error_message=last_line,
            raw=raw,
        )


__all__ = [
    "StackTraceParser",
    "ParsedStackTrace",
    "StackFrame",
]