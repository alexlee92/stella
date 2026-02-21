"""
P3.2 — Code intelligence: go-to-definition, find-references, signatures.

Uses jedi for Python analysis. Provides a unified interface that can be
extended for other languages later.
"""

import os
from typing import List, Optional

from agent.config import PROJECT_ROOT


def _abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _rel(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def goto_definition(path: str, symbol: str) -> List[dict]:
    """Find where *symbol* is defined in the file at *path*.

    Returns a list of {file, line, column, name, type} dicts.
    """
    try:
        import jedi
    except ImportError:
        return [{"error": "jedi not installed — run: pip install jedi"}]

    abs_path = _abs(path)
    if not os.path.isfile(abs_path):
        return [{"error": f"file not found: {path}"}]

    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except OSError as e:
        return [{"error": str(e)}]

    # Find the symbol position in source
    line, col = _find_symbol_position(source, symbol)
    if line is None:
        return [{"error": f"symbol '{symbol}' not found in {path}"}]

    try:
        script = jedi.Script(source, path=abs_path)
        defs = script.goto(line, col)
        results = []
        for d in defs:
            results.append({
                "file": _rel(d.module_path) if d.module_path else "builtin",
                "line": d.line,
                "column": d.column,
                "name": d.name,
                "type": d.type,
            })
        return results if results else [{"error": f"no definition found for '{symbol}'"}]
    except Exception as e:
        return [{"error": f"jedi error: {e}"}]


def find_references(path: str, symbol: str) -> List[dict]:
    """Find all references to *symbol* starting from *path*.

    Returns a list of {file, line, column, name, context} dicts.
    """
    try:
        import jedi
    except ImportError:
        return [{"error": "jedi not installed — run: pip install jedi"}]

    abs_path = _abs(path)
    if not os.path.isfile(abs_path):
        return [{"error": f"file not found: {path}"}]

    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except OSError as e:
        return [{"error": str(e)}]

    line, col = _find_symbol_position(source, symbol)
    if line is None:
        return [{"error": f"symbol '{symbol}' not found in {path}"}]

    try:
        script = jedi.Script(source, path=abs_path)
        refs = script.get_references(line, col)
        results = []
        for r in refs:
            ctx = ""
            if r.module_path and os.path.isfile(str(r.module_path)):
                try:
                    lines = open(str(r.module_path), "r", encoding="utf-8", errors="ignore").readlines()
                    if r.line and r.line <= len(lines):
                        ctx = lines[r.line - 1].strip()[:120]
                except OSError:
                    pass
            results.append({
                "file": _rel(r.module_path) if r.module_path else "?",
                "line": r.line,
                "column": r.column,
                "name": r.name,
                "context": ctx,
            })
        return results if results else [{"error": f"no references found for '{symbol}'"}]
    except Exception as e:
        return [{"error": f"jedi error: {e}"}]


def get_signature(path: str, symbol: str) -> Optional[str]:
    """Get the signature of a function/class defined in *path*.

    Returns a human-readable signature string, or None.
    """
    try:
        import jedi
    except ImportError:
        return None

    abs_path = _abs(path)
    if not os.path.isfile(abs_path):
        return None

    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
    except OSError:
        return None

    line, col = _find_symbol_position(source, symbol)
    if line is None:
        return None

    try:
        script = jedi.Script(source, path=abs_path)
        # Move to opening paren position to trigger signature help
        sigs = script.get_signatures(line, col + len(symbol) + 1)
        if sigs:
            return str(sigs[0])
        # Fallback: try goto to read the definition line
        defs = script.goto(line, col)
        for d in defs:
            desc = d.description
            if desc:
                return desc
    except Exception:
        pass
    return None


def list_symbols(path: str) -> List[dict]:
    """List all top-level symbols (classes, functions, variables) in a file.

    Returns a list of {name, type, line} dicts.
    """
    try:
        import jedi
    except ImportError:
        return []

    abs_path = _abs(path)
    if not os.path.isfile(abs_path):
        return []

    try:
        with open(abs_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        script = jedi.Script(source, path=abs_path)
        names = script.get_names(all_scopes=False)
        return [
            {"name": n.name, "type": n.type, "line": n.line}
            for n in names
            if n.type in ("class", "function", "statement")
        ]
    except Exception:
        return []


def _find_symbol_position(source: str, symbol: str) -> tuple:
    """Find the first occurrence of *symbol* as an identifier in source.

    Returns (line_1based, column_0based) or (None, None).
    """
    import re
    pattern = re.compile(r'\b' + re.escape(symbol) + r'\b')
    for i, line in enumerate(source.splitlines(), start=1):
        m = pattern.search(line)
        if m:
            return i, m.start()
    return None, None
