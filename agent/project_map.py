import ast
import os
from typing import Dict, List

from agent.config import PROJECT_ROOT
from agent.project_scan import get_python_files, load_file_content


def build_project_map() -> Dict[str, List[str]]:
    result = {}
    for path in get_python_files(PROJECT_ROOT):
        content = load_file_content(path)
        if not content:
            continue
        try:
            tree = ast.parse(content)
        except SyntaxError:
            continue

        symbols = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                symbols.append(f"class:{node.name}")
            elif isinstance(node, ast.FunctionDef):
                symbols.append(f"def:{node.name}")
            elif isinstance(node, ast.AsyncFunctionDef):
                symbols.append(f"async_def:{node.name}")

        rel = os.path.relpath(path, PROJECT_ROOT)
        result[rel] = symbols

    return result


def render_project_map(limit_files: int = 120, limit_symbols: int = 25) -> str:
    mapping = build_project_map()
    lines = []
    for idx, (path, symbols) in enumerate(sorted(mapping.items())):
        if idx >= limit_files:
            break
        shown = ", ".join(symbols[:limit_symbols]) if symbols else "(no top-level symbols)"
        lines.append(f"{path}: {shown}")
    return "\n".join(lines)
