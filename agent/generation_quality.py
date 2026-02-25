import ast
import os
import re
from typing import List, Optional

from agent.config import PROJECT_ROOT
from agent.test_selector import suggest_test_path

_STUB_PATTERNS = [
    re.compile(r"TODO", re.IGNORECASE),
    re.compile(r"NotImplementedError"),
    re.compile(r"pass\s*(#.*)?$", re.MULTILINE),
]


def _is_test_file(path: str) -> bool:
    low = (path or "").replace("\\", "/").lower()
    base = os.path.basename(low)
    return low.startswith("tests/") or base.startswith("test_") or base.endswith(
        "_test.py"
    )


def _safe_read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return ""


def _has_stub_markers(text: str) -> bool:
    return any(p.search(text or "") for p in _STUB_PATTERNS)


def _line_length_ok_ratio(text: str, limit: int = 120) -> float:
    lines = (text or "").splitlines()
    if not lines:
        return 1.0
    ok = sum(1 for ln in lines if len(ln) <= limit)
    return ok / len(lines)


def _wildcard_import_count(text: str) -> int:
    return len(re.findall(r"^\s*from\s+[A-Za-z0-9_.]+\s+import\s+\*", text or "", re.M))


def _annotation_rate(text: str) -> float:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return 0.0
    funcs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not funcs:
        return 1.0
    annotated = 0
    for fn in funcs:
        args_ok = all(a.annotation is not None for a in fn.args.args)
        ret_ok = fn.returns is not None
        if args_ok and ret_ok:
            annotated += 1
    return annotated / len(funcs)


def assess_generated_files(
    paths: List[str], project_root: Optional[str] = None
) -> dict:
    root = os.path.abspath(project_root or PROJECT_ROOT)
    normalized = []
    for p in paths or []:
        if not isinstance(p, str) or not p.strip():
            continue
        rel = p.replace("\\", "/").strip()
        normalized.append(rel)
    normalized = sorted(set(normalized))

    existing_rel = []
    for rel in normalized:
        abs_path = os.path.join(root, rel)
        if os.path.exists(abs_path):
            existing_rel.append(rel)

    py_files = [p for p in existing_rel if p.lower().endswith(".py")]
    if not py_files:
        return {
            "total_files": len(normalized),
            "existing_files": len(existing_rel),
            "python_files": 0,
            "syntax_valid_rate": 0.0,
            "stub_free_rate": 0.0,
            "line_length_ok_rate": 0.0,
            "wildcard_import_violations": 0,
            "annotation_rate": 0.0,
            "test_presence_rate": 0.0,
            "score": 0.0,
        }

    syntax_ok = 0
    stub_free = 0
    line_ratio_sum = 0.0
    wildcard_violations = 0
    ann_sum = 0.0

    src_py_files = [p for p in py_files if not _is_test_file(p)]
    test_presence = 0
    touched_set = set(existing_rel)

    for rel in py_files:
        text = _safe_read_text(os.path.join(root, rel))
        try:
            ast.parse(text)
            syntax_ok += 1
        except SyntaxError:
            pass
        if not _has_stub_markers(text):
            stub_free += 1
        line_ratio_sum += _line_length_ok_ratio(text)
        wildcard_violations += _wildcard_import_count(text)
        ann_sum += _annotation_rate(text)

    for rel in src_py_files:
        suggested = suggest_test_path(rel).replace("\\", "/")
        if suggested in touched_set or os.path.isfile(os.path.join(root, suggested)):
            test_presence += 1

    py_count = len(py_files)
    syntax_rate = syntax_ok / py_count
    stub_rate = stub_free / py_count
    line_rate = line_ratio_sum / py_count
    ann_rate = ann_sum / py_count
    test_rate = test_presence / max(1, len(src_py_files))
    wildcard_score = 1.0 if wildcard_violations == 0 else max(
        0.0, 1.0 - (wildcard_violations / max(1, py_count))
    )

    # Weighted score (0..100)
    score = (
        syntax_rate * 30.0
        + stub_rate * 25.0
        + line_rate * 10.0
        + wildcard_score * 10.0
        + ann_rate * 10.0
        + test_rate * 15.0
    )

    return {
        "total_files": len(normalized),
        "existing_files": len(existing_rel),
        "python_files": py_count,
        "syntax_valid_rate": round(syntax_rate * 100, 2),
        "stub_free_rate": round(stub_rate * 100, 2),
        "line_length_ok_rate": round(line_rate * 100, 2),
        "wildcard_import_violations": wildcard_violations,
        "annotation_rate": round(ann_rate * 100, 2),
        "test_presence_rate": round(test_rate * 100, 2),
        "score": round(score, 2),
    }
