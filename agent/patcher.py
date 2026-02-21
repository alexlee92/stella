import ast
import datetime
import difflib
import os
import shutil
from typing import Dict, List, Tuple

from agent.ast_merge import ast_merge_python_code
from agent.config import DRY_RUN, PROJECT_ROOT
from agent.partial_edits import parse_partial_edit, apply_multi_edit
from agent.ts_merge import ts_merge, is_js_ts_file  # P4.1


def _safe_abs(path: str) -> str:
    abs_path = os.path.abspath(
        path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    )
    root = os.path.abspath(PROJECT_ROOT)
    if not abs_path.startswith(root):
        raise ValueError("Path outside project root")
    return abs_path


def create_backup(filepath: str):
    abs_path = _safe_abs(filepath)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = f"{abs_path}.bak_{timestamp}"
    if os.path.exists(abs_path):
        shutil.copy(abs_path, backup)
    print(f"[patch] backup: {backup}")
    return backup


def restore_backup(filepath: str, backup_path: str):
    abs_path = _safe_abs(filepath)
    if not backup_path or not os.path.exists(backup_path):
        return False
    if DRY_RUN:
        print(f"[dry-run] restore skipped for {abs_path}")
        return True
    shutil.copy(backup_path, abs_path)
    print(f"[patch] restored from backup: {backup_path}")
    return True


def find_latest_backup(filepath: str):
    abs_path = _safe_abs(filepath)
    base = f"{abs_path}.bak_"
    directory = os.path.dirname(abs_path)
    candidates = [
        os.path.join(directory, n)
        for n in os.listdir(directory)
        if n.startswith(os.path.basename(base))
    ]
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0]


def _colorize_diff_line(line: str) -> str:
    """Colore une ligne de diff avec des codes ANSI."""
    RED = "\033[31m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    RESET = "\033[0m"
    if line.startswith("+++") or line.startswith("---"):
        return f"{CYAN}{line}{RESET}"
    if line.startswith("@@"):
        return f"{CYAN}{line}{RESET}"
    if line.startswith("+"):
        return f"{GREEN}{line}{RESET}"
    if line.startswith("-"):
        return f"{RED}{line}{RESET}"
    return line


def show_diff(old: str, new: str, filepath: str = "") -> str:
    """Affiche un diff coloré et retourne le texte brut du diff."""
    from_label = f"a/{filepath}" if filepath else "a/original"
    to_label = f"b/{filepath}" if filepath else "b/modified"
    diff_lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile=from_label, tofile=to_label, lineterm=""
    ))
    if not diff_lines:
        print("\n[patch] aucune difference detectee\n")
        return ""

    print(f"\n[patch] proposed diff ({len([l for l in diff_lines if l.startswith('+') and not l.startswith('+++')])} additions, {len([l for l in diff_lines if l.startswith('-') and not l.startswith('---')])} deletions):\n")
    for line in diff_lines:
        print(_colorize_diff_line(line))
    print()
    return "\n".join(diff_lines)


def _prepare_new_code(abs_path: str, old_code: str, new_code: str):
    # Try partial edits first (language agnostic)
    edits = parse_partial_edit(new_code)
    if edits:
        patched = apply_multi_edit(old_code, edits)
        if patched != old_code:
            print(f"[patch] applied {len(edits)} partial edits")
            # If it's python, still check syntax
            if abs_path.endswith(".py"):
                try:
                    ast.parse(patched)
                except SyntaxError as exc:
                    raise ValueError(f"partial edit produced invalid python: {exc}")
            return patched, {"partial_edits": True, "count": len(edits)}

    if abs_path.endswith(".py"):
        stripped = (new_code or "").lstrip()
        if stripped.startswith("```") or stripped.lower().startswith(
            "here's the modified code"
        ):
            raise ValueError("refusing markdown/prose patch for python file")

        merged, used, reason = ast_merge_python_code(old_code, new_code)
        candidate = merged if used else new_code
        try:
            ast.parse(candidate)
        except SyntaxError as exc:
            raise ValueError(f"refusing invalid python patch: {exc}") from exc

        if used:
            print(f"[patch] ast-aware merge applied: {reason}")
            return merged, {"ast_merge": True, "reason": reason}
        return new_code, {"ast_merge": False, "reason": reason}

    # P4.1 — JS/TS symbol-aware merge
    if is_js_ts_file(abs_path) and old_code.strip():
        merged, used, reason = ts_merge(old_code, new_code)
        if used:
            print(f"[patch] ts-merge applied: {reason}")
            return merged, {"ts_merge": True, "reason": reason}

    return new_code, {"ast_merge": False, "reason": "non_python_file"}


def apply_patch_non_interactive(filepath: str, new_code: str):
    abs_path = _safe_abs(filepath)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            old_code = f.read()
    except FileNotFoundError:
        old_code = ""

    try:
        prepared_code, ast_meta = _prepare_new_code(abs_path, old_code, new_code)
    except ValueError as exc:
        return {
            "old_code": old_code,
            "new_code": new_code,
            "backup_path": None,
            "dry_run": DRY_RUN,
            "applied": False,
            "ast_merge": False,
            "reason": str(exc),
        }

    backup_path = create_backup(abs_path)

    if DRY_RUN:
        return {
            "old_code": old_code,
            "new_code": prepared_code,
            "backup_path": backup_path,
            "dry_run": True,
            "applied": False,
            **ast_meta,
        }

    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(prepared_code)

    return {
        "old_code": old_code,
        "new_code": prepared_code,
        "backup_path": backup_path,
        "dry_run": False,
        "applied": True,
        **ast_meta,
    }


def apply_patch_interactive(filepath: str, new_code: str):
    abs_path = _safe_abs(filepath)
    try:
        with open(abs_path, "r", encoding="utf-8") as f:
            old_code = f.read()
    except FileNotFoundError:
        old_code = ""

    try:
        prepared_code, ast_meta = _prepare_new_code(abs_path, old_code, new_code)
    except ValueError as exc:
        print(f"[patch] rejected: {exc}")
        return False

    show_diff(old_code, prepared_code, filepath=filepath)

    confirm = input("\nAppliquer ce patch ? [y/n/q] ")
    if confirm.lower() != "y":
        print("[patch] cancelled")
        return False

    create_backup(abs_path)

    if DRY_RUN:
        print(f"[dry-run] patch skipped for {abs_path}")
        return True

    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(prepared_code)

    print("[patch] applied")
    if ast_meta.get("ast_merge"):
        print(f"[patch] note: ast-aware merge ({ast_meta.get('reason')})")
    return True


def validate_cross_imports(file_to_code: Dict[str, str]) -> List[str]:
    """P3.4 — Validate import coherence across files in a transaction.

    Checks that if file A imports from file B, and both are being modified,
    the symbols being imported actually exist in the new version of B.
    Returns a list of warning strings (empty = all good).
    """
    warnings = []
    py_files = {p: c for p, c in file_to_code.items() if p.endswith(".py")}
    if len(py_files) < 2:
        return warnings

    # Build a map of symbols defined in each file
    defined_symbols: Dict[str, set] = {}
    for path, code in py_files.items():
        try:
            tree = ast.parse(code)
            syms = set()
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    syms.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            syms.add(target.id)
            defined_symbols[path] = syms
        except SyntaxError:
            pass

    # Check imports between transaction files
    import re as _re
    for path, code in py_files.items():
        for m in _re.finditer(
            r"^\s*from\s+([A-Za-z0-9_.]+)\s+import\s+(.+)$",
            code,
            flags=_re.MULTILINE,
        ):
            module = m.group(1)
            imported = [s.strip().split(" as ")[0] for s in m.group(2).split(",")]
            # Find which transaction file this module maps to
            module_path = module.replace(".", "/") + ".py"
            for tx_path in py_files:
                norm = tx_path.replace("\\", "/")
                if norm == module_path or norm.endswith("/" + module_path):
                    syms = defined_symbols.get(tx_path, set())
                    for imp in imported:
                        imp = imp.strip()
                        if imp and imp != "*" and imp not in syms:
                            warnings.append(
                                f"{path} imports '{imp}' from {tx_path} but it's not defined there"
                            )
    return warnings


def apply_transaction(
    file_to_code: Dict[str, str],
) -> Tuple[bool, List[Tuple[str, str]], str]:
    # P3.4 — Cross-import validation
    cross_warnings = validate_cross_imports(file_to_code)
    if cross_warnings:
        print(f"\n  [!] Cross-import warnings:")
        for w in cross_warnings[:5]:
            print(f"    - {w}")

    backups = []
    try:
        for path, code in file_to_code.items():
            res = apply_patch_non_interactive(path, code)
            if not res.get("applied") and not res.get("dry_run"):
                raise ValueError(res.get("reason", "patch_not_applied"))
            backups.append((path, res.get("backup_path")))
        return True, backups, "transaction_applied"
    except Exception as exc:
        for path, backup in reversed(backups):
            restore_backup(path, backup)
        return False, backups, f"transaction_failed:{exc}"


def rollback_transaction(backups: List[Tuple[str, str]]) -> bool:
    ok = True
    for path, backup in reversed(backups):
        if not restore_backup(path, backup):
            ok = False
    return ok
