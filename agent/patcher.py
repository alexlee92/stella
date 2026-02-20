import ast
import datetime
import difflib
import os
import shutil
from typing import Dict, List, Tuple

from agent.ast_merge import ast_merge_python_code
from agent.config import DRY_RUN, PROJECT_ROOT
from agent.partial_edits import parse_partial_edit, apply_multi_edit


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


def show_diff(old: str, new: str):
    diff = difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="")
    print("\n[patch] proposed diff:\n")
    for line in diff:
        print(line)


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

    show_diff(old_code, prepared_code)

    confirm = input("\nApply this patch? (y/n) ")
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


def apply_transaction(
    file_to_code: Dict[str, str],
) -> Tuple[bool, List[Tuple[str, str]], str]:
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
