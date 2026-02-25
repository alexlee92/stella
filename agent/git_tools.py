import os
import subprocess
from typing import Optional

from agent.config import DRY_RUN, PROJECT_ROOT

_SAFE_DIR = PROJECT_ROOT.replace("\\", "/")


def _run_git(args):
    try:
        result = subprocess.run(
            ["git", "-c", f"safe.directory={_SAFE_DIR}", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception as exc:
        return 2, str(exc)
    out = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode, out.strip()


def is_git_repo():
    code, out = _run_git(["rev-parse", "--is-inside-work-tree"])
    return code == 0 and out.strip().lower() == "true"


def current_branch():
    code, out = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if code != 0:
        return None
    return out.strip()


def changed_files():
    code, out = _run_git(["status", "--porcelain"])
    if code != 0:
        return []

    files = []
    for line in out.splitlines():
        if not line.strip():
            continue

        # Porcelain format: XY <path> OR XY <old> -> <new>
        body = line[3:].strip()
        if " -> " in body:
            body = body.split(" -> ", 1)[1].strip()

        path = body.strip('"').replace("\\", "/")
        if not path:
            continue

        abs_path = os.path.join(PROJECT_ROOT, path)
        if os.path.exists(abs_path):
            files.append(path)

    return sorted(set(files))


def create_branch(name: str):
    if DRY_RUN:
        return 0, f"[dry-run] git checkout -b {name}"
    return _run_git(["checkout", "-b", name])


def commit_all(message: str):
    if DRY_RUN:
        return 0, f"[dry-run] git add -A && git commit -m {message}"

    code1, out1 = _run_git(["add", "-A"])
    if code1 != 0:
        return code1, out1

    code2, out2 = _run_git(["commit", "-m", message])
    if code2 != 0 and "nothing to commit" in (out2 or "").lower():
        return 0, out2

    return code2, out2


def diff_summary(target: Optional[str] = None):
    args = ["diff", "--", target] if target else ["diff"]
    code, out = _run_git(args)
    if code != 0:
        return f"git diff error: {out}"

    lines = out.splitlines()
    added = sum(
        1 for line in lines if line.startswith("+") and not line.startswith("+++")
    )
    removed = sum(
        1 for line in lines if line.startswith("-") and not line.startswith("---")
    )
    return f"changed_lines:+{added}/-{removed}\n{out[:3000]}"


# --- P3.1 : Git avance ---


def git_stash(message: Optional[str] = None):
    """Sauvegarde le travail en cours dans le stash."""
    if DRY_RUN:
        return 0, "[dry-run] git stash"
    args = ["stash", "push"]
    if message:
        args.extend(["-m", message])
    return _run_git(args)


def git_stash_pop():
    """Restaure le dernier stash."""
    if DRY_RUN:
        return 0, "[dry-run] git stash pop"
    return _run_git(["stash", "pop"])


def git_stash_list():
    """Liste les stashs disponibles."""
    return _run_git(["stash", "list"])


def git_log(file_path: Optional[str] = None, limit: int = 10):
    """Historique des commits (optionnellement filtre par fichier)."""
    args = ["log", "--oneline", f"-{limit}", "--no-color"]
    if file_path:
        args.extend(["--", file_path])
    code, out = _run_git(args)
    return out if code == 0 else f"git log error: {out}"


def git_blame(file_path: str, start_line: int = 0, end_line: int = 0):
    """Qui a ecrit quoi — blame sur un fichier (optionnellement un range de lignes)."""
    args = ["blame", "--no-color"]
    if start_line > 0 and end_line > 0:
        args.extend(["-L", f"{start_line},{end_line}"])
    args.append(file_path)
    code, out = _run_git(args)
    return out[:4000] if code == 0 else f"git blame error: {out}"


def git_diff_staged():
    """Diff des fichiers stages (git add)."""
    code, out = _run_git(["diff", "--cached"])
    if code != 0:
        return f"git diff --cached error: {out}"
    return out[:3000]
