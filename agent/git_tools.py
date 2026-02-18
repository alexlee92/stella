import subprocess
from typing import Optional

from agent.config import DRY_RUN, PROJECT_ROOT


def _run_git(args):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
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
        path = line[3:].strip()
        if path:
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
    return code2, out2


def diff_summary(target: Optional[str] = None):
    args = ["diff", "--", target] if target else ["diff"]
    code, out = _run_git(args)
    if code != 0:
        return f"git diff error: {out}"

    lines = out.splitlines()
    added = sum(1 for l in lines if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in lines if l.startswith("-") and not l.startswith("---"))
    return f"changed_lines:+{added}/-{removed}\n{out[:3000]}"
