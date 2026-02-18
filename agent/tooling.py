import os
import re
import shlex
import subprocess
from typing import List, Tuple

from agent.config import DRY_RUN, PROJECT_ROOT
from agent.project_scan import get_python_files, load_file_content

_ALLOWED_COMMAND_PREFIXES = [
    ["pytest"],
    ["python", "-m", "pytest"],
    ["python", "-m", "ruff"],
    ["python", "-m", "black"],
]


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        abs_path = os.path.abspath(path)
    else:
        abs_path = os.path.abspath(os.path.join(PROJECT_ROOT, path))

    root = os.path.abspath(PROJECT_ROOT)
    if not abs_path.startswith(root):
        raise ValueError("Path is outside project root")

    return abs_path


def is_command_allowed(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    for prefix in _ALLOWED_COMMAND_PREFIXES:
        if tokens[: len(prefix)] == prefix:
            return True
    return False


def run_safe_command(command: str, timeout: int = 300) -> Tuple[int, str]:
    if not is_command_allowed(command):
        return 2, f"Refused command: {command}"

    if DRY_RUN and "pytest" not in command:
        return 0, f"[dry-run] skipped command: {command}"

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
            check=False,
        )
    except Exception as exc:
        return 2, f"Command failed to run: {exc}"

    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode, (output.strip() or "(no output)")[:8000]


def read_file(path: str, max_chars: int = 6000) -> str:
    abs_path = _resolve_path(path)
    content = load_file_content(abs_path)
    if len(content) <= max_chars:
        return content
    return content[:max_chars] + "\n\n...[truncated]"


def read_many(
    paths: List[str], max_chars_per_file: int = 2500, max_total_chars: int = 9000
) -> str:
    chunks = []
    total = 0
    for path in paths:
        try:
            content = read_file(path, max_chars=max_chars_per_file)
        except Exception as exc:
            content = f"[error] {exc}"

        block = f"FILE: {path}\n{content}\n"
        if total + len(block) > max_total_chars:
            break
        chunks.append(block)
        total += len(block)

    return "\n".join(chunks) if chunks else "No files read"


def list_python_files(limit: int = 200) -> List[str]:
    files = get_python_files(PROJECT_ROOT)
    rel = [os.path.relpath(p, PROJECT_ROOT) for p in files]
    rel.sort()
    return rel[:limit]


def list_files(limit: int = 200, contains: str = "", ext: str = ".py") -> List[str]:
    files = list_python_files(limit=100000)
    out = []
    for rel in files:
        if ext and not rel.endswith(ext):
            continue
        if contains and contains not in rel:
            continue
        out.append(rel)
        if len(out) >= limit:
            break
    return out


def search_code(pattern: str, limit: int = 30) -> List[str]:
    try:
        rg_cmd = [
            "rg",
            "-n",
            "--hidden",
            "--glob",
            "!**/__pycache__/**",
            pattern,
            PROJECT_ROOT,
        ]
        result = subprocess.run(rg_cmd, capture_output=True, text=True, check=False)
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines[:limit]
    except Exception:
        pass

    output = []
    try:
        expr = re.compile(pattern)
    except re.error:
        expr = re.compile(re.escape(pattern))

    for file_path in get_python_files(PROJECT_ROOT):
        content = load_file_content(file_path)
        for idx, line in enumerate(content.splitlines(), start=1):
            if expr.search(line):
                rel = os.path.relpath(file_path, PROJECT_ROOT)
                output.append(f"{rel}:{idx}:{line.strip()}")
                if len(output) >= limit:
                    return output
    return output


def run_tests_detailed(
    command: str = "pytest -q", timeout: int = 180
) -> Tuple[int, str]:
    return run_safe_command(command, timeout=timeout)


def run_tests(command: str = "pytest -q", timeout: int = 180) -> str:
    code, output = run_tests_detailed(command=command, timeout=timeout)
    return f"exit_code={code}\n{output}"
