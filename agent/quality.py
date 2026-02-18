from typing import List, Optional

from agent.config import FORMAT_COMMAND, LINT_COMMAND, TEST_COMMAND
from agent.test_selector import build_targeted_pytest_command
from agent.tooling import run_safe_command


def _python_files(changed_files: Optional[List[str]]) -> List[str]:
    if not changed_files:
        return []
    return [p for p in changed_files if p.endswith(".py")]


def _build_changed_file_command(base_command: str, changed_files: Optional[List[str]]) -> str:
    py_files = _python_files(changed_files)
    if not py_files:
        return base_command

    if base_command.startswith("python -m black"):
        return "python -m black " + " ".join(py_files)
    if base_command.startswith("python -m ruff check"):
        return "python -m ruff check " + " ".join(py_files)

    return base_command


def run_quality_pipeline(
    mode: str = "full",
    changed_files: Optional[List[str]] = None,
    format_cmd: str = FORMAT_COMMAND,
    lint_cmd: str = LINT_COMMAND,
    test_cmd: str = TEST_COMMAND,
):
    results = []

    if mode not in {"fast", "full"}:
        return False, [{"stage": "config", "command": mode, "exit_code": 2, "output": "invalid mode"}]

    effective_format = _build_changed_file_command(format_cmd, changed_files) if mode == "fast" else format_cmd
    effective_lint = _build_changed_file_command(lint_cmd, changed_files) if mode == "fast" else lint_cmd
    effective_tests = build_targeted_pytest_command(changed_files or [], default_command=test_cmd) if mode == "fast" else test_cmd

    for name, cmd in [("format", effective_format), ("lint", effective_lint), ("tests", effective_tests)]:
        code, out = run_safe_command(cmd)
        results.append({"mode": mode, "stage": name, "command": cmd, "exit_code": code, "output": out[:1200]})
        if code != 0:
            return False, results

    return True, results


def run_quality_gate(changed_files: Optional[List[str]] = None):
    fast_ok, fast_results = run_quality_pipeline(mode="fast", changed_files=changed_files)
    if not fast_ok:
        return False, {"fast": fast_results, "full": []}

    full_ok, full_results = run_quality_pipeline(mode="full", changed_files=changed_files)
    return full_ok, {"fast": fast_results, "full": full_results}
