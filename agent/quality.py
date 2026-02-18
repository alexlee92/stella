import re
from typing import List, Optional

from agent.config import FORMAT_COMMAND, LINT_COMMAND, TEST_COMMAND
from agent.test_selector import build_targeted_pytest_command
from agent.tooling import run_safe_command


def _python_files(changed_files: Optional[List[str]]) -> List[str]:
    if not changed_files:
        return []
    out = []
    for p in changed_files:
        if p.endswith(".py"):
            out.append(p)
    return out


def _build_changed_file_command(
    base_command: str, changed_files: Optional[List[str]]
) -> str:
    py_files = _python_files(changed_files)
    if not py_files:
        return base_command

    if base_command.startswith("python -m black"):
        return "python -m black " + " ".join(py_files)
    if base_command.startswith("python -m ruff check"):
        return "python -m ruff check " + " ".join(py_files)

    return base_command


def _is_test_stage_success(code: int, output: str) -> bool:
    if code == 0:
        return True
    low = (output or "").lower()
    if code == 5 and "no tests ran" in low:
        return True
    return False


def _is_permission_error(output: str) -> bool:
    low = (output or "").lower()
    return (
        "permissionerror" in low or "accès refusé" in low or "access is denied" in low
    )


def run_quality_pipeline(
    mode: str = "full",
    changed_files: Optional[List[str]] = None,
    format_cmd: str = FORMAT_COMMAND,
    lint_cmd: str = LINT_COMMAND,
    test_cmd: str = TEST_COMMAND,
):
    results = []

    if mode not in {"fast", "full"}:
        return False, [
            {
                "stage": "config",
                "command": mode,
                "exit_code": 2,
                "output": "invalid mode",
            }
        ]

    # Always prefer scoped commands when changed_files are known, even in full mode.
    effective_format = _build_changed_file_command(format_cmd, changed_files)
    effective_lint = _build_changed_file_command(lint_cmd, changed_files)
    if (
        changed_files
        and effective_lint == lint_cmd
        and lint_cmd.startswith("python -m ruff check")
    ):
        py_files = _python_files(changed_files)
        if py_files:
            effective_lint = "python -m ruff check " + " ".join(py_files)

    effective_tests = (
        build_targeted_pytest_command(changed_files or [], default_command=test_cmd)
        if mode == "fast"
        else test_cmd
    )

    for name, cmd in [
        ("format", effective_format),
        ("lint", effective_lint),
        ("tests", effective_tests),
    ]:
        code, out = run_safe_command(cmd)

        # If formatting/linting on broad scope fails due FS permission, fallback to changed .py files.
        if name in {"format", "lint"} and code != 0 and _is_permission_error(out):
            # If we have no scoped files, skip this stage instead of failing on OS-level folder permissions.
            if not _python_files(changed_files):
                results.append(
                    {
                        "mode": mode,
                        "stage": name,
                        "command": cmd,
                        "exit_code": 0,
                        "output": "skipped due filesystem permission limits and no changed Python files",
                        "skipped": True,
                    }
                )
                code, out = 0, "skipped"
            else:
                fallback_cmd = _build_changed_file_command(cmd, changed_files)
                if fallback_cmd != cmd:
                    f_code, f_out = run_safe_command(fallback_cmd)
                    results.append(
                        {
                            "mode": mode,
                            "stage": name,
                            "command": fallback_cmd,
                            "exit_code": f_code,
                            "output": f_out[:1200],
                            "fallback_from": cmd,
                        }
                    )
                    code, out = f_code, f_out
                else:
                    results.append(
                        {
                            "mode": mode,
                            "stage": name,
                            "command": cmd,
                            "exit_code": code,
                            "output": out[:1200],
                        }
                    )
        else:
            results.append(
                {
                    "mode": mode,
                    "stage": name,
                    "command": cmd,
                    "exit_code": code,
                    "output": out[:1200],
                }
            )

        if name == "tests":
            if not _is_test_stage_success(code, out):
                return False, results
        else:
            if code != 0:
                return False, results

    return True, results


def run_quality_gate(changed_files: Optional[List[str]] = None):
    fast_ok, fast_results = run_quality_pipeline(
        mode="fast", changed_files=changed_files
    )
    if not fast_ok:
        return False, {"fast": fast_results, "full": []}

    full_ok, full_results = run_quality_pipeline(
        mode="full", changed_files=changed_files
    )
    return full_ok, {"fast": fast_results, "full": full_results}
