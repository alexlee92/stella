"""
P4.5 -- File watcher for automatic test re-runs.

Monitors file changes using polling (no external dependency required)
and re-runs tests when modifications are detected.
"""

import fnmatch
import os
import time
from typing import Dict, Optional, Set

from agent.config import PROJECT_ROOT
from agent.tooling import run_safe_command

# Directories to always skip
_SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "node_modules",
    ".stella", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "egg-info",
}


def _scan_files(root: str, pattern: str) -> Dict[str, float]:
    """Scan files matching pattern, return {relative_path: mtime}."""
    result = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            rel = os.path.relpath(os.path.join(dirpath, fname), root)
            rel = rel.replace("\\", "/")
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(fname, pattern):
                try:
                    mtime = os.path.getmtime(os.path.join(dirpath, fname))
                    result[rel] = mtime
                except OSError:
                    pass
    return result


def _detect_changes(
    old_state: Dict[str, float], new_state: Dict[str, float]
) -> Set[str]:
    """Return set of files that were modified, added, or deleted."""
    changed = set()
    for path, mtime in new_state.items():
        if path not in old_state or old_state[path] != mtime:
            changed.add(path)
    for path in old_state:
        if path not in new_state:
            changed.add(path)
    return changed


def run_watch(
    pattern: str = "**/*.py",
    command: Optional[str] = None,
    interval: float = 2.0,
):
    """Watch for file changes and re-run tests.

    Args:
        pattern: Glob pattern of files to watch
        command: Shell command to run on changes (default: pytest -q)
        interval: Seconds between scans
    """
    if command is None:
        from agent.config import TEST_COMMAND
        command = TEST_COMMAND or "pytest -q"

    print(f"[watch] Surveillance active : {pattern}")
    print(f"[watch] Commande : {command}")
    print(f"[watch] Intervalle : {interval}s")
    print(f"[watch] Ctrl+C pour arreter\n")

    state = _scan_files(PROJECT_ROOT, pattern)
    print(f"[watch] {len(state)} fichiers surveilles.")

    # Run tests once at start
    print(f"\n--- Lancement initial ---")
    code, output = run_safe_command(command, timeout=300)
    _print_result(code, output)

    try:
        while True:
            time.sleep(interval)
            new_state = _scan_files(PROJECT_ROOT, pattern)
            changed = _detect_changes(state, new_state)

            if changed:
                state = new_state
                _print_changes(changed)
                print(f"\n--- Re-lancement : {command} ---")
                code, output = run_safe_command(command, timeout=300)
                _print_result(code, output)
    except KeyboardInterrupt:
        print("\n[watch] Arrete.")


def _print_changes(changed: Set[str]):
    """Print changed files."""
    print(f"\n[watch] {len(changed)} fichier(s) modifie(s) :")
    for f in sorted(changed)[:10]:
        print(f"  ~ {f}")
    if len(changed) > 10:
        print(f"  ... et {len(changed) - 10} autres")


def _print_result(code: int, output: str):
    """Print test result with status indicator."""
    status = "PASS" if code == 0 else "FAIL"
    marker = "[OK]" if code == 0 else "[KO]"
    print(output[:4000])
    print(f"\n[watch] {marker} {status} (exit code {code})")
    print(f"[watch] En attente de modifications...")
