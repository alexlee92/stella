"""
P5.4 — Automatic changelog generation for agent sessions.

Generates conventional-commit-style changelogs from agent activity,
including files touched, tests affected, and migrations needed.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional

from agent.config import PROJECT_ROOT


def generate_session_changelog(
    goal: str,
    files_modified: List[str],
    steps: List[dict],
    test_results: Optional[str] = None,
) -> str:
    """Generate a changelog entry for a completed agent session.

    Args:
        goal: The original user goal
        files_modified: List of file paths modified/created
        steps: List of step dicts with action/reason/result
        test_results: Optional test output summary
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    change_type = _infer_change_type(goal, files_modified)

    lines = [
        f"## [{change_type}] {goal[:80]}",
        f"**Date**: {now}",
        "",
    ]

    if files_modified:
        lines.append("### Files changed")
        for f in files_modified[:20]:
            action = "created" if not os.path.exists(os.path.join(PROJECT_ROOT, f + ".bak_*")) else "modified"
            lines.append(f"- `{f}` ({action})")
        lines.append("")

    # Categorize files
    categories = _categorize_files(files_modified)
    if any(categories.values()):
        lines.append("### Impact")
        if categories["models"]:
            lines.append(f"- **Models**: {', '.join(categories['models'])}")
        if categories["tests"]:
            lines.append(f"- **Tests**: {', '.join(categories['tests'])}")
        if categories["migrations"]:
            lines.append(f"- **Migrations**: {', '.join(categories['migrations'])}")
        if categories["configs"]:
            lines.append(f"- **Config**: {', '.join(categories['configs'])}")
        lines.append("")

    if test_results:
        lines.append("### Test results")
        lines.append(f"```\n{test_results[:500]}\n```")
        lines.append("")

    # Steps summary
    if steps:
        lines.append("### Agent steps")
        for s in steps[:12]:
            action = s.get("action", "?")
            reason = s.get("reason", "")[:60]
            lines.append(f"- {action}: {reason}")
        lines.append("")

    return "\n".join(lines)


def _infer_change_type(goal: str, files: List[str]) -> str:
    """Infer conventional commit type from goal and files."""
    low = goal.lower()
    if any(k in low for k in ["fix", "bug", "correct", "repair", "corrig"]):
        return "fix"
    if any(k in low for k in ["test", "spec", "coverage"]):
        return "test"
    if any(k in low for k in ["refactor", "clean", "restructur"]):
        return "refactor"
    if any(k in low for k in ["doc", "readme", "comment"]):
        return "docs"
    if any(k in low for k in ["perf", "optimiz", "speed", "latenc"]):
        return "perf"
    if any(f.endswith((".toml", ".yaml", ".yml", ".json", ".cfg")) for f in files):
        return "chore"
    return "feat"


def _categorize_files(files: List[str]) -> Dict[str, List[str]]:
    """Categorize files by their role in the project."""
    categories: Dict[str, List[str]] = {
        "models": [],
        "tests": [],
        "migrations": [],
        "configs": [],
    }
    for f in files:
        low = f.lower()
        if "model" in low or "schema" in low or "orm" in low:
            categories["models"].append(f)
        elif "test" in low or "spec" in low:
            categories["tests"].append(f)
        elif "migrat" in low or "alembic" in low:
            categories["migrations"].append(f)
        elif any(f.endswith(ext) for ext in (".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".env")):
            categories["configs"].append(f)
    return categories


def append_to_changelog_file(
    entry: str,
    changelog_path: str = "CHANGELOG.md",
) -> str:
    """Append a changelog entry to the project's CHANGELOG.md."""
    abs_path = os.path.join(PROJECT_ROOT, changelog_path)
    try:
        existing = ""
        if os.path.exists(abs_path):
            with open(abs_path, "r", encoding="utf-8") as f:
                existing = f.read()

        with open(abs_path, "w", encoding="utf-8") as f:
            if existing:
                # Insert after the first heading
                lines = existing.split("\n", 1)
                f.write(lines[0] + "\n\n" + entry + "\n\n---\n\n")
                if len(lines) > 1:
                    f.write(lines[1])
            else:
                f.write("# Changelog\n\n" + entry + "\n")

        return f"Changelog updated: {changelog_path}"
    except OSError as e:
        return f"Failed to update changelog: {e}"
