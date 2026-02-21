import json
import os
from datetime import datetime

from agent.agent import index_project
from agent.auto_agent import AutonomousAgent
from agent.config import PROJECT_ROOT
from agent.git_tools import changed_files, diff_summary


def _next_action(has_changes: bool, auto_apply: bool) -> str:
    if has_changes and auto_apply:
        return "Run `python stella.py pr-ready \"<goal>\"` to package the patch."
    if has_changes and not auto_apply:
        return "Re-run with `--apply` to apply staged edits."
    return "Refine goal to a concrete file-level change and rerun `dev-task`."


def _write_run_summary(payload: dict) -> tuple[str, str]:
    out_dir = os.path.join(PROJECT_ROOT, ".stella")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "last_dev_task.json")
    md_path = os.path.join(out_dir, "last_dev_task.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    lines = [
        "# Dev Task Summary",
        "",
        f"- timestamp: {payload.get('timestamp')}",
        f"- goal: {payload.get('goal')}",
        f"- status: {payload.get('status')}",
        f"- changed_files: {payload.get('changed_files_count')}",
        f"- next_action: {payload.get('next_action')}",
        "",
        "## Agent Summary",
        payload.get("agent_summary", ""),
        "",
        "## Diff",
        "```",
        (payload.get("diff_summary") or "")[:1800],
        "```",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return json_path, md_path


def run_dev_task(
    goal: str,
    max_steps: int = 10,
    auto_apply: bool = False,
    fix_until_green: bool = False,
    with_tests: bool = False,
    profile: str | None = None,
    max_seconds: int = 0,
) -> dict:
    if profile == "safe":
        auto_apply = False
        fix_until_green = False
        with_tests = False
        max_steps = max(4, min(max_steps, 8))
    elif profile == "standard":
        auto_apply = True
        fix_until_green = False
        with_tests = True
        max_steps = max(6, min(max_steps, 10))
    elif profile == "aggressive":
        auto_apply = True
        fix_until_green = True
        with_tests = True
        max_steps = max(8, min(max_steps, 16))

    index_project()
    summary = AutonomousAgent(max_steps=max_steps).run(
        goal=goal,
        auto_apply=auto_apply,
        fix_until_green=fix_until_green,
        generate_tests=with_tests,
        max_seconds=max_seconds,
    )

    changed = changed_files()
    diff = diff_summary() if changed else ""
    status = "changes_ready" if changed else "no_changes"

    payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "goal": goal,
        "status": status,
        "changed_files_count": len(changed),
        "changed_files": changed,
        "next_action": _next_action(bool(changed), auto_apply=auto_apply),
        "agent_summary": summary,
        "diff_summary": diff,
        "options": {
            "max_steps": max_steps,
            "auto_apply": auto_apply,
            "fix_until_green": fix_until_green,
            "with_tests": with_tests,
            "profile": profile or "custom",
            "max_seconds": max_seconds,
        },
    }
    json_path, md_path = _write_run_summary(payload)
    payload["summary_json"] = json_path
    payload["summary_md"] = md_path
    return payload


def ide_shortcuts() -> dict:
    return {
        "run_dev_task_standard": "python stella.py dev-task \"<goal>\" --profile standard",
        "run_dev_task_aggressive": "python stella.py dev-task \"<goal>\" --profile aggressive --max-seconds 900",
        "run_dev_task_safe": "python stella.py dev-task \"<goal>\" --profile safe",
        "review_only": "python stella.py run \"<goal>\" --steps 8",
        "package_pr": "python stella.py pr-ready \"<goal>\"",
        "last_summary": ".stella/last_dev_task.md",
        "last_report": "eval/last_report.json",
    }
