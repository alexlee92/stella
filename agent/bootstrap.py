import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

from agent.agent import index_project
from agent.config import PROJECT_ROOT
from agent.git_tools import is_git_repo

TOOLS = ["pytest", "ruff", "black"]


def _run(cmd):
    try:
        proc = subprocess.run(
            cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return proc.returncode, out.strip()
    except Exception as exc:
        return 2, str(exc)


def run_bootstrap(
    init_git: bool = True, rebuild_index: bool = True, install_tools: bool = True
) -> dict:
    steps = []

    if init_git:
        if is_git_repo():
            steps.append(
                {"step": "git_init", "ok": True, "details": "already a git repo"}
            )
        else:
            code, out = _run(["git", "init"])
            steps.append({"step": "git_init", "ok": code == 0, "details": out[:800]})

    if rebuild_index:
        try:
            index_project(force_rebuild=True)
            steps.append(
                {"step": "memory_index", "ok": True, "details": "index rebuilt"}
            )
        except Exception as exc:
            steps.append({"step": "memory_index", "ok": False, "details": str(exc)})

    if install_tools:
        for tool in TOOLS:
            if shutil.which(tool):
                steps.append(
                    {
                        "step": f"install_{tool}",
                        "ok": True,
                        "details": "already installed",
                    }
                )
                continue

            code, out = _run([sys.executable, "-m", "pip", "install", tool])
            steps.append(
                {"step": f"install_{tool}", "ok": code == 0, "details": out[:800]}
            )

    ok = all(s.get("ok") for s in steps) if steps else True
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "ok": ok,
        "steps": steps,
    }

    out_dir = os.path.join(PROJECT_ROOT, ".stella")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bootstrap_last.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    report["report_path"] = out_path
    return report


def format_bootstrap(report: dict) -> str:
    lines = [f"Bootstrap: {'OK' if report.get('ok') else 'FAILED'}"]
    for s in report.get("steps", []):
        status = "OK" if s.get("ok") else "FAIL"
        lines.append(f"- [{status}] {s.get('step')}: {s.get('details')}")
    lines.append(f"report: {report.get('report_path')}")
    return "\n".join(lines)
