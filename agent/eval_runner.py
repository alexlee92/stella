import json
import os
import time
from datetime import datetime

from agent.agent import ask_project, index_project
from agent.auto_agent import AutonomousAgent
from agent.config import EVENT_LOG_PATH, PROJECT_ROOT


def _score(output: str, checks: list[str]) -> bool:
    low = output.lower()
    return any(c.lower() in low for c in checks)


def _safe_read_jsonl(path: str):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _compute_kpis(report: dict):
    results = report.get("results", [])
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    success_rate = round((passed / total) * 100, 2) if total else 0.0

    avg_task_time = round(sum(float(r.get("duration_s", 0.0)) for r in results) / total, 2) if total else 0.0

    events = _safe_read_jsonl(os.path.join(PROJECT_ROOT, EVENT_LOG_PATH))
    failure_events = [e for e in events if e.get("type") == "failure"]
    parse_failures = [e for e in failure_events if (e.get("payload") or {}).get("category") == "parse"]

    planner_events = [e for e in events if e.get("type") == "plan"]
    json_failure_rate = round((len(parse_failures) / max(1, len(planner_events))) * 100, 2)

    pr_ready_events = [e for e in events if e.get("type") == "pr_ready"]
    pr_success = sum(1 for e in pr_ready_events if (e.get("payload") or {}).get("ok"))
    pr_ready_usability = round((pr_success / max(1, len(pr_ready_events))) * 100, 2)

    return {
        "measured_at": datetime.utcnow().isoformat(),
        "success_rate": success_rate,
        "json_failure_rate": json_failure_rate,
        "avg_task_seconds": avg_task_time,
        "pr_ready_usability": pr_ready_usability,
        "counts": {
            "eval_total": total,
            "eval_passed": passed,
            "planner_events": len(planner_events),
            "parse_failures": len(parse_failures),
            "pr_ready_events": len(pr_ready_events),
        },
    }


def run_eval(tasks_file: str = "eval/tasks.json", max_tasks: int | None = None):
    abs_tasks = os.path.join(PROJECT_ROOT, tasks_file)
    with open(abs_tasks, "r", encoding="utf-8-sig") as f:
        tasks = json.load(f)
    if isinstance(max_tasks, int) and max_tasks > 0:
        tasks = tasks[:max_tasks]

    index_project()

    results = []
    start_all = time.time()
    for task in tasks:
        name = task["name"]
        mode = task["mode"]
        prompt = task["prompt"]
        checks = task.get("must_contain_any", [])

        t0 = time.time()
        if mode == "ask":
            output = ask_project(prompt)
        elif mode == "auto":
            output = AutonomousAgent(max_steps=6).run(prompt, auto_apply=False)
        else:
            output = f"unsupported mode: {mode}"

        dt = time.time() - t0
        passed = _score(output, checks)
        results.append(
            {
                "name": name,
                "mode": mode,
                "duration_s": round(dt, 2),
                "passed": passed,
                "output": output[:1000],
            }
        )

    duration = round(time.time() - start_all, 2)
    pass_count = sum(1 for r in results if r["passed"])
    summary = {
        "timestamp": datetime.utcnow().isoformat(),
        "total": len(results),
        "passed": pass_count,
        "pass_rate": round((pass_count / len(results)) * 100, 2) if results else 0.0,
        "duration_s": duration,
    }

    report = {"summary": summary, "results": results}
    out_path = os.path.join(PROJECT_ROOT, "eval", "last_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    kpis = _compute_kpis(report)
    kpi_path = os.path.join(PROJECT_ROOT, ".stella", "kpi_snapshot.json")
    os.makedirs(os.path.dirname(kpi_path), exist_ok=True)
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump(kpis, f, ensure_ascii=False, indent=2)

    report["kpis"] = kpis
    return report
