import json
import os
import re
import time
from datetime import datetime, UTC

from agent.agent import ask_project, index_project
from agent.auto_agent import AutonomousAgent
from agent.config import EVENT_LOG_PATH, PROJECT_ROOT
from agent.generation_quality import assess_generated_files
from agent.git_tools import changed_files
from agent.pr_ready import prepare_pr


def _score(output: str, checks: list[str]) -> bool:
    low = output.lower()
    return any(c.lower() in low for c in checks)


def _track_for_task(task: dict) -> str:
    track = task.get("track")
    if isinstance(track, str) and track.strip():
        return track.strip().lower()
    mode = str(task.get("mode", "")).lower()
    if mode == "code_edit":
        return "code_edit"
    return "qa"


def _task_bool(task: dict, key: str, default: bool = False) -> bool:
    val = task.get(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "on"}
    return default


def _task_int(task: dict, key: str, default: int) -> int:
    val = task.get(key, default)
    if isinstance(val, int):
        return val
    if isinstance(val, str) and val.isdigit():
        return int(val)
    return default


def _extract_actions_from_summary(output: str) -> list[str]:
    actions = []
    for line in (output or "").splitlines():
        # Old format: - step 1: read_file | reason
        m_old = re.match(r"^- step \d+:\s*([a-zA-Z0-9_]+)\s*\|", line.strip())
        if m_old:
            actions.append(m_old.group(1).strip().lower())
            continue

        # New format: 1. [read_file] reason -> result
        m_new = re.match(r"^\d+\.\s*\[([a-zA-Z0-9_]+)\]", line.strip())
        if m_new:
            actions.append(m_new.group(1).strip().lower())

    return actions


def _extract_steps_from_summary(output: str) -> list[dict]:
    steps = []
    for line in (output or "").splitlines():
        line = line.strip()

        # Old format: - step 1: read_file | reason | result
        m_old = re.match(r"^- step \d+:\s*([a-zA-Z0-9_]+)\s*\|\s*(.*)$", line)
        if m_old:
            action = m_old.group(1).strip().lower()
            tail = m_old.group(2)
            parts = [p.strip() for p in tail.split("|", 1)]
            reason = parts[0] if parts else ""
            result = parts[1] if len(parts) > 1 else ""
            steps.append({"action": action, "reason": reason, "result": result})
            continue

        # New format: 1. [read_file] reason -> result
        m_new = re.match(r"^\d+\.\s*\[([a-zA-Z0-9_]+)\]\s*(.*)$", line)
        if m_new:
            action = m_new.group(1).strip().lower()
            tail = m_new.group(2)
            parts = [p.strip() for p in tail.split("->", 1)]
            reason = parts[0] if parts else ""
            result = parts[1] if len(parts) > 1 else ""
            steps.append({"action": action, "reason": reason, "result": result})

    return steps


def _is_test_path(path: str) -> bool:
    low = (path or "").replace("\\", "/").lower()
    name = os.path.basename(low)
    return (
        low.startswith("tests/")
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _extract_expected_edit_paths(task: dict) -> list[str]:
    configured = task.get("expected_edit_paths")
    if isinstance(configured, list):
        out = []
        for p in configured:
            if isinstance(p, str) and p.strip():
                out.append(p.strip().replace("\\", "/").lower())
        if out:
            return out

    prompt = task.get("prompt", "")
    if not isinstance(prompt, str):
        return []
    found = re.findall(
        r"([A-Za-z0-9_./\\-]+\.(?:py|json|md|toml|ya?ml|txt))",
        prompt,
        flags=re.IGNORECASE,
    )
    out = []
    seen = set()
    for p in found:
        norm = p.replace("\\", "/").lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(norm)
    return out


def _compute_code_edit_validation(task: dict, output: str, signals: dict) -> dict:
    steps = _extract_steps_from_summary(output)
    expected_paths = _extract_expected_edit_paths(task)
    has_edit_action = bool(signals.get("has_edit_action"))

    touched_paths = []
    for step in steps:
        if step.get("action") not in {"propose_edit", "apply_edit"}:
            continue
        result = str(step.get("result", ""))
        touched_paths.extend(
            re.findall(
                r"([A-Za-z0-9_./\\-]+\.(?:py|json|md|toml|ya?ml|txt))",
                result,
                flags=re.IGNORECASE,
            )
        )
    touched_paths = [p.replace("\\", "/").lower() for p in touched_paths]
    patch_with_tests = any(_is_test_path(p) for p in touched_paths)

    expected_diff_match = False
    if expected_paths:
        expected_diff_match = any(
            any(exp in touched for touched in touched_paths) for exp in expected_paths
        )
    elif has_edit_action:
        expected_diff_match = True

    tests_ran = any(s.get("action") in {"run_tests", "run_quality"} for s in steps)
    run_tests_green = any(
        s.get("action") == "run_tests"
        and "exit_code=0" in str(s.get("result", "")).lower()
        for s in steps
    )
    run_quality_green = any(
        s.get("action") == "run_quality"
        and "ok=true" in str(s.get("result", "")).lower()
        for s in steps
    )
    tests_green = tests_ran and (run_tests_green or run_quality_green)
    strict_scope = _task_bool(task, "strict_scope", bool(expected_paths))
    unexpected_touched_paths = []
    if strict_scope and expected_paths:
        for touched in touched_paths:
            if not any(exp in touched for exp in expected_paths) and not _is_test_path(
                touched
            ):
                unexpected_touched_paths.append(touched)

    quality_rates = []
    for s in steps:
        result = str(s.get("result", ""))
        m = re.search(r"tests_quality_ok_rate=([0-9]+(?:\.[0-9]+)?)", result)
        if m:
            try:
                quality_rates.append(float(m.group(1)))
            except ValueError:
                pass
    tests_quality_ok = (max(quality_rates) if quality_rates else 0.0) >= 70.0

    score = (int(has_edit_action) + int(expected_diff_match) + int(tests_green)) / 3.0
    return {
        "expected_edit_paths": expected_paths,
        "touched_paths": touched_paths,
        "expected_diff_match": expected_diff_match,
        "tests_ran": tests_ran,
        "tests_green": tests_green,
        "patch_with_tests": patch_with_tests,
        "tests_quality_ok": tests_quality_ok,
        "unexpected_touched_paths": unexpected_touched_paths,
        "valid_patch_score": round(score * 100, 2),
    }


def _code_edit_signals(output: str) -> dict:
    actions = _extract_actions_from_summary(output)
    action_set = set(actions)
    has_edit = bool(action_set & {"propose_edit", "apply_edit", "apply_all_staged"})
    has_validation = bool(action_set & {"run_tests", "run_quality"})
    has_code_navigation = bool(action_set & {"read_file", "read_many", "search_code"})
    score = (int(has_edit) + int(has_validation) + int(has_code_navigation)) / 3.0
    return {
        "actions": actions,
        "has_edit_action": has_edit,
        "has_validation_action": has_validation,
        "has_code_navigation_action": has_code_navigation,
        "patch_signal_score": round(score * 100, 2),
    }


def _summarize_tracks(results: list[dict]) -> dict:
    by_track: dict[str, list[dict]] = {}
    for r in results:
        t = r.get("track", "qa")
        by_track.setdefault(t, []).append(r)

    out = {}
    for track, rows in by_track.items():
        total = len(rows)
        passed = sum(1 for r in rows if r.get("passed"))
        avg_time = (
            round(sum(float(r.get("duration_s", 0.0)) for r in rows) / total, 2)
            if total
            else 0.0
        )
        out[track] = {
            "total": total,
            "passed": passed,
            "pass_rate": round((passed / total) * 100, 2) if total else 0.0,
            "avg_task_seconds": avg_time,
        }
        if track == "code_edit":
            patch_scores = [
                float((r.get("code_edit_signals") or {}).get("patch_signal_score", 0.0))
                for r in rows
            ]
            valid_scores = [
                float(
                    (r.get("code_edit_validation") or {}).get("valid_patch_score", 0.0)
                )
                for r in rows
            ]
            tests_green_count = sum(
                1
                for r in rows
                if bool((r.get("code_edit_validation") or {}).get("tests_green"))
            )
            expected_diff_match_count = sum(
                1
                for r in rows
                if bool(
                    (r.get("code_edit_validation") or {}).get("expected_diff_match")
                )
            )
            patch_with_tests_count = sum(
                1
                for r in rows
                if bool((r.get("code_edit_validation") or {}).get("patch_with_tests"))
            )
            tests_quality_ok_count = sum(
                1
                for r in rows
                if bool((r.get("code_edit_validation") or {}).get("tests_quality_ok"))
            )
            out[track]["avg_patch_signal_score"] = (
                round(sum(patch_scores) / len(patch_scores), 2) if patch_scores else 0.0
            )
            out[track]["avg_valid_patch_score"] = (
                round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else 0.0
            )
            out[track]["tests_green_rate"] = (
                round((tests_green_count / total) * 100, 2) if total else 0.0
            )
            out[track]["expected_diff_match_rate"] = (
                round((expected_diff_match_count / total) * 100, 2) if total else 0.0
            )
            out[track]["patch_with_tests_rate"] = (
                round((patch_with_tests_count / total) * 100, 2) if total else 0.0
            )
            out[track]["generated_tests_quality_rate"] = (
                round((tests_quality_ok_count / total) * 100, 2) if total else 0.0
            )
    return out


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


def _parse_event_time(event: dict) -> datetime | None:
    ts = event.get("timestamp")
    if not isinstance(ts, str) or not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        return None


def _compute_kpis(
    report: dict, start_ts: datetime | None = None, end_ts: datetime | None = None
):
    results = report.get("results", [])
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    success_rate = round((passed / total) * 100, 2) if total else 0.0

    avg_task_time = (
        round(sum(float(r.get("duration_s", 0.0)) for r in results) / total, 2)
        if total
        else 0.0
    )

    events = _safe_read_jsonl(os.path.join(PROJECT_ROOT, EVENT_LOG_PATH))
    if start_ts or end_ts:
        filtered = []
        for e in events:
            e_ts = _parse_event_time(e)
            if e_ts is None:
                continue
            if start_ts and e_ts < start_ts:
                continue
            if end_ts and e_ts > end_ts:
                continue
            filtered.append(e)
        events = filtered

    failure_events = [e for e in events if e.get("type") == "failure"]
    parse_failures = [
        e for e in failure_events if (e.get("payload") or {}).get("category") == "parse"
    ]
    parse_by_class: dict[str, int] = {}
    parse_by_prompt_class: dict[str, int] = {}
    for e in parse_failures:
        payload = (e.get("payload") or {}).get("payload") or {}
        parse_class = payload.get("parse_class")
        if not isinstance(parse_class, str) or not parse_class:
            parse_class = "unspecified"
        parse_by_class[parse_class] = parse_by_class.get(parse_class, 0) + 1
        prompt_class = payload.get("prompt_class")
        if not isinstance(prompt_class, str) or not prompt_class:
            prompt_class = "unspecified"
        parse_by_prompt_class[prompt_class] = (
            parse_by_prompt_class.get(prompt_class, 0) + 1
        )

    planner_events = [e for e in events if e.get("type") == "plan"]
    json_failure_rate = round(
        (len(parse_failures) / max(1, len(planner_events))) * 100, 2
    )

    pr_ready_events = [e for e in events if e.get("type") == "pr_ready"]
    pr_success = sum(1 for e in pr_ready_events if (e.get("payload") or {}).get("ok"))
    pr_ready_usability = round((pr_success / max(1, len(pr_ready_events))) * 100, 2)

    track_summary = _summarize_tracks(results)
    code_edit_rows = [r for r in results if r.get("track") == "code_edit"]
    code_edit_patch_score = (
        round(
            sum(
                float((r.get("code_edit_signals") or {}).get("patch_signal_score", 0.0))
                for r in code_edit_rows
            )
            / max(1, len(code_edit_rows)),
            2,
        )
        if code_edit_rows
        else 0.0
    )
    code_edit_valid_patch_score = (
        round(
            sum(
                float(
                    (r.get("code_edit_validation") or {}).get("valid_patch_score", 0.0)
                )
                for r in code_edit_rows
            )
            / max(1, len(code_edit_rows)),
            2,
        )
        if code_edit_rows
        else 0.0
    )
    code_edit_patch_with_tests_rate = (
        round(
            (
                sum(
                    1
                    for r in code_edit_rows
                    if bool(
                        (r.get("code_edit_validation") or {}).get("patch_with_tests")
                    )
                )
                / max(1, len(code_edit_rows))
            )
            * 100,
            2,
        )
        if code_edit_rows
        else 0.0
    )
    code_edit_generated_tests_quality_rate = (
        round(
            (
                sum(
                    1
                    for r in code_edit_rows
                    if bool(
                        (r.get("code_edit_validation") or {}).get("tests_quality_ok")
                    )
                )
                / max(1, len(code_edit_rows))
            )
            * 100,
            2,
        )
        if code_edit_rows
        else 0.0
    )

    rollback_failures = [
        e
        for e in failure_events
        if (e.get("payload") or {}).get("category") == "rollback"
    ]
    rollback_rate = round(
        (len(rollback_failures) / max(1, len(code_edit_rows))) * 100, 2
    )
    generation_quality_score = (
        round(
            sum(
                float((r.get("generation_quality") or {}).get("score", 0.0))
                for r in code_edit_rows
            )
            / max(1, len(code_edit_rows)),
            2,
        )
        if code_edit_rows
        else 0.0
    )

    return {
        "measured_at": datetime.now(UTC).isoformat(),
        "success_rate": success_rate,
        "json_failure_rate": json_failure_rate,
        "avg_task_seconds": avg_task_time,
        "pr_ready_usability": pr_ready_usability,
        "track_summary": track_summary,
        "code_edit_patch_signal_score": code_edit_patch_score,
        "code_edit_valid_patch_score": code_edit_valid_patch_score,
        "code_edit_patch_with_tests_rate": code_edit_patch_with_tests_rate,
        "code_edit_generated_tests_quality_rate": code_edit_generated_tests_quality_rate,
        "code_edit_generation_quality_score": generation_quality_score,
        "rollback_rate_per_code_edit_task": rollback_rate,
        "counts": {
            "eval_total": total,
            "eval_passed": passed,
            "planner_events": len(planner_events),
            "parse_failures": len(parse_failures),
            "pr_ready_events": len(pr_ready_events),
            "parse_failures_by_class": parse_by_class,
            "parse_failures_by_prompt_class": parse_by_prompt_class,
        },
    }


def _run_pr_ready_probe() -> dict:
    probe_dir = os.path.join(PROJECT_ROOT, ".stella")
    os.makedirs(probe_dir, exist_ok=True)
    probe_path = os.path.join(probe_dir, "eval_pr_probe.txt")
    created = False
    try:
        with open(probe_path, "a", encoding="utf-8") as f:
            f.write(f"probe={datetime.now(UTC).isoformat()}\n")
        created = True
        return prepare_pr(
            goal="kpi-pr-ready-probe",
            branch="agent/kpi-pr-ready-probe",
            commit_message="chore(agent): kpi pr-ready probe",
            dry_run=True,
            quick_validate=True,
        )
    except Exception as exc:
        return {"ok": False, "summary": f"pr-ready probe failed: {exc}"}
    finally:
        if created and os.path.exists(probe_path):
            try:
                os.remove(probe_path)
            except OSError:
                pass


def run_eval(
    tasks_file: str = "eval/tasks.json",
    max_tasks: int | None = None,
    min_generation_quality: float | None = None,
):
    abs_tasks = os.path.join(PROJECT_ROOT, tasks_file)
    with open(abs_tasks, "r", encoding="utf-8-sig") as f:
        tasks = json.load(f)
    if isinstance(max_tasks, int) and max_tasks > 0:
        tasks = tasks[:max_tasks]

    index_project()

    results = []
    start_all = time.time()
    start_ts = datetime.now(UTC)
    for task in tasks:
        name = task["name"]
        mode = task["mode"]
        prompt = task["prompt"]
        checks = task.get("must_contain_any", [])
        track = _track_for_task(task)
        before_changed = set(changed_files())

        t0 = time.time()
        if mode == "ask":
            output = ask_project(prompt)
        elif mode == "auto":
            output = AutonomousAgent(max_steps=6).run(prompt, auto_apply=False)
        elif mode == "code_edit":
            output = AutonomousAgent(max_steps=_task_int(task, "max_steps", 8)).run(
                prompt,
                auto_apply=_task_bool(task, "auto_apply", True),
                fix_until_green=_task_bool(task, "fix_until_green", True),
                generate_tests=_task_bool(task, "with_tests", True),
                max_seconds=_task_int(task, "max_seconds", 0),
            )
        else:
            output = f"unsupported mode: {mode}"

        dt = time.time() - t0
        after_changed = set(changed_files())
        changed_delta = sorted(after_changed - before_changed)
        base_passed = _score(output, checks)
        row = {
            "name": name,
            "mode": mode,
            "track": track,
            "duration_s": round(dt, 2),
            "output": output[:1000],
        }
        if track == "code_edit":
            signals = _code_edit_signals(output)
            validation = _compute_code_edit_validation(task, output, signals)
            row["code_edit_signals"] = signals
            row["code_edit_validation"] = validation
            requires_green = _task_bool(
                task, "require_tests_green", _task_bool(task, "auto_apply", False)
            )
            requires_tests = _task_bool(task, "require_patch_with_tests", False)
            requires_quality = _task_bool(
                task, "require_generated_tests_quality", False
            )
            strict_scope = _task_bool(task, "strict_scope", False)
            has_unexpected = bool(validation.get("unexpected_touched_paths"))
            touched_for_quality = sorted(
                set(changed_delta or validation.get("touched_paths", []))
            )
            row["touched_files"] = touched_for_quality
            row["generation_quality"] = assess_generated_files(touched_for_quality)

            min_quality = (
                float(min_generation_quality)
                if min_generation_quality is not None
                else float(task.get("min_generation_quality", 65.0))
            )
            row["passed"] = bool(
                base_passed
                and validation["expected_diff_match"]
                and (not strict_scope or not has_unexpected)
                and (not requires_green or validation["tests_green"])
                and (not requires_tests or validation["patch_with_tests"])
                and (not requires_quality or validation["tests_quality_ok"])
                and (row["generation_quality"]["score"] >= min_quality)
                and (
                    requires_green
                    or validation["patch_with_tests"]
                    or bool(signals.get("has_validation_action"))
                )
            )
        else:
            row["passed"] = base_passed

        results.append(row)

    duration = round(time.time() - start_all, 2)
    probe = _run_pr_ready_probe()
    end_ts = datetime.now(UTC)
    pass_count = sum(1 for r in results if r["passed"])
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total": len(results),
        "passed": pass_count,
        "pass_rate": round((pass_count / len(results)) * 100, 2) if results else 0.0,
        "duration_s": duration,
        "pr_ready_probe_ok": bool(probe.get("ok")),
        "tracks": _summarize_tracks(results),
    }

    report = {"summary": summary, "results": results, "pr_ready_probe": probe}
    out_path = os.path.join(PROJECT_ROOT, "eval", "last_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    kpis = _compute_kpis(report, start_ts=start_ts, end_ts=end_ts)
    kpi_path = os.path.join(PROJECT_ROOT, ".stella", "kpi_snapshot.json")
    os.makedirs(os.path.dirname(kpi_path), exist_ok=True)
    with open(kpi_path, "w", encoding="utf-8") as f:
        json.dump(kpis, f, ensure_ascii=False, indent=2)

    report["kpis"] = kpis
    return report
