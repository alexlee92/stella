"""
P2.1 — Action executor extracted from AutonomousAgent.

Handles the execution of individual agent actions (read, edit, test, etc.).
"""

import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional

from agent.agent import apply_suggestion, patch_risk, propose_file_update
from agent.config import AUTO_TEST_COMMAND, PROJECT_ROOT
from agent.decision_normalizer import (
    extract_all_target_files_from_goal,
    is_allowed_edit_path,
    is_code_edit_goal,
    extract_target_file_from_goal,
)
from agent.deps import detect_and_install_deps
from agent.git_tools import commit_all, create_branch, diff_summary
from agent.memory import index_file_in_session, remember_fix_strategy
from agent.patcher import apply_transaction, restore_backup, rollback_transaction
from agent.project_map import render_project_map
from agent.quality import (
    run_quality_gate,
    run_quality_gate_normalized,
    run_quality_pipeline_normalized,
)
from agent.test_generator import apply_generated_tests
from agent.test_selector import suggest_test_path
from agent.tooling import (
    list_files,
    read_file,
    read_many,
    run_tests,
    search_code,
    write_new_file,
)

if TYPE_CHECKING:
    from agent.auto_agent import AutonomousAgent


class ActionExecutor:
    """Executes a single agent action and returns the result string."""

    def __init__(self, agent: "AutonomousAgent"):
        self.agent = agent

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self, action: str, args: dict, goal: str, auto_apply: bool, index: int) -> str:
        """Execute *action* with *args* and return a result string.

        Returns a result string. May modify agent state (staged_edits, forced_decisions, etc.).
        Raises on unexpected errors (caller should catch).
        """
        handler = getattr(self, f"_exec_{action}", None)
        if handler is None:
            self.agent.event_logger.log_failure(
                "tool", "unknown_action", {"action": action, "args": args}
            )
            return f"Unknown action: {action}"
        return handler(args, goal=goal, auto_apply=auto_apply, index=index)

    # ------------------------------------------------------------------
    # Individual action handlers
    # ------------------------------------------------------------------

    def _exec_list_files(self, args: dict, **_kw) -> str:
        out = list_files(
            limit=int(args.get("limit", 50)),
            contains=args.get("contains", ""),
            ext=args.get("ext", ".py"),
        )
        return "\n".join(out) if out else "No files found"

    def _exec_read_file(self, args: dict, **_kw) -> str:
        return read_file(args.get("path", ""))

    def _exec_read_many(self, args: dict, **_kw) -> str:
        paths = args.get("paths", [])
        return read_many(paths) if isinstance(paths, list) and paths else "Missing paths list"

    def _exec_search_code(self, args: dict, **_kw) -> str:
        pattern = args.get("pattern", "")
        limit = int(args.get("limit", 20))
        matches = search_code(pattern, limit=limit) if pattern else []
        return "\n".join(matches) if matches else "No matches"

    def _exec_create_file(self, args: dict, goal: str, **_kw) -> str:
        ag = self.agent
        path = args.get("path", "")
        description = args.get("description", "")
        if not path or not description:
            return "[error] create_file requires 'path' and 'description'"
        if path in ag.staged_edits:
            return f"Skipped: {path} already created this session"
        if os.path.isfile(os.path.join(PROJECT_ROOT, path)):
            result = f"Skipped: {path} already exists on disk (use fix to modify it)"
            all_targets = extract_all_target_files_from_goal(goal)
            if not all_targets and not ag._forced_decisions:
                ag._forced_decisions.insert(0, {
                    "action": "finish",
                    "reason": "all_target_files_already_exist",
                    "args": {"summary": f"All target files already exist on disk: {path}"},
                })
            return result

        print(f"     génère le contenu de {path}...", flush=True)
        content = ag._generate_file_content(goal, path, description)
        if not content:
            return f"[error] LLM returned empty content for {path}"

        write_result = write_new_file(path, content)
        if not write_result.startswith("ok:"):
            return f"[error] write failed: {write_result}"

        ag.staged_edits[path] = content
        risk = patch_risk(path, content)
        n_chunks = index_file_in_session(path, content)
        deps_result = detect_and_install_deps(path, content)
        result = (
            f"Created {path} ({len(content)} chars), "
            f"risk={risk['level']}:{risk['score']}, "
            f"indexed={n_chunks} chunks, "
            f"{deps_result}"
        )

        # Chain next files from goal
        all_targets = extract_all_target_files_from_goal(goal)
        pending = [t for t in all_targets if t not in ag.staged_edits]
        if pending:
            next_path = pending[0]
            ag._forced_decisions.insert(0, {
                "action": "create_file",
                "reason": "multi_file_goal_continuation",
                "args": {
                    "path": next_path,
                    "description": (
                        f"{goal[:400]} "
                        f"(already created: {', '.join(ag.staged_edits.keys())})"
                    ),
                },
            })
        else:
            created = ", ".join(ag.staged_edits.keys())
            ag._forced_decisions.insert(0, {
                "action": "finish",
                "reason": "all_files_created",
                "args": {"summary": f"Created: {created}"},
            })
        return result

    def _exec_propose_edit(self, args: dict, goal: str, **_kw) -> str:
        ag = self.agent
        path = args.get("path", "")
        instruction = args.get("instruction", "")
        if is_code_edit_goal(goal) and not is_allowed_edit_path(goal, path):
            target = extract_target_file_from_goal(goal)
            if target:
                path = target
                instruction = f"{instruction}\nKeep changes strictly in {target}."
        code = propose_file_update(path, instruction, k=ag.top_k)
        ag.staged_edits[path] = code
        risk = patch_risk(path, code)
        result = f"Staged edit for {path} ({len(code)} chars), risk={risk['level']}:{risk['score']}"
        auto_apply = _kw.get("auto_apply", False)
        if ag._fix_until_green and auto_apply and path:
            ag._forced_decisions = [
                {"action": "apply_edit", "reason": "fix_until_green_apply_staged", "args": {"path": path}},
                {"action": "run_quality", "reason": "fix_until_green_verify_patch", "args": {}},
            ] + ag._forced_decisions
        elif ag._generate_tests and path.endswith(".py"):
            test_path = suggest_test_path(path)
            if test_path not in ag.staged_edits:
                ag._forced_decisions = [
                    {
                        "action": "propose_edit",
                        "reason": "with_tests_auto_target",
                        "args": {
                            "path": test_path,
                            "instruction": (
                                f"Add focused pytest tests for {path} with one nominal case "
                                "and one edge case."
                            ),
                        },
                    }
                ] + ag._forced_decisions
        return result

    def _exec_apply_edit(self, args: dict, goal: str, auto_apply: bool, index: int, **_kw) -> str:
        ag = self.agent
        path = args.get("path", "")
        code = ag.staged_edits.get(path)
        if not path or code is None:
            ag.event_logger.log_failure(
                "tool", "apply_edit_without_staged",
                {"path": path, "staged_files": list(ag.staged_edits.keys())},
            )
            return "[no_staged_edit]"

        if not auto_apply or (ag.interactive and not ag._confirm_apply([path])):
            return f"Staged only for {path}. Re-run with --apply to apply."

        patch_result = apply_suggestion(path, code, interactive=False)
        backup_path = patch_result.get("backup_path") if isinstance(patch_result, dict) else None
        changed_scope = [path]

        result = ""
        if ag._generate_tests and path.endswith(".py"):
            gen = apply_generated_tests([path], limit=2)
            generated = gen.get("applied", [])
            if generated:
                changed_scope.extend(generated)
                ag.event_logger.log("generated_tests", {
                    "source_files": [path],
                    "generated_files": generated,
                    "quality_ok_rate": gen.get("quality_ok_rate", 0.0),
                })
                result = (
                    f"Applied {path}; generated_tests={','.join(generated)}; "
                    f"tests_quality_ok_rate={gen.get('quality_ok_rate', 0.0)}"
                )

        ok, details = run_quality_gate(changed_files=changed_scope, command_timeout=90)
        if ok:
            if result:
                result += "; quality gate passed"
            else:
                result = f"Applied {path}; quality gate passed"
        else:
            ag.event_logger.log_failure(
                "test", "quality_gate_failed_after_apply_edit",
                {"path": path, "details": details},
            )
            if ag._fix_until_green:
                normalized = run_quality_pipeline_normalized(
                    mode="fast", changed_files=[path], command_timeout=90,
                )
                failure_text = json.dumps(normalized or {"rows": []}, ensure_ascii=False)
                scheduled = ag._enqueue_replan_after_failure(
                    "quality", failure_text, auto_apply=auto_apply, fallback_path=path,
                )
                if not scheduled:
                    ag._forced_decisions.insert(0, {
                        "action": "finish",
                        "reason": "stop_max_replan_attempts",
                        "args": {"summary": "Stopped: max replan attempts reached after quality failures"},
                    })
                result = f"Applied {path}; quality gate failed; replan queued in fix-until-green mode"
            else:
                restore_ok = restore_backup(path, backup_path)
                if not restore_ok:
                    ag.event_logger.log_failure(
                        "rollback", "rollback_failed_after_apply_edit",
                        {"path": path, "backup_path": backup_path},
                    )
                result = f"Applied {path}; quality gate failed; rollback={'ok' if restore_ok else 'failed'}"
        return result

    def _exec_apply_all_staged(self, args: dict, goal: str, auto_apply: bool, **_kw) -> str:
        ag = self.agent
        if not ag.staged_edits:
            return "No staged edits"
        if not auto_apply or (ag.interactive and not ag._confirm_apply(list(ag.staged_edits.keys()))):
            return "Staged only. Re-run with --apply to apply transaction."

        success, backups, msg = apply_transaction(ag.staged_edits)
        if not success:
            ag.event_logger.log_failure("tool", "apply_transaction_failed", {"message": msg})
            return msg

        changed = list(ag.staged_edits.keys())
        changed_scope = list(changed)
        result = ""

        if ag._generate_tests:
            gen = apply_generated_tests(changed, limit=3)
            generated = gen.get("applied", [])
            if generated:
                changed_scope.extend(generated)
                ag.event_logger.log("generated_tests", {
                    "source_files": changed,
                    "generated_files": generated,
                    "quality_ok_rate": gen.get("quality_ok_rate", 0.0),
                })
                result = (
                    "Transaction applied; generated_tests="
                    + ",".join(generated)
                    + f"; tests_quality_ok_rate={gen.get('quality_ok_rate', 0.0)}"
                )

        ok, details = run_quality_gate(changed_files=changed_scope, command_timeout=90)
        if ok:
            ag.last_backups = backups
            if result:
                result += "; quality gate passed"
            else:
                result = "Transaction applied; quality gate passed"
        else:
            ag.event_logger.log_failure(
                "test", "quality_gate_failed_after_transaction", {"details": details},
            )
            if ag._fix_until_green:
                normalized = run_quality_gate_normalized(changed_files=changed, command_timeout=90)
                scheduled = ag._enqueue_replan_after_failure(
                    "quality", json.dumps(normalized, ensure_ascii=False),
                    auto_apply=auto_apply,
                    fallback_path=changed[0] if changed else None,
                )
                if not scheduled:
                    ag._forced_decisions.insert(0, {
                        "action": "finish",
                        "reason": "stop_max_replan_attempts",
                        "args": {"summary": "Stopped: max replan attempts reached after quality failures"},
                    })
                result = "Transaction applied then failed quality gate; replan queued in fix-until-green mode"
            else:
                rb = rollback_transaction(backups)
                if not rb:
                    ag.event_logger.log_failure("rollback", "rollback_transaction_failed", {})
                result = f"Transaction applied then failed quality gate; rollback={'ok' if rb else 'failed'}"
        return result

    def _exec_run_tests(self, args: dict, auto_apply: bool = False, **_kw) -> str:
        ag = self.agent
        ag._has_validation_action = True
        command = args.get("command", AUTO_TEST_COMMAND)
        result = run_tests(command)
        if "exit_code=0" not in result:
            ag.event_logger.log_failure(
                "test", "run_tests_non_zero",
                {"command": command, "result": result[:500]},
            )
            scheduled = ag._enqueue_replan_after_failure("tests", result, auto_apply=auto_apply)
            if not scheduled:
                ag._forced_decisions.insert(0, {
                    "action": "finish",
                    "reason": "stop_max_replan_attempts",
                    "args": {"summary": "Stopped: max replan attempts reached after test failures"},
                })
        elif ag._replan_attempts > 0:
            recent = " | ".join(f"{s.action}:{s.reason}" for s in ag.steps[-6:])
            remember_fix_strategy(
                issue=f"tests failure recovered for goal: {_kw.get('goal', '')[:120]}",
                strategy=recent,
                files=list(ag.staged_edits.keys())[:6],
            )
        return result

    def _exec_run_quality(self, args: dict, auto_apply: bool = False, goal: str = "", **_kw) -> str:
        ag = self.agent
        ag._has_validation_action = True
        changed_scope = list(ag.staged_edits.keys()) or None
        normalized = run_quality_pipeline_normalized(
            mode="fast" if changed_scope else "full",
            changed_files=changed_scope,
            command_timeout=90,
        )
        ok = bool(normalized.get("ok"))
        details = normalized.get("raw", [])
        result = f"ok={ok}; failed_stage={normalized.get('failed_stage')}; details={details}"
        if not ok:
            ag.event_logger.log_failure("test", "run_quality_failed", {"details": normalized})
            stage = normalized.get("failed_stage") or "quality"
            scheduled = ag._enqueue_replan_after_failure(
                str(stage), json.dumps(normalized, ensure_ascii=False), auto_apply=auto_apply,
            )
            if not scheduled:
                ag._forced_decisions.insert(0, {
                    "action": "finish",
                    "reason": "stop_max_replan_attempts",
                    "args": {"summary": "Stopped: max replan attempts reached after quality failures"},
                })
        elif ag._replan_attempts > 0:
            recent = " | ".join(f"{s.action}:{s.reason}" for s in ag.steps[-6:])
            remember_fix_strategy(
                issue=f"quality failure recovered for goal: {goal[:120]}",
                strategy=recent,
                files=list(ag.staged_edits.keys())[:6],
            )
        return result

    def _exec_project_map(self, args: dict, **_kw) -> str:
        return render_project_map()

    def _exec_git_branch(self, args: dict, **_kw) -> str:
        code, out = create_branch(args.get("name", "feature/agent-change"))
        return f"exit={code}; {out[:600]}"

    def _exec_git_commit(self, args: dict, **_kw) -> str:
        code, out = commit_all(args.get("message", "chore: agent update"))
        return f"exit={code}; {out[:600]}"

    def _exec_git_diff(self, args: dict, **_kw) -> str:
        return diff_summary()

    def _exec_finish(self, args: dict, **_kw) -> str:
        return args.get("summary") or "Task finished"
