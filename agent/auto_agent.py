import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from agent.action_schema import validate_critique_schema, validate_decision_schema
from agent.agent import apply_suggestion, patch_risk, propose_file_update
from agent.config import AUTO_TEST_COMMAND, MAX_RETRIES_JSON, TOP_K_RESULTS
from agent.event_logger import EventLogger
from agent.git_tools import commit_all, create_branch, diff_summary
from agent.llm_interface import ask_llm_json
from agent.memory import search_memory
from agent.patcher import apply_transaction, restore_backup, rollback_transaction
from agent.project_map import render_project_map
from agent.quality import run_quality_gate, run_quality_pipeline
from agent.tooling import (
    list_files,
    list_python_files,
    read_file,
    read_many,
    run_tests,
    search_code,
)


@dataclass
class AutoStep:
    step: int
    action: str
    reason: str
    result: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AutonomousAgent:
    def __init__(
        self,
        top_k: int = TOP_K_RESULTS,
        max_steps: int = 8,
        logger: Optional[Callable[[dict], None]] = None,
    ):
        self.top_k = top_k
        self.max_steps = max_steps
        self.logger = logger
        self.steps: List[AutoStep] = []
        self.staged_edits: Dict[str, str] = {}
        self.last_backups = []
        self.event_logger = EventLogger()
        self._decision_signatures: List[str] = []
        self._outcome_signatures: List[str] = []
        self._parse_fallback_count = 0

    def _summarize_context(self, goal: str) -> str:
        docs = search_memory(goal, k=self.top_k)
        if not docs:
            return "No indexed context found"

        chunks = []
        for path, content in docs:
            chunks.append(f"FILE: {path}\n{content[:900]}")
        return "\n\n".join(chunks)

    def _planner_prompt(self, goal: str) -> str:
        history = (
            "\n".join(
                f"{s.step}. {s.action} | reason={s.reason} | result={s.result[:180]}"
                for s in self.steps[-8:]
            )
            or "none"
        )

        files = list_python_files(limit=40)
        files_text = "\n".join(files) if files else "no python files"
        context = self._summarize_context(goal)

        return f"""
You are an autonomous coding agent.
Goal: {goal}

You can use one action at a time:
- list_files: {{"contains": "optional substring", "ext": ".py", "limit": 50}}
- read_file: {{"path": "relative/path.py"}}
- read_many: {{"paths": ["a.py", "b.py"]}}
- search_code: {{"pattern": "regex or text", "limit": 20}}
- propose_edit: {{"path": "relative/path.py", "instruction": "change request"}}
- apply_edit: {{"path": "relative/path.py"}}
- apply_all_staged: {{}}
- run_tests: {{"command": "pytest -q"}}
- run_quality: {{}}
- project_map: {{}}
- git_branch: {{"name": "feature/xyz"}}
- git_commit: {{"message": "feat: ..."}}
- git_diff: {{}}
- finish: {{"summary": "final answer for user"}}

Rules:
- Return strict JSON only.
- Use only listed actions and valid args.
- Keep edits minimal and safe.
- Prefer read/search/propose/apply actions first.
- Use git actions only when the goal explicitly asks for git/commit/pr operations.

Return format:
{{"action":"...","reason":"short reason","args":{{...}}}}

Project files:
{files_text}

Relevant indexed context:
{context}

Recent steps:
{history}
"""

    def _extract_target_file_from_goal(self, goal: str) -> Optional[str]:
        low = goal.strip()
        m = re.search(r"([A-Za-z0-9_./\\-]+\.py)", low)
        if not m:
            return None
        path = m.group(1).replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]
        return path

    def _is_git_goal(self, goal: str) -> bool:
        low = (goal or "").lower()
        return any(k in low for k in ["git", "commit", "branch", "pr", "pull request"])

    def _coerce_decision(self, goal: str, decision: dict) -> dict:
        action = decision.get("action", "")
        args = decision.get("args", {}) or {}

        if action.startswith("git_") and not self._is_git_goal(goal):
            return {
                "action": "search_code",
                "reason": "coerced_non_git_goal",
                "args": {
                    "pattern": "requests.post|subprocess.run|timeout|sleep",
                    "limit": 20,
                },
            }

        if action == "list_files" and args.get("contains"):
            args = dict(args)
            args["contains"] = ""
            return {
                "action": "list_files",
                "reason": "coerced_broad_listing",
                "args": args,
            }

        return decision

    def _critique_prompt(self, goal: str, decision: dict) -> str:
        return f"""
You are reviewing a planner decision for safety and schema correctness.
Goal: {goal}
Decision:
{json.dumps(decision, ensure_ascii=False)}

Return strict JSON:
{{
  "approve": true|false,
  "reason": "short reason",
  "patched_decision": {{"action":"...","reason":"...","args":{{...}}}} or null
}}

Rules:
- If decision is valid and useful, approve=true.
- If decision is invalid or risky, approve=false and provide patched_decision when possible.
- Never output markdown.
"""

    def _record(self, step: AutoStep):
        self.steps.append(step)
        record = {
            "type": "decision",
            "step": step.step,
            "action": step.action,
            "reason": step.reason,
            "result": step.result,
            "timestamp": step.timestamp,
        }
        if self.logger:
            self.logger(record)
        self.event_logger.log("decision", record)

    def _signature(self, action: str, args: dict, result: str = "") -> str:
        return json.dumps(
            {"a": action, "g": args, "r": result[:200]},
            sort_keys=True,
            ensure_ascii=False,
        )

    def _decision_loop_detected(self, action: str, args: dict) -> bool:
        sig = self._signature(action, args)
        return self._decision_signatures[-4:].count(sig) >= 2

    def _outcome_loop_detected(self, action: str, args: dict, result: str) -> bool:
        sig = self._signature(action, args, result)
        return self._outcome_signatures[-4:].count(sig) >= 2

    def _plan_with_critique(self, goal: str) -> dict:
        decision = ask_llm_json(self._planner_prompt(goal), retries=MAX_RETRIES_JSON)
        self.event_logger.log("plan", {"goal": goal, "decision": decision})

        if decision.get("_error_type") == "parse":
            self.event_logger.log_failure(
                "parse", "planner_json_parse_failed", {"decision": decision}
            )
            self._parse_fallback_count += 1
            return self._fallback_decision(goal, reason="parse_failed")
        self._parse_fallback_count = 0

        decision = self._coerce_decision(goal, decision)
        ok, msg = validate_decision_schema(decision)
        if not ok:
            self.event_logger.log_failure(
                "parse", f"planner_schema_invalid:{msg}", {"decision": decision}
            )
            return self._fallback_decision(goal, reason=f"schema_invalid:{msg}")

        critique = ask_llm_json(
            self._critique_prompt(goal, decision), retries=MAX_RETRIES_JSON
        )
        c_ok, c_msg = validate_critique_schema(critique)
        if not c_ok:
            self.event_logger.log_failure(
                "parse", f"critique_schema_invalid:{c_msg}", {"critique": critique}
            )
            return decision

        if critique.get("approve"):
            return decision

        patched = critique.get("patched_decision")
        if isinstance(patched, dict):
            patched = self._coerce_decision(goal, patched)
            p_ok, p_msg = validate_decision_schema(patched)
            if p_ok:
                self.event_logger.log(
                    "critique_patch",
                    {"reason": critique.get("reason"), "patched": patched},
                )
                return patched
            self.event_logger.log_failure(
                "parse", f"patched_schema_invalid:{p_msg}", {"patched": patched}
            )

        return {
            "action": "finish",
            "reason": "critique_rejected",
            "args": {
                "summary": f"Decision rejected by critique: {critique.get('reason', 'n/a')}"
            },
        }

    def _fallback_decision(self, goal: str, reason: str) -> dict:
        target = self._extract_target_file_from_goal(goal)
        if target:
            return {
                "action": "propose_edit",
                "reason": f"fallback_{reason}",
                "args": {"path": target, "instruction": goal[:500]},
            }

        low = (goal or "").lower()
        if self._parse_fallback_count >= 3:
            return {
                "action": "finish",
                "reason": f"fallback_{reason}",
                "args": {
                    "summary": "Planner parse unstable after 3 retries; stopping early"
                },
            }

        if any(k in low for k in ["latence", "performance", "vitesse", "speed"]):
            if self._parse_fallback_count % 2 == 1:
                return {
                    "action": "search_code",
                    "reason": f"fallback_{reason}",
                    "args": {
                        "pattern": "requests.post|subprocess.run|timeout|sleep",
                        "limit": 20,
                    },
                }
            return {
                "action": "read_many",
                "reason": f"fallback_{reason}",
                "args": {
                    "paths": [
                        "agent/llm_interface.py",
                        "agent/auto_agent.py",
                        "agent/tooling.py",
                    ]
                },
            }

        if any(k in low for k in ["pr", "commit", "git", "branch"]):
            if self._parse_fallback_count % 2 == 1:
                return {
                    "action": "git_diff",
                    "reason": f"fallback_{reason}",
                    "args": {},
                }
            return {
                "action": "read_many",
                "reason": f"fallback_{reason}",
                "args": {
                    "paths": [
                        "agent/pr_ready.py",
                        "agent/git_tools.py",
                        "agent/quality.py",
                    ]
                },
            }

        return {
            "action": "list_files",
            "reason": f"fallback_{reason}",
            "args": {"limit": 40, "ext": ".py"},
        }

    def plan_once(self, goal: str) -> dict:
        return self._plan_with_critique(goal)

    def run(self, goal: str, auto_apply: bool = False) -> str:
        for index in range(1, self.max_steps + 1):
            decision = self._plan_with_critique(goal)
            action = decision.get("action", "")
            reason = decision.get("reason", "")
            args = decision.get("args", {}) or {}

            if self._decision_loop_detected(action, args):
                self.event_logger.log_failure(
                    "tool", "decision_loop_detected", {"action": action, "args": args}
                )
                self._record(
                    AutoStep(
                        index,
                        "loop_guard",
                        "repeat_decision",
                        "Stopped repeated decision loop",
                    )
                )
                summary = "Stopped repeated decision loop; recommend narrowing goal to a concrete file-level change request"
                self._record(AutoStep(index, "finish", "auto_stop_repeat", summary))
                return self.render_summary(summary)

            self._decision_signatures.append(self._signature(action, args))
            step = AutoStep(index, action, reason, "")

            try:
                if action == "list_files":
                    out = list_files(
                        limit=int(args.get("limit", 50)),
                        contains=args.get("contains", ""),
                        ext=args.get("ext", ".py"),
                    )
                    step.result = "\n".join(out) if out else "No files found"

                elif action == "read_file":
                    step.result = read_file(args.get("path", ""))

                elif action == "read_many":
                    paths = args.get("paths", [])
                    step.result = (
                        read_many(paths)
                        if isinstance(paths, list) and paths
                        else "Missing paths list"
                    )

                elif action == "search_code":
                    pattern = args.get("pattern", "")
                    limit = int(args.get("limit", 20))
                    matches = search_code(pattern, limit=limit) if pattern else []
                    step.result = "\n".join(matches) if matches else "No matches"

                elif action == "propose_edit":
                    path = args.get("path", "")
                    instruction = args.get("instruction", "")
                    code = propose_file_update(path, instruction, k=self.top_k)
                    self.staged_edits[path] = code
                    risk = patch_risk(path, code)
                    step.result = f"Staged edit for {path} ({len(code)} chars), risk={risk['level']}:{risk['score']}"

                elif action == "apply_edit":
                    path = args.get("path", "")
                    code = self.staged_edits.get(path)
                    if not path or code is None:
                        self.event_logger.log_failure(
                            "tool",
                            "apply_edit_without_staged",
                            {
                                "path": path,
                                "staged_files": list(self.staged_edits.keys()),
                            },
                        )
                        step.result = "No staged edit for this file"
                        self._record(step)
                        self._record(
                            AutoStep(
                                index,
                                "finish",
                                "invalid_action_sequence",
                                "apply_edit requested without staged edit; stopping",
                            )
                        )
                        return self.render_summary(
                            "apply_edit requested without staged edit; stopping"
                        )
                    elif not auto_apply:
                        step.result = (
                            f"Staged only for {path}. Re-run with --apply to apply."
                        )
                    else:
                        patch_result = apply_suggestion(path, code, interactive=False)
                        backup_path = (
                            patch_result.get("backup_path")
                            if isinstance(patch_result, dict)
                            else None
                        )

                        ok, details = run_quality_gate(changed_files=[path])
                        if ok:
                            step.result = f"Applied {path}; quality gate passed"
                        else:
                            self.event_logger.log_failure(
                                "test",
                                "quality_gate_failed_after_apply_edit",
                                {"path": path, "details": details},
                            )
                            restore_ok = restore_backup(path, backup_path)
                            if not restore_ok:
                                self.event_logger.log_failure(
                                    "rollback",
                                    "rollback_failed_after_apply_edit",
                                    {"path": path, "backup_path": backup_path},
                                )
                            step.result = f"Applied {path}; quality gate failed; rollback={'ok' if restore_ok else 'failed'}"

                elif action == "apply_all_staged":
                    if not self.staged_edits:
                        step.result = "No staged edits"
                    elif not auto_apply:
                        step.result = (
                            "Staged only. Re-run with --apply to apply transaction."
                        )
                    else:
                        success, backups, msg = apply_transaction(self.staged_edits)
                        if not success:
                            self.event_logger.log_failure(
                                "tool", "apply_transaction_failed", {"message": msg}
                            )
                            step.result = msg
                        else:
                            changed = list(self.staged_edits.keys())
                            ok, details = run_quality_gate(changed_files=changed)
                            if ok:
                                self.last_backups = backups
                                step.result = "Transaction applied; quality gate passed"
                            else:
                                self.event_logger.log_failure(
                                    "test",
                                    "quality_gate_failed_after_transaction",
                                    {"details": details},
                                )
                                rb = rollback_transaction(backups)
                                if not rb:
                                    self.event_logger.log_failure(
                                        "rollback", "rollback_transaction_failed", {}
                                    )
                                step.result = f"Transaction applied then failed quality gate; rollback={'ok' if rb else 'failed'}"

                elif action == "run_tests":
                    command = args.get("command", AUTO_TEST_COMMAND)
                    step.result = run_tests(command)
                    if "exit_code=0" not in step.result:
                        self.event_logger.log_failure(
                            "test",
                            "run_tests_non_zero",
                            {"command": command, "result": step.result[:500]},
                        )

                elif action == "run_quality":
                    ok, details = run_quality_pipeline(mode="full")
                    step.result = f"ok={ok}; details={details}"
                    if not ok:
                        self.event_logger.log_failure(
                            "test", "run_quality_failed", {"details": details}
                        )

                elif action == "project_map":
                    step.result = render_project_map()

                elif action == "git_branch":
                    code, out = create_branch(args.get("name", "feature/agent-change"))
                    step.result = f"exit={code}; {out[:600]}"

                elif action == "git_commit":
                    code, out = commit_all(args.get("message", "chore: agent update"))
                    step.result = f"exit={code}; {out[:600]}"
                    if "nothing to commit" in out.lower():
                        recent_nothing = any(
                            s.action == "git_commit"
                            and "nothing to commit" in s.result.lower()
                            for s in self.steps[-2:]
                        )
                        if recent_nothing:
                            self._record(step)
                            self._record(
                                AutoStep(
                                    index,
                                    "finish",
                                    "auto_stop_redundant_commit",
                                    "No more changes to commit; stopping redundant git_commit loop",
                                )
                            )
                            return self.render_summary(
                                "No more changes to commit; stopping redundant git_commit loop"
                            )

                elif action == "git_diff":
                    step.result = diff_summary()

                elif action == "finish":
                    step.result = args.get("summary") or "Task finished"
                    self._record(step)
                    return self.render_summary(step.result)

                else:
                    self.event_logger.log_failure(
                        "tool", "unknown_action", {"action": action, "args": args}
                    )
                    step.action = "unknown_action"
                    step.result = f"Unknown action: {action}"

            except Exception as exc:
                self.event_logger.log_failure(
                    "tool",
                    "tool_execution_exception",
                    {"action": action, "error": str(exc)},
                )
                step.result = f"Tool error: {exc}"

            if self._outcome_loop_detected(action, args, step.result):
                self._record(step)
                self._record(
                    AutoStep(
                        index,
                        "loop_guard",
                        "repeat_outcome",
                        "Stopped repeated outcome loop",
                    )
                )
                return self.render_summary("Stopped: repeated outcome loop")

            self._outcome_signatures.append(self._signature(action, args, step.result))
            self._record(step)

        return self.render_summary("Reached max steps without finish action")

    def render_summary(self, final_message: str) -> str:
        lines = [f"Final: {final_message}", "", "Decisions:"]
        for s in self.steps:
            lines.append(f"- step {s.step}: {s.action} | {s.reason} | {s.result[:180]}")
        return "\n".join(lines)
