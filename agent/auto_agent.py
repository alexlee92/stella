"""
Autonomous agent loop for Stella.
Handles planning, tool execution, and self-correction.
"""

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from agent.action_schema import (
    validate_critique_schema,
    validate_decision_schema,
)
from agent.agent import patch_risk
from agent.config import (
    MAX_RETRIES_JSON,
    PROJECT_ROOT,
    TOP_K_RESULTS,
    _CFG,
)
from agent.decision_normalizer import (
    autocorrect_decision_schema,
    coerce_decision,
    extract_all_target_files_from_goal,
    extract_target_file_from_goal,
    infer_fallback_action,
    is_code_edit_goal,
    is_allowed_edit_path,
    is_git_goal,
    normalize_critique,
    normalize_decision,
)
from agent.event_logger import EventLogger
from agent.executor import ActionExecutor
from agent.llm_interface import ask_llm_json
from agent.memory import search_memory
from agent.test_selector import suggest_test_path
from agent.tooling import list_python_files


@dataclass
class AutoStep:
    step: int
    action: str
    reason: str
    result: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class AutonomousAgent:
    """
    An agent that can run multiple steps to achieve a goal.
    It uses a Planner to decide on actions and a Critique to validate them.
    """

    def __init__(
        self,
        top_k: int = TOP_K_RESULTS,
        max_steps: int = 8,
        logger: Optional[Callable[[dict], None]] = None,
        interactive: bool = False,
        config: Optional[Dict] = None,
        llm_fn: Optional[Callable] = None,
        memory_fn: Optional[Callable] = None,
    ):
        # P2.5 — Dependency injection: accept config/llm/memory as arguments
        self.config = config or dict(_CFG)
        self._llm_fn = llm_fn or ask_llm_json
        self._memory_fn = memory_fn or search_memory
        self.top_k = top_k
        self.max_steps = max_steps
        self.logger = logger
        self.interactive = interactive
        self.steps: List[AutoStep] = []
        self.staged_edits: Dict[str, str] = {}
        self.last_backups = []
        self.event_logger = EventLogger()
        self._decision_signatures: List[str] = []
        self._outcome_signatures: List[str] = []
        self._parse_fallback_count = 0
        self._forced_decisions: List[dict] = []
        self._consecutive_stalled_steps = 0
        self._total_cost = 0
        self._max_cost = max_steps * 4
        self._replan_attempts = 0
        self._max_replan_attempts = 4
        self._fix_until_green = False
        self._generate_tests = False
        self._has_validation_action = False
        self._bootstrapped_code_edit = False
        self._executor = ActionExecutor(self)

    # P2.1 — Delegate to decision_normalizer module
    def _normalize_decision(self, decision: dict) -> dict:
        return normalize_decision(decision)

    def _normalize_critique(self, critique: dict) -> dict:
        return normalize_critique(critique)

    def _infer_fallback_action(self, goal: str, args: dict) -> tuple[str, dict]:
        return infer_fallback_action(goal, args)

    def _autocorrect_decision_schema(self, goal: str, decision: dict, msg: str) -> dict:
        return autocorrect_decision_schema(goal, decision, msg)

    def _confirm_apply(self, paths: List[str]) -> bool:
        """P1.2 — Demande confirmation à l'utilisateur avant d'appliquer des patches.

        En mode non-interactif, retourne toujours True.
        """
        if not self.interactive:
            return True
        files_list = ", ".join(paths[:5])
        if len(paths) > 5:
            files_list += f" (+{len(paths) - 5} autres)"
        try:
            answer = input(
                f"\n  [?] Appliquer les modifications sur {files_list} ? [y/n/d(iff)] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  [stella] Annulation.")
            return False
        if answer == "d":
            # Afficher les diffs pour chaque fichier
            from agent.patcher import show_diff
            from agent.project_scan import load_file_content
            for path in paths:
                code = self.staged_edits.get(path)
                if code is None:
                    continue
                abs_path = os.path.join(PROJECT_ROOT, path) if not os.path.isabs(path) else path
                try:
                    old_code = load_file_content(abs_path)
                except Exception:
                    old_code = ""
                show_diff(old_code, code, filepath=path)
            try:
                answer = input("  [?] Appliquer ? [y/n] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
        return answer in {"y", "yes", "o", "oui"}

    def _session_context(self, max_chars_per_file: int = 2000) -> str:
        """Retourne le contenu des fichiers créés/modifiés dans cette session."""
        if not self.staged_edits:
            return ""
        chunks = []
        for path, content in self.staged_edits.items():
            preview = (content or "")[:max_chars_per_file]
            chunks.append(f"FILE (this session): {path}\n{preview}")
        return "\n\n".join(chunks)

    def _summarize_context(self, goal: str) -> str:
        docs = self._memory_fn(goal, k=self.top_k)
        if not docs:
            return "No indexed context found"

        chunks = []
        for path, content in docs:
            chunks.append(f"FILE: {path}\n{content[:900]}")
        return "\n\n".join(chunks)

    def _summarize_context_multi(
        self, goal: str, path: str = "", description: str = ""
    ) -> str:
        """Recherche de contexte multi-angle : goal + chemin + description.

        Combine les résultats des 3 requêtes et déduplique par chemin de fichier.
        Inclut aussi les fichiers créés dans la session courante.
        """
        import os as _os

        seen_paths: set[str] = set()
        all_chunks: list[str] = []

        def _add_docs(query: str):
            if not query.strip():
                return
            for fpath, content in self._memory_fn(query, k=self.top_k):
                rel = _os.path.relpath(fpath, ".").replace("\\", "/")
                if rel not in seen_paths:
                    seen_paths.add(rel)
                    all_chunks.append(f"FILE: {rel}\n{content[:900]}")

        # 1. Recherche par goal principal
        _add_docs(goal)

        # 2. Recherche par répertoire/nom de fichier cible
        if path:
            dir_name = _os.path.dirname(path)
            base_name = _os.path.splitext(_os.path.basename(path))[0]
            _add_docs(dir_name or base_name)

        # 3. Recherche par mots-clés de la description
        if description:
            _add_docs(description[:200])

        # 4. Fichiers de la session courante (priorité haute — toujours inclus)
        session = self._session_context(max_chars_per_file=1500)
        if session:
            all_chunks.insert(0, session)

        return "\n\n".join(all_chunks) if all_chunks else "No indexed context found"

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
        session_ctx = self._session_context(max_chars_per_file=400)

        return f"""
You are an autonomous coding agent.
Goal: {goal}

You can use one action at a time:
- list_files: {{"contains": "optional substring", "ext": ".py", "limit": 50}}
- read_file: {{"path": "relative/path.py"}}
- read_many: {{"paths": ["a.py", "b.py"]}}
- search_code: {{"pattern": "regex or text", "limit": 20}}
- create_file: {{"path": "relative/path.ext", "description": "complete description of what the file must contain and do"}}
- propose_edit: {{"path": "relative/path.py", "instruction": "change request. You can use SEARCH/REPLACE blocks for surgical edits."}}
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
- Use create_file to generate NEW files (backend, frontend, modules, configs) — any extension (.py, .js, .ts, .html, .css, etc.).
- Use propose_edit only to MODIFY existing files.
- Keep edits minimal and safe.
- Use git actions only when the goal explicitly asks for git/commit/pr operations.
- When goal says "generate tests for X/Y.py", create "tests/test_Y.py" (mirror the source filename with test_ prefix under tests/).
- Never create a file that already exists on disk — use propose_edit instead.

Return format:
{{"action":"...","reason":"short reason","args":{{...}}}}

Valid examples:
{{"action":"create_file","reason":"generate tests for users/api.py","args":{{"path":"tests/test_api.py","description":"pytest tests for users/api.py Flask Blueprint: test create/get/update/delete/login endpoints using Flask test client"}}}}
{{"action":"create_file","reason":"create user model","args":{{"path":"users/models.py","description":"SQLAlchemy User model with id, email, hashed_password, created_at fields and CRUD methods"}}}}
{{"action":"create_file","reason":"create REST API","args":{{"path":"users/api.py","description":"Flask Blueprint with GET /users, POST /users, PUT /users/<id>, DELETE /users/<id> endpoints using bcrypt for password hashing"}}}}
{{"action":"create_file","reason":"create frontend component","args":{{"path":"frontend/components/UserForm.jsx","description":"React component for user registration form with email and password fields, validation, and API call"}}}}
{{"action":"search_code","reason":"find implementation points","args":{{"pattern":"run_quality_pipeline","limit":20}}}}
{{"action":"read_file","reason":"inspect target file","args":{{"path":"agent/auto_agent.py"}}}}
{{"action":"propose_edit","reason":"prepare safe minimal change","args":{{"path":"agent/eval_runner.py","instruction":"Add parse KPI details"}}}}
{{"action":"finish","reason":"task completed","args":{{"summary":"Explained how pr-ready works"}}}}

Project files:
{files_text}

Relevant indexed context:
{context}

Files created in this session (use their content as context for new files):
{session_ctx if session_ctx else "none yet"}

Recent steps:
{history}
"""

    def _schema_repair_prompt(
        self, goal: str, decision: dict, schema_error: str
    ) -> str:
        return f"""
You must repair this planner JSON to satisfy a strict schema.
Goal: {goal}
Schema error: {schema_error}
Invalid decision:
{json.dumps(decision, ensure_ascii=False)}

Return only one strict JSON object with this exact shape:
{{"action":"...","reason":"...","args":{{...}}}}

Rules:
- action must be one of:
  list_files, read_file, read_many, search_code, propose_edit, apply_edit,
  apply_all_staged, run_tests, run_quality, project_map,
  git_branch, git_commit, git_diff, finish
- args must match action:
  read_file->{{"path":"..."}}
  read_many->{{"paths":["..."]}}
  search_code->{{"pattern":"...","limit":20?}}
  propose_edit->{{"path":"...","instruction":"..."}}
  apply_edit->{{"path":"..."}}
  run_tests->{{"command":"..."}}
  finish->{{"summary":"..."}}
- never add unknown keys in args.
- keep decision useful for the goal.
"""

    # P2.1 — Delegate to decision_normalizer module
    def _extract_target_file_from_goal(self, goal: str) -> Optional[str]:
        return extract_target_file_from_goal(goal)

    def _extract_all_target_files_from_goal(self, goal: str) -> List[str]:
        return extract_all_target_files_from_goal(goal)

    def _is_code_edit_goal(self, goal: str) -> bool:
        return is_code_edit_goal(goal)

    def _is_allowed_edit_path(self, goal: str, path: str) -> bool:
        return is_allowed_edit_path(goal, path)

    def _bootstrap_code_edit_decisions(self, goal: str, auto_apply: bool):
        if self._bootstrapped_code_edit:
            return
        target = self._extract_target_file_from_goal(goal)
        if not target:
            return
        if not self._is_code_edit_goal(goal):
            return

        queue = [
            {
                "action": "read_file",
                "reason": "code_edit_bootstrap_read",
                "args": {"path": target},
            },
            {
                "action": "propose_edit",
                "reason": "code_edit_bootstrap_patch",
                "args": {"path": target, "instruction": goal[:500]},
            },
        ]
        if self._generate_tests and target.endswith(".py"):
            queue.append(
                {
                    "action": "propose_edit",
                    "reason": "code_edit_bootstrap_tests",
                    "args": {
                        "path": suggest_test_path(target),
                        "instruction": (
                            f"Add pytest tests for {target} with one nominal case and one edge case."
                        ),
                    },
                }
            )

        if auto_apply:
            queue.append(
                {
                    "action": "apply_all_staged",
                    "reason": "code_edit_bootstrap_apply",
                    "args": {},
                }
            )
            queue.append(
                {
                    "action": "run_quality",
                    "reason": "code_edit_bootstrap_validate",
                    "args": {},
                }
            )
        queue.append(
            {
                "action": "finish",
                "reason": "code_edit_bootstrap_finish",
                "args": {"summary": f"Prepared code edit for {target}"},
            }
        )
        self._forced_decisions = queue + self._forced_decisions
        self._bootstrapped_code_edit = True

    def _is_git_goal(self, goal: str) -> bool:
        return is_git_goal(goal)

    def _coerce_decision(self, goal: str, decision: dict) -> dict:
        return coerce_decision(goal, decision)

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
        if self._decision_signatures[-4:].count(sig) >= 2:
            return True
        # Détecter les propose_edit répétés sur le même fichier sans apply_edit entre deux
        if action == "propose_edit":
            path = args.get("path", "")
            recent = self._decision_signatures[-6:]
            propose_count = sum(
                1
                for s in recent
                if json.loads(s).get("a") == "propose_edit"
                and json.loads(s).get("g", {}).get("path") == path
            )
            if propose_count >= 3:
                return True
        return False

    def _outcome_loop_detected(self, action: str, args: dict, result: str) -> bool:
        sig = self._signature(action, args, result)
        return self._outcome_signatures[-4:].count(sig) >= 2

    def _action_cost(self, action: str) -> int:
        costs = {
            "list_files": 1,
            "read_file": 1,
            "read_many": 2,
            "search_code": 2,
            "propose_edit": 3,
            "apply_edit": 4,
            "apply_all_staged": 5,
            "run_tests": 4,
            "run_quality": 5,
            "project_map": 2,
            "git_branch": 2,
            "git_commit": 3,
            "git_diff": 2,
            "finish": 0,
        }
        return costs.get(action, 2)

    def _action_timeout(self, action: str) -> int:
        """P2.3 — Timeout spécifique par type d'action (en secondes)."""
        timeouts = {
            "list_files": 10,
            "read_file": 10,
            "read_many": 15,
            "search_code": 15,
            "create_file": 180,
            "propose_edit": 180,
            "apply_edit": 60,
            "apply_all_staged": 90,
            "run_tests": 300,
            "run_quality": 300,
            "project_map": 15,
            "git_branch": 10,
            "git_commit": 30,
            "git_diff": 10,
            "finish": 5,
        }
        return timeouts.get(action, 60)

    def _is_stalled_step(self, action: str, result: str) -> bool:
        low = (result or "").lower()
        if action in {"list_files", "search_code"} and (
            "no files found" in low or "no matches" in low
        ):
            return True
        if action in {"read_file", "read_many"} and (
            "[error]" in low or not low.strip()
        ):
            return True
        if action in {"unknown_action", "loop_guard"}:
            return True
        return False

    def _failure_excerpt(self, text: str, max_lines: int = 8) -> str:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return "no failure output"
        return "\n".join(lines[:max_lines])[:800]

    def _extract_error_paths(self, text: str) -> List[str]:
        found = re.findall(r"([A-Za-z0-9_./\\-]+\.py)", text or "")
        out = []
        seen = set()
        for raw in found:
            p = raw.replace("\\", "/")
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
        return out

    def _enqueue_replan_after_failure(
        self,
        failure_kind: str,
        failure_text: str,
        auto_apply: bool,
        fallback_path: Optional[str] = None,
    ) -> bool:
        if self._replan_attempts >= self._max_replan_attempts:
            return False

        targets = self._extract_error_paths(failure_text)
        target_path = targets[0] if targets else (fallback_path or "")
        excerpt = self._failure_excerpt(failure_text)

        queue = []
        if target_path:
            queue.append(
                {
                    "action": "propose_edit",
                    "reason": f"replan_after_{failure_kind}",
                    "args": {
                        "path": target_path,
                        "instruction": (
                            f"Fix {failure_kind} issue using this diagnostic:\n{excerpt}\n"
                            "Keep patch minimal and focused on making checks pass."
                        ),
                    },
                }
            )
            if auto_apply:
                queue.append(
                    {
                        "action": "apply_edit",
                        "reason": f"apply_after_{failure_kind}_replan",
                        "args": {"path": target_path},
                    }
                )
        else:
            queue.append(
                {
                    "action": "search_code",
                    "reason": f"replan_after_{failure_kind}",
                    "args": {
                        "pattern": "test_|assert|ruff|black|traceback",
                        "limit": 20,
                    },
                }
            )

        queue.append(
            {
                "action": (
                    "run_quality" if failure_kind in {"lint", "format"} else "run_tests"
                ),
                "reason": f"verify_after_{failure_kind}_replan",
                "args": {},
            }
        )

        self._forced_decisions.extend(queue)
        self._replan_attempts += 1
        self.event_logger.log(
            "replan_policy",
            {
                "failure_kind": failure_kind,
                "target_path": target_path,
                "queued_actions": [d["action"] for d in queue],
                "attempt": self._replan_attempts,
            },
        )
        return True

    def _plan_with_critique(self, goal: str) -> dict:
        if self._forced_decisions:
            forced = self._forced_decisions.pop(0)
            self.event_logger.log(
                "forced_plan",
                {"decision": forced, "remaining": len(self._forced_decisions)},
            )
            return forced

        raw_decision = self._llm_fn(
            self._planner_prompt(goal),
            retries=MAX_RETRIES_JSON,
            prompt_class="planner",
            task_type="planning",
        )
        decision = self._normalize_decision(raw_decision)
        self.event_logger.log("plan", {"goal": goal, "decision": decision})

        if decision.get("_error_type") == "parse":
            parse_meta = (
                decision.get("_parse_meta")
                if isinstance(decision.get("_parse_meta"), dict)
                else {}
            )
            self.event_logger.log_failure(
                "parse",
                "planner_json_parse_failed",
                {
                    "decision": decision,
                    "parse_class": parse_meta.get("error_class", "unknown_parse_error"),
                    "parse_attempts": parse_meta.get("attempt_count", 0),
                    "prompt_class": parse_meta.get("prompt_class", "planner"),
                },
            )
            self._parse_fallback_count += 1
            return self._fallback_decision(goal, reason="parse_failed")
        self._parse_fallback_count = 0

        decision = self._coerce_decision(goal, decision)
        ok, msg = validate_decision_schema(decision)
        if not ok:
            corrected = self._autocorrect_decision_schema(goal, decision, msg)
            c_ok, _ = validate_decision_schema(corrected)
            if c_ok:
                self.event_logger.log(
                    "schema_autocorrect",
                    {"from": decision, "to": corrected, "issue": msg},
                )
                decision = corrected
            else:
                repaired = self._llm_fn(
                    self._schema_repair_prompt(goal, decision, msg),
                    retries=1,
                    prompt_class="planner_schema_repair",
                    task_type="planning",
                )
                repaired = self._normalize_decision(repaired)
                r_ok, r_msg = validate_decision_schema(repaired)
                if r_ok:
                    self.event_logger.log(
                        "schema_repair",
                        {"from": decision, "to": repaired, "issue": msg},
                    )
                    decision = repaired
                else:
                    self.event_logger.log_failure(
                        "parse",
                        f"planner_schema_invalid:{msg}",
                        {
                            "decision": decision,
                            "parse_class": "schema_invalid",
                            "prompt_class": "planner",
                            "repair_schema_error": r_msg,
                        },
                    )
                    return self._fallback_decision(goal, reason=f"schema_invalid:{msg}")

        import time as _time

        _t0_critique = _time.time()
        raw_critique = self._llm_fn(
            self._critique_prompt(goal, decision),
            retries=1,  # 1 seul retry : si échec, on auto-approuve (fallback sûr)
            prompt_class="critique",
            task_type="analysis",
        )
        _critique_elapsed = round(_time.time() - _t0_critique, 1)
        if _critique_elapsed > 60:
            print(
                f"  [critique] lente ({_critique_elapsed}s) — charge élevée sur Ifa1.0"
            )
        if (
            isinstance(raw_critique, dict)
            and raw_critique.get("_error_type") == "parse"
        ):
            parse_meta = (
                raw_critique.get("_parse_meta")
                if isinstance(raw_critique.get("_parse_meta"), dict)
                else {}
            )
            self.event_logger.log_failure(
                "parse",
                "critique_json_parse_failed",
                {
                    "critique": raw_critique,
                    "parse_class": parse_meta.get("error_class", "unknown_parse_error"),
                    "parse_attempts": parse_meta.get("attempt_count", 0),
                    "prompt_class": parse_meta.get("prompt_class", "critique"),
                },
            )
            return decision
        critique = self._normalize_critique(raw_critique)
        c_ok, c_msg = validate_critique_schema(critique)
        if not c_ok:
            self.event_logger.log_failure(
                "parse",
                f"critique_schema_invalid:{c_msg}",
                {
                    "critique": critique,
                    "parse_class": "critique_schema_invalid",
                    "prompt_class": "critique",
                },
            )
            return decision

        if critique.get("approve"):
            return decision

        patched = critique.get("patched_decision")
        if isinstance(patched, dict):
            patched = self._coerce_decision(goal, patched)
            p_ok, p_msg = validate_decision_schema(patched)
            if not p_ok:
                patched2 = self._autocorrect_decision_schema(goal, patched, p_msg)
                p2_ok, _ = validate_decision_schema(patched2)
                if p2_ok:
                    self.event_logger.log(
                        "schema_autocorrect",
                        {"from": patched, "to": patched2, "issue": p_msg},
                    )
                    patched = patched2
                    p_ok = True
            if p_ok:
                self.event_logger.log(
                    "critique_patch",
                    {"reason": critique.get("reason"), "patched": patched},
                )
                return patched
            self.event_logger.log_failure(
                "parse",
                f"patched_schema_invalid:{p_msg}",
                {
                    "patched": patched,
                    "parse_class": "patched_schema_invalid",
                    "prompt_class": "critique",
                },
            )

        self.event_logger.log(
            "critique_reject_fallback",
            {"reason": critique.get("reason", "n/a")},
        )
        return self._fallback_decision(goal, reason="critique_rejected")

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

    def _generate_file_content(self, goal: str, path: str, description: str) -> str:
        """Génère le contenu complet d'un nouveau fichier via le LLM."""
        from agent.llm_interface import ask_llm
        import os as _os

        ext = _os.path.splitext(path)[1].lower()
        lang_hints = {
            ".py": "Python",
            ".js": "JavaScript (ES modules)",
            ".ts": "TypeScript",
            ".jsx": "React JSX",
            ".tsx": "React TSX with TypeScript",
            ".html": "HTML5",
            ".css": "CSS",
            ".scss": "SCSS",
            ".json": "JSON",
            ".md": "Markdown",
            ".sql": "SQL",
            ".yaml": "YAML",
            ".yml": "YAML",
            ".toml": "TOML",
        }
        lang = lang_hints.get(ext, "plain text")
        context = self._summarize_context_multi(
            goal, path=path, description=description
        )

        prompt = (
            f"You are an expert software engineer. Generate the COMPLETE content of a new file.\n\n"
            f"File path  : {path}\n"
            f"Language   : {lang}\n"
            f"Description: {description}\n\n"
            f"Overall project goal: {goal}\n\n"
            f"Relevant project context (including files already created this session):\n{context}\n\n"
            f"Rules:\n"
            f"- Return ONLY the raw file content. No explanation, no markdown fences.\n"
            f"- The file must be complete and functional — no TODOs, no stubs.\n"
            f"- Follow best practices for {lang}.\n"
            f"- Include all necessary imports/dependencies.\n"
            f"- For Python: use type hints, docstrings, proper error handling.\n"
            f"- For JS/TS/JSX: use modern syntax, proper exports.\n"
            f"- For HTML: include <!DOCTYPE html> and full document structure.\n"
        )
        raw = ask_llm(prompt, task_type="backend")
        # Retirer les éventuelles fences markdown que le modèle pourrait ajouter
        raw = (raw or "").strip()
        if raw.startswith("```"):
            lines = raw.splitlines()
            end = next(
                (i for i in range(len(lines) - 1, 0, -1) if lines[i].strip() == "```"),
                -1,
            )
            if end > 0:
                raw = "\n".join(lines[1:end]).strip()
        return raw

    def plan_once(self, goal: str) -> dict:
        return self._plan_with_critique(goal)

    def run(
        self,
        goal: str,
        auto_apply: bool = False,
        fix_until_green: bool = False,
        generate_tests: bool = False,
        max_seconds: int = 0,
    ) -> str:
        self._fix_until_green = bool(fix_until_green)
        self._generate_tests = bool(generate_tests)
        self._consecutive_stalled_steps = 0
        self._total_cost = 0
        self._replan_attempts = 0
        self._forced_decisions = []
        self._has_validation_action = False
        self._bootstrapped_code_edit = False
        self._bootstrap_code_edit_decisions(goal, auto_apply=auto_apply)
        start_ts = datetime.utcnow()

        print(f"\n[stella] Démarrage de l'agent — objectif : {goal[:80]}")
        print(
            f"[stella] Paramètres : max_steps={self.max_steps}, apply={auto_apply}, fix={fix_until_green}, tests={generate_tests}"
        )
        print()

        for index in range(1, self.max_steps + 1):
            if (
                max_seconds
                and (datetime.utcnow() - start_ts).total_seconds() > max_seconds
            ):
                self._record(
                    AutoStep(
                        index,
                        "finish",
                        "max_time_reached",
                        f"Stopped: max runtime reached ({max_seconds}s)",
                    )
                )
                return self.render_summary(
                    f"Stopped: max runtime reached ({max_seconds}s)"
                )

            decision = self._plan_with_critique(goal)
            action = decision.get("action", "")
            reason = decision.get("reason", "")
            args = decision.get("args", {}) or {}

            # Feedback visuel pour l'utilisateur
            _action_labels = {
                "list_files": "liste les fichiers",
                "read_file": f"lit {args.get('path', '')}",
                "read_many": f"lit {len(args.get('paths', []))} fichier(s)",
                "search_code": f"cherche « {args.get('pattern', '')} »",
                "create_file": f"crée {args.get('path', '')}",
                "propose_edit": f"prépare un patch pour {args.get('path', '')}",
                "apply_edit": f"applique le patch sur {args.get('path', '')}",
                "apply_all_staged": "applique tous les patches en attente",
                "run_tests": "exécute les tests",
                "run_quality": "vérifie la qualité du code",
                "project_map": "génère la carte du projet",
                "git_branch": f"crée la branche {args.get('name', '')}",
                "git_commit": "committe les changements",
                "git_diff": "affiche le diff git",
                "finish": "finalise la tâche",
            }
            label = _action_labels.get(action, action)
            elapsed_so_far = round((datetime.utcnow() - start_ts).total_seconds(), 1)
            print(f"  [{index}/{self.max_steps}] {label}... ({elapsed_so_far}s)", flush=True)

            if (
                index == self.max_steps
                and self.staged_edits
                and not self._has_validation_action
                and self._fix_until_green
                and action not in {"run_tests", "run_quality"}
            ):
                action = "run_quality"
                reason = "forced_final_validation"
                args = {}

            self._total_cost += self._action_cost(action)
            if self._total_cost > self._max_cost:
                self._record(
                    AutoStep(
                        index,
                        "finish",
                        "max_cost_reached",
                        f"Stopped: max cost budget reached ({self._total_cost}/{self._max_cost})",
                    )
                )
                return self.render_summary(
                    f"Stopped: max cost budget reached ({self._total_cost}/{self._max_cost})"
                )

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
                # Pour les actions de lecture (read-only), synthétiser une réponse
                # en lien avec le goal plutôt que de retourner une erreur.
                _read_only = {"read_file", "read_many", "search_code", "list_files"}
                if action in _read_only and self.steps:
                    last_result = next(
                        (
                            s.result
                            for s in reversed(self.steps)
                            if s.result and s.action in _read_only
                        ),
                        None,
                    )
                    if last_result:
                        from agent.llm_interface import ask_llm

                        synthesis_prompt = (
                            f"Goal: {goal}\n\n"
                            f"Collected context:\n{last_result[:3000]}\n\n"
                            "Based on the context above, answer the goal concisely and clearly. "
                            "Do not repeat the code verbatim — summarize and explain."
                        )
                        summary = ask_llm(synthesis_prompt, task_type="analysis")
                        if not summary:
                            summary = last_result[:2000]
                        self._record(
                            AutoStep(
                                index, "finish", "loop_resolved_read_only", summary
                            )
                        )
                        elapsed = round(
                            (datetime.utcnow() - start_ts).total_seconds(), 1
                        )
                        print(f"\n[stella] Terminé en {elapsed}s\n")
                        return self.render_summary(summary)
                summary = "Stopped repeated decision loop; recommend narrowing goal to a concrete file-level change request"
                self._record(AutoStep(index, "finish", "auto_stop_repeat", summary))
                return self.render_summary(summary)

            self._decision_signatures.append(self._signature(action, args))
            step = AutoStep(index, action, reason, "")

            # P2.1 — Delegate action execution to ActionExecutor
            try:
                result = self._executor.execute(
                    action, args, goal=goal, auto_apply=auto_apply, index=index,
                )
                # Special handling for apply_edit without staged edit
                if action == "apply_edit" and result == "[no_staged_edit]":
                    step.result = "No staged edit for this file"
                    self._record(step)
                    self._record(AutoStep(
                        index, "finish", "invalid_action_sequence",
                        "apply_edit requested without staged edit; stopping",
                    ))
                    return self.render_summary("apply_edit requested without staged edit; stopping")
                # Special handling for git_commit redundancy
                if action == "git_commit" and "nothing to commit" in result.lower():
                    recent_nothing = any(
                        s.action == "git_commit" and "nothing to commit" in s.result.lower()
                        for s in self.steps[-2:]
                    )
                    if recent_nothing:
                        step.result = result
                        self._record(step)
                        self._record(AutoStep(
                            index, "finish", "auto_stop_redundant_commit",
                            "No more changes to commit; stopping redundant git_commit loop",
                        ))
                        return self.render_summary("No more changes to commit; stopping redundant git_commit loop")
                # Special handling for finish action
                if action == "finish":
                    step.result = result
                    self._record(step)
                    elapsed = round((datetime.utcnow() - start_ts).total_seconds(), 1)
                    print(f"\n[stella] Terminé en {elapsed}s\n")
                    return self.render_summary(step.result)
                # Unknown action
                if result.startswith("Unknown action:"):
                    step.action = "unknown_action"
                step.result = result
            except Exception as exc:
                self.event_logger.log_failure(
                    "tool", "tool_execution_exception",
                    {"action": action, "error": str(exc)},
                )
                step.result = f"Tool error: {exc}"

            if self._is_stalled_step(step.action, step.result):
                self._consecutive_stalled_steps += 1
            else:
                self._consecutive_stalled_steps = 0

            if self._consecutive_stalled_steps >= 4:
                self._record(step)
                self._record(
                    AutoStep(
                        index,
                        "finish",
                        "stop_impasse",
                        "Stopped: impasse detected (4 stalled steps in a row)",
                    )
                )
                return self.render_summary(
                    "Stopped: impasse detected (4 stalled steps in a row)"
                )

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
        # Statistiques de session
        total_steps = len(self.steps)
        files_modified = list(self.staged_edits.keys())
        actions_used = {}
        for s in self.steps:
            actions_used[s.action] = actions_used.get(s.action, 0) + 1

        lines = [
            f"Final: {final_message}",
            "",
            f"--- Resume de session ---",
            f"  Etapes executees : {total_steps}/{self.max_steps}",
            f"  Fichiers modifies : {len(files_modified)}",
        ]
        if files_modified:
            for f in files_modified[:10]:
                lines.append(f"    - {f}")
        lines.append(f"  Cout total : {self._total_cost}/{self._max_cost}")
        lines.append(f"  Replans : {self._replan_attempts}/{self._max_replan_attempts}")
        lines.append("")
        lines.append("Decisions:")
        for s in self.steps:
            lines.append(f"  {s.step}. [{s.action}] {s.reason} -> {s.result[:160]}")
        return "\n".join(lines)
