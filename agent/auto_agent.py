"""
Autonomous agent loop for Stella.
Handles planning, tool execution, and self-correction.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from agent.action_schema import (
    ALLOWED_ACTIONS,
    validate_critique_schema,
    validate_decision_schema,
)
from agent.agent import apply_suggestion, patch_risk, propose_file_update
from agent.config import (
    AUTO_TEST_COMMAND,
    MAX_RETRIES_JSON,
    PROJECT_ROOT,
    TOP_K_RESULTS,
)
from agent.event_logger import EventLogger
from agent.git_tools import commit_all, create_branch, diff_summary
from agent.llm_interface import ask_llm_json
from agent.deps import detect_and_install_deps
from agent.memory import index_file_in_session, remember_fix_strategy, search_memory
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
    list_python_files,
    read_file,
    read_many,
    run_tests,
    search_code,
    write_new_file,
)


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

    def _normalize_decision(self, decision: dict) -> dict:
        if not isinstance(decision, dict):
            return decision

        root_args = {
            k: v
            for k, v in decision.items()
            if k
            not in {
                "action",
                "tool",
                "name",
                "decision",
                "reason",
                "why",
                "args",
                "parameters",
                "input",
            }
            and not (isinstance(k, str) and k.startswith("_"))
        }

        action = (
            decision.get("action")
            or decision.get("tool")
            or decision.get("name")
            or decision.get("decision")
            or ""
        )
        action = str(action).strip().lower().replace("-", "_")
        alias = {
            "read": "read_file",
            "read_files": "read_many",
            "search": "search_code",
            "grep": "search_code",
            "create": "create_file",
            "new_file": "create_file",
            "write_file": "create_file",
            "generate_file": "create_file",
            "create_module": "create_file",
            "write_new_file": "create_file",
            "edit": "propose_edit",
            "propose": "propose_edit",
            "propose_patch": "propose_edit",
            "apply_patch": "apply_edit",
            "apply": "apply_edit",
            "apply_staged": "apply_all_staged",
            "test": "run_tests",
            "run_test": "run_tests",
            "quality": "run_quality",
            "run_quality_pipeline": "run_quality",
            "map": "project_map",
            "branch": "git_branch",
            "commit": "git_commit",
            "diff": "git_diff",
            "done": "finish",
            "final": "finish",
        }
        action = alias.get(action, action)

        reason = decision.get("reason") or decision.get("why") or "auto_normalized"
        args = decision.get("args")
        if args is None:
            args = decision.get("parameters")
        if args is None:
            args = decision.get("input")
        if not isinstance(args, dict):
            args = {}

        args = {**root_args, **dict(args)}
        if action in {"list_files", "search_code"}:
            limit = args.get("limit")
            if isinstance(limit, str) and limit.isdigit():
                args["limit"] = int(limit)
            elif isinstance(limit, float):
                args["limit"] = int(limit)
        if action == "read_many" and isinstance(args.get("paths"), str):
            args["paths"] = [args["paths"]]
        if action == "search_code" and "pattern" not in args:
            query = args.get("query") or args.get("text")
            if isinstance(query, str) and query.strip():
                args["pattern"] = query
        if action == "run_tests" and "command" not in args:
            cmd = args.get("test_command")
            if isinstance(cmd, str) and cmd.strip():
                args["command"] = cmd
        if action == "finish" and "summary" not in args:
            summary = args.get("message") or reason
            if isinstance(summary, str) and summary.strip():
                args["summary"] = summary
        if action in {
            "apply_all_staged",
            "run_quality",
            "project_map",
            "git_diff",
        }:
            args = {}

        if action not in ALLOWED_ACTIONS:
            if isinstance(args.get("path"), str) and isinstance(
                args.get("instruction"), str
            ):
                action = "propose_edit"
            elif isinstance(args.get("paths"), list):
                action = "read_many"
            elif isinstance(args.get("pattern"), str):
                action = "search_code"
            elif isinstance(args.get("path"), str):
                action = "read_file"
            elif isinstance(args.get("summary"), str):
                action = "finish"

        allowed = {
            "list_files": {"contains", "ext", "limit"},
            "read_file": {"path"},
            "read_many": {"paths"},
            "search_code": {"pattern", "limit"},
            "create_file": {"path", "description"},
            "propose_edit": {"path", "instruction"},
            "apply_edit": {"path"},
            "apply_all_staged": set(),
            "run_tests": {"command"},
            "run_quality": set(),
            "project_map": set(),
            "git_branch": {"name"},
            "git_commit": {"message"},
            "git_diff": set(),
            "finish": {"summary"},
        }.get(action, set(args.keys()))
        args = {k: v for k, v in args.items() if k in allowed}

        normalized = {"action": action, "reason": str(reason), "args": args}
        for k, v in decision.items():
            if isinstance(k, str) and k.startswith("_"):
                normalized[k] = v
        return normalized

    def _normalize_critique(self, critique: dict) -> dict:
        if not isinstance(critique, dict):
            return critique
        approve = critique.get("approve")
        if isinstance(approve, str):
            approve = approve.strip().lower() in {"true", "1", "yes", "ok"}
        if not isinstance(approve, bool):
            approve = False

        reason = critique.get("reason") or critique.get("comment") or "normalized"
        patched = critique.get("patched_decision")
        if patched is None:
            patched = critique.get("patch")
        if not isinstance(patched, dict):
            patched = None
        else:
            patched = self._normalize_decision(patched)

        normalized = {
            "approve": approve,
            "reason": str(reason),
            "patched_decision": patched,
        }
        for k, v in critique.items():
            if isinstance(k, str) and k.startswith("_"):
                normalized[k] = v
        return normalized

    def _infer_fallback_action(self, goal: str, args: dict) -> tuple[str, dict]:
        low_goal = (goal or "").lower()
        target = self._extract_target_file_from_goal(goal)
        if target:
            return "read_file", {"path": target}

        text_blob = " ".join(
            str(v) for v in args.values() if isinstance(v, (str, int, float, bool))
        ).lower()
        combined = f"{low_goal} {text_blob}"

        if any(k in combined for k in ["test", "pytest", "coverage", "fiabilite"]):
            return "search_code", {"pattern": "pytest|test_|_test.py", "limit": 20}
        if any(k in combined for k in ["performance", "latence", "speed", "optimiz"]):
            return "search_code", {
                "pattern": "timeout|sleep|requests.post|subprocess.run",
                "limit": 20,
            }
        if any(k in combined for k in ["refactor", "risk", "analyse", "architecture"]):
            return "list_files", {"limit": 40, "ext": ".py"}
        if any(k in combined for k in ["pr", "commit", "branch", "git"]):
            return "git_diff", {}
        return "list_files", {"limit": 40, "ext": ".py"}

    def _autocorrect_decision_schema(self, goal: str, decision: dict, msg: str) -> dict:
        if not isinstance(decision, dict):
            return decision
        corrected = self._normalize_decision(decision)
        action = corrected.get("action")
        args = corrected.get("args", {}) or {}
        if not isinstance(args, dict):
            args = {}

        if "invalid action" in msg:
            action, inferred_args = self._infer_fallback_action(goal, args)
            corrected["action"] = action
            args = dict(inferred_args)

        if action == "read_file" and "missing required arg 'path'" in msg:
            path = args.get("file") or args.get("target")
            if isinstance(path, str) and path:
                args["path"] = path
            else:
                target = self._extract_target_file_from_goal(goal)
                if target:
                    args["path"] = target
        elif action == "read_many" and "missing required arg 'paths'" in msg:
            path = args.get("path")
            if isinstance(path, str) and path:
                args["paths"] = [path]
            else:
                target = self._extract_target_file_from_goal(goal)
                if target:
                    args["paths"] = [target]
        elif action == "propose_edit":
            if "missing required arg 'path'" in msg:
                path = args.get("file") or args.get("target")
                if isinstance(path, str) and path:
                    args["path"] = path
                else:
                    target = self._extract_target_file_from_goal(goal)
                    if target:
                        args["path"] = target
            if "missing required arg 'instruction'" in msg:
                instruction = (
                    args.get("prompt")
                    or args.get("change")
                    or decision.get("reason", "")
                )
                if isinstance(instruction, str) and instruction:
                    args["instruction"] = instruction
                elif isinstance(goal, str) and goal.strip():
                    args["instruction"] = goal[:500]
        elif action == "finish" and "unexpected arg 'reason'" in msg:
            summary = args.get("reason")
            if isinstance(summary, str) and summary:
                args.pop("reason", None)
                args["summary"] = summary

        corrected["args"] = args
        corrected["reason"] = str(corrected.get("reason") or "auto_corrected_schema")
        return corrected

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
        docs = search_memory(goal, k=self.top_k)
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
            for fpath, content in search_memory(query, k=self.top_k):
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

    def _extract_target_file_from_goal(self, goal: str) -> Optional[str]:
        low = goal.strip()
        m = re.search(r"([A-Za-z0-9_./\\-]+\.[a-zA-Z]{1,5})", low)
        if not m:
            return None
        path = m.group(1).replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]
        return path

    def _extract_all_target_files_from_goal(self, goal: str) -> List[str]:
        """Extrait les chemins de fichiers à CRÉER depuis le goal.

        Exclut les fichiers qui existent déjà sur disque (ils sont seulement
        référencés comme contexte, pas comme cibles de création).
        """
        found = re.findall(r"([A-Za-z0-9_./\\-]+\.[a-zA-Z]{1,5})", goal)
        seen: set[str] = set()
        out: List[str] = []
        for raw in found:
            path = raw.replace("\\", "/")
            if path.startswith("./"):
                path = path[2:]
            # Filtrer les faux positifs (e.g. "1.0", "e.g")
            if not re.match(
                r".+\.(py|js|ts|jsx|tsx|html|css|scss|json|yaml|yml|toml|sql|md)$", path
            ):
                continue
            # Ne pas inclure les fichiers qui existent déjà sur disque
            abs_path = os.path.join(PROJECT_ROOT, path)
            if os.path.isfile(abs_path):
                continue
            if path not in seen:
                seen.add(path)
                out.append(path)
        return out

    def _is_code_edit_goal(self, goal: str) -> bool:
        low = (goal or "").lower()
        return "code-edit" in low or ("dans " in low and ".py" in low)

    def _is_allowed_edit_path(self, goal: str, path: str) -> bool:
        target = self._extract_target_file_from_goal(goal)
        if not target:
            return True
        norm = (path or "").replace("\\", "/").lower()
        target_norm = target.replace("\\", "/").lower()
        if norm == target_norm:
            return True
        if norm == suggest_test_path(target).replace("\\", "/").lower():
            return True
        if norm.startswith("tests/"):
            return True
        return False

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

        raw_decision = ask_llm_json(
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
                repaired = ask_llm_json(
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
        raw_critique = ask_llm_json(
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
            print(f"  [{index}/{self.max_steps}] {label}...", flush=True)

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

                elif action == "create_file":
                    path = args.get("path", "")
                    description = args.get("description", "")
                    if not path or not description:
                        step.result = (
                            "[error] create_file requires 'path' and 'description'"
                        )
                    elif path in self.staged_edits:
                        # Fichier déjà créé dans cette session → skip
                        step.result = f"Skipped: {path} already created this session"
                    elif os.path.isfile(os.path.join(PROJECT_ROOT, path)):
                        # Fichier pré-existant sur disque → skip, puis vérifier si tous les
                        # fichiers cibles du goal existent → finish automatique
                        step.result = f"Skipped: {path} already exists on disk (use fix to modify it)"
                        all_targets = self._extract_all_target_files_from_goal(goal)
                        if not all_targets and not self._forced_decisions:
                            # Tous les fichiers mentionnés dans le goal existent déjà → finish
                            self._forced_decisions.insert(
                                0,
                                {
                                    "action": "finish",
                                    "reason": "all_target_files_already_exist",
                                    "args": {
                                        "summary": f"All target files already exist on disk: {path}"
                                    },
                                },
                            )
                    else:
                        print(f"     génère le contenu de {path}...", flush=True)
                        content = self._generate_file_content(goal, path, description)
                        if not content:
                            step.result = (
                                f"[error] LLM returned empty content for {path}"
                            )
                        else:
                            write_result = write_new_file(path, content)
                            if write_result.startswith("ok:"):
                                self.staged_edits[path] = content
                                risk = patch_risk(path, content)
                                # Indexer immédiatement le fichier créé dans la mémoire
                                # pour qu'il soit trouvable par les prochaines créations.
                                n_chunks = index_file_in_session(path, content)
                                # Détecter et installer les dépendances manquantes.
                                deps_result = detect_and_install_deps(path, content)
                                step.result = (
                                    f"Created {path} ({len(content)} chars), "
                                    f"risk={risk['level']}:{risk['score']}, "
                                    f"indexed={n_chunks} chunks, "
                                    f"{deps_result}"
                                )
                                # Enchaîner automatiquement les autres fichiers du goal.
                                all_targets = self._extract_all_target_files_from_goal(
                                    goal
                                )
                                pending = [
                                    t for t in all_targets if t not in self.staged_edits
                                ]
                                if pending:
                                    # Forcer le prochain fichier comme prochaine décision
                                    next_path = pending[0]
                                    self._forced_decisions.insert(
                                        0,
                                        {
                                            "action": "create_file",
                                            "reason": "multi_file_goal_continuation",
                                            "args": {
                                                "path": next_path,
                                                "description": (
                                                    f"{goal[:400]} "
                                                    f"(already created: {', '.join(self.staged_edits.keys())})"
                                                ),
                                            },
                                        },
                                    )
                                else:
                                    # Tous les fichiers du goal sont créés → finish
                                    created = ", ".join(self.staged_edits.keys())
                                    self._forced_decisions.insert(
                                        0,
                                        {
                                            "action": "finish",
                                            "reason": "all_files_created",
                                            "args": {"summary": f"Created: {created}"},
                                        },
                                    )
                            else:
                                step.result = f"[error] write failed: {write_result}"

                elif action == "propose_edit":
                    path = args.get("path", "")
                    instruction = args.get("instruction", "")
                    if self._is_code_edit_goal(goal) and not self._is_allowed_edit_path(
                        goal, path
                    ):
                        target = self._extract_target_file_from_goal(goal)
                        if target:
                            path = target
                            instruction = (
                                f"{instruction}\nKeep changes strictly in {target}."
                            )
                    code = propose_file_update(path, instruction, k=self.top_k)
                    self.staged_edits[path] = code
                    risk = patch_risk(path, code)
                    step.result = f"Staged edit for {path} ({len(code)} chars), risk={risk['level']}:{risk['score']}"
                    if self._fix_until_green and auto_apply and path:
                        self._forced_decisions = [
                            {
                                "action": "apply_edit",
                                "reason": "fix_until_green_apply_staged",
                                "args": {"path": path},
                            },
                            {
                                "action": "run_quality",
                                "reason": "fix_until_green_verify_patch",
                                "args": {},
                            },
                        ] + self._forced_decisions
                    elif self._generate_tests and path.endswith(".py"):
                        test_path = suggest_test_path(path)
                        if test_path not in self.staged_edits:
                            self._forced_decisions = [
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
                            ] + self._forced_decisions

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
                        changed_scope = [path]
                        if self._generate_tests and path.endswith(".py"):
                            gen = apply_generated_tests([path], limit=2)
                            generated = gen.get("applied", [])
                            if generated:
                                changed_scope.extend(generated)
                                self.event_logger.log(
                                    "generated_tests",
                                    {
                                        "source_files": [path],
                                        "generated_files": generated,
                                        "quality_ok_rate": gen.get(
                                            "quality_ok_rate", 0.0
                                        ),
                                    },
                                )
                                step.result = (
                                    f"Applied {path}; generated_tests={','.join(generated)}; "
                                    f"tests_quality_ok_rate={gen.get('quality_ok_rate', 0.0)}"
                                )

                        ok, details = run_quality_gate(
                            changed_files=changed_scope, command_timeout=90
                        )
                        if ok:
                            if "generated_tests=" in step.result:
                                step.result += "; quality gate passed"
                            else:
                                step.result = f"Applied {path}; quality gate passed"
                        else:
                            self.event_logger.log_failure(
                                "test",
                                "quality_gate_failed_after_apply_edit",
                                {"path": path, "details": details},
                            )
                            if self._fix_until_green:
                                flat_rows = []
                                flat_rows.extend(details.get("fast", []))
                                flat_rows.extend(details.get("full", []))
                                normalized = run_quality_pipeline_normalized(
                                    mode="fast",
                                    changed_files=[path],
                                    command_timeout=90,
                                )
                                failure_text = json.dumps(
                                    normalized if normalized else {"rows": flat_rows},
                                    ensure_ascii=False,
                                )
                                scheduled = self._enqueue_replan_after_failure(
                                    "quality",
                                    failure_text,
                                    auto_apply=auto_apply,
                                    fallback_path=path,
                                )
                                if not scheduled:
                                    self._forced_decisions.insert(
                                        0,
                                        {
                                            "action": "finish",
                                            "reason": "stop_max_replan_attempts",
                                            "args": {
                                                "summary": "Stopped: max replan attempts reached after quality failures"
                                            },
                                        },
                                    )
                                step.result = (
                                    f"Applied {path}; quality gate failed; replan queued "
                                    "in fix-until-green mode"
                                )
                            else:
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
                            changed_scope = list(changed)
                            if self._generate_tests:
                                gen = apply_generated_tests(changed, limit=3)
                                generated = gen.get("applied", [])
                                if generated:
                                    changed_scope.extend(generated)
                                    self.event_logger.log(
                                        "generated_tests",
                                        {
                                            "source_files": changed,
                                            "generated_files": generated,
                                            "quality_ok_rate": gen.get(
                                                "quality_ok_rate", 0.0
                                            ),
                                        },
                                    )
                                    step.result = (
                                        "Transaction applied; generated_tests="
                                        + ",".join(generated)
                                        + f"; tests_quality_ok_rate={gen.get('quality_ok_rate', 0.0)}"
                                    )

                            ok, details = run_quality_gate(
                                changed_files=changed_scope, command_timeout=90
                            )
                            if ok:
                                self.last_backups = backups
                                if "generated_tests=" in step.result:
                                    step.result += "; quality gate passed"
                                else:
                                    step.result = (
                                        "Transaction applied; quality gate passed"
                                    )
                            else:
                                self.event_logger.log_failure(
                                    "test",
                                    "quality_gate_failed_after_transaction",
                                    {"details": details},
                                )
                                if self._fix_until_green:
                                    normalized = run_quality_gate_normalized(
                                        changed_files=changed, command_timeout=90
                                    )
                                    scheduled = self._enqueue_replan_after_failure(
                                        "quality",
                                        json.dumps(normalized, ensure_ascii=False),
                                        auto_apply=auto_apply,
                                        fallback_path=changed[0] if changed else None,
                                    )
                                    if not scheduled:
                                        self._forced_decisions.insert(
                                            0,
                                            {
                                                "action": "finish",
                                                "reason": "stop_max_replan_attempts",
                                                "args": {
                                                    "summary": "Stopped: max replan attempts reached after quality failures"
                                                },
                                            },
                                        )
                                    step.result = (
                                        "Transaction applied then failed quality gate; "
                                        "replan queued in fix-until-green mode"
                                    )
                                else:
                                    rb = rollback_transaction(backups)
                                    if not rb:
                                        self.event_logger.log_failure(
                                            "rollback",
                                            "rollback_transaction_failed",
                                            {},
                                        )
                                    step.result = f"Transaction applied then failed quality gate; rollback={'ok' if rb else 'failed'}"

                elif action == "run_tests":
                    self._has_validation_action = True
                    command = args.get("command", AUTO_TEST_COMMAND)
                    step.result = run_tests(command)
                    if "exit_code=0" not in step.result:
                        self.event_logger.log_failure(
                            "test",
                            "run_tests_non_zero",
                            {"command": command, "result": step.result[:500]},
                        )
                        scheduled = self._enqueue_replan_after_failure(
                            "tests",
                            step.result,
                            auto_apply=auto_apply,
                        )
                        if not scheduled:
                            self._forced_decisions.insert(
                                0,
                                {
                                    "action": "finish",
                                    "reason": "stop_max_replan_attempts",
                                    "args": {
                                        "summary": "Stopped: max replan attempts reached after test failures"
                                    },
                                },
                            )
                    elif self._replan_attempts > 0:
                        recent = " | ".join(
                            f"{s.action}:{s.reason}" for s in self.steps[-6:]
                        )
                        remember_fix_strategy(
                            issue=f"tests failure recovered for goal: {goal[:120]}",
                            strategy=recent,
                            files=list(self.staged_edits.keys())[:6],
                        )

                elif action == "run_quality":
                    self._has_validation_action = True
                    changed_scope = list(self.staged_edits.keys()) or None
                    normalized = run_quality_pipeline_normalized(
                        mode="fast" if changed_scope else "full",
                        changed_files=changed_scope,
                        command_timeout=90,
                    )
                    ok = bool(normalized.get("ok"))
                    details = normalized.get("raw", [])
                    step.result = f"ok={ok}; failed_stage={normalized.get('failed_stage')}; details={details}"
                    if not ok:
                        self.event_logger.log_failure(
                            "test", "run_quality_failed", {"details": normalized}
                        )
                        stage = normalized.get("failed_stage") or "quality"
                        scheduled = self._enqueue_replan_after_failure(
                            str(stage),
                            json.dumps(normalized, ensure_ascii=False),
                            auto_apply=auto_apply,
                        )
                        if not scheduled:
                            self._forced_decisions.insert(
                                0,
                                {
                                    "action": "finish",
                                    "reason": "stop_max_replan_attempts",
                                    "args": {
                                        "summary": "Stopped: max replan attempts reached after quality failures"
                                    },
                                },
                            )
                    elif self._replan_attempts > 0:
                        recent = " | ".join(
                            f"{s.action}:{s.reason}" for s in self.steps[-6:]
                        )
                        remember_fix_strategy(
                            issue=f"quality failure recovered for goal: {goal[:120]}",
                            strategy=recent,
                            files=list(self.staged_edits.keys())[:6],
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
                    elapsed = round((datetime.utcnow() - start_ts).total_seconds(), 1)
                    print(f"\n[stella] Terminé en {elapsed}s\n")
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
        lines = [f"Final: {final_message}", "", "Decisions:"]
        for s in self.steps:
            lines.append(f"- step {s.step}: {s.action} | {s.reason} | {s.result[:180]}")
        return "\n".join(lines)
