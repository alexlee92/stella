"""
P1.1 — Planner extrait de auto_agent.py.

Responsabilités :
- Construire le prompt du planner
- Appeler le LLM pour obtenir une décision
- Corriger / réparer le schéma JSON si invalide
- Générer les contenus de fichiers via LLM
- Produire les décisions de fallback
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from agent.action_schema import validate_decision_schema
from agent.config import MAX_RETRIES_JSON

if TYPE_CHECKING:
    from agent.auto_agent import AutonomousAgent


class Planner:
    """Génère et valide les décisions de l'agent autonome."""

    def __init__(self, agent: "AutonomousAgent"):
        self.agent = agent

    # ------------------------------------------------------------------
    # Prompts
    # ------------------------------------------------------------------

    def prompt(self, goal: str) -> str:
        from agent.tooling import list_python_files

        history = (
            "\n".join(
                f"{s.step}. {s.action} | reason={s.reason} | result={s.result[:180]}"
                for s in self.agent.steps[-8:]
            )
            or "none"
        )
        files = list_python_files(limit=40)
        files_text = "\n".join(files) if files else "no python files"
        context = self.agent._summarize_context(goal)
        session_ctx = self.agent._session_context(max_chars_per_file=400)

        # P3.2 — Contexte ERP injecté si entités détectées dans le goal
        erp_ctx = ""
        try:
            from agent.erp_knowledge import get_erp_context

            erp_ctx = get_erp_context(goal)
        except Exception:
            pass

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
- web_search: {{"query": "search terms", "limit": 5}}
- generate_migration: {{"message": "add user table"}}
- apply_migration: {{"target": "head"}}
- read_schema: {{}} or {{"files": ["users/models.py"]}}
- plan_files: {{"description": "complete billing module with models, API, tests"}}
- mcp_call: {{"server": "database", "method": "tools/list"}}
- finish: {{"summary": "final answer for user"}}

Rules:
- Return strict JSON only.
- Use only listed actions and valid args.
- For any NEW functional module (CRM, ERP, Auth, etc.), always start by using web_search to research industry standards, required fields, and best practices.
- Use web_search only when you need external documentation, latest library versions, or to resolve a cryptic error you don't understand from local context.
- Use create_file to generate NEW files (backend, frontend, modules, configs) — any extension (.py, .js, .ts, .html, .css, etc.).
- Use propose_edit only to MODIFY existing files.
- Keep edits minimal and safe.
- Use git actions only when the goal explicitly asks for git/commit/pr operations.
- When goal says "generate tests for X/Y.py", create "tests/test_Y.py" (mirror the source filename with test_ prefix under tests/).
- Never create a file that already exists on disk — use propose_edit instead.
- When goal involves creating a multi-file module (ERP, CRM, API, etc.), use plan_files FIRST to get an ordered file manifest before any create_file.
- Use read_schema before generating code that interacts with existing DB models — it injects the current table/column definitions into context.
- Use generate_migration after modifying SQLAlchemy models, then apply_migration to upgrade the DB.

Return format:
{{"action":"...","reason":"short reason","args":{{...}}}}

Valid examples:
{{"action":"create_file","reason":"generate tests for users/api.py","args":{{"path":"tests/test_api.py","description":"pytest tests for users/api.py Flask Blueprint: test create/get/update/delete/login endpoints using Flask test client"}}}}
{{"action":"create_file","reason":"create user model","args":{{"path":"users/models.py","description":"SQLAlchemy User model with id, email, hashed_password, created_at fields and CRUD methods"}}}}
{{"action":"search_code","reason":"find implementation points","args":{{"pattern":"run_quality_pipeline","limit":20}}}}
{{"action":"read_file","reason":"inspect target file","args":{{"path":"agent/auto_agent.py"}}}}
{{"action":"finish","reason":"task completed","args":{{"summary":"Explained how pr-ready works"}}}}

Project files:
{files_text}

Relevant indexed context:
{context}

Files created in this session (use their content as context for new files):
{session_ctx if session_ctx else "none yet"}

{erp_ctx if erp_ctx else ""}

Recent steps:
{history}
"""

    def schema_repair_prompt(self, goal: str, decision: dict, schema_error: str) -> str:
        import json

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
  git_branch, git_commit, git_diff, web_search,
  generate_migration, apply_migration, read_schema, plan_files, mcp_call, finish
- args must match action:
  read_file->{{"path":"..."}}
  read_many->{{"paths":["..."]}}
  search_code->{{"pattern":"...","limit":20?}}
  propose_edit->{{"path":"...","instruction":"..."}}
  apply_edit->{{"path":"..."}}
  run_tests->{{"command":"..."}}
  web_search->{{"query":"..."}}
  finish->{{"summary":"..."}}
- never add unknown keys in args.
- keep decision useful for the goal.
"""

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, goal: str) -> dict:
        """Appelle le LLM pour obtenir une décision brute (sans critique)."""
        from agent.decision_normalizer import (
            autocorrect_decision_schema,
            coerce_decision,
            normalize_decision,
        )

        ag = self.agent
        raw_decision = ag._llm_fn(
            self.prompt(goal),
            retries=MAX_RETRIES_JSON,
            prompt_class="planner",
            task_type="planning",
        )
        decision = normalize_decision(raw_decision)
        ag.event_logger.log("plan", {"goal": goal, "decision": decision})

        if decision.get("_error_type") == "parse":
            parse_meta = decision.get("_parse_meta") or {}
            ag.event_logger.log_failure(
                "parse",
                "planner_json_parse_failed",
                {
                    "decision": decision,
                    "parse_class": parse_meta.get("error_class", "unknown_parse_error"),
                    "parse_attempts": parse_meta.get("attempt_count", 0),
                    "prompt_class": parse_meta.get("prompt_class", "planner"),
                },
            )
            ag._parse_fallback_count += 1
            return self.fallback_decision(goal, reason="parse_failed")
        ag._parse_fallback_count = 0

        decision = coerce_decision(goal, decision)
        ok, msg = validate_decision_schema(decision)
        if not ok:
            corrected = autocorrect_decision_schema(goal, decision, msg)
            c_ok, _ = validate_decision_schema(corrected)
            if c_ok:
                ag.event_logger.log(
                    "schema_autocorrect",
                    {"from": decision, "to": corrected, "issue": msg},
                )
                decision = corrected
            else:
                repaired = ag._llm_fn(
                    self.schema_repair_prompt(goal, decision, msg),
                    retries=1,
                    prompt_class="planner_schema_repair",
                    task_type="planning",
                )
                repaired = normalize_decision(repaired)
                r_ok, r_msg = validate_decision_schema(repaired)
                if r_ok:
                    ag.event_logger.log(
                        "schema_repair",
                        {"from": decision, "to": repaired, "issue": msg},
                    )
                    decision = repaired
                else:
                    ag.event_logger.log_failure(
                        "parse",
                        f"planner_schema_invalid:{msg}",
                        {
                            "decision": decision,
                            "parse_class": "schema_invalid",
                            "prompt_class": "planner",
                            "repair_schema_error": r_msg,
                        },
                    )
                    return self.fallback_decision(goal, reason=f"schema_invalid:{msg}")

        return decision

    def fallback_decision(self, goal: str, reason: str) -> dict:
        """Produit une décision de fallback si le LLM échoue."""
        from agent.decision_normalizer import extract_target_file_from_goal

        ag = self.agent
        target = extract_target_file_from_goal(goal)
        if target:
            return {
                "action": "propose_edit",
                "reason": f"fallback_{reason}",
                "args": {"path": target, "instruction": goal[:500]},
            }

        low = (goal or "").lower()
        if ag._parse_fallback_count >= 3:
            return {
                "action": "finish",
                "reason": f"fallback_{reason}",
                "args": {
                    "summary": "Planner parse unstable after 3 retries; stopping early"
                },
            }

        if any(k in low for k in ["latence", "performance", "vitesse", "speed"]):
            if ag._parse_fallback_count % 2 == 1:
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
            if ag._parse_fallback_count % 2 == 1:
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

        if any(
            k in low
            for k in ["web", "search", "recherche", "doc", "documentation", "online"]
        ):
            return {
                "action": "web_search",
                "reason": f"fallback_{reason}",
                "args": {"query": goal[:200], "limit": 5},
            }

        return {
            "action": "list_files",
            "reason": f"fallback_{reason}",
            "args": {"limit": 40, "ext": ".py"},
        }

    # ------------------------------------------------------------------
    # File generation
    # ------------------------------------------------------------------

    def generate_file_content(self, goal: str, path: str, description: str) -> str:
        """Génère le contenu complet d'un nouveau fichier via le LLM."""
        from agent.llm_interface import ask_llm

        ext = os.path.splitext(path)[1].lower()
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
        context = self.agent._summarize_context_multi(
            goal, path=path, description=description
        )

        prompt = (
            "You are an expert software engineer. Generate the COMPLETE content of a new file.\n\n"
            f"File path  : {path}\n"
            f"Language   : {lang}\n"
            f"Description: {description}\n\n"
            f"Overall project goal: {goal}\n\n"
            f"Relevant project context (including files already created this session):\n{context}\n\n"
            "Rules:\n"
            "- Return ONLY the raw file content. No explanation, no markdown fences.\n"
            "- The file must be complete and functional — no TODOs, no stubs.\n"
            f"- Follow best practices for {lang}.\n"
            "- Include all necessary imports/dependencies.\n"
            "- For Python: use type hints, docstrings, proper error handling.\n"
            "- For JS/TS/JSX: use modern syntax, proper exports.\n"
            "- For HTML: include <!DOCTYPE html> and full document structure.\n"
        )
        raw = ask_llm(prompt, task_type="backend")
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
