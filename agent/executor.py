"""
P2.1 â€” Action executor extracted from AutonomousAgent.

Handles the execution of individual agent actions (read, edit, test, etc.).
"""

import json
import os
import re
from typing import TYPE_CHECKING

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
    web_search,
)

if TYPE_CHECKING:
    from agent.auto_agent import AutonomousAgent


def _validate_file_syntax(path: str, content: str) -> str | None:
    """M5: Validates syntax for .json, .js/.ts/.tsx/.jsx, and .html files.

    Returns None if OK, "warning: <message>" if a problem is detected.
    Does not raise â€” validation errors are non-blocking.
    """
    import subprocess as _sp
    import tempfile

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            return f"warning: JSON syntax error: {e}"
        return None

    if ext in (".js", ".jsx", ".ts", ".tsx"):
        try:
            with tempfile.NamedTemporaryFile(
                suffix=ext, mode="w", encoding="utf-8", delete=False
            ) as tf:
                tf.write(content)
                tmp_path = tf.name
            try:
                result = _sp.run(
                    ["node", "--check", tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    shell=False,
                )
                if result.returncode != 0:
                    err = (result.stderr or result.stdout or "").strip()
                    return f"warning: JS/TS syntax: {err[:200]}"
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except FileNotFoundError:
            pass  # node not available â€” skip
        except Exception:
            pass
        return None

    if ext in (".html", ".htm"):
        from html.parser import HTMLParser

        class _CheckParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.errors: list[str] = []

            def unknown_decl(self, data: str):
                self.errors.append(f"unknown_decl: {data[:60]}")

        parser = _CheckParser()
        try:
            parser.feed(content)
        except Exception as e:
            return f"warning: HTML parsing error: {e}"
        if parser.errors:
            return f"warning: HTML issues: {'; '.join(parser.errors[:3])}"
        return None

    return None


class ActionExecutor:
    """Executes a single agent action and returns the result string."""

    def __init__(self, agent: "AutonomousAgent"):
        self.agent = agent

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(
        self,
        action: str,
        args: dict,
        goal: str,
        auto_apply: bool,
        index: int,
        reason: str = "",
    ) -> str:
        """Execute *action* with *args* and return a result string.

        Returns a result string. May modify agent state (staged_edits, forced_decisions, etc.).
        Raises on unexpected errors (caller should catch).
        """
        # P1.1 â€” Protection mode lecture seule
        writing_actions = {
            "create_file",
            "propose_edit",
            "apply_edit",
            "apply_all_staged",
            "delete_file",
            "move_file",
            "git_branch",
            "git_commit",
        }
        if self.agent.read_only and action in writing_actions:
            self.agent.event_logger.log("read_only_blocked", {"action": action})
            return f"[blocked] Action '{action}' refusÃ©e en mode LECTURE SEULE."

        handler = getattr(self, f"_exec_{action}", None)
        if handler is None:
            self.agent.event_logger.log_failure(
                "tool", "unknown_action", {"action": action, "args": args}
            )
            return f"Unknown action: {action}"
        return handler(
            args, goal=goal, auto_apply=auto_apply, index=index, reason=reason
        )

    def _fix_missing_imports(self, path: str, content: str, goal: str) -> str:
        """Tentative de rÃ©solution automatique des imports manquants via Ruff + search_code."""
        if not path.endswith(".py"):
            return content

        import subprocess
        from agent.tooling import search_code

        # 1. Utiliser ruff pour dÃ©tecter les noms non dÃ©finis (F821)
        # On Ã©crit temporairement pour que ruff puisse lire si besoin,
        # ou on pipe le contenu. Ici on va utiliser ruff check --stdin-filename
        try:
            # I8-fix: shell=False pour la sÃ©curitÃ©
            cmd = [
                "python",
                "-m",
                "ruff",
                "check",
                "--select",
                "F821",
                "--output-format",
                "json",
                "-",
            ]
            proc = subprocess.run(
                cmd, input=content, text=True, capture_output=True, shell=False
            )
            if proc.returncode == 0 and not proc.stdout.strip():
                return content  # Pas d'erreurs F821

            errors = json.loads(proc.stdout)
            undefined_names = set()
            for err in errors:
                # Extraire le nom du message ruff (souvent "Undefined name `Name`")
                match = re.search(r"`([^`]+)`", err.get("message", ""))
                if match:
                    undefined_names.add(match.group(1))

            if not undefined_names:
                return content

            print(
                f"     [auto-import] dÃ©tection de {len(undefined_names)} noms non dÃ©finis : {list(undefined_names)}"
            )

            new_imports = []
            common_fallbacks = {
                "create_engine": "from sqlalchemy import create_engine",
                "sessionmaker": "from sqlalchemy.orm import sessionmaker",
                "relationship": "from sqlalchemy.orm import relationship",
                "backref": "from sqlalchemy.orm import backref",
                "declarative_base": "from sqlalchemy.ext.declarative import declarative_base",
                "Column": "from sqlalchemy import Column",
                "Integer": "from sqlalchemy import Integer",
                "String": "from sqlalchemy import String",
                "DateTime": "from sqlalchemy import DateTime",
                "Float": "from sqlalchemy import Float",
                "Boolean": "from sqlalchemy import Boolean",
                "ForeignKey": "from sqlalchemy import ForeignKey",
                "jsonify": "from flask import jsonify",
                "request": "from flask import request",
                "Blueprint": "from flask import Blueprint",
                "Flask": "from flask import Flask",
                "datetime": "import datetime",
                "timedelta": "from datetime import timedelta",
                "os": "import os",
                "sys": "import sys",
                "json": "import json",
                "time": "import time",
                "re": "import re",
                "logging": "import logging",
                "requests": "import requests",
                "pd": "import pandas as pd",
                "np": "import numpy as np",
                "plt": "import matplotlib.pyplot as plt",
                "pytest": "import pytest",
                "unittest": "import unittest",
                "ABC": "from abc import ABC",
                "abstractmethod": "from abc import abstractmethod",
                "Any": "from typing import Any",
                "List": "from typing import List",
                "Dict": "from typing import Dict",
                "Optional": "from typing import Optional",
                "Union": "from typing import Union",
            }

            for name in undefined_names:
                # 1. Chercher dans le projet
                found_locally = False
                matches = search_code(f"class {name}|def {name}", limit=5)
                for m in matches:
                    parts = m.split(":", 2)
                    if len(parts) >= 3:
                        m_path = parts[0].replace("\\", "/")
                        if m_path == path.replace("\\", "/"):
                            continue
                        m_mod = m_path.replace(".py", "").replace("/", ".")
                        new_imports.append(f"from {m_mod} import {name}")
                        found_locally = True
                        break

                # 2. Si non trouvÃ© localement, chercher dans les fallbacks communs
                if not found_locally and name in common_fallbacks:
                    new_imports.append(common_fallbacks[name])

            if new_imports:
                import_block = "\n".join(new_imports) + "\n"
                print(f"     [auto-import] ajout de : {import_block.strip()}")
                # InsÃ©rer au dÃ©but du fichier (aprÃ¨s les docstrings si possible, sinon tout en haut)
                if content.startswith('"""'):
                    end_ds = content.find('"""', 3)
                    if end_ds != -1:
                        return (
                            content[: end_ds + 3]
                            + "\n"
                            + import_block
                            + content[end_ds + 3 :]
                        )
                return import_block + content

        except Exception as e:
            print(f"     [auto-import] Ã©chec : {e}")

        return content

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
        return (
            read_many(paths)
            if isinstance(paths, list) and paths
            else "Missing paths list"
        )

    def _exec_search_code(self, args: dict, **_kw) -> str:
        pattern = args.get("pattern", "")
        limit = int(args.get("limit", 20))
        matches = search_code(pattern, limit=limit) if pattern else []
        return "\n".join(matches) if matches else "No matches"

    def _exec_create_file(self, args: dict, goal: str, **_kw) -> str:
        ag = self.agent
        decision_reason = _kw.get("reason", "")
        path = args.get("path", "")
        description = args.get("description", "")
        auto_apply = bool(_kw.get("auto_apply", False))
        if not path or not description:
            return "[error] create_file requires 'path' and 'description'"
        if path in ag.staged_edits:
            return f"Skipped: {path} already created this session"
        if os.path.isfile(os.path.join(PROJECT_ROOT, path)):
            result = f"Skipped: {path} already exists on disk (use fix to modify it)"
            all_targets = extract_all_target_files_from_goal(goal)
            if not all_targets and not ag._forced_decisions:
                ag._forced_decisions.insert(
                    0,
                    {
                        "action": "finish",
                        "reason": "all_target_files_already_exist",
                        "args": {
                            "summary": f"All target files already exist on disk: {path}"
                        },
                    },
                )
            return result

        print(f"     gÃ©nÃ¨re le contenu de {path}...", flush=True)
        content = ag._generate_file_content(goal, path, description)
        if not content:
            # R1 â€” Abort remaining plan_files creations on failure
            if decision_reason == "plan_files_ordered_creation":
                ag._forced_decisions = [
                    d
                    for d in ag._forced_decisions
                    if not (
                        d.get("action") == "create_file"
                        and d.get("reason") == "plan_files_ordered_creation"
                    )
                ]
                ag._forced_decisions.insert(
                    0,
                    {
                        "action": "finish",
                        "reason": "plan_files_aborted",
                        "args": {
                            "summary": f"plan_files aborted: empty content for {path}"
                        },
                    },
                )
            return f"[error] LLM returned empty content for {path}"

        # P1.1 â€” Tentative de correction automatique des imports manquants
        if path.endswith(".py"):
            fixed_content = self._fix_missing_imports(path, content, goal)
            if fixed_content != content:
                content = fixed_content
                print(f"     [auto-import] contenu de {path} mis Ã  jour.")

        # M5 â€” Validation syntaxe non-Python (JSON, JS/TS, HTML)
        syntax_warning = _validate_file_syntax(path, content)

        if auto_apply:
            write_result = write_new_file(path, content)
            if not write_result.startswith("ok:"):
                # R1 â€” Abort remaining plan_files creations on write failure
                if decision_reason == "plan_files_ordered_creation":
                    ag._forced_decisions = [
                        d
                        for d in ag._forced_decisions
                        if not (
                            d.get("action") == "create_file"
                            and d.get("reason") == "plan_files_ordered_creation"
                        )
                    ]
                    ag._forced_decisions.insert(
                        0,
                        {
                            "action": "finish",
                            "reason": "plan_files_aborted",
                            "args": {
                                "summary": f"plan_files aborted: write failed for {path}: {write_result}"
                            },
                        },
                    )
                return f"[error] write failed: {write_result}"

        ag.staged_edits[path] = content
        risk = patch_risk(path, content)
        n_chunks = index_file_in_session(path, content)
        if decision_reason == "plan_files_ordered_creation":
            deps_result = "deps:skipped (fast plan_files mode)"
        else:
            deps_result = detect_and_install_deps(path, content)
        mode = "Created" if auto_apply else "Staged creation for"
        result = (
            f"{mode} {path} ({len(content)} chars), "
            f"risk={risk['level']}:{risk['score']}, "
            f"indexed={n_chunks} chunks, "
            f"{deps_result}"
        )
        if syntax_warning:
            result += f" [syntax-warning: {syntax_warning}]"

        # For plan_files queue, continue creating all queued files before quality/finalize.
        if decision_reason == "plan_files_ordered_creation":
            remaining_plan_actions = any(
                (d.get("reason") or "").startswith("plan_files_ordered_")
                for d in ag._forced_decisions
            )
            if remaining_plan_actions:
                return result

        # Chain next files from goal
        all_targets = extract_all_target_files_from_goal(goal)
        pending = [t for t in all_targets if t not in ag.staged_edits]
        if pending:
            next_path = pending[0]
            ag._forced_decisions.insert(
                0,
                {
                    "action": "create_file",
                    "reason": "multi_file_goal_continuation",
                    "args": {
                        "path": next_path,
                        "description": (
                            f"{goal[:400]} "
                            f"(already created: {', '.join(ag.staged_edits.keys())})"
                        ),
                    },
                },
            )
        else:
            created = ", ".join(ag.staged_edits.keys())
            if auto_apply:
                ag._forced_decisions = [
                    {
                        "action": "run_quality",
                        "reason": "validate_created_files",
                        "args": {},
                    },
                    {
                        "action": "finish",
                        "reason": "all_files_created",
                        "args": {"summary": f"Created: {created}"},
                    },
                ] + ag._forced_decisions
            else:
                ag._forced_decisions.insert(
                    0,
                    {
                        "action": "finish",
                        "reason": "all_files_staged",
                        "args": {
                            "summary": (
                                f"Staged new files (not yet written): {created}. "
                                "Re-run with --apply to write changes."
                            )
                        },
                    },
                )
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

        # B4-fix: injecter le contenu des fichiers crÃ©Ã©s dans la session
        # pour que le LLM puisse voir les dÃ©pendances non encore Ã©crites sur disque.
        session_ctx = ag._session_context(max_chars_per_file=3000)
        if session_ctx:
            instruction = (
                f"{instruction}\n\n"
                "Files created/modified in this session (use as context for imports and types):\n"
                f"{session_ctx}"
            )

        code = propose_file_update(path, instruction, k=ag.top_k)
        ag.staged_edits[path] = code
        risk = patch_risk(path, code)
        result = f"Staged edit for {path} ({len(code)} chars), risk={risk['level']}:{risk['score']}"
        auto_apply = _kw.get("auto_apply", False)
        if ag._fix_until_green and auto_apply and path:
            ag._forced_decisions = [
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

    def _exec_apply_edit(
        self, args: dict, goal: str, auto_apply: bool, index: int, **_kw
    ) -> str:
        ag = self.agent
        path = args.get("path", "")
        code = ag.staged_edits.get(path)
        if not path or code is None:
            ag.event_logger.log_failure(
                "tool",
                "apply_edit_without_staged",
                {"path": path, "staged_files": list(ag.staged_edits.keys())},
            )
            return "[no_staged_edit]"

        if not auto_apply or (ag.interactive and not ag._confirm_apply([path])):
            return f"Staged only for {path}. Re-run with --apply to apply."

        patch_result = apply_suggestion(path, code, interactive=False)
        backup_path = (
            patch_result.get("backup_path") if isinstance(patch_result, dict) else None
        )
        changed_scope = [path]

        result = ""
        if ag._generate_tests and path.endswith(".py"):
            gen = apply_generated_tests([path], limit=2)
            generated = gen.get("applied", [])
            if generated:
                changed_scope.extend(generated)
                ag.event_logger.log(
                    "generated_tests",
                    {
                        "source_files": [path],
                        "generated_files": generated,
                        "quality_ok_rate": gen.get("quality_ok_rate", 0.0),
                    },
                )
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
                "test",
                "quality_gate_failed_after_apply_edit",
                {"path": path, "details": details},
            )
            if ag._fix_until_green:
                normalized = run_quality_pipeline_normalized(
                    mode="fast",
                    changed_files=[path],
                    command_timeout=90,
                )
                failure_text = json.dumps(
                    normalized or {"rows": []}, ensure_ascii=False
                )
                scheduled = ag._enqueue_replan_after_failure(
                    "quality",
                    failure_text,
                    auto_apply=auto_apply,
                    fallback_path=path,
                )
                if not scheduled:
                    ag._forced_decisions.insert(
                        0,
                        {
                            "action": "finish",
                            "reason": "stop_max_replan_attempts",
                            "args": {
                                "summary": "Stopped: max replan attempts reached after quality failures"
                            },
                        },
                    )
                result = f"Applied {path}; quality gate failed; replan queued in fix-until-green mode"
            else:
                # P1.1 â€” Interactive force apply
                force_apply = False
                if ag.interactive:
                    try:
                        print(f"\n  [!] Ã‰chec du Quality Gate pour {path}.")
                        ans = (
                            input("      Garder les modifications quand mÃªme ? [y/n] ")
                            .strip()
                            .lower()
                        )
                        force_apply = ans in {"y", "yes", "o", "oui"}
                    except (EOFError, KeyboardInterrupt):
                        force_apply = False

                if force_apply:
                    result = f"Applied {path}; quality gate failed (forced by user)"
                else:
                    restore_ok = restore_backup(path, backup_path)
                    if not restore_ok:
                        ag.event_logger.log_failure(
                            "rollback",
                            "rollback_failed_after_apply_edit",
                            {"path": path, "backup_path": backup_path},
                        )
                    result = f"Applied {path}; quality gate failed; rollback={'ok' if restore_ok else 'failed'}"
        return result

    def _exec_apply_all_staged(
        self, args: dict, goal: str, auto_apply: bool, **_kw
    ) -> str:
        ag = self.agent
        if not ag.staged_edits:
            return "No staged edits"
        if not auto_apply or (
            ag.interactive and not ag._confirm_apply(list(ag.staged_edits.keys()))
        ):
            return "Staged only. Re-run with --apply to apply transaction."

        success, backups, msg = apply_transaction(ag.staged_edits)
        if not success:
            ag.event_logger.log_failure(
                "tool", "apply_transaction_failed", {"message": msg}
            )
            return msg

        changed = list(ag.staged_edits.keys())
        changed_scope = list(changed)
        result = ""

        if ag._generate_tests:
            gen = apply_generated_tests(changed, limit=3)
            generated = gen.get("applied", [])
            if generated:
                changed_scope.extend(generated)
                ag.event_logger.log(
                    "generated_tests",
                    {
                        "source_files": changed,
                        "generated_files": generated,
                        "quality_ok_rate": gen.get("quality_ok_rate", 0.0),
                    },
                )
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
                "test",
                "quality_gate_failed_after_transaction",
                {"details": details},
            )
            if ag._fix_until_green:
                normalized = run_quality_gate_normalized(
                    changed_files=changed, command_timeout=90
                )
                scheduled = ag._enqueue_replan_after_failure(
                    "quality",
                    json.dumps(normalized, ensure_ascii=False),
                    auto_apply=auto_apply,
                    fallback_path=changed[0] if changed else None,
                )
                if not scheduled:
                    ag._forced_decisions.insert(
                        0,
                        {
                            "action": "finish",
                            "reason": "stop_max_replan_attempts",
                            "args": {
                                "summary": "Stopped: max replan attempts reached after quality failures"
                            },
                        },
                    )
                result = "Transaction applied then failed quality gate; replan queued in fix-until-green mode"
            else:
                # P1.1 â€” Interactive force apply
                force_apply = False
                if ag.interactive:
                    try:
                        print(
                            f"\n  [!] Ã‰chec du Quality Gate pour la transaction ({len(changed)} fichiers)."
                        )
                        ans = (
                            input("      Garder les modifications quand mÃªme ? [y/n] ")
                            .strip()
                            .lower()
                        )
                        force_apply = ans in {"y", "yes", "o", "oui"}
                    except (EOFError, KeyboardInterrupt):
                        force_apply = False

                if force_apply:
                    ag.last_backups = backups
                    result = (
                        "Transaction applied then failed quality gate (forced by user)"
                    )
                else:
                    rb = rollback_transaction(backups)
                    if not rb:
                        ag.event_logger.log_failure(
                            "rollback", "rollback_transaction_failed", {}
                        )
                    result = f"Transaction applied then failed quality gate; rollback={'ok' if rb else 'failed'}"
        return result

    def _exec_run_tests(self, args: dict, auto_apply: bool = False, **_kw) -> str:
        ag = self.agent
        ag._has_validation_action = True
        command = args.get("command", AUTO_TEST_COMMAND)
        result = run_tests(command)
        if "exit_code=0" not in result:
            ag.event_logger.log_failure(
                "test",
                "run_tests_non_zero",
                {"command": command, "result": result[:500]},
            )
            scheduled = ag._enqueue_replan_after_failure(
                "tests", result, auto_apply=auto_apply
            )
            if not scheduled:
                ag._forced_decisions.insert(
                    0,
                    {
                        "action": "finish",
                        "reason": "stop_max_replan_attempts",
                        "args": {
                            "summary": "Stopped: max replan attempts reached after test failures"
                        },
                    },
                )
        elif ag._replan_attempts > 0:
            recent = " | ".join(f"{s.action}:{s.reason}" for s in ag.steps[-6:])
            remember_fix_strategy(
                issue=f"tests failure recovered for goal: {_kw.get('goal', '')[:120]}",
                strategy=recent,
                files=list(ag.staged_edits.keys())[:6],
            )
        return result

    def _exec_run_quality(
        self, args: dict, auto_apply: bool = False, goal: str = "", **_kw
    ) -> str:
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
        result = (
            f"ok={ok}; failed_stage={normalized.get('failed_stage')}; details={details}"
        )
        if not ok:
            ag.event_logger.log_failure(
                "test", "run_quality_failed", {"details": normalized}
            )
            stage = normalized.get("failed_stage") or "quality"
            scheduled = ag._enqueue_replan_after_failure(
                str(stage),
                json.dumps(normalized, ensure_ascii=False),
                auto_apply=auto_apply,
            )
            if not scheduled:
                ag._forced_decisions.insert(
                    0,
                    {
                        "action": "finish",
                        "reason": "stop_max_replan_attempts",
                        "args": {
                            "summary": "Stopped: max replan attempts reached after quality failures"
                        },
                    },
                )
        elif ag._replan_attempts > 0:
            recent = " | ".join(f"{s.action}:{s.reason}" for s in ag.steps[-6:])
            remember_fix_strategy(
                issue=f"quality failure recovered for goal: {goal[:120]}",
                strategy=recent,
                files=list(ag.staged_edits.keys())[:6],
            )
        return result

    # ------------------------------------------------------------------
    # M1 â€” delete_file
    # ------------------------------------------------------------------

    def _exec_delete_file(self, args: dict, **_kw) -> str:
        from agent.tooling import _resolve_path, invalidate_tool_cache

        path = args.get("path", "")
        if not path:
            return "[error] delete_file requires 'path'"
        try:
            abs_path = _resolve_path(path)
        except ValueError as exc:
            return f"[error] {exc}"
        if not os.path.isfile(abs_path):
            return f"[error] File not found: {path}"
        from agent.patcher import create_backup

        backup = create_backup(path)
        os.remove(abs_path)
        invalidate_tool_cache()
        # Retirer des staged_edits si prÃ©sent
        self.agent.staged_edits.pop(path, None)
        return f"Deleted {path} (backup: {backup})"

    # ------------------------------------------------------------------
    # M2 â€” move_file
    # ------------------------------------------------------------------

    def _exec_move_file(self, args: dict, **_kw) -> str:
        import shutil
        from agent.tooling import _resolve_path, invalidate_tool_cache

        src = args.get("src", "")
        dst = args.get("dst", "")
        if not src or not dst:
            return "[error] move_file requires 'src' and 'dst'"
        try:
            abs_src = _resolve_path(src)
            abs_dst = _resolve_path(dst)
        except ValueError as exc:
            return f"[error] {exc}"
        if not os.path.isfile(abs_src):
            return f"[error] Source not found: {src}"
        if os.path.exists(abs_dst):
            return f"[error] Destination already exists: {dst}"
        os.makedirs(os.path.dirname(abs_dst), exist_ok=True)
        shutil.move(abs_src, abs_dst)
        invalidate_tool_cache()
        # Mettre Ã  jour staged_edits si src Ã©tait staged
        ag = self.agent
        if src in ag.staged_edits:
            ag.staged_edits[dst] = ag.staged_edits.pop(src)
        return f"Moved {src} â†’ {dst}"

    # ------------------------------------------------------------------
    # M3 â€” read_file_range
    # ------------------------------------------------------------------

    def _exec_read_file_range(self, args: dict, **_kw) -> str:
        from agent.tooling import _resolve_path

        path = args.get("path", "")
        start_line = int(args.get("start_line", 1))
        end_line = int(args.get("end_line", 50))
        if not path:
            return "[error] read_file_range requires 'path', 'start_line', 'end_line'"
        try:
            abs_path = _resolve_path(path)
        except ValueError as exc:
            return f"[error] {exc}"
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError as exc:
            return f"[error] Cannot read {path}: {exc}"
        total = len(lines)
        s = max(0, start_line - 1)
        e = min(total, end_line)
        excerpt = "".join(
            f"{i + 1:4d} | {l}" for i, l in enumerate(lines[s:e], start=s)
        )
        return f"{path} (lines {start_line}-{end_line} of {total}):\n{excerpt}"

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

    def _exec_web_search(self, args: dict, **_kw) -> str:
        query = args.get("query", "")
        limit = int(args.get("limit", 5))
        goal = (_kw.get("goal") or "").strip()

        def _enqueue_local_plan_fallback() -> None:
            low_goal = goal.lower()
            creation_hints = (
                "create",
                "crÃ©er",
                "creer",
                "gÃ©nÃ©rer",
                "generer",
                "module",
                "frontend",
                "backend",
            )
            if not (goal and any(h in low_goal for h in creation_hints)):
                return
            ag = self.agent
            already_queued = any(
                d.get("action") == "plan_files"
                and d.get("reason") == "fallback_after_web_error"
                for d in ag._forced_decisions
            )
            if not already_queued:
                ag._forced_decisions = [
                    {
                        "action": "plan_files",
                        "reason": "fallback_after_web_error",
                        "args": {"description": goal[:500]},
                    }
                ] + ag._forced_decisions

        if not query:
            return "[error] web_search requires a 'query'"

        # P2.5 â€” Consulter le cache de documentation avant une vraie recherche
        try:
            from agent.global_memory import cache_web_result, search_doc_cache

            cached = search_doc_cache(query, k=1)
            if cached:
                _, cached_text = cached[0]
                print(
                    "     [web_search] rÃ©sultat depuis le cache doc global", flush=True
                )
                if "No web results found" in cached_text:
                    _enqueue_local_plan_fallback()
                    return (
                        f"[cached] {cached_text}\n"
                        "[fallback] cache web insuffisant -> planification locale du module"
                    )
                return f"[cached] {cached_text}"
        except Exception:
            cache_web_result = None  # type: ignore[assignment]

        result = web_search(query, limit=limit)

        if isinstance(result, str) and (
            result.startswith("[error]") or "No web results found" in result
        ):
            _enqueue_local_plan_fallback()
            return f"{result}\n[fallback] web indisponible/insuffisant -> planification locale du module"

        # P2.5 â€” Sauvegarder le rÃ©sultat pour les sessions futures
        try:
            if (
                cache_web_result is not None
                and result
                and not result.startswith("[error]")
            ):
                cache_web_result(query, result)
        except Exception:
            pass

        return result

    def _exec_mcp_call(self, args: dict, **_kw) -> str:
        from agent.mcp_client import format_mcp_result, mcp_call

        server = args.get("server", "")
        method = args.get("method", "")
        params = args.get("params") or {}
        if not server or not method:
            return "[error] mcp_call requires 'server' and 'method'"
        result = mcp_call(server, method, params)
        return format_mcp_result(result)

    def _exec_read_schema(self, args: dict, **_kw) -> str:
        from agent.schema_reader import read_schema

        files = args.get("files") or None
        summary = read_schema(files=files)
        return summary

    def _exec_plan_files(self, args: dict, goal: str, auto_apply: bool, **_kw) -> str:
        ag = self.agent
        description = args.get("description", goal)
        if not description:
            return "[error] plan_files requires a 'description'"

        def _local_default_plan(desc: str) -> list[dict]:
            low = (desc or "").lower()
            wants_react = "react" in low or "frontend" in low
            wants_backend = "backend" in low or "api" in low or "module" in low
            plan: list[dict] = []

            if wants_backend:
                plan.extend(
                    [
                        {
                            "path": "users/__init__.py",
                            "description": "Initialize users backend package",
                        },
                        {
                            "path": "users/models.py",
                            "description": "User domain model and role definitions",
                        },
                        {
                            "path": "users/schemas.py",
                            "description": "Validation schemas for user payloads",
                        },
                        {
                            "path": "users/service.py",
                            "description": "Business logic for create/list/update/authenticate users",
                        },
                        {
                            "path": "users/api.py",
                            "description": "HTTP endpoints for user CRUD and auth actions",
                        },
                    ]
                )
            if wants_react:
                plan.extend(
                    [
                        {
                            "path": "frontend/src/modules/users/api.ts",
                            "description": "HTTP client for users module",
                        },
                        {
                            "path": "frontend/src/modules/users/types.ts",
                            "description": "TypeScript types for users",
                        },
                        {
                            "path": "frontend/src/modules/users/UsersPage.tsx",
                            "description": "Users management page",
                        },
                        {
                            "path": "frontend/src/modules/users/components/UserForm.tsx",
                            "description": "Create/Edit user form",
                        },
                        {
                            "path": "frontend/src/modules/users/components/UserTable.tsx",
                            "description": "Users list table",
                        },
                    ]
                )
            if not plan:
                plan = [
                    {
                        "path": "users/__init__.py",
                        "description": "Initialize users module",
                    },
                    {"path": "users/models.py", "description": "User model"},
                    {"path": "users/api.py", "description": "Users API"},
                ]
            return plan

        from agent.llm_interface import ask_llm_json

        prompt = (
            "You are a software architect planning a multi-file module.\n\n"
            f"Goal: {description}\n\n"
            "Return a JSON array of file objects in dependency order (models first, then APIs, then tests).\n"
            'Each object: {"path": "relative/path.ext", "description": "what this file does"}\n\n'
            "Rules:\n"
            "- Include all files needed (models, schemas, services, routes, tests, __init__.py, etc.)\n"
            "- Use paths like: users/models.py, users/api.py, users/schemas.py, tests/test_users.py\n"
            "- Maximum 20 files.\n"
            "- Return ONLY a raw JSON array. No explanation, no markdown fences.\n"
            '- Example: [{"path":"users/models.py","description":"SQLAlchemy User model"}, ...]\n'
        )

        raw = ask_llm_json(
            prompt, retries=2, prompt_class="plan_files", task_type="planning"
        )

        # B1-fix: ask_llm_json peut maintenant retourner une list ou un dict
        file_plan: list[dict] = []
        if isinstance(raw, list):
            file_plan = [
                f
                for f in raw
                if isinstance(f, dict) and "path" in f and "description" in f
            ]
        elif isinstance(raw, dict):
            if "files" in raw:
                file_plan = raw["files"]
            elif "path" in raw and "description" in raw:
                # LLM a retournÃ© un seul objet au lieu d'un array
                file_plan = [raw]

        default_plan = _local_default_plan(description)
        if not file_plan:
            file_plan = default_plan
            ag.event_logger.log(
                "plan_files_local_fallback",
                {"reason": "invalid_llm_plan", "count": len(file_plan)},
            )
        else:
            low_desc = (description or "").lower()
            is_full_module_goal = any(
                k in low_desc
                for k in (
                    "module",
                    "backend",
                    "frontend",
                    "react",
                    "gestion",
                    "management",
                )
            )
            if is_full_module_goal and len(file_plan) < 4:
                # LLM plan too small for a full module request: merge with robust local defaults.
                by_path = {f.get("path"): f for f in file_plan if isinstance(f, dict)}
                for f in default_plan:
                    if f["path"] not in by_path:
                        file_plan.append(f)

        # Partition plan into files to create vs existing files to update.
        to_create: list[dict] = []
        to_update: list[dict] = []
        for f in file_plan:
            path = f["path"]
            if path in ag.staged_edits:
                continue
            abs_path = os.path.join(PROJECT_ROOT, path)
            if os.path.isfile(abs_path):
                to_update.append(f)
            else:
                to_create.append(f)

        if not to_create and not to_update:
            return f"plan_files: all {len(file_plan)} files already staged"

        # Inject ordered actions: create missing files, then complete/update existing ones.
        new_decisions = []
        for f in to_create:
            new_decisions.append(
                {
                    "action": "create_file",
                    "reason": "plan_files_ordered_creation",
                    "args": {
                        "path": f["path"],
                        "description": f"{f['description']} â€” part of: {description[:200]}",
                    },
                }
            )
        for f in to_update:
            new_decisions.append(
                {
                    "action": "propose_edit",
                    "reason": "plan_files_ordered_update",
                    "args": {
                        "path": f["path"],
                        "instruction": (
                            f"Complete/align this file with the module goal: {description[:300]}.\n"
                            f"File role: {f['description']}.\n"
                            "Keep imports consistent and provide production-ready code."
                        ),
                    },
                }
            )
            if auto_apply:
                new_decisions.append(
                    {
                        "action": "apply_edit",
                        "reason": "plan_files_ordered_update_apply",
                        "args": {"path": f["path"]},
                    }
                )

        new_decisions.append(
            {
                "action": "finish",
                "reason": "plan_files_all_done",
                "args": {
                    "summary": (
                        f"Module plan executed: created={len(to_create)}, updated={len(to_update)}"
                    )
                },
            }
        )
        ag._forced_decisions = new_decisions + ag._forced_decisions

        plan_summary = "\n".join(
            [f"  [create] {f['path']}: {f['description']}" for f in to_create]
            + [f"  [update] {f['path']}: {f['description']}" for f in to_update]
        )
        return f"Plan files (create={len(to_create)}, update={len(to_update)}) :\n{plan_summary}"

    def _exec_generate_migration(self, args: dict, **_kw) -> str:
        from agent.migration_generator import (
            find_model_files,
            generate_migration,
            is_alembic_configured,
        )

        if not is_alembic_configured():
            models = find_model_files()
            hint = (
                f"ModÃ¨les dÃ©tectÃ©s : {', '.join(models[:5])}"
                if models
                else "Aucun modÃ¨le SQLAlchemy dÃ©tectÃ©."
            )
            return (
                "[error] Alembic non configurÃ©. "
                "CrÃ©ez alembic.ini via 'alembic init alembic' et configurez sqlalchemy.url. "
                f"{hint}"
            )

        message = args.get("message", "auto_migration")
        result = generate_migration(message=message)
        if result["ok"]:
            rev = result.get("revision_file") or "inconnu"
            return f"Migration gÃ©nÃ©rÃ©e : {rev}\n{result['output']}"
        return f"[error] gÃ©nÃ©ration Ã©chouÃ©e :\n{result['output']}"

    def _exec_apply_migration(self, args: dict, **_kw) -> str:
        from agent.migration_generator import apply_migration, is_alembic_configured

        if not is_alembic_configured():
            return (
                "[error] Alembic non configurÃ©. "
                "CrÃ©ez alembic.ini via 'alembic init alembic' avant d'appliquer des migrations."
            )

        target = args.get("target", "head")
        result = apply_migration(target=target)
        if result["ok"]:
            return f"Migrations appliquÃ©es (cible={target}) :\n{result['output']}"
        return f"[error] application Ã©chouÃ©e (cible={target}) :\n{result['output']}"
