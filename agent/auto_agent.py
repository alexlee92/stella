"""
Autonomous agent loop for Stella.

Orchestrateur principal : délègue à :
- agent/planner.py      : génération et réparation des décisions LLM
- agent/critic.py       : validation critique des décisions
- agent/loop_controller.py : détection de boucles et gestion du budget
- agent/replan.py       : stratégies de replanification après échec
- agent/executor.py     : exécution des actions individuelles
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Callable, Dict, List, Optional

from agent.config import (
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

# P1.1 — Modules extraits
from agent.planner import Planner
from agent.critic import Critic
from agent.loop_controller import LoopController
from agent.replan import ReplanEngine


@dataclass
class AutoStep:
    step: int
    action: str
    reason: str
    result: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


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
        self.read_only = False
        self._executor = ActionExecutor(self)
        # P1.1 — Sous-composants extraits
        self._planner_obj = Planner(self)
        self._critic_obj = Critic(self)
        self._loop = LoopController(self)
        self._replan_engine = ReplanEngine(self)

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
            answer = (
                input(
                    f"\n  [?] Appliquer les modifications sur {files_list} ? [y/n/d(iff)] "
                )
                .strip()
                .lower()
            )
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
                abs_path = (
                    os.path.join(PROJECT_ROOT, path)
                    if not os.path.isabs(path)
                    else path
                )
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

    # P2.6 — Budget contextuel dynamique
    _COMPLEX_KW = {
        "erp",
        "crm",
        "module",
        "facturation",
        "billing",
        "invoice",
        "complet",
        "complete",
        "full",
        "système",
        "system",
        "architecture",
        "multi-fichier",
        "multi-file",
        "plan",
        "génère un",
        "crée un",
        "generate a",
        "create a",
        "stock",
        "inventory",
        "purchase",
        "workflow",
        "rbac",
        "auth",
        "saas",
        "tenant",
        "dashboard",
        "analytics",
        "reporting",
        "audit",
    }
    _MEDIUM_KW = {
        "refactor",
        "refactorise",
        "test",
        "tests",
        "fix",
        "fixe",
        "debug",
        "optimise",
        "optimize",
        "update",
        "mise à jour",
        "améliore",
        "improve",
        "ajoute",
        "add",
        "migration",
        "schema",
    }

    def _context_budget(self, goal: str) -> tuple[int, int]:
        """Retourne (chars_per_file, top_k) selon la complexité du goal.

        P2.6 — Budget contextuel dynamique :
        - Simple  (lecture/recherche) → 1200 chars, k=3
        - Medium  (fix/refactor/test) → 3000 chars, k=5
        - Complex (module ERP complet)→ 8000 chars, k=8
        """
        low = goal.lower()
        if any(k in low for k in self._COMPLEX_KW):
            return 8000, 8
        if any(k in low for k in self._MEDIUM_KW):
            return 3000, 5
        return 1200, 3

    def _summarize_context(self, goal: str) -> str:
        chars, k = self._context_budget(goal)
        docs = self._memory_fn(goal, k=k)

        chunks = []
        for path, content in docs:
            chunks.append(f"FILE: {path}\n{content[:chars]}")

        # P2.4 — Enrichir avec la mémoire globale cross-sessions
        try:
            from agent.global_memory import search_global_memory

            global_docs = search_global_memory(goal, k=2)
            for label, text in global_docs:
                chunks.append(f"GLOBAL_MEMORY: {label}\n{text[:600]}")
        except Exception:
            pass

        if not chunks:
            return "No indexed context found"
        return "\n\n".join(chunks)

    def _summarize_context_multi(
        self, goal: str, path: str = "", description: str = ""
    ) -> str:
        """Recherche de contexte multi-angle : goal + chemin + description.

        Combine les résultats des 3 requêtes et déduplique par chemin de fichier.
        Inclut aussi les fichiers créés dans la session courante.
        P2.6 : chars_per_file et k adaptés dynamiquement à la complexité du goal.
        """
        import os as _os

        chars, k = self._context_budget(goal)
        # Les fichiers de session ont droit à plus de contexte
        session_chars = min(chars * 2, 16000)

        seen_paths: set[str] = set()
        all_chunks: list[str] = []

        def _add_docs(query: str):
            if not query.strip():
                return
            for fpath, content in self._memory_fn(query, k=k):
                rel = _os.path.relpath(fpath, ".").replace("\\", "/")
                if rel not in seen_paths:
                    seen_paths.add(rel)
                    all_chunks.append(f"FILE: {rel}\n{content[:chars]}")

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
        session = self._session_context(max_chars_per_file=session_chars)
        if session:
            all_chunks.insert(0, session)

        return "\n\n".join(all_chunks) if all_chunks else "No indexed context found"

    def _planner_prompt(self, goal: str) -> str:
        return self._planner_obj.prompt(goal)

    def _schema_repair_prompt(
        self, goal: str, decision: dict, schema_error: str
    ) -> str:
        return self._planner_obj.schema_repair_prompt(goal, decision, schema_error)

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
        return self._critic_obj.prompt(goal, decision)

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
        return self._loop.signature(action, args, result)

    def _decision_loop_detected(self, action: str, args: dict) -> bool:
        return self._loop.decision_loop_detected(action, args)

    def _outcome_loop_detected(self, action: str, args: dict, result: str) -> bool:
        return self._loop.outcome_loop_detected(action, args, result)

    def _action_cost(self, action: str) -> int:
        return self._loop.action_cost(action)

    def _action_timeout(self, action: str) -> int:
        return self._loop.action_timeout(action)

    def _is_stalled_step(self, action: str, result: str) -> bool:
        return self._loop.is_stalled_step(action, result)

    def _failure_excerpt(self, text: str, max_lines: int = 8) -> str:
        return self._replan_engine.failure_excerpt(text, max_lines)

    def _extract_error_paths(self, text: str) -> List[str]:
        return self._replan_engine.extract_error_paths(text)

    def _enqueue_replan_after_failure(
        self,
        failure_kind: str,
        failure_text: str,
        auto_apply: bool,
        fallback_path: Optional[str] = None,
    ) -> bool:
        return self._replan_engine.enqueue_replan_after_failure(
            failure_kind, failure_text, auto_apply, fallback_path
        )

    def _plan_with_critique(self, goal: str) -> dict:
        if self._forced_decisions:
            forced = self._forced_decisions.pop(0)
            self.event_logger.log(
                "forced_plan",
                {"decision": forced, "remaining": len(self._forced_decisions)},
            )
            return forced
        decision = self._planner_obj.plan(goal)
        return self._critic_obj.critique(goal, decision)

    def _fallback_decision(self, goal: str, reason: str) -> dict:
        return self._planner_obj.fallback_decision(goal, reason)

    def _generate_file_content(self, goal: str, path: str, description: str) -> str:
        return self._planner_obj.generate_file_content(goal, path, description)

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
        # P1.1 — Detection mode lecture seule (read-only)
        low_goal = goal.lower()
        read_only_keywords = [
            "ne modifie pas",
            "ne cree pas",
            "aucune modification",
            "aucune creation",
            "strictement aucune",
            "read only",
            "read-only",
            "don't modify",
            "don't create",
        ]
        self.read_only = any(k in low_goal for k in read_only_keywords)

        self._fix_until_green = bool(fix_until_green)
        self._generate_tests = bool(generate_tests)

        # P1.1 — Auto-adjust budget for code_edit + tests
        if self._generate_tests and self.max_steps <= 10:
            self.max_steps = 15
            self._max_cost = self.max_steps * 4

        self._consecutive_stalled_steps = 0
        self._total_cost = 0
        self._replan_attempts = 0
        self._forced_decisions = []
        self._has_validation_action = False
        self._bootstrapped_code_edit = False

        if not self.read_only:
            self._bootstrap_code_edit_decisions(goal, auto_apply=auto_apply)

        # P1.1 — Proactive web search bootstrap for research and complex modules
        research_keywords = [
            "web",
            "recherche",
            "online",
            "search",
            "internet",
            "doc",
            "crm",
            "erp",
            "facturation",
            "billing",
            "inventory",
            "stock",
            "auth",
            "jwt",
            "oauth",
            "dashboard",
            "analytics",
            "standard",
        ]
        if not self._forced_decisions and any(k in low_goal for k in research_keywords):
            # Use a cleaner, more technical English query for better results
            clean_goal = (
                goal.replace("Génère ", "")
                .replace("Crée ", "")
                .replace("un module ", "")
                .strip()
            )
            self._forced_decisions.append(
                {
                    "action": "web_search",
                    "reason": "proactive_research_standards",
                    "args": {
                        "query": f"industry standard features and database schema for {clean_goal[:100]} best practices",
                        "limit": 5,
                    },
                }
            )

        from datetime import UTC

        start_ts = datetime.now(UTC)

        print(f"\n[stella] Démarrage de l'agent — objectif : {goal[:80]}")
        if self.read_only:
            print("[stella] MODE LECTURE SEULE ACTIF (aucune modification autorisée)")
        print(
            f"[stella] Paramètres : max_steps={self.max_steps}, apply={auto_apply}, fix={fix_until_green}, tests={generate_tests}"
        )
        print()

        for index in range(1, self.max_steps + 1):
            if (
                max_seconds
                and (datetime.now(UTC) - start_ts).total_seconds() > max_seconds
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
                "web_search": f"recherche sur le web « {args.get('query', '')} »",
                "finish": "finalise la tâche",
            }
            label = _action_labels.get(action, action)
            elapsed_so_far = round((datetime.now(UTC) - start_ts).total_seconds(), 1)
            print(
                f"  [{index}/{self.max_steps}] {label}... ({elapsed_so_far}s)",
                flush=True,
            )

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
                            f"Collected context:\n{last_result[:4000]}\n\n"
                            "Based on the context above, provide a comprehensive and ACTIONABLE response. "
                            "If the goal is complex (architecture, ERP, planning), your response MUST include:\n"
                            "1. A structured summary of the current situation.\n"
                            "2. A detailed file-by-file implementation plan or analysis.\n"
                            "3. Potential risks and integration points identified.\n"
                            "Use Markdown formatting for clarity. Be precise and avoid generic advice."
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
                            (datetime.now(UTC) - start_ts).total_seconds(), 1
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
                    action,
                    args,
                    goal=goal,
                    auto_apply=auto_apply,
                    index=index,
                )
                # Special handling for apply_edit without staged edit
                if action == "apply_edit" and result == "[no_staged_edit]":
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
                # Special handling for git_commit redundancy
                if action == "git_commit" and "nothing to commit" in result.lower():
                    recent_nothing = any(
                        s.action == "git_commit"
                        and "nothing to commit" in s.result.lower()
                        for s in self.steps[-2:]
                    )
                    if recent_nothing:
                        step.result = result
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
                # Special handling for finish action
                if action == "finish":
                    step.result = result
                    self._record(step)
                    elapsed = round((datetime.now(UTC) - start_ts).total_seconds(), 1)
                    print(f"\n[stella] Terminé en {elapsed}s\n")
                    return self.render_summary(step.result)
                # Unknown action
                if result.startswith("Unknown action:"):
                    step.action = "unknown_action"
                step.result = result
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
        # Statistiques de session
        total_steps = len(self.steps)
        files_modified = list(self.staged_edits.keys())
        actions_used = {}
        for s in self.steps:
            actions_used[s.action] = actions_used.get(s.action, 0) + 1

        lines = [
            f"Final: {final_message}",
            "",
            "--- Resume de session ---",
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
