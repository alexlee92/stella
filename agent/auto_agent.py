"""
Autonomous agent loop for Stella.

Orchestrateur principal : dÃ©lÃ¨gue Ã  :
- agent/planner.py      : gÃ©nÃ©ration et rÃ©paration des dÃ©cisions LLM
- agent/critic.py       : validation critique des dÃ©cisions
- agent/loop_controller.py : dÃ©tection de boucles et gestion du budget
- agent/replan.py       : stratÃ©gies de replanification aprÃ¨s Ã©chec
- agent/executor.py     : exÃ©cution des actions individuelles
"""

import atexit
import json
import os
import sys
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
from agent.config import PROJECT_ROOT as _PROJECT_ROOT
from agent.event_logger import EventLogger

STAGED_RECOVERY_PATH = os.path.join(_PROJECT_ROOT, ".stella", "staged_recovery.json")
from agent.executor import ActionExecutor
from agent.llm_interface import ask_llm_json
from agent.memory import search_memory

# P1.1 â€” Modules extraits
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
        # P2.5 â€” Dependency injection: accept config/llm/memory as arguments
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
        self._last_goal: str = ""
        self._executor = ActionExecutor(self)
        # P1.1 â€” Sous-composants extraits
        self._planner_obj = Planner(self)
        self._critic_obj = Critic(self)
        self._loop = LoopController(self)
        self._replan_engine = ReplanEngine(self)

    # P2.1 â€” Delegate to decision_normalizer module
    def _normalize_decision(self, decision: dict) -> dict:
        return normalize_decision(decision)

    def _normalize_critique(self, critique: dict) -> dict:
        return normalize_critique(critique)

    def _infer_fallback_action(self, goal: str, args: dict) -> tuple[str, dict]:
        return infer_fallback_action(goal, args)

    def _autocorrect_decision_schema(self, goal: str, decision: dict, msg: str) -> dict:
        return autocorrect_decision_schema(goal, decision, msg)

    def _confirm_apply(self, paths: List[str]) -> bool:
        """P1.2 â€” Demande confirmation Ã  l'utilisateur avant d'appliquer des patches.

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
        """Retourne le contenu des fichiers crÃ©Ã©s/modifiÃ©s dans cette session."""
        if not self.staged_edits:
            return ""
        chunks = []
        for path, content in self.staged_edits.items():
            preview = (content or "")[:max_chars_per_file]
            chunks.append(f"FILE (this session): {path}\n{preview}")
        return "\n\n".join(chunks)

    # P2.6 â€” Budget contextuel dynamique
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
        "systÃ¨me",
        "system",
        "architecture",
        "multi-fichier",
        "multi-file",
        "plan",
        "gÃ©nÃ¨re un",
        "crÃ©e un",
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
        "mise Ã  jour",
        "amÃ©liore",
        "improve",
        "ajoute",
        "add",
        "migration",
        "schema",
    }

    def _context_budget(self, goal: str) -> tuple[int, int]:
        """Retourne (chars_per_file, top_k) selon la complexitÃ© du goal.

        P2.6 â€” Budget contextuel dynamique :
        - Simple  (lecture/recherche) â†’ 1200 chars, k=3
        - Medium  (fix/refactor/test) â†’ 3000 chars, k=5
        - Complex (module ERP complet)â†’ 8000 chars, k=8
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

        # P2.4 â€” Enrichir avec la mÃ©moire globale cross-sessions
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

        Combine les rÃ©sultats des 3 requÃªtes et dÃ©duplique par chemin de fichier.
        Inclut aussi les fichiers crÃ©Ã©s dans la session courante.
        P2.6 : chars_per_file et k adaptÃ©s dynamiquement Ã  la complexitÃ© du goal.
        """
        import os as _os

        chars, k = self._context_budget(goal)
        # Les fichiers de session ont droit Ã  plus de contexte
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

        # 2. Recherche par rÃ©pertoire/nom de fichier cible
        if path:
            dir_name = _os.path.dirname(path)
            base_name = _os.path.splitext(_os.path.basename(path))[0]
            _add_docs(dir_name or base_name)

        # 3. Recherche par mots-clÃ©s de la description
        if description:
            _add_docs(description[:200])

        # 4. Fichiers de la session courante (prioritÃ© haute â€” toujours inclus)
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

    # P2.1 â€” Delegate to decision_normalizer module
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

    def _color(self, text: str, code: str) -> str:
        if not sys.stdout.isatty():
            return text
        return f"\033[{code}m{text}\033[0m"

    def _render_progress(self, step: int, total: int) -> str:
        total = max(1, total)
        width = 24
        ratio = min(max(step / total, 0.0), 1.0)
        filled = int(width * ratio)
        bar = ("#" * filled) + ("-" * (width - filled))
        pct = int(ratio * 100)
        return f"[{self._color(bar, '36')}] {step}/{total} ({pct:3d}%)"

    def _fallback_decision(self, goal: str, reason: str) -> dict:
        return self._planner_obj.fallback_decision(goal, reason)

    def _generate_file_content(self, goal: str, path: str, description: str) -> str:
        return self._planner_obj.generate_file_content(goal, path, description)

    # M8 â€” Persistance staged_edits sur interruption
    def _save_staged_recovery(self):
        if not self.staged_edits:
            return
        try:
            os.makedirs(os.path.dirname(STAGED_RECOVERY_PATH), exist_ok=True)
            with open(STAGED_RECOVERY_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {"goal": self._last_goal, "edits": self.staged_edits}, f, indent=2
                )
            print(f"[stella] staged_edits sauvegardÃ©s â†’ {STAGED_RECOVERY_PATH}")
        except Exception as exc:
            print(f"[stella] Impossible de sauvegarder staged_recovery: {exc}")

    def plan_once(self, goal: str) -> dict:
        return self._plan_with_critique(goal)

    def run(
        self,
        goal: str,
        auto_apply: bool = False,
        fix_until_green: bool = False,
        generate_tests: bool = False,
        max_seconds: int = 0,
        max_auto_continuations: int = 2,
        _continuation_idx: int = 0,
        _resume: bool = False,
        _started_at: Optional[datetime] = None,
     ) -> str:
        window_start_step_count = len(self.steps)
        # M8 â€” Enregistrer goal pour la sauvegarde atexit
        if not _resume:
            self._last_goal = goal
            atexit.register(self._save_staged_recovery)

        # M8 â€” Recharger staged_edits depuis la derniÃ¨re session interrompue si le goal correspond
        if not _resume and os.path.isfile(STAGED_RECOVERY_PATH):
            try:
                with open(STAGED_RECOVERY_PATH, "r", encoding="utf-8") as f:
                    recovery = json.load(f)
                if recovery.get("goal") == goal and recovery.get("edits"):
                    self.staged_edits.update(recovery["edits"])
                    print(
                        f"[stella] staged_edits rechargÃ©s depuis {STAGED_RECOVERY_PATH} "
                        f"({len(self.staged_edits)} fichiers)"
                    )
                    os.remove(STAGED_RECOVERY_PATH)
            except Exception:
                pass

        # P1.1 â€” Detection mode lecture seule (read-only)
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

        # P1.1 â€” Auto-adjust budget for code_edit + tests
        if self._generate_tests and self.max_steps <= 10:
            self.max_steps = 15
            self._max_cost = self.max_steps * 4

        # M7-fix: augmenter le budget pour les goals multi-fichiers (plan_files)
        _multi_file_kw = [
            "plan_files",
            "module",
            "erp",
            "crm",
            "crÃ©e un",
            "create a",
            "gÃ©nÃ¨re un",
            "generate a",
            "scaffold",
            "complet",
            "complete",
        ]
        if any(k in goal.lower() for k in _multi_file_kw) and self.max_steps <= 12:
            self.max_steps = 20
            self._max_cost = self.max_steps * 4

        if not _resume:
            self._consecutive_stalled_steps = 0
            self._total_cost = 0
            self._replan_attempts = 0
            self._forced_decisions = []
            self._has_validation_action = False
            self._bootstrapped_code_edit = False

            if not self.read_only:
                self._bootstrap_code_edit_decisions(goal, auto_apply=auto_apply)

        # P1.1 â€” Proactive web search bootstrap for research and complex modules
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
        if (not _resume) and (not self._forced_decisions) and any(
            k in low_goal for k in research_keywords
        ):
            # Use a cleaner, more technical English query for better results
            clean_goal = (
                goal.replace("GÃ©nÃ¨re ", "")
                .replace("CrÃ©e ", "")
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

        start_ts = _started_at or datetime.now(UTC)

        if _resume:
            self._max_cost += self.max_steps * 4
            print(
                f"\n[stella] reprise automatique {_continuation_idx}/{max_auto_continuations} (fenetre +{self.max_steps} etapes)"
            )
        else:
            print(f"\n[stella] DÃ©marrage de l'agent â€” objectif : {goal[:80]}")
            if self.read_only:
                print("[stella] MODE LECTURE SEULE ACTIF (aucune modification autorisÃ©e)")
            print(
                f"[stella] ParamÃ¨tres : max_steps={self.max_steps}, apply={auto_apply}, fix={fix_until_green}, tests={generate_tests}"
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

            elapsed_so_far = round((datetime.now(UTC) - start_ts).total_seconds(), 1)
            print(
                f"  [{index}/{self.max_steps}] planification LLM en cours... ({elapsed_so_far}s)",
                flush=True,
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
                "search_code": f"cherche Â« {args.get('pattern', '')} Â»",
                "create_file": f"crÃ©e {args.get('path', '')}",
                "propose_edit": f"prÃ©pare un patch pour {args.get('path', '')}",
                "apply_edit": f"applique le patch sur {args.get('path', '')}",
                "apply_all_staged": "applique tous les patches en attente",
                "run_tests": "exÃ©cute les tests",
                "run_quality": "vÃ©rifie la qualitÃ© du code",
                "project_map": "gÃ©nÃ¨re la carte du projet",
                "git_branch": f"crÃ©e la branche {args.get('name', '')}",
                "git_commit": "committe les changements",
                "git_diff": "affiche le diff git",
                "web_search": f"recherche sur le web Â« {args.get('query', '')} Â»",
                "finish": "finalise la tÃ¢che",
            }
            label = _action_labels.get(action, action)
            elapsed_so_far = round((datetime.now(UTC) - start_ts).total_seconds(), 1)
            gauge = self._render_progress(index, self.max_steps)
            print(
                f"  {gauge} {label}... ({elapsed_so_far}s)",
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
                # Pour les actions de lecture (read-only), synthÃ©tiser une rÃ©ponse
                # en lien avec le goal plutÃ´t que de retourner une erreur.
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
                        print(f"\n[stella] TerminÃ© en {elapsed}s\n")
                        return self.render_summary(summary)
                summary = "Stopped repeated decision loop; recommend narrowing goal to a concrete file-level change request"
                self._record(AutoStep(index, "finish", "auto_stop_repeat", summary))
                return self.render_summary(summary)

            self._decision_signatures.append(self._signature(action, args))
            step = AutoStep(index, action, reason, "")

            # P2.1 â€” Delegate action execution to ActionExecutor
            try:
                result = self._executor.execute(
                    action,
                    args,
                    goal=goal,
                    auto_apply=auto_apply,
                    index=index,
                    reason=reason,
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
                    print(f"\n[stella] TerminÃ© en {elapsed}s\n")
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

        if auto_apply and (not self.read_only) and _continuation_idx < max_auto_continuations:
            window_steps = self.steps[window_start_step_count:]
            meaningful_actions = {
                "create_file",
                "propose_edit",
                "apply_edit",
                "apply_all_staged",
                "plan_files",
                "run_quality",
                "run_tests",
            }
            made_progress = any(s.action in meaningful_actions for s in window_steps)
            if made_progress or self._forced_decisions:
                return self.run(
                    goal=goal,
                    auto_apply=auto_apply,
                    fix_until_green=fix_until_green,
                    generate_tests=generate_tests,
                    max_seconds=max_seconds,
                    max_auto_continuations=max_auto_continuations,
                    _continuation_idx=_continuation_idx + 1,
                    _resume=True,
                    _started_at=start_ts,
                )
            return self.render_summary(
                "Stopped: max steps reached and no meaningful progress in last window"
            )

        if _continuation_idx >= max_auto_continuations and auto_apply and (not self.read_only):
            return self.render_summary(
                f"Stopped: max steps reached after {max_auto_continuations} automatic continuations"
            )

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
        if total_steps < self.max_steps:
            lines.append(
                "  Statut boucle : termine normalement avant max_steps (action finish)"
            )
        if files_modified:
            for f in files_modified[:10]:
                lines.append(f"    - {f}")
        lines.append(f"  Cout total : {self._total_cost}/{self._max_cost}")
        lines.append(f"  Replans : {self._replan_attempts}/{self._max_replan_attempts}")

        syntax_warnings = []
        for s in self.steps:
            if "[syntax-warning:" in (s.result or ""):
                syntax_warnings.append(
                    f"  - step {s.step} ({s.action}): {s.result[:220]}"
                )
        if syntax_warnings:
            lines.append("  Warnings syntaxe JS/TS:")
            lines.extend(syntax_warnings[:10])
        lines.append("")
        lines.append("Decisions:")
        for s in self.steps:
            lines.append(f"  {s.step}. [{s.action}] {s.reason} -> {s.result[:160]}")
        return "\n".join(lines)


