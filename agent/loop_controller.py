"""
P1.1 — Contrôleur de boucle extrait de auto_agent.py.

Responsabilités :
- Calcul du coût et du timeout par action
- Détection de boucles de décision et de résultats identiques
- Détection des étapes bloquées (stalled steps)
- Calcul des signatures pour comparaison
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.auto_agent import AutonomousAgent


class LoopController:
    """Détecte et prévient les boucles infinies dans la boucle agent."""

    def __init__(self, agent: "AutonomousAgent"):
        self.agent = agent

    # ------------------------------------------------------------------
    # Coût et timeout
    # ------------------------------------------------------------------

    def action_cost(self, action: str) -> int:
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
            "web_search": 4,
            "finish": 0,
        }
        return costs.get(action, 2)

    def action_timeout(self, action: str) -> int:
        """Timeout spécifique par type d'action (en secondes)."""
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
            "web_search": 30,
            "finish": 5,
        }
        return timeouts.get(action, 60)

    # ------------------------------------------------------------------
    # Signatures
    # ------------------------------------------------------------------

    def signature(self, action: str, args: dict, result: str = "") -> str:
        return json.dumps(
            {"a": action, "g": args, "r": result[:200]},
            sort_keys=True,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------
    # Détection de boucles
    # ------------------------------------------------------------------

    def decision_loop_detected(self, action: str, args: dict) -> bool:
        sig = self.signature(action, args)
        if self.agent._decision_signatures[-4:].count(sig) >= 2:
            return True
        if action == "propose_edit":
            path = args.get("path", "")
            recent = self.agent._decision_signatures[-6:]
            propose_count = sum(
                1
                for s in recent
                if json.loads(s).get("a") == "propose_edit"
                and json.loads(s).get("g", {}).get("path") == path
            )
            if propose_count >= 3:
                return True
        return False

    def outcome_loop_detected(self, action: str, args: dict, result: str) -> bool:
        sig = self.signature(action, args, result)
        return self.agent._outcome_signatures[-4:].count(sig) >= 2

    def is_stalled_step(self, action: str, result: str) -> bool:
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
