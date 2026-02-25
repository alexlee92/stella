"""
P1.1 — Critique extrait de auto_agent.py.

Responsabilités :
- Construire le prompt de critique
- Valider la décision du planner via un second appel LLM
- Appliquer le patch proposé par la critique si valide
- Fallback en auto-approbation si la critique échoue
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from agent.action_schema import validate_critique_schema, validate_decision_schema
from agent.decision_normalizer import (
    autocorrect_decision_schema,
    coerce_decision,
    normalize_critique,
)

if TYPE_CHECKING:
    from agent.auto_agent import AutonomousAgent


class Critic:
    """Valide et corrige les décisions du planner."""

    def __init__(self, agent: "AutonomousAgent"):
        self.agent = agent

    def prompt(self, goal: str, decision: dict) -> str:
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

    def critique(self, goal: str, decision: dict) -> dict:
        """
        Soumet la décision à la critique et retourne la décision finale approuvée.
        En cas d'échec de la critique, la décision originale est auto-approuvée.
        """
        ag = self.agent

        t0 = time.time()
        raw_critique = ag._llm_fn(
            self.prompt(goal, decision),
            retries=1,
            prompt_class="critique",
            task_type="analysis",
        )
        elapsed = round(time.time() - t0, 1)
        if elapsed > 60:
            print(f"  [critique] lente ({elapsed}s) — charge élevée sur le modèle")

        # Échec de parsing de la critique → auto-approbation
        if (
            isinstance(raw_critique, dict)
            and raw_critique.get("_error_type") == "parse"
        ):
            parse_meta = raw_critique.get("_parse_meta") or {}
            ag.event_logger.log_failure(
                "parse",
                "critique_json_parse_failed",
                {
                    "critique": raw_critique,
                    "parse_class": parse_meta.get("error_class", "unknown_parse_error"),
                    "parse_attempts": parse_meta.get("attempt_count", 0),
                    "prompt_class": parse_meta.get("prompt_class", "critique"),
                },
            )
            return decision  # auto-approve

        critique = normalize_critique(raw_critique)
        c_ok, c_msg = validate_critique_schema(critique)
        if not c_ok:
            ag.event_logger.log_failure(
                "parse",
                f"critique_schema_invalid:{c_msg}",
                {
                    "critique": critique,
                    "parse_class": "critique_schema_invalid",
                    "prompt_class": "critique",
                },
            )
            return decision  # auto-approve

        if critique.get("approve"):
            return decision

        # La critique propose un patch
        patched = critique.get("patched_decision")
        if isinstance(patched, dict):
            patched = coerce_decision(goal, patched)
            p_ok, p_msg = validate_decision_schema(patched)
            if not p_ok:
                patched2 = autocorrect_decision_schema(goal, patched, p_msg)
                p2_ok, _ = validate_decision_schema(patched2)
                if p2_ok:
                    ag.event_logger.log(
                        "schema_autocorrect",
                        {"from": patched, "to": patched2, "issue": p_msg},
                    )
                    patched = patched2
                    p_ok = True
            if p_ok:
                ag.event_logger.log(
                    "critique_patch",
                    {"reason": critique.get("reason"), "patched": patched},
                )
                return patched
            ag.event_logger.log_failure(
                "parse",
                f"patched_schema_invalid:{p_msg}",
                {
                    "patched": patched,
                    "parse_class": "patched_schema_invalid",
                    "prompt_class": "critique",
                },
            )

        from agent.planner import Planner

        ag.event_logger.log(
            "critique_reject_fallback", {"reason": critique.get("reason", "n/a")}
        )
        return Planner(ag).fallback_decision(goal, reason="critique_rejected")
