"""
P1.1 — Moteur de replanification extrait de auto_agent.py.

Responsabilités :
- Analyser les sorties d'erreurs (lint, tests, format)
- Extraire les chemins de fichiers concernés
- Construire et enqueue une séquence de correction
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from agent.auto_agent import AutonomousAgent


class ReplanEngine:
    """Stratégies de replanification après un échec de qualité ou de tests."""

    def __init__(self, agent: "AutonomousAgent"):
        self.agent = agent

    def failure_excerpt(self, text: str, max_lines: int = 8) -> str:
        lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
        if not lines:
            return "no failure output"
        return "\n".join(lines[:max_lines])[:800]

    def extract_error_paths(self, text: str) -> List[str]:
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

    def enqueue_replan_after_failure(
        self,
        failure_kind: str,
        failure_text: str,
        auto_apply: bool,
        fallback_path: Optional[str] = None,
    ) -> bool:
        ag = self.agent
        if ag._replan_attempts >= ag._max_replan_attempts:
            return False

        targets = self.extract_error_paths(failure_text)
        target_path = targets[0] if targets else (fallback_path or "")
        excerpt = self.failure_excerpt(failure_text)

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

        ag._forced_decisions.extend(queue)
        ag._replan_attempts += 1
        ag.event_logger.log(
            "replan_policy",
            {
                "failure_kind": failure_kind,
                "target_path": target_path,
                "queued_actions": [d["action"] for d in queue],
                "attempt": ag._replan_attempts,
            },
        )
        return True
