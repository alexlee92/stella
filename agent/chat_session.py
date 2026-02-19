import json
import os
from datetime import datetime
from typing import List

from agent.auto_agent import AutonomousAgent
from agent.config import CHAT_HISTORY_PATH, PROJECT_ROOT, TOP_K_RESULTS
from agent.event_logger import EventLogger
from agent.llm_interface import ask_llm
from agent.memory import search_memory


class ChatSession:
    def __init__(
        self, history_path: str = CHAT_HISTORY_PATH, top_k: int = TOP_K_RESULTS
    ):
        self.top_k = top_k
        self.messages: List[dict] = []
        self.decisions: List[dict] = []
        self.event_logger = EventLogger()

        abs_history = os.path.join(PROJECT_ROOT, history_path)
        os.makedirs(os.path.dirname(abs_history), exist_ok=True)
        self.history_path = abs_history
        self._load_history()

    def _load_history(self):
        if not os.path.exists(self.history_path):
            return

        try:
            with open(self.history_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if rec.get("type") == "message":
                        role = rec.get("role")
                        content = rec.get("content", "")
                        if role in {"user", "assistant"}:
                            self.messages.append({"role": role, "content": content})
                    elif rec.get("type") == "decision":
                        self.decisions.append(rec)
        except OSError:
            pass

    def _persist(self, record: dict):
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_decision(self, record: dict):
        self.decisions.append(record)
        self._persist(record)

    def _build_history_text(self, max_turns: int = 8) -> str:
        recent = self.messages[-max_turns:]
        if not recent:
            return "no prior turns"
        return "\n".join(f"{m['role']}: {m['content']}" for m in recent)

    def _build_context(self, question: str) -> str:
        docs = search_memory(question, k=self.top_k)
        if not docs:
            return "No indexed context"

        chunks = []
        for path, content in docs:
            chunks.append(f"FILE: {path}\n{content[:1200]}")
        return "\n\n".join(chunks)

    def ask(self, question: str) -> str:
        context = self._build_context(question)
        history = self._build_history_text()

        prompt = f"""
You are a local coding assistant in a continuous chat session.
Use project context and conversation history.
If context is missing, say what to inspect next.

Conversation history:
{history}

User question:
{question}

Project context:
{context}
"""
        answer = ask_llm(prompt).strip()

        user_record = {
            "type": "message",
            "role": "user",
            "content": question,
            "timestamp": datetime.utcnow().isoformat(),
        }
        assistant_record = {
            "type": "message",
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.utcnow().isoformat(),
        }

        self.messages.append({"role": "user", "content": question})
        self.messages.append({"role": "assistant", "content": answer})

        self._persist(user_record)
        self._persist(assistant_record)
        self.event_logger.log(
            "chat_turn", {"question": question, "answer_preview": answer[:400]}
        )

        return answer

    def run_auto(
        self,
        goal: str,
        auto_apply: bool = False,
        max_steps: int = 8,
        fix_until_green: bool = False,
        generate_tests: bool = False,
        max_seconds: int = 0,
    ) -> str:
        agent = AutonomousAgent(
            top_k=self.top_k, max_steps=max_steps, logger=self.log_decision
        )
        summary = agent.run(
            goal=goal,
            auto_apply=auto_apply,
            fix_until_green=fix_until_green,
            generate_tests=generate_tests,
            max_seconds=max_seconds,
        )

        record = {
            "type": "auto_summary",
            "goal": goal,
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._persist(record)
        self.event_logger.log(
            "auto_summary", {"goal": goal, "summary_preview": summary[:500]}
        )
        return summary

    def show_decisions(self, limit: int = 12) -> str:
        recent = self.decisions[-limit:]
        if not recent:
            return "No decisions yet in this session"

        lines = []
        for d in recent:
            lines.append(f"step {d.get('step')}: {d.get('action')} | {d.get('reason')}")
        return "\n".join(lines)
