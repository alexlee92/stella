import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional

from agent.auto_agent import AutonomousAgent
from agent.config import CHAT_HISTORY_PATH, PROJECT_ROOT, TOP_K_RESULTS
from agent.event_logger import EventLogger
from agent.llm_interface import ask_llm, ask_llm_stream_print
from agent.memory import search_memory

# P2.2 — Sessions directory
_SESSIONS_DIR = os.path.join(PROJECT_ROOT, ".stella", "sessions")


def list_sessions(limit: int = 20) -> List[dict]:
    """P2.2 — List recent sessions with metadata."""
    if not os.path.isdir(_SESSIONS_DIR):
        return []
    entries = []
    for name in os.listdir(_SESSIONS_DIR):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(_SESSIONS_DIR, name)
        sid = name.replace(".jsonl", "")
        try:
            mtime = os.path.getmtime(path)
            # Read first line for metadata
            with open(path, "r", encoding="utf-8") as f:
                first = f.readline().strip()
            meta = json.loads(first) if first else {}
        except Exception:
            meta = {}
            mtime = 0
        entries.append({
            "id": sid,
            "path": path,
            "mtime": mtime,
            "goal": meta.get("goal", ""),
            "type": meta.get("type", ""),
        })
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return entries[:limit]


def get_latest_session_id() -> Optional[str]:
    """P2.2 — Get the ID of the most recent session."""
    sessions = list_sessions(limit=1)
    return sessions[0]["id"] if sessions else None


class ChatSession:
    def __init__(
        self,
        history_path: str = CHAT_HISTORY_PATH,
        top_k: int = TOP_K_RESULTS,
        session_id: Optional[str] = None,
        config: Optional[Dict] = None,
        llm_fn=None,
        memory_fn=None,
    ):
        # P2.5 — Dependency injection
        self._config = config
        self._llm_fn = llm_fn
        self._memory_fn = memory_fn or search_memory
        self.top_k = top_k
        self.messages: List[dict] = []
        self.decisions: List[dict] = []
        self.staged_edits: Dict[str, str] = {}
        self.event_logger = EventLogger()

        abs_history = os.path.join(PROJECT_ROOT, history_path)
        os.makedirs(os.path.dirname(abs_history), exist_ok=True)
        self.history_path = abs_history

        # P2.2 — Session persistence
        self.session_id = session_id or f"s{int(time.time())}"
        os.makedirs(_SESSIONS_DIR, exist_ok=True)
        self.session_path = os.path.join(_SESSIONS_DIR, f"{self.session_id}.jsonl")
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
        # P2.2 — Also persist to session file
        self._persist_session(record)

    def _persist_session(self, record: dict):
        """P2.2 — Write record to session-specific JSONL file."""
        try:
            with open(self.session_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def save_staged_edits(self):
        """P2.2 — Persist staged_edits so they survive a crash."""
        if not self.staged_edits:
            return
        record = {
            "type": "staged_edits",
            "edits": self.staged_edits,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._persist_session(record)

    def load_session(self, session_id: str) -> bool:
        """P2.2 — Restore a previous session by ID (messages, decisions, staged_edits)."""
        path = os.path.join(_SESSIONS_DIR, f"{session_id}.jsonl")
        if not os.path.exists(path):
            return False
        self.session_id = session_id
        self.session_path = path
        self.messages.clear()
        self.decisions.clear()
        self.staged_edits.clear()
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    rtype = rec.get("type")
                    if rtype == "message":
                        role = rec.get("role")
                        content = rec.get("content", "")
                        if role in {"user", "assistant"}:
                            self.messages.append({"role": role, "content": content})
                    elif rtype == "decision":
                        self.decisions.append(rec)
                    elif rtype == "staged_edits":
                        self.staged_edits = rec.get("edits", {})
        except OSError:
            return False
        return True

    def log_decision(self, record: dict):
        self.decisions.append(record)
        self._persist(record)

    def _build_history_text(self, max_turns: int = 10, max_chars: int = 4000) -> str:
        """P3.6 — Fenetre de conversation glissante avec budget de caracteres."""
        recent = self.messages[-max_turns:]
        if not recent:
            return "no prior turns"
        lines = []
        total = 0
        for m in reversed(recent):
            line = f"{m['role']}: {m['content'][:500]}"
            if total + len(line) > max_chars:
                break
            lines.insert(0, line)
            total += len(line)
        return "\n".join(lines)

    def _build_context(self, question: str) -> str:
        docs = self._memory_fn(question, k=self.top_k)
        if not docs:
            return "No indexed context"

        chunks = []
        for path, content in docs:
            rel = os.path.relpath(path, PROJECT_ROOT) if os.path.isabs(path) else path
            chunks.append(f"FILE: {rel}\n{content[:1200]}")
        return "\n\n".join(chunks)

    def _read_explicit_files(self, question: str) -> str:
        """Read files explicitly mentioned in the question (e.g. users/api.py)."""
        import re as _re
        from agent.project_scan import load_file_content
        pattern = r"([A-Za-z0-9_./\\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|yaml|yml|md|toml|sql|xml))"
        refs = list(dict.fromkeys(_re.findall(pattern, question)))
        sections = []
        for ref in refs[:3]:
            # Try direct and plural/singular variants
            candidates = [ref]
            parts = ref.replace("\\", "/").split("/")
            if parts:
                candidates.append("/".join([parts[0] + "s"] + parts[1:]))
                candidates.append("/".join([parts[0].rstrip("s")] + parts[1:]))
            for c in candidates:
                abs_path = os.path.join(PROJECT_ROOT, c) if not os.path.isabs(c) else c
                if os.path.isfile(abs_path):
                    try:
                        content = load_file_content(abs_path)
                        rel = os.path.relpath(abs_path, PROJECT_ROOT)
                        numbered = "\n".join(
                            f"{i+1:4d} | {line}" for i, line in enumerate(content.splitlines())
                        )
                        sections.append(f"=== {rel} (full source) ===\n{numbered}")
                    except OSError:
                        pass
                    break
        return "\n\n".join(sections)

    def ask(self, question: str) -> str:
        context = self._build_context(question)
        history = self._build_history_text()
        file_context = self._read_explicit_files(question)

        if file_context:
            prompt = f"""You are a senior coding assistant analyzing a project.
Answer in clear prose, NOT in JSON. Be precise and reference actual code.

Conversation history:
{history}

User question:
{question}

Here is the exact source code of the mentioned file(s) — analyze it carefully:
{file_context}

Instructions:
- Base your answer ONLY on the actual code shown above.
- Reference specific line numbers, function names, and variable names.
- Do NOT invent or hallucinate code that is not present.
- If asked about imports, list the exact import statements from the file.
- If asked about structure, describe the actual fields/methods present.

Additional project context:
{context}
"""
        else:
            prompt = f"""You are a senior coding assistant in a continuous chat session.
Answer in clear prose, NOT in JSON. Use the project context provided.

Conversation history:
{history}

User question:
{question}

Project context:
{context}

Instructions:
- Base your answer on the provided code context.
- Reference specific file names, function names, and line numbers when possible.
- Do NOT invent code that is not shown in the context.
"""
        answer = ask_llm_stream_print(prompt).strip()

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
            top_k=self.top_k, max_steps=max_steps, logger=self.log_decision,
            config=self._config, llm_fn=self._llm_fn, memory_fn=self._memory_fn,
        )
        summary = agent.run(
            goal=goal,
            auto_apply=auto_apply,
            fix_until_green=fix_until_green,
            generate_tests=generate_tests,
            max_seconds=max_seconds,
        )
        # P2.2 — Persist staged edits from agent for crash recovery
        if agent.staged_edits:
            self.staged_edits.update(agent.staged_edits)
            self.save_staged_edits()

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

    def show_context(self) -> str:
        """P3.6 — Affiche le contexte actuel de la session."""
        lines = [
            "--- Contexte de session ---",
            f"  Session ID : {self.session_id}",
            f"  Messages en memoire : {len(self.messages)}",
            f"  Decisions enregistrees : {len(self.decisions)}",
            f"  Staged edits non appliques : {len(self.staged_edits)}",
            f"  Historique : {self.history_path}",
            f"  Session file : {self.session_path}",
        ]
        if self.messages:
            lines.append(f"  Derniers echanges :")
            for m in self.messages[-6:]:
                content = m['content'][:100].replace('\n', ' ')
                lines.append(f"    [{m['role']}] {content}...")
        return "\n".join(lines)

    def show_decisions(self, limit: int = 12) -> str:
        recent = self.decisions[-limit:]
        if not recent:
            return "No decisions yet in this session"

        lines = []
        for d in recent:
            lines.append(f"step {d.get('step')}: {d.get('action')} | {d.get('reason')}")
        return "\n".join(lines)
