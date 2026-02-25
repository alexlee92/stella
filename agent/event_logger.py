import json
import os
from datetime import datetime, UTC

from agent.config import EVENT_LOG_PATH, PROJECT_ROOT


class EventLogger:
    def __init__(self, rel_path: str = EVENT_LOG_PATH):
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        self.path = abs_path

    def log(self, event_type: str, payload: dict):
        record = {
            "type": event_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "payload": payload,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_failure(self, category: str, message: str, payload: dict | None = None):
        self.log(
            "failure",
            {
                "category": category,
                "message": message,
                "payload": payload or {},
            },
        )
