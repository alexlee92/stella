import json
import requests

from agent.config import MODEL, OLLAMA_URL, REQUEST_TIMEOUT


def ask_llm(prompt: str) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "")
    except Exception as e:
        print(f"[llm] error: {e}")
        return ""


def ask_llm_json(prompt: str, retries: int = 3) -> dict:
    strict_prompt = (
        prompt
        + "\n\nReturn strict JSON only. No markdown, no prose."
        + " If previous response was invalid, fix it and return valid JSON now."
    )

    last_error = ""
    for _ in range(max(1, retries)):
        raw = ask_llm(strict_prompt)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            last_error = str(exc)
            continue

    return {
        "action": "finish",
        "reason": "json_fallback",
        "args": {"summary": f"Planner JSON error: {last_error}"},
        "_error_type": "parse",
        "_error": last_error,
    }
