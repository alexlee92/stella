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


def _extract_json_candidate(text: str) -> str:
    raw = text.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()

    if raw.startswith("{") and raw.endswith("}"):
        return raw

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        return raw[start : end + 1]

    return raw


def ask_llm_json(prompt: str, retries: int = 3) -> dict:
    strict_prompt = (
        prompt
        + "\n\nReturn strict JSON only. No markdown, no prose."
        + " If previous response was invalid, fix it and return valid JSON now."
    )

    last_error = ""
    for _ in range(max(1, retries)):
        raw = ask_llm(strict_prompt)
        candidate = _extract_json_candidate(raw)

        try:
            return json.loads(candidate)
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
