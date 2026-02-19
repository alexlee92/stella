import json
import re
from collections import OrderedDict
from typing import Any

import requests

from agent.config import MODEL, OLLAMA_URL, REQUEST_TIMEOUT

_JSON_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_JSON_CACHE_SIZE = 64


def _strip_fences(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            first = lines[0].strip().lower()
            if first in {"```", "```json"}:
                raw = "\n".join(lines[1:-1]).strip()
    return raw


def _extract_json_candidates(text: str) -> list[str]:
    raw = _strip_fences(text)
    candidates: list[str] = []

    if raw.startswith("{") and raw.endswith("}"):
        candidates.append(raw)

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        candidates.append(raw[start : end + 1])

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            _, end_idx = decoder.raw_decode(raw[idx:])
            candidates.append(raw[idx : idx + end_idx])
        except json.JSONDecodeError:
            continue

    seen = set()
    out = []
    for c in candidates:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _repair_json_text(candidate: str) -> str:
    text = candidate.strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def _parse_json_candidate(candidate: str) -> dict[str, Any] | None:
    for payload in (candidate, _repair_json_text(candidate)):
        try:
            parsed = json.loads(payload)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    return None


def _repair_prompt(raw: str) -> str:
    clipped = (raw or "")[:5000]
    return (
        "Convert the following text into one strict JSON object.\n"
        "Rules: JSON only, double quotes, no markdown, no explanation.\n\n"
        f"{clipped}"
    )


def _build_json_strict_prompt(prompt: str) -> str:
    return (
        prompt
        + "\n\nReturn strict JSON only. No markdown, no prose."
        + " If previous response was invalid, fix it and return valid JSON now."
    )


def _build_json_constrained_prompt(prompt: str) -> str:
    return (
        prompt
        + "\n\nIMPORTANT FORMAT RULES:"
        + "\n- Output exactly one JSON object."
        + "\n- Use only double quotes."
        + "\n- Do not include markdown fences."
        + "\n- Do not include commentary before or after JSON."
        + "\n- Keep keys and value types stable."
    )


def _parse_json_response(raw: str) -> tuple[dict[str, Any] | None, str]:
    text = (raw or "").strip()
    if not text:
        return None, "empty_response"

    candidates = _extract_json_candidates(text)
    if not candidates:
        return None, "no_json_object"

    non_object_seen = False
    decode_fail_seen = False
    for candidate in candidates:
        for payload in (candidate, _repair_json_text(candidate)):
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, dict):
                    return parsed, "ok"
                non_object_seen = True
            except json.JSONDecodeError:
                decode_fail_seen = True

    if non_object_seen:
        return None, "non_object_json"
    if decode_fail_seen:
        return None, "json_decode_error"
    return None, "unknown_parse_error"


def _fallback_json_error(
    error_class: str, attempts: list[str], prompt_class: str
) -> dict[str, Any]:
    return {
        "action": "finish",
        "reason": "json_fallback",
        "args": {"summary": f"Planner JSON error: {error_class}"},
        "_error_type": "parse",
        "_error": error_class,
        "_parse_meta": {
            "error_class": error_class,
            "attempts": attempts,
            "attempt_count": len(attempts),
            "prompt_class": prompt_class,
        },
    }


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
    except Exception as exc:
        print(f"[llm] error: {exc}")
        return ""


def ask_llm_json(
    prompt: str, retries: int = 3, prompt_class: str = "generic"
) -> dict[str, Any]:
    strict_prompt = _build_json_strict_prompt(prompt)

    cached = _JSON_CACHE.get(strict_prompt)
    if cached is not None:
        _JSON_CACHE.move_to_end(strict_prompt)
        return cached

    attempt_errors: list[str] = []
    max_retries = max(1, retries)
    stages = [
        _build_json_strict_prompt(prompt),
        _build_json_constrained_prompt(prompt),
    ]

    for stage_prompt in stages:
        for _ in range(max_retries):
            raw = ask_llm(stage_prompt)
            parsed, err = _parse_json_response(raw)
            if parsed is not None:
                _JSON_CACHE[strict_prompt] = parsed
                _JSON_CACHE.move_to_end(strict_prompt)
                while len(_JSON_CACHE) > _JSON_CACHE_SIZE:
                    _JSON_CACHE.popitem(last=False)
                return parsed

            attempt_errors.append(err)
            if raw:
                repaired_raw = ask_llm(_repair_prompt(raw))
                repaired, repaired_err = _parse_json_response(repaired_raw)
                if repaired is not None:
                    _JSON_CACHE[strict_prompt] = repaired
                    _JSON_CACHE.move_to_end(strict_prompt)
                    while len(_JSON_CACHE) > _JSON_CACHE_SIZE:
                        _JSON_CACHE.popitem(last=False)
                    return repaired
                attempt_errors.append(f"repair:{repaired_err}")

    final_error = attempt_errors[-1] if attempt_errors else "unknown_parse_error"
    return _fallback_json_error(final_error, attempt_errors, prompt_class)
