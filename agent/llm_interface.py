import json
import re
from collections import OrderedDict
from typing import Any, Generator

import requests

from agent.config import (
    MODEL,
    OLLAMA_URL,
    REQUEST_TIMEOUT,
    OPENAI_API_KEY,
)

_JSON_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_JSON_CACHE_SIZE = 64


def _strip_fences(text: str) -> str:
    """B2-fix: strip any ``` fence regardless of language tag."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            # Drop the opening fence line (```json, ```python, ```yaml, `` ` ``, …)
            raw = "\n".join(lines[1:-1]).strip()
    return raw


def _extract_json_candidates(text: str) -> list[str]:
    """B1-fix: also parse JSON arrays in addition to objects.
    Arrays are prioritised when the raw text starts with '['.
    """
    raw = _strip_fences(text)
    candidates: list[str] = []

    # Prioritise: if raw text itself is an array/object, add it first
    if raw.startswith("[") and raw.endswith("]"):
        candidates.append(raw)
    elif raw.startswith("{") and raw.endswith("}"):
        candidates.append(raw)

    # --- arrays ---
    start_arr = raw.find("[")
    end_arr = raw.rfind("]")
    if start_arr >= 0 and end_arr > start_arr:
        candidates.append(raw[start_arr : end_arr + 1])

    # --- objects ---
    start_obj = raw.find("{")
    end_obj = raw.rfind("}")
    if start_obj >= 0 and end_obj > start_obj:
        candidates.append(raw[start_obj : end_obj + 1])

    decoder = json.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch not in ("{", "["):
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


def _parse_json_response(raw: str) -> tuple[dict[str, Any] | list | None, str]:
    """B1-fix: accept both JSON objects (dict) and arrays (list)."""
    text = (raw or "").strip()
    if not text:
        return None, "empty_response"

    candidates = _extract_json_candidates(text)
    if not candidates:
        return None, "no_json_object"

    decode_fail_seen = False
    for candidate in candidates:
        for payload in (candidate, _repair_json_text(candidate)):
            try:
                parsed = json.loads(payload)
                if isinstance(parsed, (dict, list)):
                    return parsed, "ok"
            except json.JSONDecodeError:
                decode_fail_seen = True

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



def ask_llm(
    prompt: str,
    system_prompt: str = "You are a helpful coding assistant.",
    task_type: str | None = None,  # no-op: kept for API compatibility
    concise: bool = False,  # no-op: kept for API compatibility
    json_mode: bool = False,  # P1.2 — force Ollama format:json
) -> str:
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if OPENAI_API_KEY and (MODEL.startswith("gpt-") or MODEL.startswith("o1-")):
            from openai import OpenAI

            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                timeout=REQUEST_TIMEOUT,
            )
            return response.choices[0].message.content

        ollama_body: dict[str, Any] = {
            "model": MODEL,
            "messages": messages,
            "stream": False,
        }
        if json_mode:
            ollama_body["format"] = "json"  # P1.2 â€" JSON schema enforcement
        response = requests.post(
            OLLAMA_URL,
            json=ollama_body,
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")
    except requests.ConnectionError:
        print("[llm] Serveur inaccessible. VÃ©rifiez qu'Ollama/routing est dÃ©marrÃ©.")
        return ""
    except requests.Timeout:
        print(
            f"[llm] Timeout aprÃ¨s {REQUEST_TIMEOUT}s. Le modÃ¨le est peut-Ãªtre surchargÃ©."
        )
        return ""
    except Exception as exc:
        print(f"[llm] Erreur inattendue : {exc}")
        return ""


def ask_llm_json(
    prompt: str,
    retries: int = 3,
    prompt_class: str = "generic",
    system_prompt: str = "You are a helpful coding assistant that only outputs valid JSON.",
    task_type: str | None = None,
) -> dict[str, Any] | list:
    """B1-fix: can now return a list when LLM outputs a JSON array (e.g. plan_files)."""
    # task_type par défaut pour JSON = modèle analytique (single model)
    effective_task_type = task_type or "json"
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
            raw = ask_llm(
                stage_prompt,
                system_prompt=system_prompt,
                task_type=effective_task_type,
                json_mode=True,
            )
            parsed, err = _parse_json_response(raw)
            if parsed is not None:
                _JSON_CACHE[strict_prompt] = parsed
                _JSON_CACHE.move_to_end(strict_prompt)
                while len(_JSON_CACHE) > _JSON_CACHE_SIZE:
                    _JSON_CACHE.popitem(last=False)
                return parsed

            attempt_errors.append(err)
            if raw:
                repaired_raw = ask_llm(
                    _repair_prompt(raw), task_type="json", json_mode=True
                )
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


def ask_llm_stream(
    prompt: str,
    system_prompt: str = "You are a helpful coding assistant.",
    task_type: str | None = None,  # no-op: kept for API compatibility
) -> Generator[str, None, None]:
    """Génère les tokens LLM progressivement via Ollama streaming.

    Usage:
        for token in ask_llm_stream(prompt):
            print(token, end="", flush=True)
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            OLLAMA_URL,
            json={"model": MODEL, "messages": messages, "stream": True},
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue
    except Exception as exc:
        print(f"\n[llm] Stream error: {exc}")
        return


def ask_llm_stream_print(
    prompt: str,
    system_prompt: str = "You are a helpful coding assistant.",
    task_type: str | None = None,
) -> str:
    """Stream les tokens vers stdout et retourne le texte complet."""
    parts = []
    for token in ask_llm_stream(
        prompt, system_prompt=system_prompt, task_type=task_type
    ):
        print(token, end="", flush=True)
        parts.append(token)
    print()  # newline finale
    return "".join(parts)
