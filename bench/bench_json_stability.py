"""
bench_json_stability.py - Measure strict JSON reliability with the single model.

Usage:
    python bench/bench_json_stability.py --direct
    python bench/bench_json_stability.py --url http://localhost:5000/query --runs 5
"""

import argparse
import json
import sys
import time

import requests

MODEL = "qwen2.5-coder:14b-instruct-q5_K_M"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

PROMPTS = [
    {
        "name": "planner_read_file",
        "task_type": "planning",
        "prompt": (
            'Return strict JSON only: {"action":"read_file","reason":"inspect file","args":{"path":"agent/llm_interface.py"}}'
        ),
        "required_keys": ["action", "reason", "args"],
    },
    {
        "name": "planner_search_code",
        "task_type": "planning",
        "prompt": (
            'Return strict JSON only: {"action":"search_code","reason":"find ask_llm","args":{"pattern":"def ask_llm"}}'
        ),
        "required_keys": ["action", "reason", "args"],
    },
    {
        "name": "critic_approve",
        "task_type": "analysis",
        "prompt": (
            'Return strict JSON only: {"approve":true,"reason":"valid decision","patched_decision":null}'
        ),
        "required_keys": ["approve", "reason"],
    },
    {
        "name": "finish_action",
        "task_type": "json",
        "prompt": (
            'Return strict JSON only: {"action":"finish","reason":"done","args":{"summary":"task complete"}}'
        ),
        "required_keys": ["action", "reason", "args"],
    },
]

VALID_ACTIONS = {
    "list_files",
    "read_file",
    "read_many",
    "search_code",
    "propose_edit",
    "apply_edit",
    "apply_all_staged",
    "run_tests",
    "run_quality",
    "project_map",
    "git_branch",
    "git_commit",
    "git_diff",
    "finish",
}


def _extract_json(text: str) -> tuple[bool, dict | None, str]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return False, None, str(exc)

    if not isinstance(parsed, dict):
        return False, None, "json_not_object"
    return True, parsed, ""


def _query_direct(prompt: str, timeout: int = 120) -> str:
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def run_direct(runs: int = 5):
    print(f"\n{'=' * 80}")
    print("JSON stability benchmark - direct Ollama mode")
    print(f"Model: {MODEL}")
    print(f"Runs per prompt: {runs}")
    print(f"{'=' * 80}\n")

    results: dict[str, dict[str, int | float]] = {}
    for case in PROMPTS:
        ok_count = 0
        parse_failures = 0
        key_failures = 0
        action_failures = 0

        print(f"\n[{case['name']}]")
        for i in range(runs):
            try:
                t0 = time.time()
                raw = _query_direct(case["prompt"])
                elapsed = round(time.time() - t0, 2)
                ok, parsed, err = _extract_json(raw)
                if not ok:
                    parse_failures += 1
                    print(
                        f"  run {i + 1}/{runs}: [KO] parse error ({err}) - {elapsed}s"
                    )
                    continue

                missing = [k for k in case["required_keys"] if k not in parsed]
                if missing:
                    key_failures += 1
                    print(
                        f"  run {i + 1}/{runs}: [KO] missing keys {missing} - {elapsed}s"
                    )
                    continue

                if "action" in parsed and parsed["action"] not in VALID_ACTIONS:
                    action_failures += 1
                    print(
                        f"  run {i + 1}/{runs}: [KO] invalid action {parsed['action']} - {elapsed}s"
                    )
                    continue

                ok_count += 1
                print(f"  run {i + 1}/{runs}: [OK] valid JSON - {elapsed}s")
            except Exception as exc:
                parse_failures += 1
                print(f"  run {i + 1}/{runs}: [KO] request error ({exc})")

        success = round((ok_count / runs) * 100, 1) if runs else 0.0
        results[case["name"]] = {
            "success_rate": success,
            "ok_count": ok_count,
            "parse_failures": parse_failures,
            "key_failures": key_failures,
            "action_failures": action_failures,
        }
        print(
            f"  => success: {success}% ({ok_count}/{runs}) "
            f"| parse:{parse_failures} key:{key_failures} action:{action_failures}"
        )

    global_success = round(
        sum(v["success_rate"] for v in results.values()) / len(results), 1
    )
    print(f"\n{'=' * 52}")
    print(f"Global JSON success rate: {global_success}%")
    print("Target minimum: 80%")
    print("Target optimal: 90%")


def run_via_routing(url: str, runs: int = 5):
    print(f"\n{'=' * 80}")
    print("JSON stability benchmark - routing server mode")
    print(f"URL: {url}")
    print(f"Runs per prompt: {runs}")
    print(f"{'=' * 80}\n")

    for case in PROMPTS:
        ok_count = 0
        print(f"\n[{case['name']}]")

        for i in range(runs):
            try:
                t0 = time.time()
                resp = requests.post(
                    url,
                    json={"prompt": case["prompt"], "task_type": case["task_type"]},
                    timeout=200,
                )
                data = resp.json()
                raw = data.get("response", "")
                model_used = data.get("model_used", "?")
                elapsed = round(time.time() - t0, 2)

                ok, parsed, err = _extract_json(raw)
                if not ok:
                    print(
                        f"  run {i + 1}/{runs}: [KO] parse error ({err}) - {elapsed}s | model={model_used}"
                    )
                    continue

                missing = [k for k in case["required_keys"] if k not in parsed]
                if missing:
                    print(
                        f"  run {i + 1}/{runs}: [KO] missing keys {missing} - {elapsed}s | model={model_used}"
                    )
                    continue

                if "action" in parsed and parsed["action"] not in VALID_ACTIONS:
                    print(
                        f"  run {i + 1}/{runs}: [KO] invalid action {parsed['action']} - {elapsed}s | model={model_used}"
                    )
                    continue

                ok_count += 1
                print(
                    f"  run {i + 1}/{runs}: [OK] valid JSON - {elapsed}s | model={model_used}"
                )
            except Exception as exc:
                print(f"  run {i + 1}/{runs}: [KO] request error ({exc})")

        success = round((ok_count / runs) * 100, 1) if runs else 0.0
        print(f"  => success: {success}% ({ok_count}/{runs})")


def main():
    parser = argparse.ArgumentParser(
        description="JSON stability benchmark (single model)"
    )
    parser.add_argument("--url", default="http://localhost:5000/query")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--direct", action="store_true", help="Call Ollama directly")
    args = parser.parse_args()

    if args.direct:
        run_direct(args.runs)
        return

    try:
        requests.get(args.url.replace("/query", "/health"), timeout=5)
    except Exception:
        print(f"[!] Routing server is not accessible at {args.url}")
        print("    Use direct mode: python bench/bench_json_stability.py --direct")
        sys.exit(1)

    run_via_routing(args.url, args.runs)


if __name__ == "__main__":
    main()
