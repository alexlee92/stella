"""
bench_latency.py - Measure latency with the single Stella model.

Usage:
    python bench/bench_latency.py --direct
    python bench/bench_latency.py --url http://localhost:5000/query --runs 3
"""

import argparse
import sys
import time

import requests

MODEL = "qwen2.5-coder:14b-instruct-q5_K_M"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

TASKS = [
    ("simple_question", "optimization", "What is the complexity of O(n log n)?"),
    (
        "short_analysis",
        "analysis",
        "Analyze this Python code: def add(a, b): return a + b",
    ),
    (
        "long_analysis",
        "analysis",
        "Analyze an autonomous coding agent architecture with planner, memory, quality gate and rollback.",
    ),
    (
        "simple_generation",
        "backend",
        "Generate a Python function for HTTP GET with timeout and error handling.",
    ),
    (
        "complex_generation",
        "backend",
        "Generate a complete FastAPI CRUD class for User with JWT auth and validation.",
    ),
    (
        "json_strict",
        "json",
        'Return JSON only: {"action":"read_file","reason":"inspect file","args":{"path":"agent/llm_interface.py"}}',
    ),
    (
        "debug_simple",
        "debug",
        "This code raises KeyError: d = {}; print(d['a']). Explain and fix.",
    ),
    (
        "planning_json",
        "planning",
        "You are an autonomous agent. Return strict JSON with one action among read_file/search_code/propose_edit/finish.",
    ),
]


def _query_ollama_direct(prompt: str, timeout: int = 300) -> tuple[str, str]:
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
    content = resp.json().get("message", {}).get("content", "")
    return MODEL, content


def run_bench_direct(runs: int = 3):
    print(f"\n{'=' * 80}")
    print("Latency benchmark - direct Ollama mode")
    print(f"Model: {MODEL}")
    print(f"Runs per task: {runs}")
    print(f"{'=' * 80}\n")

    header = f"{'Task':<22} {'Model':<36} {'Avg (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Chars'}"
    print(header)
    print("-" * len(header))

    for name, _task_type, prompt in TASKS:
        times = []
        lengths = []
        errors = 0
        for _ in range(runs):
            t0 = time.time()
            try:
                _model, content = _query_ollama_direct(prompt, timeout=300)
                times.append(round(time.time() - t0, 2))
                lengths.append(len(content))
            except Exception:
                errors += 1
                times.append(round(time.time() - t0, 2))
                lengths.append(0)

        avg = round(sum(times) / len(times), 2) if times else 0
        mn = round(min(times), 2) if times else 0
        mx = round(max(times), 2) if times else 0
        avg_len = round(sum(lengths) / len(lengths)) if lengths else 0
        err_str = f" [ERR:{errors}]" if errors else ""
        print(f"{name:<22} {MODEL:<36} {avg:<10} {mn:<10} {mx:<10} {avg_len}{err_str}")


def run_bench_via_routing(url: str, runs: int = 3):
    print(f"\n{'=' * 80}")
    print("Latency benchmark - routing server mode")
    print(f"URL: {url}")
    print(f"Runs per task: {runs}")
    print(f"{'=' * 80}\n")

    header = f"{'Task':<22} {'Model used':<36} {'Avg (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Chars'}"
    print(header)
    print("-" * len(header))

    total_errors = 0
    for name, task_type, prompt in TASKS:
        times = []
        lengths = []
        model_used = "?"
        errors = 0

        for _ in range(runs):
            t0 = time.time()
            try:
                resp = requests.post(
                    url, json={"prompt": prompt, "task_type": task_type}, timeout=300
                )
                data = resp.json()
                elapsed = round(time.time() - t0, 2)
                response_text = data.get("response", "")
                model_used = data.get("model_used", "?")
                times.append(elapsed)
                lengths.append(len(response_text))
            except Exception:
                errors += 1
                total_errors += 1
                times.append(round(time.time() - t0, 2))
                lengths.append(0)

        avg = round(sum(times) / len(times), 2) if times else 0
        mn = round(min(times), 2) if times else 0
        mx = round(max(times), 2) if times else 0
        avg_len = round(sum(lengths) / len(lengths)) if lengths else 0
        err_str = f" [ERR:{errors}]" if errors else ""
        print(
            f"{name:<22} {model_used[:34]:<36} {avg:<10} {mn:<10} {mx:<10} {avg_len}{err_str}"
        )

    print(f"\nTotal errors: {total_errors}/{len(TASKS) * runs}")


def main():
    parser = argparse.ArgumentParser(description="Latency benchmark (single model)")
    parser.add_argument("--url", default="http://localhost:5000/query")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--direct", action="store_true", help="Call Ollama directly")
    args = parser.parse_args()

    if args.direct:
        run_bench_direct(args.runs)
        return

    try:
        requests.get(args.url.replace("/query", "/health"), timeout=5)
    except Exception:
        print(f"[!] Routing server is not accessible at {args.url}")
        print("    Use direct mode: python bench/bench_latency.py --direct")
        sys.exit(1)

    run_bench_via_routing(args.url, args.runs)


if __name__ == "__main__":
    main()
