"""
bench_model_comparison.py - Evaluation du modele unique sur des taches variees.

Usage:
    python bench/bench_model_comparison.py --direct
    python bench/bench_model_comparison.py --url http://localhost:5000/query
"""

import argparse
import sys
import time

import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5-coder:14b-instruct-q5_K_M"

TASKS = {
    "analyse_architecture": {
        "task_type": "analysis",
        "prompt": (
            "Analyse cette fonction et identifie tous les problemes potentiels:\n"
            "def process(data):\n"
            "    result = []\n"
            "    for item in data:\n"
            "        result.append(item['value'] / item['total'])\n"
            "    return result"
        ),
        "expected_keywords": [
            "division",
            "zero",
            "keyerror",
            "none",
            "erreur",
            "exception",
        ],
    },
    "generation_fonction": {
        "task_type": "backend",
        "prompt": (
            "Genere une fonction Python safe_divide(a, b) qui:\n"
            "- Retourne None si b == 0\n"
            "- Leve ValueError si a ou b n'est pas un nombre\n"
            "- Retourne a / b sinon"
        ),
        "expected_keywords": [
            "def safe_divide",
            "if b == 0",
            "return none",
            "valueerror",
            "isinstance",
        ],
    },
    "debug_traceback": {
        "task_type": "debug",
        "prompt": (
            "Ce code Python leve une exception:\n"
            "lst = [1, 2, 3]\n"
            "print(lst[10])\n"
            "Quel est le type d'exception et comment corriger?"
        ),
        "expected_keywords": ["indexerror", "index", "out of range", "10", "len"],
    },
    "json_strict": {
        "task_type": "json",
        "prompt": (
            "Retourne UNIQUEMENT ce JSON, sans modification:\n"
            '{"action":"finish","reason":"tache complete","args":{"summary":"Analyse terminee avec succes"}}'
        ),
        "expected_keywords": ['"action"', '"finish"', '"summary"'],
    },
}


def _score(response: str, keywords: list[str]) -> tuple[int, int]:
    low = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in low)
    return hits, len(keywords)


def _query_ollama(prompt: str, timeout: int = 120) -> str:
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


def run_evaluation(url: str | None = None):
    direct = url is None
    print(f"\n{'=' * 80}")
    print("Evaluation du modele unique")
    print(f"Modele: {MODEL}")
    print(f"Mode : {'direct Ollama' if direct else url}")
    print(f"{'=' * 80}\n")

    if not direct:
        try:
            health_url = url.replace("/query", "/health")
            requests.get(health_url, timeout=5)
        except Exception:
            print("[!] Serveur de routing non accessible.")
            print(
                "    Utilisez le mode direct : python bench/bench_model_comparison.py --direct"
            )
            sys.exit(1)

    for task_name, task in TASKS.items():
        t0 = time.time()
        try:
            if direct:
                response = _query_ollama(task["prompt"])
                model_used = MODEL
            else:
                resp = requests.post(
                    url,
                    json={"prompt": task["prompt"], "task_type": task["task_type"]},
                    timeout=200,
                )
                data = resp.json()
                response = data.get("response", "")
                model_used = data.get("model_used", "?")
            elapsed = round(time.time() - t0, 1)
            hits, total = _score(response, task["expected_keywords"])
            score_pct = round((hits / total) * 100) if total else 0
            status = (
                "[OK]" if score_pct >= 60 else ("[!!]" if score_pct >= 40 else "[KO]")
            )
            print(
                f"{status} {task_name:<22} | model: {model_used:<34} | score: {hits}/{total} ({score_pct}%) | {elapsed}s"
            )
        except Exception as exc:
            print(f"[KO] {task_name:<22} | ERREUR - {exc}")

    print(f"\n{'=' * 80}")
    print("Interpretation:")
    print("  [OK]  >=60% de mots-cles trouves")
    print("  [!!]  40-59%")
    print("  [KO]  <40% ou erreur")


def main():
    parser = argparse.ArgumentParser(description="Evaluation du modele unique")
    parser.add_argument("--url", default="http://localhost:5000/query")
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Appeler Ollama directement sans serveur de routing",
    )
    args = parser.parse_args()

    if args.direct:
        run_evaluation(url=None)
    else:
        run_evaluation(url=args.url)


if __name__ == "__main__":
    main()
