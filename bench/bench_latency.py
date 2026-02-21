"""
bench_latency.py — Mesure la latence des modèles Orisha selon le type de tâche.

Usage :
    python bench/bench_latency.py
    python bench/bench_latency.py --url http://localhost:5000 --runs 3
"""
import argparse
import sys
import time

import requests

TASKS = [
    ("simple_question",    "optimization", "Quelle est la complexité algorithmique de O(n log n) ?"),
    ("analyse_courte",     "analysis",     "Analyse ce code Python : def add(a, b): return a + b"),
    ("analyse_longue",     "analysis",
     "Analyse l'architecture complète d'un agent IA autonome qui utilise un planner LLM, "
     "une mémoire vectorielle BM25+cosine, un système de quality gate (format, lint, tests), "
     "un patcher transactionnel avec rollback et un mode fix-until-green."),
    ("generation_simple",  "backend",
     "Génère une fonction Python qui fait une requête HTTP GET avec gestion d'erreur et timeout."),
    ("generation_complexe","backend",
     "Génère une classe Python complète FastAPI avec CRUD complet pour une entité User "
     "avec authentification JWT, validation Pydantic et gestion des erreurs HTTP."),
    ("json_strict",        "json",
     'Retourne UNIQUEMENT ce JSON sans aucune modification ni explication : '
     '{"action":"read_file","reason":"inspecter le fichier","args":{"path":"agent/llm_interface.py"}}'),
    ("debug_simple",       "debug",
     "Ce code lève KeyError : d = {}; print(d['a']). Explique l'erreur et donne la correction."),
    ("refactor",           "refactor",
     "Refactorise cette fonction pour la rendre plus lisible : "
     "def f(x,y,z): return x*y+z if x>0 else y*z+x if y>0 else x+y+z"),
    ("planning_json",      "planning",
     'Tu es un agent autonome. Goal: corriger un bug dans agent/llm.py. '
     'Retourne strict JSON: {"action":"...","reason":"...","args":{...}} '
     'Actions valides: read_file, search_code, propose_edit, finish'),
]


def run_bench(url: str, runs: int = 3):
    print(f"\n{'='*80}")
    print(f"Benchmark latence Orisha — {url}")
    print(f"Runs par tâche : {runs}")
    print(f"{'='*80}\n")

    header = f"{'Tâche':<25} {'Modèle':<22} {'Moy (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Longueur'}"
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
                    url,
                    json={"prompt": prompt, "task_type": task_type},
                    timeout=300,
                )
                data = resp.json()
                elapsed = round(time.time() - t0, 2)
                response_text = data.get("response", "")
                model_used = data.get("model_used", "?")
                times.append(elapsed)
                lengths.append(len(response_text))
            except Exception as e:
                errors += 1
                total_errors += 1
                elapsed = round(time.time() - t0, 2)
                times.append(elapsed)
                lengths.append(0)

        avg = round(sum(times) / len(times), 2) if times else 0
        mn = round(min(times), 2) if times else 0
        mx = round(max(times), 2) if times else 0
        avg_len = round(sum(lengths) / len(lengths)) if lengths else 0
        err_str = f" [ERR:{errors}]" if errors else ""

        print(f"{name:<25} {model_used:<22} {avg:<10} {mn:<10} {mx:<10} {avg_len}{err_str}")

    print(f"\nTotal erreurs : {total_errors}/{len(TASKS) * runs}")
    print("\nSeuils cibles :")
    print("  simple_question   < 5s")
    print("  analyse_longue    < 30s")
    print("  generation_complexe < 45s")
    print("  json_strict       < 10s  (taux succès JSON attendu > 85%)")


OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODELS = {
    "analysis": "Orisha-ifa1.0:latest",
    "refactor": "Orisha-ifa1.0:latest",
    "planning": "Orisha-ifa1.0:latest",
    "json": "Orisha-ifa1.0:latest",
    "debug": "Orisha-Oba1.0:latest",
    "optimization": "Orisha-Oba1.0:latest",
    "frontend": "Orisha-Oba1.0:latest",
    "backend": "Orisha-Oba1.0:latest",
    "generation": "Orisha-Oba1.0:latest",
}

def _query_ollama_direct(prompt: str, task_type: str, timeout: int = 300) -> tuple[str, str]:
    model = OLLAMA_MODELS.get(task_type, "Orisha-Oba1.0:latest")
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "")
    return model, content


def run_bench_direct(runs: int = 3):
    """Mode direct : interroge Ollama sans passer par Flask."""
    print(f"\n{'='*80}")
    print(f"Benchmark latence — mode direct Ollama")
    print(f"Runs par tâche : {runs}")
    print(f"{'='*80}\n")

    header = f"{'Tâche':<25} {'Modèle':<22} {'Moy (s)':<10} {'Min (s)':<10} {'Max (s)':<10} {'Longueur'}"
    print(header)
    print("-" * len(header))

    for name, task_type, prompt in TASKS:
        times = []
        lengths = []
        model_used = "?"
        errors = 0
        for _ in range(runs):
            t0 = time.time()
            try:
                model_used, content = _query_ollama_direct(prompt, task_type, timeout=300)
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
        print(f"{name:<25} {model_used[:20]:<22} {avg:<10} {mn:<10} {mx:<10} {avg_len}{err_str}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark latence Orisha")
    parser.add_argument("--url", default="http://localhost:5000/query")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--direct", action="store_true",
                        help="Appeler Ollama directement sans le serveur Flask")
    args = parser.parse_args()

    if args.direct:
        run_bench_direct(args.runs)
        return

    try:
        requests.get(args.url.replace("/query", "/health"), timeout=5)
    except Exception:
        print(f"[!] Le serveur Orisha n'est pas accessible à {args.url}")
        print("    Lancez d'abord : python orisha_server.py")
        print("    Ou utilisez le mode direct : python bench/bench_latency.py --direct")
        sys.exit(1)

    run_bench(args.url, args.runs)


if __name__ == "__main__":
    main()
