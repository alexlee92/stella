"""
bench_json_stability.py — Mesure la stabilité des réponses JSON des modèles.

Ce benchmark vérifie que les modèles Orisha retournent du JSON valide de manière fiable
pour les prompts critiques utilisés par le planner de Stella.

Usage :
    python bench/bench_json_stability.py --direct
    python bench/bench_json_stability.py --runs 5 --url http://localhost:5000
"""
import argparse
import json
import sys
import time

import requests

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODELS = {
    "planning": "Orisha-ifa1.0:latest",
    "analysis": "Orisha-ifa1.0:latest",
    "json": "Orisha-ifa1.0:latest",
    "debug": "Orisha-Oba1.0:latest",
    "optimization": "Orisha-Oba1.0:latest",
    "backend": "Orisha-Oba1.0:latest",
}


def _query_ollama_direct(prompt: str, task_type: str, timeout: int = 120) -> str:
    model = OLLAMA_MODELS.get(task_type, "Orisha-ifa1.0:latest")
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")

PLANNER_PROMPTS = [
    {
        "name": "action_read_file",
        "task_type": "planning",
        "prompt": (
            'Tu es un agent autonome. Goal: inspecter le fichier agent/llm_interface.py pour comprendre comment ask_llm est implémenté.\n'
            'Retourne strict JSON uniquement: {"action":"...","reason":"...","args":{...}}\n'
            'Actions valides: read_file, search_code, propose_edit, finish'
        ),
        "expected_keys": ["action", "reason", "args"],
        "expected_action": "read_file",
    },
    {
        "name": "action_search_code",
        "task_type": "planning",
        "prompt": (
            'Tu es un agent autonome. Goal: trouver où ask_llm est défini.\n'
            'Retourne strict JSON: {"action":"...","reason":"...","args":{...}}\n'
            'Actions valides: read_file, search_code, propose_edit, finish'
        ),
        "expected_keys": ["action", "reason", "args"],
        "expected_action": None,  # N'importe quelle action valide
    },
    {
        "name": "critique_approve",
        "task_type": "analysis",
        "prompt": (
            'Évalue cette décision d\'agent pour sa sécurité et son utilité.\n'
            'Décision: {"action":"read_file","reason":"inspecter","args":{"path":"agent/llm.py"}}\n'
            'Retourne strict JSON uniquement:\n'
            '{"approve": true, "reason": "décision valide et utile", "patched_decision": null}'
        ),
        "expected_keys": ["approve", "reason"],
        "expected_action": None,
    },
    {
        "name": "finish_action",
        "task_type": "planning",
        "prompt": (
            'Retourne strict JSON uniquement:\n'
            '{"action":"finish","reason":"tâche terminée","args":{"summary":"Le bug a été corrigé dans agent/llm_interface.py"}}'
        ),
        "expected_keys": ["action", "args"],
        "expected_action": "finish",
    },
    {
        "name": "propose_edit",
        "task_type": "planning",
        "prompt": (
            'Retourne strict JSON uniquement:\n'
            '{"action":"propose_edit","reason":"corriger le bug","args":{"path":"agent/llm.py","instruction":"Corriger la fonction ask_llm"}}'
        ),
        "expected_keys": ["action", "args"],
        "expected_action": "propose_edit",
    },
]

VALID_ACTIONS = {
    "list_files", "read_file", "read_many", "search_code",
    "propose_edit", "apply_edit", "apply_all_staged",
    "run_tests", "run_quality", "project_map",
    "git_branch", "git_commit", "git_diff", "finish",
}


def _is_valid_json(text: str) -> tuple[bool, dict | None, str]:
    """Tente de parser le JSON. Retourne (ok, parsed, erreur)."""
    text = text.strip()
    # Retirer les fences markdown si présentes
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    # Chercher le premier objet JSON
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return True, parsed, ""
        return False, None, "not_a_dict"
    except json.JSONDecodeError as e:
        return False, None, str(e)


def run_stability_bench_direct(runs: int = 5):
    """Mode direct : appelle Ollama sans Flask."""
    print(f"\n{'='*80}")
    print("Benchmark stabilité JSON — mode direct Ollama")
    print(f"Runs par prompt : {runs}")
    print(f"{'='*80}\n")

    all_results = {}

    for task in PLANNER_PROMPTS:
        name = task["name"]
        task_type = task["task_type"]
        prompt = task["prompt"]
        expected_keys = task["expected_keys"]
        expected_action = task["expected_action"]

        successes = 0
        key_failures = 0
        action_failures = 0
        parse_failures = 0
        times = []

        for _ in range(runs):
            t0 = time.time()
            try:
                response_text = _query_ollama_direct(prompt, task_type)
                elapsed = round(time.time() - t0, 2)
                times.append(elapsed)

                ok, parsed, err = _is_valid_json(response_text)
                if not ok:
                    parse_failures += 1
                    continue

                missing_keys = [k for k in expected_keys if k not in parsed]
                if missing_keys:
                    key_failures += 1
                    continue

                if expected_action and parsed.get("action") != expected_action:
                    action_failures += 1
                    continue

                if task_type == "planning" and "action" in parsed:
                    if parsed.get("action") not in VALID_ACTIONS:
                        action_failures += 1
                        continue

                successes += 1

            except Exception:
                parse_failures += 1
                times.append(round(time.time() - t0, 2))

        avg_time = round(sum(times) / len(times), 2) if times else 0
        success_rate = round((successes / runs) * 100)
        status = "[OK]" if success_rate >= 80 else ("[!!]" if success_rate >= 50 else "[KO]")

        print(f"{status} {name:<25} | succes: {successes}/{runs} ({success_rate}%) | moy: {avg_time}s")
        if parse_failures:
            print(f"     parse_failures={parse_failures}")
        if key_failures:
            print(f"     key_failures={key_failures}")
        if action_failures:
            print(f"     action_failures={action_failures}")

        all_results[name] = {
            "success_rate": success_rate,
            "avg_latency": avg_time,
            "parse_failures": parse_failures,
            "key_failures": key_failures,
            "action_failures": action_failures,
        }

    avg_success = round(sum(r["success_rate"] for r in all_results.values()) / len(all_results))
    print(f"\n{'='*50}")
    print(f"Taux de succès JSON global : {avg_success}%")
    print(f"Seuil minimal attendu      : 80%")
    if avg_success >= 90:
        print("Résultat : [OK] EXCELLENT")
    elif avg_success >= 80:
        print("Résultat : [OK] BON")
    elif avg_success >= 60:
        print("Résultat : [!!]  MOYEN")
    else:
        print("Résultat : [KO] INSUFFISANT")


def run_stability_bench(url: str, runs: int = 5):
    print(f"\n{'='*80}")
    print(f"Benchmark stabilité JSON — {url}")
    print(f"Runs par prompt : {runs}")
    print(f"{'='*80}\n")

    all_results = {}

    for task in PLANNER_PROMPTS:
        name = task["name"]
        task_type = task["task_type"]
        prompt = task["prompt"]
        expected_keys = task["expected_keys"]
        expected_action = task["expected_action"]

        successes = 0
        key_failures = 0
        action_failures = 0
        parse_failures = 0
        times = []

        for _ in range(runs):
            t0 = time.time()
            try:
                resp = requests.post(
                    url,
                    json={"prompt": prompt, "task_type": task_type},
                    timeout=200,
                )
                data = resp.json()
                response_text = data.get("response", "")
                elapsed = round(time.time() - t0, 2)
                times.append(elapsed)

                ok, parsed, err = _is_valid_json(response_text)
                if not ok:
                    parse_failures += 1
                    continue

                # Vérifier les clés attendues
                missing_keys = [k for k in expected_keys if k not in parsed]
                if missing_keys:
                    key_failures += 1
                    continue

                # Vérifier l'action si spécifiée
                if expected_action and parsed.get("action") != expected_action:
                    action_failures += 1
                    continue

                # Vérifier que l'action est valide si c'est un planner
                if task_type == "planning" and "action" in parsed:
                    if parsed.get("action") not in VALID_ACTIONS:
                        action_failures += 1
                        continue

                successes += 1

            except Exception as e:
                parse_failures += 1
                times.append(round(time.time() - t0, 2))

        avg_time = round(sum(times) / len(times), 2) if times else 0
        success_rate = round((successes / runs) * 100)
        status = "[OK]" if success_rate >= 80 else ("[!!]" if success_rate >= 50 else "[KO]")

        print(f"{status} {name:<25} | succes: {successes}/{runs} ({success_rate}%) | moy: {avg_time}s")
        if parse_failures:
            print(f"     parse_failures={parse_failures}")
        if key_failures:
            print(f"     key_failures={key_failures}")
        if action_failures:
            print(f"     action_failures={action_failures}")

        all_results[name] = {
            "success_rate": success_rate,
            "avg_latency": avg_time,
            "parse_failures": parse_failures,
            "key_failures": key_failures,
            "action_failures": action_failures,
        }

    # Résumé global
    avg_success = round(sum(r["success_rate"] for r in all_results.values()) / len(all_results))
    print(f"\n{'='*50}")
    print(f"Taux de succès JSON global : {avg_success}%")
    print(f"Seuil minimal attendu      : 80%")
    print(f"Seuil optimal attendu      : 90%")
    if avg_success >= 90:
        print("Résultat : [OK] EXCELLENT")
    elif avg_success >= 80:
        print("Résultat : [OK] BON")
    elif avg_success >= 60:
        print("Résultat : [!!]  MOYEN — améliorer les system prompts des modèles")
    else:
        print("Résultat : [KO] INSUFFISANT — vérifier les modèles Ollama et les system prompts")


def main():
    parser = argparse.ArgumentParser(description="Benchmark stabilité JSON Orisha")
    parser.add_argument("--url", default="http://localhost:5000/query")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--direct", action="store_true",
                        help="Appeler Ollama directement sans le serveur Flask")
    args = parser.parse_args()

    if args.direct:
        run_stability_bench_direct(args.runs)
        return

    try:
        requests.get(args.url.replace("/query", "/health"), timeout=5)
    except Exception:
        print(f"[!] Serveur Orisha non accessible à {args.url}")
        print("    Lancez : python orisha_server.py")
        print("    Ou utilisez le mode direct : python bench/bench_json_stability.py --direct")
        sys.exit(1)

    run_stability_bench(args.url, args.runs)


if __name__ == "__main__":
    main()
