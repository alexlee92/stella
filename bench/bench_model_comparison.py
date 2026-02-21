"""
bench_model_comparison.py — Compare Orisha-Ifa1.0 vs Orisha-Oba1.0 sur des tâches croisées.

Ce benchmark teste chaque modèle sur son domaine de prédilection ET en dehors,
pour valider que le routing est justifié.

Usage :
    python bench/bench_model_comparison.py --direct
    python bench/bench_model_comparison.py --url http://localhost:5000
"""
import argparse
import sys
import time

import requests

ORISHA_URL = "http://localhost:5000/query"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"

TASKS = {
    "analyse_architecture": {
        "prompt": (
            "Analyse cette fonction et identifie tous les problèmes potentiels :\n"
            "def process(data):\n"
            "    result = []\n"
            "    for item in data:\n"
            "        result.append(item['value'] / item['total'])\n"
            "    return result"
        ),
        "expected_keywords": ["division", "zero", "keyerror", "none", "erreur", "exception"],
        "ideal_model": "Orisha-Ifa1.0",
    },
    "generation_fonction": {
        "prompt": (
            "Génère une fonction Python `safe_divide(a, b)` qui :\n"
            "- Retourne None si b == 0\n"
            "- Lève ValueError si a ou b n'est pas un nombre\n"
            "- Retourne le résultat de a / b sinon"
        ),
        "expected_keywords": ["def safe_divide", "if b == 0", "return none", "valueerror", "isinstance"],
        "ideal_model": "Orisha-Oba1.0",
    },
    "debug_traceback": {
        "prompt": (
            "Ce code Python lève une exception :\n"
            "```python\n"
            "lst = [1, 2, 3]\n"
            "print(lst[10])\n"
            "```\n"
            "Quel est le type d'exception ? Comment corriger ?"
        ),
        "expected_keywords": ["indexerror", "index", "out of range", "10", "len"],
        "ideal_model": "Orisha-Oba1.0",
    },
    "refactoring": {
        "prompt": (
            "Refactorise ce code pour le rendre plus lisible et maintenable :\n"
            "def f(l):\n"
            "    r = []\n"
            "    for i in range(len(l)):\n"
            "        if l[i] > 0:\n"
            "            r.append(l[i] * 2)\n"
            "    return r"
        ),
        "expected_keywords": ["def", "for", "if", "append", "comprehension", "list"],
        "ideal_model": "Orisha-Ifa1.0",
    },
    "json_strict": {
        "prompt": (
            'Retourne UNIQUEMENT ce JSON, sans modification :\n'
            '{"action":"finish","reason":"tâche complète","args":{"summary":"Analyse terminée avec succès"}}'
        ),
        "expected_keywords": ['"action"', '"finish"', '"summary"'],
        "ideal_model": "Orisha-Ifa1.0",
    },
}

TASK_TYPES_BY_MODEL = [
    ("Orisha-Ifa1.0", "analysis", "Orisha-ifa1.0:latest"),
    ("Orisha-Oba1.0", "optimization", "Orisha-Oba1.0:latest"),
]


def _score(response: str, keywords: list[str]) -> tuple[int, int]:
    low = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in low)
    return hits, len(keywords)


def _query_ollama(model: str, prompt: str, timeout: int = 120) -> str:
    resp = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "")


def run_comparison(url: str | None = None):
    direct = url is None
    print(f"\n{'='*80}")
    print("Comparaison Orisha-Ifa1.0 vs Orisha-Oba1.0")
    print(f"Mode : {'direct Ollama' if direct else url}")
    print(f"{'='*80}\n")

    if not direct:
        try:
            health_url = url.replace("/query", "/health")
            requests.get(health_url, timeout=5)
        except Exception:
            print("[!] Serveur Orisha non accessible. Lancez : python orisha_server.py")
            print("    Ou utilisez le mode direct : python bench/bench_model_comparison.py --direct")
            sys.exit(1)

    for task_name, task in TASKS.items():
        print(f"\n--- Tâche : {task_name} (modèle idéal : {task['ideal_model']}) ---")
        for model_label, task_type, ollama_model in TASK_TYPES_BY_MODEL:
            t0 = time.time()
            try:
                if direct:
                    response = _query_ollama(ollama_model, task["prompt"])
                else:
                    resp = requests.post(
                        url,
                        json={"prompt": task["prompt"], "task_type": task_type},
                        timeout=200,
                    )
                    response = resp.json().get("response", "")
                elapsed = round(time.time() - t0, 1)
                hits, total = _score(response, task["expected_keywords"])
                score_pct = round((hits / total) * 100) if total else 0
                is_ideal = "[*]" if model_label == task["ideal_model"] else "  "
                status = "[OK]" if score_pct >= 60 else ("[!!]" if score_pct >= 40 else "[KO]")
                print(f"  {is_ideal} {status} {model_label:<22} | score: {hits}/{total} ({score_pct}%) | {elapsed}s")
            except Exception as e:
                print(f"     {model_label}: ERREUR — {e}")

    print(f"\n{'='*80}")
    print("Interprétation :")
    print("  [*] = modèle idéal pour cette tâche selon le routing")
    print("  [OK] = bonne réponse (>=60% de mots-clés trouvés)")
    print("  [!!]  = réponse partielle (40-59%)")
    print("  [KO] = réponse insuffisante (<40%)")
    print("\nSi le modèle idéal [*] obtient un meilleur score que l'autre,")
    print("le routing est justifié et fonctionne correctement.")


def main():
    parser = argparse.ArgumentParser(description="Comparaison Orisha-Ifa1.0 vs Orisha-Oba1.0")
    parser.add_argument("--url", default="http://localhost:5000/query")
    parser.add_argument("--direct", action="store_true",
                        help="Appeler Ollama directement sans le serveur Flask")
    args = parser.parse_args()

    if args.direct:
        run_comparison(url=None)
    else:
        run_comparison(url=args.url)


if __name__ == "__main__":
    main()
