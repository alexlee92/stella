"""
Benchmark progressif des reponses de Stella.

5 niveaux de difficulte :
  1. Questions factuelles simples (lecture de code)
  2. Questions de comprehension (logique, patterns)
  3. Detection de bugs / problemes
  4. Propositions de refactoring
  5. Generation de code complexe (multi-fichiers)

Usage:
  python bench/bench_stella_qa.py [--level N] [--verbose]
"""

import json
import os
import sys
import time

# Ajouter le projet au path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.agent import index_project
from agent.llm_interface import ask_llm
from agent.memory import search_memory

# ---------------------------------------------------------------------------
# Test cases par niveau
# ---------------------------------------------------------------------------

TESTS = [
    # --- NIVEAU 1 : Questions factuelles simples ---
    {
        "level": 1,
        "name": "Lister les endpoints",
        "question": "Quels sont les endpoints HTTP definis dans users/api.py ? Liste-les avec leurs methodes HTTP.",
        "expected_keywords": ["POST", "GET", "PUT", "DELETE", "/users", "login", "logout"],
        "min_keywords": 4,
    },
    {
        "level": 1,
        "name": "Identifier le modele ORM",
        "question": "Quelle est la structure du modele User dans users/models.py ? Quels champs a-t-il ?",
        "expected_keywords": ["id", "email", "hashed_password", "created_at", "is_active", "Column"],
        "min_keywords": 4,
    },
    {
        "level": 1,
        "name": "Trouver un import",
        "question": "Quelles bibliotheques sont importees dans users/api.py ?",
        "expected_keywords": ["flask", "sqlalchemy", "werkzeug", "Blueprint", "jsonify"],
        "min_keywords": 3,
    },
    # --- NIVEAU 2 : Comprehension de la logique ---
    {
        "level": 2,
        "name": "Comprendre le flow d'authentification",
        "question": "Comment fonctionne le processus de login dans users/api.py ? Decris les etapes.",
        "expected_keywords": ["email", "password", "check_password_hash", "401", "credentials"],
        "min_keywords": 3,
    },
    {
        "level": 2,
        "name": "Comprendre le pattern de session DB",
        "question": "Comment les sessions SQLAlchemy sont-elles gerees dans users/api.py ? Y a-t-il un probleme avec cette approche ?",
        "expected_keywords": ["Session", "sessionmaker", "commit", "rollback"],
        "min_keywords": 3,
    },
    {
        "level": 2,
        "name": "Expliquer to_dict",
        "question": "A quoi sert la methode to_dict() du modele User et pourquoi le mot de passe n'y est pas inclus ?",
        "expected_keywords": ["to_dict", "password", "securite", "json"],
        "min_keywords": 2,
    },
    # --- NIVEAU 3 : Detection de bugs et problemes ---
    {
        "level": 3,
        "name": "Trouver les failles de securite",
        "question": "Quels problemes de securite vois-tu dans users/api.py ? Liste tous les risques.",
        "expected_keywords": ["session", "injection", "validation", "token", "rate"],
        "min_keywords": 2,
    },
    {
        "level": 3,
        "name": "Detecter les fuites de sessions DB",
        "question": "Y a-t-il un probleme de fuite de sessions de base de donnees dans users/api.py ? Si oui, lesquelles ?",
        "expected_keywords": ["session", "close", "fuite", "leak"],
        "min_keywords": 2,
    },
    {
        "level": 3,
        "name": "Identifier la deprecation",
        "question": "Y a-t-il du code deprecie (deprecated) dans users/models.py ? Si oui, lequel ?",
        "expected_keywords": ["utcnow", "deprec", "datetime"],
        "min_keywords": 2,
    },
    # --- NIVEAU 4 : Propositions de refactoring ---
    {
        "level": 4,
        "name": "Refactoring des routes",
        "question": "Comment refactorer users/api.py pour suivre les bonnes pratiques Flask ? Propose des ameliorations concretes.",
        "expected_keywords": ["context", "session", "close", "error", "validation"],
        "min_keywords": 2,
    },
    {
        "level": 4,
        "name": "Architecture REST",
        "question": "L'API users/api.py respecte-t-elle les conventions REST ? Quelles ameliorations proposes-tu ?",
        "expected_keywords": ["REST", "status", "204", "pagination", "response"],
        "min_keywords": 2,
    },
    # --- NIVEAU 5 : Generation de code complexe ---
    {
        "level": 5,
        "name": "Generer un middleware d'auth",
        "question": "Ecris un decorateur Flask pour proteger les routes avec un JWT token. Il doit verifier le header Authorization et extraire l'utilisateur.",
        "expected_keywords": ["def ", "decorator", "token", "Authorization", "Bearer", "return"],
        "min_keywords": 3,
    },
    {
        "level": 5,
        "name": "Generer des tests unitaires",
        "question": "Ecris des tests pytest pour l'endpoint POST /users de users/api.py. Utilise un client de test Flask et mocke la base de donnees.",
        "expected_keywords": ["def test_", "pytest", "client", "post", "assert", "json"],
        "min_keywords": 3,
    },
]


def _read_explicit_files(question: str) -> str:
    """Read files explicitly mentioned in the question."""
    import re as _re
    from agent.project_scan import load_file_content
    from agent.config import PROJECT_ROOT as _ROOT

    pattern = r"([A-Za-z0-9_./\\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|yaml|yml|md|toml|sql|xml))"
    refs = list(dict.fromkeys(_re.findall(pattern, question)))
    sections = []
    for ref in refs[:3]:
        candidates = [ref]
        parts = ref.replace("\\", "/").split("/")
        if parts:
            candidates.append("/".join([parts[0] + "s"] + parts[1:]))
            candidates.append("/".join([parts[0].rstrip("s")] + parts[1:]))
        for c in candidates:
            abs_path = os.path.join(_ROOT, c)
            if os.path.isfile(abs_path):
                try:
                    content = load_file_content(abs_path)
                    rel = os.path.relpath(abs_path, _ROOT)
                    numbered = "\n".join(
                        f"{i+1:4d} | {line}" for i, line in enumerate(content.splitlines())
                    )
                    sections.append(f"=== {rel} (full source) ===\n{numbered}")
                except OSError:
                    pass
                break
    return "\n\n".join(sections)


def run_single_test(test: dict, verbose: bool = False) -> dict:
    """Execute un test et analyse la reponse."""
    question = test["question"]
    t0 = time.time()

    # Lire directement les fichiers mentionnes dans la question
    file_context = _read_explicit_files(question)

    # Chercher le contexte pertinent via memoire vectorielle
    docs = search_memory(question, k=5)
    if not docs:
        context = "No indexed context"
    else:
        chunks = []
        for path, content in docs:
            rel = os.path.relpath(path, os.getcwd()) if os.path.isabs(path) else path
            chunks.append(f"FILE: {rel}\n{content[:1200]}")
        context = "\n\n".join(chunks)

    if file_context:
        prompt = f"""You are a senior coding assistant. Answer in clear prose, NOT in JSON.

Question: {question}

Here is the exact source code of the mentioned file(s) — analyze it carefully:
{file_context}

Instructions:
- Base your answer ONLY on the actual code shown above.
- Reference specific line numbers, function names, and variable names.
- Do NOT invent or hallucinate code that is not present.
- If asked about imports, list the exact import statements from the file.
- If asked about structure, describe the actual fields/methods/classes present.
- If asked about bugs/issues, reference the specific lines.

Additional project context:
{context}

Answer in detail:"""
    else:
        prompt = f"""You are a senior coding assistant. Answer in clear prose, NOT in JSON.

Question: {question}

Project context:
{context}

Instructions:
- Base your answer on the provided code context.
- Reference specific file names and function names.
- Do NOT invent code that is not shown in the context.

Answer in detail:"""

    try:
        answer = ask_llm(prompt, task_type="analysis")
    except Exception as e:
        answer = f"[ERROR] {e}"

    elapsed = round(time.time() - t0, 1)

    # Analyse de la reponse
    answer_lower = answer.lower()
    found_keywords = []
    missing_keywords = []
    for kw in test["expected_keywords"]:
        if kw.lower() in answer_lower:
            found_keywords.append(kw)
        else:
            missing_keywords.append(kw)

    score = len(found_keywords)
    min_required = test["min_keywords"]
    passed = score >= min_required
    quality = round(score / len(test["expected_keywords"]) * 100)

    result = {
        "name": test["name"],
        "level": test["level"],
        "passed": passed,
        "score": f"{score}/{len(test['expected_keywords'])}",
        "quality": f"{quality}%",
        "elapsed": f"{elapsed}s",
        "found": found_keywords,
        "missing": missing_keywords,
        "answer_length": len(answer),
    }

    if verbose:
        print(f"\n{'='*70}")
        print(f"[L{test['level']}] {test['name']}")
        print(f"Question: {question[:100]}...")
        print(f"Reponse ({len(answer)} chars, {elapsed}s):")
        print(f"  {answer[:500]}...")
        print(f"Keywords trouves: {found_keywords}")
        print(f"Keywords manquants: {missing_keywords}")
        status = "[OK]" if passed else "[KO]"
        print(f"Resultat: {status} {score}/{len(test['expected_keywords'])} (min: {min_required})")

    return result


def run_benchmark(max_level: int = 5, verbose: bool = False):
    """Lance tous les tests jusqu'au niveau specifie."""
    print("="*70)
    print("BENCHMARK STELLA QA — Tests progressifs")
    print("="*70)
    print(f"Niveaux: 1-{max_level}")
    print(f"Tests: {sum(1 for t in TESTS if t['level'] <= max_level)}")
    print()

    # Indexer le projet d'abord
    print("[...] Indexation du projet...")
    index_project()
    print("[OK] Projet indexe.\n")

    results_by_level = {}
    all_results = []

    for test in TESTS:
        if test["level"] > max_level:
            continue

        level = test["level"]
        level_names = {
            1: "Factuel simple",
            2: "Comprehension",
            3: "Detection bugs",
            4: "Refactoring",
            5: "Generation code",
        }

        if level not in results_by_level:
            results_by_level[level] = {"passed": 0, "total": 0, "label": level_names.get(level, f"Niveau {level}")}
            print(f"\n--- Niveau {level}: {level_names.get(level, '?')} ---")

        status_char = "." if not verbose else ""
        result = run_single_test(test, verbose=verbose)
        all_results.append(result)
        results_by_level[level]["total"] += 1

        if result["passed"]:
            results_by_level[level]["passed"] += 1
            if not verbose:
                print(f"  [OK] {test['name']} ({result['score']}, {result['elapsed']})")
        else:
            if not verbose:
                print(f"  [KO] {test['name']} ({result['score']}, {result['elapsed']}) — manquants: {result['missing']}")

    # Resume
    print(f"\n{'='*70}")
    print("RESUME")
    print(f"{'='*70}")

    total_passed = 0
    total_tests = 0
    for level in sorted(results_by_level.keys()):
        info = results_by_level[level]
        total_passed += info["passed"]
        total_tests += info["total"]
        pct = round(info["passed"] / info["total"] * 100) if info["total"] else 0
        bar = "#" * info["passed"] + "." * (info["total"] - info["passed"])
        print(f"  Niveau {level} ({info['label']:20s}): [{bar}] {info['passed']}/{info['total']} ({pct}%)")

    overall_pct = round(total_passed / total_tests * 100) if total_tests else 0
    print(f"\n  TOTAL: {total_passed}/{total_tests} ({overall_pct}%)")

    # Latences
    latencies = [float(r["elapsed"].rstrip("s")) for r in all_results]
    avg_latency = round(sum(latencies) / len(latencies), 1) if latencies else 0
    max_latency = round(max(latencies), 1) if latencies else 0
    print(f"  Latence moyenne: {avg_latency}s | max: {max_latency}s")
    print(f"  Longueur moyenne reponse: {round(sum(r['answer_length'] for r in all_results) / len(all_results))} chars")

    # Verdict
    print()
    if overall_pct >= 80:
        print("  VERDICT: Stella repond correctement a la majorite des questions.")
    elif overall_pct >= 50:
        print("  VERDICT: Stella a des lacunes sur certaines categories.")
    else:
        print("  VERDICT: Stella a besoin d'ameliorations significatives.")

    return all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=5, help="Niveau max (1-5)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Afficher les reponses completes")
    args = parser.parse_args()

    run_benchmark(max_level=args.level, verbose=args.verbose)
