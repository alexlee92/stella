import sys

sys.path.insert(0, ".")

from agent.llm_interface import _detect_task_type
from agent.quality import scan_secrets_in_files

# Test routing LLM corrigé
tests = [
    ("Cree un module de facturation ERP", "planning"),
    ("Corrige le bug dans users/api.py", "debug"),
    ("Analyse l architecture du projet", "analysis"),
    ("Optimise les requetes SQL", "optimization"),
    ("Cree un composant React pour le dashboard", "frontend"),
    ("Genere une API FastAPI pour les commandes", "backend"),
    ("Optimise les index de la base de donnees", "optimization"),
    ("nouveau module stock pour ERP", "planning"),
]

print("=== Routing LLM apres corrections ===")
ok = 0
for prompt, expected in tests:
    detected = _detect_task_type(prompt)
    status = "OK" if detected == expected else "FAIL (got: " + detected + ")"
    print("  [" + status + "] " + prompt[:50])
    if detected == expected:
        ok += 1
print("  Score: " + str(ok) + "/" + str(len(tests)))

# Test scan secrets
print("\n=== Scan secrets apres correction auth.py ===")
# auth.py ne doit plus avoir de secrets hardcodes
findings = scan_secrets_in_files(["agent/auth.py"])
print("  Secrets dans auth.py: " + str(len(findings)))
if findings:
    for f in findings:
        print("  - " + str(f))
else:
    print("  [OK] Aucun secret hardcode detecte")
