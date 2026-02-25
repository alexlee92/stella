import sys

sys.path.insert(0, ".")

from agent.tooling import list_files, search_code
from agent.scaffolder import list_templates
from agent.quality import scan_secrets_in_files
from agent.llm_interface import _detect_task_type, _parse_json_response
from agent.memory import _chunk_text

# Test 1: task type detection
tests_types = [
    ("Cree un module de facturation ERP", "planning"),
    ("Corrige le bug dans users/api.py", "debug"),
    ("Analyse l architecture du projet", "analysis"),
    ("Optimise les requetes SQL", "optimization"),
    ("Cree un composant React pour le dashboard", "frontend"),
    ("Genere une API FastAPI pour les commandes", "backend"),
]
print("=== Test routing types ===")
ok = 0
for prompt, expected in tests_types:
    detected = _detect_task_type(prompt)
    status = "OK" if detected == expected else "FAIL (got: " + detected + ")"
    print("  [" + status + "] " + prompt[:40] + " -> " + detected)
    if detected == expected:
        ok += 1
print("  Score: " + str(ok) + "/" + str(len(tests_types)))

# Test 2: JSON parsing
print("\n=== Test JSON parsing ===")
json_tests = [
    ('{"action": "finish", "reason": "done", "args": {"summary": "ok"}}', True),
    ("not json at all", False),
]
ok = 0
for text, should_parse in json_tests:
    parsed, err = _parse_json_response(text)
    success = (parsed is not None) == should_parse
    print(
        "  [" + ("OK" if success else "FAIL") + "] -> " + ("parsed" if parsed else err)
    )
    if success:
        ok += 1
print("  Score: " + str(ok) + "/" + str(len(json_tests)))

# Test 3: templates
print("\n=== Templates disponibles ===")
templates = list_templates()
for name, desc in templates.items():
    print("  - " + name + ": " + desc)

# Test 4: secrets detection
print("\n=== Test detection secrets ===")
findings = scan_secrets_in_files(["agent/auth.py"])
print("  Secrets detectes dans auth.py: " + str(len(findings)))
for f in findings:
    print("  - ligne " + str(f["line"]) + ": " + f["type"] + " | " + f["excerpt"])

# Test 5: memory chunking
print("\n=== Test chunking memoire ===")
text = "def hello(): pass\n" * 100
chunks = _chunk_text(text)
print("  Texte de " + str(len(text)) + " chars -> " + str(len(chunks)) + " chunks")

# Test 6: list files
print("\n=== Test list_files ===")
files = list_files(limit=10, ext=".py")
print("  Fichiers Python trouves: " + str(len(files)))
for f in files[:5]:
    print("  - " + f)

# Test 7: search code
print("\n=== Test search_code ===")
results = search_code("def ask_llm", limit=5)
print('  Resultats pour "def ask_llm": ' + str(len(results)))
for r in results:
    print("  - " + r[:80])
