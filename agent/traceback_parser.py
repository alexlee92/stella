"""
agent/traceback_parser.py — Parse les tracebacks pytest/Python en objets structurés.

Transforme la sortie texte de pytest en une liste de dicts avec:
  - file: chemin relatif du fichier
  - line: numéro de ligne (int)
  - error_type: classe d'erreur (AssertionError, TypeError, etc.)
  - message: message d'erreur concis
  - context: lignes de code autour de l'erreur
"""

import re
from typing import Any

_TRACEBACK_HEADER = re.compile(r"^_{3,}\s*(.+?)\s*_{3,}$")
_FILE_LINE = re.compile(r'^\s*File "([^"]+)", line (\d+), in (.+)$')
_PYTEST_FAILED = re.compile(r"^FAILED (.+?)(?:::\w+)* - (.+)$")
_PYTEST_ERROR_LINE = re.compile(r"^E\s+(.+)$")
_SHORT_TEST_SUMMARY = re.compile(r"^(?:FAILED|ERROR)\s+(.+?)(?:\s+-\s+(.+))?$")


def parse_pytest_output(output: str) -> dict[str, Any]:
    """Parse la sortie complète de pytest et retourne un résumé structuré."""
    lines = (output or "").splitlines()
    failures: list[dict] = []
    errors: list[dict] = []
    passed = 0
    failed = 0
    total = 0

    # Compter les résultats globaux
    for line in lines:
        m = re.search(r"(\d+) passed", line)
        if m:
            passed = int(m.group(1))
        m = re.search(r"(\d+) failed", line)
        if m:
            failed = int(m.group(1))
        m = re.search(r"(\d+) error", line)
        if m:
            int(m.group(1))  # errors_count — compté mais non utilisé actuellement
    total = passed + failed

    # Parser les blocs d'échec
    current_failure: dict | None = None
    current_error_lines: list[str] = []
    in_short_summary = False

    for line in lines:
        # Début section FAILURES ou ERRORS
        if re.match(r"^=+ FAILURES? =+$", line) or re.match(r"^=+ ERRORS? =+$", line):
            in_short_summary = False
            continue

        # Début section short test summary
        if re.match(r"^=+ short test summary", line):
            in_short_summary = True
            continue

        # Séparateur de bloc d'échec
        m = _TRACEBACK_HEADER.match(line)
        if m and not in_short_summary:
            if current_failure and current_error_lines:
                current_failure["error_lines"] = current_error_lines[-3:]
            current_failure = {
                "test": m.group(1),
                "frames": [],
                "error_type": "",
                "message": "",
                "error_lines": [],
            }
            current_error_lines = []
            failures.append(current_failure)
            continue

        if in_short_summary:
            m = _SHORT_TEST_SUMMARY.match(line)
            if m:
                test_id = m.group(1).strip()
                msg = (m.group(2) or "").strip()
                errors.append({"test": test_id, "message": msg})
            continue

        if current_failure is None:
            continue

        # Frame de traceback
        m = _FILE_LINE.match(line)
        if m:
            current_failure["frames"].append(
                {"file": m.group(1), "line": int(m.group(2)), "func": m.group(3)}
            )
            continue

        # Ligne d'erreur (préfixe E )
        m = _PYTEST_ERROR_LINE.match(line)
        if m:
            current_error_lines.append(m.group(1))
            if not current_failure["error_type"]:
                err_text = m.group(1)
                colon_pos = err_text.find(":")
                if colon_pos > 0:
                    current_failure["error_type"] = err_text[:colon_pos].strip()
                    current_failure["message"] = err_text[colon_pos + 1 :].strip()[:200]
                else:
                    current_failure["message"] = err_text[:200]

    if current_failure and current_error_lines:
        current_failure["error_lines"] = current_error_lines[-3:]

    # Extraire les infos de localisation depuis le dernier frame
    structured = []
    for f in failures:
        frames = f.get("frames", [])
        last_frame = frames[-1] if frames else {}
        structured.append(
            {
                "test": f.get("test", ""),
                "file": last_frame.get("file", ""),
                "line": last_frame.get("line", 0),
                "func": last_frame.get("func", ""),
                "error_type": f.get("error_type", ""),
                "message": f.get("message", ""),
                "context": f.get("error_lines", []),
            }
        )

    return {
        "passed": passed,
        "failed": failed,
        "total": total,
        "failures": structured,
        "short_summary": errors,
        "has_failures": len(structured) > 0,
    }


def format_failures_for_llm(parsed: dict[str, Any], max_failures: int = 3) -> str:
    """Formate les échecs parsés pour inclusion dans un prompt LLM."""
    if not parsed.get("has_failures"):
        return "No test failures found."

    lines = [f"Test results: {parsed['passed']} passed, {parsed['failed']} failed\n"]
    for i, f in enumerate(parsed["failures"][:max_failures]):
        lines.append(f"Failure {i + 1}: {f['test']}")
        if f["file"]:
            lines.append(f"  File: {f['file']}, line {f['line']}")
        if f["error_type"]:
            lines.append(f"  Error: {f['error_type']}: {f['message']}")
        elif f["message"]:
            lines.append(f"  Message: {f['message']}")
        if f["context"]:
            lines.append("  Context: " + " | ".join(f["context"]))
        lines.append("")

    return "\n".join(lines)
