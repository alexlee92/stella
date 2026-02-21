import os
import re
from typing import Dict, List, Optional, Tuple

from agent.config import PROJECT_ROOT
from agent.llm_interface import ask_llm
from agent.patcher import apply_patch_non_interactive
from agent.project_scan import load_file_content
from agent.test_selector import suggest_test_path
from agent.tooling import run_tests_detailed


def _to_abs(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_ROOT, path)


def _is_python_source(path: str) -> bool:
    low = path.replace("\\", "/").lower()
    return low.endswith(".py") and "/tests/" not in f"/{low}/" and not low.startswith("tests/")


def _strip_code_fences(text: str) -> str:
    raw = (text or "").strip()
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return raw


def analyze_generated_test_code(code: str) -> dict:
    names = re.findall(r"^\s*def\s+(test_[A-Za-z0-9_]+)\s*\(", code or "", re.MULTILINE)
    edge_tokens = {
        "edge",
        "empty",
        "none",
        "null",
        "invalid",
        "error",
        "boundary",
        "zero",
    }
    has_edge = any(any(tok in n.lower() for tok in edge_tokens) for n in names)
    has_nominal = any(not any(tok in n.lower() for tok in edge_tokens) for n in names)
    return {
        "test_functions": names,
        "test_count": len(names),
        "has_nominal_case": has_nominal,
        "has_edge_case": has_edge,
        "quality_ok": len(names) >= 2 and has_nominal and has_edge,
    }


def build_test_targets(changed_files: List[str], limit: int = 3) -> List[Tuple[str, str]]:
    out = []
    seen = set()
    for path in changed_files:
        rel = path.replace("\\", "/")
        if not _is_python_source(rel):
            continue
        target = suggest_test_path(rel)
        key = (rel, target)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= limit:
            break
    return out


def _get_coverage_report(source_path: str, test_path: str) -> Optional[str]:
    """P4.3 — Exécute pytest-cov et retourne les lignes non couvertes.

    Retourne None si pytest-cov n'est pas disponible ou si aucun test n'existe.
    """
    test_abs = os.path.join(PROJECT_ROOT, test_path)
    if not os.path.exists(test_abs):
        return None

    # Convertit le chemin de fichier en module dot-notation pour --cov
    module = source_path.replace("\\", "/").replace("/", ".").removesuffix(".py")
    cmd = f"pytest {test_path} --cov={module} --cov-report=term-missing -q"
    try:
        code, output = run_tests_detailed(cmd, timeout=60)
        if "no module named" in output.lower() or "ModuleNotFoundError" in output:
            return None
        return output[:3000]
    except Exception:
        return None


def generate_tests_for_changes(changed_files: List[str], limit: int = 3) -> Dict[str, str]:
    generated: Dict[str, str] = {}
    targets = build_test_targets(changed_files, limit=limit)
    for source_path, test_path in targets:
        source_abs = _to_abs(source_path)
        source_code = load_file_content(source_abs)

        test_abs = _to_abs(test_path)
        current_test = ""
        if os.path.exists(test_abs):
            current_test = load_file_content(test_abs)

        # P4.3 — coverage-guided context
        coverage_report = _get_coverage_report(source_path, test_path)
        coverage_section = ""
        if coverage_report:
            coverage_section = f"\nCoverage report (lines not yet covered):\n{coverage_report}\n"

        prompt = f"""
You generate pytest tests for a changed Python file.
Return only complete valid Python code for the target test file.

Requirements:
- cover at least one nominal case
- cover at least one edge case
- keep tests deterministic
- avoid network/filesystem side effects when possible
{coverage_section}
Source file: {source_path}
Target test file: {test_path}

Source code:
{source_code[:6000]}

Existing test file content (may be empty):
{current_test[:5000]}
"""
        test_code = _strip_code_fences(ask_llm(prompt))
        quality = analyze_generated_test_code(test_code)
        if not quality["quality_ok"]:
            repair_prompt = f"""
Regenerate the pytest file and satisfy all constraints strictly.
Return only Python code.

Mandatory:
- at least 2 test functions
- at least 1 nominal test
- at least 1 explicit edge-case test (name should contain edge/invalid/empty/none/boundary/zero)

Source file: {source_path}
Target test file: {test_path}
"""
            test_code = _strip_code_fences(ask_llm(repair_prompt))
        if test_code.strip():
            generated[test_path] = test_code
    return generated


def apply_generated_tests(changed_files: List[str], limit: int = 3) -> dict:
    generated = generate_tests_for_changes(changed_files, limit=limit)
    applied = []
    failed = []
    quality = {}
    for test_path, code in generated.items():
        quality[test_path] = analyze_generated_test_code(code)
        abs_path = _to_abs(test_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        result = apply_patch_non_interactive(abs_path, code)
        if isinstance(result, dict) and (result.get("applied") or result.get("dry_run")):
            applied.append(test_path)
        else:
            failed.append({"path": test_path, "result": result})
    quality_ok = sum(1 for q in quality.values() if q.get("quality_ok"))
    return {
        "generated": list(generated.keys()),
        "applied": applied,
        "failed": failed,
        "quality": quality,
        "quality_ok_rate": round((quality_ok / max(1, len(quality))) * 100, 2),
    }
