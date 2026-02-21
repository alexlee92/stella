import difflib
import os
import re
from typing import Dict

from agent.config import PROJECT_ROOT, TOP_K_RESULTS
from agent.llm_interface import ask_llm
from agent.memory import build_memory, budget_context
from agent.patcher import apply_patch_interactive, apply_patch_non_interactive
from agent.project_scan import load_file_content
from agent.risk import compute_patch_risk


def _strip_code_fences(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""

    # If response is a pure fenced block
    if raw.startswith("```") and raw.endswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()

    # If response contains prose + fenced code, extract first fenced block
    match = re.search(
        r"```(?:python)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip()

    return raw


def _to_abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def index_project(project_root=PROJECT_ROOT, force_rebuild: bool = False):
    build_memory(project_root, force_rebuild=force_rebuild)


def _resolve_file_fuzzy(path: str) -> str | None:
    """Essaie de résoudre un chemin de fichier, en testant des variantes (pluriel, etc.)."""
    candidates = [path]
    # Essai avec s final sur le premier segment (user → users)
    parts = path.replace("\\", "/").split("/")
    if parts:
        candidates.append("/".join([parts[0] + "s"] + parts[1:]))
        candidates.append("/".join([parts[0].rstrip("s")] + parts[1:]))
    for candidate in candidates:
        abs_path = _to_abs(candidate)
        if os.path.isfile(abs_path):
            return abs_path
    return None


def _extract_file_refs(question: str) -> list[str]:
    """Extrait les chemins de fichiers mentionnés dans une question."""
    pattern = (
        r"([A-Za-z0-9_./\\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|yaml|yml|md|toml|sql))"
    )
    return list(dict.fromkeys(re.findall(pattern, question)))


def ask_project(question: str, k: int = TOP_K_RESULTS) -> str:
    context = budget_context(question, k=max(4, k))

    # Lire directement les fichiers mentionnés dans la question
    file_sections: list[str] = []
    for ref in _extract_file_refs(question):
        abs_path = _resolve_file_fuzzy(ref)
        if abs_path:
            try:
                content = load_file_content(abs_path)
                rel = os.path.relpath(abs_path, PROJECT_ROOT)
                # Ajouter les numéros de ligne pour que le LLM ne les invente pas
                numbered = "\n".join(
                    f"{i+1:4d} | {line}" for i, line in enumerate(content.splitlines())
                )
                file_sections.append(f"=== {rel} ===\n{numbered}")
            except OSError:
                pass

    file_context = "\n\n".join(file_sections)

    prompt = f"""Tu es un assistant de développement senior. Réponds en prose claire, PAS en JSON.

Question: {question}
{f"""
Voici le contenu exact du fichier mentionné — analyse-le ligne par ligne:
{file_context}

Instructions:
- Liste uniquement les vraies erreurs (NameError, ImportError, logique incorrecte).
- Pour chaque erreur: numéro de ligne, description courte, correction proposée en code.
- Si le fichier est correct, dis-le explicitement.
- Ne réinvente pas des erreurs qui n'existent pas.
""" if file_context else f"""
Contexte du projet:
{context}
"""}"""
    return ask_llm(prompt, task_type="analysis").strip()


def propose_file_update(
    file_path: str, instruction: str, k: int = TOP_K_RESULTS
) -> str:
    abs_path = _to_abs(file_path)
    current_code = load_file_content(abs_path)
    context = budget_context(instruction, k=max(5, k))

    prompt = f"""
You are a coding agent.
Task: modify the target file according to the instruction.

You can return either:
1. The FULL content of the file (for new files or small files).
2. One or more SEARCH/REPLACE blocks for surgical edits (preferred for large files to preserve formatting).

Format for SEARCH/REPLACE blocks:
<<<<<<< SEARCH
[exact code to find]
=======
[code to replace with]
>>>>>>> REPLACE

Instruction:
{instruction}

Target file:
{file_path}

Current content:
{current_code}

Related project context:
{context}
"""

    suggestion = ask_llm(prompt, task_type="optimization")
    return _strip_code_fences(suggestion)


def propose_multi_file_update(
    instruction_by_file: Dict[str, str], k: int = TOP_K_RESULTS
) -> Dict[str, str]:
    result = {}
    for path, instruction in instruction_by_file.items():
        result[path] = propose_file_update(path, instruction, k=k)
    return result


def review_file_update(file_path: str, suggestion: str) -> str:
    abs_path = _to_abs(file_path)
    old = load_file_content(abs_path)
    diff = difflib.unified_diff(
        old.splitlines(),
        suggestion.splitlines(),
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}",
        lineterm="",
    )
    return "\n".join(diff)


def patch_risk(file_path: str, suggestion: str):
    abs_path = _to_abs(file_path)
    old = load_file_content(abs_path)
    return compute_patch_risk(file_path=file_path, old_code=old, new_code=suggestion)


def apply_suggestion(file_path: str, suggested_code: str, interactive: bool = True):
    abs_path = _to_abs(file_path)
    if interactive:
        return apply_patch_interactive(abs_path, suggested_code)
    return apply_patch_non_interactive(abs_path, suggested_code)
