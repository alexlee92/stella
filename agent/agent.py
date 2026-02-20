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


def ask_project(question: str, k: int = TOP_K_RESULTS) -> str:
    context = budget_context(question, k=max(4, k))

    prompt = f"""
You are a senior coding assistant.
Answer using the project context only when possible.
If uncertain, say what is missing.

Question:
{question}

Project context:
{context}
"""
    return ask_llm(prompt).strip()


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

    suggestion = ask_llm(prompt)
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
