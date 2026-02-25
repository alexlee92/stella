"""P4.1 — Merge JS/TS symbol-aware (regex-based).

Fusionne du nouveau code JS/TS dans l'existant en remplaçant
les symboles modifiés sans écraser les fonctions/classes non touchées.

Supporte : function, class, const/let/var = arrow/function expression
Extensions : .js .jsx .ts .tsx
"""

import re
from typing import Dict, List, Tuple

# Pattern pour détecter un symbole de haut niveau
_TOP_LEVEL = re.compile(
    r"^(?P<prefix>(?:export\s+(?:default\s+)?)?(?:async\s+)?)"
    r"(?:"
    r"function\s+(?P<fn>[A-Za-z_$][A-Za-z0-9_$]*)"
    r"|class\s+(?P<cls>[A-Za-z_$][A-Za-z0-9_$]*)"
    r"|(?:const|let|var)\s+(?P<var>[A-Za-z_$][A-Za-z0-9_$]*)\s*="
    r")",
    re.MULTILINE,
)


def _extract_symbols(code: str) -> Dict[str, Tuple[int, int]]:
    """Retourne {nom_symbol: (start_line_idx, end_line_idx)} (0-indexed, end exclusif)."""
    lines = code.splitlines()
    symbols: Dict[str, Tuple[int, int]] = {}
    i = 0
    while i < len(lines):
        m = _TOP_LEVEL.match(lines[i])
        if m:
            name = m.group("fn") or m.group("cls") or m.group("var")
            if name:
                start = i
                end = _find_symbol_end(lines, i)
                symbols[name] = (start, end)
                i = end
                continue
        i += 1
    return symbols


def _find_symbol_end(lines: List[str], start: int) -> int:
    """Trouve la fin d'un bloc en comptant les accolades/parenthèses."""
    depth = 0
    found_open = False
    for i in range(start, len(lines)):
        for ch in lines[i]:
            if ch in "{(":
                depth += 1
                found_open = True
            elif ch in "})":
                depth -= 1
        if found_open and depth <= 0:
            return i + 1
    return len(lines)


def _get_symbol_text(lines: List[str], start: int, end: int) -> str:
    return "\n".join(lines[start:end])


def ts_merge(old_code: str, new_code: str) -> Tuple[str, bool, str]:
    """Fusionne new_code dans old_code en remplaçant uniquement les symboles modifiés.

    Returns:
        (merged_code, was_merged, reason)
        was_merged=False si le merge n'a pas pu être appliqué (fallback = remplacer tout)
    """
    old_symbols = _extract_symbols(old_code)
    new_symbols = _extract_symbols(new_code)

    if not old_symbols and not new_symbols:
        # Fichier sans symboles reconnaissables : remplacer tout
        return new_code, False, "no_symbols_detected"

    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    # Construire la liste des remplacements : {nom: nouveau_texte}
    replacements: Dict[str, str] = {}
    additions: List[str] = []

    for name, (ns, ne) in new_symbols.items():
        new_text = _get_symbol_text(new_lines, ns, ne)
        if name in old_symbols:
            old_text = _get_symbol_text(old_lines, *old_symbols[name])
            if old_text.strip() != new_text.strip():
                replacements[name] = new_text
        else:
            additions.append(new_text)

    if not replacements and not additions:
        return old_code, True, "no_changes_detected"

    # Appliquer les remplacements dans old_code en ordre
    result_lines: List[str] = []
    i = 0
    while i < len(old_lines):
        replaced = False
        for name, (os_, oe) in old_symbols.items():
            if i == os_ and name in replacements:
                result_lines.extend(replacements[name].splitlines())
                i = oe
                replaced = True
                break
        if not replaced:
            result_lines.append(old_lines[i])
            i += 1

    # Ajouter les nouveaux symboles à la fin
    for add_text in additions:
        result_lines.append("")
        result_lines.extend(add_text.splitlines())

    merged = "\n".join(result_lines) + "\n"
    reason = f"replaced={len(replacements)},added={len(additions)}"
    return merged, True, reason


def is_js_ts_file(path: str) -> bool:
    return path.lower().endswith((".js", ".jsx", ".ts", ".tsx"))
