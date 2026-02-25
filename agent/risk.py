import difflib
from typing import Dict

SENSITIVE_PATTERNS = [
    "auth",
    "security",
    "config",
    "patcher",
    "auto_agent",
    "stella.py",
]


def _changed_line_count(old_code: str, new_code: str) -> int:
    diff = difflib.unified_diff(
        old_code.splitlines(), new_code.splitlines(), lineterm=""
    )
    changed = 0
    for line in diff:
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+") or line.startswith("-"):
            changed += 1
    return changed


def compute_patch_risk(
    file_path: str, old_code: str, new_code: str
) -> Dict[str, object]:
    changed = _changed_line_count(old_code, new_code)
    score = min(100, changed)

    lowered = file_path.lower()
    sensitive_hits = [p for p in SENSITIVE_PATTERNS if p in lowered]
    if sensitive_hits:
        score = min(100, score + 25)

    if "import " in new_code and "import " not in old_code:
        score = min(100, score + 8)

    if "subprocess" in new_code and "subprocess" not in old_code:
        score = min(100, score + 12)

    if score < 20:
        level = "low"
    elif score < 50:
        level = "medium"
    else:
        level = "high"

    return {
        "score": score,
        "level": level,
        "changed_lines": changed,
        "sensitive_hits": sensitive_hits,
    }
