import os
from typing import List

from agent.config import PROJECT_ROOT


def _walk_test_files() -> List[str]:
    roots = [os.path.join(PROJECT_ROOT, "tests"), PROJECT_ROOT]
    out = []
    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.startswith("test_") and name.endswith(".py"):
                    out.append(os.path.join(dirpath, name))
                elif name.endswith("_test.py"):
                    out.append(os.path.join(dirpath, name))
    dedup = sorted(set(out))
    return dedup


def suggest_test_path(changed_file: str) -> str:
    rel = changed_file.replace("\\", "/")
    base = os.path.basename(rel)
    stem = os.path.splitext(base)[0]

    if rel.startswith("agent/"):
        return f"tests/agent/test_{stem}.py"
    if rel.startswith("tests/"):
        return rel
    return f"tests/test_{stem}.py"


def select_test_files(changed_files: List[str]) -> List[str]:
    tests = _walk_test_files()
    if not tests:
        return []

    selected = []
    lowered_tests = {t: t.lower() for t in tests}

    for changed in changed_files:
        base = os.path.basename(changed)
        stem = os.path.splitext(base)[0].lower()
        candidates = {f"test_{stem}.py", f"{stem}_test.py"}
        suggested = suggest_test_path(changed).replace("\\", "/").lower()

        for path, low in lowered_tests.items():
            filename = os.path.basename(low)
            if low.endswith(suggested) or filename in candidates or stem in filename:
                selected.append(path)

    return sorted(set(selected))


def build_targeted_pytest_command(
    changed_files: List[str], default_command: str = "pytest -q", limit: int = 8
) -> str:
    targets = select_test_files(changed_files)
    if not targets:
        return default_command

    rel = [os.path.relpath(p, PROJECT_ROOT) for p in targets[:limit]]
    args = " ".join(rel)
    return f"pytest -q {args}".strip()
