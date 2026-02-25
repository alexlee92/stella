import json
import os
import platform
import shutil
from datetime import datetime, UTC

import requests

from agent.config import (
    EMBED_MODEL,
    MEMORY_INDEX_DIR,
    MODEL,
    OLLAMA_BASE_URL,
    PROJECT_ROOT,
)
from agent.git_tools import is_git_repo


def run_doctor() -> dict:
    checks = []

    def add(name: str, ok: bool, details: str):
        checks.append({"name": name, "ok": ok, "details": details})

    add("python", True, f"{platform.python_version()}")

    ollama_ok = False
    models_ok = False
    tags = []
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=8)
        if r.status_code == 200:
            ollama_ok = True
            data = r.json()
            tags = [m.get("name", "") for m in data.get("models", [])]
            models_ok = any(MODEL in t for t in tags) and any(
                EMBED_MODEL in t for t in tags
            )
    except Exception as exc:
        add("ollama_api", False, str(exc))

    if ollama_ok:
        add("ollama_api", True, f"reachable: {OLLAMA_BASE_URL}")
        add(
            "ollama_models",
            models_ok,
            f"required=({MODEL}, {EMBED_MODEL}), found={len(tags)}",
        )

    idx_dir = os.path.join(PROJECT_ROOT, MEMORY_INDEX_DIR)
    docs = os.path.join(idx_dir, "docs.json")
    vec = os.path.join(idx_dir, "vectors.npy")
    add(
        "memory_index",
        os.path.exists(docs) and os.path.exists(vec),
        f"index_dir={idx_dir}",
    )

    add("git_repo", is_git_repo(), f"root={PROJECT_ROOT}")

    for cmd in ["pytest", "ruff", "black"]:
        add(f"tool_{cmd}", shutil.which(cmd) is not None, f"path={shutil.which(cmd)}")

    import importlib.util

    ddgs_installed = importlib.util.find_spec("duckduckgo_search") is not None
    add(
        "lib_ddgs",
        ddgs_installed,
        (
            "duckduckgo-search installed"
            if ddgs_installed
            else "missing duckduckgo-search"
        ),
    )

    ok_count = sum(1 for c in checks if c["ok"])
    result = {
        "timestamp": datetime.now(UTC).isoformat(),
        "total": len(checks),
        "ok": ok_count,
        "failed": len(checks) - ok_count,
        "checks": checks,
    }

    out_dir = os.path.join(PROJECT_ROOT, ".stella")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "doctor_last.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def format_doctor(result: dict) -> str:
    lines = [f"Doctor: {result['ok']}/{result['total']} checks passed"]
    for c in result["checks"]:
        status = "OK" if c["ok"] else "FAIL"
        lines.append(f"- [{status}] {c['name']}: {c['details']}")
    return "\n".join(lines)
