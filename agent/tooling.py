import os
import re
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from agent.config import DRY_RUN, PROJECT_ROOT
from agent.project_scan import get_python_files, get_source_files, load_file_content

# P3.3 — TTL cache for read_file and list_files (30s TTL)
_TOOL_CACHE_TTL = 30.0
_read_file_cache: Dict[str, Tuple[float, str]] = {}
_list_files_cache: Dict[str, Tuple[float, list]] = {}


def _ttl_get(cache: dict, key: str):
    entry = cache.get(key)
    if entry and time.time() - entry[0] < _TOOL_CACHE_TTL:
        return entry[1]
    if entry:
        del cache[key]
    return None


def _ttl_set(cache: dict, key: str, val):
    cache[key] = (time.time(), val)


def invalidate_tool_cache():
    """Vide les caches TTL (à appeler après écriture de fichier)."""
    _read_file_cache.clear()
    _list_files_cache.clear()

_ALLOWED_COMMAND_PREFIXES = [
    ["pytest"],
    ["python", "-m", "pytest"],
    ["python", "-m", "ruff"],
    ["python", "-m", "black"],
    ["python", "-m", "mypy"],
    ["python", "-m", "bandit"],
    ["python", "-m", "pip", "install"],
    ["python", "-m", "pip", "list"],
    ["pip", "install"],
    ["pip", "list"],
    ["npm", "install"],
    ["npm", "test"],
    ["npm", "run"],
    ["node"],
    ["python"],
    ["git", "status"],
    ["git", "log"],
    ["git", "diff"],
    ["git", "blame"],
    ["git", "stash"],
]

# Commandes dangereuses jamais autorisees
_BLOCKED_PATTERNS = [
    "rm -rf", "rmdir /s", "del /f", "format ",
    "drop database", "drop table", "truncate table",
    "> /dev/null", ":(){ :|:& };:",
    "mkfs", "dd if=",
]


def _resolve_path(path: str) -> str:
    if os.path.isabs(path):
        abs_path = os.path.abspath(path)
    else:
        abs_path = os.path.abspath(os.path.join(PROJECT_ROOT, path))

    root = os.path.abspath(PROJECT_ROOT)
    if not abs_path.startswith(root):
        raise ValueError("Path is outside project root")

    return abs_path


def is_command_blocked(command: str) -> bool:
    """Verifie si une commande est dans la blocklist de securite.

    P5.5 — En production, bloque aussi les commandes destructives SQL.
    """
    low = command.lower()
    if any(pat in low for pat in _BLOCKED_PATTERNS):
        return True
    # P5.5 — Extra safety in production/staging environments
    try:
        from agent.config import _CFG
        env = _CFG.get("STELLA_ENV", "development")
    except Exception:
        env = "development"
    if env in ("production", "staging"):
        prod_blocked = ["drop ", "truncate ", "delete from", "alter table", "migrate"]
        if any(pat in low for pat in prod_blocked):
            return True
    return False


def is_command_allowed(command: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    for prefix in _ALLOWED_COMMAND_PREFIXES:
        if tokens[: len(prefix)] == prefix:
            return True
    return False


def run_safe_command(command: str, timeout: int = 300, ask_user: bool = False) -> Tuple[int, str]:
    # P3.3 — Blocklist de securite
    if is_command_blocked(command):
        return 2, f"[BLOCKED] Commande dangereuse refusee : {command}"

    if not is_command_allowed(command):
        if ask_user:
            # P3.3 — Mode sandbox : demande confirmation pour les commandes inconnues
            try:
                answer = input(
                    f"\n  [?] Commande non-whitelistee : {command}\n"
                    f"      Autoriser l'execution ? [y/n] "
                ).strip().lower()
                if answer not in {"y", "yes", "o", "oui"}:
                    return 2, f"Refused by user: {command}"
            except (EOFError, KeyboardInterrupt):
                return 2, f"Refused (interrupted): {command}"
        else:
            return 2, f"Refused command: {command}"

    if DRY_RUN and "pytest" not in command:
        return 0, f"[dry-run] skipped command: {command}"

    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=True,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return 2, f"Command timed out after {timeout}s: {command}"
    except Exception as exc:
        return 2, f"Command failed to run: {exc}"

    output = (result.stdout or "") + ("\n" + result.stderr if result.stderr else "")
    return result.returncode, (output.strip() or "(no output)")[:8000]


def read_file(path: str, max_chars: int = 6000) -> str:
    abs_path = _resolve_path(path)
    # P3.3 — TTL cache
    cache_key = f"{abs_path}:{max_chars}"
    cached = _ttl_get(_read_file_cache, cache_key)
    if cached is not None:
        return cached
    content = load_file_content(abs_path)
    result = content if len(content) <= max_chars else content[:max_chars] + "\n\n...[truncated]"
    _ttl_set(_read_file_cache, cache_key, result)
    return result


def read_many(
    paths: List[str], max_chars_per_file: int = 2500, max_total_chars: int = 9000
) -> str:
    # P3.2 — parallel reads via ThreadPoolExecutor
    def _read_one(path: str) -> Tuple[str, str]:
        try:
            content = read_file(path, max_chars=max_chars_per_file)
        except Exception as exc:
            content = f"[error] {exc}"
        return path, content

    results: Dict[str, str] = {}
    if paths:
        with ThreadPoolExecutor(max_workers=min(8, len(paths))) as executor:
            for path, content in executor.map(_read_one, paths):
                results[path] = content

    chunks = []
    total = 0
    for path in paths:
        content = results.get(path, "[error] not read")
        block = f"FILE: {path}\n{content}\n"
        if total + len(block) > max_total_chars:
            break
        chunks.append(block)
        total += len(block)

    return "\n".join(chunks) if chunks else "No files read"


_SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    ".md",
    ".html",
    ".css",
    ".scss",
}


def list_python_files(limit: int = 200) -> List[str]:
    files = get_python_files(PROJECT_ROOT)
    rel = [os.path.relpath(p, PROJECT_ROOT) for p in files]
    rel.sort()
    return rel[:limit]


def list_files(limit: int = 200, contains: str = "", ext: str = ".py") -> List[str]:
    # P3.3 — TTL cache
    cache_key = f"list:{limit}:{contains}:{ext}"
    cached = _ttl_get(_list_files_cache, cache_key)
    if cached is not None:
        return cached
    extensions = {ext} if ext else _SOURCE_EXTENSIONS
    files = get_source_files(PROJECT_ROOT, extensions=extensions)
    out = []
    for abs_path in sorted(files):
        rel = os.path.relpath(abs_path, PROJECT_ROOT)
        if contains and contains not in rel:
            continue
        out.append(rel)
        if len(out) >= limit:
            break
    _ttl_set(_list_files_cache, cache_key, out)
    return out


def search_code(pattern: str, limit: int = 30) -> List[str]:
    try:
        rg_cmd = [
            "rg",
            "-n",
            "--hidden",
            "--glob",
            "!**/__pycache__/**",
            pattern,
            PROJECT_ROOT,
        ]
        result = subprocess.run(
            rg_cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=10,  # P1.1 — timeout individuel
        )
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return lines[:limit]
    except Exception:
        pass

    output = []
    try:
        expr = re.compile(pattern)
    except re.error:
        expr = re.compile(re.escape(pattern))

    for file_path in get_python_files(PROJECT_ROOT):
        content = load_file_content(file_path)
        for idx, line in enumerate(content.splitlines(), start=1):
            if expr.search(line):
                rel = os.path.relpath(file_path, PROJECT_ROOT)
                output.append(f"{rel}:{idx}:{line.strip()}")
                if len(output) >= limit:
                    return output
    return output


def write_new_file(path: str, content: str) -> str:
    """Écrit un nouveau fichier sur disque, crée les dossiers parents si nécessaire.

    Retourne un message de résultat (success ou erreur).
    """
    abs_path = _resolve_path(path)
    parent = os.path.dirname(abs_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        with open(abs_path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return f"ok:{abs_path}"
    except OSError as exc:
        return f"error:{exc}"


def run_tests_detailed(
    command: str = "pytest -q", timeout: int = 180
) -> Tuple[int, str]:
    return run_safe_command(command, timeout=timeout)


def run_tests(command: str = "pytest -q", timeout: int = 180) -> str:
    code, output = run_tests_detailed(command=command, timeout=timeout)
    return f"exit_code={code}\n{output}"
