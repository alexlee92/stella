"""
agent/deps.py — Détection et installation automatique des dépendances.

Analyse les imports d'un fichier généré, détecte les packages manquants
et les installe via pip (Python) ou npm (JS/TS/JSX/TSX).
"""

import importlib.util
import os
import re
import subprocess
import sys
from typing import List, Tuple

from agent.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Mapping import name → nom PyPI réel
# ---------------------------------------------------------------------------
_PYPI_ALIASES: dict[str, str] = {
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "yaml": "PyYAML",
    "bs4": "beautifulsoup4",
    "dotenv": "python-dotenv",
    "flask": "Flask",
    "sqlalchemy": "SQLAlchemy",
    "bcrypt": "bcrypt",
    "jwt": "PyJWT",
    "requests": "requests",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "celery": "celery",
    "redis": "redis",
    "psycopg2": "psycopg2-binary",
    "pymongo": "pymongo",
    "boto3": "boto3",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
    "torch": "torch",
    "transformers": "transformers",
    "aiohttp": "aiohttp",
    "httpx": "httpx",
    "starlette": "starlette",
    "alembic": "alembic",
    "werkzeug": "Werkzeug",
    "jinja2": "Jinja2",
    "click": "click",
    "typer": "typer",
    "rich": "rich",
    "tqdm": "tqdm",
    "paramiko": "paramiko",
    "cryptography": "cryptography",
    "passlib": "passlib",
    "marshmallow": "marshmallow",
    "cerberus": "cerberus",
    "peewee": "peewee",
    "tortoise": "tortoise-orm",
    "motor": "motor",
    "beanie": "beanie",
    "loguru": "loguru",
    "arrow": "arrow",
    "pendulum": "pendulum",
    "attrs": "attrs",
    "cattrs": "cattrs",
    "orjson": "orjson",
    "ujson": "ujson",
}

# Modules stdlib Python à ignorer (non installables via pip)
_STDLIB: set[str] = (
    set(sys.stdlib_module_names)
    if hasattr(sys, "stdlib_module_names")
    else {
        "os",
        "sys",
        "re",
        "json",
        "time",
        "datetime",
        "math",
        "random",
        "collections",
        "itertools",
        "functools",
        "pathlib",
        "typing",
        "abc",
        "io",
        "hashlib",
        "hmac",
        "base64",
        "struct",
        "copy",
        "threading",
        "multiprocessing",
        "subprocess",
        "shutil",
        "glob",
        "tempfile",
        "traceback",
        "logging",
        "warnings",
        "inspect",
        "ast",
        "dis",
        "importlib",
        "contextlib",
        "dataclasses",
        "enum",
        "argparse",
        "configparser",
        "csv",
        "sqlite3",
        "socket",
        "ssl",
        "http",
        "urllib",
        "email",
        "html",
        "xml",
        "unittest",
        "string",
        "textwrap",
        "difflib",
        "heapq",
        "bisect",
        "array",
        "queue",
        "weakref",
        "gc",
        "platform",
        "signal",
        "atexit",
        "site",
        "builtins",
        "types",
        "operator",
        "decimal",
        "fractions",
        "statistics",
        "cmath",
        "numbers",
        "codecs",
        "unicodedata",
        "locale",
        "gettext",
        "pprint",
        "reprlib",
        "errno",
    }
)

# Modules locaux du projet à ignorer
_PROJECT_LOCAL: set[str] = {
    "agent",
    "bench",
    "eval",
    "tests",
    "users",
    "frontend",
    "models",
    "config",
    "settings",
    "main",
    "stella",
}


# ---------------------------------------------------------------------------
# Détection — Python
# ---------------------------------------------------------------------------


def _extract_python_imports(content: str) -> List[str]:
    """Extrait les noms de modules de premier niveau depuis le code Python."""
    names: set[str] = set()
    for m in re.finditer(
        r"^\s*(?:import|from)\s+([A-Za-z_][A-Za-z0-9_]*)",
        content,
        re.MULTILINE,
    ):
        names.add(m.group(1))
    return list(names)


def _is_python_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _is_valid_package_name(name: str) -> bool:
    """Vérifie qu'un nom de package est plausible (pas un mot-clé ou nom local)."""
    if len(name) < 2:
        return False
    # Doit contenir uniquement des caractères valides pour un nom de package
    if not re.match(r"^[A-Za-z][A-Za-z0-9_\-\.]*$", name):
        return False
    # Rejeter les noms qui ressemblent à des modules locaux ou fichiers de config
    _local_hints = {
        "database",
        "config",
        "settings",
        "models",
        "utils",
        "helpers",
        "constants",
        "exceptions",
        "errors",
        "middleware",
        "routes",
        "schemas",
        "serializers",
        "validators",
        "db",
    }
    if name.lower() in _local_hints:
        return False
    return True


def _missing_python_packages(content: str) -> List[str]:
    """Retourne la liste des packages PyPI manquants pour ce fichier Python."""
    imports = _extract_python_imports(content)
    missing = []
    for mod in imports:
        if mod in _STDLIB or mod in _PROJECT_LOCAL:
            continue
        if not _is_valid_package_name(mod):
            continue
        if _is_python_installed(mod):
            continue
        # Utiliser l'alias PyPI si disponible
        pypi_name = _PYPI_ALIASES.get(mod, mod)
        if pypi_name not in missing:
            missing.append(pypi_name)
    return missing


# ---------------------------------------------------------------------------
# Détection — JavaScript / TypeScript
# ---------------------------------------------------------------------------


def _extract_js_imports(content: str) -> List[str]:
    """Extrait les noms de packages npm depuis du code JS/TS (exclut imports relatifs)."""
    names: set[str] = set()

    # import X from 'package'  /  import 'package'
    for m in re.finditer(
        r"""(?:import\s+[^'"]*from\s+|import\s+)['"]([^'"]+)['"]""",
        content,
    ):
        pkg = m.group(1)
        if pkg.startswith("."):
            continue
        # Scoped: @org/pkg → garder @org/pkg
        if pkg.startswith("@"):
            parts = pkg.split("/")
            if len(parts) >= 2:
                names.add(f"{parts[0]}/{parts[1]}")
        else:
            names.add(pkg.split("/")[0])

    # require('package')
    for m in re.finditer(
        r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""",
        content,
    ):
        pkg = m.group(1)
        if pkg.startswith("."):
            continue
        names.add(pkg.split("/")[0])

    return list(names)


def _is_npm_installed(package: str) -> bool:
    """Vérifie si un package npm est présent dans node_modules/."""
    node_modules = os.path.join(PROJECT_ROOT, "node_modules", package)
    return os.path.isdir(node_modules)


def _missing_npm_packages(content: str) -> List[str]:
    """Retourne la liste des packages npm manquants pour ce fichier JS/TS."""
    imports = _extract_js_imports(content)
    return [pkg for pkg in imports if not _is_npm_installed(pkg)]


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------


def _install_python(packages: List[str]) -> Tuple[bool, str]:
    if not packages:
        return True, "nothing to install"
    cmd = [sys.executable, "-m", "pip", "install"] + packages
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        out = (result.stdout + "\n" + result.stderr).strip()
        return result.returncode == 0, out[:1200]
    except Exception as exc:
        return False, str(exc)


def _install_npm(packages: List[str]) -> Tuple[bool, str]:
    if not packages:
        return True, "nothing to install"
    # Vérifier que npm est disponible
    cmd = ["npm", "install"] + packages
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=180,
        )
        out = (result.stdout + "\n" + result.stderr).strip()
        return result.returncode == 0, out[:1200]
    except FileNotFoundError:
        return False, "npm not found — install Node.js first"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def detect_and_install_deps(path: str, content: str) -> str:
    """Détecte les dépendances manquantes d'un fichier et les installe.

    Retourne un résumé lisible (installé / déjà présent / erreur).
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".py":
        missing = _missing_python_packages(content)
        if not missing:
            return "deps:ok (all python packages already installed)"
        print(f"     [deps] packages manquants : {', '.join(missing)}", flush=True)
        ok, out = _install_python(missing)
        status = "installed" if ok else "install_failed"
        last_line = out.splitlines()[-1] if out else ""
        return f"deps:{status} pip install {' '.join(missing)} -> {last_line}"

    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        missing = _missing_npm_packages(content)
        if not missing:
            return "deps:ok (all npm packages already installed)"
        print(f"     [deps] packages npm manquants : {', '.join(missing)}", flush=True)
        ok, out = _install_npm(missing)
        status = "installed" if ok else "install_failed"
        last_line = out.splitlines()[-1] if out else ""
        return f"deps:{status} npm install {' '.join(missing)} -> {last_line}"

    return "deps:skipped (unsupported file type)"
