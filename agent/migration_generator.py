"""
P1.2 — Générateur de migrations Alembic.

Responsabilités :
- Vérifier la présence d'Alembic dans le projet
- Détecter les fichiers de modèles SQLAlchemy
- Générer des révisions via alembic --autogenerate
- Appliquer des migrations via alembic upgrade
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def is_alembic_configured() -> bool:
    """Vérifie si Alembic est configuré dans le projet (alembic.ini ou alembic/)."""
    return os.path.isfile("alembic.ini") or os.path.isdir("alembic")


def find_model_files(root: str = ".") -> list[str]:
    """Retourne les chemins relatifs des fichiers contenant des modèles SQLAlchemy."""
    found = []
    markers = (
        "Base",
        "DeclarativeBase",
        "DeclarativeMeta",
        "declarative_base",
        "Column",
        "mapped_column",
    )
    try:
        for dirpath, _dirs, filenames in os.walk(root):
            # Skip hidden dirs and virtual envs
            _dirs[:] = [
                d
                for d in _dirs
                if not d.startswith(".")
                and d not in {"__pycache__", ".venv", "venv", "env", "node_modules"}
            ]
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                    if any(m in text for m in markers):
                        found.append(os.path.relpath(fpath, root).replace("\\", "/"))
                except OSError:
                    continue
    except OSError:
        pass
    return found


def generate_migration(
    message: str = "auto_migration",
    timeout: int = 60,
) -> dict:
    """
    Génère une révision Alembic via --autogenerate.

    Returns:
        {
            "ok": bool,
            "output": str,
            "revision_file": Optional[str],  # chemin relatif si détecté
        }
    """
    if not is_alembic_configured():
        return {
            "ok": False,
            "output": (
                "Alembic n'est pas configuré dans ce projet. "
                "Créez alembic.ini avec 'alembic init alembic' et configurez "
                "sqlalchemy.url dans alembic/env.py avant de générer des migrations."
            ),
            "revision_file": None,
        }

    cmd = ["python", "-m", "alembic", "revision", "--autogenerate", "-m", message]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout + proc.stderr).strip()

        # Extraire le chemin du fichier de révision depuis la sortie alembic
        revision_file: Optional[str] = None
        for line in output.splitlines():
            if "Generating" in line and ".py" in line:
                # "  Generating /abs/path/alembic/versions/abc123_xxx.py ..."
                parts = line.strip().split()
                for part in parts:
                    if part.endswith(".py"):
                        try:
                            revision_file = os.path.relpath(part).replace("\\", "/")
                        except ValueError:
                            revision_file = part
                        break

        return {
            "ok": proc.returncode == 0,
            "output": output[:2000],
            "revision_file": revision_file,
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "output": f"Timeout ({timeout}s) lors de la génération de migration.",
            "revision_file": None,
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "output": "alembic introuvable. Installez-le : pip install alembic",
            "revision_file": None,
        }


def apply_migration(
    target: str = "head",
    timeout: int = 120,
) -> dict:
    """
    Applique les migrations Alembic jusqu'à la cible spécifiée.

    Args:
        target: Révision cible (par défaut "head"). Exemples : "head", "+1", "-1", "<rev_id>"
        timeout: Délai maximal en secondes.

    Returns:
        {"ok": bool, "output": str}
    """
    if not is_alembic_configured():
        return {
            "ok": False,
            "output": (
                "Alembic n'est pas configuré dans ce projet. "
                "Créez alembic.ini avec 'alembic init alembic' avant d'appliquer des migrations."
            ),
        }

    cmd = ["python", "-m", "alembic", "upgrade", target]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout + proc.stderr).strip()
        return {
            "ok": proc.returncode == 0,
            "output": output[:2000],
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "output": f"Timeout ({timeout}s) lors de l'application des migrations.",
        }
    except FileNotFoundError:
        return {
            "ok": False,
            "output": "alembic introuvable. Installez-le : pip install alembic",
        }


def migration_status(timeout: int = 30) -> dict:
    """Retourne l'état courant des migrations (alembic current + history)."""
    if not is_alembic_configured():
        return {"ok": False, "output": "Alembic non configuré."}

    try:
        proc = subprocess.run(
            ["python", "-m", "alembic", "current"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout + proc.stderr).strip()
        return {"ok": proc.returncode == 0, "output": output[:1000]}
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return {"ok": False, "output": str(exc)}
