"""
P5.3 — Migration helper for database schema changes.

Detects model changes and suggests migration commands for:
- Alembic (SQLAlchemy)
- Django migrations
- Odoo module upgrades
"""

import os
from typing import List, Optional

from agent.config import PROJECT_ROOT


def detect_migration_framework() -> Optional[str]:
    """Detect which migration framework the project uses."""
    # Alembic
    if os.path.isdir(os.path.join(PROJECT_ROOT, "alembic")) or os.path.isfile(
        os.path.join(PROJECT_ROOT, "alembic.ini")
    ):
        return "alembic"

    # Django
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if "migrations" in dirs:
            init = os.path.join(root, "migrations", "__init__.py")
            if os.path.isfile(init):
                return "django"
        dirs[:] = [
            d for d in dirs if d not in {".venv", "__pycache__", ".git", "node_modules"}
        ]

    # Odoo
    for root, dirs, files in os.walk(PROJECT_ROOT):
        if "__manifest__.py" in files or "__openerp__.py" in files:
            return "odoo"
        dirs[:] = [d for d in dirs if d not in {".venv", "__pycache__", ".git"}]

    return None


def suggest_migration_commands(changed_files: List[str]) -> List[str]:
    """Suggest migration commands based on changed model files."""
    model_files = [f for f in changed_files if _is_model_file(f)]
    if not model_files:
        return []

    framework = detect_migration_framework()
    suggestions = []

    if framework == "alembic":
        suggestions.append("alembic revision --autogenerate -m 'auto: model changes'")
        suggestions.append("alembic upgrade head")
    elif framework == "django":
        # Detect app names from file paths
        apps = set()
        for f in model_files:
            parts = f.replace("\\", "/").split("/")
            if len(parts) >= 2:
                apps.add(parts[0])
        for app in sorted(apps):
            suggestions.append(f"python manage.py makemigrations {app}")
        suggestions.append("python manage.py migrate")
    elif framework == "odoo":
        suggestions.append(
            "# Restart Odoo server with -u <module_name> to apply changes"
        )
    else:
        suggestions.append(
            "# No migration framework detected. Consider using Alembic or Django migrations."
        )

    return suggestions


def validate_model_migration_coherence(changed_files: List[str]) -> List[str]:
    """Check if model changes have corresponding migrations."""
    warnings = []
    model_files = [f for f in changed_files if _is_model_file(f)]
    migration_files = [f for f in changed_files if _is_migration_file(f)]

    if model_files and not migration_files:
        warnings.append(
            f"Model file(s) changed ({', '.join(model_files)}) "
            "but no migration file was created. Run migration command."
        )

    return warnings


def _is_model_file(path: str) -> bool:
    low = path.lower().replace("\\", "/")
    return ("model" in low and low.endswith(".py")) or (
        "schema" in low and low.endswith(".py")
    )


def _is_migration_file(path: str) -> bool:
    low = path.lower().replace("\\", "/")
    return "migration" in low or "alembic/versions" in low
