"""
P2.2 — Awareness du schéma de base de données.

Responsabilités :
- Scanner les fichiers Python à la recherche de modèles SQLAlchemy / Django ORM
- Extraire tables, colonnes, types, clés étrangères, relations
- Produire un résumé injecté dans le contexte du planner
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Détection des fichiers modèles
# ---------------------------------------------------------------------------


def find_model_files(root: str = ".") -> list[str]:
    """Retourne les chemins relatifs des fichiers contenant des modèles ORM."""
    found = []
    sqlalchemy_markers = {
        "Column",
        "mapped_column",
        "relationship",
        "declarative_base",
        "DeclarativeBase",
    }
    django_markers = {"models.Model", "models.CharField", "models.ForeignKey"}
    all_markers = sqlalchemy_markers | django_markers

    try:
        for dirpath, dirs, filenames in os.walk(root):
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".")
                and d
                not in {
                    "__pycache__",
                    ".venv",
                    "venv",
                    "env",
                    "node_modules",
                    "migrations",
                }
            ]
            for fname in filenames:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(dirpath, fname)
                try:
                    text = Path(fpath).read_text(encoding="utf-8", errors="ignore")
                    if any(m in text for m in all_markers):
                        found.append(os.path.relpath(fpath, root).replace("\\", "/"))
                except OSError:
                    continue
    except OSError:
        pass
    return found


# ---------------------------------------------------------------------------
# Extraction AST — SQLAlchemy
# ---------------------------------------------------------------------------


def _extract_sqlalchemy_models(source: str, filepath: str) -> list[dict]:
    """
    Parse un fichier Python et extrait les modèles SQLAlchemy.
    Retourne une liste de dicts : {name, table, columns, relationships}.
    """
    models = []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return models

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Vérifier que la classe hérite d'une base SQLAlchemy
        bases = [ast.unparse(b) for b in node.bases]
        is_sqla = any(
            "Base" in b or "Model" in b or "DeclarativeBase" in b for b in bases
        )
        if not is_sqla:
            continue

        model: dict = {
            "name": node.name,
            "file": filepath,
            "table": None,
            "columns": [],
            "relationships": [],
        }

        for item in ast.walk(node):
            # __tablename__
            if (
                isinstance(item, ast.Assign)
                and len(item.targets) == 1
                and isinstance(item.targets[0], ast.Name)
                and item.targets[0].id == "__tablename__"
                and isinstance(item.value, ast.Constant)
            ):
                model["table"] = item.value.value
                continue

            # Column() / mapped_column()
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                col_name = item.target.id
                annotation = ast.unparse(item.annotation) if item.annotation else ""
                model["columns"].append({"name": col_name, "type": annotation[:80]})
                continue

            if isinstance(item, ast.Assign):
                for tgt in item.targets:
                    if not isinstance(tgt, ast.Name):
                        continue
                    col_name = tgt.id
                    val_src = ast.unparse(item.value)
                    if "Column(" in val_src or "mapped_column(" in val_src:
                        # Extraire le type brut
                        col_type = val_src[:60]
                        model["columns"].append({"name": col_name, "type": col_type})
                    elif "relationship(" in val_src:
                        model["relationships"].append(col_name)

        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Extraction regex — Django
# ---------------------------------------------------------------------------


def _extract_django_models(source: str, filepath: str) -> list[dict]:
    """Extraction simplifiée des modèles Django via regex."""
    models = []
    # Trouver les classes héritant de models.Model
    class_blocks = re.findall(
        r"class\s+(\w+)\s*\(([^)]*models\.Model[^)]*)\)(.*?)(?=\nclass |\Z)",
        source,
        re.DOTALL,
    )
    for class_name, _, body in class_blocks:
        model: dict = {
            "name": class_name,
            "file": filepath,
            "table": class_name.lower() + "s",  # convention Django
            "columns": [],
            "relationships": [],
        }
        # Colonnes
        for line in body.splitlines():
            line = line.strip()
            m = re.match(r"(\w+)\s*=\s*models\.(\w+)\(", line)
            if m:
                field_name, field_type = m.group(1), m.group(2)
                if (
                    "ForeignKey" in field_type
                    or "ManyToMany" in field_type
                    or "OneToOne" in field_type
                ):
                    model["relationships"].append(field_name)
                else:
                    model["columns"].append({"name": field_name, "type": field_type})
        models.append(model)
    return models


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def read_schema(
    root: str = ".",
    files: Optional[list[str]] = None,
    max_models: int = 20,
) -> str:
    """
    Scanne les modèles ORM et retourne un résumé lisible par le LLM.

    Args:
        root: Répertoire racine du projet.
        files: Liste de fichiers à scanner (si None, auto-détection).
        max_models: Nombre maximum de modèles à inclure dans le résumé.

    Returns:
        Texte résumant le schéma, prêt à être injecté dans le contexte.
    """
    if files is None:
        files = find_model_files(root)

    if not files:
        return "Aucun modèle ORM détecté dans le projet."

    all_models = []
    for fpath in files:
        abs_path = os.path.join(root, fpath) if not os.path.isabs(fpath) else fpath
        try:
            source = Path(abs_path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        sqla = _extract_sqlalchemy_models(source, fpath)
        if sqla:
            all_models.extend(sqla)
        else:
            dj = _extract_django_models(source, fpath)
            all_models.extend(dj)

    if not all_models:
        return (
            f"Fichiers modèles trouvés ({len(files)}) : {', '.join(files[:5])}\n"
            "Impossible d'extraire les modèles (syntaxe non reconnue ou classes vides)."
        )

    # Limiter et formater
    all_models = all_models[:max_models]
    lines = [f"=== Schéma de la base de données ({len(all_models)} modèles) ===\n"]
    for m in all_models:
        table_label = f" [table={m['table']}]" if m["table"] else ""
        lines.append(f"Model: {m['name']}{table_label}  ({m['file']})")
        for col in m["columns"][:15]:
            lines.append(f"  - {col['name']}: {col['type']}")
        if m["relationships"]:
            lines.append(f"  ~ relations: {', '.join(m['relationships'][:8])}")
        lines.append("")

    return "\n".join(lines)
