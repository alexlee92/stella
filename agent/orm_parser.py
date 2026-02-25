"""
P5.2 — ORM relationship parser for ERP projects.

Extracts model definitions and relationships from popular ORM frameworks:
- SQLAlchemy (Flask, FastAPI)
- Django ORM
- Odoo models

Builds an entity-relationship summary that can be included in agent context.
"""

import ast
import os
from typing import List, Set

from agent.config import PROJECT_ROOT
from agent.project_scan import get_python_files


def _rel(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def _extract_models_from_file(file_path: str) -> List[dict]:
    """Extract ORM model definitions from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return []

    models = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue

        # Check if it's a model class
        base_names = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                base_names.append(base.id)
            elif isinstance(base, ast.Attribute):
                base_names.append(base.attr)

        is_model = any(
            b in {"Model", "Base", "DeclarativeBase", "AbstractModel", "TransientModel"}
            for b in base_names
        )
        if not is_model:
            continue

        model_info = {
            "name": node.name,
            "file": _rel(file_path),
            "line": node.lineno,
            "bases": base_names,
            "fields": [],
            "relationships": [],
        }

        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        field_info = _parse_field_assignment(
                            target.id, item.value, source
                        )
                        if field_info:
                            if field_info.get("relation"):
                                model_info["relationships"].append(field_info)
                            else:
                                model_info["fields"].append(field_info)
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                field_info = _parse_annotated_field(item.target.id, item, source)
                if field_info:
                    if field_info.get("relation"):
                        model_info["relationships"].append(field_info)
                    else:
                        model_info["fields"].append(field_info)

        models.append(model_info)

    return models


def _parse_field_assignment(name: str, value, source: str) -> dict | None:
    """Parse a field assignment like: name = Column(String) or name = ForeignKey(...)"""
    if not isinstance(value, ast.Call):
        return None

    func_name = ""
    if isinstance(value.func, ast.Name):
        func_name = value.func.id
    elif isinstance(value.func, ast.Attribute):
        func_name = value.func.attr

    # Known ORM field constructors
    relation_funcs = {
        "ForeignKey",
        "relationship",
        "ForeignKeyField",
        "ManyToManyField",
        "OneToOneField",
        "Many2one",
        "Many2many",
        "One2many",
    }
    field_funcs = {
        "Column",
        "Field",
        "fields",
        "mapped_column",
        "CharField",
        "IntegerField",
        "TextField",
        "BooleanField",
        "DateTimeField",
        "FloatField",
        "DecimalField",
    }

    if func_name in relation_funcs:
        target = ""
        if value.args:
            arg0 = value.args[0]
            if isinstance(arg0, ast.Constant):
                target = str(arg0.value)
            elif isinstance(arg0, ast.Name):
                target = arg0.id
        return {"name": name, "type": func_name, "target": target, "relation": True}

    if func_name in field_funcs:
        col_type = ""
        if value.args:
            arg0 = value.args[0]
            if isinstance(arg0, ast.Name):
                col_type = arg0.id
            elif isinstance(arg0, ast.Call) and isinstance(arg0.func, ast.Name):
                col_type = arg0.func.id
        return {"name": name, "type": col_type or func_name, "relation": False}

    return None


def _parse_annotated_field(name: str, node: ast.AnnAssign, source: str) -> dict | None:
    """Parse SQLAlchemy 2.0 style: name: Mapped[str] = mapped_column(...)"""
    if node.value and isinstance(node.value, ast.Call):
        return _parse_field_assignment(name, node.value, source)
    return {"name": name, "type": "annotated", "relation": False}


def scan_orm_models(project_root: str = PROJECT_ROOT) -> List[dict]:
    """Scan the entire project for ORM model definitions."""
    files = get_python_files(project_root)
    all_models = []
    for f in files:
        models = _extract_models_from_file(f)
        all_models.extend(models)
    return all_models


def render_er_summary(models: List[dict] | None = None) -> str:
    """Render a text-based entity-relationship summary.

    Useful for including in agent context when modifying models.
    """
    if models is None:
        models = scan_orm_models()

    if not models:
        return "No ORM models found in the project."

    lines = ["Entity-Relationship Summary:", ""]
    for m in models:
        fields_str = ", ".join(f["name"] for f in m["fields"][:8])
        lines.append(f"  [{m['name']}] ({m['file']}:{m['line']})")
        if fields_str:
            lines.append(f"    fields: {fields_str}")
        for rel in m["relationships"]:
            lines.append(f"    -> {rel['type']}({rel['target']}) via .{rel['name']}")
        lines.append("")

    return "\n".join(lines)


def get_related_models(model_name: str, models: List[dict] | None = None) -> List[str]:
    """Find models related to *model_name* through FK/relationships."""
    if models is None:
        models = scan_orm_models()

    related: Set[str] = set()
    for m in models:
        if m["name"] == model_name:
            for rel in m["relationships"]:
                if rel["target"]:
                    related.add(rel["target"])
        else:
            for rel in m["relationships"]:
                if rel["target"] == model_name:
                    related.add(m["name"])

    return sorted(related)
