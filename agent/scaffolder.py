"""
P3.7 -- Code scaffolding / template generation.

Generates boilerplate files for common patterns:
- FastAPI endpoint
- Django model / view
- React component
- Python module with tests
- Test file for existing module
"""

import os
from typing import Dict, Optional

from agent.config import PROJECT_ROOT
from agent.tooling import write_new_file

# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_TEMPLATES: Dict[str, Dict[str, str]] = {
    "fastapi-endpoint": {
        "description": "FastAPI router with CRUD endpoints",
        "extension": ".py",
        "template": '''\
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter(prefix="/{name}", tags=["{name}"])


class {Name}Base(BaseModel):
    name: str


class {Name}Create({Name}Base):
    pass


class {Name}Response({Name}Base):
    id: int

    class Config:
        from_attributes = True


@router.get("/", response_model=List[{Name}Response])
async def list_{name}s():
    """List all {name}s."""
    return []


@router.get("/{{item_id}}", response_model={Name}Response)
async def get_{name}(item_id: int):
    """Get a single {name} by ID."""
    raise HTTPException(status_code=404, detail="{Name} not found")


@router.post("/", response_model={Name}Response, status_code=201)
async def create_{name}(data: {Name}Create):
    """Create a new {name}."""
    return {Name}Response(id=1, **data.model_dump())


@router.delete("/{{item_id}}", status_code=204)
async def delete_{name}(item_id: int):
    """Delete a {name}."""
    pass
''',
    },
    "django-model": {
        "description": "Django model with admin registration",
        "extension": ".py",
        "template": '''\
from django.db import models


class {Name}(models.Model):
    """Model for {name}."""

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "{name}"
        verbose_name_plural = "{name}s"

    def __str__(self):
        return self.name
''',
    },
    "django-view": {
        "description": "Django class-based views (List + Detail + Create)",
        "extension": ".py",
        "template": '''\
from django.views.generic import ListView, DetailView, CreateView
from django.urls import reverse_lazy


class {Name}ListView(ListView):
    template_name = "{name}/list.html"
    context_object_name = "{name}s"
    paginate_by = 20


class {Name}DetailView(DetailView):
    template_name = "{name}/detail.html"
    context_object_name = "{name}"


class {Name}CreateView(CreateView):
    template_name = "{name}/form.html"
    fields = ["name", "description"]
    success_url = reverse_lazy("{name}-list")
''',
    },
    "react-component": {
        "description": "React functional component with TypeScript",
        "extension": ".tsx",
        "template": '''\
import React from "react";

interface {Name}Props {{
  title?: string;
  children?: React.ReactNode;
}}

export const {Name}: React.FC<{Name}Props> = ({{ title, children }}) => {{
  return (
    <div className="{name}">
      {{title && <h2>{{title}}</h2>}}
      {{children}}
    </div>
  );
}};

export default {Name};
''',
    },
    "python-module": {
        "description": "Python module with docstring and basic structure",
        "extension": ".py",
        "template": '''\
"""
{Name} module.

Provides functionality for {name} operations.
"""

from typing import List, Optional


class {Name}:
    """Main class for {name} operations."""

    def __init__(self):
        self._items: List[str] = []

    def add(self, item: str) -> None:
        """Add an item."""
        self._items.append(item)

    def get_all(self) -> List[str]:
        """Return all items."""
        return list(self._items)

    def find(self, query: str) -> Optional[str]:
        """Find first item matching query."""
        for item in self._items:
            if query.lower() in item.lower():
                return item
        return None
''',
    },
    "test": {
        "description": "pytest test file skeleton",
        "extension": ".py",
        "template": '''\
"""Tests for {name}."""

import pytest


class Test{Name}:
    """Test suite for {Name}."""

    def test_creation(self):
        """Test basic creation."""
        assert True  # TODO: implement

    def test_basic_operation(self):
        """Test basic operation."""
        assert True  # TODO: implement

    def test_edge_case(self):
        """Test edge cases."""
        assert True  # TODO: implement
''',
    },
}

# ---------------------------------------------------------------------------
# Custom templates from .stella/templates/
# ---------------------------------------------------------------------------


def _load_custom_templates() -> Dict[str, Dict[str, str]]:
    """Load user-defined templates from .stella/templates/."""
    templates_dir = os.path.join(PROJECT_ROOT, ".stella", "templates")
    if not os.path.isdir(templates_dir):
        return {}
    custom = {}
    for fname in os.listdir(templates_dir):
        if not fname.endswith((".py", ".tsx", ".ts", ".js", ".html")):
            continue
        tpl_name = os.path.splitext(fname)[0]
        ext = os.path.splitext(fname)[1]
        path = os.path.join(templates_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            custom[tpl_name] = {
                "description": f"Custom template: {tpl_name}",
                "extension": ext,
                "template": content,
            }
        except OSError:
            pass
    return custom


def list_templates() -> Dict[str, str]:
    """Return available template types with descriptions."""
    all_tpls = {**_TEMPLATES, **_load_custom_templates()}
    return {name: tpl["description"] for name, tpl in all_tpls.items()}


def scaffold(template_type: str, name: str, output_dir: str = "") -> str:
    """Generate a file from a template.

    Args:
        template_type: One of the registered template types
        name: The entity name (e.g., "user", "product", "invoice")
        output_dir: Optional subdirectory to place the file in

    Returns:
        Result message with created file path
    """
    all_tpls = {**_TEMPLATES, **_load_custom_templates()}

    if template_type not in all_tpls:
        available = ", ".join(sorted(all_tpls.keys()))
        return f"[!] Template inconnu : '{template_type}'. Disponibles : {available}"

    tpl = all_tpls[template_type]
    # Prepare template variables
    clean_name = name.strip().replace("-", "_").replace(" ", "_").lower()
    capitalized = "".join(w.capitalize() for w in clean_name.split("_"))

    content = tpl["template"].replace("{name}", clean_name).replace("{Name}", capitalized)

    # Determine output file path
    ext = tpl["extension"]
    if template_type == "test":
        filename = f"test_{clean_name}{ext}"
        if not output_dir:
            output_dir = "tests"
    elif template_type == "react-component":
        filename = f"{capitalized}{ext}"
        if not output_dir:
            output_dir = "src/components"
    else:
        filename = f"{clean_name}{ext}"

    if output_dir:
        rel_path = os.path.join(output_dir, filename)
    else:
        rel_path = filename

    # Check if file already exists
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    if os.path.exists(abs_path):
        return f"[!] Le fichier existe deja : {rel_path}"

    result = write_new_file(rel_path, content)
    if result.startswith("ok:"):
        lines = content.count("\n") + 1
        return f"[scaffold] Cree : {rel_path} ({lines} lignes, template '{template_type}')"
    return f"[scaffold] Erreur : {result}"
