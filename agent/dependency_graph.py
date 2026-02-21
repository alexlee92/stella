"""
agent/dependency_graph.py — Graphe de dépendances inter-fichiers.

Construit un graphe d'imports Python pour permettre à Stella d'inclure
automatiquement les fichiers liés dans le contexte de recherche.
"""
import ast
import os
import re
from functools import lru_cache
from typing import Dict, Set

from agent.config import PROJECT_ROOT
from agent.project_scan import get_python_files


def _rel(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")


def _module_to_path(module: str) -> str | None:
    """Convertit un nom de module Python en chemin relatif probable."""
    parts = module.split(".")
    # Essai direct : agent.memory → agent/memory.py
    candidate = os.path.join(PROJECT_ROOT, *parts) + ".py"
    if os.path.isfile(candidate):
        return _rel(candidate)
    # Essai package : agent.memory → agent/memory/__init__.py
    candidate2 = os.path.join(PROJECT_ROOT, *parts, "__init__.py")
    if os.path.isfile(candidate2):
        return _rel(candidate2)
    return None


def _extract_imports_ast(file_path: str) -> Set[str]:
    """Extrait les imports d'un fichier Python via AST."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            source = f.read()
        tree = ast.parse(source)
    except Exception:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def build_dependency_graph() -> Dict[str, Set[str]]:
    """Construit le graphe complet : {fichier → ensemble de fichiers importés}."""
    graph: Dict[str, Set[str]] = {}
    files = get_python_files(PROJECT_ROOT)

    for abs_path in files:
        rel = _rel(abs_path)
        imports = _extract_imports_ast(abs_path)
        deps: Set[str] = set()
        for mod in imports:
            dep_path = _module_to_path(mod)
            if dep_path and dep_path != rel:
                deps.add(dep_path)
        graph[rel] = deps

    return graph


@lru_cache(maxsize=1)
def _cached_graph() -> Dict[str, Set[str]]:
    return build_dependency_graph()


def get_related_files(target_rel: str, depth: int = 1) -> list[str]:
    """Retourne les fichiers directement liés à target_rel.

    depth=1 : imports directs seulement
    depth=2 : imports de second niveau inclus
    """
    graph = _cached_graph()
    norm = target_rel.replace("\\", "/")

    related: Set[str] = set()

    # Fichiers que target importe
    direct_deps = graph.get(norm, set())
    related.update(direct_deps)

    # Fichiers qui importent target (reverse graph)
    for path, deps in graph.items():
        if norm in deps:
            related.add(path)

    if depth >= 2:
        second_level: Set[str] = set()
        for rel in list(related):
            second_level.update(graph.get(rel, set()))
            for path, deps in graph.items():
                if rel in deps:
                    second_level.add(path)
        related.update(second_level)

    related.discard(norm)
    return sorted(related)[:10]


def invalidate_cache():
    """Invalide le cache du graphe (à appeler après création/modification de fichiers)."""
    _cached_graph.cache_clear()
