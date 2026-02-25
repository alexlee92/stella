"""
P2.4 — Mémoire cross-sessions et cross-projets.

Stocke les décisions, corrections et patterns réussis dans
~/.stella/global_memory/ — persistant entre sessions et projets.

Utilise une recherche lexicale BM25 légère (pas d'Ollama requis),
ce qui garantit que la mémoire globale fonctionne même hors ligne.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Répertoire global
# ---------------------------------------------------------------------------


def _global_dir() -> Path:
    """Répertoire ~/.stella/global_memory/ (créé si inexistant)."""
    d = Path.home() / ".stella" / "global_memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _strategy_path() -> Path:
    return _global_dir() / "fix_strategies.jsonl"


def _pattern_path() -> Path:
    return _global_dir() / "erp_patterns.jsonl"


# ---------------------------------------------------------------------------
# Écriture
# ---------------------------------------------------------------------------


def remember_globally(
    issue: str,
    strategy: str,
    project: Optional[str] = None,
    files: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> None:
    """
    Sauvegarde une décision/correction réussie dans la mémoire globale.

    Args:
        issue: Description du problème rencontré.
        strategy: Solution ou séquence d'actions qui a fonctionné.
        project: Nom ou chemin du projet source (pour traçabilité).
        files: Fichiers impliqués.
        tags: Tags libres (ex: ["sqlalchemy", "migration", "erp"]).
    """
    record = {
        "issue": (issue or "")[:400],
        "strategy": (strategy or "")[:2000],
        "project": (project or "")[:200],
        "files": (files or [])[:8],
        "tags": (tags or [])[:10],
    }
    try:
        with _strategy_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def remember_erp_pattern(
    entity: str,
    pattern: str,
    framework: str = "",
    tags: Optional[list[str]] = None,
) -> None:
    """
    Sauvegarde un pattern ERP réutilisable (modèle, workflow, API, etc.).

    Args:
        entity: Nom de l'entité (ex: "Invoice", "PurchaseOrder").
        pattern: Description ou code du pattern.
        framework: Framework concerné (ex: "sqlalchemy", "fastapi", "django").
        tags: Tags libres.
    """
    record = {
        "entity": (entity or "")[:100],
        "pattern": (pattern or "")[:2000],
        "framework": (framework or "")[:50],
        "tags": (tags or [])[:10],
    }
    try:
        with _pattern_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Recherche lexicale BM25 légère
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", text.lower())


def _bm25_score(
    query_terms: list[str],
    doc_text: str,
    n_docs: int,
    df: dict[str, int],
    avg_len: float,
) -> float:
    if not query_terms:
        return 0.0
    tokens = _tokenize(doc_text)
    if not tokens:
        return 0.0
    tf = Counter(tokens)
    doc_len = max(1, len(tokens))
    k1, b = 1.2, 0.75
    score = 0.0
    import numpy as np

    for term in query_terms:
        term_tf = tf.get(term, 0)
        if term_tf == 0:
            continue
        idf = np.log((n_docs - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1.0)
        denom = term_tf + k1 * (1 - b + b * (doc_len / avg_len))
        score += idf * ((term_tf * (k1 + 1)) / max(1e-9, denom))
    return float(score)


def _load_jsonl(path: Path, limit: int = 500) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return rows[-limit:]


def _record_to_text(record: dict) -> str:
    """Convertit un enregistrement en texte plat pour le scoring BM25."""
    parts = [
        record.get("issue", ""),
        record.get("strategy", ""),
        record.get("entity", ""),
        record.get("pattern", ""),
        record.get("framework", ""),
        " ".join(record.get("tags", [])),
        " ".join(record.get("files", [])),
    ]
    return " ".join(p for p in parts if p)


# ---------------------------------------------------------------------------
# Point d'entrée principal
# ---------------------------------------------------------------------------


def search_global_memory(query: str, k: int = 4) -> list[tuple[str, str]]:
    """
    Recherche dans la mémoire globale par similarité lexicale (BM25).

    Returns:
        Liste de (source_label, text) triée par score décroissant.
        Compatible avec le format attendu par search_memory().
    """
    strategies = _load_jsonl(_strategy_path())
    patterns = _load_jsonl(_pattern_path())

    all_records: list[tuple[str, dict]] = [
        ("memory://global/fix_strategies", r) for r in strategies
    ] + [("memory://global/erp_patterns", r) for r in patterns]

    if not all_records:
        return []

    texts = [_record_to_text(r) for _, r in all_records]
    query_terms = [t for t in _tokenize(query) if len(t) > 2]

    # Build document frequency for BM25
    df: dict[str, int] = Counter()
    total_len = 0
    for t in texts:
        toks = set(_tokenize(t))
        total_len += max(1, len(_tokenize(t)))
        for tok in toks:
            df[tok] += 1
    avg_len = max(1.0, total_len / max(1, len(texts)))
    n_docs = len(texts)

    scored = []
    for (label, record), text in zip(all_records, texts):
        score = _bm25_score(query_terms, text, n_docs, df, avg_len)
        if score > 0:
            scored.append((score, label, record))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for _, label, record in scored[:k]:
        text = _record_to_text(record)[:1200]
        results.append((label, text))

    return results


def global_memory_stats() -> dict:
    """Retourne des statistiques sur la mémoire globale."""
    strats = _load_jsonl(_strategy_path())
    patterns = _load_jsonl(_pattern_path())
    doc_cache = _load_jsonl(_doc_cache_path())
    return {
        "fix_strategies": len(strats),
        "erp_patterns": len(patterns),
        "doc_cache_entries": len(doc_cache),
        "global_dir": str(_global_dir()),
    }


# ---------------------------------------------------------------------------
# P2.5 — Cache de documentation web (évite les recherches redondantes)
# ---------------------------------------------------------------------------


def _doc_cache_path() -> Path:
    return _global_dir() / "doc_cache.jsonl"


def cache_web_result(query: str, result: str) -> None:
    """
    Sauvegarde le résultat d'une recherche web dans le cache global.
    Évite de refaire la même recherche lors de sessions futures.
    """
    record = {
        "query": (query or "")[:200],
        "result": (result or "")[:3000],
    }
    try:
        with _doc_cache_path().open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass


def search_doc_cache(query: str, k: int = 2) -> list[tuple[str, str]]:
    """
    Recherche dans le cache de documentation par similarité lexicale.

    Returns:
        Liste de (source_label, text) si des résultats pertinents sont trouvés.
    """
    entries = _load_jsonl(_doc_cache_path(), limit=300)
    if not entries:
        return []

    query_terms = [t for t in _tokenize(query) if len(t) > 2]
    if not query_terms:
        return []

    texts = [e.get("query", "") + " " + e.get("result", "") for e in entries]

    df: dict[str, int] = Counter()
    total_len = 0
    for t in texts:
        toks = set(_tokenize(t))
        total_len += max(1, len(_tokenize(t)))
        for tok in toks:
            df[tok] += 1
    avg_len = max(1.0, total_len / max(1, len(texts)))
    n_docs = len(texts)

    scored = []
    for i, (entry, text) in enumerate(zip(entries, texts)):
        score = _bm25_score(query_terms, text, n_docs, df, avg_len)
        if score > 1.0:  # Seuil minimal pour éviter les faux positifs
            scored.append((score, entry))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for _, entry in scored[:k]:
        label = f"memory://doc_cache/{entry.get('query', '')[:60]}"
        text = entry.get("result", "")[:1500]
        results.append((label, text))
    return results
