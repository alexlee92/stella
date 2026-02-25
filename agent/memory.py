"""
Vector memory and indexing for Stella.
Supports semantic search (Ollama embeddings) and lexical search (BM25).
"""

import hashlib
import json
import os
import re
from collections import OrderedDict, Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import requests
from tqdm import tqdm

from agent.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTEXT_BUDGET_CHARS,
    CONTEXT_MAX_CHUNKS,
    CONTEXT_MAX_PER_FILE,
    EMBED_MODEL,
    MAX_CHUNKS_PER_FILE,
    MEMORY_INDEX_DIR,
    MEMORY_QUERY_CACHE_SIZE,
    OLLAMA_EMBED_URL,
    PROJECT_ROOT,
    REQUEST_TIMEOUT,
)
from agent.git_tools import changed_files
from agent.project_scan import get_source_files, load_file_content


@dataclass
class MemoryDoc:
    path: str
    chunk_id: int
    text: str
    symbol: str = ""


documents: List[MemoryDoc] = []
vectors: List[np.ndarray] = []
_index_loaded = False
_query_cache: "OrderedDict[str, List[Tuple[str, str]]]" = OrderedDict()
_token_df: Dict[str, int] = {}
_avg_doc_len = 1.0
_project_cache_token = ""

SOURCE_EXTENSIONS = {
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".json",
    ".toml",
    ".yaml",
    ".yml",
    # P5.1 — ERP-related extensions
    ".sql",
    ".xml",
    ".html",
    ".css",
    ".scss",
}


def _ensure_index_dir() -> str:
    index_dir = os.path.join(PROJECT_ROOT, MEMORY_INDEX_DIR)
    os.makedirs(index_dir, exist_ok=True)
    return index_dir


def _index_files() -> tuple[str, str]:
    d = _ensure_index_dir()
    return os.path.join(d, "docs.json"), os.path.join(d, "vectors.npy")


def _strategy_file() -> str:
    return os.path.join(_ensure_index_dir(), "fix_strategies.jsonl")


def remember_fix_strategy(issue: str, strategy: str, files: List[str] | None = None):
    record = {
        "issue": (issue or "")[:300],
        "strategy": (strategy or "")[:1500],
        "files": files or [],
    }
    try:
        with open(_strategy_file(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        return

    # P2.4 — Propager vers la mémoire globale cross-sessions
    try:
        from agent.global_memory import remember_globally

        remember_globally(
            issue=issue,
            strategy=strategy,
            project=PROJECT_ROOT,
            files=files,
        )
    except Exception:
        pass


def _load_fix_strategy_docs(limit: int = 8) -> List[Tuple[str, str]]:
    path = _strategy_file()
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                issue = str(obj.get("issue", ""))
                strategy = str(obj.get("strategy", ""))
                files = obj.get("files") or []
                files_text = ", ".join(files[:4]) if isinstance(files, list) else ""
                text = (
                    f"issue: {issue}\nstrategy: {strategy}\nfiles: {files_text}".strip()
                )
                rows.append(("memory://fix_strategies", text))
    except OSError:
        return []
    return rows[-limit:]


def _normalize(vec: np.ndarray):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{1,}", text.lower())


def _rebuild_lexical_stats():
    global _token_df, _avg_doc_len

    df = Counter()
    total_len = 0
    for doc in documents:
        tokens = _tokenize(doc.text)
        total_len += max(1, len(tokens))
        for tok in set(tokens):
            df[tok] += 1

    _token_df = dict(df)
    _avg_doc_len = max(1.0, total_len / max(1, len(documents)))


def _cache_get(key: str):
    if key not in _query_cache:
        return None
    _query_cache.move_to_end(key)
    return _query_cache[key]


def _cache_put(key: str, value):
    _query_cache[key] = value
    _query_cache.move_to_end(key)
    while len(_query_cache) > MEMORY_QUERY_CACHE_SIZE:
        _query_cache.popitem(last=False)


def embed(text: str):
    try:
        r = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": EMBED_MODEL, "prompt": text[:6000]},
            timeout=REQUEST_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return None

    emb = data.get("embedding")
    if not emb:
        return None

    return _normalize(np.array(emb, dtype=np.float32))


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append((text[start:end], "window"))
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return chunks[:MAX_CHUNKS_PER_FILE]


def _chunk_python_by_symbols(content: str):
    import ast

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    lines = content.splitlines()
    chunks = []

    nodes = []
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and hasattr(node, "lineno")
            and hasattr(node, "end_lineno")
        ):
            nodes.append(node)

    if not nodes:
        return []

    first_line = max(1, min(n.lineno for n in nodes))
    prelude = "\n".join(lines[: first_line - 1]).strip()
    if prelude:
        chunks.append((prelude, "module_prelude"))

    for node in nodes:
        start = max(1, node.lineno)
        end = max(start, node.end_lineno)
        text = "\n".join(lines[start - 1 : end]).strip()
        if text:
            chunks.append((text, f"symbol:{node.name}"))

    return chunks[:MAX_CHUNKS_PER_FILE]


def _chunk_jsts_by_symbols(content: str):
    lines = content.splitlines()
    chunks = []
    current_name = ""
    start = None
    pattern = re.compile(
        r"^\s*(?:export\s+)?(?:async\s+)?(?:function|class|const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)"
    )
    for idx, line in enumerate(lines, start=1):
        m = pattern.search(line)
        if m:
            if start is not None and current_name:
                text = "\n".join(lines[start - 1 : idx - 1]).strip()
                if text:
                    chunks.append((text, f"symbol:{current_name}"))
            current_name = m.group(1)
            start = idx

    if start is not None and current_name:
        text = "\n".join(lines[start - 1 :]).strip()
        if text:
            chunks.append((text, f"symbol:{current_name}"))

    return chunks[:MAX_CHUNKS_PER_FILE]


def _chunk_sql_by_statements(content: str):
    """P5.1 — Chunk SQL files by statement (CREATE, ALTER, INSERT, etc.)."""
    statements = re.split(r";\s*\n", content)
    chunks = []
    for stmt in statements:
        stmt = stmt.strip()
        if not stmt or len(stmt) < 10:
            continue
        # Extract a label for the statement
        m = re.match(
            r"(CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SELECT)\s+(?:TABLE|INDEX|VIEW|INTO|FROM)?\s*(\S+)",
            stmt,
            re.IGNORECASE,
        )
        label = f"sql:{m.group(1).lower()}_{m.group(2)}" if m else "sql:statement"
        chunks.append((stmt[:CHUNK_SIZE], label))
    return chunks[:MAX_CHUNKS_PER_FILE] if chunks else []


def _chunk_xml_by_elements(content: str):
    """P5.1 — Chunk XML files by top-level elements (for Odoo, configs, etc.)."""
    chunks = []
    # Split on top-level opening tags
    parts = re.split(
        r"(?=<(?:record|template|menuitem|field|form|tree|data)\s)",
        content,
        flags=re.IGNORECASE,
    )
    for part in parts:
        part = part.strip()
        if not part or len(part) < 20:
            continue
        m = re.match(r"<(\w+)", part)
        label = f"xml:{m.group(1)}" if m else "xml:element"
        chunks.append((part[:CHUNK_SIZE], label))
    return chunks[:MAX_CHUNKS_PER_FILE] if chunks else []


def _chunk_for_file(path: str, content: str):
    if path.endswith(".py"):
        symbol_chunks = _chunk_python_by_symbols(content)
        if symbol_chunks:
            return symbol_chunks
    if path.endswith((".js", ".jsx", ".ts", ".tsx")):
        symbol_chunks = _chunk_jsts_by_symbols(content)
        if symbol_chunks:
            return symbol_chunks
    # P5.1 — SQL and XML chunking for ERP projects
    if path.endswith(".sql"):
        sql_chunks = _chunk_sql_by_statements(content)
        if sql_chunks:
            return sql_chunks
    if path.endswith(".xml"):
        xml_chunks = _chunk_xml_by_elements(content)
        if xml_chunks:
            return xml_chunks
    return _chunk_text(content)


def _file_hash(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8", errors="ignore")).hexdigest()


def _load_index() -> bool:
    global _index_loaded
    docs_path, vec_path = _index_files()
    if not (os.path.exists(docs_path) and os.path.exists(vec_path)):
        return False

    try:
        with open(docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        arr = np.load(vec_path)
    except Exception:
        return False

    documents.clear()
    vectors.clear()

    for item in data.get("docs", []):
        documents.append(
            MemoryDoc(
                path=item["path"],
                chunk_id=item["chunk_id"],
                text=item["text"],
                symbol=item.get("symbol", ""),
            )
        )

    for row in arr:
        vectors.append(row.astype(np.float32))

    _rebuild_lexical_stats()
    _index_loaded = True
    return True


def _save_index(meta: dict):
    docs_path, vec_path = _index_files()
    payload = {
        "meta": meta,
        "docs": [
            {"path": d.path, "chunk_id": d.chunk_id, "text": d.text, "symbol": d.symbol}
            for d in documents
        ],
    }

    with open(docs_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    matrix = (
        np.array(vectors, dtype=np.float32)
        if vectors
        else np.zeros((0, 1), dtype=np.float32)
    )
    np.save(vec_path, matrix)


def _load_persisted_hashes() -> Dict[str, str]:
    """P3.5 — Load file hashes from the persisted index metadata."""
    docs_path, _ = _index_files()
    if not os.path.exists(docs_path):
        return {}
    try:
        with open(docs_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("meta", {}).get("file_hashes", {})
    except Exception:
        return {}


def build_memory(project_root: str = PROJECT_ROOT, force_rebuild: bool = False):
    global _index_loaded

    if _index_loaded and not force_rebuild:
        print(f"[memory] loaded in-process index with {len(vectors)} vectors")
        return

    if not force_rebuild and _load_index():
        # P3.5 — Incremental: check if any files changed since last index
        changed = _incremental_update(project_root)
        if changed > 0:
            print(f"[memory] incremental update: re-indexed {changed} changed file(s)")
        else:
            print(f"[memory] loaded persisted index with {len(vectors)} vectors")
        return

    _full_rebuild(project_root)


def _incremental_update(project_root: str) -> int:
    """P3.5 — Re-index only files that changed since last full build."""
    old_hashes = _load_persisted_hashes()
    if not old_hashes:
        return 0

    files = get_source_files(project_root, extensions=SOURCE_EXTENSIONS)
    changed_count = 0
    new_hashes = {}

    for file_path in files:
        content = load_file_content(file_path)
        if not content.strip():
            continue
        h = _file_hash(content)
        new_hashes[file_path] = h

        old_h = old_hashes.get(file_path)
        if old_h == h:
            continue

        # File is new or changed — remove old chunks and re-index
        _remove_file_chunks(file_path)
        chunks = _chunk_for_file(file_path, content)
        for idx, (chunk, symbol) in enumerate(chunks):
            vec = embed(chunk)
            if vec is not None:
                documents.append(
                    MemoryDoc(path=file_path, chunk_id=idx, text=chunk, symbol=symbol)
                )
                vectors.append(vec)
        changed_count += 1

    # Remove chunks for deleted files
    current_paths = set(new_hashes.keys())
    for old_path in list(old_hashes.keys()):
        if old_path not in current_paths:
            _remove_file_chunks(old_path)
            changed_count += 1

    if changed_count > 0:
        _rebuild_lexical_stats()
        _query_cache.clear()
        _save_index(
            meta={
                "file_hashes": new_hashes,
                "chunking": "symbol_or_window_multilang",
                "extensions": sorted(SOURCE_EXTENSIONS),
            }
        )

    return changed_count


def _remove_file_chunks(file_path: str):
    """Remove all chunks for a specific file from the in-memory index."""
    indices_to_remove = [i for i, d in enumerate(documents) if d.path == file_path]
    for i in sorted(indices_to_remove, reverse=True):
        documents.pop(i)
        vectors.pop(i)


def _full_rebuild(project_root: str):
    """Full rebuild of the memory index (original logic)."""
    global _index_loaded

    documents.clear()
    vectors.clear()
    _query_cache.clear()

    files = get_source_files(project_root, extensions=SOURCE_EXTENSIONS)
    file_hashes = {}
    all_chunks_to_embed = []  # list of (path, idx, text, symbol)

    for file_path in files:
        content = load_file_content(file_path)
        if not content.strip():
            continue

        file_hashes[file_path] = _file_hash(content)
        chunks = _chunk_for_file(file_path, content)
        for idx, (chunk, symbol) in enumerate(chunks):
            all_chunks_to_embed.append((file_path, idx, chunk, symbol))

    print(
        f"[memory] embedding {len(all_chunks_to_embed)} chunks using parallel workers..."
    )

    def _embed_and_wrap(item):
        path, idx, text, symbol = item
        vec = embed(text)
        if vec is None:
            return None
        return MemoryDoc(path=path, chunk_id=idx, text=text, symbol=symbol), vec

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            tqdm(
                executor.map(_embed_and_wrap, all_chunks_to_embed),
                total=len(all_chunks_to_embed),
                desc="Embedding chunks",
            )
        )

    for res in results:
        if res:
            doc, vec = res
            documents.append(doc)
            vectors.append(vec)

    _rebuild_lexical_stats()
    _save_index(
        meta={
            "file_hashes": file_hashes,
            "chunking": "symbol_or_window_multilang",
            "extensions": sorted(SOURCE_EXTENSIONS),
        }
    )
    _index_loaded = True
    print(f"[memory] indexed {len(vectors)} chunks from {len(files)} files")


def _bm25_lite_score(query_terms: List[str], doc_text: str) -> float:
    if not query_terms:
        return 0.0

    tokens = _tokenize(doc_text)
    if not tokens:
        return 0.0

    tf = Counter(tokens)
    doc_len = max(1, len(tokens))
    n_docs = max(1, len(documents))

    k1 = 1.2
    b = 0.75

    score = 0.0
    for term in query_terms:
        term_tf = tf.get(term, 0)
        if term_tf == 0:
            continue
        df = _token_df.get(term, 0)
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        denom = term_tf + k1 * (1 - b + b * (doc_len / _avg_doc_len))
        score += idf * ((term_tf * (k1 + 1)) / max(1e-9, denom))

    return float(score)


def _project_token() -> str:
    try:
        dirty = changed_files()
    except Exception:
        dirty = []
    key = "|".join(sorted(dirty))[:4000]
    return hashlib.sha1((PROJECT_ROOT + "|" + key).encode("utf-8")).hexdigest()


def _extract_import_tokens(path: str, content: str) -> set[str]:
    tokens = set()
    if path.endswith(".py"):
        for m in re.finditer(
            r"^\s*(?:from|import)\s+([A-Za-z_][A-Za-z0-9_\.]*)",
            content,
            flags=re.MULTILINE,
        ):
            root = m.group(1).split(".")[0].lower()
            if len(root) > 2:
                tokens.add(root)
    elif path.endswith((".js", ".jsx", ".ts", ".tsx")):
        for m in re.finditer(
            r"(?:from\s+['\"]([^'\"]+)['\"]|require\(\s*['\"]([^'\"]+)['\"]\s*\))",
            content,
        ):
            val = (m.group(1) or m.group(2) or "").split("/")[-1].lower()
            val = val.replace(".js", "").replace(".ts", "")
            if re.match(r"[a-z_][a-z0-9_]{2,}", val):
                tokens.add(val)
    return tokens


def _build_hot_path_signals(limit: int = 20) -> tuple[set[str], set[str]]:
    hot = set()
    import_tokens = set()
    try:
        for rel in changed_files()[:limit]:
            norm = rel.replace("\\", "/").lower()
            hot.add(norm)
            abs_path = os.path.join(PROJECT_ROOT, rel)
            if os.path.exists(abs_path):
                import_tokens.update(
                    _extract_import_tokens(norm, load_file_content(abs_path))
                )
    except Exception:
        pass

    try:
        candidates = get_source_files(PROJECT_ROOT, extensions=SOURCE_EXTENSIONS)
        candidates.sort(
            key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0.0,
            reverse=True,
        )
        for abs_path in candidates[: min(12, limit)]:
            hot.add(os.path.relpath(abs_path, PROJECT_ROOT).replace("\\", "/").lower())
    except OSError:
        pass

    return hot, import_tokens


def _jaccard(a: str, b: str) -> float:
    """Similarité Jaccard sur les tokens — proxy léger pour MMR."""
    ta = set(_tokenize(a[:600]))
    tb = set(_tokenize(b[:600]))
    if not ta and not tb:
        return 0.0
    return len(ta & tb) / max(1, len(ta | tb))


def _mmr_rerank(
    scored: List[Tuple[float, "MemoryDoc"]],
    k: int,
    lambda_: float = 0.7,
) -> List[Tuple[float, "MemoryDoc"]]:
    """P2.3 — Maximal Marginal Relevance : équilibre pertinence et diversité.

    lambda_=1.0 → pure pertinence ; lambda_=0.0 → pure diversité.
    """
    if not scored or k <= 0:
        return []

    selected: List[Tuple[float, "MemoryDoc"]] = []
    remaining = list(scored)

    while remaining and len(selected) < k:
        if not selected:
            # Premier doc : le plus pertinent
            best = remaining[0]
        else:
            sel_texts = [d.text for _, d in selected]
            best = None
            best_mmr = float("-inf")
            for item in remaining:
                score, doc = item
                max_sim = max(_jaccard(doc.text, st) for st in sel_texts)
                mmr = lambda_ * score - (1 - lambda_) * max_sim
                if mmr > best_mmr:
                    best_mmr = mmr
                    best = item
            if best is None:
                break

        selected.append(best)
        remaining.remove(best)

    return selected


def _extract_explicit_file_refs(query: str) -> list[str]:
    """Extract file paths explicitly mentioned in a query (e.g. 'users/api.py')."""
    pattern = r"([A-Za-z0-9_./\\-]+\.(?:py|js|ts|jsx|tsx|html|css|json|yaml|yml|md|toml|sql|xml))"
    return list(dict.fromkeys(re.findall(pattern, query)))


def search_memory(query: str, k: int = 3):
    global _project_cache_token

    if not vectors:
        build_memory(PROJECT_ROOT)

    if not vectors:
        return []

    token = _project_token()
    if token != _project_cache_token:
        _query_cache.clear()
        _project_cache_token = token

    cache_key = f"{_project_cache_token}|{query}|{k}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # --- Explicit file reference detection ---
    # If the query mentions specific files, resolve them and normalize for boosting
    explicit_refs = _extract_explicit_file_refs(query)
    explicit_paths: set[str] = set()
    for ref in explicit_refs:
        ref_norm = ref.replace("\\", "/").lower()
        explicit_paths.add(ref_norm)
        # Try plural/singular variant (user/api.py -> users/api.py)
        parts = ref_norm.split("/")
        if parts:
            explicit_paths.add("/".join([parts[0] + "s"] + parts[1:]))
            explicit_paths.add("/".join([parts[0].rstrip("s")] + parts[1:]))

    q = embed(query)
    if q is None:
        return []

    semantic_scores = [float(np.dot(q, v)) for v in vectors]
    top_idx = np.argsort(semantic_scores)[-max(k * 6, k) :][::-1]

    query_terms = [t for t in _tokenize(query) if len(t) > 2]
    hot_paths, import_tokens = _build_hot_path_signals()

    # P3.5 — dependency graph boost: find files related to hot paths
    dep_related: set[str] = set()
    try:
        from agent.dependency_graph import get_related_files

        for hp in list(hot_paths)[:8]:
            for rel_file in get_related_files(hp, depth=1):
                dep_related.add(rel_file.lower())
    except Exception:
        pass

    reranked = []
    for i in top_idx:
        doc = documents[i]
        semantic = semantic_scores[i]
        lexical = _bm25_lite_score(query_terms, doc.text)
        rel = os.path.relpath(doc.path, PROJECT_ROOT).replace("\\", "/").lower()
        # Explicit file ref: very strong boost when the query names a specific file
        explicit_boost = 0.35 if any(rel.endswith(ep) for ep in explicit_paths) else 0.0
        path_boost = 0.05 if any(t in doc.path.lower() for t in query_terms) else 0.0
        hot_boost = 0.12 if rel in hot_paths else 0.0
        graph_boost = 0.06 if any(tok in rel for tok in import_tokens) else 0.0
        # P3.5 — Boost files in the dependency graph of recently changed files
        dep_boost = 0.08 if rel in dep_related else 0.0
        final_score = 0.78 * semantic + 0.22 * lexical + path_boost
        final_score += hot_boost + graph_boost + dep_boost + explicit_boost
        reranked.append((final_score, doc))

    # Also ensure explicitly referenced files appear even if not in top_idx
    if explicit_paths:
        seen_paths = {
            os.path.relpath(documents[i].path, PROJECT_ROOT).replace("\\", "/").lower()
            for i in top_idx
        }
        for i, doc in enumerate(documents):
            rel = os.path.relpath(doc.path, PROJECT_ROOT).replace("\\", "/").lower()
            if any(rel.endswith(ep) for ep in explicit_paths) and rel not in seen_paths:
                semantic = float(np.dot(q, vectors[i]))
                lexical = _bm25_lite_score(query_terms, doc.text)
                explicit_boost = 0.35
                final_score = 0.78 * semantic + 0.22 * lexical + explicit_boost
                reranked.append((final_score, doc))
                seen_paths.add(rel)

    reranked.sort(key=lambda x: x[0], reverse=True)

    # P2.3 — MMR : diversifie les résultats
    reranked_mmr = _mmr_rerank(reranked, k=k)
    out = [(doc.path, doc.text) for _, doc in reranked_mmr]

    # --- Ensure explicitly referenced files appear in results ---
    # If the query names specific files, force their best chunks to the front
    if explicit_paths:
        existing_paths = {
            os.path.relpath(p, PROJECT_ROOT).replace("\\", "/").lower() for p, _ in out
        }
        explicit_docs = []
        for i, doc in enumerate(documents):
            rel = os.path.relpath(doc.path, PROJECT_ROOT).replace("\\", "/").lower()
            if (
                any(rel.endswith(ep) for ep in explicit_paths)
                and rel not in existing_paths
            ):
                explicit_docs.append((doc.path, doc.text))
                existing_paths.add(rel)
        # Prepend explicit file chunks, push others down
        if explicit_docs:
            out = explicit_docs[:k] + out
            out = out[:k]

    strategy_docs = _load_fix_strategy_docs(limit=2)
    if strategy_docs:
        out.extend(strategy_docs)
        out = out[:k]
    _cache_put(cache_key, out)
    return out


def index_file_in_session(path: str, content: str) -> int:
    """Indexe immédiatement un fichier créé/modifié en session sans rebuild complet.

    Les chunks sont ajoutés aux listes globales `documents` et `vectors`,
    rendant le fichier trouvable par `search_memory` dans la même session.
    Retourne le nombre de chunks indexés.
    """
    if not content.strip():
        return 0

    chunks = _chunk_for_file(path, content)
    added = 0
    for idx, (chunk, symbol) in enumerate(chunks):
        vec = embed(chunk)
        if vec is None:
            continue
        documents.append(MemoryDoc(path=path, chunk_id=idx, text=chunk, symbol=symbol))
        vectors.append(vec)
        added += 1

    if added > 0:
        _rebuild_lexical_stats()
        _query_cache.clear()  # invalider le cache pour forcer re-recherche

    return added


def budget_context(query: str, k: int = 6, budget_chars: int = CONTEXT_BUDGET_CHARS):
    candidates = search_memory(query, k=max(k * 2, CONTEXT_MAX_CHUNKS * 2))
    if not candidates:
        return "No indexed context found."

    per_file_count: Dict[str, int] = {}
    chosen = []
    chosen_files = set()
    total = 0

    for path, text in candidates:
        if len(chosen) >= CONTEXT_MAX_CHUNKS:
            break

        per_file_count[path] = per_file_count.get(path, 0)
        if per_file_count[path] >= CONTEXT_MAX_PER_FILE:
            continue

        rel = os.path.relpath(path, PROJECT_ROOT)
        header = f"FILE: {rel}\n"
        body_budget = max(300, budget_chars // max(1, CONTEXT_MAX_CHUNKS) - len(header))
        body = text[:body_budget].strip()
        block = header + body

        if total + len(block) > budget_chars:
            continue

        chosen.append(block)
        chosen_files.add(rel)
        per_file_count[path] += 1
        total += len(block)

    omitted_files = sorted(
        {os.path.relpath(p, PROJECT_ROOT) for p, _ in candidates} - chosen_files
    )

    summary_tail = ""
    if omitted_files:
        summary_tail = "\n\n[context] omitted files: " + ", ".join(omitted_files[:6])

    return "\n\n".join(chosen) + summary_tail


# P2.1 — Mots-clés pour calibrer la complexité du goal
_COMPLEX_KW = {
    "architecture",
    "refactor",
    "migrate",
    "redesign",
    "restructure",
    "all files",
    "entire",
    "system",
    "global",
    "tous les fichiers",
    "architecture",
    "refactoriser",
    "migrer",
    "restructurer",
    "système",
}
_SIMPLE_KW = {
    "fix",
    "typo",
    "rename",
    "import",
    "correct",
    "add line",
    "corriger",
    "typo",
    "renommer",
    "ajouter",
    "ligne",
}


def _estimate_project_scale() -> str:
    """P3.5 — Estimate project scale: small / medium / large."""
    try:
        files = get_source_files(PROJECT_ROOT, extensions=SOURCE_EXTENSIONS)
        n = len(files)
    except Exception:
        return "medium"
    if n > 200:
        return "large"
    if n > 50:
        return "medium"
    return "small"


def summarize_module(file_path: str, max_chars: int = 800) -> str:
    """P3.5 — Generate a compact summary of a large module.

    Extracts: imports, class names, function signatures (first line of each).
    Used when the full file is too large for the context window.
    """
    try:
        content = load_file_content(file_path)
    except Exception:
        return ""
    if not content.strip():
        return ""

    lines = content.splitlines()
    parts = []

    # Imports block
    imports = [ln for ln in lines[:50] if ln.strip().startswith(("import ", "from "))]
    if imports:
        parts.append("IMPORTS: " + "; ".join(ln.strip() for ln in imports[:8]))

    # Class/function signatures
    if file_path.endswith(".py"):
        import ast

        try:
            tree = ast.parse(content)
            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    bases = ", ".join(
                        getattr(b, "id", getattr(b, "attr", "?")) for b in node.bases
                    )
                    methods = [
                        n.name
                        for n in node.body
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    parts.append(f"class {node.name}({bases}): methods={methods[:8]}")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sig = (
                        lines[node.lineno - 1].strip()
                        if node.lineno <= len(lines)
                        else node.name
                    )
                    parts.append(sig)
        except SyntaxError:
            pass

    summary = "\n".join(parts)
    return summary[:max_chars]


def adaptive_budget_context(query: str, goal: str = "") -> str:
    """P3.5 — Adapte le budget de contexte selon complexité ET taille du projet.

    - Goal complexe (architecture, refactor…) → k=12, budget=12 000 chars
    - Goal simple (fix, typo…)               → k=4,  budget=3 000 chars
    - Sinon                                  → k=6,  budget=CONTEXT_BUDGET_CHARS

    Sur gros projets (>200 fichiers), augmente le k et le budget de 50%.
    """
    text = (goal or query).lower()
    if any(kw in text for kw in _COMPLEX_KW):
        k, budget = 12, 12_000
    elif any(kw in text for kw in _SIMPLE_KW):
        k, budget = 4, 3_000
    else:
        k, budget = 6, CONTEXT_BUDGET_CHARS

    # P3.5 — Scale up for large projects
    scale = _estimate_project_scale()
    if scale == "large":
        k = int(k * 1.5)
        budget = int(budget * 1.5)

    return budget_context(query, k=k, budget_chars=budget)
