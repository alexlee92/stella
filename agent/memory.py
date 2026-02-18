import hashlib
import json
import os
import re
from collections import OrderedDict, Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import requests

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
from agent.project_scan import get_python_files, load_file_content


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


def _ensure_index_dir() -> str:
    index_dir = os.path.join(PROJECT_ROOT, MEMORY_INDEX_DIR)
    os.makedirs(index_dir, exist_ok=True)
    return index_dir


def _index_files() -> tuple[str, str]:
    d = _ensure_index_dir()
    return os.path.join(d, "docs.json"), os.path.join(d, "vectors.npy")


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


def _chunk_for_file(path: str, content: str):
    if path.endswith(".py"):
        symbol_chunks = _chunk_python_by_symbols(content)
        if symbol_chunks:
            return symbol_chunks
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


def build_memory(project_root: str = PROJECT_ROOT, force_rebuild: bool = False):
    global _index_loaded

    if _index_loaded and not force_rebuild:
        print(f"[memory] loaded in-process index with {len(vectors)} vectors")
        return

    if not force_rebuild and _load_index():
        print(f"[memory] loaded persisted index with {len(vectors)} vectors")
        return

    documents.clear()
    vectors.clear()
    _query_cache.clear()

    files = get_python_files(project_root)
    file_hashes = {}

    for file_path in files:
        content = load_file_content(file_path)
        if not content.strip():
            continue

        file_hashes[file_path] = _file_hash(content)
        chunks = _chunk_for_file(file_path, content)
        for idx, (chunk, symbol) in enumerate(chunks):
            vec = embed(chunk)
            if vec is None:
                continue
            documents.append(
                MemoryDoc(path=file_path, chunk_id=idx, text=chunk, symbol=symbol)
            )
            vectors.append(vec)

    _rebuild_lexical_stats()
    _save_index(meta={"file_hashes": file_hashes, "chunking": "symbol_or_window"})
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


def search_memory(query: str, k: int = 3):
    if not vectors:
        build_memory(PROJECT_ROOT)

    if not vectors:
        return []

    cache_key = f"{query}|{k}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    q = embed(query)
    if q is None:
        return []

    semantic_scores = [float(np.dot(q, v)) for v in vectors]
    top_idx = np.argsort(semantic_scores)[-max(k * 6, k) :][::-1]

    query_terms = [t for t in _tokenize(query) if len(t) > 2]
    reranked = []
    for i in top_idx:
        doc = documents[i]
        semantic = semantic_scores[i]
        lexical = _bm25_lite_score(query_terms, doc.text)
        path_boost = 0.05 if any(t in doc.path.lower() for t in query_terms) else 0.0
        final_score = 0.78 * semantic + 0.22 * lexical + path_boost
        reranked.append((final_score, doc))

    reranked.sort(key=lambda x: x[0], reverse=True)

    out = [(doc.path, doc.text) for _, doc in reranked[:k]]
    _cache_put(cache_key, out)
    return out


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
