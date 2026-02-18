import os
import tomllib


def _read_toml(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("utf-8-sig")
    return tomllib.loads(text)


def _get(cfg: dict, section: str, key: str, default):
    return cfg.get(section, {}).get(key, default)


def load_settings() -> dict:
    root = os.path.abspath(os.getcwd())
    file_cfg = _read_toml(os.path.join(root, "settings.toml"))

    project_root = os.path.abspath(
        os.getenv("PROJECT_ROOT", _get(file_cfg, "project", "root", root))
    )
    base_url = os.getenv(
        "OLLAMA_BASE_URL",
        _get(file_cfg, "ollama", "base_url", "http://localhost:11434"),
    )

    settings = {
        "MODEL": os.getenv(
            "MODEL", _get(file_cfg, "models", "main", "deepseek-coder:6.7b")
        ),
        "EMBED_MODEL": os.getenv(
            "EMBED_MODEL", _get(file_cfg, "models", "embed", "nomic-embed-text")
        ),
        "PROJECT_ROOT": project_root,
        "MAX_FILE_SIZE": int(
            os.getenv(
                "MAX_FILE_SIZE", _get(file_cfg, "project", "max_file_size", 100000)
            )
        ),
        "OLLAMA_BASE_URL": base_url,
        "OLLAMA_URL": f"{base_url}/api/generate",
        "OLLAMA_EMBED_URL": f"{base_url}/api/embeddings",
        "REQUEST_TIMEOUT": int(
            os.getenv(
                "REQUEST_TIMEOUT", _get(file_cfg, "ollama", "request_timeout", 120)
            )
        ),
        "TOP_K_RESULTS": int(
            os.getenv("TOP_K_RESULTS", _get(file_cfg, "agent", "top_k_results", 4))
        ),
        "AUTO_MAX_STEPS": int(
            os.getenv("AUTO_MAX_STEPS", _get(file_cfg, "agent", "auto_max_steps", 8))
        ),
        "AUTO_TEST_COMMAND": os.getenv(
            "AUTO_TEST_COMMAND",
            _get(file_cfg, "agent", "auto_test_command", "pytest -q"),
        ),
        "DRY_RUN": str(
            os.getenv("DRY_RUN", _get(file_cfg, "agent", "dry_run", False))
        ).lower()
        in {"1", "true", "yes"},
        "MAX_RETRIES_JSON": int(
            os.getenv(
                "MAX_RETRIES_JSON", _get(file_cfg, "agent", "max_retries_json", 3)
            )
        ),
        "CHAT_HISTORY_PATH": os.getenv(
            "CHAT_HISTORY_PATH",
            _get(file_cfg, "session", "history_path", ".stella/session_history.jsonl"),
        ),
        "EVENT_LOG_PATH": os.getenv(
            "EVENT_LOG_PATH",
            _get(file_cfg, "session", "log_path", ".stella/agent_events.jsonl"),
        ),
        "MEMORY_INDEX_DIR": os.getenv(
            "MEMORY_INDEX_DIR", _get(file_cfg, "memory", "index_dir", ".stella/memory")
        ),
        "CHUNK_SIZE": int(
            os.getenv("CHUNK_SIZE", _get(file_cfg, "memory", "chunk_size", 1200))
        ),
        "CHUNK_OVERLAP": int(
            os.getenv("CHUNK_OVERLAP", _get(file_cfg, "memory", "chunk_overlap", 160))
        ),
        "MAX_CHUNKS_PER_FILE": int(
            os.getenv(
                "MAX_CHUNKS_PER_FILE",
                _get(file_cfg, "memory", "max_chunks_per_file", 25),
            )
        ),
        "CONTEXT_BUDGET_CHARS": int(
            os.getenv(
                "CONTEXT_BUDGET_CHARS",
                _get(file_cfg, "memory", "context_budget_chars", 4500),
            )
        ),
        "CONTEXT_MAX_CHUNKS": int(
            os.getenv(
                "CONTEXT_MAX_CHUNKS", _get(file_cfg, "memory", "context_max_chunks", 6)
            )
        ),
        "CONTEXT_MAX_PER_FILE": int(
            os.getenv(
                "CONTEXT_MAX_PER_FILE",
                _get(file_cfg, "memory", "context_max_per_file", 2),
            )
        ),
        "MEMORY_QUERY_CACHE_SIZE": int(
            os.getenv(
                "MEMORY_QUERY_CACHE_SIZE",
                _get(file_cfg, "memory", "query_cache_size", 128),
            )
        ),
        "FORMAT_COMMAND": os.getenv(
            "FORMAT_COMMAND",
            _get(file_cfg, "quality", "format_command", "python -m black ."),
        ),
        "LINT_COMMAND": os.getenv(
            "LINT_COMMAND",
            _get(file_cfg, "quality", "lint_command", "python -m ruff check ."),
        ),
        "TEST_COMMAND": os.getenv(
            "TEST_COMMAND", _get(file_cfg, "quality", "test_command", "pytest -q")
        ),
    }

    return settings
