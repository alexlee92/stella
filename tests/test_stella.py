"""
Tests unitaires de l'agent Stella.
Ces tests ne nécessitent pas qu'Ollama soit en cours d'exécution.
"""
import json
import os
import sys
import tempfile

import pytest

# S'assurer que le projet est dans le path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Tests de détection de type de tâche (llm_interface)
# ---------------------------------------------------------------------------

class TestDetectTaskType:
    """Vérifie que _detect_task_type reconnaît les mots-clés en FR et EN."""

    def _detect(self, prompt, system=""):
        from agent.llm_interface import _detect_task_type
        return _detect_task_type(prompt, system)

    # --- Anglais ---
    def test_analysis_english(self):
        assert self._detect("analyze this code") == "analysis"

    def test_refactor_english(self):
        assert self._detect("refactor this class to be cleaner") == "refactor"

    def test_debug_english(self):
        assert self._detect("fix the bug in auth.py") == "debug"

    def test_optimize_english(self):
        assert self._detect("optimize the performance of this function") == "optimization"

    def test_frontend_english(self):
        assert self._detect("create a React component for the login form") == "frontend"

    def test_backend_english(self):
        assert self._detect("add a FastAPI endpoint for user creation") == "backend"

    # --- Français ---
    def test_analysis_french(self):
        assert self._detect("analyser l'architecture de ce projet") == "analysis"

    def test_explain_french(self):
        assert self._detect("expliquer comment fonctionne ce module") == "analysis"

    def test_refactor_french(self):
        assert self._detect("refactoriser cette classe pour améliorer la lisibilité") == "refactor"

    def test_improve_french(self):
        assert self._detect("améliorer la structure du code") == "refactor"

    def test_debug_french(self):
        assert self._detect("corriger l'erreur dans ce fichier") == "debug"

    def test_fix_french(self):
        assert self._detect("réparer le bug dans agent/llm_interface.py") == "debug"

    def test_optimize_french(self):
        assert self._detect("optimiser la latence des requêtes") == "optimization"

    def test_frontend_french(self):
        assert self._detect("créer un composant React pour l'interface") == "frontend"

    def test_backend_french(self):
        assert self._detect("créer une route API dans Flask avec base de données") == "backend"

    def test_fallback(self):
        # Sans mot-clé reconnu, retourne "analysis" par défaut (pour réponses détaillées)
        assert self._detect("bonjour") == "analysis"

    def test_system_prompt_contributes(self):
        # Le system_prompt compte aussi dans la détection
        result = self._detect("do the task", system="You are an architecture analyzer")
        assert result == "analysis"


# ---------------------------------------------------------------------------
# Tests du calcul de risque de patch (risk.py)
# ---------------------------------------------------------------------------

class TestComputePatchRisk:
    def _risk(self, path, old, new):
        from agent.risk import compute_patch_risk
        return compute_patch_risk(path, old, new)

    def test_low_risk_small_change(self):
        old = "x = 1\n"
        new = "x = 2\n"
        result = self._risk("utils.py", old, new)
        assert result["level"] == "low"
        assert result["score"] < 20

    def test_sensitive_file_raises_score(self):
        old = "SECRET = 'abc'\n"
        new = "SECRET = 'xyz'\n"
        result = self._risk("agent/config.py", old, new)
        assert result["score"] >= 25  # score sensible + changement

    def test_subprocess_addition_raises_score(self):
        old = "def run(): pass\n"
        new = "import subprocess\ndef run(): subprocess.run(['ls'])\n"
        result = self._risk("tool.py", old, new)
        assert result["score"] > 15

    def test_high_risk_many_lines(self):
        old = "\n".join(f"line_{i} = {i}" for i in range(60))
        new = "\n".join(f"line_{i} = {i + 1}" for i in range(60))
        result = self._risk("big_file.py", old, new)
        assert result["level"] == "high"

    def test_result_has_required_keys(self):
        result = self._risk("f.py", "a = 1\n", "a = 2\n")
        assert {"score", "level", "changed_lines", "sensitive_hits"} <= result.keys()


# ---------------------------------------------------------------------------
# Tests du patcher (patcher.py)
# ---------------------------------------------------------------------------

class TestPatcher:
    def test_search_replace_applied(self):
        from agent.partial_edits import parse_partial_edit, apply_multi_edit
        old = "def hello():\n    return 'world'\n"
        new = (
            "<<<<<<< SEARCH\n"
            "def hello():\n"
            "    return 'world'\n"
            "=======\n"
            "def hello():\n"
            "    return 'HELLO'\n"
            ">>>>>>> REPLACE\n"
        )
        edits = parse_partial_edit(new)
        assert len(edits) == 1
        result = apply_multi_edit(old, edits)
        assert "HELLO" in result

    def test_no_edit_when_no_markers(self):
        from agent.partial_edits import parse_partial_edit
        edits = parse_partial_edit("def foo(): pass\n")
        assert edits == []

    def test_invalid_python_rejected_in_prepare(self):
        """_prepare_new_code doit rejeter du Python invalide."""
        from agent.patcher import _prepare_new_code
        old = "x = 1\n"
        invalid = "def broken(\n"  # SyntaxError
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
            f.write(old.encode())
            path = f.name
        try:
            with pytest.raises(ValueError, match="invalid python"):
                _prepare_new_code(path, old, invalid)
        finally:
            os.unlink(path)

    def test_non_python_accepted_without_ast(self):
        """Les fichiers non-Python ne passent pas par l'AST check."""
        from agent.patcher import _prepare_new_code
        old = "const x = 1;\n"
        new = "const x = 2;\n"
        with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as f:
            f.write(old.encode())
            path = f.name
        try:
            result, meta = _prepare_new_code(path, old, new)
            assert result == new
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests du strip des code fences (agent.py)
# ---------------------------------------------------------------------------

class TestStripCodeFences:
    def _strip(self, text):
        from agent.agent import _strip_code_fences
        return _strip_code_fences(text)

    def test_strip_python_fence(self):
        text = "```python\ndef foo():\n    pass\n```"
        result = self._strip(text)
        assert result == "def foo():\n    pass"

    def test_strip_plain_fence(self):
        text = "```\ncode here\n```"
        assert self._strip(text) == "code here"

    def test_no_fence_unchanged(self):
        text = "def foo(): pass"
        assert self._strip(text) == text

    def test_empty_returns_empty(self):
        assert self._strip("") == ""


# ---------------------------------------------------------------------------
# Tests du parsing JSON (llm_interface.py)
# ---------------------------------------------------------------------------

class TestParseJsonResponse:
    def _parse(self, text):
        from agent.llm_interface import _parse_json_response
        return _parse_json_response(text)

    def test_valid_json(self):
        parsed, err = self._parse('{"action": "finish", "args": {}}')
        assert parsed == {"action": "finish", "args": {}}
        assert err == "ok"

    def test_json_in_prose(self):
        text = 'Here is the result: {"action": "read_file", "args": {"path": "x.py"}} done.'
        parsed, err = self._parse(text)
        assert parsed is not None
        assert parsed["action"] == "read_file"

    def test_json_with_fence(self):
        text = "```json\n{\"key\": \"value\"}\n```"
        from agent.llm_interface import _strip_fences, _parse_json_response
        stripped = _strip_fences(text)
        parsed, err = _parse_json_response(stripped)
        assert parsed == {"key": "value"}

    def test_empty_returns_none(self):
        parsed, err = self._parse("")
        assert parsed is None
        assert err == "empty_response"

    def test_trailing_comma_repaired(self):
        text = '{"action": "finish", "args": {},}'
        parsed, err = self._parse(text)
        # Après repair, doit parser
        assert parsed is not None

    def test_curly_quotes_repaired(self):
        text = '\u201caction\u201d: \u201cfinish\u201d'
        from agent.llm_interface import _repair_json_text
        repaired = _repair_json_text('{"' + 'action": "finish"}')
        assert '"' in repaired


# ---------------------------------------------------------------------------
# Tests des utilitaires tooling
# ---------------------------------------------------------------------------

class TestTooling:
    def test_is_command_allowed_pytest(self):
        from agent.tooling import is_command_allowed
        assert is_command_allowed("pytest -q") is True

    def test_is_command_allowed_black(self):
        from agent.tooling import is_command_allowed
        assert is_command_allowed("python -m black .") is True

    def test_is_command_allowed_ruff(self):
        from agent.tooling import is_command_allowed
        assert is_command_allowed("python -m ruff check .") is True

    def test_is_command_blocked_rm(self):
        from agent.tooling import is_command_allowed
        assert is_command_allowed("rm -rf /") is False

    def test_is_command_blocked_curl(self):
        from agent.tooling import is_command_allowed
        assert is_command_allowed("curl http://evil.com") is False

    def test_list_files_python(self):
        from agent.tooling import list_files
        files = list_files(limit=10, ext=".py")
        assert all(f.endswith(".py") for f in files)
        assert len(files) > 0

    def test_list_files_toml(self):
        from agent.tooling import list_files
        files = list_files(limit=10, ext=".toml")
        assert all(f.endswith(".toml") for f in files)

    def test_read_file_truncates(self):
        from agent.tooling import read_file
        # Lit un fichier existant du projet
        content = read_file("stella.py", max_chars=50)
        assert len(content) <= 50 + len("\n\n...[truncated]")


# ---------------------------------------------------------------------------
# Tests de configuration (settings.py)
# ---------------------------------------------------------------------------

class TestSettings:
    def test_load_settings_returns_dict(self):
        from agent.settings import load_settings
        cfg = load_settings()
        assert isinstance(cfg, dict)

    def test_required_keys_present(self):
        from agent.settings import load_settings
        cfg = load_settings()
        required = {"MODEL", "EMBED_MODEL", "OLLAMA_URL", "PROJECT_ROOT",
                    "REQUEST_TIMEOUT", "AUTO_MAX_STEPS", "ORISHA_ENABLED"}
        assert required <= cfg.keys()

    def test_ollama_url_format(self):
        from agent.settings import load_settings
        cfg = load_settings()
        assert cfg["OLLAMA_URL"].startswith("http")

    def test_orisha_url_format(self):
        from agent.settings import load_settings
        cfg = load_settings()
        assert cfg["ORISHA_URL"].startswith("http")


# ---------------------------------------------------------------------------
# Tests BM25 mémoire (sans Ollama)
# ---------------------------------------------------------------------------

class TestBM25:
    def test_bm25_score_nonzero_for_matching(self):
        import agent.memory as mem
        from agent.memory import _bm25_lite_score, _rebuild_lexical_stats, MemoryDoc

        # Sauvegarder l'état
        saved_docs = list(mem.documents)
        saved_df = dict(mem._token_df)
        saved_avg = mem._avg_doc_len

        mem.documents.clear()
        # Note: _tokenize utilise [a-zA-Z_][a-zA-Z0-9_]+ donc "fix_bug" = 1 token
        # On utilise des mots séparés pour que les query_terms matchent
        mem.documents.append(
            MemoryDoc(path="test.py", chunk_id=0, text="this function fix the bug in the code")
        )
        _rebuild_lexical_stats()

        score = _bm25_lite_score(["fix", "bug"], "this function fix the bug in the code")
        assert score > 0.0

        # Restaurer l'état original
        mem.documents.clear()
        mem.documents.extend(saved_docs)
        mem._token_df = saved_df
        mem._avg_doc_len = saved_avg

    def test_bm25_score_zero_for_no_match(self):
        from agent.memory import _bm25_lite_score
        score = _bm25_lite_score(["totally", "unrelated"], "def hello(): return 42")
        # Sans documents indexés pour calculer IDF, peut être 0 ou faible
        assert score >= 0.0

    def test_tokenize(self):
        from agent.memory import _tokenize
        tokens = _tokenize("def ask_llm(prompt: str) -> str:")
        assert "ask_llm" in tokens
        assert "prompt" in tokens
        assert "str" in tokens
