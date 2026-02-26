import os
from types import SimpleNamespace
from unittest.mock import patch


class _DummyLogger:
    def log(self, *_args, **_kwargs):
        return None

    def log_failure(self, *_args, **_kwargs):
        return None


class TestCriticModule:
    def test_parse_error_auto_approves_original_decision(self):
        from agent.critic import Critic

        original = {"action": "list_files", "reason": "x", "args": {"limit": 10}}
        ag = SimpleNamespace(
            _llm_fn=lambda *_a, **_k: {
                "_error_type": "parse",
                "_parse_meta": {"error_class": "json_decode_error"},
            },
            event_logger=_DummyLogger(),
        )
        out = Critic(ag).critique("goal", original)
        assert out == original

    def test_reject_with_valid_patch_returns_patched(self):
        from agent.critic import Critic

        original = {"action": "list_files", "reason": "x", "args": {"limit": 10}}
        critique_payload = {
            "approve": False,
            "reason": "bad",
            "patched_decision": {
                "action": "read_file",
                "reason": "inspect",
                "args": {"path": "agent/config.py"},
            },
        }
        ag = SimpleNamespace(
            _llm_fn=lambda *_a, **_k: critique_payload,
            event_logger=_DummyLogger(),
        )
        out = Critic(ag).critique("goal", original)
        assert out["action"] == "read_file"
        assert out["args"]["path"] == "agent/config.py"


class TestReplanModule:
    def test_extract_error_paths_deduplicates(self):
        from agent.replan import ReplanEngine

        ag = SimpleNamespace()
        rp = ReplanEngine(ag)
        text = "agent/a.py:10 error\nagent/a.py:20\nagent/b.py:3"
        assert rp.extract_error_paths(text) == ["agent/a.py", "agent/b.py"]

    def test_enqueue_replan_after_failure_auto_apply(self):
        from agent.replan import ReplanEngine

        ag = SimpleNamespace(
            _replan_attempts=0,
            _max_replan_attempts=3,
            _forced_decisions=[],
            event_logger=_DummyLogger(),
        )
        rp = ReplanEngine(ag)
        ok = rp.enqueue_replan_after_failure(
            "tests", "tests/test_api.py:42 assert", auto_apply=True
        )
        assert ok is True
        assert ag._replan_attempts == 1
        actions = [d["action"] for d in ag._forced_decisions]
        assert actions[:3] == ["propose_edit", "apply_edit", "run_tests"]


class TestLoopControllerModule:
    def test_decision_loop_detected_for_repeated_signature(self):
        """R2: les non-read-only conservent la logique originale (≥2 dans les 4 derniers)."""
        from agent.loop_controller import LoopController

        ag = SimpleNamespace(_decision_signatures=[], _outcome_signatures=[])
        lc = LoopController(ag)
        # web_search n'est pas read-only → logique originale : 2 occurrences dans les 4 derniers
        sig = lc.signature("web_search", {"query": "fastapi", "limit": 3})
        ag._decision_signatures = [sig, sig, "x", "y"]
        assert (
            lc.decision_loop_detected("web_search", {"query": "fastapi", "limit": 3})
            is True
        )

    def test_read_only_consecutive_is_loop(self):
        """R2: deux read_file consécutifs identiques = vraie boucle."""
        from agent.loop_controller import LoopController

        ag = SimpleNamespace(_decision_signatures=[], _outcome_signatures=[])
        lc = LoopController(ag)
        sig = lc.signature("read_file", {"path": "config.py"})
        # Dernier élément est le même sig → consécutif
        ag._decision_signatures = ["x", sig]
        assert lc.decision_loop_detected("read_file", {"path": "config.py"}) is True

    def test_read_only_non_consecutive_single_repeat_not_loop(self):
        """R2: read A → autre chose → read A = PAS une boucle (faux positif corrigé)."""
        from agent.loop_controller import LoopController

        ag = SimpleNamespace(_decision_signatures=[], _outcome_signatures=[])
        lc = LoopController(ag)
        sig = lc.signature("read_file", {"path": "config.py"})
        other = lc.signature(
            "propose_edit", {"path": "config.py", "instruction": "fix"}
        )
        # history: [sig, sig, other, ?] — non-consécutif, seulement 2 occurrences
        ag._decision_signatures = [sig, sig, other]
        assert lc.decision_loop_detected("read_file", {"path": "config.py"}) is False

    def test_read_only_three_occurrences_is_loop(self):
        """R2: 3+ occurrences non-consécutives d'un read = boucle."""
        from agent.loop_controller import LoopController

        ag = SimpleNamespace(_decision_signatures=[], _outcome_signatures=[])
        lc = LoopController(ag)
        sig = lc.signature("read_file", {"path": "models.py"})
        other = lc.signature("search_code", {"pattern": "class User"})
        ag._decision_signatures = [sig, other, sig, other, sig]
        assert lc.decision_loop_detected("read_file", {"path": "models.py"}) is True

    def test_is_stalled_step(self):
        from agent.loop_controller import LoopController

        ag = SimpleNamespace(_decision_signatures=[], _outcome_signatures=[])
        lc = LoopController(ag)
        assert lc.is_stalled_step("search_code", "No matches") is True
        assert lc.is_stalled_step("read_file", "[error] file not found") is True
        assert lc.is_stalled_step("read_file", "some content") is False


class TestMcpClientModule:
    def test_format_mcp_result(self):
        from agent.mcp_client import format_mcp_result

        assert "[mcp error]" in format_mcp_result({"ok": False, "error": "boom"})
        assert format_mcp_result({"ok": True, "result": "hello"}) == "hello"

    def test_mcp_call_server_not_found(self):
        from agent.mcp_client import mcp_call

        with patch("agent.mcp_client._find_server", return_value=None):
            out = mcp_call("missing", "tools/list")
        assert out["ok"] is False
        assert "not found" in out["error"].lower()


class TestMigrationModule:
    def test_generate_migration_not_configured(self):
        from agent.migration_generator import generate_migration

        with patch(
            "agent.migration_generator.is_alembic_configured", return_value=False
        ):
            out = generate_migration("x")
        assert out["ok"] is False
        assert "alembic" in out["output"].lower()

    def test_apply_migration_not_configured(self):
        from agent.migration_generator import apply_migration

        with patch(
            "agent.migration_generator.is_alembic_configured", return_value=False
        ):
            out = apply_migration("head")
        assert out["ok"] is False

    def test_migration_helper_suggestions_alembic(self):
        from agent.migration_helper import suggest_migration_commands

        with patch(
            "agent.migration_helper.detect_migration_framework", return_value="alembic"
        ):
            cmds = suggest_migration_commands(["users/models.py"])
        assert any("alembic revision" in c for c in cmds)

    def test_migration_helper_coherence_warning(self):
        from agent.migration_helper import validate_model_migration_coherence

        warnings = validate_model_migration_coherence(["users/models.py"])
        assert warnings
        assert "no migration file" in warnings[0].lower()


class TestDoctorAndDevTaskModule:
    def test_format_doctor(self):
        from agent.doctor import format_doctor

        text = format_doctor(
            {
                "ok": 1,
                "total": 2,
                "checks": [{"name": "python", "ok": True, "details": "3.13"}],
            }
        )
        assert "Doctor: 1/2 checks passed" in text
        assert "[OK] python" in text

    def test_next_action_matrix(self):
        from agent.dev_task import _next_action

        assert "pr-ready" in _next_action(True, True)
        assert "--apply" in _next_action(True, False)
        assert "Refine goal" in _next_action(False, False)

    def test_run_dev_task_profile_standard(self):
        from agent.dev_task import run_dev_task

        with (
            patch("agent.dev_task.index_project"),
            patch("agent.dev_task.AutonomousAgent") as mock_agent_cls,
            patch("agent.dev_task.changed_files", return_value=["agent/x.py"]),
            patch("agent.dev_task.diff_summary", return_value="diff"),
            patch("agent.dev_task._write_run_summary", return_value=("a.json", "a.md")),
        ):
            mock_agent = mock_agent_cls.return_value
            mock_agent.run.return_value = "summary"
            out = run_dev_task("improve x", profile="standard")

        assert out["status"] == "changes_ready"
        assert out["options"]["auto_apply"] is True
        assert out["options"]["with_tests"] is True


class TestGenerationQualityModule:
    def test_assess_generated_files_scores_python(self):
        from agent.generation_quality import assess_generated_files
        from agent.config import PROJECT_ROOT
        import shutil

        rel_dir = os.path.join("tests", "__tmp_gen_quality")
        abs_dir = os.path.join(PROJECT_ROOT, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)
        src = os.path.join(abs_dir, "sample.py")
        testf = os.path.join(abs_dir, "test_sample.py")
        try:
            with open(src, "w", encoding="utf-8") as f:
                f.write(
                    '"""module"""\n\n'
                    "def add(a: int, b: int) -> int:\n"
                    "    return a + b\n"
                )
            with open(testf, "w", encoding="utf-8") as f:
                f.write("def test_add():\n    assert True\n")
            score = assess_generated_files(
                [f"{rel_dir}/sample.py", f"{rel_dir}/test_sample.py"]
            )
            assert score["python_files"] == 2
            assert score["syntax_valid_rate"] == 100.0
            assert score["score"] > 60.0
        finally:
            shutil.rmtree(abs_dir, ignore_errors=True)

    def test_assess_generated_files_handles_no_python(self):
        from agent.generation_quality import assess_generated_files

        score = assess_generated_files(["README.md"])
        assert score["python_files"] == 0
        assert score["score"] == 0.0


class TestEvalRunnerGenerationQuality:
    def test_run_eval_with_quality_threshold_override(self):
        from agent.eval_runner import run_eval
        import tempfile
        import json

        task = [
            {
                "name": "t1",
                "mode": "code_edit",
                "track": "code_edit",
                "prompt": "fix agent/tooling.py",
                "must_contain_any": ["Final"],
                "auto_apply": False,
                "fix_until_green": False,
                "with_tests": False,
                "max_steps": 1,
            }
        ]
        with tempfile.NamedTemporaryFile(
            suffix=".json", mode="w", encoding="utf-8", delete=False
        ) as f:
            json.dump(task, f)
            tmp_path = f.name
        try:
            with (
                patch("agent.eval_runner.index_project"),
                patch("agent.eval_runner.AutonomousAgent") as mock_agent_cls,
                patch(
                    "agent.eval_runner._run_pr_ready_probe", return_value={"ok": True}
                ),
                patch(
                    "agent.eval_runner.changed_files",
                    side_effect=[[], ["agent/tooling.py"]],
                ),
                patch(
                    "agent.eval_runner.assess_generated_files",
                    return_value={"score": 10.0},
                ),
            ):
                mock_agent = mock_agent_cls.return_value
                mock_agent.run.return_value = (
                    "Final\nDecisions:\n1. [propose_edit] x -> agent/tooling.py"
                )
                # Very high override -> fail
                report = run_eval(tasks_file=tmp_path, min_generation_quality=90.0)
            assert report["results"][0]["passed"] is False
            assert report["results"][0]["generation_quality"]["score"] == 10.0
        finally:
            os.unlink(tmp_path)

    def test_parse_event_time_upgrades_naive_to_utc(self):
        from agent.eval_runner import _parse_event_time

        dt = _parse_event_time({"timestamp": "2026-02-25T12:00:00"})
        assert dt is not None
        assert dt.tzinfo is not None
