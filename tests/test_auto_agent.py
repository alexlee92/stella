"""
P2.4 — Tests d'intégration pour la boucle agent autonome.

Teste la logique de normalisation, auto-correction, détection de boucle,
et exécution de la boucle avec des décisions pré-scriptées.
"""

import os
import shutil
from unittest.mock import patch

from agent.decision_normalizer import (
    normalize_decision,
    normalize_critique,
    coerce_decision,
    infer_fallback_action,
    autocorrect_decision_schema,
    extract_target_file_from_goal,
    is_code_edit_goal,
    is_git_goal,
)

# ---------------------------------------------------------------------------
# Decision normalizer tests
# ---------------------------------------------------------------------------


class TestNormalizeDecision:
    def test_standard_decision(self):
        d = {"action": "read_file", "reason": "test", "args": {"path": "foo.py"}}
        result = normalize_decision(d)
        assert result["action"] == "read_file"
        assert result["args"]["path"] == "foo.py"

    def test_alias_resolution(self):
        d = {"action": "read", "reason": "test", "args": {"path": "a.py"}}
        result = normalize_decision(d)
        assert result["action"] == "read_file"

    def test_alias_grep_to_search(self):
        d = {"action": "grep", "reason": "find", "args": {"pattern": "foo"}}
        result = normalize_decision(d)
        assert result["action"] == "search_code"

    def test_alias_done_to_finish(self):
        d = {"action": "done", "reason": "ok", "args": {"summary": "done"}}
        result = normalize_decision(d)
        assert result["action"] == "finish"

    def test_tool_key_fallback(self):
        d = {"tool": "search_code", "why": "test", "parameters": {"pattern": "foo"}}
        result = normalize_decision(d)
        assert result["action"] == "search_code"
        assert result["reason"] == "test"
        assert result["args"]["pattern"] == "foo"

    def test_non_dict_passthrough(self):
        assert normalize_decision("not a dict") == "not a dict"

    def test_limit_string_coercion(self):
        d = {"action": "list_files", "reason": "t", "args": {"limit": "10"}}
        result = normalize_decision(d)
        assert result["args"]["limit"] == 10

    def test_read_many_paths_string_to_list(self):
        d = {"action": "read_many", "reason": "t", "args": {"paths": "a.py"}}
        result = normalize_decision(d)
        assert result["args"]["paths"] == ["a.py"]

    def test_infer_action_from_args_shape(self):
        d = {
            "action": "unknown_stuff",
            "reason": "t",
            "args": {"path": "f.py", "instruction": "fix bug"},
        }
        result = normalize_decision(d)
        assert result["action"] == "propose_edit"

    def test_filter_disallowed_args(self):
        d = {
            "action": "read_file",
            "reason": "t",
            "args": {"path": "a.py", "extra": "bad"},
        }
        result = normalize_decision(d)
        assert "extra" not in result["args"]

    def test_root_args_merged(self):
        d = {"action": "read_file", "reason": "t", "path": "a.py"}
        result = normalize_decision(d)
        assert result["args"]["path"] == "a.py"

    def test_private_keys_preserved(self):
        d = {"action": "finish", "reason": "t", "args": {}, "_error_type": "parse"}
        result = normalize_decision(d)
        assert result["_error_type"] == "parse"


class TestNormalizeCritique:
    def test_standard_approve(self):
        c = {"approve": True, "reason": "ok"}
        result = normalize_critique(c)
        assert result["approve"] is True

    def test_string_approve(self):
        c = {"approve": "yes", "reason": "ok"}
        result = normalize_critique(c)
        assert result["approve"] is True

    def test_string_reject(self):
        c = {"approve": "false", "reason": "nope"}
        result = normalize_critique(c)
        assert result["approve"] is False

    def test_patched_decision_normalized(self):
        c = {
            "approve": False,
            "reason": "bad",
            "patched_decision": {
                "action": "read",
                "reason": "fix",
                "args": {"path": "a.py"},
            },
        }
        result = normalize_critique(c)
        assert result["patched_decision"]["action"] == "read_file"

    def test_non_dict_passthrough(self):
        assert normalize_critique(42) == 42


class TestCoerceDecision:
    def test_git_action_on_non_git_goal(self):
        d = {"action": "git_commit", "reason": "t", "args": {"message": "fix"}}
        result = coerce_decision("fix the bug in utils.py", d)
        assert result["action"] == "search_code"

    def test_git_action_on_git_goal(self):
        d = {"action": "git_commit", "reason": "t", "args": {"message": "fix"}}
        result = coerce_decision("commit all changes", d)
        assert result["action"] == "git_commit"


class TestGoalHelpers:
    def test_extract_target_file(self):
        assert (
            extract_target_file_from_goal("fix bug in agent/config.py")
            == "agent/config.py"
        )

    def test_extract_no_file(self):
        assert extract_target_file_from_goal("improve performance") is None

    def test_is_code_edit_goal(self):
        assert is_code_edit_goal("code-edit agent/utils.py") is True
        assert is_code_edit_goal("explain how it works") is False

    def test_is_git_goal(self):
        assert is_git_goal("commit the changes") is True
        assert is_git_goal("fix the bug") is False


class TestInferFallback:
    def test_with_file_target(self):
        action, args = infer_fallback_action("fix agent/config.py", {})
        assert action == "read_file"
        assert args["path"] == "agent/config.py"

    def test_test_keyword(self):
        action, args = infer_fallback_action("improve test coverage", {})
        assert action == "search_code"

    def test_performance_keyword(self):
        action, args = infer_fallback_action("optimize latence", {})
        assert action == "search_code"

    def test_generic_fallback(self):
        action, args = infer_fallback_action("do something", {})
        assert action == "list_files"


class TestAutocorrect:
    def test_fix_missing_path(self):
        d = {"action": "read_file", "reason": "t", "args": {"file": "test.py"}}
        result = autocorrect_decision_schema(
            "fix test.py", d, "missing required arg 'path' for action 'read_file'"
        )
        assert result["args"]["path"] == "test.py"

    def test_fix_invalid_action(self):
        d = {"action": "invalid", "reason": "t", "args": {}}
        result = autocorrect_decision_schema(
            "fix agent/config.py", d, "invalid action: invalid"
        )
        assert result["action"] == "read_file"
        assert result["args"]["path"] == "agent/config.py"

    def test_fix_finish_reason_to_summary(self):
        # After normalize_decision, args.reason is stripped so autocorrect
        # falls back to decision reason field for the summary.
        d = {"action": "finish", "reason": "all done", "args": {"reason": "all done"}}
        result = autocorrect_decision_schema(
            "task done", d, "unexpected arg 'reason' for action 'finish'"
        )
        assert "summary" in result["args"]


# ---------------------------------------------------------------------------
# Agent loop integration tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestAgentLoop:
    @patch("agent.auto_agent.ask_llm_json")
    @patch(
        "agent.executor.list_files", return_value=["agent/config.py", "agent/memory.py"]
    )
    @patch("agent.tooling.list_python_files", return_value=["agent/config.py"])
    @patch("agent.auto_agent.search_memory", return_value=[])
    def test_simple_list_then_finish(self, mock_mem, mock_pyfiles, mock_list, mock_llm):
        from agent.auto_agent import AutonomousAgent

        mock_llm.side_effect = [
            # Planner: list_files
            {"action": "list_files", "reason": "explore", "args": {"limit": 20}},
            # Critique: approve
            {"approve": True, "reason": "ok"},
            # Planner: finish
            {
                "action": "finish",
                "reason": "done",
                "args": {"summary": "explored files"},
            },
            # Critique: approve
            {"approve": True, "reason": "ok"},
        ]

        agent = AutonomousAgent(max_steps=4)
        result = agent.run("list project files", auto_apply=False)
        assert "explored files" in result
        assert len(agent.steps) >= 2

    @patch("agent.auto_agent.ask_llm_json")
    @patch("agent.tooling.list_python_files", return_value=[])
    @patch("agent.auto_agent.search_memory", return_value=[])
    def test_loop_detection_stops_agent(self, mock_mem, mock_pyfiles, mock_llm):
        from agent.auto_agent import AutonomousAgent

        # Return the same decision repeatedly to trigger loop detection
        same_decision = {
            "action": "list_files",
            "reason": "loop",
            "args": {"limit": 20},
        }
        approve = {"approve": True, "reason": "ok"}

        mock_llm.side_effect = [same_decision, approve] * 10

        with patch("agent.executor.list_files", return_value=["a.py"]):
            agent = AutonomousAgent(max_steps=10)
            result = agent.run("test loop", auto_apply=False)
            # Should stop before max_steps due to loop detection
            assert (
                "loop" in result.lower()
                or "repeat" in result.lower()
                or len(agent.steps) < 10
            )

    @patch("agent.auto_agent.ask_llm_json")
    @patch("agent.tooling.list_python_files", return_value=[])
    @patch("agent.auto_agent.search_memory", return_value=[])
    def test_cost_budget_stops_agent(self, mock_mem, mock_pyfiles, mock_llm):
        from agent.auto_agent import AutonomousAgent

        # Use varied expensive actions so loop detection doesn't trigger first
        # Each pair costs 5 (run_quality), and max_cost = max_steps * 4 = 20
        # So 5 calls of cost=5 => total_cost=25 > 20 should trigger budget stop
        calls = []
        for i in range(20):
            calls.append({"action": "run_quality", "reason": f"check_{i}", "args": {}})
            calls.append({"approve": True, "reason": "ok"})
        mock_llm.side_effect = calls

        with patch(
            "agent.executor.run_quality_pipeline_normalized",
            return_value={"ok": True, "raw": []},
        ):
            agent = AutonomousAgent(max_steps=20)
            result = agent.run("check quality repeatedly", auto_apply=False)
            # Should stop due to cost budget or loop detection
            assert (
                "cost" in result.lower()
                or "loop" in result.lower()
                or "repeat" in result.lower()
            )


class TestCreateFileApplyMode:
    @patch("agent.executor.detect_and_install_deps", return_value="deps:ok")
    @patch("agent.executor.index_file_in_session", return_value=1)
    @patch("agent.executor.patch_risk", return_value={"level": "low", "score": 0})
    def test_create_file_no_apply_stages_without_writing(
        self, _risk, _index, _deps
    ):
        from agent.auto_agent import AutonomousAgent
        from agent.executor import ActionExecutor
        from agent.config import PROJECT_ROOT

        rel_dir = os.path.join("tests", "__tmp_create_file_no_apply")
        rel_path = os.path.join(rel_dir, "new_module.py")
        abs_dir = os.path.join(PROJECT_ROOT, rel_dir)
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        shutil.rmtree(abs_dir, ignore_errors=True)
        try:
            agent = AutonomousAgent(max_steps=1)
            agent._generate_file_content = lambda *_args, **_kwargs: "x = 1\n"
            ex = ActionExecutor(agent)

            out = ex.execute(
                "create_file",
                {"path": rel_path, "description": "demo"},
                goal="create module file",
                auto_apply=False,
                index=1,
            )

            assert "Staged creation for" in out
            assert rel_path in agent.staged_edits
            assert not os.path.exists(abs_path)
        finally:
            shutil.rmtree(abs_dir, ignore_errors=True)

    @patch("agent.executor.detect_and_install_deps", return_value="deps:ok")
    @patch("agent.executor.index_file_in_session", return_value=1)
    @patch("agent.executor.patch_risk", return_value={"level": "low", "score": 0})
    def test_create_file_apply_writes_to_disk(self, _risk, _index, _deps):
        from agent.auto_agent import AutonomousAgent
        from agent.executor import ActionExecutor
        from agent.config import PROJECT_ROOT

        rel_dir = os.path.join("tests", "__tmp_create_file_apply")
        rel_path = os.path.join(rel_dir, "new_module.py")
        abs_dir = os.path.join(PROJECT_ROOT, rel_dir)
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        shutil.rmtree(abs_dir, ignore_errors=True)
        try:
            agent = AutonomousAgent(max_steps=1)
            agent._generate_file_content = lambda *_args, **_kwargs: "x = 1\n"
            ex = ActionExecutor(agent)

            out = ex.execute(
                "create_file",
                {"path": rel_path, "description": "demo"},
                goal="create module file",
                auto_apply=True,
                index=1,
            )

            assert "Created" in out
            assert os.path.exists(abs_path)
        finally:
            shutil.rmtree(abs_dir, ignore_errors=True)


class TestTransactionRollbackIntegration:
    @patch("agent.executor.rollback_transaction", return_value=True)
    @patch("agent.executor.run_quality_gate", return_value=(False, "quality failed"))
    @patch(
        "agent.executor.apply_transaction",
        return_value=(True, [("a.py", "a.py.bak")], "transaction_applied"),
    )
    def test_apply_all_staged_rolls_back_on_quality_failure(
        self, _apply_tx, _quality_gate, _rollback
    ):
        from agent.auto_agent import AutonomousAgent
        from agent.executor import ActionExecutor

        ag = AutonomousAgent(max_steps=1)
        ag.staged_edits = {"a.py": "x = 1\n"}
        ag.interactive = False
        ag._fix_until_green = False

        ex = ActionExecutor(ag)
        out = ex.execute(
            "apply_all_staged",
            {},
            goal="apply staged",
            auto_apply=True,
            index=1,
        )
        assert "rollback=ok" in out
