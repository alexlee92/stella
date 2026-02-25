"""
P2.1 — Decision normalization and autocorrection logic.

Pure functions extracted from AutonomousAgent to reduce auto_agent.py complexity.
"""

import os
import re
from typing import Dict, List, Optional, Tuple

from agent.action_schema import ALLOWED_ACTIONS
from agent.config import PROJECT_ROOT

# ---------------------------------------------------------------------------
# Goal analysis helpers
# ---------------------------------------------------------------------------


def extract_target_file_from_goal(goal: str) -> Optional[str]:
    low = goal.strip()
    m = re.search(r"([A-Za-z0-9_./\\-]+\.[a-zA-Z]{1,5})", low)
    if not m:
        return None
    path = m.group(1).replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    return path


def extract_all_target_files_from_goal(goal: str) -> List[str]:
    """Extract file paths to CREATE from the goal.

    Excludes files that already exist on disk.
    """
    found = re.findall(r"([A-Za-z0-9_./\\-]+\.[a-zA-Z]{1,5})", goal)
    seen: set[str] = set()
    out: List[str] = []
    for raw in found:
        path = raw.replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]
        if not re.match(
            r".+\.(py|js|ts|jsx|tsx|html|css|scss|json|yaml|yml|toml|sql|md)$", path
        ):
            continue
        abs_path = os.path.join(PROJECT_ROOT, path)
        if os.path.isfile(abs_path):
            continue
        if path not in seen:
            seen.add(path)
            out.append(path)
    return out


def is_code_edit_goal(goal: str) -> bool:
    low = (goal or "").lower()
    return "code-edit" in low or ("dans " in low and ".py" in low)


def is_git_goal(goal: str) -> bool:
    low = (goal or "").lower()
    return any(k in low for k in ["git", "commit", "branch", "pr", "pull request"])


def is_allowed_edit_path(goal: str, path: str) -> bool:
    from agent.test_selector import suggest_test_path

    target = extract_target_file_from_goal(goal)
    if not target:
        return True
    norm = (path or "").replace("\\", "/").lower()
    target_norm = target.replace("\\", "/").lower()
    if norm == target_norm:
        return True
    if norm == suggest_test_path(target).replace("\\", "/").lower():
        return True
    if norm.startswith("tests/"):
        return True
    return False


# ---------------------------------------------------------------------------
# Action alias map
# ---------------------------------------------------------------------------

_ACTION_ALIASES: Dict[str, str] = {
    "read": "read_file",
    "read_files": "read_many",
    "search": "search_code",
    "grep": "search_code",
    "create": "create_file",
    "new_file": "create_file",
    "write_file": "create_file",
    "generate_file": "create_file",
    "create_module": "create_file",
    "write_new_file": "create_file",
    "edit": "propose_edit",
    "propose": "propose_edit",
    "propose_patch": "propose_edit",
    "apply_patch": "apply_edit",
    "apply": "apply_edit",
    "apply_staged": "apply_all_staged",
    "test": "run_tests",
    "run_test": "run_tests",
    "quality": "run_quality",
    "run_quality_pipeline": "run_quality",
    "map": "project_map",
    "branch": "git_branch",
    "commit": "git_commit",
    "diff": "git_diff",
    "done": "finish",
    "final": "finish",
}

_ALLOWED_ARGS_PER_ACTION: Dict[str, set] = {
    "list_files": {"contains", "ext", "limit"},
    "read_file": {"path"},
    "read_many": {"paths"},
    "search_code": {"pattern", "limit"},
    "create_file": {"path", "description"},
    "propose_edit": {"path", "instruction"},
    "apply_edit": {"path"},
    "apply_all_staged": set(),
    "run_tests": {"command"},
    "run_quality": set(),
    "project_map": set(),
    "git_branch": {"name"},
    "git_commit": {"message"},
    "git_diff": set(),
    "finish": {"summary"},
}


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def normalize_decision(decision: dict) -> dict:
    if not isinstance(decision, dict):
        return decision

    root_args = {
        k: v
        for k, v in decision.items()
        if k
        not in {
            "action",
            "tool",
            "name",
            "decision",
            "reason",
            "why",
            "args",
            "parameters",
            "input",
        }
        and not (isinstance(k, str) and k.startswith("_"))
    }

    action = (
        decision.get("action")
        or decision.get("tool")
        or decision.get("name")
        or decision.get("decision")
        or ""
    )
    action = str(action).strip().lower().replace("-", "_")
    action = _ACTION_ALIASES.get(action, action)

    reason = decision.get("reason") or decision.get("why") or "auto_normalized"
    args = decision.get("args")
    if args is None:
        args = decision.get("parameters")
    if args is None:
        args = decision.get("input")
    if not isinstance(args, dict):
        args = {}

    args = {**root_args, **dict(args)}

    # Type coercion
    if action in {"list_files", "search_code"}:
        limit = args.get("limit")
        if isinstance(limit, str) and limit.isdigit():
            args["limit"] = int(limit)
        elif isinstance(limit, float):
            args["limit"] = int(limit)
    if action == "read_many" and isinstance(args.get("paths"), str):
        args["paths"] = [args["paths"]]
    if action == "search_code" and "pattern" not in args:
        query = args.get("query") or args.get("text")
        if isinstance(query, str) and query.strip():
            args["pattern"] = query
    if action == "run_tests" and "command" not in args:
        cmd = args.get("test_command")
        if isinstance(cmd, str) and cmd.strip():
            args["command"] = cmd
    if action == "finish" and "summary" not in args:
        summary = args.get("message") or reason
        if isinstance(summary, str) and summary.strip():
            args["summary"] = summary
    if action in {"apply_all_staged", "run_quality", "project_map", "git_diff"}:
        args = {}

    # Infer action from args shape
    if action not in ALLOWED_ACTIONS:
        if isinstance(args.get("path"), str) and isinstance(
            args.get("instruction"), str
        ):
            action = "propose_edit"
        elif isinstance(args.get("paths"), list):
            action = "read_many"
        elif isinstance(args.get("pattern"), str):
            action = "search_code"
        elif isinstance(args.get("path"), str):
            action = "read_file"
        elif isinstance(args.get("summary"), str):
            action = "finish"

    # Filter to allowed args
    allowed = _ALLOWED_ARGS_PER_ACTION.get(action, set(args.keys()))
    args = {k: v for k, v in args.items() if k in allowed}

    normalized = {"action": action, "reason": str(reason), "args": args}
    for k, v in decision.items():
        if isinstance(k, str) and k.startswith("_"):
            normalized[k] = v
    return normalized


def normalize_critique(critique: dict) -> dict:
    if not isinstance(critique, dict):
        return critique
    approve = critique.get("approve")
    if isinstance(approve, str):
        approve = approve.strip().lower() in {"true", "1", "yes", "ok"}
    if not isinstance(approve, bool):
        approve = False

    reason = critique.get("reason") or critique.get("comment") or "normalized"
    patched = critique.get("patched_decision")
    if patched is None:
        patched = critique.get("patch")
    if not isinstance(patched, dict):
        patched = None
    else:
        patched = normalize_decision(patched)

    normalized = {
        "approve": approve,
        "reason": str(reason),
        "patched_decision": patched,
    }
    for k, v in critique.items():
        if isinstance(k, str) and k.startswith("_"):
            normalized[k] = v
    return normalized


def coerce_decision(goal: str, decision: dict) -> dict:
    action = decision.get("action", "")
    args = decision.get("args", {}) or {}

    if action.startswith("git_") and not is_git_goal(goal):
        return {
            "action": "search_code",
            "reason": "coerced_non_git_goal",
            "args": {
                "pattern": "requests.post|subprocess.run|timeout|sleep",
                "limit": 20,
            },
        }

    if action == "list_files" and args.get("contains"):
        args = dict(args)
        args["contains"] = ""
        return {
            "action": "list_files",
            "reason": "coerced_broad_listing",
            "args": args,
        }

    return decision


# ---------------------------------------------------------------------------
# Fallback inference
# ---------------------------------------------------------------------------


def infer_fallback_action(goal: str, args: dict) -> Tuple[str, dict]:
    low_goal = (goal or "").lower()
    target = extract_target_file_from_goal(goal)
    if target:
        return "read_file", {"path": target}

    text_blob = " ".join(
        str(v) for v in args.values() if isinstance(v, (str, int, float, bool))
    ).lower()
    combined = f"{low_goal} {text_blob}"

    if any(k in combined for k in ["test", "pytest", "coverage", "fiabilite"]):
        return "search_code", {"pattern": "pytest|test_|_test.py", "limit": 20}
    if any(k in combined for k in ["performance", "latence", "speed", "optimiz"]):
        return "search_code", {
            "pattern": "timeout|sleep|requests.post|subprocess.run",
            "limit": 20,
        }
    if any(k in combined for k in ["refactor", "risk", "analyse", "architecture"]):
        return "list_files", {"limit": 40, "ext": ".py"}
    if any(k in combined for k in ["pr", "commit", "branch", "git"]):
        return "git_diff", {}
    return "list_files", {"limit": 40, "ext": ".py"}


def autocorrect_decision_schema(goal: str, decision: dict, msg: str) -> dict:
    if not isinstance(decision, dict):
        return decision
    corrected = normalize_decision(decision)
    action = corrected.get("action")
    args = corrected.get("args", {}) or {}
    if not isinstance(args, dict):
        args = {}

    if "invalid action" in msg:
        action, inferred_args = infer_fallback_action(goal, args)
        corrected["action"] = action
        args = dict(inferred_args)

    if action == "read_file" and "missing required arg 'path'" in msg:
        path = args.get("file") or args.get("target")
        if isinstance(path, str) and path:
            args["path"] = path
        else:
            target = extract_target_file_from_goal(goal)
            if target:
                args["path"] = target
    elif action == "read_many" and "missing required arg 'paths'" in msg:
        path = args.get("path")
        if isinstance(path, str) and path:
            args["paths"] = [path]
        else:
            target = extract_target_file_from_goal(goal)
            if target:
                args["paths"] = [target]
    elif action == "propose_edit":
        if "missing required arg 'path'" in msg:
            path = args.get("file") or args.get("target")
            if isinstance(path, str) and path:
                args["path"] = path
            else:
                target = extract_target_file_from_goal(goal)
                if target:
                    args["path"] = target
        if "missing required arg 'instruction'" in msg:
            instruction = (
                args.get("prompt") or args.get("change") or decision.get("reason", "")
            )
            if isinstance(instruction, str) and instruction:
                args["instruction"] = instruction
            elif isinstance(goal, str) and goal.strip():
                args["instruction"] = goal[:500]
    elif action == "finish" and "unexpected arg 'reason'" in msg:
        summary = args.get("reason")
        if isinstance(summary, str) and summary:
            args.pop("reason", None)
            args["summary"] = summary

    corrected["args"] = args
    corrected["reason"] = str(corrected.get("reason") or "auto_corrected_schema")
    return corrected
