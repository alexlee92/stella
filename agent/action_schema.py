from typing import Any

ALLOWED_ACTIONS = {
    "list_files",
    "read_file",
    "read_many",
    "search_code",
    "create_file",
    "propose_edit",
    "apply_edit",
    "apply_all_staged",
    "run_tests",
    "run_quality",
    "project_map",
    "git_branch",
    "git_commit",
    "git_diff",
    "finish",
}


def _is_type(value: Any, expected: str) -> bool:
    if expected == "str":
        return isinstance(value, str)
    if expected == "int":
        return isinstance(value, int)
    if expected == "list_str":
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    if expected == "dict":
        return isinstance(value, dict)
    if expected == "bool":
        return isinstance(value, bool)
    return False


def validate_decision_schema(decision: dict) -> tuple[bool, str]:
    if not isinstance(decision, dict):
        return False, "decision must be a JSON object"

    action = decision.get("action")
    reason = decision.get("reason")
    args = decision.get("args")

    if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
        return False, f"invalid action: {action}"

    if not isinstance(reason, str) or not reason.strip():
        return False, "reason must be a non-empty string"

    if not isinstance(args, dict):
        return False, "args must be an object"

    rules = {
        "list_files": {
            "required": {},
            "optional": {"contains": "str", "ext": "str", "limit": "int"},
        },
        "read_file": {"required": {"path": "str"}, "optional": {}},
        "read_many": {"required": {"paths": "list_str"}, "optional": {}},
        "search_code": {"required": {"pattern": "str"}, "optional": {"limit": "int"}},
        "create_file": {
            "required": {"path": "str", "description": "str"},
            "optional": {},
        },
        "propose_edit": {
            "required": {"path": "str", "instruction": "str"},
            "optional": {},
        },
        "apply_edit": {"required": {"path": "str"}, "optional": {}},
        "apply_all_staged": {"required": {}, "optional": {}},
        "run_tests": {"required": {}, "optional": {"command": "str"}},
        "run_quality": {"required": {}, "optional": {}},
        "project_map": {"required": {}, "optional": {}},
        "git_branch": {"required": {}, "optional": {"name": "str"}},
        "git_commit": {"required": {}, "optional": {"message": "str"}},
        "git_diff": {"required": {}, "optional": {}},
        "finish": {"required": {}, "optional": {"summary": "str"}},
    }

    spec = rules[action]
    allowed_keys = set(spec["required"].keys()) | set(spec["optional"].keys())

    for key, typ in spec["required"].items():
        if key not in args:
            return False, f"missing required arg '{key}' for action '{action}'"
        if not _is_type(args[key], typ):
            return False, f"invalid type for arg '{key}' in action '{action}'"

    for key, value in args.items():
        if key not in allowed_keys:
            return False, f"unexpected arg '{key}' for action '{action}'"
        expected = spec["required"].get(key) or spec["optional"].get(key)
        if expected and not _is_type(value, expected):
            return False, f"invalid type for arg '{key}' in action '{action}'"

    return True, "ok"


def validate_critique_schema(critique: dict) -> tuple[bool, str]:
    if not isinstance(critique, dict):
        return False, "critique must be object"

    approve = critique.get("approve")
    reason = critique.get("reason")

    if not isinstance(approve, bool):
        return False, "approve must be bool"
    if not isinstance(reason, str) or not reason.strip():
        return False, "reason must be non-empty string"

    patched = critique.get("patched_decision")
    if patched is not None and not isinstance(patched, dict):
        return False, "patched_decision must be object when provided"

    return True, "ok"
