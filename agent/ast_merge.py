import ast
from typing import Tuple


def _top_level_named_nodes(tree: ast.Module):
    nodes = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            nodes[node.name] = node
    return nodes


def ast_merge_python_code(old_code: str, new_code: str) -> Tuple[str, bool, str]:
    try:
        old_tree = ast.parse(old_code)
        new_tree = ast.parse(new_code)
    except SyntaxError as exc:
        return new_code, False, f"syntax_error:{exc}"

    old_named = _top_level_named_nodes(old_tree)
    new_named = _top_level_named_nodes(new_tree)

    if not old_named or not new_named:
        return new_code, False, "no_top_level_symbols"

    shared = set(old_named.keys()) & set(new_named.keys())
    if not shared:
        return new_code, False, "no_shared_symbols"

    merged_body = []
    used_names = set()
    for node in old_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in new_named:
            merged_body.append(new_named[node.name])
            used_names.add(node.name)
        else:
            merged_body.append(node)

    for node in new_tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name not in used_names:
            merged_body.append(node)

    merged = ast.Module(body=merged_body, type_ignores=[])
    ast.fix_missing_locations(merged)

    try:
        merged_code = ast.unparse(merged)
    except Exception as exc:
        return new_code, False, f"unparse_failed:{exc}"

    if not merged_code.strip():
        return new_code, False, "empty_merge_result"

    return merged_code + "\n", True, "ast_merge_applied"
