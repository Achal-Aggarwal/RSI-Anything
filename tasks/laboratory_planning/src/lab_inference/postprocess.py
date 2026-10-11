"""Action-signature repair and final-plan postprocessing."""

from __future__ import annotations

import ast

import hashlib

import re

def signature_map(action_pool: str) -> dict[str, list[str]]:
    tree = ast.parse(action_pool)
    return {item.name: [arg.arg for arg in item.args.posonlyargs + item.args.args] for item in tree.body if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))}

def code_from_completion(text: str) -> str:
    fenced = re.search('```(?:python)?\\s*(.*?)```', text, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    open_fence = re.search('```(?:python)?\\s*', text, re.IGNORECASE)
    if open_fence:
        return text[open_fence.end():].strip()
    return text.strip()

def repair(completion: str, signatures: dict[str, list[str]]) -> tuple[str, dict]:
    source = code_from_completion(completion)
    lines = source.splitlines()
    dropped = 0
    while lines:
        try:
            tree = ast.parse('\n'.join(lines))
            break
        except SyntaxError:
            lines.pop()
            dropped += 1
    else:
        return (completion, {'changed': False, 'reason': 'no_parseable_prefix', 'dropped_lines': dropped})
    converted = 0
    for statement in tree.body:
        call = None
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call):
            call = statement.value
        elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
        if not call or not call.args or (not isinstance(call.func, ast.Name)):
            continue
        params = signatures.get(call.func.id)
        existing = {keyword.arg for keyword in call.keywords}
        if params is None or len(call.args) > len(params) or any((name in existing for name in params[:len(call.args)])):
            continue
        call.keywords = [ast.keyword(arg=name, value=value) for name, value in zip(params, call.args)] + call.keywords
        converted += len(call.args)
        call.args = []
    if not converted and (not dropped):
        return (completion, {'changed': False, 'reason': 'no_repair_needed', 'dropped_lines': 0})
    ast.fix_missing_locations(tree)
    return (ast.unparse(tree), {'changed': True, 'positional_args_converted': converted, 'dropped_lines': dropped})

def is_action_call(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call) or (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call))

def cap_completion(completion: str, max_steps: int) -> tuple[str, dict]:
    source = code_from_completion(completion)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return (completion, {'changed': False, 'reason': 'unparseable', 'original_steps': None})
    count = 0
    end = None
    for index, statement in enumerate(tree.body):
        if is_action_call(statement):
            count += 1
            if count == max_steps:
                end = index + 1
                break
    if count < max_steps or end is None:
        return (completion, {'changed': False, 'reason': 'at_or_under_cap', 'original_steps': count})
    total = sum((is_action_call(statement) for statement in tree.body))
    if total <= max_steps:
        return (completion, {'changed': False, 'reason': 'at_or_under_cap', 'original_steps': total})
    tree.body = tree.body[:end]
    ast.fix_missing_locations(tree)
    return (ast.unparse(tree), {'changed': True, 'reason': 'capped_by_T_length_prior', 'original_steps': total, 'retained_steps': max_steps})

def preprocess(text, action_pool):
    signatures = {n.name: [a.arg for a in n.args.posonlyargs + n.args.args] for n in ast.parse(action_pool).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    lines = code_from_completion(text).splitlines()
    while lines:
        try:
            tree = ast.parse('\n'.join(lines))
            break
        except SyntaxError:
            lines.pop()
    else:
        return text
    changed = len(lines) != len(code_from_completion(text).splitlines())
    for statement in tree.body:
        call = statement.value if isinstance(statement, (ast.Assign, ast.Expr)) else None
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or (not call.args):
            continue
        params = signatures.get(call.func.id)
        existing = {kw.arg for kw in call.keywords}
        if params is None or len(call.args) > len(params) or any((n in existing for n in params[:len(call.args)])):
            continue
        call.keywords = [ast.keyword(arg=n, value=v) for n, v in zip(params, call.args)] + call.keywords
        call.args = []
        changed = True
    if len(tree.body) > 13:
        tree.body = tree.body[:13]
        changed = True
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) if changed else text

def stable_seed(seed, task_id):
    return int.from_bytes(hashlib.sha256(f'{seed}|{task_id}'.encode()).digest()[:4], 'little')
