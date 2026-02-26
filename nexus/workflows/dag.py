"""
DAG utilities for workflow graph traversal and condition evaluation.

All functions operate on WorkflowStep / WorkflowEdge lists and are pure
(no side effects, no I/O) so they can be called safely from validator,
manager, and execution engine alike.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from typing import Any

from nexus.exceptions import WorkflowValidationError
from nexus.types import WorkflowEdge, WorkflowStep


# ── Traversal helpers ─────────────────────────────────────────────────────────


def get_entry_points(
    steps: list[WorkflowStep], edges: list[WorkflowEdge]
) -> list[str]:
    """Return step IDs with no incoming edges (DAG roots)."""
    target_ids = {e.target_step_id for e in edges}
    return [s.id for s in steps if s.id not in target_ids]


def get_exit_points(
    steps: list[WorkflowStep], edges: list[WorkflowEdge]
) -> list[str]:
    """Return step IDs with no outgoing edges (DAG leaves)."""
    source_ids = {e.source_step_id for e in edges}
    return [s.id for s in steps if s.id not in source_ids]


def get_children(
    step_id: str, edges: list[WorkflowEdge]
) -> list[tuple[str, WorkflowEdge]]:
    """Return (target_step_id, edge) pairs for all outgoing edges of step_id."""
    return [(e.target_step_id, e) for e in edges if e.source_step_id == step_id]


def get_parents(
    step_id: str, edges: list[WorkflowEdge]
) -> list[tuple[str, WorkflowEdge]]:
    """Return (source_step_id, edge) pairs for all incoming edges of step_id."""
    return [(e.source_step_id, e) for e in edges if e.target_step_id == step_id]


def get_parallel_group(
    step_id: str,
    steps: list[WorkflowStep],
    edges: list[WorkflowEdge],
) -> list[str]:
    """
    Return all step IDs that share the exact same set of parent steps as step_id.

    Steps with identical parent sets belong to the same parallel fork and should
    converge downstream.  A step with no parents is its own singleton group.
    """
    step_parents = frozenset(
        e.source_step_id for e in edges if e.target_step_id == step_id
    )
    if not step_parents:
        return [step_id]

    return [
        s.id
        for s in steps
        if frozenset(
            e.source_step_id for e in edges if e.target_step_id == s.id
        ) == step_parents
    ]


# ── Topological sort (Kahn's algorithm) ──────────────────────────────────────


def topological_sort(
    steps: list[WorkflowStep], edges: list[WorkflowEdge]
) -> list[str]:
    """
    Return step IDs in topological order.

    Uses Kahn's BFS algorithm:
      1. Compute in-degree for every node.
      2. Seed queue with zero-in-degree nodes (entry points).
      3. BFS: pop node, emit it, decrement in-degrees of its children.
      4. If emitted count < total nodes → cycle exists.

    Raises:
        WorkflowValidationError: if the graph contains a cycle, with a
            description of which nodes are involved.
    """
    step_ids = [s.id for s in steps]
    if not step_ids:
        return []

    in_degree: dict[str, int] = {sid: 0 for sid in step_ids}
    adjacency: dict[str, list[str]] = {sid: [] for sid in step_ids}

    for edge in edges:
        src, tgt = edge.source_step_id, edge.target_step_id
        # Only count edges whose endpoints exist (validator checks the rest)
        if src in adjacency and tgt in in_degree:
            adjacency[src].append(tgt)
            in_degree[tgt] += 1

    queue: deque[str] = deque(
        sid for sid in step_ids if in_degree[sid] == 0
    )
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child in adjacency[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(step_ids):
        # Identify cycle participants — nodes never reached
        cycle_nodes = [sid for sid in step_ids if sid not in set(order)]
        raise WorkflowValidationError(
            f"Cycle detected in workflow graph. Involved step IDs: {cycle_nodes}",
            violations=[f"Cycle includes steps: {cycle_nodes}"],
        )

    return order


# ── Safe condition evaluator ──────────────────────────────────────────────────


def evaluate_condition(condition: str, context: dict[str, Any]) -> bool:
    """
    Evaluate a workflow edge condition expression safely.  Never calls eval().

    Supported syntax (after ${var} substitution):
      - Literals: "true", "false", "1", "0", "yes", "no"
      - Comparisons: a == b, a != b, a < b, a <= b, a > b, a >= b
      - Membership: a in b, a not in b
      - Boolean: and, or, not
      - Grouped: (expr)

    Variables are referenced as ${key} or ${nested.key}.  Unresolved variables
    substitute as None (falsy).

    Args:
        condition: Expression string, e.g. "${status} == 'completed'"
        context:   Runtime data dict, e.g. {"status": "completed"}

    Returns:
        bool result of the expression.

    Raises:
        WorkflowValidationError: on syntax errors or unsupported node types.
    """
    if not condition or not condition.strip():
        return True

    normalized = condition.strip()

    # Fast path: boolean literals
    lower = normalized.lower()
    if lower in ("true", "yes", "1"):
        return True
    if lower in ("false", "no", "0"):
        return False

    # Substitute ${var} and ${nested.key} references with repr() of value
    def _substitute(match: re.Match) -> str:  # type: ignore[type-arg]
        key_path = match.group(1).strip()
        val: Any = context
        for part in key_path.split("."):
            if isinstance(val, dict):
                val = val.get(part)
            else:
                val = getattr(val, part, None)
            if val is None:
                break
        return repr(val)

    expr = re.sub(r"\$\{([^}]+)\}", _substitute, normalized)

    # Parse with ast — raises SyntaxError on bad input
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise WorkflowValidationError(
            f"Invalid condition syntax: {condition!r}",
            violations=[str(exc)],
        ) from exc

    return bool(_eval_ast_node(tree.body))


def _eval_ast_node(node: ast.expr) -> Any:
    """
    Recursively evaluate a safe subset of the Python AST.

    Allowed node types:
      ast.Constant    — literal values (str, int, float, bool, None)
      ast.Compare     — ==, !=, <, <=, >, >=, in, not in
      ast.BoolOp      — and, or
      ast.UnaryOp Not — not
      ast.Tuple/List  — for 'in' RHS membership tests

    All other node types raise WorkflowValidationError (no function calls,
    attribute access, subscripts, assignments, etc.).
    """
    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_eval_ast_node(el) for el in node.elts)

    if isinstance(node, ast.Compare):
        left = _eval_ast_node(node.left)
        result = True
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval_ast_node(comparator)
            result = result and _apply_compare_op(op, left, right)
            left = right
        return result

    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(_eval_ast_node(v) for v in node.values)
        if isinstance(node.op, ast.Or):
            return any(_eval_ast_node(v) for v in node.values)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not _eval_ast_node(node.operand)

    raise WorkflowValidationError(
        f"Unsupported expression node '{type(node).__name__}' in condition. "
        "Only comparisons, boolean operators, and literals are allowed.",
        violations=[f"Disallowed AST node: {type(node).__name__}"],
    )


def _apply_compare_op(op: ast.cmpop, left: Any, right: Any) -> bool:
    """Apply a single comparison operator."""
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right  # type: ignore[operator]
    if isinstance(op, ast.LtE):
        return left <= right  # type: ignore[operator]
    if isinstance(op, ast.Gt):
        return left > right  # type: ignore[operator]
    if isinstance(op, ast.GtE):
        return left >= right  # type: ignore[operator]
    if isinstance(op, ast.In):
        return left in right  # type: ignore[operator]
    if isinstance(op, ast.NotIn):
        return left not in right  # type: ignore[operator]
    raise WorkflowValidationError(
        f"Unsupported comparison operator: {type(op).__name__}",
        violations=[f"Disallowed operator: {type(op).__name__}"],
    )
