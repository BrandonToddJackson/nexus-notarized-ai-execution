"""Built-in data_transform tool — safe, eval-free data pipeline."""

import copy
import math
import re
import json
from datetime import datetime
from typing import Any

from nexus.types import ToolDefinition, RiskLevel
from nexus.tools.plugin import _registered_tools


# ─── Dot-notation helpers ────────────────────────────────────────────────────

def _get_nested(item: Any, field: str) -> Any:
    """Get value from nested dict using dot notation."""
    for part in field.split("."):
        if isinstance(item, dict):
            item = item.get(part)
        else:
            return None
    return item


def _set_nested(item: dict, field: str, value: Any) -> None:
    """Set value in nested dict using dot notation, creating intermediate dicts."""
    parts = field.split(".")
    for part in parts[:-1]:
        if part not in item or not isinstance(item[part], dict):
            item[part] = {}
        item = item[part]
    item[parts[-1]] = value


def _del_nested(item: dict, field: str) -> None:
    """Delete key from nested dict using dot notation."""
    parts = field.split(".")
    for part in parts[:-1]:
        if not isinstance(item, dict):
            return
        item = item.get(part)
        if item is None:
            return
    if isinstance(item, dict):
        item.pop(parts[-1], None)


# ─── Filter conditions ────────────────────────────────────────────────────────

def _check_condition(item: dict, field: str, operator: str, value: Any) -> bool:
    actual = _get_nested(item, field)

    if operator == "eq":
        return actual == value
    elif operator == "ne":
        return actual != value
    elif operator == "gt":
        return actual is not None and actual > value
    elif operator == "gte":
        return actual is not None and actual >= value
    elif operator == "lt":
        return actual is not None and actual < value
    elif operator == "lte":
        return actual is not None and actual <= value
    elif operator == "contains":
        if isinstance(actual, str):
            return str(value) in actual
        elif isinstance(actual, list):
            return value in actual
        return False
    elif operator == "not_contains":
        if isinstance(actual, str):
            return str(value) not in actual
        elif isinstance(actual, list):
            return value not in actual
        return True
    elif operator == "starts_with":
        return isinstance(actual, str) and actual.startswith(str(value))
    elif operator == "ends_with":
        return isinstance(actual, str) and actual.endswith(str(value))
    elif operator == "in":
        return actual in (value if isinstance(value, list) else [value])
    elif operator == "not_in":
        return actual not in (value if isinstance(value, list) else [value])
    elif operator == "is_null":
        return actual is None
    elif operator == "is_not_null":
        return actual is not None
    return False


def _op_filter(data: Any, op_spec: dict) -> Any:
    if not isinstance(data, list):
        return data
    conditions = op_spec.get("conditions", [])
    logic = op_spec.get("logic", "and")

    # Single condition shorthand
    if "field" in op_spec and "operator" in op_spec:
        conditions = [{"field": op_spec["field"], "operator": op_spec["operator"], "value": op_spec.get("value")}]
        logic = "and"

    def matches(item: dict) -> bool:
        results = [_check_condition(item, c["field"], c["operator"], c.get("value")) for c in conditions]
        if not results:
            return True
        return all(results) if logic == "and" else any(results)

    return [item for item in data if matches(item)]


# ─── Map ─────────────────────────────────────────────────────────────────────

def _op_map(data: Any, op_spec: dict) -> Any:
    if not isinstance(data, list):
        return data

    output_field = op_spec.get("output_field")

    # Template substitution
    if "template" in op_spec:
        template = op_spec["template"]
        result = []
        for item in data:
            new_item = copy.copy(item)
            def replacer(m):
                val = _get_nested(item, m.group(1))
                return str(val) if val is not None else ""
            value = re.sub(r'\{(\w+(?:\.\w+)*)\}', replacer, template)
            if output_field:
                _set_nested(new_item, output_field, value)
            else:
                new_item = value
            result.append(new_item)
        return result

    # Math operation
    if "math_op" in op_spec:
        math_op = op_spec["math_op"]
        field = op_spec.get("field", "")
        operand = op_spec.get("operand")
        result = []
        for item in data:
            new_item = copy.copy(item)
            val = _get_nested(item, field)
            if val is not None:
                try:
                    val = float(val)
                    if math_op == "add":
                        val = val + operand
                    elif math_op == "subtract":
                        val = val - operand
                    elif math_op == "multiply":
                        val = val * operand
                    elif math_op == "divide":
                        val = val / operand if operand else val
                    elif math_op == "round":
                        val = round(val, operand if operand is not None else 0)
                    elif math_op == "abs":
                        val = abs(val)
                    elif math_op == "floor":
                        val = math.floor(val)
                    elif math_op == "ceil":
                        val = math.ceil(val)
                    # Keep as int if it's a whole number
                    if isinstance(val, float) and val.is_integer() and math_op not in ("divide", "round"):
                        val = int(val)
                except (TypeError, ValueError):
                    pass
                _set_nested(new_item, output_field or field, val)
            result.append(new_item)
        return result

    return data


# ─── Sort ────────────────────────────────────────────────────────────────────

def _sort_key(item: dict, field: str, type_: str):
    val = _get_nested(item, field)
    if val is None:
        return (1, 0)  # None values last; 0 is comparable regardless of other types
    if type_ == "date":
        try:
            return (0, datetime.fromisoformat(str(val)))
        except (ValueError, TypeError):
            return (1, 0)  # unparseable dates sorted last
    elif type_ == "number":
        try:
            return (0, float(val))
        except (TypeError, ValueError):
            return (1, 0)  # non-numeric values sorted last
    else:
        return (0, str(val))


def _op_sort(data: Any, op_spec: dict) -> Any:
    if not isinstance(data, list):
        return data
    field = op_spec.get("field", "")
    order = op_spec.get("order", "asc")
    type_ = op_spec.get("type", "string")
    return sorted(data, key=lambda x: _sort_key(x, field, type_), reverse=(order == "desc"))


# ─── Group by ────────────────────────────────────────────────────────────────

def _op_group_by(data: Any, op_spec: dict) -> Any:
    if not isinstance(data, list):
        return data
    field = op_spec.get("field", "")
    result = {}
    for item in data:
        key = _get_nested(item, field)
        key_str = str(key) if key is not None else "null"
        if key_str not in result:
            result[key_str] = []
        result[key_str].append(item)
    return result


# ─── Flatten ─────────────────────────────────────────────────────────────────

def _op_flatten(data: Any, op_spec: dict) -> Any:
    field = op_spec.get("field")
    if field:
        # Flatten by field: merge field list into parent items
        if not isinstance(data, list):
            return data
        result = []
        for item in data:
            sub = _get_nested(item, field)
            if isinstance(sub, list):
                result.extend(sub)
            else:
                result.append(item)
        return result
    else:
        # Flatten top-level list of lists
        if not isinstance(data, list):
            return data
        result = []
        for sub in data:
            if isinstance(sub, list):
                result.extend(sub)
            else:
                result.append(sub)
        return result


# ─── Pick ────────────────────────────────────────────────────────────────────

def _op_pick(data: Any, op_spec: dict) -> Any:
    if not isinstance(data, list):
        return data
    fields = op_spec.get("fields", [])
    result = []
    for item in data:
        new_item = {}
        for f in fields:
            val = _get_nested(item, f)
            if val is not None:
                _set_nested(new_item, f, val)
        result.append(new_item)
    return result


# ─── Omit ────────────────────────────────────────────────────────────────────

def _op_omit(data: Any, op_spec: dict) -> Any:
    if not isinstance(data, list):
        return data
    fields = op_spec.get("fields", [])
    result = []
    for item in data:
        new_item = copy.deepcopy(item)
        for f in fields:
            _del_nested(new_item, f)
        result.append(new_item)
    return result


# ─── Rename ──────────────────────────────────────────────────────────────────

def _op_rename(data: Any, op_spec: dict) -> Any:
    if not isinstance(data, list):
        return data
    mapping = op_spec.get("mapping", {})  # {old_name: new_name}
    result = []
    for item in data:
        new_item = copy.deepcopy(item)
        for old_field, new_field in mapping.items():
            val = _get_nested(new_item, old_field)
            if val is not None:
                _del_nested(new_item, old_field)
                _set_nested(new_item, new_field, val)
        result.append(new_item)
    return result


# ─── Deduplicate ─────────────────────────────────────────────────────────────

def _op_deduplicate(data: Any, op_spec: dict) -> Any:
    if not isinstance(data, list):
        return data
    field = op_spec.get("field")
    if field:
        seen = set()
        result = []
        for item in data:
            val = _get_nested(item, field)
            key = str(val)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result
    else:
        # Full equality via JSON serialization
        seen = set()
        result = []
        for item in data:
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


# ─── Aggregate ───────────────────────────────────────────────────────────────

def _op_aggregate(data: Any, op_spec: dict) -> Any:
    field = op_spec.get("field", "")
    operation = op_spec.get("operation", "count")
    output_field = op_spec.get("output_field", f"{operation}_{field}")

    if not isinstance(data, list):
        return data

    values = [_get_nested(item, field) for item in data if _get_nested(item, field) is not None]

    if operation == "count":
        result = len(data)
    elif operation == "sum":
        result = sum(float(v) for v in values)
        if isinstance(result, float) and result.is_integer():
            result = int(result)
    elif operation == "avg":
        result = sum(float(v) for v in values) / len(values) if values else 0
    elif operation == "min":
        result = min(values) if values else None
    elif operation == "max":
        result = max(values) if values else None
    elif operation == "first":
        result = values[0] if values else None
    elif operation == "last":
        result = values[-1] if values else None
    elif operation == "concat":
        separator = op_spec.get("separator", "")
        result = separator.join(str(v) for v in values)
    elif operation == "distinct":
        seen = []
        for v in values:
            if v not in seen:
                seen.append(v)
        result = seen
    else:
        result = len(data)

    return {output_field: result}


# ─── Cast ────────────────────────────────────────────────────────────────────

def _op_cast(data: Any, op_spec: dict) -> Any:
    if not isinstance(data, list):
        return data
    field = op_spec.get("field", "")
    to_type = op_spec.get("to", "str")
    result = []
    for item in data:
        new_item = copy.copy(item)
        val = _get_nested(item, field)
        if val is not None:
            try:
                if to_type == "int":
                    val = int(float(str(val)))
                elif to_type == "float":
                    val = float(str(val))
                elif to_type == "str":
                    val = str(val)
                elif to_type == "bool":
                    if isinstance(val, str):
                        val = val.lower() not in ("false", "0", "no", "")
                    else:
                        val = bool(val)
            except (ValueError, TypeError):
                pass
            _set_nested(new_item, field, val)
        result.append(new_item)
    return result


# ─── Dispatch ────────────────────────────────────────────────────────────────

def _apply_operation(data: Any, op_spec: dict) -> Any:
    op = op_spec.get("op")
    if op == "filter":
        return _op_filter(data, op_spec)
    elif op == "map":
        return _op_map(data, op_spec)
    elif op == "sort":
        return _op_sort(data, op_spec)
    elif op == "group_by":
        return _op_group_by(data, op_spec)
    elif op == "flatten":
        return _op_flatten(data, op_spec)
    elif op == "pick":
        return _op_pick(data, op_spec)
    elif op == "omit":
        return _op_omit(data, op_spec)
    elif op == "rename":
        return _op_rename(data, op_spec)
    elif op == "deduplicate":
        return _op_deduplicate(data, op_spec)
    elif op == "limit":
        count = op_spec.get("count", len(data) if isinstance(data, list) else 1)
        return data[:count] if isinstance(data, list) else data
    elif op == "skip":
        count = op_spec.get("count", 0)
        return data[count:] if isinstance(data, list) else data
    elif op == "aggregate":
        return _op_aggregate(data, op_spec)
    elif op == "merge":
        if isinstance(data, list):
            return {k: v for d in data for k, v in (d.items() if isinstance(d, dict) else {}.items())}
        return data
    elif op == "set":
        if isinstance(data, list):
            result = []
            for item in data:
                new_item = copy.copy(item)
                _set_nested(new_item, op_spec["field"], op_spec["value"])
                result.append(new_item)
            return result
        return data
    elif op == "cast":
        return _op_cast(data, op_spec)
    else:
        raise ValueError(f"Unknown operation: {op}")


# ─── Main function ────────────────────────────────────────────────────────────

async def data_transform(input_data: Any = None, operations: list = None, **kwargs) -> dict:
    """Transform data through a pipeline of operations."""
    # Support both positional and kwargs-based calling
    if input_data is None:
        input_data = kwargs.get("input_data")
    if operations is None:
        operations = kwargs.get("operations", [])
    if operations is None:
        operations = []

    data = copy.deepcopy(input_data)
    input_count = len(data) if isinstance(data, list) else 1

    for op_spec in operations:
        data = _apply_operation(data, op_spec)

    output_count = len(data) if isinstance(data, list) else 1
    result: dict = {"result": data}
    if isinstance(input_data, list):
        result["input_count"] = input_count
    if isinstance(data, list):
        result["output_count"] = output_count
    return result


# ─── Schema and registration ─────────────────────────────────────────────────

_DATA_TRANSFORM_SCHEMA = {
    "type": "object",
    "properties": {
        "input_data": {
            "description": "Input data — list of dicts or a single dict",
        },
        "operations": {
            "type": "array",
            "description": "List of operations to apply sequentially",
            "items": {
                "type": "object",
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": ["filter", "map", "sort", "group_by", "flatten", "pick", "omit", "rename",
                                 "deduplicate", "limit", "skip", "aggregate", "merge", "set", "cast"],
                    },
                    "field": {"type": "string"},
                    "fields": {"type": "array", "items": {"type": "string"}},
                    "operator": {"type": "string"},
                    "value": {},
                    "order": {"type": "string", "enum": ["asc", "desc"]},
                    "type": {"type": "string"},
                    "template": {"type": "string"},
                    "math_op": {"type": "string"},
                    "operand": {"type": "number"},
                    "output_field": {"type": "string"},
                    "mapping": {"type": "object"},
                    "operation": {"type": "string"},
                    "count": {"type": "integer"},
                    "separator": {"type": "string"},
                    "conditions": {"type": "array"},
                    "logic": {"type": "string", "enum": ["and", "or"]},
                    "to": {"type": "string", "enum": ["int", "float", "str", "bool"]},
                },
                "required": ["op"],
            },
        },
    },
    "required": ["input_data", "operations"],
}

_registered_tools["data_transform"] = (
    ToolDefinition(
        name="data_transform",
        description="Transform data through a safe, eval-free pipeline of 15 operations",
        parameters=_DATA_TRANSFORM_SCHEMA,
        risk_level=RiskLevel.LOW,
        resource_pattern="*",
        timeout_seconds=60,
        requires_approval=False,
    ),
    data_transform,
)
