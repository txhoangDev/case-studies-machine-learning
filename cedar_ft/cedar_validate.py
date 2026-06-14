from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import cedarpy


SchemaT = Union[str, Dict[str, Any]]


@dataclass(frozen=True)
class CedarValidation:
    ok: bool
    errors: Tuple[str, ...]


def validate_cedar_policies(policies_text: str, schema: SchemaT) -> CedarValidation:
    """Validate Cedar policy text against a Cedar schema.

    Uses `cedarpy.validate_policies`.
    """
    result = cedarpy.validate_policies(policies_text, schema)
    errors: List[str] = []
    if not result.validation_passed:
        # cedarpy returns structured ValidationError objects; stringify for logging.
        errors = [str(e) for e in (result.errors or [])]
        if not errors:
            errors = ["Validation failed (no detailed errors returned)."]
    return CedarValidation(ok=bool(result.validation_passed), errors=tuple(errors))


def first_error_string(errors: Iterable[str], *, max_chars: int = 800) -> str:
    joined = "\n".join(list(errors))
    if len(joined) <= max_chars:
        return joined
    return joined[: max_chars - 3] + "..."


def strip_code_fences(text: str) -> str:
    """Best-effort cleanup when a model wraps Cedar in Markdown fences."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        t = t.strip("`").strip()
        # If a language is specified on the first line (e.g., ```cedar), drop it.
        lines = t.splitlines()
        if lines and lines[0].strip().isalpha():
            t = "\n".join(lines[1:]).strip()
    return t.strip()

