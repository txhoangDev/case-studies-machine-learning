from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import sys

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cedar_ft.cedar_validate import first_error_string, strip_code_fences, validate_cedar_policies
from cedar_ft.jsonl import read_jsonl, write_json


def extract_assistant(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path, default=Path("datasets/cedar_policy_sft.jsonl"))
    ap.add_argument("--report", dest="report_path", type=Path, default=Path("datasets/cedar_policy_sft.validation_report.json"))
    args = ap.parse_args()

    total = 0
    passed = 0
    first_failure: Tuple[str, str] | None = None

    for row in read_jsonl(args.in_path):
        total += 1
        schema = row.get("cedar_schema")
        policy = strip_code_fences(extract_assistant(row.get("messages")))
        if not schema or not policy:
            if first_failure is None:
                first_failure = ("Missing schema or policy", json.dumps(row)[:500])
            continue
        res = validate_cedar_policies(policy, schema)
        if res.ok:
            passed += 1
            continue
        if first_failure is None:
            first_failure = (first_error_string(res.errors), policy[:500])

    report = {
        "input": str(args.in_path),
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": (passed / total) if total else 0.0,
        "first_failure": {"error": first_failure[0], "policy_excerpt": first_failure[1]} if first_failure else None,
    }
    write_json(args.report_path, report)
    print(json.dumps(report, indent=2))
    print(f"Wrote report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
