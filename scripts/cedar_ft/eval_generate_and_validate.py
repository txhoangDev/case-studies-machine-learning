from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cedar_ft.cedar_validate import first_error_string, strip_code_fences, validate_cedar_policies
from cedar_ft.jsonl import read_jsonl, write_json


def extract_schema_and_user(messages: Any) -> Tuple[Optional[dict], str]:
    schema: Optional[dict] = None
    user_text = ""
    if isinstance(messages, list):
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") == "user":
                user_text = str(msg.get("content") or "")
                try:
                    payload = json.loads(user_text)
                    if isinstance(payload, dict) and isinstance(payload.get("cedar_schema"), dict):
                        schema = payload["cedar_schema"]
                except Exception:
                    pass
    return schema, user_text


@torch.inference_mode()
def generate_policy(model, tokenizer, messages: List[Dict[str, str]], *, max_new_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    out = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    completion_ids = out[0][inputs.shape[-1] :]
    text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return strip_code_fences(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model name/path (base or merged)")
    ap.add_argument("--data", type=Path, default=Path("datasets/cedar_policy_sft.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("datasets/cedar_eval_report.json"))
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")
    model.eval()

    total = 0
    valid = 0
    failures: List[Dict[str, Any]] = []

    for row in read_jsonl(args.data):
        messages = row.get("messages")
        if not isinstance(messages, list):
            continue

        schema, _ = extract_schema_and_user(messages)
        if not schema:
            continue

        pred = generate_policy(model, tokenizer, messages[:-1], max_new_tokens=args.max_new_tokens)
        res = validate_cedar_policies(pred, schema)
        total += 1
        if res.ok:
            valid += 1
        else:
            failures.append(
                {
                    "use_case": row.get("use_case"),
                    "prediction": pred,
                    "error": first_error_string(res.errors),
                }
            )

        if args.limit and total >= args.limit:
            break

    report = {
        "model": args.model,
        "data": str(args.data),
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "valid_rate": (valid / total) if total else 0.0,
        "first_failures": failures[:10],
    }
    write_json(args.out, report)
    print(json.dumps(report, indent=2))
    print(f"Wrote report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
