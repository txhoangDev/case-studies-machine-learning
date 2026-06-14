from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

# Allow running as a script without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def require(package: str) -> None:
    try:
        __import__(package)
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            f"Missing optional dependency `{package}`. Install your training deps, then retry. ({e})"
        ) from e


require("datasets")
require("peft")
require("transformers")
require("trl")
require("torch")

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback

from cedar_ft.cedar_validate import strip_code_fences, validate_cedar_policies


def to_text(tokenizer, example: Dict[str, Any]) -> Dict[str, Any]:
    example["text"] = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return example


class CedarEvalCallback(TrainerCallback):
    def __init__(self, tokenizer, *, max_new_tokens: int = 256, limit: int = 64):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.limit = limit

    @torch.inference_mode()
    def on_evaluate(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer")
        model = kwargs.get("model")
        eval_dataset = kwargs.get("eval_dataset")
        if trainer is None or model is None or eval_dataset is None:
            return

        # Small generation-based eval: "does it validate under cedarpy?"
        total = 0
        valid = 0
        for ex in eval_dataset:
            messages = ex.get("messages")
            schema = ex.get("cedar_schema")
            if not isinstance(messages, list) or not isinstance(schema, dict):
                continue

            inputs = self.tokenizer.apply_chat_template(
                messages[:-1],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)

            out = model.generate(
                inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            completion_ids = out[0][inputs.shape[-1] :]
            pred = strip_code_fences(self.tokenizer.decode(completion_ids, skip_special_tokens=True))
            res = validate_cedar_policies(pred, schema)
            total += 1
            valid += int(res.ok)
            if self.limit and total >= self.limit:
                break

        if total:
            trainer.log({"cedar_valid_rate": valid / total, "cedar_eval_n": total})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Base HF model name/path")
    ap.add_argument("--train", type=Path, default=Path("datasets/cedar_policy_sft.jsonl"))
    ap.add_argument("--eval", type=Path, default=Path("datasets/cedar_policy_sft.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("lora_out/cedar"))
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--eval-limit", type=int, default=64)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto")

    ds_train = load_dataset("json", data_files=str(args.train), split="train")
    ds_eval = load_dataset("json", data_files=str(args.eval), split="train")

    ds_train = ds_train.map(lambda ex: to_text(tokenizer, ex), remove_columns=[])
    ds_eval = ds_eval.map(lambda ex: to_text(tokenizer, ex), remove_columns=[])

    # LoRA config: adjust as needed.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # TRL API differences across versions: prefer SFTConfig if available.
    from trl import SFTTrainer

    try:
        from trl import SFTConfig  # newer TRL

        sft_args = SFTConfig(
            output_dir=str(args.out),
            dataset_text_field="text",
            max_length=args.max_length,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=200,
            save_steps=200,
        )

        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=ds_train,
            eval_dataset=ds_eval,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
    except Exception:
        # Older TRL: fall back to tokenizer= + dataset_text_field= kwargs.
        from transformers import TrainingArguments

        sft_args = TrainingArguments(
            output_dir=str(args.out),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            learning_rate=2e-4,
            num_train_epochs=1,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
        )
        trainer = SFTTrainer(
            model=model,
            args=sft_args,
            train_dataset=ds_train,
            eval_dataset=ds_eval,
            peft_config=peft_config,
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=args.max_length,
        )

    trainer.add_callback(CedarEvalCallback(tokenizer, max_new_tokens=args.max_new_tokens, limit=args.eval_limit))
    trainer.train()
    trainer.save_model(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
