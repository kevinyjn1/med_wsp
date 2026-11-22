"""EXP-04b: MedQA with free-form reasoning (greedy generation), to compare with letter-logit scoring.

Each model reasons step by step and must end with "Answer: X". The last such line is parsed. This is the
protocol where a fine-tune could differ from its base even when their first-token letter choices agree.

Usage:  python experiments/exp_04_medqa_baseline/run_cot.py --split test --limit 400 [--models ...]
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp import medqa, models

OUT = Path(__file__).resolve().parent / "processed"
COT_INSTRUCTION = (
    "The following is a multiple choice question about medical knowledge. Think step by step, then finish "
    'with a final line of the form "Answer: X", where X is the letter of the correct option.'
)
ANSWER_RE = re.compile(r"answer\s*(?:is)?\s*[:\-]?\s*\**\(?([ABCD])\b", re.IGNORECASE)


def cot_prompt(tokenizer, item: dict) -> str:
    options = "\n".join(f"{letter}. {item['options'][letter]}" for letter in medqa.LETTERS)
    content = f"{COT_INSTRUCTION}\n\n{item['question']}\n\n{options}"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )


def parse_answer(text: str) -> str | None:
    found = ANSWER_RE.findall(text)
    return found[-1].upper() if found else None


@torch.no_grad()
def run(model, tokenizer, items: list[dict], batch_size: int, max_new_tokens: int) -> list[dict]:
    prompts = [cot_prompt(tokenizer, item) for item in items]
    order = sorted(range(len(items)), key=lambda i: len(prompts[i]))
    records: dict[int, dict] = {}
    for start in range(0, len(order), batch_size):
        batch = order[start : start + batch_size]
        enc = tokenizer(
            [prompts[i] for i in batch], return_tensors="pt", padding=True, add_special_tokens=False
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id
        )
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        for row, i in enumerate(batch):
            text = tokenizer.decode(new_tokens[row], skip_special_tokens=True)
            pred = parse_answer(text)
            n_tokens = int((new_tokens[row] != tokenizer.pad_token_id).sum())
            records[i] = {"id": items[i]["id"], "answer": items[i]["answer"], "pred": pred, "parsed": pred is not None,
                          "correct": pred == items[i]["answer"], "new_tokens": n_tokens, "text": text}  # fmt: skip
        del out, enc
        torch.mps.empty_cache()
        print(f"  {min(start + batch_size, len(order))}/{len(order)}", flush=True)
    return [records[i] for i in range(len(items))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["phi", "mediphi", "mediphi_instruct"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    items = medqa.load_medqa(args.split)[: args.limit]
    tokenizer = models.load_tokenizer("phi")
    for name in args.models:
        out = OUT / f"cot_{name}_{args.split}_{len(items)}.jsonl"
        if out.exists():
            print(f"{name}: already done", flush=True)
            continue
        model = models.load_model(name, eager_attention=False)
        start = time.time()
        records = run(model, tokenizer, items, args.batch_size, args.max_new_tokens)
        medqa.save_jsonl(records, out)
        n = len(records)
        print(json.dumps({
            "model": name, "split": args.split, "n": n, "acc": round(sum(r["correct"] for r in records) / n, 4),
            "parsed": round(sum(r["parsed"] for r in records) / n, 4),
            "mean_new_tokens": round(sum(r["new_tokens"] for r in records) / n, 1), "seconds": round(time.time() - start),
        }), flush=True)  # fmt: skip
        del model
        gc.collect()
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
