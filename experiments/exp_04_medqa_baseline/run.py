"""EXP-04: Phi-3.5-mini-instruct vs MediPhi on MedQA (USMLE, 4 options), letter-logit scoring.

Scores both models on the test split (headline numbers), the dev split and a fixed pool of training
questions (the discovery pool, so circuit discovery and edits never touch test questions).

Usage:  python experiments/exp_04_medqa_baseline/run.py [--models phi mediphi] [--train-pool 3000]
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp import medqa, models

OUT = Path(__file__).resolve().parent / "processed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["phi", "mediphi"])
    parser.add_argument("--splits", nargs="+", default=["test", "dev", "train"])
    parser.add_argument("--train-pool", type=int, default=3000, help="first N training questions")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--style", default="chat", choices=medqa.STYLES,
                        help='"chat": our chat prompt (files <model>_<split>.jsonl); "bare": the lm-evaluation-harness prompt (bare_<model>_<split>.jsonl)')  # fmt: skip
    args = parser.parse_args()

    data = {split: medqa.load_medqa(split) for split in args.splits}
    if "train" in data:
        data["train"] = data["train"][: args.train_pool]
    tokenizer = models.load_tokenizer("phi")

    for name in args.models:
        model = models.load_model(name)
        for split, items in data.items():
            out = OUT / f"{'bare_' if args.style == 'bare' else ''}{name}_{split}.jsonl"
            if out.exists() and len(medqa.load_jsonl(out)) == len(items):
                print(f"{name} {split}: already done", flush=True)
                continue
            start = time.time()
            records = medqa.evaluate(
                model, tokenizer, items, batch_size=args.batch_size, progress=False, style=args.style
            )
            medqa.save_jsonl(records, out)
            accuracy = sum(r["correct"] for r in records) / len(records)
            print(json.dumps({"model": name, "style": args.style, "split": split, "n": len(records), "acc": round(accuracy, 4),
                              "seconds": round(time.time() - start)}), flush=True)  # fmt: skip
        del model
        gc.collect()
        torch.mps.empty_cache()


if __name__ == "__main__":
    main()
