"""EXP-07: training-free layer looping on MedQA (single-pass letter scoring).

Re-runs a window of decoder layers at inference with one of three update rules (see wsp/looping.py) and
compares every configuration with the unmodified model on the same questions (paired).

Sweeps (`--sweep`):
  defaults   the papers' default recipes: 4-layer window near mid-depth, Chen K=2/3 and Lys uniform R=3
  position   4-layer windows at starts 2, 4, ..., 26 with Chen K=2 and naive K=2
  around     around `--center`: lengths 2/3/4/6, Chen K 2/3/4 with beta 0 and 0.5, Lys uniform/moving

Usage:  python experiments/exp_07_layer_looping/run.py --model phi --split dev --limit 600 --sweep defaults position

The script takes the "batch" GPU lock once per configuration (wsp/gpu_lock.py); do not wrap it in the lock command.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp import medqa, models
from wsp.gpu_lock import gpu_lock
from wsp.looping import LoopSpec, looped

OUT = Path(__file__).resolve().parent / "processed"


def sweep_specs(name: str, center: int) -> list[LoopSpec]:
    if name == "defaults":  # Phi-3.5-mini has 32 layers: [14, 17] is centered at depth fraction 0.48
        return [
            LoopSpec(14, 4, "chen", 2), LoopSpec(14, 4, "chen", 3),
            LoopSpec(14, 4, "lys", 3, mode="uniform"), LoopSpec(14, 4, "naive", 2),
        ]  # fmt: skip
    if name == "position":
        return [LoopSpec(start, 4, rule, 2) for start in range(2, 28, 2) for rule in ("chen", "naive")]
    if name == "around":
        specs = [
            LoopSpec(center, n, "chen", k, beta=beta)
            for n in (2, 3, 4, 6)
            for k in (2, 3, 4)
            for beta in (0.0, 0.5)
        ]
        specs += [LoopSpec(center, 4, "lys", r, mode="uniform") for r in (2, 3)]
        specs += [
            LoopSpec(center, 4, "lys", r, mode="moving", eta=eta) for r in (2, 3) for eta in (0.25, 0.5, 0.75)
        ]
        return specs
    raise ValueError(name)


def summarize(label: str, records: list[dict], baseline: list[dict] | None, seconds: float) -> dict:
    correct = np.array([r["correct"] for r in records])
    margin = np.array([r["margin"] for r in records])
    row = {"config": label, "n": len(records), "acc": float(correct.mean()), "mean_margin": float(margin.mean()),
           "seconds": round(seconds)}  # fmt: skip
    if baseline is not None:
        base_correct = np.array([r["correct"] for r in baseline])
        base_margin = np.array([r["margin"] for r in baseline])
        fixed, broken = int((correct & ~base_correct).sum()), int((~correct & base_correct).sum())
        delta = margin - base_margin
        row.update({
            "acc_delta": float(correct.mean() - base_correct.mean()), "fixed": fixed, "broken": broken,
            "mcnemar_p": float(binomtest(fixed, fixed + broken, 0.5).pvalue) if fixed + broken else 1.0,
            "margin_delta": float(delta.mean()), "margin_delta_se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
        })  # fmt: skip
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="phi")
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--sweep", nargs="+", default=["defaults", "position"])
    parser.add_argument("--center", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    items = medqa.load_medqa(args.split)[: args.limit]
    tokenizer = models.load_tokenizer("phi")
    model = models.load_model(args.model)
    tag = f"{args.model}_{args.split}_{len(items)}"
    results_path = OUT / f"scan_{tag}.jsonl"
    per_question = OUT / f"per_question_{tag}"
    per_question.mkdir(parents=True, exist_ok=True)
    done = (
        {json.loads(line)["config"] for line in results_path.read_text().splitlines()}
        if results_path.exists()
        else set()
    )

    def evaluate(label: str, spec: LoopSpec | None, baseline: list[dict] | None) -> list[dict]:
        path = per_question / f"{label}.jsonl"
        if label in done and path.exists():
            return medqa.load_jsonl(path)
        # One lock per configuration, not per sweep, so a lab pass can run between two configurations.
        with gpu_lock(f"exp07 {label}", name="batch"):
            start = time.time()
            if spec is None:
                records = medqa.evaluate(model, tokenizer, items, batch_size=args.batch_size, progress=False)
            else:
                with looped(model, spec):
                    records = medqa.evaluate(
                        model, tokenizer, items, batch_size=args.batch_size, progress=False
                    )
        medqa.save_jsonl(records, path)
        row = summarize(label, records, baseline, time.time() - start)
        if spec is not None:
            row.update({"start": spec.start, "length": spec.length, "rule": spec.rule, "steps": spec.steps,
                        "beta": spec.beta, "mode": spec.mode, "eta": spec.eta})  # fmt: skip
        with open(results_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        return records

    baseline = evaluate("baseline", None, None)
    for sweep in args.sweep:
        for spec in sweep_specs(sweep, args.center):
            evaluate(spec.label(), spec, baseline)


if __name__ == "__main__":
    main()
