"""EXP-05b: which layers carry the difference between a fine-tune (donor) and its base on MedQA?

Two interventions on questions from the discovery pool (MedQA dev + first 3,000 training questions):

  activation patching  the donor's attention output, MLP output or residual stream of one layer is placed
                       into the base model's forward pass (all token positions)
  weight swapping      one layer's attention weights or MLP weights are copied from the donor into the base

Question sets, defined by letter scoring of both models:
  gain   donor right, base wrong      loss   base right, donor wrong      both   a random sample both get right

For every question and intervention we store the gold-answer margin (log-prob of the gold letter minus the
best other letter). Recovery = (patched - base) / (donor - base), computed per set afterwards.

Usage:  python experiments/exp_05_medical_circuit/patch.py --donor mediphi --per-set 60 --mode activation weights

The script takes the GPU locks itself, once per question (activation: both locks, two models are loaded) or once
per weight configuration ("batch"), so a lab pass can run in between; do not wrap it in the lock command.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp import medqa, models
from wsp.checkpoints import Checkpoint
from wsp.gpu_lock import gpu_lock, gpu_locks
from wsp.patching import COMPONENTS, Patch, capture, layer_tensor_names, run_patched, swapped_weights

EXP = Path(__file__).resolve().parent
BASELINE = EXP.parent / "exp_04_medqa_baseline" / "processed"


def question_sets(donor: str, per_set: int, style: str = "chat", seed: int = 0) -> dict[str, list[dict]]:
    prefix = "bare_" if style == "bare" else ""
    items, base, tuned = [], [], []
    for split in ("dev", "train"):
        base_records = medqa.load_jsonl(BASELINE / f"{prefix}phi_{split}.jsonl")
        items += medqa.load_medqa(split)[: len(base_records)]
        base += base_records
        tuned += medqa.load_jsonl(BASELINE / f"{prefix}{donor}_{split}.jsonl")
    assert [r["id"] for r in base] == [r["id"] for r in tuned] == [i["id"] for i in items]
    groups = {"gain": [], "loss": [], "both": []}
    for item, b, t in zip(items, base, tuned, strict=True):
        key = (
            "gain"
            if t["correct"] and not b["correct"]
            else "loss"
            if b["correct"] and not t["correct"]
            else "both"
            if b["correct"] and t["correct"]
            else None
        )
        if key:
            groups[key].append(item)
    rng = random.Random(seed)
    return {key: rng.sample(group, min(per_set, len(group))) for key, group in groups.items()}


def margin_of(logits: torch.Tensor, letter_ids: dict, gold: str) -> list[float]:
    return [round(float(x), 4) for x in medqa.gold_margins(logits, letter_ids, gold)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", default="mediphi")
    parser.add_argument("--per-set", type=int, default=60)
    parser.add_argument(
        "--mode", nargs="+", default=["activation", "weights"], choices=["activation", "weights"]
    )
    parser.add_argument("--rows", type=int, default=24, help="patches evaluated per forward pass")
    parser.add_argument(
        "--style", default="chat", choices=medqa.STYLES, help="prompt format (see wsp/medqa.py)"
    )
    args = parser.parse_args()

    out = EXP / "processed" / (f"patch_{args.donor}" + ("_bare" if args.style == "bare" else ""))
    out.mkdir(parents=True, exist_ok=True)
    sets = question_sets(args.donor, args.per_set, args.style)
    (out / "question_sets.json").write_text(
        json.dumps({k: [i["id"] for i in v] for k, v in sets.items()}, indent=1)
    )
    print({k: len(v) for k, v in sets.items()}, flush=True)

    tokenizer = models.load_tokenizer("phi")
    letter_ids = medqa.letter_token_ids(tokenizer, args.style)
    base = models.load_model("phi")
    n_layers = base.config.num_hidden_layers

    if "activation" in args.mode:
        donor = models.load_model(args.donor)
        patches = [Patch(layer, component) for layer in range(n_layers) for component in COMPONENTS]
        path = out / "activation.jsonl"
        done = {json.loads(line)["id"] for line in path.read_text().splitlines()} if path.exists() else set()
        start = time.time()
        with open(path, "a") as f:
            for set_name, items in sets.items():
                for item in items:
                    if item["id"] in done:
                        continue
                    with gpu_locks(f"exp05 patching {item['id']}", ("batch", "gpu")):
                        enc = tokenizer(
                            [medqa.build_prompt(tokenizer, item, args.style)],
                            return_tensors="pt",
                            add_special_tokens=False,
                        )
                        enc = {k: v.to(base.device) for k, v in enc.items()}
                        donor_cache = capture(donor, enc, layers=list(range(n_layers)))
                        base_margin = margin_of(
                            base(**enc, use_cache=False).logits[:, -1, :], letter_ids, item["answer"]
                        )[0]
                        donor_margin = margin_of(donor_cache["logits"], letter_ids, item["answer"])[0]
                        margins = []
                        for begin in range(0, len(patches), args.rows):
                            logits = run_patched(base, enc, donor_cache, patches[begin : begin + args.rows])
                            margins += margin_of(logits, letter_ids, item["answer"])
                        del donor_cache, enc
                        models.release_gpu_cache()
                    f.write(json.dumps({"id": item["id"], "set": set_name, "base": base_margin, "donor": donor_margin,
                                        "patched": dict(zip([p.label() for p in patches], margins, strict=True))}) + "\n")  # fmt: skip
                    f.flush()
            print(f"activation patching done in {time.time() - start:.0f}s", flush=True)
        del donor
        torch.mps.empty_cache()

    if "weights" in args.mode:
        checkpoint = Checkpoint(args.donor)
        everything = [item for items in sets.values() for item in items]
        set_of = {item["id"]: name for name, items in sets.items() for item in items}
        path = out / "weights.jsonl"
        done = (
            {json.loads(line)["config"] for line in path.read_text().splitlines()} if path.exists() else set()
        )

        def evaluate(label: str) -> None:
            if label in done:
                return
            with gpu_lock(f"exp05 weight swap {label}", name="batch"):
                records = medqa.evaluate(
                    base, tokenizer, everything, batch_size=16, progress=False, style=args.style
                )
            row = {"config": label, "margins": {r["id"]: round(r["margin"], 4) for r in records},
                   "acc": {name: sum(r["correct"] for r in records if set_of[r["id"]] == name) / len(items)
                           for name, items in sets.items()}}  # fmt: skip
            with open(path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(label, row["acc"], flush=True)

        evaluate("base")
        for layer in range(n_layers):
            for part in ("attn", "mlp"):
                tensors = {name: checkpoint.get(name) for name in layer_tensor_names(layer, part)}
                with swapped_weights(base, tensors):
                    evaluate(f"L{layer}.{part}")
        full = {
            name: checkpoint.get(name) for name in checkpoint.names() if name in dict(base.named_parameters())
        }
        with swapped_weights(base, full):
            evaluate("all")


if __name__ == "__main__":
    with torch.no_grad():
        main()
