"""EXP-04: which MedQA test questions each model gets right, and where the medical models and Phi-3.5 disagree.

Reads the per-question results that already exist (no model runs) and writes
`processed/picks_test.csv`: one row per test question with the correct letter and the letter each of the three
models picks under the bare harness prompt (Protocol C) and under the chat prompt (Protocol A).

Usage:  python experiments/exp_04_medqa_baseline/contrast.py [--examples 3]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp import medqa

PROCESSED = Path(__file__).resolve().parent / "processed"
MODELS = {"phi": "Phi-3.5", "mediphi": "MediPhi", "mediphi_instruct": "MediPhi-Instruct"}


def load_picks() -> tuple[list[dict], dict[tuple[str, str], list[str]]]:
    """The test questions and, per (model, prompt), the picked letter for each of them in the same order."""
    items = medqa.load_medqa("test")
    picks = {}
    for model in MODELS:
        bare = medqa.load_jsonl(PROCESSED / f"lmeval_raw_{model}_test.jsonl")
        chat = medqa.load_jsonl(PROCESSED / f"{model}_test.jsonl")
        # the harness-format files index the same questions in the same order; check it through the answer key
        assert [medqa.LETTERS[r["label"]] for r in bare] == [item["answer"] for item in items], model
        assert [r["id"] for r in chat] == [item["id"] for item in items], model
        picks[model, "bare"] = [medqa.LETTERS[r["pred"]] for r in bare]
        picks[model, "chat"] = [r["pred"] for r in chat]
    return items, picks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=3, help="how many example questions to print")
    args = parser.parse_args()
    items, picks = load_picks()
    gold = [item["answer"] for item in items]
    n = len(items)
    right = {key: [p == g for p, g in zip(letters, gold, strict=True)] for key, letters in picks.items()}

    print(f"MedQA test split, {n} questions: right / wrong")
    for prompt in ("bare", "chat"):
        for model, name in MODELS.items():
            k = sum(right[model, prompt])
            print(f"  {prompt:<4} {name:<17} {k} / {n - k}  ({100 * k / n:.2f} %)")

    for other in ("mediphi_instruct", "mediphi"):
        for prompt in ("bare", "chat"):
            a, b = right["phi", prompt], right[other, prompt]
            cells = Counter(zip(a, b, strict=True))
            print(f"  {prompt} prompt, Phi-3.5 vs {MODELS[other]}: both right {cells[True, True]}, only {MODELS[other]} "
                  f"{cells[False, True]}, only Phi-3.5 {cells[True, False]}, neither {cells[False, False]}")  # fmt: skip

    # the questions Filippos's plan starts from: MediPhi-Instruct right, Phi-3.5 wrong, bare prompt
    gain = [i for i in range(n) if right["mediphi_instruct", "bare"][i] and not right["phi", "bare"][i]]
    print(f"\nBare prompt: {len(gain)} questions only MediPhi-Instruct gets right. On those questions:")
    print("  Phi-3.5's wrong pick:", dict(sorted(Counter(picks["phi", "bare"][i] for i in gain).items())))
    print("  correct letter:      ", dict(sorted(Counter(gold[i] for i in gain).items())))
    print("  Phi-3.5 is right under the chat prompt on", sum(right["phi", "chat"][i] for i in gain))
    print("  MediPhi is right (bare prompt) on", sum(right["mediphi", "bare"][i] for i in gain))
    print(
        "  MediPhi-Instruct is right under the chat prompt on",
        sum(right["mediphi_instruct", "chat"][i] for i in gain),
    )

    with open(PROCESSED / "picks_test.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "correct", *[f"{model}_{prompt}" for prompt in ("bare", "chat") for model in MODELS],
                         "only_mediphi_instruct_right_bare", "question_start"])  # fmt: skip
        for i, item in enumerate(items):
            writer.writerow([item["id"], gold[i], *[picks[model, prompt][i] for prompt in ("bare", "chat") for model in MODELS],
                             int(i in set(gain)), item["question"][:100]])  # fmt: skip
    print("\nwrote", PROCESSED / "picks_test.csv")

    # examples: the shortest of those questions, so they can be read at a glance
    for i in sorted(gain, key=lambda i: len(items[i]["question"]))[: args.examples]:
        item = items[i]
        print(f"\n[{item['id']}] {item['question']}")
        for letter in medqa.LETTERS:
            print(f"   {letter}. {item['options'][letter]}")
        print(f"   correct: {gold[i]} | bare prompt: " + ", ".join(f"{name} {picks[m, 'bare'][i]}" for m, name in MODELS.items())
              + " | chat prompt: " + ", ".join(f"{name} {picks[m, 'chat'][i]}" for m, name in MODELS.items()))  # fmt: skip


if __name__ == "__main__":
    main()
