"""EXP-04: which MedQA test questions each model gets right, and where the medical models and Phi-3.5 disagree.

Reads the per-question results that already exist (no model runs) and writes
- `processed/picks_test.csv`: one row per test question with the correct letter, the letter each of the three
  models picks under the bare harness prompt (Protocol C) and under the chat prompt (Protocol A), and which of the
  models are right under each prompt;
- `vault/attachments/exp04_venn_plain_text.png` and `exp04_venn_chat.png`: a three-circle diagram of those counts,
  with the questions Phi-3.5 misses and a medical model gets highlighted.

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
FIGURES = REPO / "vault" / "attachments"
MODELS = {"phi": "Phi-3.5", "mediphi": "MediPhi", "mediphi_instruct": "MediPhi-Instruct"}
PROMPTS = {"bare": "plain-text prompt (the paper's setting)", "chat": "chat prompt"}


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


def who_is_right(flags: tuple[bool, bool, bool]) -> str:
    """A readable name for one region of the diagram, e.g. "MediPhi + MediPhi-Instruct"."""
    names = [name for name, flag in zip(MODELS.values(), flags, strict=True) if flag]
    return " + ".join(names) if names else "none"


def draw_venn(counts: Counter, prompt: str, path: Path) -> None:
    """Three circles (areas not to scale) with the number of questions in each region. The regions where Phi-3.5
    is wrong and a medical model is right are the highlighted ones."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    centers = {"phi": (0.0, 0.62), "mediphi": (-0.58, -0.38), "mediphi_instruct": (0.58, -0.38)}
    neutral, highlight, ink, accent = "#E6EAF0", "#FBD3A6", "#1F2933", "#9A3412"
    figure, axes = plt.subplots(figsize=(11.2, 6.6), dpi=200)
    # the two medical circles in the highlight color, then Phi-3.5's circle on top in the neutral one: what stays
    # highlighted is exactly "a medical model right, Phi-3.5 wrong"
    for model in ("mediphi", "mediphi_instruct"):
        axes.add_patch(Circle(centers[model], 1.0, facecolor=highlight, edgecolor="none", zorder=1))
    axes.add_patch(Circle(centers["phi"], 1.0, facecolor=neutral, edgecolor="none", zorder=2))
    for model, style in (("phi", "-"), ("mediphi", (0, (6, 3))), ("mediphi_instruct", "-")):
        axes.add_patch(Circle(centers[model], 1.0, facecolor="none", edgecolor=ink, linewidth=1.8, linestyle=style, zorder=3))  # fmt: skip

    def n(phi: bool, mediphi: bool, instruct: bool) -> int:
        return counts[phi, mediphi, instruct]

    regions = {  # (Phi-3.5, MediPhi, MediPhi-Instruct) right -> where its number goes
        (True, False, False): (0.0, 1.22), (False, True, False): (-1.12, -0.72), (False, False, True): (1.12, -0.72),
        (True, True, False): (-0.68, 0.42), (True, False, True): (0.68, 0.42), (False, True, True): (0.0, -0.84),
        (True, True, True): (0.0, 0.02),
    }  # fmt: skip
    for flags, (x, y) in regions.items():
        focus = not flags[0]
        axes.text(x, y, f"{counts[flags]:,}", ha="center", va="center", fontsize=24 if focus else 20,
                  fontweight="bold" if focus else "normal", color=accent if focus else ink, zorder=4)  # fmt: skip
    total = sum(counts.values())
    right = {model: sum(v for flags, v in counts.items() if flags[i]) for i, model in enumerate(MODELS)}

    def note(x: float, y: float, text: str, *, size: float = 11.5, bold: bool = False, color: str = ink, align: str = "center") -> None:  # fmt: skip
        axes.text(x, y, text, ha=align, va="center", fontsize=size, fontweight="bold" if bold else "normal", color=color,
                  linespacing=1.4)  # fmt: skip

    # each circle is named next to it, with the number of questions that model gets right
    note(0.0, 1.76, f"{MODELS['phi']} right: {right['phi']:,}", size=13, bold=True)
    note(
        -1.7,
        -0.38,
        f"{MODELS['mediphi']}\nright: {right['mediphi']:,}\n(dashed circle)",
        size=13,
        bold=True,
        align="right",
    )
    note(
        1.7,
        -0.38,
        f"{MODELS['mediphi_instruct']}\nright: {right['mediphi_instruct']:,}",
        size=13,
        bold=True,
        align="left",
    )
    missed = n(False, True, False) + n(False, True, True) + n(False, False, True)
    note(2.35, 1.62, "Phi-3.5 wrong,\na medical model right:", size=12.5, bold=True, color=accent)
    note(2.35, 1.24, f"{missed:,} questions", size=14, bold=True, color=accent)
    note(2.35, 0.82, f"{n(False, True, False):,} MediPhi only\n{n(False, True, True):,} both medical models\n"
         f"{n(False, False, True):,} MediPhi-Instruct only")  # fmt: skip
    note(-2.35, 1.62, "None of the three right:", size=12.5, bold=True)
    note(-2.35, 1.36, f"{n(False, False, False):,} questions", size=14, bold=True)
    axes.set_title(f"MedQA test, {total:,} questions, {PROMPTS[prompt]}:\nwhich models answer each question correctly",
                   fontsize=14, color=ink, pad=14)  # fmt: skip
    axes.text(0.0, -1.62, "Each number is a count of questions. Circle areas are not to scale.", ha="center",
              va="center", fontsize=10, color="#52606D")  # fmt: skip
    axes.set_xlim(-3.45, 3.45)
    axes.set_ylim(-1.78, 1.98)
    axes.set_aspect("equal")
    axes.axis("off")
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=3, help="how many example questions to print")
    args = parser.parse_args()
    items, picks = load_picks()
    gold = [item["answer"] for item in items]
    n = len(items)
    right = {key: [p == g for p, g in zip(letters, gold, strict=True)] for key, letters in picks.items()}
    flags = {
        prompt: [tuple(right[model, prompt][i] for model in MODELS) for i in range(n)] for prompt in PROMPTS
    }

    print(f"MedQA test split, {n} questions: right / wrong")
    for prompt in PROMPTS:
        for model, name in MODELS.items():
            k = sum(right[model, prompt])
            print(f"  {prompt:<4} {name:<17} {k} / {n - k}  ({100 * k / n:.2f} %)")
    for prompt in PROMPTS:
        counts = Counter(flags[prompt])
        print(f"\n{prompt} prompt, questions by which models are right:")
        for key, count in sorted(counts.items(), reverse=True):
            print(f"  {who_is_right(key):<45} {count}")
        name = "plain_text" if prompt == "bare" else prompt
        draw_venn(counts, prompt, FIGURES / f"exp04_venn_{name}.png")
        print("  wrote", FIGURES / f"exp04_venn_{name}.png")

    # the questions Filippos's plan starts from: MediPhi-Instruct right, Phi-3.5 wrong, bare prompt
    gain = [i for i in range(n) if right["mediphi_instruct", "bare"][i] and not right["phi", "bare"][i]]
    print(
        f"\nBare prompt: {len(gain)} questions MediPhi-Instruct gets right and Phi-3.5 does not. On those questions:"
    )
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
        writer.writerow(["id", "correct", *[f"{model}_{prompt}" for prompt in PROMPTS for model in MODELS],
                         "right_bare", "right_chat", "question_start"])  # fmt: skip
        for i, item in enumerate(items):
            writer.writerow([item["id"], gold[i], *[picks[model, prompt][i] for prompt in PROMPTS for model in MODELS],
                             who_is_right(flags["bare"][i]), who_is_right(flags["chat"][i]), item["question"][:100]])  # fmt: skip
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
