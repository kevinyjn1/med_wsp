"""Score EXP-01: parse Astra's final answers and compare with MIT's ground truth and MIT's own runs.

Ground truth comes from MIT's eval/heads.json (each head measured by running the full network; a
copying head is one with induction score > 0.2). Writes processed/scores.json and prints a table.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent
MIT = EXP.parents[1] / "external" / "lm_understand-logs" / "exp3_language"
COPY_CUTOFF = 0.2


def parse_answer(text: str) -> dict:
    corpus = re.search(r"corpus:\s*\**\s*(python|romaji|english)", text, re.IGNORECASE)
    order = re.search(r"order:\s*\**\s*(as is|shuffled)", text, re.IGNORECASE)
    probs = {
        k: float(v)
        for k, v in re.findall(r"(python|romaji|english|as is|shuffled)\s*=\s*([0-9.]+)", text, re.IGNORECASE)
    }
    labels = {}
    for head, label in re.findall(
        r"(L[0-3]H[0-5])\W{0,6}(previous|local|broad|copy|unclear)", text, re.IGNORECASE
    ):
        labels.setdefault(head, label.lower())
    return {
        "corpus_guess": corpus.group(1).lower() if corpus else None,
        "order_guess": order.group(1).lower() if order else None,
        "probabilities": probs, "head_labels": labels,
        "copy_named": sorted(h for h, label in labels.items() if label == "copy"),
    }  # fmt: skip


def mit_probability(probabilities, language: str):
    """MIT stores the probabilities either as a dict or as the raw 'python=0.56 romaji=...' line."""
    if isinstance(probabilities, dict):
        return probabilities.get(language)
    found = re.search(rf"{language}\s*=\s*([0-9.]+)", probabilities or "")
    return float(found.group(1)) if found else None


def main() -> None:
    truth_heads = json.loads((MIT / "eval" / "heads.json").read_text())
    mit_scores = {
        s["id"].split("_")[0] + "_" + s["id"].split("_")[-1]: s
        for s in json.loads((MIT / "eval" / "scores.json").read_text())
    }
    rows = []
    for run_dir in sorted((EXP / "raw" / "runs").iterdir()):
        final = run_dir / "final_answer.md"
        if not final.exists():
            continue
        language = run_dir.name.split("_")[0]
        parsed = parse_answer(final.read_text())
        heads = truth_heads[f"{language}_L4_H6_d384_s0"]
        true_copy = sorted(h for h, v in heads.items() if v["induct"] > COPY_CUTOFF)
        state = json.loads((run_dir / "state.json").read_text())
        mit = mit_scores.get(run_dir.name, {})
        named = set(parsed["copy_named"])
        rows.append({
            "run": run_dir.name, "truth": language, **parsed,
            "corpus_correct": parsed["corpus_guess"] == language,
            "p_correct": parsed["probabilities"].get(language),
            "true_copy": true_copy,
            "copy_true_positive": sorted(named & set(true_copy)),
            "copy_false_positive": sorted(named - set(true_copy)),
            "copy_missed": sorted(set(true_copy) - named),
            "n_labels": len(parsed["head_labels"]), "code_runs": state["runs"].get("python", 0),
            "turns": state["turn"], "wall_s": round(state["wall_s"]),
            "mit_corpus_guess": mit.get("corpus_guess"), "mit_copy_named": mit.get("copy_named"),
            "mit_p_correct": mit_probability(mit.get("probabilities"), language), "mit_code_runs": mit.get("runs"),
        })  # fmt: skip
    (EXP / "processed").mkdir(exist_ok=True)
    (EXP / "processed" / "scores.json").write_text(json.dumps(rows, indent=1))
    print(
        "run | ours: corpus (p) | copy named | truth copy | FP | missed | runs || MIT: corpus (p) | copy named | runs"
    )
    for r in rows:
        print(f"{r['run']} | {r['corpus_guess']} ({r['p_correct']}) {'OK' if r['corpus_correct'] else 'WRONG'} | "
              f"{r['copy_named']} | {r['true_copy']} | {r['copy_false_positive']} | {r['copy_missed']} | {r['code_runs']} || "
              f"{r['mit_corpus_guess']} ({r['mit_p_correct']}) | {r['mit_copy_named']} | {r['mit_code_runs']}")  # fmt: skip
    n = len(rows)
    print(f"\ncorpus correct: {sum(r['corpus_correct'] for r in rows)}/{n}; "
          f"runs with copy heads exactly right: {sum(not r['copy_false_positive'] and not r['copy_missed'] for r in rows)}/{n}; "
          f"all 24 heads labelled: {sum(r['n_labels'] == 24 for r in rows)}/{n}")  # fmt: skip


if __name__ == "__main__":
    sys.exit(main())
