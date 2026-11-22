```python
# Hypothesis: baseline failures contain a systematic, localizable pattern rather than only isolated knowledge gaps.
import json
import pathlib
import numpy as np
import pandas as pd
import torch

practice = pathlib.Path(PRACTICE_DIR)
work = pathlib.Path(WORK)
questions = [json.loads(s) for s in (practice / "questions.jsonl").read_text().splitlines() if s.strip()]
baseline = [json.loads(s) for s in (practice / "original_results.jsonl").read_text().splitlines() if s.strip()]
q_by_id = {str(q["id"]): q for q in questions}
b_by_id = {str(r["id"]): r for r in baseline}

rng = np.random.default_rng(20261122)
all_ids = np.array(sorted(q_by_id), dtype=object)
permuted = rng.permutation(all_ids).tolist()
confirmation_ids = permuted[:1024]
discovery_ids = permuted[1024:2224]
selection_ids = permuted[2224:3248]
reserve_ids = permuted[3248:]
development_ids = discovery_ids + selection_ids + reserve_ids
splits = {
    "discovery": discovery_ids,
    "selection": selection_ids,
    "development_reserve": reserve_ids,
    "final_confirmation": confirmation_ids,
}
(work / "splits.json").write_text(json.dumps(splits, indent=2))

letters = list("ABCD")
def baseline_row(qid):
    q, r = q_by_id[qid], b_by_id[qid]
    lp = r["letter_logprobs"]
    lp = np.array([lp[a] for a in letters] if isinstance(lp, dict) else lp, dtype=float)
    p = np.exp(lp - lp.max())
    p /= p.sum()
    return {
        "id": qid, "answer": q["answer"], "pred": r["pred"],
        "correct": bool(r["correct"]), "margin": float(r["margin"]),
        "confidence": float(p.max()),
        **{f"p_{a}": float(p[i]) for i, a in enumerate(letters)},
    }

# Confirmation outcomes are deliberately not accessed.
dev = pd.DataFrame([baseline_row(qid) for qid in development_ids])
print("Split sizes:", {k: len(v) for k, v in splits.items()})
print("Prompt format:\n", (practice / "eval_prompt_format.txt").read_text()[:3500])
print("Development accuracy:", round(100 * dev.correct.mean(), 2), "%")
letter_summary = pd.DataFrame(index=letters)
letter_summary["true_n"] = dev.answer.value_counts().reindex(letters, fill_value=0)
letter_summary["pred_n"] = dev.pred.value_counts().reindex(letters, fill_value=0)
letter_summary["mean_probability"] = [dev[f"p_{a}"].mean() for a in letters]
letter_summary["recall"] = [dev.loc[dev.answer == a, "correct"].mean() for a in letters]
print("\nLetter diagnostics:\n", letter_summary.round(4).to_string())
print("\nConfusion matrix, true rows and predicted columns:\n",
      pd.crosstab(dev.answer, dev.pred).reindex(index=letters, columns=letters, fill_value=0).to_string())
print("\nMargin quantiles by correctness:\n",
      dev.groupby("correct").margin.quantile([0, .1, .25, .5, .75, .9, 1]).round(3).to_string())
dev["confidence_bin"] = pd.cut(dev.confidence, [0, .4, .6, .8, .9, .95, 1.000001])
print("\nConfidence diagnostics:\n",
      dev.groupby("confidence_bin", observed=True).agg(n=("id", "size"), accuracy=("correct", "mean")).round(4).to_string())

patterns = {
    "diagnosis": r"diagnos|most likely cause",
    "management": r"next step|management|treatment|therapy|most appropriate",
    "mechanism": r"mechanism|pathophysiolog|responsible for|most likely explains",
    "drug_related": r"medication|drug|pharmac|adverse effect",
    "statistics": r"sensitivity|specificity|odds ratio|relative risk|confidence interval|study design",
    "negative_question": r"\bexcept\b|\bleast likely\b|\bnot indicated\b|\bcontraindicat",
}
text_by_id = {qid: str(q_by_id[qid]["question"]) for qid in development_ids}
topic_rows = []
for name, pattern in patterns.items():
    import re
    mask = dev.id.map(lambda qid: bool(re.search(pattern, text_by_id[qid], re.I)))
    sub = dev[mask]
    topic_rows.append({
        "pattern": name, "n": len(sub),
        "accuracy": sub.correct.mean(),
        "mean_margin": sub.margin.mean(),
        "confident_errors": int(((~sub.correct) & (sub.confidence >= .9)).sum()),
    })
print("\nOverlapping question-type diagnostics:\n", pd.DataFrame(topic_rows).round(4).to_string(index=False))
print("\nEight confident errors, excerpts only:")
for row in dev[~dev.correct].sort_values("margin").head(8).itertuples():
    excerpt = text_by_id[row.id][:240].replace("\n", " ").encode("ascii", "replace").decode()
    print(json.dumps({"id": row.id, "true": row.answer, "pred": row.pred,
                      "margin": round(row.margin, 3), "excerpt": excerpt}, ensure_ascii=True))

notes = (
    "# Weight-editing experiment\n\n"
    "No edits yet; no forward passes used.\n"
    "Seed: 20261122. Split membership is recorded in splits.json.\n"
    "Final confirmation: 1024 randomly reserved questions. Do not inspect their outcomes "
    "or activations until the final candidate is locked.\n"
    "Discovery: 1200; selection: 1024; additional development reserve: 1024.\n"
    "Baseline diagnosis uses only the 3248 development questions.\n"
    f"Development accuracy: {100 * dev.correct.mean():.3f}%.\n\n"
    "Planned budget: approximately 6 activation/localization passes, 18 intervention and "
    "dose-response passes, 4 robustness passes, and 2 final confirmation passes. "
    "Adapt allocation while preserving untouched confirmation.\n"
    "Constraints: no training, no network simulation, no embedding/output/final-norm edits, "
    "and no whole-tensor scalar edits. Retain only independently supported improvements.\n\n"
    "Letter diagnostics:\n" + letter_summary.round(4).to_string() + "\n\n"
    "Question-type diagnostics:\n" + pd.DataFrame(topic_rows).round(4).to_string(index=False) + "\n"
)
(work / "notes.md").write_text(notes)
print("\nInitial edit inventory:", edited())
```