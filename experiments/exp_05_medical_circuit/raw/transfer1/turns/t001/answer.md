I will first separate prompt-format effects from question-dependent improvements, before examining weight differences. I will reserve 1,024 practice questions for final confirmation and exclude their results from all exploratory analysis. On the remaining questions, I will compare errors, answer-letter preferences, and total probability assigned to answer letters under both prompts. A higher probability of producing a letter does not itself improve the evaluated accuracy, so I will also check whether changes in relative letter preferences explain the donor's advantage. I will then use activation patching and direct weight interventions to test causality, reserving forward passes for controls and independent confirmation.

```python
# Hypothesis: the donor's BARE advantage may combine prompt-format effects with question-dependent improvements.
import json, random, hashlib, gc
from pathlib import Path
import numpy as np
import torch

work = Path(WORK)
practice = Path(PRACTICE_DIR)
questions = [json.loads(line) for line in open(practice / "questions.jsonl")]
qid = lambda q: str(q["id"])
q_by_id = {qid(q): q for q in questions}
all_ids = [qid(q) for q in questions]

split_rng = random.Random(815731)
shuffled_ids = all_ids.copy()
split_rng.shuffle(shuffled_ids)
confirmation_ids = shuffled_ids[:1024]
confirmation_set = set(confirmation_ids)
dev_ids = shuffled_ids[1024:]
dev_set = set(dev_ids)
(work / "confirmation_ids.json").write_text(json.dumps(confirmation_ids))
(work / "development_ids.json").write_text(json.dumps(dev_ids))
print("Questions:", len(all_ids), "development:", len(dev_ids),
      "sealed confirmation:", len(confirmation_ids))
print("Prompt definitions:\n", (practice / "eval_prompt_format.txt").read_text())

def read_development_results(filename):
    out = {}
    with open(practice / filename) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            rid = str(row.get("id", row.get("question_id", all_ids[i])))
            if rid in dev_set:
                out[rid] = row
    assert set(out) == dev_set, (filename, len(out))
    return out

results = {
    "base_bare": read_development_results("original_results.jsonl"),
    "donor_bare": read_development_results("donor_results.jsonl"),
    "base_chat": read_development_results("original_results_chat.jsonl"),
    "donor_chat": read_development_results("donor_results_chat.jsonl"),
}
letters = list("ABCD")
def lp_array(rows, ids):
    def convert(row):
        v = row["letter_logprobs"]
        return [v[k] for k in letters] if isinstance(v, dict) else v
    return np.asarray([convert(rows[i]) for i in ids], dtype=np.float64)

answer_index = {k: i for i, k in enumerate(letters)}
y_dev = np.array([answer_index[q_by_id[i]["answer"]] for i in dev_ids])
L = {k: lp_array(v, dev_ids) for k, v in results.items()}
summary = {}
for name, rows in results.items():
    pred = L[name].argmax(1)
    mass = np.array([rows[i]["letter_mass"] for i in dev_ids], dtype=float)
    summary[name] = {
        "accuracy_pct": round(100 * np.mean(pred == y_dev), 3),
        "predicted_ABCD_pct": np.round(100 * np.bincount(pred, minlength=4) / len(pred), 2).tolist(),
        "letter_mass_mean": round(float(mass.mean()), 6),
        "letter_mass_quantiles": np.round(np.quantile(mass, [.1, .5, .9]), 6).tolist(),
    }
print("Development summaries:", json.dumps(summary, indent=2))
print("Correct-answer ABCD percentages:",
      np.round(100 * np.bincount(y_dev, minlength=4) / len(y_dev), 2).tolist())

diagnostics = {}
for prompt in ["bare", "chat"]:
    b, d = L["base_" + prompt], L["donor_" + prompt]
    bc, dc = b.argmax(1) == y_dev, d.argmax(1) == y_dev
    delta = d - b
    relative_delta = delta - delta.mean(1, keepdims=True)
    offset = relative_delta.mean(0)
    info = {
        "donor_only_correct": int((dc & ~bc).sum()),
        "base_only_correct": int((bc & ~dc).sum()),
        "both_correct": int((bc & dc).sum()),
        "neither_correct": int((~bc & ~dc).sum()),
        "mean_relative_logprob_shift_ABCD": np.round(offset, 4).tolist(),
        "base_plus_label_free_mean_shift_acc_pct":
            round(100 * np.mean((b + offset).argmax(1) == y_dev), 3),
        "donor_minus_mean_shift_acc_pct":
            round(100 * np.mean((d - offset).argmax(1) == y_dev), 3),
        "constant_shift_fraction_of_squared_relative_delta":
            round(float((offset ** 2).sum() / np.mean((relative_delta ** 2).sum(1))), 4),
    }
    diagnostics[prompt] = info
print("Error overlap and descriptive format-offset diagnostic:",
      json.dumps(diagnostics, indent=2))

notes_path = work / "notes.md"
notes_path.write_text(
    "# Medical weight transfer\n"
    "Forward passes used: 0/30.\n"
    "Split: seed 815731; 1024 confirmation IDs sealed in confirmation_ids.json; "
    "3248 development IDs in development_ids.json. Confirmation results not inspected.\n"
    "Excluded edits: embeddings, lm_head, final norm, whole-tensor scalar changes.\n"
    "Plan: diagnose outputs; patch components; test sparse donor weight transfer; "
    "dose-response and size-matched random control; validate BARE and CHAT independently.\n"
    "Current edit: none.\n\nDevelopment summaries:\n" + json.dumps(summary, indent=2)
    + "\nDevelopment descriptive diagnostics:\n" + json.dumps(diagnostics, indent=2) + "\n"
)
print("Existing edits:", edited())
```
