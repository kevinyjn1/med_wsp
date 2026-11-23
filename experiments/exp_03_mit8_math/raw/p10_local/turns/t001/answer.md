I will first inspect the available weights and practice data, then measure how the unedited model fails under the full evaluation protocol. I will distinguish incorrect mathematics from repetition, missing answers, and token-limit cutoffs before choosing a mechanism to investigate. The tentative budget is one diagnostic generation pass, several localization and intervention passes, and two final paired comparisons on untouched practice problems. Any candidate edit will need both a predicted behavioral effect and an accuracy improvement larger than its uncertainty.

```python
# Hypothesis: full-length practice generations will reveal a specific failure mechanism worth localizing.
import os, json, random, math, collections
from pathlib import Path
import numpy as np
import torch

work = Path(WORK)
practice = Path(PRACTICE_DIR)
model_dir = Path(MODEL_DIR)

datasets = {}
for source, filename in [
    ("gsm8k", "gsm8k_train.jsonl"),
    ("math", "math_train.jsonl"),
    ("aime", "aime.jsonl"),
]:
    with open(practice / filename) as f:
        datasets[source] = [json.loads(line) for line in f if line.strip()]

by_id = {str(row["id"]): row for rows in datasets.values() for row in rows}
rng = random.Random(20261122)


def math_level(row):
    value = str(row.get("level", ""))
    digits = [c for c in value if c.isdigit()]
    return int(digits[-1]) if digits else None


gsm_pool = list(datasets["gsm8k"])
rng.shuffle(gsm_pool)
math_pools = {level: [r for r in datasets["math"] if math_level(r) == level] for level in range(1, 6)}
for pool in math_pools.values():
    rng.shuffle(pool)

diagnostic_rows = gsm_pool[:32]
for level, count in [(1, 4), (2, 6), (3, 8), (4, 8), (5, 6)]:
    diagnostic_rows.extend(math_pools[level][:count])

used_ids = {str(r["id"]) for r in diagnostic_rows}
reserved_rows = []
remaining_gsm = [r for r in gsm_pool if str(r["id"]) not in used_ids]
remaining_math = [r for pool in math_pools.values() for r in pool if str(r["id"]) not in used_ids]
rng.shuffle(remaining_math)
for batch in range(2):
    rows = remaining_gsm[16 * batch : 16 * (batch + 1)]
    rows += remaining_math[16 * batch : 16 * (batch + 1)]
    rng.shuffle(rows)
    reserved_rows.append(rows)

diagnostic_ids = [r["id"] for r in diagnostic_rows]
validation_ids = [[r["id"] for r in rows] for rows in reserved_rows]
assert len(diagnostic_ids) == 64
assert len({str(x) for group in [diagnostic_ids] + validation_ids for x in group}) == 128

plan = {
    "seed": 20261122,
    "diagnostic_ids": diagnostic_ids,
    "reserved_validation_ids": validation_ids,
    "planned_budget": {
        "diagnostic_generation": 1,
        "localization_and_candidate_tests": 7,
        "untouched_paired_validation": 2,
    },
}
(work / "experiment_plan.json").write_text(json.dumps(plan, indent=2))
(work / "notes.md").write_text(
    "# Weight-edit experiment\n\n"
    "Initial state: no edits made by this session.\n"
    "Plan: diagnose full-length failures, localize a mechanism, test causal "
    "interventions at multiple strengths, then validate on two untouched paired batches.\n"
    "No training, no model execution outside the forward tool, and no edits to "
    "embeddings, output head, final norm, or whole-tensor rescalings.\n\n"
    "Planned pass 1: original model; n=1; max_tokens=16384; exact diagnostic IDs:\n"
    + json.dumps(diagnostic_ids)
    + "\n"
    "Reserved validation IDs, not to be used for candidate selection:\n" + json.dumps(validation_ids) + "\n"
)

print("Existing edits:", edited())
print("Practice counts:", {k: len(v) for k, v in datasets.items()})
print("MATH level counts:", {k: len(v) for k, v in math_pools.items()})
print("Diagnostic source counts:", {"gsm8k": 32, "math": 32})
print("Language-model tensor count:", len(NAMES))
print("Representative layer-0 tensor names:", [n for n in NAMES if ".layers.0." in n][:18])
config_path = model_dir / "config.json"
if config_path.exists():
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    keys = [
        "model_type",
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "linear_key_head_dim",
        "linear_value_head_dim",
        "tie_word_embeddings",
    ]
    print("Relevant configuration:", {k: text_config[k] for k in keys if k in text_config})
print("Evaluation prompt format:", (practice / "eval_prompt_format.txt").read_text()[:1800])
print(
    "PASS_1_REQUEST:",
    json.dumps(
        {"type": "generate", "model": "original", "problems": diagnostic_ids, "n": 1, "max_tokens": 16384}
    ),
)
```
