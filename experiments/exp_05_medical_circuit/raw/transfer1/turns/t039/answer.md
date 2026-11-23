The separate CHAT panel has exactly unchanged accuracy, with 13 improvements and 13 regressions, although margins decline. This supports proceeding cautiously, not claiming proven non-degradation. I will now test whether the selected MLP units add anything specific beyond the early attention transfer by replacing them with the same number of random units in the same layers. After this control, I will keep the final candidate fixed while collecting mechanism evidence and the sealed confirmation results.

```python
# Hypothesis: activation-selected early MLP units add more useful computation than equally numerous random early units.
unit_control_rng = np.random.default_rng(622081)
random_early_units = {}
for l in range(8):
    available = np.setdiff1d(np.arange(8192), np.array(sparse_units[l]))
    random_early_units[l] = unit_control_rng.choice(available, size=256, replace=False).tolist()
set_structured_layers(random_early_units)
(work / "random_early_units.json").write_text(json.dumps(random_early_units))

def conditional_extraction_diagnostic(bare_pass, chat_pass):
    br = {str(r["id"]): r for r in
          map(json.loads, open(work / f"score_{bare_pass}.jsonl"))}
    cr = {str(r["id"]): r for r in
          map(json.loads, open(work / f"score_{chat_pass}.jsonl"))}
    out = {}
    for chat_correct in [True, False]:
        rr = [r for i, r in br.items() if bool(cr[i]["original"]["correct"]) == chat_correct]
        changes = np.array([int(r["edited"]["correct"]) - int(r["original"]["correct"])
                            for r in rr])
        rescues = [r for r in rr if r["edited"]["correct"] and not r["original"]["correct"]]
        losses = [r for r in rr if r["original"]["correct"] and not r["edited"]["correct"]]
        out[str(chat_correct)] = {
            "n": len(rr), "rescues": len(rescues), "losses": len(losses),
            "bare_accuracy_change_pp": float(100 * changes.mean()),
        }
    return out
extraction_diagnostic = conditional_extraction_diagnostic(18, 19)
print("Validation BARE effects conditional on unedited CHAT correctness:",
      json.dumps(extraction_diagnostic, indent=2))
print("Validation output diagnostic:", json.dumps(score_analysis(18), indent=2))
with notes_path.open("a") as f:
    f.write("\nForward passes used: 19/30. Independent development n512: "
            "score_18 BARE +1.95 +/-1.68 pp (42/32 flips), margin +0.1316 +/-0.0684; "
            "score_19 CHAT 0.00 +/-1.00 pp (13/13 flips), margin -0.3750 +/-0.1681. "
            "Not yet clear accuracy success.\n"
            "Experiment M: same early attention/norms, replace selected256 MLP units/layer "
            "with256 random nonselected units/layer, fixed seed622081.\n"
            "Conditional extraction diagnostic: " + json.dumps(extraction_diagnostic) + "\n")
```
