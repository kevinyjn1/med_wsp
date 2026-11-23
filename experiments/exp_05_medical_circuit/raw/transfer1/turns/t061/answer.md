The checkpoint identity and the confirmation panel’s separation from all earlier experiments are verified. Before scoring, I will record one final descriptive decomposition using development data only: how much a fixed, label-free letter-preference shift reproduces, versus changes that vary by question. Increasing total letter probability alone cannot change the evaluator’s four-letter argmax, so this distinction matters. I will then assemble the remaining confirmation identifiers without reading their answers or precomputed results.

```python
# Hypothesis: a fixed letter-preference shift explains only part of the candidate's development behavior.
def score_lp_matrix(rows, model):
    return np.array([r[model]["letter_logprobs"] for r in rows], dtype=float)

train_rows = read_score(13)
check_rows = read_score(18)
trb, tre = [score_lp_matrix(train_rows, k) for k in ["original","edited"]]
cb, ce = [score_lp_matrix(check_rows, k) for k in ["original","edited"]]
trdelta = tre - trb
trdelta -= trdelta.mean(axis=1, keepdims=True)
fixed_offset = trdelta.mean(axis=0)
cy = np.array(["ABCD".index(r["answer"]) for r in check_rows])
acc_of = lambda a: float((a.argmax(axis=1) == cy).mean())
development_decomposition = {
    "n":len(check_rows),
    "offset_estimated_on_separate_exploration":fixed_offset.tolist(),
    "base_accuracy":acc_of(cb),
    "candidate_accuracy":acc_of(ce),
    "base_plus_fixed_offset_accuracy":acc_of(cb+fixed_offset),
    "candidate_minus_fixed_offset_accuracy":acc_of(ce-fixed_offset),
    "base_letter_mass":float(np.exp(cb).sum(axis=1).mean()),
    "candidate_letter_mass":float(np.exp(ce).sum(axis=1).mean()),
    "caution":"Descriptive output decomposition, not an additive causal decomposition of accuracy."
}
print(json.dumps(development_decomposition))
(W / "development_decomposition.json").write_text(json.dumps(development_decomposition, indent=2))
with (W / "notes.md").open("a") as f:
    f.write("\nPre-confirmation, development-only fixed-offset diagnostic: "
            + json.dumps(development_decomposition) + "\n")
print("CONFIRMATION IDS SECOND HALF:")
print(json.dumps(confirmation_ids[512:], separators=(",", ":")))
```
