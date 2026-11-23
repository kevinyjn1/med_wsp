The first CHAT check shows nearly unchanged accuracy, -0.39 ± 1.79 points, but a negative margin shift, so the guardrail is not yet established. I will save this candidate and test a half-strength version, using interpolation toward the donor rather than scaling whole tensors. I will also separate its common letter-preference shift from question-dependent changes using the recorded logits. A useful dose-response should show a consistent BARE effect that weakens as the donor contribution is reduced.

```python
# Hypothesis: the sparse-prefix effect is donor-dose-dependent and partly reflects prompt-format calibration.
import shutil
from scipy.special import logsumexp

snapshot = work / "candidate_sparse_full"
if snapshot.exists():
    shutil.rmtree(snapshot)
shutil.copytree(work / "edited", snapshot)

def score_analysis(k):
    rows = [json.loads(line) for line in open(work / f"score_{k}.jsonl")]
    ids = [str(r["id"]) for r in rows]
    yy = np.array([answer_index[q_by_id[i]["answer"]] for i in ids])
    b = lp_array({str(r["id"]): r["original"] for r in rows}, ids)
    e = lp_array({str(r["id"]): r["edited"] for r in rows}, ids)
    delta = e - b
    rel = delta - delta.mean(1, keepdims=True)
    offset = rel.mean(0)
    out = {
        "pass": k,
        "mean_letter_mass_base_edit": np.exp(logsumexp(np.stack([b, e]), axis=2)).mean(1).tolist(),
        "relative_letter_offset_ABCD": offset.tolist(),
        "base_accuracy": float((b.argmax(1) == yy).mean()),
        "edited_accuracy": float((e.argmax(1) == yy).mean()),
        "base_plus_common_offset_accuracy": float(((b + offset).argmax(1) == yy).mean()),
        "edited_minus_common_offset_accuracy": float(((e - offset).argmax(1) == yy).mean()),
        "common_fraction_of_relative_delta_energy":
            float(np.sum(offset**2) / np.mean(np.sum(rel**2, axis=1))),
    }
    return out

format_diagnostics = [score_analysis(k) for k in [5, 8, 11, 13, 14]]
print(json.dumps(format_diagnostics, indent=2))
set_sparse_prefix(sparse_units, alpha=0.5)
with notes_path.open("a") as f:
    f.write("\nForward passes used: 14/30. Sparse prefix score_13 BARE +4.30 +/-2.23 pp "
            "(22 rescues,11 regressions), margin +0.1400 +/-0.1060. "
            "score_14 CHAT -0.39 +/-1.79 pp (10 rescues,11 regressions), "
            "margin -0.4137 +/-0.2812. No independent success established.\n"
            "Full-dose candidate snapshot: candidate_sparse_full. "
            "Experiment J: same sparse-prefix edit at alpha=0.5.\n"
            "Output diagnostics:\n" + json.dumps(format_diagnostics, indent=2) + "\n")
```
