The stronger layer-26 edit corrected six answers and harmed two, a 3.13-point screening gain, but its 2.20-point standard error is still too large to establish improvement. The mean margin decreased, so this is not a simple confidence increase and needs careful checking. I will preserve this candidate provisionally and test it on the full 512-question development batch. If it survives, I will separate the neuron's direct letter-selective output from its other output directions to test where any benefit comes from.

```python
# Hypothesis: reversing L26 neuron 4131 changes answer selection beneficially, but the apparent gain needs a larger paired test.
import shutil

changes15 = paired_details(15)
shutil.copytree(work / "edited", work / "candidate_l26_reverse", dirs_exist_ok=True)
rows15 = score_records[15]
old15 = np.stack([lp_array(r["original"]) for r in rows15])
truth15 = np.array([letters.index(r["answer"]) for r in rows15])
offset_only_pred = (old15 + changes15.mean(0)[None, :]).argmax(1)
actual_pred15 = np.array([letters.index(r["edited"]["pred"]) for r in rows15])
print("Descriptive constant-offset approximation, not a model forward pass:")
print(
    {
        "offset_only_accuracy": round(float((offset_only_pred == truth15).mean()), 4),
        "actual_edited_accuracy": round(float((actual_pred15 == truth15).mean()), 4),
        "prediction_agreement": round(float((offset_only_pred == actual_pred15).mean()), 4),
    }
)
v26 = candidate_original[:, candidate_unit]
v26_letter = letter_basis @ (letter_basis.T @ v26)
v26_other = v26 - v26_letter
print("L26:4131 column norm fraction in letter subspace:", round(float(v26_letter.norm() / v26.norm()), 4))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 15, L26:4131 scale=-1, paired n=128: 57.8125% -> 60.9375%; "
        "delta=+3.125 pp, SE=2.20 pp; W->R=6, R->W=2; "
        "margin delta=-0.3794, SE=0.2654. Provisional only, not independently established.\n"
        "Saved candidate copy to candidate_l26_reverse. "
        "Dose sequence scales 0.5,0,-1 gave screening accuracy deltas 0,+1.5625,+3.125 pp, "
        "but mean margins did not improve.\n"
        "Next: paired n=512 at unchanged scale=-1. If promising, test letter-subspace "
        "and orthogonal-component interventions separately for causal attribution.\n"
    )
```