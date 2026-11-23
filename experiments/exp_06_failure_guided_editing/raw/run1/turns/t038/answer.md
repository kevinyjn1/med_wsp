On 512 questions, the layer-26 reversal gained 0.98 percentage points, but that is only about 1.3 standard errors and the mean margin worsened. Most of the net accuracy gain came from the initial screening questions, so this is still not a validated improvement. I will now isolate the neuron's letter-selective output component while leaving its other directions unchanged. Comparing this with the full edit and its complementary intervention will test whether the apparent benefit is primarily answer-letter correction or a broader circuit effect.

```python
# Hypothesis: the small L26 reversal gain is carried by its letter-selective component rather than its other output directions.
changes16 = paired_details(16)
rows16 = score_records[16]
fresh16 = rows16[128:]
d = np.array([int(r["edited"]["correct"]) - int(r["original"]["correct"]) for r in fresh16])
print(
    "Previously unscreened 384-question subset:",
    {
        "accuracy_delta_pp": round(100 * float(d.mean()), 3),
        "se_pp": round(100 * float(d.std(ddof=1) / np.sqrt(len(d))), 3),
        "wrong_to_right": int((d > 0).sum()),
        "right_to_wrong": int((d < 0).sum()),
    },
)
old16 = np.stack([lp_array(r["original"]) for r in rows16])
truth16 = np.array([letters.index(r["answer"]) for r in rows16])
print(
    "Descriptive constant-shift approximation accuracy:",
    round(float(((old16 + changes16.mean(0)).argmax(1) == truth16).mean()), 4),
)


def set_neuron_components(letter_scale=1.0, other_scale=1.0):
    clear_current_edits()
    w = candidate_original.clone()
    w[:, candidate_unit] = float(letter_scale) * v26_letter + float(other_scale) * v26_other
    save_edit(candidate_name, w)
    candidate_history.append(
        {
            "kind": "L26_neuron4131_directional",
            "letter_scale": float(letter_scale),
            "other_scale": float(other_scale),
        }
    )
    return float((w - candidate_original).norm() / candidate_original.norm())


change = set_neuron_components(letter_scale=-1.0, other_scale=1.0)
print("Installed letter-component reversal only; relative tensor change:", round(change, 6))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 16, full L26:4131 reversal, paired n=512: 56.8359% -> 57.8125%; "
        "delta=+0.9765625 pp, SE=0.76 pp; W->R=10, R->W=5; "
        "margin delta=-0.2949, SE=0.1429. Not clearly established. "
        "The 384 questions outside the initial screen contribute only one net correction.\n"
        "Directional causal control: reflect only column4131's projection onto the rank-3 "
        "fixed answer-letter subspace; keep its orthogonal component original. "
        "Then compare the complementary intervention.\n"
    )
```