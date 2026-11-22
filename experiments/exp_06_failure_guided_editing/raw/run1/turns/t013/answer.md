Removing the stable final-layer projections shifted two predictions toward A but again produced no net accuracy gain on 128 questions. This supports the direction of the intervention, but its usefulness remains unproven. I will examine the actual per-letter probability changes and then test a stronger, predeclared dose of three times the projection removal. If the stronger dose still lacks a consistent benefit, I will avoid treating the baseline's unequal letter counts as sufficient justification for a repair.

```python
# Hypothesis: the final-layer projection edit has the predicted letter-specific effect, but strength 1 may be too weak for a decisive test.
def lp_array(record):
    v = record["letter_logprobs"]
    return np.array([v[a] for a in letters] if isinstance(v, dict) else v, dtype=float)


def paired_details(k):
    rows = read_score(k)
    old = np.stack([lp_array(r["original"]) for r in rows])
    new = np.stack([lp_array(r["edited"]) for r in rows])
    old_centered = old - old.mean(1, keepdims=True)
    new_centered = new - new.mean(1, keepdims=True)
    changes = new_centered - old_centered
    truth = np.array([letters.index(r["answer"]) for r in rows])
    details = []
    for j, a in enumerate(letters):
        idx = truth == j
        ds = np.array([r["edited"]["margin"] - r["original"]["margin"] for r in rows])[idx]
        details.append(
            {
                "true": a,
                "n": int(idx.sum()),
                "original_correct": sum(r["original"]["correct"] for r, keep in zip(rows, idx) if keep),
                "edited_correct": sum(r["edited"]["correct"] for r, keep in zip(rows, idx) if keep),
                "margin_delta": float(ds.mean()),
            }
        )
    print("Pass", k, "centered log-probability change mean ABCD:", np.round(changes.mean(0), 3).tolist())
    print("Pass", k, "centered log-probability change SD ABCD:", np.round(changes.std(0), 3).tolist())
    print(pd.DataFrame(details).round(3).to_string(index=False))
    return changes


changes2 = paired_details(2)
changes3 = paired_details(3)
changes5 = paired_details(5)
change = set_bias_strength(3.0)
print("Installed projection-removal strength=3.0; relative tensor change:", round(change, 6))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 5, L31 stable anti-A projection removal strength=1, paired n=128: "
        "delta accuracy=0 pp, SE=1.11 pp; W->R=1, R->W=1; "
        "margin delta=0.0347, SE=0.0811; A predictions=26 -> 28. No established benefit.\n"
        "Next predeclared strength=3.0, with no other edits.\n"
        "Measured centered log-probability shift at strength 1: "
        + str(np.round(changes5.mean(0), 4).tolist())
        + "\n"
    )
```