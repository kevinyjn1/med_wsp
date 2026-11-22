Halving neuron 4131 produced no accuracy gain: one answer improved and one worsened, with a mean margin change of only 0.035 plus or minus 0.063 standard error. A predictions increased by just one, much less than its direct residual contribution suggested. This weakens the idea that the neuron's apparent anti-A contribution reaches the output unchanged; downstream compensation is a plausible explanation. I will test complete removal next to establish whether there is a meaningful dose-response before spending more evaluation power on this candidate.

```python
# Hypothesis: complete removal of L26 neuron 4131 reveals whether downstream compensation masks a partial intervention.
score_records = {}


def read_score(k):
    rows = [json.loads(s) for s in (work / f"score_{k}.jsonl").read_text().splitlines() if s.strip()]
    score_records[k] = rows
    return rows


rows2 = read_score(2)
print("Score record schema:", json.dumps(rows2[0], ensure_ascii=True)[:2200])
relative_change = set_single_neuron_scale(0.0)
print("Current intervention: L26 neuron 4131 outgoing column zeroed.")
print("Relative tensor change:", round(relative_change, 6))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 2, L26:4131 scale=0.5, paired selection n=128: "
        "accuracy 57.8125% -> 57.8125%; delta=0.000 pp, SE=1.11 pp; "
        "wrong-to-right=1, right-to-wrong=1; margin delta=0.0347, SE=0.0628. "
        "A predictions 26 -> 27. No established benefit.\n"
        "Next dose: L26:4131 scale=0.0, all other weights original.\n"
    )
```