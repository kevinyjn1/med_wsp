The early MLP down-projection transfer has essentially zero margin effect, so that writeout-only hypothesis is not supported. Taken together, the negative local transfers suggest either interactions among early components or a source outside the components tested so far. Before spending more passes, I will check how much donor advantage is actually present on the exploration panel and summarize all interventions consistently. I will then test a bounded early-layer prefix as a sufficiency experiment, not as an automatic final choice.

```python
# Hypothesis: early component interactions, rather than isolated writeout changes, account for the residual-patch benefit.
def panel_baseline(ids):
    yy = np.array([answer_index[q_by_id[i]["answer"]] for i in ids])
    out = {}
    for name, rows in results.items():
        lp = lp_array(rows, ids)
        out[name] = round(float(100 * (lp.argmax(1) == yy).mean()), 3)
    return out
print("Exploration baseline accuracies:", panel_baseline(explore_ids))
print("Paired score row schema:", json.dumps(json.loads(
    (work / "score_10.jsonl").read_text().splitlines()[0]))[:2000])
prefix_names = [n for n in NAMES if n.startswith("model.layers.")
                and int(n.split(".")[2]) < 8]
set_full_components(prefix_names)
print("Prefix active tensors:", len(edited()))
print("Layers 0-7 weight slots:", 8 * (4 * 3072**2 + 3 * 3072 * 8192 + 2 * 3072))
with notes_path.open("a") as f:
    f.write("\nForward passes used: 10/30. Early down-projection-only score_10: "
            "BARE +1.17 +/-1.30 pp, margin -0.0002 +/-0.0285; rejected.\n"
            "Exploration unedited accuracies: " + json.dumps(panel_baseline(explore_ids)) +
            "\nExperiment G: complete donor layers0-7 (8/32 blocks) as bounded sufficiency "
            "test of interactions. This is broader than desired final edit and will require "
            "decomposition, dose/control, CHAT and independent confirmation.\n")
```
