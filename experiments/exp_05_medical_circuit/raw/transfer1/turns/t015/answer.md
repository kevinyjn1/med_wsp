Head patching substantially narrows the activity pathway: layer 25 head 9, layer 18 heads 5 and 7, and layer 7 head 1 have the largest positive effects. Layer 18 head 9 and layer 7 head 16 provide smaller but consistent signals. I will transfer only these six heads' Q, K, V rows and output-projection columns, without the normalization edit, to test whether their own weights are sufficient. If this fails, these heads are likely readers of changes established upstream rather than the main storage sites of the advantage.

```python
# Hypothesis: six causally selected attention heads contain sufficient local weight changes for improved BARE answers.
selected_heads = {7: [1, 16], 18: [5, 7, 9], 25: [9]}
def set_head_edit(heads_by_layer, alpha=1.0, parts=("q", "k", "v", "o"), extra_norms=()):
    clear_edits()
    for layer, heads in heads_by_layer.items():
        for kind in ["qkv", "o"]:
            if kind == "qkv" and not any(p in parts for p in ["q", "k", "v"]):
                continue
            if kind == "o" and "o" not in parts:
                continue
            name = f"model.layers.{layer}.self_attn.{kind}_proj.weight"
            b, d = load_original(name), load_donor(name)
            out = b.clone()
            for h in heads:
                if kind == "qkv":
                    for offset, p in [(0, "q"), (3072, "k"), (6144, "v")]:
                        if p in parts:
                            s = slice(offset + 96 * h, offset + 96 * (h + 1))
                            out[s] = b[s] + alpha * (d[s] - b[s])
                else:
                    s = slice(96 * h, 96 * (h + 1))
                    out[:, s] = b[:, s] + alpha * (d[:, s] - b[:, s])
            save_edit(name, out)
            del b, d, out
            gc.collect()
    for name in extra_norms:
        b, d = load_original(name), load_donor(name)
        save_edit(name, b + alpha * (d - b))

set_head_edit(selected_heads)
head_slot_count = sum(map(len, selected_heads.values())) * 4 * 96 * 3072
print("Selected heads:", selected_heads, "editable head weight slots:", head_slot_count)
patch6 = [json.loads(line) for line in open(work / "patch_6.jsonl")]
for layer, heads in selected_heads.items():
    for h in heads:
        key = f"L{layer}.head{h}"
        v = np.array([r["patched"][key] - r["base"] for r in patch6])
        weighted = sum(population_weights[g] * v[ix].mean()
                       for g, ix in panel_group_indices.items())
        print(key, "mean delta", round(v.mean(), 4), "weighted delta", round(weighted, 4))
with notes_path.open("a") as f:
    f.write("\nForward passes used: 6/30. Norm0-only score_5: BARE +0.78 +/-0.78 pp, "
            "margin +0.0438 +/-0.0135 (256 questions), insufficient accuracy evidence.\n"
            "Head patch_6: L25H9 +0.3477 +/-0.1373 margin, "
            "L18H5 +0.2168 +/-0.0528, L18H7 +0.1929 +/-0.0525, "
            "L7H1 +0.1860 +/-0.0429, L18H9 +0.1030 +/-0.0292, "
            "L7H16 +0.0933 +/-0.0219.\n"
            "Experiment C: six selected heads Q/K/V rows and O columns, alpha=1, "
            "no norms or MLP changes. Weight slots " + str(head_slot_count) + ".\n")
```
