The baseline under-predicts A substantially: A is correct on 804 development questions but predicted on only 575, and its recall is 49.0%, versus 62.8% to 66.0% for the other letters. This is a concrete hypothesis worth testing, although unequal recall does not prove an internal letter bias. The errors are also often extremely confident, so merely changing confidence would not address the task. I will inspect internal weight directions associated with answer letters and prepare a random discovery batch for activation measurements. A small set of unusually strong, consistently active letter-selective components would suggest a targeted intervention; diffuse or question-dependent effects would argue against that explanation.

```python
# Hypothesis: a concentrated internal letter-selective component contributes to the deficit in A predictions.
torch.set_num_threads(8)
probe_ids = discovery_ids[:128]
screen_ids = selection_ids[:128]

letter_token_ids = {}
for a in letters:
    variants = [tok.encode(s, add_special_tokens=False).ids for s in (a, " " + a)]
    letter_token_ids[a] = sorted({v[0] for v in variants if len(v) == 1})
    print("Letter tokenization:", a, variants)
assert all(letter_token_ids[a] for a in letters)

head = load_original("lm_head.weight")
final_norm = load_original("model.norm.weight")
letter_readout = torch.stack([head[letter_token_ids[a]].mean(0) for a in letters])
letter_readout *= final_norm[None, :]
letter_readout -= letter_readout.mean(0, keepdim=True)
letter_basis = torch.linalg.svd(letter_readout.T, full_matrices=False).U[:, :3].contiguous()
del head
weight_scan = []
letter_projections = {}
column_norms = {}
for layer in range(32):
    name = f"model.layers.{layer}.mlp.down_proj.weight"
    w = load_original(name)
    norms = w.norm(dim=0)
    projections = letter_readout @ w
    letter_projections[layer] = projections.numpy().copy()
    column_norms[layer] = norms.numpy().copy()
    concentration = (letter_basis.T @ w).norm(dim=0) / norms.clamp_min(1e-12)
    top = torch.topk(concentration, 3)
    weight_scan.append(
        {
            "layer": layer,
            "column_norm_median": float(norms.median()),
            "column_norm_max": float(norms.max()),
            "letter_fraction_max": float(top.values[0]),
            "top_letter_units": top.indices.tolist(),
            "A_contrast_min": float(projections[0].min()),
            "A_contrast_max": float(projections[0].max()),
        }
    )
    del w

print("\nWeight-only screen, not causal evidence:")
print(pd.DataFrame(weight_scan).round(4).to_string(index=False))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nBaseline diagnosis: A true=804, predicted=575, recall=49.0%; "
        "other recalls=62.8%-66.0%. Investigate letter-selective components, "
        "without assuming this explains all errors. Errors commonly have large negative margins.\n"
        "First activation batch: first 128 discovery IDs, chosen without outcomes. "
        "First causal screen: first 128 selection IDs.\n"
    )
print("\nDiscovery forward request:")
print(
    json.dumps(
        {
            "type": "teacher_forced",
            "model": "original",
            "questions": probe_ids,
            "return": {
                "positions": "last",
                "token_logprobs": True,
                "topk": 8,
                "hidden_states": list(range(33)),
                "mlp_act": list(range(32)),
            },
        }
    )
)
```