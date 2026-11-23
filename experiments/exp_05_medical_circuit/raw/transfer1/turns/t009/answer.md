The strongest individual candidates are attention in layers 18 and 25 and the MLP in layer 11; layer 11's MLP has the clearest margin signal relative to its uncertainty. However, donor activation patches can carry improvements inherited from earlier layers, so they do not prove that the corresponding local weight changes are sufficient. I will first test that distinction by transferring only layer 11's two MLP matrices, leaving its normalization and every other component unchanged. This is a localization experiment, not yet a proposed final edit.

```python
# Hypothesis: donor changes in layer 11's MLP weights themselves reproduce its beneficial activation-patch effect.
def clear_edits():
    for name in list(edited()):
        drop_edit(name)

def set_full_components(names, alpha=1.0):
    clear_edits()
    for name in names:
        original = load_original(name)
        donor = load_donor(name)
        value = original + alpha * (donor - original)
        save_edit(name, value)
        del original, donor, value
        gc.collect()

mlp11_names = [
    "model.layers.11.mlp.gate_up_proj.weight",
    "model.layers.11.mlp.down_proj.weight",
]
set_full_components(mlp11_names)
explore_ids = tuning_ids[:256]
(work / "explore_ids.json").write_text(json.dumps(explore_ids))
print("Exploration IDs:", json.dumps(explore_ids))
for name in mlp11_names:
    b, d = load_original(name), load_donor(name)
    delta = d - b
    print(name, "shape", list(b.shape), "relative delta norm",
          round(float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(b)), 6))
    del b, d, delta
gc.collect()
print("Edited tensors:", edited())
with notes_path.open("a") as f:
    f.write("\nExperiment A: full donor L11 gate_up and down matrices, alpha=1, "
            "no norm transfer. Selected by patch_3. Exploration uses first 256 tuning IDs, "
            "disjoint from discovery and sealed confirmation. Awaiting direct-weight score.\n")
```
