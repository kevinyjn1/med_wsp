Gate/up changes alone add little beyond the small normalization effect, so the full-prefix result appears to depend on interactions rather than one isolated matrix family. I will test a structured approximation to that prefix: retain the early attention changes, but transfer only 256 MLP units per layer instead of all 8,192. Units will be chosen without answer labels, using their observed base-to-donor activation differences weighted by their writeout-column norms. This is a hypothesis about which features carry the changed representation, and a size-matched random-unit control will be required if it works.

```python
# Hypothesis: a small set of changed early MLP features mediates the interaction benefit on top of early attention changes.
sparse_units = {}
unit_selection_info = []
for l in range(8):
    a = read_npy(fwd_base, f"mlp_act_{l}")[:64]
    z = read_npy(fwd_donor, f"mlp_act_{l}")[:64]
    down = load_original(f"model.layers.{l}.mlp.down_proj.weight")
    column_norm2 = down.square().sum(0).numpy()
    score = np.mean((z - a) ** 2, axis=0) * column_norm2
    units = np.argsort(score)[-256:]
    sparse_units[l] = units.tolist()
    unit_selection_info.append([l, round(float(score[units].sum() / score.sum()), 4)])
    del a, z, down
gc.collect()

def set_sparse_prefix(units_by_layer, alpha=1.0, attention=True, norm=True):
    clear_edits()
    if attention:
        for l in range(8):
            for suffix in ["self_attn.qkv_proj.weight", "self_attn.o_proj.weight"]:
                name = f"model.layers.{l}.{suffix}"
                b, d = load_original(name), load_donor(name)
                save_edit(name, b + alpha * (d - b))
                del b, d
    if norm:
        for l in range(8):
            for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
                name = f"model.layers.{l}.{suffix}"
                b, d = load_original(name), load_donor(name)
                if not torch.equal(b, d):
                    save_edit(name, b + alpha * (d - b))
                del b, d
    for l, units in units_by_layer.items():
        ix = torch.tensor(units, dtype=torch.long)
        for suffix in ["gate_up_proj", "down_proj"]:
            name = f"model.layers.{l}.mlp.{suffix}.weight"
            b, d = load_original(name), load_donor(name)
            out = b.clone()
            if suffix == "gate_up_proj":
                rows = torch.cat([ix, ix + 8192])
                out[rows] = b[rows] + alpha * (d[rows] - b[rows])
            else:
                out[:, ix] = b[:, ix] + alpha * (d[:, ix] - b[:, ix])
            save_edit(name, out)
            del b, d, out
            gc.collect()

set_sparse_prefix(sparse_units)
(work / "sparse_units.json").write_text(json.dumps(sparse_units))
print("Selected 256 units/layer; fraction of weighted activation-difference energy:",
      unit_selection_info)
print("Approximate touched slots:", 8 * 4 * 3072**2 + 8 * 256 * 3 * 3072 + 16 * 3072)
with notes_path.open("a") as f:
    f.write("\nForward passes used: 12/30. Early gate_up + norm0 score_12: "
            "BARE +0.78 +/-0.96 pp, margin +0.0571 +/-0.0263; weak.\n"
            "Experiment I: attention/norms layers0-7 plus 256 MLP units per layer, "
            "selected label-free from weighted captured activation differences. "
            "Units saved in sparse_units.json. Selection is descriptive, requires direct tests and control.\n")
```
