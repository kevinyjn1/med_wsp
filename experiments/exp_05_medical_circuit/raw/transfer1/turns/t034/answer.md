Layer 0 alone does not improve on the tiny normalization edit, so it does not explain the broader candidate's accuracy gain. I will retain the eight-layer sparse candidate only provisionally and now run the required size-matched random-location control. The control will transfer the same kinds and numbers of attention weights and MLP units, but from eight randomly selected later layers with randomly selected units. If it performs similarly, the claimed localization to early computation would be weakened.

```python
# Hypothesis: early locations matter; a size-matched transfer at random later locations should not reproduce the candidate's gain.
control_rng = np.random.default_rng(474191)
control_layers = sorted(control_rng.choice(np.arange(8, 32), size=8, replace=False).tolist())
control_units = {l: control_rng.choice(8192, size=256, replace=False).tolist()
                 for l in control_layers}

def set_structured_layers(units_by_layer, alpha=1.0, attn_parts=("q", "k", "v", "o"),
                          include_norms=True):
    layers = sorted(units_by_layer)
    clear_edits()
    for l in layers:
        for kind in ["qkv", "o"]:
            if kind == "o" and "o" not in attn_parts:
                continue
            if kind == "qkv" and not any(p in attn_parts for p in ["q", "k", "v"]):
                continue
            name = f"model.layers.{l}.self_attn.{kind}_proj.weight"
            b, d = load_original(name), load_donor(name)
            out = b.clone()
            if kind == "qkv":
                for offset, part in [(0, "q"), (3072, "k"), (6144, "v")]:
                    if part in attn_parts:
                        sl = slice(offset, offset + 3072)
                        out[sl] = b[sl] + alpha * (d[sl] - b[sl])
            else:
                out = b + alpha * (d - b)
            save_edit(name, out)
            del b, d, out
        if include_norms:
            for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight"]:
                name = f"model.layers.{l}.{suffix}"
                b, d = load_original(name), load_donor(name)
                if not torch.equal(b, d):
                    save_edit(name, b + alpha * (d - b))
                del b, d
        ix = torch.tensor(units_by_layer[l], dtype=torch.long)
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

set_structured_layers(control_units)
print("Random control layers:", control_layers)
(work / "random_location_control.json").write_text(json.dumps(control_units))
with notes_path.open("a") as f:
    f.write("\nForward passes used: 16/30. Full L0 score_16: BARE +0.78 +/-0.96 pp, "
            "margin +0.0502 +/-0.0214; little beyond norm0, not selected.\n"
            "Experiment L: size-matched random-location control, 8 later layers "
            + json.dumps(control_layers) +
            ", complete attention plus256 random MLP units each and available donor norm changes. "
            "Same matrix slot count; norm differences can be zero in later layers.\n")
```
