The complete prefix gained the same 1.95 percentage points as the sparse edit on the development panel, despite changing substantially more weights. Its stronger margin signal is interesting, but it does not justify the extra replacement, so I will reject that broader candidate. I will make one additional, fixed dose test of the sparse edit: 1.5 times the donor-minus-base change, alongside the existing zero, half, and full doses. This tests whether stronger prompt adaptation improves accuracy or instead disrupts the base; CHAT remains a required check.

```python
# Hypothesis: the sparse prefix's nonlinear dose response reflects a threshold in answer-format handling.
def install_sparse_transfer(alpha, units_by_layer):
    for name in list(edited()):
        drop_edit(name)
    for layer_key, unit_list in sorted(units_by_layer.items(), key=lambda kv: int(kv[0])):
        layer = int(layer_key)
        prefix = f"model.layers.{layer}."
        for suffix in ["input_layernorm.weight", "post_attention_layernorm.weight",
                       "self_attn.qkv_proj.weight", "self_attn.o_proj.weight"]:
            name = prefix + suffix
            b = load_original(name)
            d = load_donor(name)
            if not torch.equal(b, d):
                e = b + alpha * (d - b)
                save_edit(name, e)
                del e
            del b, d
        u = torch.tensor(unit_list, dtype=torch.long)
        for suffix in ["mlp.gate_up_proj.weight", "mlp.down_proj.weight"]:
            name = prefix + suffix
            b = load_original(name)
            d = load_donor(name)
            if "gate_up" in suffix:
                rr = torch.cat([u, u + 8192])
                b[rr] += alpha * (d[rr] - b[rr])
            else:
                b[:, u] += alpha * (d[:, u] - b[:, u])
            save_edit(name, b)
            del b, d
    gc.collect()

selected_unit_map = json.loads((W / "sparse_units.json").read_text())
print("Unit-map structure:", type(selected_unit_map).__name__,
      [(k, len(v)) for k,v in selected_unit_map.items()])
install_sparse_transfer(1.5, selected_unit_map)
print("Installed fixed 1.5-dose sparse prefix:", len(edited()), "tensors")
print("Full prefix versus sparse on development:", compare_scores(21, 18))
with (W / "notes.md").open("a") as f:
    f.write("\nForward passes used: 21/30. Full prefix score21 on separate development "
            "n512: BARE +1.95 +/-1.77 pp (46/36 flips), margin +0.2157 +/-0.0803. "
            "Same accuracy as sparse prefix with more changed weights: reject broader edit. "
            "Experiment O: one fixed extrapolated sparse-prefix dose alpha=1.5. "
            "Existing doses alpha=0,0.5,1 provide reference; no gradient or fitting. "
            "Use same development n512, then CHAT if viable. Confirmation remains sealed.\n")
```
