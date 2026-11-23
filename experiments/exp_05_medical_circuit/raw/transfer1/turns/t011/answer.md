Copying layer 11's MLP weights did not reproduce its activation-patch benefit: accuracy changed by only +0.78 ± 0.78 percentage points, while mean margin decreased. I will reject that edit and distinguish local computation from upstream conditioning more carefully. Next I will inspect the allowed layer-normalization changes and prepare a very small normalization-only intervention; these weights can affect many downstream activations without replacing their computations. I will also continue head-level localization rather than assume that a beneficial block patch identifies beneficial block weights.

```python
# Hypothesis: distributed changes to residual-channel normalization, rather than L11 MLP weights, condition answer extraction.
clear_edits()
norm_names = [n for n in NAMES if n.startswith("model.layers.") and "layernorm.weight" in n]
norm_stats = []
for name in norm_names:
    b, d = load_original(name), load_donor(name)
    delta = d - b
    scale = float(torch.dot(b, d) / torch.dot(b, b))
    orthogonal = d - scale * b
    norm_stats.append({
        "name": name,
        "relative_change": float(delta.norm() / b.norm()),
        "scale": scale,
        "nonscalar_fraction": float(orthogonal.norm() / delta.norm().clamp_min(1e-20)),
        "max_coordinate_change": float(delta.abs().max()),
    })
print(pd.DataFrame(norm_stats).sort_values("relative_change", ascending=False)
      .head(20).round(6).to_string(index=False))
print("All allowed normalization coordinates:", len(norm_names) * 3072)
print("Norm relative-change range:",
      min(r["relative_change"] for r in norm_stats),
      max(r["relative_change"] for r in norm_stats))
with notes_path.open("a") as f:
    f.write("\nForward passes used: 4/30. REJECT L11 MLP full donor transfer: "
            "256 BARE, accuracy +0.78 +/- 0.78 pp, flips 3 better/1 worse, "
            "margin -0.0107 +/- 0.0061. No active edits. "
            "Beneficial activation patch did not establish local weight sufficiency.\n")
```
