The normalization changes are highly concentrated: layer 0's attention input norm changes by 13.4%, layer 1's by 5.5%, and most later norms are effectively unchanged. These are channel-specific changes, not whole-tensor scaling, and the early attention patches already showed positive causal effects. That makes the first attention input norm a sharply defined candidate requiring only 3,072 weight values. I will test it alone before adding any other component.

```python
# Hypothesis: layer 0's channel-specific attention normalization change is sufficient for a meaningful BARE gain.
norm0_name = "model.layers.0.input_layernorm.weight"
set_full_components([norm0_name])
b0, d0 = load_original(norm0_name), load_donor(norm0_name)
norm0_delta = d0 - b0
top_dims = torch.argsort(norm0_delta.abs(), descending=True)[:24]
print("L0 input norm largest changes [dimension, base, donor, delta]:")
print([[int(j), round(float(b0[j]), 7), round(float(d0[j]), 7),
        round(float(norm0_delta[j]), 7)] for j in top_dims])
print("Nonzero changed coordinates:", int((norm0_delta != 0).sum()))
print("Top 8/32/128 coordinates' delta energy fractions:",
      [round(float(torch.sort(norm0_delta.square(), descending=True).values[:k].sum()
                   / norm0_delta.square().sum()), 5) for k in [8, 32, 128]])
with notes_path.open("a") as f:
    f.write("\nNormalization inspection: L0 input relative delta 0.133918, "
            "L1 input 0.054556, L0 post-attention 0.018321, L2 input 0.006757. "
            "L0 input delta is 99.99% nonscalar by norm. "
            "Experiment B: only donor model.layers.0.input_layernorm.weight.\n")
```
