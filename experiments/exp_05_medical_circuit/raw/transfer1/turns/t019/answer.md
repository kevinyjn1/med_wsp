The eight-layer attention transfer gives a noisy +3.12-point accuracy change, but its margin improvement is not distinguishable from its standard error. That is not enough evidence to retain roughly 302 million transferred weights. I will now isolate the four materially changed early normalization vectors, removing all attention matrices; this tests whether their coordinated channel reweighting accounts for the useful part of the broader intervention. A similar or cleaner gain from these 12,288 slots would be a much stronger localization result.

```python
# Hypothesis: the substantial early normalization changes, rather than broad attention-matrix replacement, provide the useful conditioning.
early_norm_names = [
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.1.input_layernorm.weight",
    "model.layers.2.input_layernorm.weight",
]
set_full_components(early_norm_names)
print("Active edits:", edited())
with notes_path.open("a") as f:
    f.write("\nForward passes used: 8/30. Early attention-only layers0-7 score_8: "
            "BARE +3.12 +/-2.21 pp, margin +0.0923 +/-0.1010; inconclusive and broad, not retained.\n"
            "Experiment E: four early normalization vectors (L0 input/post, L1 input, L2 input), "
            "alpha=1; all matrices base. 12,288 weight slots.\n")
```
