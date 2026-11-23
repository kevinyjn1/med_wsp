At half dose, the sparse prefix's BARE effect nearly disappears, while full dose gives the larger exploratory gain. This is compatible with a nonlinear interaction, but the uncertainties are too large to call it a strong dose-response yet. Before accepting an eight-layer intervention, I will test the complete first layer alone, where the causal screen and normalization test both found positive effects. This can reveal whether the broader transfer adds unnecessary variance and CHAT risk around a simpler early source.

```python
# Hypothesis: the first transformer layer alone provides a smaller, more reliable source intervention than an eight-layer mixture.
layer0_names = [n for n in NAMES if n.startswith("model.layers.0.")]
set_full_components(layer0_names)
print("First-layer tensors:", edited())
with notes_path.open("a") as f:
    f.write("\nForward passes used: 15/30. Sparse prefix half-dose score_15: "
            "BARE +0.39 +/-1.41 pp, margin +0.0162 +/-0.0567; "
            "full dose +4.30 pp/+0.1400 margin. Dose pattern nonlinear and uncertain.\n"
            "Experiment K: complete donor layer0 only, testing a simpler early source "
            "and avoiding unnecessary downstream interventions. Sparse full snapshot retained.\n")
```
