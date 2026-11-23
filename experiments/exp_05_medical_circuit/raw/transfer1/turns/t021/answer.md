The four-normalization edit is not stronger than the single-vector edit on margins, so normalization is only a minor contributor. The causal screen also showed a run of beneficial early MLP-output patches, unlike many harmful late MLP patches. I will test a different, constrained source hypothesis: donor changes in those early MLP output projections may alter what the existing base features write into the residual stream. I will preserve all gate/up projections, attention weights, and norms, transferring only the down projections in the positively screened early MLPs.

```python
# Hypothesis: early MLP output-projection changes write the donor's format-sensitive residual signal while preserving base feature detectors.
source_mlp_layers = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12]
source_down_names = [f"model.layers.{l}.mlp.down_proj.weight" for l in source_mlp_layers]
set_full_components(source_down_names)
print("Transferred down-projection slots:", len(source_mlp_layers) * 3072 * 8192)
print("Layers:", source_mlp_layers)
with notes_path.open("a") as f:
    f.write("\nForward passes used: 9/30. Four-early-norm edit score_9: "
            "BARE +1.17 +/-0.87 pp, margin +0.0280 +/-0.0158; not established.\n"
            "Experiment F: down-projection-only transfer for positive early MLP patches "
            "at layers " + json.dumps(source_mlp_layers) +
            ". All feature detectors, attention, norms, and late MLPs remain base. "
            "Broad source experiment before sparse-unit decomposition.\n")
```
