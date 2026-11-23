The complete eight-layer prefix improves the exploration margin by 0.261 ± 0.121, larger than the attention-only or down-projection-only interventions, although its accuracy evidence remains preliminary. This points toward early feature-formation changes or their interactions, not the late reader heads. I will remove the prefix's attention and down-projection matrices and retain only its gate/up matrices plus the previously positive layer 0 input norm. This tests whether most of the useful effect survives while leaving substantially more of the base untouched.

```python
# Hypothesis: early MLP gate/up changes, with minimal early conditioning, account for the useful part of the full-prefix transfer.
early_gateup_names = [f"model.layers.{l}.mlp.gate_up_proj.weight" for l in range(8)]
gate_candidate_names = early_gateup_names + [norm0_name]
set_full_components(gate_candidate_names)
print("Gate/up plus norm weight slots:", 8 * 16384 * 3072 + 3072)
with notes_path.open("a") as f:
    f.write("\nForward passes used: 11/30. Full layers0-7 score_11: "
            "BARE +4.30 +/-2.61 pp; margin +0.2609 +/-0.1206; "
            "letter counts A/B/C/D 75/51/64/66 vs base98/39/74/45. "
            "Broad exploratory gain, not yet independent confirmation.\n"
            "Experiment H: layers0-7 gate_up matrices plus L0 input norm only, "
            "all attention matrices, down projections, other norms and later layers base.\n")
```
