The centered final-layer ablation increased the mean margin significantly, but it corrected no answers and introduced one error. That is evidence of changing confidence, not evidence of improving medical answer selection, so I will not retain it. I will return to the layer-26 neuron, whose full ablation previously corrected three answers and harmed one, and test a stronger intervention before a larger comparison. Reversing its outgoing column doubles the change from the original and will reveal whether those corrections form a useful dose-response or were small-sample noise.

```python
# Hypothesis: the earlier L26 neuron-4131 effect may change answer selection, unlike the confidence-only L31 intervention.
changes14 = paired_details(14)
relative_change = set_single_neuron_scale(-1.0)
print("Installed L26 neuron4131 outgoing scale=-1.0; relative tensor change:", round(relative_change, 6))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 14, centered L31:5937 ablation, n=128: accuracy delta=-0.78125 pp, "
        "SE=0.78 pp; W->R=0, R->W=1; margin delta=+0.2683, SE=0.0775. "
        "Rejected as a retention candidate: margin improvement without answer corrections "
        "is consistent with confidence changes, not improved task accuracy.\n"
        "Return to L26:4131, previously 3 corrections and 1 regression at full ablation. "
        "Test scale=-1 (column reversal), twice the original-to-zero perturbation. "
        "All other edits removed.\n"
    )
```