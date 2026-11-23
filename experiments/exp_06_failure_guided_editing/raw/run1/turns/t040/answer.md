Reversing only the letter-selective component produced no accuracy gain and significantly worsened the mean margin on the screening batch. That argues against retaining the direct letter-direction change by itself. I will now test the complementary edit: keep that component intact and reverse only the orthogonal output component. A gain under this control would localize a downstream circuit effect more specifically than the original anti-A attribution.

```python
# Hypothesis: any useful effect of L26 neuron 4131 reversal arises from downstream processing of its non-letter output component.
changes17 = paired_details(17)
change = set_neuron_components(letter_scale=1.0, other_scale=-1.0)
print("Installed orthogonal-component reversal only; relative tensor change:", round(change, 6))
print(
    "Numerical direct-letter projection of edited direction:",
    np.round((letter_readout @ (-2 * v26_other)).numpy(), 7).tolist(),
)
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 17, L26:4131 letter-component-only reflection, n=128: "
        "accuracy delta=0 pp, SE=1.11 pp; W->R=1,R->W=1; "
        "margin delta=-0.4282, SE=0.1570. No support for isolated direct-letter edit.\n"
        "Next complementary control: letter component unchanged, orthogonal component reflected. "
        "The immediate perturbation has zero projection onto the chosen answer-letter contrasts; "
        "later layers can nevertheless transform it into letter or content effects.\n"
    )
```