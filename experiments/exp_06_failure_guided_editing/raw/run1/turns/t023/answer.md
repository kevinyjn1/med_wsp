Increasing head 9 produced only one net additional correct answer, while the mean margin decreased by 0.089 with a 0.049 standard error. That does not support the proposed benefit of strengthening it. I will now test the opposite, equally sized intervention: reducing this head's output to half its original value. If margins improve under attenuation and worsen under amplification, that would establish a more informative dose-response and suggest that this option-comparison signal is sometimes overweighted rather than too weak.

```python
# Hypothesis: layer-25 head 9 may overweight a misleading option-comparison signal rather than provide insufficient evidence.
changes9 = paired_details(9)
change = set_head_scale(25, 9, 0.5)
print("Installed L25 head 9 output scale=0.5; relative tensor change:", round(change, 6))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 9, L25 head9 scale=1.5, paired n=128: accuracy delta=+0.78125 pp, "
        "SE=1.36 pp; W->R=2, R->W=1; margin delta=-0.0886, SE=0.0485. "
        "No established benefit; test symmetric attenuation to scale=0.5.\n"
    )
```