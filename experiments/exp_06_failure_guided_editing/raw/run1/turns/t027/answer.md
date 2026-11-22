Halving layer-30 head 24 changed no answers and had a negligible mean-margin effect, despite a substantial weight perturbation. I will remove that candidate; strong weight alignment and suggestive attention patterns were not enough to make it behaviorally important. The first activation sample was small and unusually difficult, so I will expand the diagnostic measurements to the 512-question selection batch already used for candidate testing. This will let me check whether late answer reversals and harmful neuron contributions recur across a broader sample before proposing another edit.

```python
# Hypothesis: a broader activation sample can separate reproducible late-layer error contributions from small-sample artifacts.
changes11 = paired_details(11)
clear_current_edits()
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 11, L30 head24 scale=0.5, paired n=128: no answer changes; "
        "margin delta=0.0017, SE=0.0083. Candidate removed for negligible behavioral effect.\n"
        "Next diagnostic pass: original model on the already-exposed first 512 selection questions, "
        "hidden layers 20:32 and MLP layers 20:31. These are development data, not final confirmation.\n"
        "All current edits cleared. No candidate has earned retention yet.\n"
    )
print("Current edits:", edited())
print("Diagnostic sample size:", len(selection_large_ids))
```