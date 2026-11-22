Neither increasing nor decreasing layer-25 head 9 improved the mean margin, so I will not pursue that head as a repair on the current evidence. Another localized candidate is layer 30, head 24: in the inspected cases it attends mainly to a prompt-start position and demographic or vital-sign tokens, despite having strongly letter-selective outputs. Such attention may be useful, so I will test rather than assume that it is distracting. I will halve this head's output and look for question-dependent improvements beyond any average letter shift.

```python
# Hypothesis: layer-30 head 24 can overweight demographic or prompt-summary cues during late answer selection.
changes10 = paired_details(10)
change = set_head_scale(30, 24, 0.5)
print("Installed L30 head 24 output scale=0.5; relative tensor change:", round(change, 6))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 10, L25 head9 scale=0.5, paired n=128: accuracy delta=0 pp, "
        "SE=1.11 pp; W->R=1, R->W=1; margin delta=-0.0515, SE=0.0473. "
        "Neither amplification nor attenuation supported; removed this edit.\n"
        "Next candidate: L30 head24 output scale=0.5, columns 2304:2400. "
        "This head has letter-subspace fraction=0.1604 and attends mainly to prompt start "
        "and age/vital-sign tokens in inspected discovery failures. "
        "Potential over-weighting of demographic cues is a hypothesis, not an established failure.\n"
    )
```