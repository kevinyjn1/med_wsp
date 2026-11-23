**No edit retained. `WORK/edited` is verified empty.**

I diagnosed answer-letter imbalance and tested targeted neuron and attention-head interventions with dose-response controls. The strongest candidate modified only layer 26 MLP neuron 4131's outgoing direction, preserving its projection onto the dominant letter-token contrasts.

Its development gain did not reliably generalize:
- **Confirmation, 1,024 questions:** +0.68 ± 0.54 percentage points.
- **19 corrections, 12 regressions; p = 0.281.**
- **All 1,280 independent questions:** +0.39 ± 0.46 points.

Letter preference remained a confound: subtracting the development-estimated constant letter shift eliminated the confirmation accuracy gain, although this is descriptive rather than causal attribution. No reliable improvement in medical content use was established.

I restored the original weights rather than submit an unsupported change. Findings and final statistics are saved in `notes.md` and `final_results.json`.