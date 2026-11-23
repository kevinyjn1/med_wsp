The orthogonal-only reversal corrected three answers and broke one, with a mean-margin improvement of 0.390 ± 0.199 on 128 questions. That is more encouraging than the letter-only intervention, but the accuracy gain is only one standard error and could be noise. I will keep its strength fixed and expand the paired comparison to the 512 development questions already used for other candidates. I will also inspect the changes by answer letter, because an orthogonal initial perturbation can still produce a letter bias through later layers.

```python
# Hypothesis: the orthogonal reversal improves downstream answer discrimination rather than merely imposing a constant letter preference.
import numpy as np, pandas as pd, torch

records18 = [json.loads(s) for s in (Path(WORK) / "score_18.jsonl").read_text().splitlines()]
print("Score record schema:", str(records18[0])[:1400])
print(
    "Analysis helpers:",
    [k for k, v in list(globals().items()) if callable(v) and getattr(v, "__module__", "") == "__main__"],
)
with (Path(WORK) / "notes.md").open("a") as f:
    f.write(
        "\nForward 18, L26:4131 orthogonal-component reflection, paired n=128: "
        "W->R=3, R->W=1; accuracy delta=+1.5625 pp, SE=1.56 pp; "
        "margin delta=+0.3896, SE=0.1989. Letter counts ABCD 26,34,35,33 -> 27,35,35,31. "
        "Provisional only. Expand at unchanged strength to the first 512 selection questions.\n"
    )
print(
    "EXPANDED_REQUEST:",
    json.dumps(
        {"type": "score", "model": "both", "questions": split_state["selection"][:512]}, separators=(",", ":")
    ),
)
```
