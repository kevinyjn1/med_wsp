The full neuron reversal has a small, unconfirmed accuracy gain, while reversing only its answer-letter component mainly shifts probability toward A without improving accuracy. The complementary edit is now installed: it reverses the component orthogonal to those letter contrasts, leaving the direct letter component unchanged. I will first recover the saved question splits and notebook state so the next comparison uses the same screening questions and does not expose the reserved confirmation set. A benefit from this complementary edit would implicate downstream processing, although it would not by itself prove better medical reasoning.

```python
# Hypothesis: the orthogonal-component intervention can be tested without exposing reserved confirmation data.
import os, json
from pathlib import Path

print("WORK:", str(WORK))
print("Edited tensors:", edited())
split_state = json.loads((Path(WORK) / "splits.json").read_text())
print(
    "Split structure:",
    {
        k: {"type": type(v).__name__, "size": len(v) if hasattr(v, "__len__") else None}
        for k, v in split_state.items()
    },
)
print("Notebook beginning:")
print((Path(WORK) / "notes.md").read_text()[:13500])
print("Kernel variable inventory:")
print(
    [
        (
            k,
            type(v).__name__,
            tuple(v.shape) if hasattr(v, "shape") else len(v) if isinstance(v, (list, dict, tuple)) else None,
        )
        for k, v in list(globals().items())
        if not k.startswith("_") and not callable(v) and not isinstance(v, type(os))
    ][:160]
)
```
