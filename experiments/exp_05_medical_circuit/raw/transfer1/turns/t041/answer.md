The random-unit control does not establish that the selected MLP units are special. Early attention changes remain a plausible source of the benefit, but the independent development result is still inconclusive. I will recover the experiment definitions and diagnostic details before allocating the remaining ten passes. The confirmation questions will remain sealed.

```python
# Hypothesis: early attention changes explain more of the gain than the particular selected MLP units.
from pathlib import Path
import json, numpy as np, torch, gc
W = Path(WORK)
notes_text = (W / "notes.md").read_text()
print("EARLY NOTES\n", notes_text[:6500])
print("\nKERNEL VARIABLES")
print([(k, type(v).__name__) for k, v in list(globals().items())
       if not k.startswith("_") and not callable(v)
       and k not in {"notes_text"}][:160])
print("\nSCORE SCHEMA")
row20 = json.loads((W / "score_20.jsonl").read_text().splitlines()[0])
print(str(row20)[:2400])
print("\nID FILE SIZES")
for filename in ["explore_ids.json", "development_check_ids.json",
                 "validation_ids.json", "confirmation_ids.json",
                 "discovery_ids.json", "tuning_ids.json"]:
    ids_tmp = json.loads((W / filename).read_text())
    print(filename, type(ids_tmp).__name__, len(ids_tmp))
print("\nSNAPSHOT FILES", [p.name for p in (W / "candidate_sparse_full").iterdir()][:8])
print("\nFORWARD METADATA")
meta1 = json.loads((W / "fwd_1" / "meta.json").read_text())
print(str(meta1)[:2200])
with (W / "notes.md").open("a") as f:
    f.write("\nForward passes used: 20/30. Random early MLP-unit control score_20: "
            "BARE +2.73 +/-2.24 pp (20/13 flips), margin +0.0979 +/-0.1023. "
            "Selected units' superiority is not established. Current edit is this control, "
            "not the selected candidate. Confirmation IDs remain sealed.\n")
```