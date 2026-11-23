The larger request failed before returning a result, so it provides no evidence for or against the edit and did not consume a pass. I will verify the saved weights after the restart and split the planned 512-question comparison into two 256-question batches. Their paired outcomes can be pooled without changing the candidate between batches. I will also record the revised layer-32 convention so later activation analyses do not mix incompatible arrays.

```python
# Hypothesis: the failed request was operational; the saved intervention and prespecified comparison can be recovered unchanged.
import json, os, gc, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
WORK = Path(WORK)
PRACTICE_DIR = Path(PRACTICE_DIR)
splits = json.loads((WORK/"splits.json").read_text())
def get_score(k):
    return [json.loads(s) for s in (WORK/f"score_{k}.jsonl").read_text().splitlines()]
def summarize_scores(rows):
    a = np.array([int(r["original"]["correct"]) for r in rows])
    b = np.array([int(r["edited"]["correct"]) for r in rows])
    d = b-a
    m = np.array([r["edited"]["margin"]-r["original"]["margin"] for r in rows])
    return dict(n=len(rows), original=float(a.mean()), edited=float(b.mean()),
                delta_pp=float(100*d.mean()), se_pp=float(100*d.std(ddof=1)/len(d)**0.5),
                wr=int((d==1).sum()), rw=int((d==-1).sum()),
                margin=float(m.mean()), margin_se=float(m.std(ddof=1)/len(m)**0.5))
name26 = "model.layers.26.mlp.down_proj.weight"
assert edited() == [name26], edited()
w26_original = load_original(name26)
w26_current = load(name26)
changed_columns = torch.nonzero((w26_current!=w26_original).any(dim=0)).flatten().tolist()
assert changed_columns == [4131], changed_columns
v26 = w26_original[:,4131].clone()
orth26 = (v26-w26_current[:,4131])/4.0
parallel26 = v26-orth26
print("Saved edit audit:", {"changed_columns": changed_columns,
    "relative_tensor_change": float(torch.linalg.vector_norm(w26_current-w26_original)/torch.linalg.vector_norm(w26_original)),
    "parallel_orthogonal_cosine": float(torch.dot(parallel26,orth26)/(parallel26.norm()*orth26.norm()))})
def install_orthogonal_scale(scale):
    for name in list(edited()):
        drop_edit(name)
    w = w26_original.clone()
    w[:,4131] = parallel26 + float(scale)*orth26
    save_edit(name26,w)
    return float(torch.linalg.vector_norm(w-w26_original)/torch.linalg.vector_norm(w26_original))
del w26_current
gc.collect()
with (WORK/"notes.md").open("a") as f:
    f.write("\nForward 20, orthogonal scale=-3, n=128: "+json.dumps(summarize_scores(get_score(20)))+
        "\nKernel restarted after pass20. Forward21 was first interrupted, then failed with WorkerError; neither counted. "
        "Saved edit verified to change only L26 down-projection column4131. Plan: same 512 questions in two fixed 256 batches. "
        "IMPORTANT: teacher-forced layer32 from passes<=20 is POST final RMSNorm; future layer32 is PRE final RMSNorm. "
        "Layers0:31 unchanged. Do not compare old/new layer32 arrays directly.\n")
print("REQUEST_21:",json.dumps({"type":"score","model":"both","questions":splits["selection"][:256]},separators=(",",":")))
```
