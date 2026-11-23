The recovered diagnostics point mainly to prompt handling: the donor assigns much more probability to answer letters, including on nonmedical probes, while it is not better under CHAT. They also show why activation patches alone were misleading: several helpful patched components failed when only their local weights were transferred. Before choosing another intervention, I will compare the existing edits question by question and inspect how early the residual patches become effective. This can distinguish evidence for a distributed early-layer change from evidence for a smaller sufficient edit.

```python
# Hypothesis: the useful change is distributed through the early prefix, rather than localized to the selected MLP units.
from scipy.stats import binomtest

def read_score(k):
    return [json.loads(s) for s in (W / f"score_{k}.jsonl").read_text().splitlines()]

def compare_scores(k1, k0):
    r1 = {r["id"]: r for r in read_score(k1)}
    r0 = {r["id"]: r for r in read_score(k0)}
    common_ids = sorted(set(r1) & set(r0))
    a = np.array([r1[i]["edited"]["correct"] for i in common_ids], dtype=float)
    b = np.array([r0[i]["edited"]["correct"] for i in common_ids], dtype=float)
    m = np.array([r1[i]["edited"]["margin"] - r0[i]["edited"]["margin"]
                  for i in common_ids])
    z = a - b
    return {"passes": [k1, k0], "n": len(z),
            "accuracy_difference_pp": round(100*z.mean(), 3),
            "paired_se_pp": round(100*z.std(ddof=1)/np.sqrt(len(z)), 3),
            "margin_difference": round(m.mean(), 4),
            "margin_se": round(m.std(ddof=1)/np.sqrt(len(m)), 4),
            "better_worse": [int((z>0).sum()), int((z<0).sum())]}

print("DIRECT EDIT CONTRASTS")
for pair in [(13,20), (13,8), (11,13), (16,5), (9,5), (13,17)]:
    print(compare_scores(*pair))
print("\nID PANEL SIZES")
for filename in ["explore_ids.json", "development_check_ids.json",
                 "validation_ids.json", "confirmation_ids.json",
                 "discovery_ids.json", "tuning_ids.json"]:
    x = json.loads((W / filename).read_text())
    print(filename, len(x))
print("\nPATCH TABLE COLUMNS", list(patch_df.columns))
print(patch_df[patch_df["patch"].str.contains("resid")].to_string(index=False))
print("\nFORWARD FILES", sorted(p.name for p in (W / "fwd_1").iterdir())[:14])
print("\nSNAPSHOT FILES", sorted(p.name for p in (W / "candidate_sparse_full").iterdir())[:5])
print("\nRESULT KEYS", list(results.keys()))
print("\nHELPER NAMES",
      [k for k,v in list(globals().items()) if callable(v) and
       getattr(v, "__module__", "") == "__main__" and not k.startswith("_")])
```
