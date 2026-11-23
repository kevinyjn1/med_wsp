Replacing the donor's residual stream by approximately the middle of the network recovers the discovery panel's margin advantage, but that intervention includes everything computed upstream. Many individual late MLP patches actually hurt, so copying late layers wholesale would be poorly justified. I will inspect the complete component results, including early layers omitted from the displayed ranking, and separate effects on donor-rescued versus donor-harmed questions. The stratified discovery panel is deliberately enriched for disagreements, so its accuracy changes will not be treated as population estimates.

```python
# Hypothesis: useful changes accumulate upstream, while several late donor components are unnecessary or harmful.
patch3 = [json.loads(line) for line in open(work / "patch_3.jsonl")]
print("Patch file row count:", len(patch3))
print("First row structure:",
      {k: (list(v)[:8] if isinstance(v, dict) else
           ("list length " + str(len(v)) if isinstance(v, list) else v))
       for k, v in patch3[0].items()})
print("First row truncated:", json.dumps(patch3[0])[:2200])
print("Meta keys:", list(json.loads((fwd_base / "meta.json").read_text()).keys()))
with notes_path.open("a") as f:
    f.write("\nForward passes used: 3/30. patch_3: exhaustive 96 patches on discovery panel. "
            "Base accuracy 42.19%, donor 56.25%; donor margin gap 0.6016. "
            "L14-16 residual patches recover about 104-105% of gap; late individual MLP "
            "patches often harmful. Residual patches include upstream computation and "
            "do not identify local weight causality.\n"
            "Top-token diagnosis: base says The first on 59/64 medical questions; donor "
            "often prefers letters, with increased letter probabilities on simple nonmedical probes too.\n")
```
