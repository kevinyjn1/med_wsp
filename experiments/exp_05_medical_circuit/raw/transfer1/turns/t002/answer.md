The donor's BARE gain is accompanied by a large format change: mean probability on the four answer letters rises from 4.0% to 41.1%, and the base's strong preference for A becomes less pronounced. A question-independent shift in letter preferences improves the base by only 1.85 percentage points, compared with the donor's 6.37-point advantage, so that simple explanation is incomplete. The remaining difference is question-dependent, but that does not yet establish additional medical knowledge; better extraction of an answer from the same knowledge could also explain it. I will create a discovery panel containing both donor improvements and donor regressions, then inspect actual next-token alternatives and internal activations. Separate development questions will be used to evaluate weight edits.

```python
# Hypothesis: format-sensitive answer extraction explains part, but not necessarily all, of the donor's advantage.
bc = L["base_bare"].argmax(1) == y_dev
dc = L["donor_bare"].argmax(1) == y_dev
groups = {
    "donor_only": np.flatnonzero(dc & ~bc),
    "base_only": np.flatnonzero(bc & ~dc),
    "both": np.flatnonzero(dc & bc),
    "neither": np.flatnonzero(~dc & ~bc),
}
panel_rng = np.random.default_rng(62104)
panel_counts = {"donor_only": 24, "base_only": 16, "both": 12, "neither": 12}
discovery_ids = []
discovery_group = {}
for group, indices in groups.items():
    chosen = panel_rng.choice(indices, size=panel_counts[group], replace=False)
    for j in chosen:
        rid = dev_ids[int(j)]
        discovery_ids.append(rid)
        discovery_group[rid] = group
discovery_set = set(discovery_ids)
remaining_dev = [i for i in dev_ids if i not in discovery_set]
random.Random(93102).shuffle(remaining_dev)
tuning_ids = remaining_dev[:512]
development_check_ids = remaining_dev[512:1536]
(work / "discovery_ids.json").write_text(json.dumps(discovery_ids))
(work / "tuning_ids.json").write_text(json.dumps(tuning_ids))
(work / "development_check_ids.json").write_text(json.dumps(development_check_ids))

for name in ["base", "donor"]:
    bare_pred = L[name + "_bare"].argmax(1)
    chat_pred = L[name + "_chat"].argmax(1)
    print(name, "BARE/CHAT answer agreement:",
          round(100 * np.mean(bare_pred == chat_pred), 2))
for group, indices in groups.items():
    print(group, "n =", len(indices),
          "base/donor BARE median letter mass =",
          [round(float(np.median([results[m + "_bare"][dev_ids[int(j)]]["letter_mass"]
                                  for j in indices])), 6) for m in ["base", "donor"]])

probe_texts = [
    "Question:\nA. option\nB. option\nC. option\nD. option\nAnswer:",
    "Question: Which option contains the word apple?\nA. pear\nB. apple\nC. plum\nD. grape\nAnswer:",
    "Question: Which option contains the word apple?\nA. pear\nB. plum\nC. grape\nD. apple\nAnswer:",
    "Question: What is 2 + 2?\nA. 3\nB. 5\nC. 4\nD. 6\nAnswer:",
]
request = {
    "type": "teacher_forced", "model": "original", "prompt": "bare",
    "questions": discovery_ids, "texts": probe_texts,
    "return": {
        "positions": "last", "token_logprobs": True, "topk": 12,
        "hidden_states": list(range(33)), "mlp_act": list(range(32))
    }
}
print("NEXT_FORWARD_REQUEST =", json.dumps(request))
with notes_path.open("a") as f:
    f.write("\nDiscovery: 64 stratified questions (24 donor-only correct, 16 base-only, "
            "12 both, 12 neither); not population-representative. "
            "Separate tuning panel 512 and development-check panel 1024 saved.\n"
            "BARE letter mass 0.0403 -> 0.4106; fixed letter shift alone "
            "raises development accuracy 47.660% -> 49.507%, versus donor 54.033%. "
            "Question dependence is not yet evidence of medical knowledge.\n")
```
