The downstream patch effects largely shrink after the early transfer: layer 25 head 9 falls from a +0.348 margin change to +0.021, and layer 11’s MLP falls from +0.420 to +0.102. This supports the upstream-source interpretation, although residual patch benefits show that the candidate does not reproduce the whole donor. I will now verify the exact edit and open only the identifiers of the sealed 1,024-question panel. Both final prompts will use the same unchanged candidate; retention requires a BARE gain above two paired standard errors and no observed CHAT accuracy loss.

```python
# Hypothesis: the frozen sparse candidate generalizes beyond every question used for selection.
confirmation_ids = json.loads((W / "confirmation_ids.json").read_text())
assert len(confirmation_ids) == len(set(confirmation_ids)) == 1024
seen_score_ids = set()
for p in W.glob("score_*.jsonl"):
    for line in p.read_text().splitlines():
        seen_score_ids.add(json.loads(line)["id"])
assert set(confirmation_ids).isdisjoint(seen_score_ids)
assert set(confirmation_ids).isdisjoint(discovery_ids)

verified_names = []
for name in edited():
    assert name.startswith("model.layers.")
    layer = int(name.split(".")[2])
    assert 0 <= layer < 8
    b = load_original(name)
    d = load_donor(name)
    e = load(name)
    u = torch.tensor(selected_unit_map[str(layer)], dtype=torch.long)
    if ".mlp.gate_up_proj." in name:
        rr = torch.cat([u, u+8192])
        b[rr] = d[rr]
        expected = b
    elif ".mlp.down_proj." in name:
        b[:,u] = d[:,u]
        expected = b
    else:
        expected = d
    assert torch.equal(e.to(torch.bfloat16), expected.to(torch.bfloat16)), name
    verified_names.append(name)
    del b,d,e,expected
gc.collect()
assert len(verified_names) == 46
print("Verified exact sparse donor transfer in", len(verified_names), "tensors.")
print("Confirmation overlaps: zero. Candidate remains frozen.")
with (W / "notes.md").open("a") as f:
    f.write("\nForward passes used: 28/30. Donor patch effects on discovery diminish "
            "after candidate transfer: L25H9 margin +0.3477 -> +0.0205; L18H7 "
            "+0.1929 -> +0.0127; L7H1 +0.1860 -> +0.0176; L11MLP "
            "+0.4204 -> +0.1016; L18 attention +0.4326 -> +0.1128. "
            "Consistent with downstream readers of an upstream change, not unique proof. "
            "Candidate exact support and donor values verified after bfloat16. "
            "Sealed n1024 confirmed disjoint from all prior score and discovery IDs. "
            "Final two passes will score this unchanged candidate under BARE and CHAT. "
            "No alternative candidate will be selected using confirmation outcomes.\n")
print("CONFIRMATION IDS FIRST HALF:")
print(json.dumps(confirmation_ids[:512], separators=(",", ":")))
```
