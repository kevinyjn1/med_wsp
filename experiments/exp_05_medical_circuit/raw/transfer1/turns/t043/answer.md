The paired contrasts give modest evidence for the selected MLP units: they improve the margin over both random units and attention alone, but account for only three or four additional correct answers. Transferring the complete first eight layers produces a larger margin improvement than the sparse version, suggesting that omitted MLP changes may contribute collectively. I will test that broader prefix on the separate 512-question development panel, then check CHAT before considering it further. It is a sufficiency experiment, not a decision to submit a quarter-model replacement.

```python
# Hypothesis: omitted early MLP changes improve the transfer collectively, beyond the sparse prefix.
for n in list(edited()):
    drop_edit(n)
full_prefix_names = [n for n in NAMES
                     if any(n.startswith(f"model.layers.{l}.") for l in range(8))]
prefix_changed_slots = 0
prefix_touched_slots = 0
for n in full_prefix_names:
    base_tensor = load_original(n)
    donor_tensor = load_donor(n)
    if not torch.equal(base_tensor, donor_tensor):
        save_edit(n, donor_tensor)
        prefix_changed_slots += int(torch.count_nonzero(donor_tensor != base_tensor))
        prefix_touched_slots += base_tensor.numel()
    del base_tensor, donor_tensor
gc.collect()
validation_ids = json.loads((W / "validation_ids.json").read_text())
assert len(validation_ids) == 512
assert not (set(validation_ids) & set(confirmation_ids))
print("Full-prefix edited tensors:", len(edited()))
print("Touched slots:", prefix_touched_slots, "numerically changed:", prefix_changed_slots)
print("BARE request:")
print(json.dumps({"type": "score", "model": "both", "prompt": "bare",
                  "questions": validation_ids}, separators=(",", ":")))
with (W / "notes.md").open("a") as f:
    f.write("\nExperiment N: full donor layers0-7 restored; testing on the same separate "
            "512 development questions used for sparse-prefix scores18/19. "
            "Direct contrasts on exploration: selected vs random MLP units +1.562 pp "
            "(4/0 flips), margin +0.0421 +/-0.0176; selected vs attention-only "
            "+1.172 pp (3/0), margin +0.0477 +/-0.0186; full prefix vs sparse "
            "0 pp (11/11), margin +0.1208 +/-0.0343. These are exploratory contrasts.\n")
```
