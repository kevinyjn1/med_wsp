The 1.5 dose improved BARE but reduced CHAT accuracy by 0.98 percentage points and substantially reduced its margin. The accuracy loss is uncertain, yet it is the wrong direction for the guardrail, so I will not carry that dose into confirmation. I will restore the full-dose sparse candidate, whose development CHAT accuracy was unchanged, and stop selecting new weight variants. The remaining work will strengthen the controls and characterize prompt versus content effects before the sealed confirmation.

```python
# Hypothesis: the full-dose sparse transfer offers a safer prompt-adaptation tradeoff than the amplified dose.
print("Dose 1.5 versus 1.0, BARE:", compare_scores(22, 18))
print("Dose 1.5 versus 1.0, CHAT:", compare_scores(23, 19))
install_sparse_transfer(1.0, selected_unit_map)
print("Restored candidate:", len(edited()), "edited tensors")
print("\nTeacher-forced metadata:")
m = json.loads((W / "fwd_1" / "meta.json").read_text())
print(str(m)[:3700])
print("\nNon-hidden output files:")
for p in sorted((W / "fwd_1").iterdir()):
    if p.suffix == ".npy" and not p.name.startswith(("hidden_", "mlp_")):
        a = np.load(p, mmap_mode="r")
        print(p.name, a.shape, a.dtype)
print("\nExisting probe texts:", [t[:250] for t in probe_texts])
print("\nControl mapping structure:")
cm = json.loads((W / "random_location_control.json").read_text())
print(str(cm)[:500])
with (W / "notes.md").open("a") as f:
    f.write("\nForward passes used: 23/30. Sparse alpha1.5 development n512: "
            "score22 BARE +4.30 +/-1.97 pp (62/40 flips), margin +0.3906 +/-0.0987; "
            "score23 CHAT -0.98 +/-1.28 pp (19/24 flips), margin -1.0198 +/-0.2513. "
            "Reject amplified dose for guardrail concern. Restored sparse alpha1.0. "
            "Weight-variant selection is now closed. Final candidate is sparse alpha1.0, "
            "subject to independent confirmation; no fallback will be chosen from "
            "confirmation results. Final decision: retain only with clear BARE gain "
            "and no observed CHAT accuracy decline on sealed n1024; otherwise empty edits. "
            "Remaining passes: larger size-matched location control, mechanistic "
            "teacher-forced/patch diagnostics, then sealed BARE and CHAT confirmation.\n")
```
