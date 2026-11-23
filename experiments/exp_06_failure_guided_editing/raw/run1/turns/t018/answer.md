The larger test did not support the letter-bias repair: accuracy fell by 0.78 percentage points, with seven corrections and eleven regressions. I will remove that edit rather than equate more balanced letter counts with better answers. Next I will investigate questions where an intermediate residual representation favors the correct answer but the final model chooses incorrectly, looking for late attention components that overwrite useful information. These intermediate projections are only diagnostic leads; a candidate will still need a targeted intervention and independent confirmation.

```python
# Hypothesis: some errors arise from late changes in content-dependent answer selection, rather than a removable constant letter preference.
changes7 = paired_details(7)
clear_current_edits()
head_tmp = load_original("lm_head.weight")
unweighted_readout = torch.stack([head_tmp[letter_token_ids[a]].mean(0) for a in letters])
unweighted_readout -= unweighted_readout.mean(0, keepdim=True)
del head_tmp
final_coordinates = hidden[32] @ unweighted_readout.numpy().T
print(
    "Final exported-state letter projection agrees with baseline on:",
    int((final_coordinates.argmax(1) == probe_pred).sum()),
    "of",
    len(probe_ids),
)

early_coordinates = {l: hidden[l] @ R.T for l in range(19, 32)}
flip_summary = []
for l, c in early_coordinates.items():
    early_pred = c.argmax(1)
    flip_summary.append(
        {
            "residual_layer": l,
            "projection_correct": int((early_pred == probe_y).sum()),
            "correct_then_final_wrong": int(((early_pred == probe_y) & (~probe_correct)).sum()),
            "wrong_then_final_correct": int(((early_pred != probe_y) & probe_correct).sum()),
        }
    )
print("\nIntermediate-answer transitions:")
print(pd.DataFrame(flip_summary).to_string(index=False))
late_flip_indices = np.where((early_coordinates[26].argmax(1) == probe_y) & (~probe_correct))[0]
attention_probe_ids = [probe_ids[int(i)] for i in late_flip_indices[:4]]
if len(attention_probe_ids) < 4:
    attention_probe_ids += [q for q in probe_ids if q not in attention_probe_ids][
        : 4 - len(attention_probe_ids)
    ]
print("Attention probe IDs:", json.dumps(attention_probe_ids))
for qid in attention_probe_ids:
    q = q_by_id[qid]
    print(
        json.dumps(
            {
                "id": qid,
                "answer": q["answer"],
                "pred": b_by_id[qid]["pred"],
                "excerpt": str(q["question"])[:180],
            },
            ensure_ascii=True,
        )
    )

head_weight_rows = []
for layer in [21, 25, 27, 28, 29, 30, 31]:
    o = load_original(f"model.layers.{layer}.self_attn.o_proj.weight")
    for h in range(32):
        block = o[:, 96 * h : 96 * (h + 1)]
        fraction = float((letter_basis.T @ block).norm() / block.norm())
        head_weight_rows.append(
            {
                "layer": layer,
                "head": h,
                "letter_fraction": fraction,
                "A_projection_norm": float((letter_readout[0] @ block).norm()),
                "block_norm": float(block.norm()),
            }
        )
    del o
head_weight_table = pd.DataFrame(head_weight_rows)
print("\nMost letter-selective attention output blocks:")
print(
    head_weight_table.sort_values("letter_fraction", ascending=False).head(16).round(4).to_string(index=False)
)
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 7, bias strength=3, paired selection n=512: 56.8359% -> 56.0547%; "
        "delta=-0.78125 pp, SE=0.83 pp; W->R=7, R->W=11; "
        "margin delta=-0.0965, SE=0.1209. Bias edit removed: no demonstrated benefit.\n"
        "IMPORTANT CORRECTION: hidden_32 is already final-normalized: RMS(hidden_32/gamma)=1. "
        "Earlier claim of strong final-layer compensation conflated normalization with cancellation. "
        "Pre-final-layer raw coordinate attenuation was 11.303 -> 8.550.\n"
        "Next hypothesis: late attention overwrites useful content-dependent answer signals. "
        "Intermediate readout transitions are leads, not causal evidence.\n"
        + pd.DataFrame(flip_summary).to_string(index=False)
        + "\n"
    )
print("Current edits:", edited())
```