The complete 512-question activation batch is now available, in addition to the original 128 discovery questions. I will check whether intermediate-to-final reversals replicate and rank individual late-MLP contributions after separating their average letter offset from their question-dependent variation. I will also compare attribution signs across the two disjoint samples, which helps reject fragile discoveries. This remains a way to nominate components, not proof that editing them helps; the next intervention must establish that causally.

```python
# Hypothesis: reproducibly harmful, question-dependent late-MLP contributions can be distinguished from constant letter offsets.
meta12 = json.loads((work / "fwd_12" / "meta.json").read_text())
meta13 = json.loads((work / "fwd_13" / "meta.json").read_text())
assert meta12["ids"] + meta13["ids"] == selection_large_ids
large_hidden = {}
large_acts = {}
for layer in range(20, 33):
    large_hidden[layer] = np.concatenate(
        [np.load(work / f"fwd_{k}" / f"hidden_{layer}.npy").astype(np.float32) for k in [12, 13]]
    )
for layer in range(20, 32):
    large_acts[layer] = np.concatenate(
        [np.load(work / f"fwd_{k}" / f"mlp_act_{layer}.npy").astype(np.float32) for k in [12, 13]]
    )
large_y = np.array([letters.index(q_by_id[q]["answer"]) for q in selection_large_ids])
large_pred = np.array([letters.index(b_by_id[q]["pred"]) for q in selection_large_ids])
large_correct = large_y == large_pred
large_transitions = []
for layer in range(20, 33):
    c = large_hidden[layer] @ (unweighted_readout.numpy().T if layer == 32 else R.T)
    pred = c.argmax(1)
    large_transitions.append(
        {
            "residual_layer": layer,
            "correct": int((pred == large_y).sum()),
            "correct_to_final_wrong": int(((pred == large_y) & ~large_correct).sum()),
            "wrong_to_final_correct": int(((pred != large_y) & large_correct).sum()),
        }
    )
print("Larger-sample intermediate readout transitions:")
print(pd.DataFrame(large_transitions).to_string(index=False))


def best_other_ids(qids):
    result = []
    for qid in qids:
        p = lp_array(b_by_id[qid]).copy()
        p[letters.index(q_by_id[qid]["answer"])] = -np.inf
        result.append(int(p.argmax()))
    return np.array(result)


probe_other = best_other_ids(probe_ids)
large_other = best_other_ids(selection_large_ids)
unit_attribution_rows = []
attribution_cache = {}
for layer in range(20, 32):
    p = letter_projections[layer]
    sample_stats = []
    for x, y, other in [(acts[layer], probe_y, probe_other), (large_acts[layer], large_y, large_other)]:
        # Attribute a fixed true-versus-best-other direction to measured unit activity.
        # Centering removes each unit's constant activation contribution.
        contrast = p[y] - p[other]
        raw = x * contrast
        centered = (x - x.mean(0, keepdims=True)) * contrast
        sample_stats.append((raw.mean(0), centered.mean(0), centered.std(0) / np.sqrt(len(y))))
    d_raw, d_content, d_se = sample_stats[0]
    s_raw, s_content, s_se = sample_stats[1]
    attribution_cache[layer] = sample_stats
    consistent = (d_content < 0) & (s_content < 0)
    rank = np.where(consistent, (d_content + s_content) / 2, np.inf)
    top = np.argsort(rank)[:8]
    for u in top:
        unit_attribution_rows.append(
            {
                "layer": layer,
                "unit": int(u),
                "discovery_raw": float(d_raw[u]),
                "selection_raw": float(s_raw[u]),
                "discovery_centered": float(d_content[u]),
                "selection_centered": float(s_content[u]),
                "selection_se": float(s_se[u]),
                "selectivity": np.round(p[:, u], 2).tolist(),
                "activation_mean": float(large_acts[layer][:, u].mean()),
                "activation_sd": float(large_acts[layer][:, u].std()),
            }
        )
attribution_table = pd.DataFrame(unit_attribution_rows)
attribution_table["average_centered"] = (
    attribution_table.discovery_centered + attribution_table.selection_centered
) / 2
attribution_table = attribution_table.sort_values("average_centered")
print("\nReproducibly negative centered contributions, coordinate units:")
print(attribution_table.head(20).round(3).to_string(index=False))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForwards 12 and 13: original activations for 128+384=512 exposed selection questions. "
        "Forward 12 accidentally requested only 128; forward 13 completed the intended batch.\n"
        "Larger intermediate transitions:\n" + pd.DataFrame(large_transitions).to_string(index=False) + "\n"
        "Centered neuron attribution is descriptive and can reflect confidence suppression; "
        "do not equate its sign with a causal accuracy effect.\n"
        + attribution_table.head(12).round(3).to_string(index=False)
        + "\n"
    )
```