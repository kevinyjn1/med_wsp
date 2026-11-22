The activation pass completed successfully, giving measurements from every layer on the discovery batch. I will now trace where the residual stream acquires its A-versus-other-letter preference and identify which MLP neurons contribute most to that change. These will be descriptive projections onto fixed weight directions, not a substitute for running the model or evidence that removing a neuron will help. A relatively constant anti-A contribution would support a bias-repair experiment, while contributions that track the correct option would warn that the same neurons carry useful content.

```python
# Hypothesis: the anti-A tendency can be localized to a few measured MLP contributions rather than an undifferentiated output effect.
fwd1 = work / "fwd_1"
meta1 = json.loads((fwd1 / "meta.json").read_text())
print("Metadata keys:", list(meta1))
print("Metadata preview:", json.dumps(meta1, ensure_ascii=True)[:1400])
probe_y = np.array([letters.index(q_by_id[qid]["answer"]) for qid in probe_ids])
probe_pred = np.array([letters.index(b_by_id[qid]["pred"]) for qid in probe_ids])
probe_correct = probe_y == probe_pred
R = letter_readout.numpy()
acts = {}
hidden = {}
for layer in range(33):
    hidden[layer] = np.load(fwd1 / f"hidden_{layer}.npy").astype(np.float32)
for layer in range(32):
    acts[layer] = np.load(fwd1 / f"mlp_act_{layer}.npy").astype(np.float32)

# Fixed-direction projections are descriptive measurements, not predicted edited-model outputs.
neuron_rows = []
layer_rows = []
for layer in range(32):
    x = acts[layer]
    p = letter_projections[layer]
    mu, sd = x.mean(0), x.std(0)
    h_before = hidden[layer] @ R.T
    h_after = hidden[layer + 1] @ R.T
    mlp_mean = p @ mu
    delta = (h_after - h_before).mean(0)
    layer_rows.append(
        {
            "layer": layer,
            "A_residual_delta": float(delta[0]),
            "A_mlp_mean": float(mlp_mean[0]),
            "A_attention_remainder": float(delta[0] - mlp_mean[0]),
            "fixed_direction_accuracy": float((h_after.argmax(1) == probe_y).mean()),
            "A_coordinate_mean": float(h_after[:, 0].mean()),
        }
    )
    a_mean = mu * p[0]
    top = np.argsort(a_mean)[:8]
    for unit in top:
        c = x[:, unit] * p[0, unit]
        by_truth = [float(c[probe_y == i].mean()) for i in range(4)]
        by_pred = [float(c[probe_pred == i].mean()) if np.any(probe_pred == i) else None for i in range(4)]
        neuron_rows.append(
            {
                "layer": layer,
                "unit": int(unit),
                "A_mean": float(c.mean()),
                "A_sd": float(c.std()),
                "act_mean": float(mu[unit]),
                "act_sd": float(sd[unit]),
                "positive_fraction": float((x[:, unit] > 0).mean()),
                "letter_projections": np.round(p[:, unit], 3).tolist(),
                "by_true_letter": np.round(by_truth, 2).tolist(),
                "by_pred_letter": [None if z is None else round(z, 2) for z in by_pred],
            }
        )
neuron_table = pd.DataFrame(neuron_rows).sort_values("A_mean")
print("\nDiscovery truth counts:", np.bincount(probe_y, minlength=4).tolist())
print("Discovery baseline accuracy:", round(100 * probe_correct.mean(), 2))
print("\nLayer projections, in unnormalized coordinate units:")
print(pd.DataFrame(layer_rows).round(3).to_string(index=False))
print("\nLargest anti-A neuron contributions:")
print(neuron_table.head(24).round(3).to_string(index=False))

# Include strongest late-layer letter-selective neurons irrespective of contribution sign.
print("\nLate-layer strongest mean letter contributions:")
for layer in [22, 27, 28, 29, 30, 31]:
    c = letter_projections[layer] * acts[layer].mean(0)[None, :]
    units = np.argsort(np.linalg.norm(c, axis=0))[-6:][::-1]
    print(
        "layer",
        layer,
        [
            {
                "unit": int(u),
                "mean_ABCD": np.round(c[:, u], 2).tolist(),
                "act_mean": round(float(acts[layer][:, u].mean()), 3),
                "act_sd": round(float(acts[layer][:, u].std()), 3),
            }
            for u in units
        ],
    )

with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 1: original, 128 discovery questions, all-layer final-position hidden states and MLP activations.\n"
    )
    f.write("Layer coordinate measurements (descriptive, not causal):\n")
    f.write(pd.DataFrame(layer_rows).round(3).to_string(index=False) + "\n")
    f.write("Top anti-A contributors:\n" + neuron_table.head(12).round(3).to_string(index=False) + "\n")
print("\nScreen IDs:", json.dumps(screen_ids))
```