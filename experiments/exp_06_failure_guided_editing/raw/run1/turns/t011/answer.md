The edited activation measurements are now available, allowing a direct check of how the layer-26 intervention propagates. I will compare the original and edited letter-direction coordinates across the remaining layers, then remove that candidate because its behavioral evidence is inconclusive. The next candidate targets four layer-31 neurons whose anti-A contributions were unusually consistent across discovery questions. I will remove only their output components along the A-versus-other-letter direction, preserving their other output directions and making an explicitly letter-bias-focused intervention.

```python
# Hypothesis: stable anti-A components in the final MLP provide a more direct bias-repair target than the compensated layer-26 neuron.
read_score(3)
propagation = []
for layer in range(26, 33):
    he = np.load(work / "fwd_4" / f"hidden_{layer}.npy").astype(np.float32)
    delta = (he - hidden[layer]) @ R.T
    propagation.append(
        {
            "residual_layer": layer,
            "mean_A_change": float(delta[:, 0].mean()),
            "sd_A_change": float(delta[:, 0].std()),
            "mean_ABCD": np.round(delta.mean(0), 3).tolist(),
        }
    )
print("Measured propagation of L26 knockout:")
print(pd.DataFrame(propagation).round(3).to_string(index=False))

bias_name = "model.layers.31.mlp.down_proj.weight"
bias_original = load_original(bias_name)
bias_units = [4775, 6367, 6760, 13]
a_direction = letter_readout[0] / letter_readout[0].norm()
bias_component = a_direction[:, None] * (a_direction @ bias_original[:, bias_units])[None, :]
print("\nProposed final-layer components:")
for u in bias_units:
    c = acts[31][:, u] * letter_projections[31][0, u]
    print(
        {
            "unit": u,
            "A_mean": round(float(c.mean()), 3),
            "A_sd": round(float(c.std()), 3),
            "relative_sd": round(float(c.std() / abs(c.mean())), 3),
        }
    )
bias_discovery_coordinate = acts[31][:, bias_units] @ (letter_readout @ (-bias_component)).numpy().T
print(
    "Removal's direct coordinate change mean ABCD:", np.round(bias_discovery_coordinate.mean(0), 3).tolist()
)
print("Removal's direct coordinate change SD ABCD:", np.round(bias_discovery_coordinate.std(0), 3).tolist())


def set_bias_strength(strength):
    clear_current_edits()
    w = bias_original.clone()
    w[:, bias_units] -= float(strength) * bias_component
    save_edit(bias_name, w)
    candidate_history.append(
        {"kind": "L31_stable_antiA_projection_removal", "strength": float(strength), "units": bias_units}
    )
    return float((w - bias_original).norm() / bias_original.norm())


change = set_bias_strength(1.0)
print("Installed strength=1.0; relative tensor change:", round(change, 6))
with (work / "notes.md").open("a") as f:
    f.write(
        "\nForward 3, L26:4131 scale=0.0, paired n=128: delta accuracy=+1.5625 pp, "
        "SE=1.56 pp; W->R=3, R->W=1; margin delta=-0.0283, SE=0.1235. "
        "A predictions 26 -> 27. Not sufficient to retain.\n"
        "Forward 4 measured edited propagation on the original 128 discovery questions:\n"
        + pd.DataFrame(propagation).round(3).to_string(index=False)
        + "\n"
        "Removed L26 edit. New candidate: L31 down-projection columns [4775,6367,6760,13], "
        "subtract each column's projection onto fixed normalized A-centered readout direction. "
        "All perpendicular components preserved. Strength=1.0. "
        "This candidate explicitly targets letter bias, not medical knowledge.\n"
    )
```