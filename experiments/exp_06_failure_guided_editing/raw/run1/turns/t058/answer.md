Both activation passes completed, and the edited model's average prompt-token log probability changed only slightly. I will now compare the recorded states and attribute changes in fixed letter contrasts to downstream MLP outputs. This analysis can identify candidate pathways, but it cannot establish that any downstream neuron is necessary without another intervention. Afterward, the remaining five passes will evaluate the locked edit on independent questions, including all 1,024 reserved confirmation questions.

```python
# Hypothesis: the edited column leaves the dominant immediate letter contrasts nearly unchanged but changes their downstream processing.
h0 = {l:np.load(WORK/"fwd_24"/f"hidden_{l}.npy").astype(np.float32) for l in range(26,33)}
h1 = {l:np.load(WORK/"fwd_25"/f"hidden_{l}.npy").astype(np.float32) for l in range(26,33)}
a0 = {l:np.load(WORK/"fwd_24"/f"mlp_act_{l}.npy").astype(np.float32) for l in range(26,32)}
a1 = {l:np.load(WORK/"fwd_25"/f"mlp_act_{l}.npy").astype(np.float32) for l in range(26,32)}
print("Before-edit checks:",{"hidden26_max_difference":float(abs(h1[26]-h0[26]).max()),
                            "mlp26_max_difference":float(abs(a1[26]-a0[26]).max())})
coord_delta = {l:(h1[l]-h0[l])@R_fixed.T for l in range(26,33)}
propagation=[]
for l in range(26,33):
    propagation.append({"residual_layer":l,
        "mean_change_ABCD":np.round(coord_delta[l].mean(0),3).tolist(),
        "sd_change_ABCD":np.round(coord_delta[l].std(0),3).tolist(),
        "mean_residual_delta_norm":round(float(np.linalg.norm(h1[l]-h0[l],axis=1).mean()),3)})
print("Unnormalized fixed-direction propagation:")
print(pd.DataFrame(propagation).to_string(index=False))
mediation=[]
for l in range(27,32):
    w=load_original(f"model.layers.{l}.mlp.down_proj.weight")
    p=(torch.from_numpy(R_fixed)@w).numpy()
    da=a1[l]-a0[l]
    dm=da@p.T
    dr=coord_delta[l+1]-coord_delta[l]
    per_neuron_A=da.mean(0)*p[0]
    inds=np.argsort(abs(per_neuron_A))[-3:][::-1]
    mediation.append({"layer":l,"MLP_mean_ABCD":np.round(dm.mean(0),3).tolist(),
        "non_MLP_remainder_ABCD":np.round((dr-dm).mean(0),3).tolist(),
        "largest_A_changes":[(int(j),round(float(per_neuron_A[j]),3)) for j in inds]})
    del w,p,da
print("Descriptive downstream attribution:")
print(pd.DataFrame(mediation).to_string(index=False))
top0=np.load(WORK/"fwd_24"/"topk_ids.npy")
top1=np.load(WORK/"fwd_25"/"topk_ids.npy")
primary_ids=[raw_letter_ids[s]["space"] for s in "ABCD"]
print("Primary letter tokens among top1, original/edited:",int(np.isin(top0[:,0],primary_ids).sum()),int(np.isin(top1[:,0],primary_ids).sum()))
print("Corrected development examples (excerpts only):")
for r in [r for r in rows_strong if not r["original"]["correct"] and r["edited"]["correct"]][:6]:
    q=q_by_id[r["id"]]
    print(r["id"],r["original"]["pred"]+"->"+r["edited"]["pred"],
          str(q["question"]).replace("\n"," ")[:160].encode("ascii","replace").decode())
with (WORK/"notes.md").open("a") as f:
    f.write("\nForward24/25 matched original/locked-edit activations, screening128, NEW layer32 convention. "
        "Fixed readout matches dominant space-prefixed tokens; orthogonality does NOT hold for all bare-token contrasts. "
        "Downstream propagation is descriptive; no claim of independently established medical reasoning circuit.\n"+
        pd.DataFrame(propagation).to_string(index=False)+"\n"+
        pd.DataFrame(mediation).to_string(index=False)+"\n")
gc.collect()
```
