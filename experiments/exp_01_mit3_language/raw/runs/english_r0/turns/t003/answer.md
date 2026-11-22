```python
import numpy as np
from scipy.linalg import svd, svdvals
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=3, suppress=True, linewidth=190)
with threadpool_limits(limits=2):
    print("Longer singular-value profiles")
    for name in ("W_v.0", "W_fc.0", "W_proj.0", "W_fc.3", "W_proj.3"):
        s = svdvals(W[name].astype(float))
        print(name, np.round(s[:80], 3))

    sl = slice(128, 192)
    v = W["W_v.0"][sl].astype(float)
    o = W["W_o.0"][:, sl].astype(float)
    _, sv, vt = svd(v, full_matrices=False)
    print("\nL0H2 unscaled V singular values:", np.round(sv, 3))
    t = vt[:27].T
    f = W["W_fc.0"].astype(float) * W["ln2_w.0"]
    f -= f.mean(axis=1, keepdims=True)
    b = o @ (v * W["ln1_w.0"])
    b -= b.mean(axis=1, keepdims=True)
    direct = f @ t
    previous = f @ b @ t
    print("Direct readout singular values:", np.round(svdvals(direct), 3))
    print("OV-composed readout singular values:", np.round(svdvals(previous), 3))

    def whiten_columns(z, keep):
        u, s, vt = svd(z - z.mean(0), full_matrices=False)
        return u[:, :keep] * np.sqrt(len(z)), s

    direct_z, _ = whiten_columns(direct, 24)
    previous_z, _ = whiten_columns(previous, 24)
    for label, z in (("direct", direct_z), ("OV-composed", previous_z)):
        z = z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
        print("\n", label, "neuron-direction clustering")
        for nc in (16, 20, 24, 28, 32, 40):
            fit = KMeans(n_clusters=nc, n_init=10, random_state=1).fit(z)
            counts = np.bincount(fit.labels_)
            cos = np.sum(z * fit.cluster_centers_[fit.labels_], axis=1) / np.maximum(
                np.linalg.norm(fit.cluster_centers_[fit.labels_], axis=1), 1e-12
            )
            print(
                f"k={nc}: mean_cos={cos.mean():.3f}, median_cos={np.median(cos):.3f}, counts={sorted(counts.tolist(), reverse=True)}"
            )

    print("\nFirst-layer MLP row alignment with its own output columns")
    fc = W["W_fc.0"].astype(float)
    proj = W["W_proj.0"].astype(float).T
    cos = np.sum(fc * proj, axis=1) / (np.linalg.norm(fc, axis=1) * np.linalg.norm(proj, axis=1))
    print("cosine percentiles:", np.percentile(cos, [0, 10, 25, 50, 75, 90, 100]))
    print("Effective LN-scaled first MLP column energy, largest 30 coordinates:")
    e = np.linalg.norm(f, axis=0)
    ix = np.argsort(e)[-30:][::-1]
    print(list(zip(ix.tolist(), np.round(e[ix], 3).tolist())))
```
