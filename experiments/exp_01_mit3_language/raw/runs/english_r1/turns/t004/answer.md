```python
import numpy as np
from scipy.linalg import svd, svdvals
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

threadpool_limits(limits=2)
np.set_printoptions(precision=4, suppress=True, linewidth=180)


def scaled(a, w):
    a = a.astype(float) * w.astype(float)
    return a - a.mean(axis=1, keepdims=True)


s = slice(128, 192)
q = scaled(W["W_q.0"][s], W["ln1_w.0"])
k = scaled(W["W_k.0"][s], W["ln1_w.0"])
v = scaled(W["W_v.0"][s], W["ln1_w.0"])
C = W["W_o.0"][:, s].astype(float) @ v
uq, sq, vq = svd(q.T @ k / 8, full_matrices=False)
uc, sc, vc = svd(C, full_matrices=False)
F = scaled(W["W_fc.0"], W["ln2_w.0"])

print("L0H2 OV singular values:", sc[:45])
print("Top-27 OV read/write subspace overlap:", svdvals(uc[:, :27].T @ vc[:27].T))
print("Top-27 OV input vs top-22 QK positional-candidate overlap:", svdvals(vc[:27] @ uq[:, :22]))
print("First MLP spectrum, indices 40..100:", svdvals(F)[40:101])

# Cluster weight directions, not activations or network inputs.
T = vc[:27].T
X = F @ T
X -= X.mean(axis=0, keepdims=True)
X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
print("\nClustering first-MLP encoder directions restricted to L0H2's OV read subspace")
for n in (16, 20, 22, 24, 26, 28, 30, 32, 40):
    km = KMeans(n_clusters=n, n_init=10, random_state=0).fit(X)
    sil = silhouette_score(X, km.labels_, sample_size=1200, random_state=0)
    print("clusters", n, "silhouette", round(sil, 4), "sizes", np.sort(np.bincount(km.labels_))[::-1])

# A complementary directional structure measure uses the previous-head
# weight composition F @ C, without applying any nonlinearity.
Y = F @ C @ T
Y -= Y.mean(axis=0, keepdims=True)
Y /= np.maximum(np.linalg.norm(Y, axis=1, keepdims=True), 1e-12)
print("\nClustering composed encoder/OV weight directions")
for n in (20, 24, 28, 32):
    km = KMeans(n_clusters=n, n_init=10, random_state=0).fit(Y)
    sil = silhouette_score(Y, km.labels_, sample_size=1200, random_state=0)
    print("clusters", n, "silhouette", round(sil, 4), "sizes", np.sort(np.bincount(km.labels_))[::-1])

print("\nFinal MLP decoder column norm quantiles:")
for l in range(4):
    a = W[f"W_proj.{l}"].astype(float)
    print(l, np.quantile(np.linalg.norm(a, axis=0), [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]))
```
