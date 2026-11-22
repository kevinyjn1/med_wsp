```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse.csgraph import laplacian

np.set_printoptions(precision=3, suppress=True, linewidth=200)
d = 384
P = np.eye(d) - np.ones((d, d)) / d
F = (W["W_fc.0"].astype(float) * W["ln2_w.0"]) @ P
sl = slice(192, 256)
B = P @ W["W_o.0"][:, sl].astype(float) @ (W["W_v.0"][sl].astype(float) * W["ln1_w.0"]) @ P
G = F @ B
Gc = G - G.mean(0)
u, s, vt = svd(Gc, full_matrices=False)
print("Singular values of the static weight product W_fc.0 D_ln2.0 P B_L0H3:", s[:50])
for r, k in [(20, 22), (24, 24), (28, 28)]:
    T = vt[:r].T / (s[:r] ** 0.7)
    X = Gc @ T
    Y = (F - F.mean(0)) @ T
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    km = KMeans(n_clusters=k, n_init=15, random_state=0).fit(X)
    C = km.cluster_centers_
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    cosx = X @ C.T
    cosy = Y @ C.T
    ix = np.argmax(cosx, axis=1)
    iy = np.argmax(cosy, axis=1)
    print("\nrank/k", r, k, "silhouette", silhouette_score(X, ix, sample_size=1000, random_state=0))
    print(
        "nearest center cosine, prev/current",
        np.quantile(cosx.max(1), [0.1, 0.5, 0.9]),
        np.quantile(cosy.max(1), [0.1, 0.5, 0.9]),
    )
    print("cluster counts prev", np.bincount(ix, minlength=k))
    print("cluster counts current", np.bincount(iy, minlength=k))
    if r == 24:
        tab = np.zeros((k, k), dtype=int)
        for a, b in zip(ix, iy):
            tab[a, b] += 1
        print("Paired static-weight cluster counts: row=F B, column=F")
        print(tab)
        H = tab + tab.T
        np.fill_diagonal(H, 0)
        ev, vec = np.linalg.eigh(laplacian(H, normed=True))
        print("Normalized graph Laplacian eigenvalues", ev)
        print("Largest-eigenvalue sign partition", np.where(vec[:, -1] >= 0)[0], np.where(vec[:, -1] < 0)[0])
        print("Largest-eigenvector coordinates", vec[:, -1])
```
