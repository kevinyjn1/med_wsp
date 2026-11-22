```python
import numpy as np
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

threadpool_limits(limits=2)
np.set_printoptions(precision=3, suppress=True, linewidth=180)

g = np.asarray(W["ln1_w.0"], dtype=float)
sl = slice(192, 256)
b = np.asarray(W["W_o.0"][:, sl], dtype=float) @ (np.asarray(W["W_v.0"][sl], dtype=float) * g)
u, s, vt = svd(b, full_matrices=False, check_finite=False)
f = np.asarray(W["W_fc.0"], dtype=float) * np.asarray(W["ln2_w.0"], dtype=float)

# Cluster only weight-derived directions. No layer is evaluated on an input.
for rank in (18, 22, 26):
    r = vt[:rank].T
    current = f @ r
    preceding = (f @ b) @ r
    for whiten in (False, True):
        x = np.vstack([current, preceding])
        x = x - x.mean(axis=0)
        if whiten:
            _, sx, vx = svd(x, full_matrices=False, check_finite=False)
            x = (x @ vx.T) / sx
        x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        print("\nWEIGHT-DIRECTION CLUSTERING rank", rank, "whiten", whiten)
        for k in (12, 16, 20, 22, 24, 26, 28, 32):
            km = KMeans(n_clusters=k, n_init=5, max_iter=200, random_state=19).fit(x)
            lab = km.labels_
            sil = silhouette_score(x, lab, sample_size=1200, random_state=21)
            centers = km.cluster_centers_
            cn = np.linalg.norm(centers, axis=1)
            counts = np.bincount(lab, minlength=k)
            print(
                "k",
                k,
                "silhouette",
                round(float(sil), 4),
                "center_norm median/max",
                np.round([np.median(cn), cn.max()], 3),
                "counts_desc",
                sorted(counts.tolist(), reverse=True),
            )
        if rank == 22:
            k = 24
            km = KMeans(n_clusters=k, n_init=15, random_state=19).fit(x)
            labels = km.labels_
            centers = km.cluster_centers_
            unit = centers / np.linalg.norm(centers, axis=1, keepdims=True)
            sims = x @ unit.T
            confidence = np.sort(sims, axis=1)[:, -1] - np.sort(sims, axis=1)[:, -2]
            c, p = labels[:1536], labels[1536:]
            good = (confidence[:1536] > 0.12) & (confidence[1536:] > 0.12)
            graph = np.zeros((k, k), dtype=int)
            for i in np.flatnonzero(good):
                graph[p[i], c[i]] += 1
            print("rank22 k24 confident weight-pairs:", int(good.sum()))
            print("Rows=preceding-weight cluster; columns=current-weight cluster")
            print(graph)
            print("Cluster-center cosine matrix:")
            print(unit @ unit.T)

print("\nL0H3 OV SPECTRAL ENERGY")
for rank in (5, 10, 15, 18, 20, 22, 24, 26, 28):
    print(rank, round(float(np.sum(s[:rank] ** 2) / np.sum(s**2)), 5))
```