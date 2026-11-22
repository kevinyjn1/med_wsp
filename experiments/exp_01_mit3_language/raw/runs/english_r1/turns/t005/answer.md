```python
import numpy as np
from scipy.linalg import svd, orth, svdvals
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from threadpoolctl import threadpool_limits

threadpool_limits(limits=2)
np.set_printoptions(precision=4, suppress=True, linewidth=180)


def scaled(a, w):
    a = a.astype(float) * w.astype(float)
    return a - a.mean(1, keepdims=True)


F = scaled(W["W_fc.0"], W["ln2_w.0"])
pn = np.linalg.norm(W["W_proj.0"].astype(float), axis=0)
fn = np.linalg.norm(F, axis=1)
print("First MLP decoder norms, counts above thresholds:")
for t in (0.3, 0.5, 0.7, 1.0, 1.5, 2.0):
    print(t, int((pn > t).sum()))
print("Encoder/decoder norm correlation:", np.corrcoef(fn, pn)[0, 1])
print("Encoder norm quantiles:", np.quantile(fn, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1]))

Cs = []
output_spaces = []
for j in range(6):
    sl = slice(j * 64, (j + 1) * 64)
    v = scaled(W["W_v.0"][sl], W["ln1_w.0"])
    C = W["W_o.0"][:, sl].astype(float) @ v
    Cs.append(C)
    u, s, vt = svd(C, full_matrices=False)
    r = 36 if j == 2 else (6 if j == 3 else 3)
    output_spaces.append(u[:, :r])

sl = slice(128, 192)
q = scaled(W["W_q.0"][sl], W["ln1_w.0"])
k = scaled(W["W_k.0"][sl], W["ln1_w.0"])
u, s, vt = svd(q.T @ k / 8, full_matrices=False)
N = orth(np.concatenate(output_spaces + [u[:, :24], vt[:24].T], axis=1))
Fc = F - (F @ N) @ N.T
active = pn > 0.7
u, s, vt = svd(Fc[active], full_matrices=False)
print("Projected active encoder singular values:", s[:60])
T = vt[:27].T
X = Fc[active] @ T
X -= X.mean(0, keepdims=True)
X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
print("\nActive encoder clustering after removing attention-write and QK subspaces:")
for n in (16, 20, 22, 24, 26, 28, 30, 32, 36, 40):
    km = KMeans(n_clusters=n, n_init=20, random_state=0).fit(X)
    print(
        n,
        "silhouette",
        round(silhouette_score(X, km.labels_), 4),
        "sizes",
        np.sort(np.bincount(km.labels_))[::-1],
    )

# Also report spectra rather than assume an alphabet size.
print("\nEffective ranks of active first-MLP encoder residual:")
for energy in (0.8, 0.9, 0.95, 0.99):
    print(energy, int(np.searchsorted(np.cumsum(s * s) / np.sum(s * s), energy) + 1))
```
