```python
import numpy as np
from scipy.linalg import svd, orth
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
active = pn > 0.7
C = []
spaces = []
for j in range(6):
    sl = slice(j * 64, j * 64 + 64)
    c = W["W_o.0"][:, sl].astype(float) @ scaled(W["W_v.0"][sl], W["ln1_w.0"])
    C.append(c)
    u, s, vt = svd(c, full_matrices=False)
    spaces.append(u[:, : 64 if j == 2 else (6 if j == 3 else 3)])
N = orth(np.concatenate(spaces, axis=1))
Fres = F - (F @ N) @ N.T
Fa = Fres[active]
Fa -= Fa.mean(0, keepdims=True)
u, s, vt = svd(Fa, full_matrices=False)
print("Centered active encoder spectrum after removing OV-write subspaces:", s[:45])
T = vt[:27].T
X = Fa @ T
Y = F[active] @ C[2] @ T
Y -= Y.mean(0, keepdims=True)

for label, A in [("current-side", X), ("previous-side", Y)]:
    U, S, V = svd(A, full_matrices=False)
    print("\n", label, "singular values:", S)
    for whitening in (0.0, 0.5, 1.0):
        Z = (A @ V.T) / (S**whitening)
        Z /= np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        print("whitening", whitening)
        for n in (20, 24, 28, 32, 40):
            km = KMeans(n_clusters=n, n_init=15, random_state=1).fit(Z)
            print(n, round(silhouette_score(Z, km.labels_), 4), end="; ")
        print()

# Cross-view similarities between weight-derived encoder directions.
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
Yn = Y / np.linalg.norm(Y, axis=1, keepdims=True)
sim = Xn @ Yn.T
print(
    "\nCross-view nearest-direction cosine quantiles:",
    np.quantile(sim.max(1), [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]),
)
print(
    "Same-neuron cross-view cosine quantiles:", np.quantile(np.diag(sim), [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
)
```
