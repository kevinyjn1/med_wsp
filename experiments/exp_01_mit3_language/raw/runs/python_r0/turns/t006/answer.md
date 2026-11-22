```python
import numpy as np
from scipy.linalg import svd
from scipy.optimize import linear_sum_assignment
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=180)


def c(a):
    return a - a.mean(-1, keepdims=True)


g = W["ln1_w.0"].astype(float)
_, _, vh = svd(
    np.vstack([c(W["W_v.0"][64 * j : 64 * j + 64].astype(float) * g) for j in (1, 5)]), full_matrices=False
)
T = vh[:28].T
F = c(W["W_fc.0"].astype(float) * W["ln2_w.0"].astype(float))
maps = [np.eye(384)]
for j in (5, 1):
    s = slice(64 * j, 64 * j + 64)
    maps.append(W["W_o.0"][:, s].astype(float) @ c(W["W_v.0"][s].astype(float) * g))
As = [F @ m @ T for m in maps]
Ss = []
Ds = []
for i, a in enumerate(As):
    fit = FastICA(n_components=28, whiten="unit-variance", random_state=13, max_iter=5000, tol=0.0005)
    z = fit.fit_transform(a)
    d = fit.mixing_.copy()
    signs = np.where(np.mean(z**3, axis=0) < 0, -1, 1)
    z *= signs
    d *= signs
    Ss.append(z)
    Ds.append(d)
    print("family", i, "iterations", fit.n_iter_, "kurtoses", kurtosis(z, axis=0))
base = Ds[0] / np.linalg.norm(Ds[0], axis=0)
for i in (1, 2):
    other = Ds[i] / np.linalg.norm(Ds[i], axis=0)
    similarity = base.T @ other
    ii, jj = linear_sum_assignment(-similarity)
    order = jj[np.argsort(ii)]
    Ds[i] = Ds[i][:, order]
    Ss[i] = Ss[i][:, order]
    print("family", i, "matched dictionary cosines:", similarity[np.arange(28), order])
    print(
        "family",
        i,
        "top-vs-second dictionary cosine gaps:",
        np.sort(similarity, axis=1)[:, -1] - np.sort(similarity, axis=1)[:, -2],
    )

# Statistics of coefficients in a decomposition of the weights, not activations.
for source in (1, 2):
    curr = Ss[0]
    past = Ss[source]
    mask = (curr.max(1) > 3) & (past.max(1) > 3)
    pairs = np.zeros((28, 28), int)
    for n in np.flatnonzero(mask):
        pairs[np.argmax(past[n]), np.argmax(curr[n])] += 1
    print("\nCoefficient co-occurrences: family", source, "to direct; qualifying weight rows", mask.sum())
    for i in range(28):
        ids = np.argsort(pairs[i])[::-1][:8]
        print(i, [(int(j), int(pairs[i, j])) for j in ids if pairs[i, j] > 0])

print("\nStrongest direct-family weight rows and their lag-family largest coefficients")
for a in range(28):
    inds = np.argsort(Ss[0][:, a])[-3:][::-1]
    print(
        "axis",
        a,
        [
            (
                int(n),
                round(Ss[0][n, a], 2),
                [(int(k), round(Ss[1][n, k], 2)) for k in np.argsort(Ss[1][n])[-2:][::-1]],
                [(int(k), round(Ss[2][n, k], 2)) for k in np.argsort(Ss[2][n])[-2:][::-1]],
            )
            for n in inds
        ],
    )
```
