```python
import numpy as np
from scipy.linalg import svd, svdvals
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=180)


def c(a):
    return a - a.mean(-1, keepdims=True)


g0 = W["ln1_w.0"].astype(float)
v = np.vstack([c(W["W_v.0"][64 * j : 64 * j + 64].astype(float) * g0) for j in (1, 5)])
_, ss, vh = svd(v, full_matrices=False)
T = vh[:27].T

F = c(W["W_fc.0"].astype(float) * W["ln2_w.0"].astype(float))
maps = [np.eye(384)]
for j in (5, 1):
    sl = slice(64 * j, 64 * j + 64)
    maps.append(W["W_o.0"][:, sl].astype(float) @ c(W["W_v.0"][sl].astype(float) * g0))
A = [F @ m @ T for m in maps]

print("MLP0 weight projections: direct, through H5, through H1")
for a in A:
    print("norm", np.linalg.norm(a), "spectrum", svdvals(a)[:27])

ica = FastICA(n_components=27, whiten="unit-variance", random_state=13, max_iter=3000, tol=0.0005)
S = ica.fit_transform(A[0])
mix = ica.mixing_
# Each axis below is fitted to weight rows only, not model activations.
for i in range(27):
    if np.mean(S[:, i] ** 3) < 0:
        S[:, i] *= -1
        mix[:, i] *= -1
D = [a @ np.linalg.pinv(mix).T for a in A]
# Center each family of weight coefficients, without thresholding or nonlinear layers.
D = [a - a.mean(0) for a in D]
D = [a / (a.std(0) + 1e-20) for a in D]
print("ICA iterations", ica.n_iter_)
print("Current-weight ICA excess kurtoses:", kurtosis(S, axis=0))
print("Cross-family diagonal correlations H5 and H1:")
print(np.mean(D[0] * D[1], axis=0))
print(np.mean(D[0] * D[2], axis=0))
print("Per-axis strong-positive coefficient counts in each weight family (z>2):")
print(np.array([[(d[:, i] > 2).sum() for d in D] for i in range(27)]))
print("Per-axis three-family simultaneous positive coefficient counts (all z>1.5):")
print(np.sum((D[0] > 1.5) & (D[1] > 1.5) & (D[2] > 1.5), axis=0))

print("\nL0 MLP projection-output norms of strongest repeated-axis candidate weight rows")
rep = np.max(np.minimum(np.minimum(D[0], D[1]), D[2]), axis=1)
ids = np.argsort(rep)[-25:][::-1]
for n in ids:
    axis = np.argmax(np.minimum(np.minimum(D[0][n], D[1][n]), D[2][n]))
    print(
        int(n),
        "axis",
        int(axis),
        "coefficients",
        np.array([d[n, axis] for d in D]),
        "proj-column-norm",
        np.linalg.norm(W["W_proj.0"][:, n]),
    )

print("\nFinal MLP projection singular values:")
print(svdvals(W["W_proj.3"].astype(float))[:65])
```
