```python
import numpy as np
from scipy.linalg import svd, svdvals
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=190)


def c(a):
    return a - a.mean(-1, keepdims=True)


def norm(a):
    return np.linalg.norm(a)


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
scales = [norm(a - a.mean(0)) for a in As]
stack = np.vstack([(a - a.mean(0)) / scale for a, scale in zip(As, scales)])
fit = FastICA(n_components=28, whiten="unit-variance", random_state=31, max_iter=5000, tol=0.0003)
z = fit.fit_transform(stack)
sgn = np.where(np.mean(z**3, 0) < 0, -1, 1)
z *= sgn
zs = [z[i * 1536 : (i + 1) * 1536] for i in range(3)]
print("Joint weight dictionary iterations:", fit.n_iter_)
print("Joint coefficient kurtoses:", kurtosis(z, axis=0))
for i, a in enumerate(zs):
    print("family", i, "coefficient std:", a.std(0))
    print("family", i, "kurtoses:", kurtosis(a, axis=0))

# Tabulate coincidences among large coefficients of different weight products.
# These are descriptive statistics of the weights, not a layer evaluation.
Z = [(a - a.mean(0)) / (a.std(0) + 1e-30) for a in zs]
print("\nPer-axis coefficient correlations, repeated-axis counts, and marginal counts")
for j in range(28):
    cc = [np.mean(Z[0][:, j] * Z[i][:, j]) for i in (1, 2)]
    both = [int(np.sum((Z[0][:, j] > 2.5) & (Z[i][:, j] > 2.5))) for i in (1, 2)]
    triple = int(np.sum((Z[0][:, j] > 2) & (Z[1][:, j] > 2) & (Z[2][:, j] > 2)))
    counts = [int(np.sum(a[:, j] > 2.5)) for a in Z]
    print(j, "corr", np.round(cc, 3), "pairs", both, "triple", triple, "marginals", counts)

print("\nStructural read strengths of joint weight-dictionary axes")
mix = fit.mixing_ * sgn
D = T @ np.linalg.pinv(mix).T
D /= np.linalg.norm(D, axis=0)
for j in (0, 2, 3, 4):
    s = slice(64 * j, 64 * j + 64)
    vals = []
    for name in ("W_q", "W_k", "W_v"):
        vals.append(np.linalg.norm(c(W[f"{name}.0"][s].astype(float) * g) @ D, axis=0))
    print("head", j, "Q", vals[0], "K", vals[1], "V", vals[2])

print("\nFinal weight-only summary: QK Frobenius norm, key-bias norm, and ratio")
for l in range(4):
    gg = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s = slice(64 * j, 64 * j + 64)
        q = c(W[f"W_q.{l}"][s].astype(float) * gg)
        k = c(W[f"W_k.{l}"][s].astype(float) * gg)
        m = q.T @ k / 8
        kb = k.T @ (W[f"W_q.{l}"][s].astype(float) @ b) / 8
        print(f"L{l}H{j}", np.round([norm(m), norm(kb), norm(kb) / norm(m)], 5))
```
