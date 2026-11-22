```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=180)


def c(a):
    return a - a.mean(-1, keepdims=True)


def nr(a):
    return np.trace(a) / (svdvals(a).sum() + 1e-30)


g0 = W["ln1_w.0"].astype(float)
_, _, vh = svd(
    np.vstack([c(W["W_v.0"][64 * j : 64 * j + 64].astype(float) * g0) for j in (1, 5)]), full_matrices=False
)
M = []
A = []
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        s = slice(64 * j, 64 * j + 64)
        q = c(W[f"W_q.{l}"][s].astype(float) * g)
        k = c(W[f"W_k.{l}"][s].astype(float) * g)
        M.append(q.T @ k / 8)
        A.append(W[f"W_o.{l}"][:, s].astype(float) @ c(W[f"W_v.{l}"][s].astype(float) * g))

print("Robustness of key-side and query-side composition through L0H5")
print("Each pair is trace/nuclear norm; token-reference ranks 20,24,28,32,40.")
for i in (7, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23):
    rows = []
    for r in (20, 24, 28, 32, 40):
        t = vh[:r].T
        rows.append((nr(t.T @ M[i] @ A[5] @ t), nr(t.T @ A[5].T @ M[i] @ t)))
    print(f"L{i // 6}H{i % 6}", np.array(rows))

# A deterministic permutation control for the same algebraic statistic.
# Permutes weight-derived reference directions at one end of the composition.
t = vh[:28].T
rng = np.random.default_rng(2026)
print("\nPermutation controls: observed key-side ratio, null mean/std, null maximum")
for i in (13, 19, 23):
    a = t.T @ M[i] @ A[5] @ t
    denom = svdvals(a).sum()
    observed = np.trace(a) / denom
    null = np.array([np.trace(a[:, rng.permutation(28)]) / denom for _ in range(2000)])
    print(f"L{i // 6}H{i % 6}", np.array([observed, null.mean(), null.std(), null.max()]))

print("\nExact matrix slices supporting the broad first-layer head:")
for name in ("W_q", "W_k", "W_v"):
    a = W[f"{name}.0"][256:320].astype(float)
    print(name + ".0[256:320]", "Frobenius", np.linalg.norm(a))
o = W["W_o.0"][:, 256:320].astype(float)
v = W["W_v.0"][256:320].astype(float)
print("W_o.0[:,256:320] @ W_v.0[256:320] top singular values:", svdvals(o @ v)[:5])
```
