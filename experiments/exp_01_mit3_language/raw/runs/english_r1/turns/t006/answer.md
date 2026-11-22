```python
import numpy as np
from scipy.linalg import svd, svdvals, eig
from threadpoolctl import threadpool_limits

threadpool_limits(limits=2)
np.set_printoptions(precision=4, suppress=True, linewidth=180)


def scaled(a, w):
    a = a.astype(float) * w.astype(float)
    return a - a.mean(1, keepdims=True)


B, C, g = {}, {}, {}
for l in range(4):
    w = W[f"ln1_w.{l}"]
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        sl = slice(64 * j, 64 * j + 64)
        q, k, v = [scaled(W[f"W_{t}.{l}"][sl], w) for t in ("q", "k", "v")]
        B[l, j] = q.T @ k / 8
        C[l, j] = W[f"W_o.{l}"][:, sl].astype(float) @ v
        g[l, j] = (W[f"W_q.{l}"][sl].astype(float) @ b) @ k / 8

# A shared L1 weight pattern, used only as a matrix-comparison reference.
R = sum(B[1, j] / np.linalg.norm(B[1, j]) for j in range(6)) / 6
u, s, vt = svd(R, full_matrices=False)
print("Shared L1 QK template singular values:", s[:16])
R2 = (u[:, :2] * s[:2]) @ vt[:2]
R2 /= np.linalg.norm(R2)
gg = sum(g[1, j] / np.linalg.norm(g[1, j]) for j in range(6))
gg /= np.linalg.norm(gg)
Cprev = C[0, 2]

print("\nPer-head matrix diagnostics:")
print(
    "name: shared-L1-pattern coefficient; coefficient/QK-norm; key-linear projection; induction trace/nuclear norm"
)
for l in range(4):
    for j in range(6):
        a = B[l, j]
        coefficient = np.sum(a * R2)
        t = a @ Cprev
        copy = np.trace(t) / svdvals(t).sum()
        print(f"L{l}H{j}", *np.round([coefficient, coefficient / np.linalg.norm(a), g[l, j] @ gg, copy], 4))

print("\nL0H2 eigenplane comparison with L0H3")
e, z = eig(B[0, 2])
ix = [i for i in range(len(e)) if e[i].imag > 1e-6 and abs(e[i]) > 0.08]
ix = sorted(ix, key=lambda i: np.angle(e[i]))
for i in ix:
    U, _ = np.linalg.qr(np.column_stack([z[:, i].real, z[:, i].imag]))
    m2 = U.T @ B[0, 2] @ U
    m3 = U.T @ B[0, 3] @ U
    print(
        "eigenvalue",
        np.round(e[i], 4),
        "phase",
        round(float(np.angle(e[i])), 4),
        "H2_plane",
        np.round(m2.ravel(), 4),
        "H3_plane",
        np.round(m3.ravel(), 4),
    )

print("\nInduction alignment versus coordinate-permutation controls")
rng = np.random.default_rng(827)
for l, j in [(1, 2), (2, 1), (3, 0), (3, 1), (3, 4), (3, 5)]:
    a = B[l, j]
    denominator = np.linalg.norm(a) * np.linalg.norm(Cprev)
    observed = np.trace(a @ Cprev) / denominator
    controls = []
    for _ in range(200):
        p = rng.permutation(384)
        cp = Cprev[np.ix_(p, p)]
        controls.append(np.sum(a * cp.T) / denominator)
    controls = np.array(controls)
    print(
        f"L{l}H{j}",
        "observed",
        round(observed, 5),
        "control_mean_sd",
        np.round([controls.mean(), controls.std()], 5),
        "control_range",
        np.round([controls.min(), controls.max()], 5),
    )
```
