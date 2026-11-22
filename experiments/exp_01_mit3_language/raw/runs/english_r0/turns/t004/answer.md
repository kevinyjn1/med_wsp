```python
import numpy as np
from scipy.linalg import svd, svdvals, eigvalsh
from scipy.stats import skew, kurtosis
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
import warnings

np.set_printoptions(precision=3, suppress=True, linewidth=180)
warnings.filterwarnings("ignore", category=UserWarning)

with threadpool_limits(limits=2):
    A, B = {}, {}
    for l in range(4):
        g = W[f"ln1_w.{l}"].astype(float)
        for j in range(6):
            sl = slice(64 * j, 64 * (j + 1))
            q = W[f"W_q.{l}"][sl].astype(float) * g
            k = W[f"W_k.{l}"][sl].astype(float) * g
            v = W[f"W_v.{l}"][sl].astype(float) * g
            q -= q.mean(1, keepdims=True)
            k -= k.mean(1, keepdims=True)
            v -= v.mean(1, keepdims=True)
            A[l, j] = q.T @ k / 8
            B[l, j] = W[f"W_o.{l}"][:, sl].astype(float) @ v

    print("Within-layer QK Frobenius cosines")
    for l in range(4):
        z = np.stack([A[l, j].ravel() / np.linalg.norm(A[l, j]) for j in range(6)])
        print("Layer", l, "\n", z @ z.T)

    print("\nFirst-layer OV products: singular values and output-subspace overlaps")
    bases = {}
    for j in range(6):
        u, s, vt = svd(B[0, j], full_matrices=False)
        bases[j] = u[:, :3]
        print(f"L0H{j}: scaled OV singular values {s[:12]}")
    for j in range(6):
        print(
            f"L0H{j} top3 output overlap:",
            [round(np.linalg.norm(bases[j].T @ bases[h]) ** 2 / 3, 3) for h in range(6)],
        )

    print("\nLater QK energy on first-layer broad-head output span")
    u, s, vt = svd(np.concatenate([B[0, j] for j in (0, 1, 4, 5)], axis=1), full_matrices=False)
    print("Joint OV singular values:", s[:20])
    p = u[:, :4]
    for l in range(1, 4):
        for j in range(6):
            a = A[l, j]
            z = np.linalg.norm(a) ** 2
            print(
                f"L{l}H{j}: query={np.linalg.norm(p.T @ a) ** 2 / z:.3f}, key={np.linalg.norm(a @ p) ** 2 / z:.3f}, both={np.linalg.norm(p.T @ a @ p) ** 2 / z:.3f}"
            )

    print("\nDirect previous-head composition diagnostics")
    for l in range(1, 4):
        for j in range(6):
            c = A[l, j] @ B[0, 2]
            sy = (c + c.T) / 2
            e = eigvalsh(sy)
            pos = np.sum(e[e > 0] ** 2) / np.sum(e**2)
            print(
                f"L{l}H{j}: trace={np.trace(c):.3f}, symmetric_positive_energy={pos:.3f}, top_sym_eigenvalues={e[-4:]}"
            )

    print("\nICA of first MLP weight-row coordinates, not activations")
    _, s, vt = svd(W["W_v.0"][128:192].astype(float), full_matrices=False)
    t = vt[:24].T
    f = W["W_fc.0"].astype(float) * W["ln2_w.0"]
    f -= f.mean(1, keepdims=True)
    d = f @ t
    p = f @ B[0, 2] @ t
    d = d / np.linalg.norm(d) * np.sqrt(d.size)
    p = p / np.linalg.norm(p) * np.sqrt(p.size)
    z = np.concatenate([d, p])
    ica = FastICA(n_components=24, random_state=1, whiten="unit-variance", max_iter=3000, tol=0.0005)
    c = ica.fit_transform(z)
    sign = np.where(skew(c, axis=0) >= 0, 1.0, -1.0)
    c *= sign
    print("ICA component excess kurtosis:", np.round(kurtosis(c, axis=0), 2))
    print("ICA component skewness:", np.round(skew(c, axis=0), 2))
    corr = np.corrcoef(c[:1536].T, c[1536:].T)[:24, 24:]
    print("Same-component direct/OV-composed coefficient correlations:", np.round(np.diag(corr), 3))
    print("Largest cross-component coefficient correlations:")
    pairs = [(corr[i, j], i, j) for i in range(24) for j in range(24) if i != j]
    print(sorted(pairs, reverse=True)[:25])
```
