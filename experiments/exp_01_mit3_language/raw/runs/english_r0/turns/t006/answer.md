```python
import numpy as np
from scipy.linalg import svd, svdvals, eigvalsh
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=180)
with threadpool_limits(limits=2):
    A, B, Kbias = {}, {}, {}
    for l in range(4):
        g = W[f"ln1_w.{l}"].astype(float)
        bias = W[f"ln1_b.{l}"].astype(float)
        for j in range(6):
            sl = slice(64 * j, 64 * (j + 1))
            qr = W[f"W_q.{l}"][sl].astype(float)
            kr = W[f"W_k.{l}"][sl].astype(float)
            q, k, v = [z * g for z in (qr, kr, W[f"W_v.{l}"][sl].astype(float))]
            q -= q.mean(1, keepdims=True)
            k -= k.mean(1, keepdims=True)
            v -= v.mean(1, keepdims=True)
            A[l, j] = q.T @ k / 8
            B[l, j] = W[f"W_o.{l}"][:, sl].astype(float) @ v
            Kbias[l, j] = k.T @ (qr @ bias) / 8

    u, s, vt = svd(np.concatenate([B[0, j] for j in (0, 1, 4, 5)], axis=1), full_matrices=False)
    p = u[:, :4]
    template = sum(Kbias[1, j] / np.linalg.norm(Kbias[1, j]) for j in range(6))
    template /= np.linalg.norm(template)
    print("Key-selection bias geometry")
    print("shared-span fraction uses top4 output directions of L0H0/H1/H4/H5")
    for l in range(4):
        for j in range(6):
            b = Kbias[l, j]
            n = np.linalg.norm(b)
            frac = np.linalg.norm(p.T @ b) ** 2 / n**2
            print(
                f"L{l}H{j}: norm={n:.4f}, shared_span_fraction={frac:.3f}, cosine_to_L1_consensus={b @ template / n:.3f}"
            )

    _, s, vt = svd(W["W_v.0"][128:192].astype(float), full_matrices=False)
    t = vt[:24].T
    print("\nComposition restricted to top24 input directions of L0H2 V")
    for l in range(1, 4):
        for j in range(6):
            c = t.T @ A[l, j] @ B[0, 2] @ t
            e = eigvalsh((c + c.T) / 2)
            positive = np.sum(e[e > 0] ** 2) / np.sum(e**2)
            tr = np.trace(c) / (np.sqrt(24) * np.linalg.norm(c))
            print(
                f"L{l}H{j}: normalized_trace={tr:.4f}, positive_symmetric_energy={positive:.4f}, symmetric_eigenvalue_range=({e[0]:.4f},{e[-1]:.4f})"
            )

    print("\nCoordinate-permutation null for L3H1/L0H2 composition alignment")
    a = A[3, 1]
    b = B[0, 2]
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    observed = np.sum(a * b.T) / denom
    rng = np.random.default_rng(7)
    null = []
    for _ in range(1000):
        perm = rng.permutation(384)
        bp = b[np.ix_(perm, perm)]
        null.append(np.sum(a * bp.T) / denom)
    null = np.array(null)
    print(
        f"observed={observed:.5f}, null_mean={null.mean():.5f}, null_sd={null.std():.5f}, null_range=({null.min():.5f},{null.max():.5f})"
    )

    print("\nLast MLP output spectrum energy")
    s = svdvals(W["W_proj.3"].astype(float))
    for r in (16, 20, 23, 24, 26, 27, 28, 29, 32, 40):
        print(
            f"rank {r}: cumulative_squared_singular_value_fraction={np.sum(s[:r] ** 2) / np.sum(s**2):.4f}, next_singular_value={s[r]:.4f}"
        )
```
