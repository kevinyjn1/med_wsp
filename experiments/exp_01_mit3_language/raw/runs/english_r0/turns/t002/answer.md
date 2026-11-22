```python
import numpy as np
from scipy.linalg import svd, svdvals, eigvals
from threadpoolctl import threadpool_limits

np.set_printoptions(precision=4, suppress=True, linewidth=180)


def center_columns(a):
    return a - a.mean(axis=1, keepdims=True)


with threadpool_limits(limits=2):
    A, B = {}, {}
    for l in range(4):
        g = W[f"ln1_w.{l}"].astype(float)
        b = W[f"ln1_b.{l}"].astype(float)
        for j in range(6):
            sl = slice(64 * j, 64 * (j + 1))
            q = W[f"W_q.{l}"][sl].astype(float)
            k = W[f"W_k.{l}"][sl].astype(float)
            v = W[f"W_v.{l}"][sl].astype(float)
            o = W[f"W_o.{l}"][:, sl].astype(float)
            qg, kg, vg = [center_columns(z * g) for z in (q, k, v)]
            a, ov = qg.T @ kg / 8, o @ vg
            A[l, j], B[l, j] = a, ov
            s = svdvals(a)
            ev = eigvals(a)
            ev = ev[np.argsort(abs(ev))[::-1]][:10]
            key_bias = kg.T @ (q @ b) / 8
            query_bias = qg.T @ (k @ b) / 8
            print(
                f"L{l}H{j}: scaled_QK fro={np.linalg.norm(a):.4f} top={np.round(s[:6], 4)} "
                f"trace={np.trace(a):.4f} key_bias_norm={np.linalg.norm(key_bias):.4f} query_bias_norm={np.linalg.norm(query_bias):.4f}"
            )
            print("  QK eigenvalues:", np.round(ev, 4))

    print("\nCross-layer product alignment: tr(QK_later OV_earlier)/(||QK||_F ||OV||_F)")
    print("For each later head, the 5 largest positive alignments and the largest negative alignment.")
    for (l, j), a in A.items():
        if l == 0:
            continue
        scores = []
        for (m, h), ov in B.items():
            if m < l:
                score = np.trace(a @ ov) / (np.linalg.norm(a) * np.linalg.norm(ov))
                scores.append((float(score), f"L{m}H{h}"))
        scores.sort(reverse=True)
        print(f"L{l}H{j}: positive={[(n, round(s, 4)) for s, n in scores[:5]]}; minimum={scores[-1]}")

    print("\nFirst-layer singular-value profiles")
    for j in range(6):
        sl = slice(64 * j, 64 * (j + 1))
        for name in ("W_q", "W_k", "W_v"):
            z = W[f"{name}.0"][sl].astype(float)
            s = svdvals(center_columns(z * W["ln1_w.0"]))
            print(f"L0H{j} {name}: {np.round(s[:40], 3)}")

    print("\nFirst-layer subspace overlap")
    q = center_columns(W["W_q.0"][128:192].astype(float) * W["ln1_w.0"])
    k = center_columns(W["W_k.0"][128:192].astype(float) * W["ln1_w.0"])
    _, s, vt = svd(np.concatenate([q, k]), full_matrices=False)
    p = vt[:32].T
    print("Candidate positional joint Q/K spectrum:", np.round(s[:50], 3))
    for l in range(4):
        for j in range(6):
            a = A[l, j]
            total = np.linalg.norm(a) ** 2
            pp = np.linalg.norm(p.T @ a @ p) ** 2 / total
            left = np.linalg.norm(p.T @ a) ** 2 / total
            right = np.linalg.norm(a @ p) ** 2 / total
            print(f"L{l}H{j}: candidate_position_energy QQ={left:.3f} KK={right:.3f} QK={pp:.3f}")
```
