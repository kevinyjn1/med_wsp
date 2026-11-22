```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True)


def norm(a):
    return np.linalg.norm(a)


def centered_columns(a):
    return a - a.mean(axis=1, keepdims=True)


def erank(s):
    p = s * s / np.sum(s * s)
    return np.exp(-np.sum(p * np.log(p + 1e-30)))


A, B, Q, K, V, O = {}, {}, {}, {}, {}, {}
print("AFFINE WEIGHT GEOMETRY")
print("D is the learned ln1 scale; C is the fixed mean-removal projector.")
print("A = C D Q.T K D C / 8; B = O V D C.")
for l in range(4):
    d = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        tag = f"L{l}H{j}"
        s = slice(j * 64, (j + 1) * 64)
        qr, kr = W[f"W_q.{l}"][s].astype(float), W[f"W_k.{l}"][s].astype(float)
        vr, o = W[f"W_v.{l}"][s].astype(float), W[f"W_o.{l}"][:, s].astype(float)
        q, k, v = [centered_columns(t * d) for t in [qr, kr, vr]]
        a, z = q.T @ k / 8, o @ v
        A[tag], B[tag], Q[tag], K[tag], V[tag], O[tag] = a, z, q, k, v, o
        sa = svdvals(a)[:64]
        key_linear = k.T @ (qr @ b) / 8
        print(
            f"{tag}: |A|={norm(a):.4f} top4={sa[:4]} erank={erank(sa):.1f} "
            f"symcos={np.sum(a * a.T) / norm(a) ** 2:+.3f} "
            f"trace/|A|={np.trace(a) / norm(a):+.3f} "
            f"|key_bias|={norm(key_linear):.4f} |B|={norm(z):.3f}"
        )

print("\nLAYER 0 SPECTRA (learned scales included)")
for name in ["W_q", "W_k", "W_v", "W_fc"]:
    ln = "ln2_w" if name == "W_fc" else "ln1_w"
    a = centered_columns(W[f"{name}.0"].astype(float) * W[f"{ln}.0"])
    s = svdvals(a)
    print(name, "top 45:", np.round(s[:45], 3), "tail:", np.round(s[[63, 95, 127, 191, 255, 383]], 3))

print("\nLAYER 0 QK PAIR COSINES")
tags = [f"L0H{j}" for j in range(6)]
print(np.array([[np.sum(A[t] * A[u]) / (norm(A[t]) * norm(A[u])) for u in tags] for t in tags]))

print("\nKEY-COMPOSITION WITH EARLIER HEADS")
print("For each later head, rank earlier OV maps by positive trace of A_later @ B_earlier.")
print("Reported: normalized trace, product norm, Frobenius alignment A_later with B_earlier.T.")
for l in range(1, 4):
    for j in range(6):
        t = f"L{l}H{j}"
        rows = []
        for m in range(l):
            for k in range(6):
                u = f"L{m}H{k}"
                p = A[t] @ B[u]
                rows.append(
                    (
                        np.trace(p) / (norm(p) + 1e-30),
                        u,
                        norm(p),
                        np.sum(A[t] * B[u].T) / (norm(A[t]) * norm(B[u])),
                    )
                )
        rows.sort(reverse=True)
        print(t, [(u, round(r, 3), round(n, 3), round(c, 3)) for r, u, n, c in rows[:5]])

print("\nLAYER 0 VALUE SUBSPACE OVERLAPS")
for t in tags:
    uq = svd(Q[t], full_matrices=False)[2][:16].T
    uk = svd(K[t], full_matrices=False)[2][:16].T
    print(t, "top16 Q versus K overlap", round(norm(uq.T @ uk) ** 2 / 16, 3))
```