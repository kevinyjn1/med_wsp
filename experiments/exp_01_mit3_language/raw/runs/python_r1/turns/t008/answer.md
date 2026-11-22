```python
import numpy as np
from scipy.linalg import svd, eig
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=160)


def norm(x):
    return np.linalg.norm(x)


def cent(x):
    return x - x.mean(axis=1, keepdims=True)


def trscore(x):
    return float(np.trace(x) / (norm(x) + 1e-30))


A, B = {}, {}
for l in range(4):
    d = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        sl = slice(64 * j, 64 * (j + 1))
        q, k, v = [cent(W[f"{name}.{l}"][sl].astype(float) * d) for name in ["W_q", "W_k", "W_v"]]
        A[l, j] = q.T @ k / 8
        B[l, j] = W[f"W_o.{l}"][:, sl].astype(float) @ v

vstack = np.concatenate(
    [
        cent(W["W_v.0"][64:128].astype(float) * W["ln1_w.0"]),
        cent(W["W_v.0"][320:384].astype(float) * W["ln1_w.0"]),
    ],
    axis=0,
)
c = svd(vstack, full_matrices=False)[2][:27].T

print("KEY VERSUS QUERY COMPOSITION ON THE SHARED VALUE SUBSPACE")
print("Each entry: earlier head, key trace/norm, query trace/norm, key symmetric positive-energy fraction.")
for l, j in [
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
    (2, 4),
    (2, 5),
    (3, 0),
    (3, 1),
    (3, 2),
    (3, 3),
    (3, 4),
    (3, 5),
]:
    rows = []
    for m, h in [(0, 1), (0, 5), (1, 2), (2, 2)]:
        if m >= l:
            continue
        z = c.T @ A[l, j] @ B[m, h] @ c
        qz = c.T @ B[m, h].T @ A[l, j] @ c
        ev = np.linalg.eigvalsh((z + z.T) / 2)
        positive = np.sum(ev[ev > 0] ** 2) / (np.sum(ev**2) + 1e-30)
        rows.append((f"L{m}H{h}", round(trscore(z), 3), round(trscore(qz), 3), round(positive, 3)))
    print(f"L{l}H{j}:", rows)

print("\nFIRST-LAYER SHARED ROTATIONAL MODES")
p = svd(np.concatenate([A[0, 1], A[0, 1].T, A[0, 5], A[0, 5].T], axis=1), full_matrices=False)[0][:, :26]
a = p.T @ A[0, 5] @ p
ev, z = eig(a.T, a)
modes = []
for k in range(len(ev)):
    v = z[:, k]
    strength = abs(np.vdot(v, a @ v))
    if ev[k].imag > 0.03 and abs(abs(ev[k]) - 1) < 0.02 and strength > 0.01:
        modes.append(v)
print("Mode strengths, head order H0 H1 H2 H3 H4 H5:")
for v in modes:
    vals = np.array([np.vdot(v, (p.T @ A[0, j] @ p) @ v) for j in range(6)])
    print("absolute:", np.round(abs(vals), 4), "phases:", np.round(np.angle(vals), 3))

print("\nLATER QK GEOMETRY AFTER REMOVING SHARED LAYER-ONE KEY-BIAS DIRECTION")
bias = []
for j in range(6):
    sl = slice(64 * j, 64 * (j + 1))
    qr = W["W_q.1"][sl].astype(float)
    k = cent(W["W_k.1"][sl].astype(float) * W["ln1_w.1"])
    b = k.T @ (qr @ W["ln1_b.1"]) / 8
    bias.append(b / norm(b))
g = svd(np.stack(bias, axis=1), full_matrices=False)[0][:, 0]
for l in range(1, 4):
    print("Layer", l, [(j, round(norm(A[l, j] @ g) ** 2 / norm(A[l, j]) ** 2, 3)) for j in range(6)])
```