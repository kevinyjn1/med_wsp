```python
import numpy as np
from scipy.linalg import svd, eig
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=160)


def cent(a):
    return a - a.mean(axis=1, keepdims=True)


def norm(a):
    return np.linalg.norm(a)


d = W["ln1_w.0"].astype(float)
A = []
for j in range(6):
    sl = slice(64 * j, 64 * (j + 1))
    q, k = [cent(W[f"{name}.0"][sl].astype(float) * d) for name in ["W_q", "W_k"]]
    A.append(q.T @ k / 8)
u, s, vt = svd(np.concatenate([A[1], A[1].T, A[5], A[5].T], axis=1), full_matrices=False)

print("GENERALIZED SPECTRAL COMPARISON OF L0H1 AND L0H5")
print("All vectors below are generalized eigenvectors of weight products, not model inputs.")
for r in [18, 20, 22, 24, 26, 28, 32]:
    p = u[:, :r]
    a = p.T @ A[5] @ p
    b = p.T @ A[1] @ p
    ev, z = eig(a.T, a)
    rows = []
    for k in range(r):
        if ev[k].imag <= 0.03 or not (0.7 < abs(ev[k]) < 1.3):
            continue
        v = z[:, k]
        x = np.vdot(v, a @ v)
        y = np.vdot(v, b @ v)
        rows.append((np.angle(x), np.angle(y), abs(x), abs(y), abs(ev[k])))
    if not rows:
        continue
    rows = np.array(rows)
    weight = np.sqrt(rows[:, 2] * rows[:, 3])
    print("\nRank", r, "modes", len(rows))
    print("phase(H5), phase(H1), strength(H5), strength(H1), eigenvalue modulus:")
    print(rows)
    scores = []
    for multiple in range(-6, 7):
        score = np.sum(weight * np.cos(rows[:, 1] - multiple * rows[:, 0])) / weight.sum()
        scores.append((round(float(score), 4), multiple))
    print("Best phase-multiple fits:", sorted(scores, reverse=True)[:5])

print("\nFIRST-LAYER QK ENERGY AFTER COMMON SUBSPACE REMOVAL")
p = u[:, :26]
for j, a in enumerate(A):
    residual = a - p @ (p.T @ a) - (a @ p) @ p.T + p @ (p.T @ a @ p) @ p.T
    print(
        f"H{j}: residual norm={norm(residual):.4f}, original={norm(a):.4f}, "
        f"residual top5={svd(residual, compute_uv=False)[:5]}"
    )
```