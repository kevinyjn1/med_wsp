```python
import numpy as np
from scipy.linalg import svdvals

np.set_printoptions(precision=4, suppress=True, linewidth=180)
d = 384
P = np.eye(d) - np.ones((d, d)) / d
A = {}
B = {}
bias = {}
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    b = W[f"ln1_b.{l}"].astype(float)
    for h in range(6):
        sl = slice(64 * h, 64 * h + 64)
        q = W[f"W_q.{l}"][sl].astype(float)
        k = W[f"W_k.{l}"][sl].astype(float)
        v = W[f"W_v.{l}"][sl].astype(float)
        o = W[f"W_o.{l}"][:, sl].astype(float)
        A[l, h] = P @ (q * g).T @ (k * g) @ P / 8
        B[l, h] = P @ o @ (v * g) @ P
        bias[l, h] = ((q @ b) @ (k * g) @ P) / 8

print("Key-composition alignment: tr(A_lh B_mj)/(||A_lh||F ||B_mj||F)")
for l in range(1, 4):
    for m in range(l):
        tab = np.array(
            [
                [
                    np.trace(A[l, h] @ B[m, j]) / np.linalg.norm(A[l, h]) / np.linalg.norm(B[m, j])
                    for j in range(6)
                ]
                for h in range(6)
            ]
        )
        print("QK layer", l, "OV layer", m)
        print(tab)
print(
    "\nFor each later head, strongest previous-head key composition: trace/nuclear norm and symmetric positive mass"
)
for l in range(1, 4):
    for h in range(6):
        candidates = []
        for m in range(l):
            for j in range(6):
                C = A[l, h] @ B[m, j]
                score = np.trace(C) / svdvals(C).sum()
                candidates.append((score, m, j, C))
        candidates.sort(key=lambda t: -t[0])
        rows = []
        for score, m, j, C in candidates[:3]:
            ev = np.linalg.eigvalsh((C + C.T) / 2)
            rows.append((m, j, round(score, 4), round(np.sum(ev[ev > 0]) / np.sum(np.abs(ev)), 4)))
        print("L%dH%d" % (l, h), rows)

print("\nDirect QK trace/nuclear norm")
for l in range(4):
    print(l, [round(np.trace(A[l, h]) / svdvals(A[l, h]).sum(), 4) for h in range(6)])

print("\nProjection of later key-bias directions onto early OV output singular vectors")
for m in [0, 1]:
    for j in range(6):
        u, s, vt = np.linalg.svd(B[m, j], full_matrices=False)
        print("source L%dH%d" % (m, j), "OV top3", s[:3])
        for l in range(m + 1, 4):
            dirs = np.array([bias[l, h] / np.linalg.norm(bias[l, h]) for h in range(6)])
            print(" target L", l, "cos with first 3 output directions", np.round(dirs @ u[:, :3], 3))
```
