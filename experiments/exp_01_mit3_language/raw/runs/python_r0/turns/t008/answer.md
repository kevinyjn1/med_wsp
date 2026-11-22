```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=190)


def c(a):
    return a - a.mean(-1, keepdims=True)


def norm(a):
    return np.linalg.norm(a)


def co(a, b):
    return np.sum(a * b) / (norm(a) * norm(b) + 1e-30)


Q = []
K = []
V = []
for j in range(6):
    s = slice(64 * j, 64 * j + 64)
    Q.append(c(W["W_q.0"][s].astype(float)))
    K.append(c(W["W_k.0"][s].astype(float)))
    V.append(c(W["W_v.0"][s].astype(float)))
_, _, vt = svd(np.vstack([V[1], V[5]]), full_matrices=False)
T = vt[:28].T
R = np.eye(384) - T @ T.T
_, ss, vh = svd(np.vstack([Q[1] @ R, K[1] @ R, Q[5] @ R, K[5] @ R]), full_matrices=False)

for rank in (12, 20, 28):
    P = vh[:rank].T
    A = [P.T @ q.T @ k @ P / 8 for q, k in zip(Q, K)]
    u, _, v = svd(A[5])
    shift = u @ v
    print("\nPositional-reference rank", rank)
    for j, a in enumerate(A):
        total = Q[j].T @ K[j] / 8
        print(
            "H",
            j,
            "projected energy fraction",
            round(norm(a) ** 2 / norm(total) ** 2, 4),
            "cosines with shift powers 0..8",
            np.round([co(a, np.linalg.matrix_power(shift, r)) for r in range(9)], 4),
        )
    vals, vecs = np.linalg.eig(A[5])
    ii = [i for i in np.argsort(-abs(vals)) if vals[i].imag > 1e-4][:10]
    print("H5 mode phase, then each head's complex quadratic coefficient magnitude and phase")
    for i in ii:
        v = vecs[:, i]
        zs = [v.conj() @ a @ v for a in A]
        print(round(np.angle(vals[i]), 4), [(round(abs(z), 4), round(np.angle(z), 4)) for z in zs])

print("\nLow-rank L0 structural-head read-direction overlaps")
g = W["ln1_w.0"].astype(float)
refs = {}
for j in (0, 2, 3, 4):
    for name, mat in (("Q", Q[j]), ("K", K[j]), ("V", V[j])):
        _, s, vh0 = svd(c(mat * g), full_matrices=False)
        refs[f"H{j}{name}"] = vh0[:2].T
        print(f"H{j}{name} top2 energy", np.sum(s[:2] ** 2) / np.sum(s**2))
names = list(refs)
print("Reference names", names)
print(np.array([[norm(refs[a].T @ refs[b]) ** 2 / 2 for b in names] for a in names]))

# Check direct-versus-reversed composition with the candidate L2 previous head.
Ms = []
OV = []
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    for j in range(6):
        s = slice(64 * j, 64 * j + 64)
        Ms.append(c(W[f"W_q.{l}"][s].astype(float) * g).T @ c(W[f"W_k.{l}"][s].astype(float) * g) / 8)
        OV.append(W[f"W_o.{l}"][:, s].astype(float) @ c(W[f"W_v.{l}"][s].astype(float) * g))
print("\nL3 QK composition through L2H2: key vs query signed nuclear ratios")
for j in range(6):
    m = Ms[18 + j]
    aa = m @ OV[14]
    bb = OV[14].T @ m
    print(
        j,
        np.trace(aa) / svdvals(aa).sum(),
        np.trace(bb) / svdvals(bb).sum(),
        "normalized traces",
        np.trace(aa) / (norm(m) * norm(OV[14])),
        np.trace(bb) / (norm(m) * norm(OV[14])),
    )
```
