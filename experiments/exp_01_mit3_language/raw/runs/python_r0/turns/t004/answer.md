```python
import numpy as np
from scipy.linalg import svd, svdvals, eigh
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=4, suppress=True, linewidth=200)


def c(a):
    return a - a.mean(-1, keepdims=True)


def norm(a):
    return np.linalg.norm(a)


def cos(a, b):
    return np.sum(a * b) / (norm(a) * norm(b) + 1e-30)


# Unscaled L0 H1/H5 V row-space is used only as a weight-derived reference space.
_, _, vh = svd(
    np.vstack([c(W["W_v.0"][64 * j : 64 * j + 64].astype(float)) for j in (1, 5)]), full_matrices=False
)
T = vh[:28].T
R = np.eye(384) - T @ T.T
Q0, K0 = [], []
for j in range(6):
    s = slice(64 * j, 64 * j + 64)
    Q0.append(c(W["W_q.0"][s].astype(float)))
    K0.append(c(W["W_k.0"][s].astype(float)))

_, ss, vh = svd(np.vstack([Q0[1] @ R, K0[1] @ R, Q0[5] @ R, K0[5] @ R]), full_matrices=False)
print("Candidate positional joint spectrum:", ss[:70])
for rank in (20, 28, 36, 48):
    B = vh[:rank].T
    a = B.T @ Q0[5].T @ K0[5] @ B / 8
    b = B.T @ Q0[1].T @ K0[1] @ B / 8
    vals, vecs = np.linalg.eig(a)
    ii = [i for i in np.argsort(-np.abs(vals)) if vals[i].imag > 0.0001]
    print("\nPosition candidate rank", rank)
    pairs = []
    for i in ii[:14]:
        v = vecs[:, i]
        z5 = v.conj() @ a @ v
        z1 = v.conj() @ b @ v
        pairs.append((abs(vals[i]), np.angle(z5), np.angle(z1), abs(z1)))
    print("H5 eigenmode: magnitude, H5 phase, H1 phase, H1 magnitude")
    print(np.array(pairs))
    u, _, v = svd(a)
    polar = u @ v
    print(
        "H1 cosine with polar(H5)^r, r=-4..4:",
        [round(cos(b, np.linalg.matrix_power(polar, r)), 4) for r in range(-4, 5)],
    )

# Residual-coordinate reference spaces obtained from weight directions.
g0 = W["ln1_w.0"].astype(float)
_, _, vh_res = svd(
    np.vstack([c(Q0[1] * g0), c(K0[1] * g0), c(Q0[5] * g0), c(K0[5] * g0)]), full_matrices=False
)
Pos = vh_res[:32].T
_, _, tv = svd(
    np.vstack([c(W["W_v.0"][64 * j : 64 * j + 64].astype(float) * g0) for j in (1, 5)]), full_matrices=False
)
Tok = tv[:28].T
print("\nReference Pos/Tok squared overlap:", norm(Pos.T @ Tok) ** 2 / 28)

Ms = []
Ks = []
Bs = []
OVs = []
for l in range(4):
    g = W[f"ln1_w.{l}"].astype(float)
    bias = W[f"ln1_b.{l}"].astype(float)
    for j in range(6):
        s = slice(64 * j, 64 * j + 64)
        q = c(W[f"W_q.{l}"][s].astype(float) * g)
        k = c(W[f"W_k.{l}"][s].astype(float) * g)
        m = q.T @ k / 8
        Ms.append(m)
        Ks.append(k)
        Bs.append(k.T @ (W[f"W_q.{l}"][s].astype(float) @ bias) / 8)
        OVs.append(W[f"W_o.{l}"][:, s].astype(float) @ c(W[f"W_v.{l}"][s].astype(float) * g))

print("\nAll heads: QK position-reference energy; QK direct-token energy; key-bias position fraction")
for i, m in enumerate(Ms):
    print(
        f"L{i // 6}H{i % 6}",
        np.array(
            [
                norm(Pos.T @ m @ Pos) ** 2 / norm(m) ** 2,
                norm(Tok.T @ m @ Tok) ** 2 / norm(m) ** 2,
                norm(Pos.T @ Bs[i]) ** 2 / (norm(Bs[i]) ** 2 + 1e-30),
            ]
        ),
    )
print("\nL1 QK pairwise cosines")
print(np.array([[cos(a, b) for b in Ms[6:12]] for a in Ms[6:12]]))
print("L1 key-bias pairwise cosines")
print(np.array([[cos(a, b) for b in Bs[6:12]] for a in Bs[6:12]]))

print("\nToken-restricted induction composition through L0H1 and L0H5")
print("Fields: source, trace, signed nuclear ratio, diagonal/symmetric energy fraction")
for i in range(6, 24):
    row = []
    for src in (1, 5):
        a = Tok.T @ Ms[i] @ OVs[src] @ Tok
        ns = svdvals(a).sum()
        row.append(
            (
                src,
                round(np.trace(a), 4),
                round(np.trace(a) / (ns + 1e-30), 4),
                round(norm((a + a.T) / 2) ** 2 / (norm(a) ** 2 + 1e-30), 4),
            )
        )
    print(f"L{i // 6}H{i % 6}", row)
```
