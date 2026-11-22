```python
import numpy as np
from scipy.linalg import svd, svdvals
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=160)


def norm(a):
    return np.linalg.norm(a)


def cent(a):
    return a - a.mean(axis=1, keepdims=True)


d = W["ln1_w.0"].astype(float)
qs, ks, vs, os, aa = [], [], [], [], []
for j in range(6):
    sl = slice(64 * j, 64 * (j + 1))
    q, k, v = [cent(W[f"{name}.0"][sl].astype(float) * d) for name in ["W_q", "W_k", "W_v"]]
    o = W["W_o.0"][:, sl].astype(float)
    qs.append(q)
    ks.append(k)
    vs.append(v)
    os.append(o)
    aa.append(q.T @ k / 8)

sources = {}
for j in range(6):
    sources[f"V{j}"] = svd(vs[j], full_matrices=False)[2][:24].T
    sources[f"O{j}"] = svd(os[j], full_matrices=False)[0][:, :24]
fc = cent(W["W_fc.0"].astype(float) * W["ln2_w.0"])
sources["FC_top28"] = svd(fc, full_matrices=False)[2][:28].T
sources["FC_top64"] = svd(fc, full_matrices=False)[2][:64].T

print("FIRST-LAYER QK SUBSPACE OVERLAPS")
for j in range(6):
    u, s, vt = svd(aa[j], full_matrices=False)
    r = 24 if j in [1, 5] else 4
    left, right = u[:, :r], vt[:r].T
    print(
        f"H{j} top{r} (left,right):",
        {
            name: (round(norm(left.T @ z) ** 2 / r, 3), round(norm(right.T @ z) ** 2 / r, 3))
            for name, z in sources.items()
        },
    )

print("\nDOMINANT FIRST-LAYER QK EIGENVALUES")
for j in range(6):
    ev = np.linalg.eigvals(aa[j])
    ev = ev[np.argsort(-abs(ev))]
    print(f"H{j}:", ev[:24])

print("\nLAYER-ONE PAIRWISE QK AND KEY-BIAS COSINES")
a1, bias1 = [], []
d, b = W["ln1_w.1"].astype(float), W["ln1_b.1"].astype(float)
for j in range(6):
    sl = slice(j * 64, (j + 1) * 64)
    qr, kr = [W[f"{name}.1"][sl].astype(float) for name in ["W_q", "W_k"]]
    q, k = cent(qr * d), cent(kr * d)
    a1.append(q.T @ k / 8)
    bias1.append(k.T @ (qr @ b) / 8)
print("QK:", np.array([[np.sum(x * y) / (norm(x) * norm(y)) for y in a1] for x in a1]))
print("Key bias:", np.array([[x @ y / (norm(x) * norm(y)) for y in bias1] for x in bias1]))
print("Key-bias overlaps with earlier OV left top3:")
for j, c in enumerate(bias1):
    print(
        "L1H" + str(j),
        [
            (h, round(norm(svd(os[h] @ vs[h], full_matrices=False)[0][:, :3].T @ c) ** 2 / norm(c) ** 2, 3))
            for h in range(6)
        ],
    )

print("\nMLP OUTPUT TOP35 SINGULAR VALUES")
for l in range(4):
    print(f"W_proj.{l}:", svdvals(W[f"W_proj.{l}"].astype(float))[:35])
```