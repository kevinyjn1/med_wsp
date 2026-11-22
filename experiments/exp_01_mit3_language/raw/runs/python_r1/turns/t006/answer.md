```python
import numpy as np
from scipy.linalg import svd, svdvals, eigh
from threadpoolctl import threadpool_limits

threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=160)


def norm(x):
    return np.linalg.norm(x)


def cent(x):
    return x - x.mean(axis=1, keepdims=True)


print("RAW L0 VALUE SPECTRA")
for j in [1, 5]:
    v = W["W_v.0"][64 * j : 64 * (j + 1)].astype(float)
    print(f"H{j}:", svdvals(v))

# The common row space of the two character-transport candidates,
# estimated before the strongly anisotropic learned LN scale.
v1 = W["W_v.0"][64:128].astype(float)
v5 = W["W_v.0"][320:384].astype(float)
u1 = svd(v1, full_matrices=False)[2][:27].T
u5 = svd(v5, full_matrices=False)[2][:27].T
print("Raw top27 common-space canonical cosines:", svdvals(u1.T @ u5))
_, s, vt = svd(np.concatenate([v1, v5], axis=0), full_matrices=False)
print("Combined raw V spectrum:", s[:48])
rawc = vt[:27].T
c = cent((rawc.T * W["ln1_w.0"]).astype(float)).T
c = np.linalg.qr(c)[0]

print("\nMLP WEIGHT GEOMETRY IN SHARED CHARACTER-TRANSPORT SPACE")
f = cent(W["W_fc.0"].astype(float) * W["ln2_w.0"])
o1 = W["W_o.0"][:, 64:128].astype(float)
o5 = W["W_o.0"][:, 320:384].astype(float)
b1 = o1 @ cent(v1 * W["ln1_w.0"])
b5 = o5 @ cent(v5 * W["ln1_w.0"])
blocks = [f @ c, f @ b5 @ c, f @ b1 @ c]
for i, z in enumerate(blocks):
    print("Block", i, "spectrum", svdvals(z)[:27])

print("\nCURRENT/PREVIOUS CHARACTER-DIRECTION COUPLINGS")
for i in [1, 2]:
    x, y = blocks[0], blocks[i]
    x = x - x.mean(0)
    y = y - y.mean(0)
    cross = x.T @ y
    ev = np.linalg.eigvals(cross)
    ev = ev[np.argsort(-abs(ev))]
    print("Block", i, "cross norm", norm(cross), "trace/norm", np.trace(cross) / norm(cross))
    print("Dominant eigenvalues:", ev[:12])
    ss = svdvals(cross)
    print("Cross singular values:", ss[:12])
    corr = np.sum(x * y, axis=1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-20)
    print("Per-neuron same-direction cosine quantiles:", np.quantile(corr, [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1]))

print("\nSTRONGEST SHARED CHARACTER DIRECTIONS BY L0 MLP")
u, s, vt = svd(blocks[0], full_matrices=False)
for k in range(8):
    direction = c @ vt[k]
    print(
        k,
        "s",
        round(s[k], 3),
        "H1-self",
        round(direction @ b1 @ direction, 3),
        "H5-self",
        round(direction @ b5 @ direction, 3),
        "H1-cross",
        round((f @ direction) @ (f @ b1 @ direction), 3),
        "H5-cross",
        round((f @ direction) @ (f @ b5 @ direction), 3),
    )
```