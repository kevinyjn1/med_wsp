```python
import numpy as np
from scipy.linalg import svd
from scipy.stats import kurtosis, skew
from sklearn.decomposition import FastICA
from threadpoolctl import threadpool_limits
import warnings

threadpool_limits(4)
np.set_printoptions(precision=3, suppress=True, linewidth=160)


def cent(x):
    return x - x.mean(axis=1, keepdims=True)


v1 = W["W_v.0"][64:128].astype(float)
v5 = W["W_v.0"][320:384].astype(float)
rawc = svd(np.concatenate([v1, v5], axis=0), full_matrices=False)[2][:27].T
c = np.linalg.qr(cent(rawc.T * W["ln1_w.0"]).T)[0]
f = cent(W["W_fc.0"].astype(float) * W["ln2_w.0"])
b1 = W["W_o.0"][:, 64:128].astype(float) @ cent(v1 * W["ln1_w.0"])
b5 = W["W_o.0"][:, 320:384].astype(float) @ cent(v5 * W["ln1_w.0"])

print("INDEPENDENT-COMPONENT ANALYSIS OF MLP WEIGHTS ONLY")
print("No inferred directions are passed through normalization, attention, or ReLU.")
x = f @ c
ica = FastICA(n_components=27, whiten="unit-variance", random_state=0, max_iter=3000, tol=1e-5)
with warnings.catch_warnings(record=True) as ww:
    s = ica.fit_transform(x)
    print("Warnings:", [str(w.message) for w in ww])
sign = np.where(skew(s, axis=0) >= 0, 1.0, -1.0)
s *= sign
direction = c @ ica.components_.T * sign
print("Input-component excess kurtoses:", np.round(kurtosis(s, axis=0), 2))
print("Input-component skewness:", np.round(skew(s, axis=0), 2))
print("Component maximum absolute correlation:", np.max(np.abs(np.corrcoef(s.T) - np.eye(27))))

for tag, b in [("H5", b5), ("H1", b1)]:
    y = f @ b @ direction
    y -= y.mean(0)
    y /= np.sqrt(np.mean(y * y, axis=0))
    corr = s.T @ y / len(s)
    print("\n", tag, "same-component correlations:", np.diag(corr))
    inds = np.dstack(np.unravel_index(np.argsort(abs(corr).ravel())[-20:][::-1], corr.shape))[0]
    print(
        "Strongest component pairs (current, transported, correlation):",
        [(int(i), int(j), round(corr[i, j], 3)) for i, j in inds],
    )

print("\nFINAL MLP OUTPUT-WEIGHT ICA")
p = W["W_proj.3"].astype(float).T
ica2 = FastICA(n_components=28, whiten="unit-variance", random_state=0, max_iter=3000, tol=1e-5)
with warnings.catch_warnings(record=True) as ww:
    z = ica2.fit_transform(p)
    print("Warnings:", [str(w.message) for w in ww])
print("Sorted excess kurtoses:", np.sort(kurtosis(z, axis=0))[::-1])
print("Sorted absolute skewness:", np.sort(abs(skew(z, axis=0)))[::-1])
print("Final output ICA mixing-direction norms, sorted:", np.sort(np.linalg.norm(ica2.mixing_, axis=0))[::-1])
```