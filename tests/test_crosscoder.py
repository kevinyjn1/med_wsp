"""A crosscoder trained on synthetic paired activations must separate shared from model-exclusive directions."""

from __future__ import annotations

import numpy as np
import torch

from wsp.crosscoder import CrosscoderConfig, analyze, train


def synthetic(n_tokens: int = 6000, d: int = 24, seed: int = 0) -> tuple[np.ndarray, dict]:
    """6 sparse directions: 4 shared by both models, 1 only in the base, 1 only in the tuned model."""
    rng = np.random.default_rng(seed)
    directions = np.linalg.qr(rng.normal(size=(d, d)))[0][:6]
    codes = (rng.random((n_tokens, 6)) < 0.15) * rng.uniform(1.0, 2.0, size=(n_tokens, 6))
    use = np.ones((2, 6))
    use[1, 4] = 0.0  # direction 4: base only
    use[0, 5] = 0.0  # direction 5: tuned only
    activations = np.stack([(codes * use[m]) @ directions for m in range(2)], axis=1)
    activations += 0.01 * rng.normal(size=activations.shape)
    return activations.astype(np.float32), {"base_only": 4, "tuned_only": 5}


def test_crosscoder_finds_shared_and_exclusive_features():
    torch.manual_seed(0)
    activations, _ = synthetic()
    config = CrosscoderConfig(d_model=24, n_features=24, target_l0=1.5, lr=1e-2, batch_size=512, steps=4000)
    model, _, scales = train(config, activations, torch.device("cpu"), log_every=4000)
    result = analyze(model, activations, scales, torch.device("cpu"))
    assert min(result["explained_variance"]) > 0.8, result["explained_variance"]
    alive = result["firing_rate"] > 0.02
    relative = result["relative_norm"][alive]
    assert (relative < 0.15).sum() >= 1, "no base-only feature found"
    assert (relative > 0.85).sum() >= 1, "no tuned-only feature found"
    assert ((relative > 0.35) & (relative < 0.65)).sum() >= 3, "shared features missing"
    shared = alive & (np.abs(result["relative_norm"] - 0.5) < 0.15)
    assert np.median(result["decoder_cosine"][shared]) > 0.9
