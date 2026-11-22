"""Sparse crosscoder for model diffing (Lindsey et al. 2024): one dictionary over the same layer of two models.

    f(x)   = ReLU( sum_m W_enc[m] a_m(x) + b_enc )            m in {base, tuned}
    a_m'   = W_dec[m] f(x) + b_dec[m]
    loss   = sum_m ||a_m - a_m'||^2 + lambda * sum_i f_i * ( ||W_dec[base, i]|| + ||W_dec[tuned, i]|| )

A feature that only one model uses ends up with a decoder vector in that model only, so the relative decoder
norm  r_i = ||W_dec[tuned, i]|| / (||W_dec[base, i]|| + ||W_dec[tuned, i]||)  sorts features into
base-only (r near 0), shared (r near 0.5) and tuned-only (r near 1).

The source gives no L1 coefficient, so lambda is steered during training toward a target number of active
features per token (our choice).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class CrosscoderConfig:
    d_model: int
    n_features: int = 16_384
    n_models: int = 2
    target_l0: float = 80.0
    lr: float = 2e-4
    batch_size: int = 2048
    steps: int = 8000
    seed: int = 0


class Crosscoder(nn.Module):
    def __init__(self, config: CrosscoderConfig):
        super().__init__()
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        decoder = torch.randn(config.n_features, config.n_models, config.d_model, generator=generator)
        decoder = 0.1 * decoder / decoder.norm(dim=-1, keepdim=True)
        self.W_dec = nn.Parameter(decoder)
        self.W_enc = nn.Parameter(decoder.permute(1, 2, 0).clone())  # [models, d, features], tied at init
        self.b_enc = nn.Parameter(torch.zeros(config.n_features))
        self.b_dec = nn.Parameter(torch.zeros(config.n_models, config.d_model))

    def encode(self, x: torch.Tensor) -> torch.Tensor:  # x: [batch, models, d]
        return torch.relu(torch.einsum("bmd,mdf->bf", x, self.W_enc) + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bf,fmd->bmd", f, self.W_dec) + self.b_dec

    def decoder_norms(self) -> torch.Tensor:  # [features, models]
        return self.W_dec.norm(dim=-1)

    def loss(self, x: torch.Tensor, l1: float) -> tuple[torch.Tensor, dict]:
        f = self.encode(x)
        reconstruction = self.decode(f)
        mse = (x - reconstruction).pow(2).sum(dim=(1, 2)).mean()
        sparsity = (f * self.decoder_norms().sum(dim=1)).sum(dim=1).mean()
        active = (f > 0).float().sum(dim=1).mean()
        stats = {"mse": mse.item(), "sparsity": sparsity.item(), "l0": active.item()}
        return mse + l1 * sparsity, stats


def train(
    config: CrosscoderConfig, activations: np.ndarray, device: torch.device, log_every: int = 500
) -> tuple[Crosscoder, list[dict], np.ndarray]:
    """activations: [tokens, models, d], any float dtype (a memory map is fine). Returns model, log and per-model scales."""
    rng = np.random.default_rng(config.seed)
    sample = np.asarray(
        activations[rng.choice(len(activations), size=min(20_000, len(activations)), replace=False)],
        np.float32,
    )
    # Scale each model's space so its mean squared norm equals d_model: both spaces weigh the same in the loss.
    scales = np.sqrt(config.d_model / (sample**2).sum(axis=-1).mean(axis=0))
    model = Crosscoder(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999))
    scale_t = torch.tensor(scales, dtype=torch.float32, device=device)[None, :, None]
    l1, log = 0.02, []
    for step in range(1, config.steps + 1):
        rows = np.sort(rng.choice(len(activations), size=config.batch_size, replace=False))
        x = torch.from_numpy(np.asarray(activations[rows], np.float32)).to(device) * scale_t
        total, stats = model.loss(x, l1 * min(1.0, step / (0.05 * config.steps)))  # sparsity warm-up
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        # Steer lambda toward the target number of active features (multiplicative, slow).
        l1 *= float(np.exp(0.01 * np.clip(np.log(max(stats["l0"], 1e-3) / config.target_l0), -1, 1)))
        if step % log_every == 0 or step == config.steps:
            variance = float(x.pow(2).sum(dim=(1, 2)).mean())
            log.append({"step": step, "l1": l1, "explained": 1.0 - stats["mse"] / variance, **stats})
            print(log[-1], flush=True)
    return model, log, scales


@torch.no_grad()
def analyze(
    model: Crosscoder, activations: np.ndarray, scales: np.ndarray, device: torch.device, chunk: int = 8192
) -> dict:
    """Relative decoder norms, decoder cosines, firing rates and the strongest tokens of every feature."""
    norms = model.decoder_norms().cpu().numpy()
    relative = norms[:, 1] / np.maximum(norms.sum(axis=1), 1e-12)
    decoder = model.W_dec.detach()
    cosine = torch.nn.functional.cosine_similarity(decoder[:, 0], decoder[:, 1], dim=-1).cpu().numpy()
    n_features = model.config.n_features
    fired = np.zeros(n_features, np.int64)
    top_values = torch.zeros(8, n_features, device=device)
    top_rows = torch.zeros(8, n_features, dtype=torch.long, device=device)
    scale_t = torch.tensor(scales, dtype=torch.float32, device=device)[None, :, None]
    explained_num, explained_den = np.zeros(2), np.zeros(2)
    for begin in range(0, len(activations), chunk):
        x = torch.from_numpy(np.asarray(activations[begin : begin + chunk], np.float32)).to(device) * scale_t
        f = model.encode(x)
        reconstruction = model.decode(f)
        explained_num += (x - reconstruction).pow(2).sum(dim=(0, 2)).cpu().numpy()
        explained_den += (x - x.mean(dim=0, keepdim=True)).pow(2).sum(dim=(0, 2)).cpu().numpy()
        fired += (f > 0).sum(dim=0).cpu().numpy()
        values = torch.cat([top_values, f], dim=0)
        rows = torch.cat(
            [top_rows, torch.arange(begin, begin + len(f), device=device)[:, None].expand(-1, n_features)],
            dim=0,
        )
        best = values.topk(8, dim=0)
        top_values, top_rows = best.values, rows.gather(0, best.indices)
    return {
        "decoder_norms": norms, "relative_norm": relative, "decoder_cosine": cosine, "firing_rate": fired / len(activations),
        "top_values": top_values.T.cpu().numpy(), "top_rows": top_rows.T.cpu().numpy(),
        "explained_variance": (1.0 - explained_num / explained_den).tolist(),
    }  # fmt: skip
