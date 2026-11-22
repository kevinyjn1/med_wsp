"""EXP-05a: where do MediPhi's weights differ from Phi-3.5-mini-instruct?

For every tensor: norm of the difference relative to the base tensor, share of entries that changed,
and the singular-value spectrum of the difference (is it low-rank?). For attention the difference is
also split per head (Q, K, V rows and O columns); for the MLP per neuron (gate, up rows and down columns).

Outputs (processed/):
  weight_diff_summary.json   per-tensor scalars and the top singular values
  weight_diff_maps.npz       per-head maps (layers x heads) and per-neuron maps (layers x neurons)

Usage:  python experiments/exp_05_medical_circuit/weight_diff.py [--base phi] [--tuned mediphi]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp.checkpoints import Checkpoint

OUT = Path(__file__).resolve().parent / "processed"
TOP_SINGULAR = 64


def singular_values(delta: np.ndarray) -> np.ndarray:
    """All singular values, via the eigenvalues of the smaller Gram matrix (fast for wide/tall matrices)."""
    d = delta.astype(np.float64)
    gram = d.T @ d if d.shape[0] >= d.shape[1] else d @ d.T
    return np.sqrt(np.clip(np.linalg.eigvalsh(gram)[::-1], 0.0, None))


def spectrum_stats(s: np.ndarray) -> dict:
    energy = s**2
    total = float(energy.sum())
    if total == 0.0:
        return {"stable_rank": 0.0, "participation_rank": 0.0, "rank90": 0, "rank99": 0, "top": []}
    cumulative = np.cumsum(energy) / total
    return {
        "stable_rank": float(total / energy[0]),
        "participation_rank": float(total**2 / (energy**2).sum()),
        "rank90": int(np.searchsorted(cumulative, 0.90) + 1),
        "rank99": int(np.searchsorted(cumulative, 0.99) + 1),
        "top": [round(float(x), 6) for x in s[:TOP_SINGULAR]],
    }


def rel(delta_norm: np.ndarray, base_norm: np.ndarray) -> np.ndarray:
    return delta_norm / np.maximum(base_norm, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="phi")
    parser.add_argument("--tuned", default="mediphi")
    args = parser.parse_args()

    base, tuned = Checkpoint(args.base), Checkpoint(args.tuned)
    assert base.names() == tuned.names(), "checkpoints do not have the same tensors"
    config = json.loads((base.root / "config.json").read_text())
    n_layers, n_heads, d_model = (
        config["num_hidden_layers"],
        config["num_attention_heads"],
        config["hidden_size"],
    )
    d_mlp, head_dim = config["intermediate_size"], config["hidden_size"] // config["num_attention_heads"]

    head_maps = {part: np.zeros((n_layers, n_heads), np.float32) for part in ("q", "k", "v", "o")}
    neuron_maps = {part: np.zeros((n_layers, d_mlp), np.float32) for part in ("gate", "up", "down")}
    summary, start = [], time.time()

    for key in base.names():
        w, w2 = base.f32(key), tuned.f32(key)
        delta = w2 - w
        record = {
            "name": key, "shape": list(w.shape), "base_norm": float(np.linalg.norm(w)),
            "delta_norm": float(np.linalg.norm(delta)), "frac_changed": float((delta != 0).mean()),
            "max_abs_delta": float(np.abs(delta).max()),
        }  # fmt: skip
        record["rel_delta"] = record["delta_norm"] / max(record["base_norm"], 1e-12)
        match = re.match(r"model\.layers\.(\d+)\.(.+)\.weight", key)
        if match:
            record["layer"], record["component"] = int(match.group(1)), match.group(2)
        if delta.ndim == 2:
            record.update(spectrum_stats(singular_values(delta)))
        if match and delta.ndim == 2:
            layer, component = record["layer"], record["component"]
            if component == "self_attn.qkv_proj":  # rows: [Q | K | V], each split into heads
                for index, part in enumerate(("q", "k", "v")):
                    rows = slice(index * d_model, (index + 1) * d_model)
                    d_heads = delta[rows].reshape(n_heads, head_dim * d_model)
                    w_heads = w[rows].reshape(n_heads, head_dim * d_model)
                    head_maps[part][layer] = rel(
                        np.linalg.norm(d_heads, axis=1), np.linalg.norm(w_heads, axis=1)
                    )
            elif component == "self_attn.o_proj":  # columns are grouped by head
                d_heads = delta.reshape(d_model, n_heads, head_dim).transpose(1, 0, 2).reshape(n_heads, -1)
                w_heads = w.reshape(d_model, n_heads, head_dim).transpose(1, 0, 2).reshape(n_heads, -1)
                head_maps["o"][layer] = rel(np.linalg.norm(d_heads, axis=1), np.linalg.norm(w_heads, axis=1))
            elif component == "mlp.gate_up_proj":  # rows: [gate | up], one row per neuron
                for index, part in enumerate(("gate", "up")):
                    rows = slice(index * d_mlp, (index + 1) * d_mlp)
                    neuron_maps[part][layer] = rel(
                        np.linalg.norm(delta[rows], axis=1), np.linalg.norm(w[rows], axis=1)
                    )
            elif component == "mlp.down_proj":  # one column per neuron
                neuron_maps["down"][layer] = rel(np.linalg.norm(delta, axis=0), np.linalg.norm(w, axis=0))
        summary.append(record)
        print(f"{key}: rel {record['rel_delta']:.4f} changed {record['frac_changed']:.3f} "
              f"stable_rank {record.get('stable_rank', 0):.1f} [{time.time() - start:.0f}s]", flush=True)  # fmt: skip

    OUT.mkdir(parents=True, exist_ok=True)
    meta = {"base": args.base, "tuned": args.tuned, "n_layers": n_layers, "n_heads": n_heads,
            "d_model": d_model, "d_mlp": d_mlp, "head_dim": head_dim}  # fmt: skip
    (OUT / "weight_diff_summary.json").write_text(json.dumps({"meta": meta, "tensors": summary}, indent=1))
    np.savez_compressed(OUT / "weight_diff_maps.npz", **{f"head_{k}": v for k, v in head_maps.items()},
                        **{f"neuron_{k}": v for k, v in neuron_maps.items()})  # fmt: skip
    print("saved", OUT)


if __name__ == "__main__":
    main()
