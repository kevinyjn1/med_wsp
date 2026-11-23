"""EXP-05d: a sparse crosscoder over one residual layer of Phi-3.5 and a fine-tune, on MedQA prompts.

Steps: (1) run both models on the same prompts and store the residual stream after `--layer` at every token;
(2) train one crosscoder on the paired activations; (3) report relative decoder norms (base-only, shared,
tuned-only features), decoder cosines, and the tokens on which the most model-specific features fire.

The prompts are MedQA training questions 3000.. (outside the practice pool and outside the test split).

Usage:  python experiments/exp_05_medical_circuit/crosscoder_run.py --donor mediphi --layer 16 --questions 1000
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp import medqa, models
from wsp.crosscoder import CrosscoderConfig, analyze, train

EXP = Path(__file__).resolve().parent


@torch.no_grad()
def collect(
    names: list[str], items: list[dict], layer: int, cache: Path
) -> tuple[np.memmap, np.ndarray, np.ndarray]:
    """Residual stream after `layer` at every token, for each model: memory map [tokens, models, d] (float16)."""
    tokenizer = models.load_tokenizer("phi")
    encoded = [
        tokenizer(medqa.build_prompt(tokenizer, item), add_special_tokens=False)["input_ids"]
        for item in items
    ]
    offsets = np.concatenate([[0], np.cumsum([len(e) for e in encoded])])
    token_ids = np.concatenate(encoded).astype(np.int32)
    d_model = 3072
    path = cache / f"resid_L{layer}_{'_'.join(names)}_{len(items)}.f16"
    done = cache / (path.name + ".done")
    store = np.memmap(
        path,
        dtype=np.float16,
        mode="r+" if done.exists() else "w+",
        shape=(int(offsets[-1]), len(names), d_model),
    )
    if not done.exists():
        order = sorted(range(len(items)), key=lambda i: len(encoded[i]))
        for column, name in enumerate(names):
            model = models.load_model(name)
            for begin in range(0, len(order), 8):
                rows = order[begin : begin + 8]
                width = max(len(encoded[i]) for i in rows)
                input_ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
                mask = torch.zeros((len(rows), width), dtype=torch.long)
                for k, i in enumerate(rows):
                    input_ids[k, width - len(encoded[i]) :] = torch.tensor(encoded[i])
                    mask[k, width - len(encoded[i]) :] = 1
                hidden = model(input_ids=input_ids.to(model.device), attention_mask=mask.to(model.device), use_cache=False,
                               output_hidden_states=True).hidden_states[layer]  # fmt: skip
                for k, i in enumerate(rows):
                    store[offsets[i] : offsets[i + 1], column] = (
                        hidden[k, width - len(encoded[i]) :].cpu().numpy()
                    )
                del hidden
                models.release_gpu_cache()
            del model
            gc.collect()
            torch.mps.empty_cache()
            print(f"collected {name}", flush=True)
        store.flush()
        done.write_text("ok")
    return store, token_ids, offsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor", default="mediphi")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--questions", type=int, default=1000)
    parser.add_argument("--features", type=int, default=16_384)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--target-l0", type=float, default=80.0)
    args = parser.parse_args()

    cache = EXP / "cache"
    cache.mkdir(exist_ok=True)
    items = medqa.load_medqa("train")[3000 : 3000 + args.questions]
    activations, token_ids, offsets = collect(["phi", args.donor], items, args.layer, cache)
    print("activations", activations.shape, flush=True)

    # Raw agreement of the two residual streams, before any dictionary.
    sample = np.asarray(activations[:: max(1, len(activations) // 20_000)], np.float32)
    cosine = (sample[:, 0] * sample[:, 1]).sum(-1) / (
        np.linalg.norm(sample[:, 0], axis=-1) * np.linalg.norm(sample[:, 1], axis=-1)
    )
    difference = np.linalg.norm(sample[:, 1] - sample[:, 0], axis=-1) / np.linalg.norm(sample[:, 0], axis=-1)

    device = models.device()
    config = CrosscoderConfig(
        d_model=activations.shape[-1], n_features=args.features, steps=args.steps, target_l0=args.target_l0
    )
    model, log, scales = train(config, activations, device)
    result = analyze(model, activations, scales, device)

    tokenizer = models.load_tokenizer("phi")
    alive = result["firing_rate"] > 1e-5
    relative = result["relative_norm"]

    def describe(feature: int) -> dict:
        examples = []
        for value, row in zip(
            result["top_values"][feature][:5], result["top_rows"][feature][:5], strict=True
        ):
            question = int(np.searchsorted(offsets, row, side="right") - 1)
            lo, hi = max(offsets[question], row - 12), min(offsets[question + 1], row + 3)
            text = (
                tokenizer.decode(token_ids[lo:row])
                + " [["
                + tokenizer.decode(token_ids[row : row + 1])
                + "]] "
                + tokenizer.decode(token_ids[row + 1 : hi])
            )
            examples.append(
                {
                    "activation": round(float(value), 3),
                    "context": text.encode("ascii", "replace").decode()[-160:],
                }
            )
        return {"feature": feature, "relative_norm": round(float(relative[feature]), 4),
                "decoder_cosine": round(float(result["decoder_cosine"][feature]), 4),
                "firing_rate": float(result["firing_rate"][feature]), "examples": examples}  # fmt: skip

    ranked = np.argsort(relative)
    ranked = ranked[alive[ranked]]
    summary = {
        "donor": args.donor, "layer": args.layer, "questions": len(items), "tokens": len(activations),
        "features": args.features, "alive": int(alive.sum()), "explained_variance": result["explained_variance"],
        "final": log[-1],
        "raw_residual": {"mean_cosine_base_vs_tuned": float(cosine.mean()), "mean_relative_difference": float(difference.mean())},
        "counts": {"base_only_r<0.1": int((relative[alive] < 0.1).sum()), "shared_0.4-0.6": int(((relative[alive] > 0.4) & (relative[alive] < 0.6)).sum()),
                   "tuned_only_r>0.9": int((relative[alive] > 0.9).sum()), "other": int(((relative[alive] >= 0.1) & (relative[alive] <= 0.4)).sum() + ((relative[alive] >= 0.6) & (relative[alive] <= 0.9)).sum())},
        "relative_norm_histogram": np.histogram(relative[alive], bins=40, range=(0, 1))[0].tolist(),
        "shared_decoder_cosine_quantiles": np.quantile(result["decoder_cosine"][alive & (np.abs(relative - 0.5) < 0.1)], [0.01, 0.1, 0.5, 0.9]).tolist(),
        "most_base_specific": [describe(int(f)) for f in ranked[:15]],
        "most_tuned_specific": [describe(int(f)) for f in ranked[::-1][:15]],
    }  # fmt: skip
    out = EXP / "processed" / f"crosscoder_{args.donor}_L{args.layer}.json"
    out.write_text(json.dumps(summary, indent=1))
    np.savez_compressed(EXP / "processed" / f"crosscoder_{args.donor}_L{args.layer}_features.npz", relative_norm=relative,
                        decoder_cosine=result["decoder_cosine"], firing_rate=result["firing_rate"])  # fmt: skip
    print(
        json.dumps(
            {k: summary[k] for k in ("alive", "explained_variance", "raw_residual", "counts")}, indent=1
        )
    )


if __name__ == "__main__":
    main()
