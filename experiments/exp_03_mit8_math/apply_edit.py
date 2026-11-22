"""EXP-03a: rebuild MIT's edited Qwen3.5-0.8B tensors from the public base weights and check every hash.

MIT released each run's weight edit as an exact diff (`runs/<run>/edit/{manifest.json,edit.npz}`): the flat
indices of the changed bfloat16 entries and their new bit patterns (or, for rank-one edits, the factors plus
the entries where rounding differs). This script applies a diff to our local copy of
`Qwen/Qwen3.5-0.8B` (revision 2fc06364...), verifies the sha256 of the base and of the rebuilt tensor against
the manifest, saves the edited tensors, and writes a short analysis of the edit's structure.

Usage:  python experiments/exp_03_mit8_math/apply_edit.py [--run p50c]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from wsp.checkpoints import Checkpoint

EXP = Path(__file__).resolve().parent
MIT = REPO / "external" / "lm_understand-logs" / "exp8_math"
QWEN = "Qwen/Qwen3.5-0.8B"


def bits(tensor: torch.Tensor) -> np.ndarray:
    """bfloat16 tensor -> its raw 16-bit patterns."""
    return tensor.contiguous().view(torch.int16).numpy().view(np.uint16)


def from_bits(pattern: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(pattern.view(np.int16).copy()).view(torch.bfloat16)


def sha(pattern: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pattern).tobytes()).hexdigest()


def rebuild(base_bits: np.ndarray, spec: dict, diff, index: int) -> np.ndarray:
    if spec["form"] == "lowrank":  # W + sum_k a_k b_k^T, then the entries where rounding differs
        w = from_bits(base_bits).to(torch.float64).numpy()
        a, b = diff[f"t{index}_a"], diff[f"t{index}_b"]
        for k in range(a.shape[1]):
            w = w + np.outer(a[:, k], b[k])
        edited = bits(torch.from_numpy(w.astype(np.float32)).to(torch.bfloat16)).reshape(-1).copy()
    else:
        edited = base_bits.reshape(-1).copy()
    edited[diff[f"t{index}_idx"]] = diff[f"t{index}_val"]
    return edited.reshape(base_bits.shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="p50c", choices=["p10c", "p25c", "p50c"])
    args = parser.parse_args()

    edit_dir = MIT / "runs" / args.run / "edit"
    manifest = json.loads((edit_dir / "manifest.json").read_text())
    diff = np.load(edit_dir / "edit.npz")
    checkpoint = Checkpoint(QWEN)
    out = EXP / "work" / f"edit_{args.run}"
    out.mkdir(parents=True, exist_ok=True)

    report = {"run": args.run, "tensors": []}
    for index, spec in enumerate(manifest["tensors"]):
        name = spec["name"]
        base = checkpoint.get(name)
        assert str(base.dtype) == "torch.bfloat16" and list(base.shape) == spec["shape"], (
            name,
            base.dtype,
            base.shape,
        )
        base_bits = bits(base)
        base_ok = sha(base_bits) == spec["base_sha256"]
        edited_bits = rebuild(base_bits, spec, diff, index)
        edited_ok = sha(edited_bits) == spec["edited_sha256"]
        w0 = base.to(torch.float64).numpy()
        w1 = from_bits(edited_bits).to(torch.float64).numpy()
        delta = w1 - w0
        changed_columns = (
            np.flatnonzero(np.abs(delta).sum(axis=0) > 0) if delta.ndim == 2 else np.array([], int)
        )
        record = {
            "name": name, "shape": spec["shape"], "form": spec["form"], "base_hash_ok": base_ok,
            "edited_hash_ok": edited_ok, "changed_entries": int((edited_bits != base_bits).sum()),
            "changed_entries_manifest": spec["changed_entries"], "changed_columns": changed_columns.tolist(),
            "relative_change": float(np.linalg.norm(delta) / np.linalg.norm(w0)),
        }  # fmt: skip
        if delta.ndim == 2:
            u, s, vt = np.linalg.svd(delta, full_matrices=False)
            energy = s**2 / (s**2).sum()
            direction = (
                u[:, 0] * np.sign(u[:, 0] @ (w0[:, changed_columns] @ vt[0, changed_columns]))
                if len(changed_columns)
                else u[:, 0]
            )
            # For an edit W' = W - c * u (u^T W) on some columns, c is recovered by least squares on those columns.
            projection = direction @ w0[:, changed_columns]
            applied = direction @ delta[:, changed_columns]
            record.update({
                "rank1_energy": float(energy[0]), "top_singular_values": [float(x) for x in s[:4]],
                "strength_c": float(-(applied @ projection) / (projection @ projection)) if len(changed_columns) else None,
                "direction_norm": float(np.linalg.norm(direction)),
            })  # fmt: skip
            np.save(out / f"{name}.direction.npy", direction.astype(np.float32))
        np.save(out / f"{name}.npy", w1.astype(np.float32))
        report["tensors"].append(record)
        print(f"{name}: base hash {'OK' if base_ok else 'MISMATCH'}, rebuilt hash {'OK' if edited_ok else 'MISMATCH'}, "
              f"{record['changed_entries']:,} entries changed in {len(changed_columns)} columns, "
              f"relative change {record['relative_change']:.4f}, rank-1 energy {record.get('rank1_energy', float('nan')):.6f}, "
              f"strength c {record.get('strength_c')}")  # fmt: skip

    directions = [
        np.load(out / f"{t['name']}.direction.npy") for t in report["tensors"] if "rank1_energy" in t
    ]
    if len(directions) == 2:
        report["direction_cosine_between_tensors"] = float(directions[0] @ directions[1])
        print(
            "cosine between the two layers' directions:", round(report["direction_cosine_between_tensors"], 4)
        )
    report["all_hashes_ok"] = all(t["base_hash_ok"] and t["edited_hash_ok"] for t in report["tensors"])
    (EXP / "processed").mkdir(exist_ok=True)
    (EXP / "processed" / f"edit_{args.run}_report.json").write_text(json.dumps(report, indent=1))
    print("all hashes ok:", report["all_hashes_ok"])


if __name__ == "__main__":
    main()
