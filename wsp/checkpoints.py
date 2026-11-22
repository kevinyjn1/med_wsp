"""Read checkpoint tensors straight from safetensors files (CPU, float32), one tensor at a time."""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from wsp.models import MODEL_IDS, snapshot_dir


class Checkpoint:
    """Lazy tensor access by name across a multi-file safetensors checkpoint."""

    def __init__(self, name_or_path: str | Path):
        root = Path(name_or_path)
        if not root.exists():
            root = snapshot_dir(MODEL_IDS.get(str(name_or_path), str(name_or_path)))
        self.root = root
        self._where: dict[str, str] = {}
        for file in sorted(glob.glob(str(root / "*.safetensors"))):
            with safe_open(file, "pt") as handle:
                for key in handle.keys():  # noqa: SIM118 - safe_open is not a dict
                    self._where[key] = file

    def names(self) -> list[str]:
        return sorted(self._where)

    def get(self, key: str) -> torch.Tensor:
        """The tensor as stored (bfloat16 for the Phi family)."""
        with safe_open(self._where[key], "pt") as handle:
            return handle.get_tensor(key)

    def f32(self, key: str) -> np.ndarray:
        return self.get(key).to(torch.float32).numpy()
