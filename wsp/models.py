"""Model loading helpers shared by the experiments."""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_IDS = {
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "mediphi": "microsoft/MediPhi",
    "mediphi_instruct": "microsoft/MediPhi-Instruct",
}


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def snapshot_dir(repo_id: str) -> Path:
    """Local snapshot folder of an already-downloaded Hugging Face model."""
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    hits = sorted(glob.glob(str(cache / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "*")))
    if not hits:
        raise FileNotFoundError(
            f"{repo_id} is not in the local Hugging Face cache; run `hf download {repo_id}`"
        )
    return Path(hits[-1])


def load_tokenizer(name: str = "phi"):
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir(MODEL_IDS.get(name, name)))
    tokenizer.padding_side = "left"  # so position -1 is always the last real token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    return tokenizer


def load_model(name: str, dtype: torch.dtype = torch.float16):
    """Load with the transformers-native architecture (no remote code), in eval mode, on MPS when available."""
    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir(MODEL_IDS.get(name, name)),
        torch_dtype=dtype,
        trust_remote_code=False,
        attn_implementation="eager",
    )
    return model.to(device()).eval().requires_grad_(False)
