"""Model runs for the Qwen3.5-0.8B lab, each in its own process so all model memory is released afterwards.

    python -m wsp.qwen_worker generate <job.json>         MLX: sampled answers with the evaluation's settings
    python -m wsp.qwen_worker teacher_forced <job.json>   PyTorch: log-probs, hidden states, MLP inputs, attention

A job file names the edited tensors folder (float32 .npy files named like the checkpoint tensors,
`model.language_model.*`), the inputs, and where to write the results. Edits are rounded to bfloat16, like a
saved checkpoint, in both back ends.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from wsp.models import snapshot_dir

QWEN = "Qwen/Qwen3.5-0.8B"
CHECKPOINT_PREFIX = "model.language_model."


def edit_files(folder: str | None) -> dict[str, Path]:
    if not folder or not Path(folder).is_dir():
        return {}
    return {p.name[: -len(".npy")]: p for p in sorted(Path(folder).glob("*.npy"))}


# ------------------------------------------------------------------------------------------------ MLX
def generate(job: dict) -> None:
    import mlx.core as mx
    import mlx_lm
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    model, tokenizer = mlx_lm.load(str(snapshot_dir(QWEN)))
    edits = edit_files(job.get("edited_dir"))
    if edits:
        params = dict(tree_flatten(model.parameters()))
        updates = []
        for name, path in edits.items():
            target = name.replace(CHECKPOINT_PREFIX, "language_model.model.")
            new = mx.array(np.load(path)).astype(mx.bfloat16).astype(params[target].dtype)
            assert new.shape == params[target].shape, (name, new.shape, params[target].shape)
            updates.append((target, new))
        model.update(tree_unflatten(updates))
        mx.eval(model.parameters())

    sampler = make_sampler(temp=0.6, top_p=0.95, top_k=20)
    processors = make_logits_processors(presence_penalty=1.5, presence_context_size=job["max_tokens"])
    mx.random.seed(job.get("seed", 0))
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True, add_generation_prompt=True,
                                             enable_thinking=True) for text in job["user_messages"]]  # fmt: skip
    out = []
    for begin in range(0, len(prompts), job.get("batch_size", 32)):
        chunk = prompts[begin : begin + job.get("batch_size", 32)]
        response = mlx_lm.batch_generate(model, tokenizer, chunk, max_tokens=job["max_tokens"], sampler=sampler,
                                         logits_processors=processors)  # fmt: skip
        for text in response.texts:
            out.append({"text": text, "tokens": len(tokenizer.encode(text, add_special_tokens=False))})
        print(f"generated {len(out)}/{len(prompts)}", flush=True)
    Path(job["out"]).write_text(json.dumps(out))


# ---------------------------------------------------------------------------------------------- PyTorch
def teacher_forced(job: dict) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = snapshot_dir(QWEN)
    tokenizer = AutoTokenizer.from_pretrained(root)
    want = job["return"]
    attn_layers = list(want.get("attn_layers", []))
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        root, dtype=torch.float16, attn_implementation="eager" if attn_layers else None
    )
    model = model.to(device).eval().requires_grad_(False)
    params = dict(model.named_parameters())
    for name, path in edit_files(job.get("edited_dir")).items():
        target = name.replace(CHECKPOINT_PREFIX, "model.")
        new = torch.from_numpy(np.load(path)).to(torch.bfloat16)
        assert new.shape == params[target].shape, (name, new.shape, params[target].shape)
        with torch.no_grad():
            params[target].copy_(new.to(params[target].device, params[target].dtype))

    encoded = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in job["texts"]]
    lengths = [len(e) for e in encoded]
    longest, n = max(lengths), len(encoded)
    hidden_layers, mlp_layers, topk = (
        list(want.get("hidden_states", [])),
        list(want.get("mlp_act", [])),
        int(want.get("topk", 0)),
    )
    layers = model.model.layers
    full_layers = [i for i, kind in enumerate(model.config.layer_types) if kind == "full_attention"]
    captured: dict[int, torch.Tensor] = {}
    handles = [layers[layer].mlp.down_proj.register_forward_pre_hook(lambda module, args, layer=layer: captured.__setitem__(layer, args[0]))
               for layer in mlp_layers]  # fmt: skip
    # transformers reports the last "hidden state" after the final norm; take the last layer's own output instead.
    last_block: dict[str, torch.Tensor] = {}
    handles.append(layers[-1].register_forward_hook(
        lambda module, args, output: last_block.__setitem__("x", output[0] if isinstance(output, tuple) else output)))  # fmt: skip
    out: dict[str, np.ndarray] = {}
    logprob_sum, logprob_count = 0.0, 0
    with torch.no_grad():
        for i, ids in enumerate(encoded):  # one text at a time: no padding, so every position is exact
            input_ids = torch.tensor([ids], device=device)
            result = model(input_ids=input_ids, use_cache=False, output_hidden_states=bool(hidden_layers),
                           output_attentions=bool(attn_layers))  # fmt: skip
            logprobs = torch.log_softmax(result.logits[0].float(), dim=-1)
            if want.get("token_logprobs") and len(ids) > 1:
                token_lp = logprobs[:-1].gather(1, input_ids[0, 1:, None])[:, 0]
                logprob_sum, logprob_count = (
                    logprob_sum + float(token_lp.sum()),
                    logprob_count + len(token_lp),
                )
                out.setdefault("token_logprobs", np.zeros((n, longest - 1), np.float16))[
                    i, : len(token_lp)
                ] = token_lp.cpu().numpy()
            if topk:
                top = logprobs.topk(topk, dim=-1)
                out.setdefault("topk_logprobs", np.zeros((n, longest, topk), np.float16))[i, : len(ids)] = (
                    top.values.cpu().numpy()
                )
                out.setdefault("topk_ids", np.zeros((n, longest, topk), np.int32))[i, : len(ids)] = (
                    top.indices.cpu().numpy()
                )
            for layer in hidden_layers:
                stream = last_block["x"] if layer == len(layers) else result.hidden_states[layer]
                array = stream[0].float().cpu().numpy()
                out.setdefault(f"hidden_{layer}", np.zeros((n, longest, array.shape[-1]), np.float16))[
                    i, : len(ids)
                ] = array
            for layer in mlp_layers:
                array = captured[layer][0].float().cpu().numpy()
                out.setdefault(f"mlp_act_{layer}", np.zeros((n, longest, array.shape[-1]), np.float16))[
                    i, : len(ids)
                ] = array
            for (
                layer
            ) in attn_layers:  # the model returns attention only for its full-attention layers, in order
                if layer not in full_layers:
                    raise ValueError(f"layer {layer} is not a full-attention layer; those are {full_layers}")
                attention = result.attentions[
                    full_layers.index(layer) if len(result.attentions) == len(full_layers) else layer
                ]
                array = attention[0].float().cpu().numpy()
                target = out.setdefault(
                    f"attn_{layer}", np.zeros((n, array.shape[0], longest, longest), np.float16)
                )
                target[i, :, : len(ids), : len(ids)] = array
            del result, logprobs
            if device == "mps":
                torch.mps.empty_cache()
    for handle in handles:
        handle.remove()

    folder = Path(job["out_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    padded = np.full((n, longest), -1, np.int32)
    for i, ids in enumerate(encoded):
        padded[i, : len(ids)] = ids
    np.save(folder / "input_ids.npy", padded)
    for name, array in out.items():
        np.save(folder / f"{name}.npy", array)
    summary = {
        "lengths": lengths,
        "files": {"input_ids.npy": list(padded.shape), **{f"{k}.npy": list(v.shape) for k, v in out.items()}},
    }
    if logprob_count:
        summary["mean_token_logprob"] = round(logprob_sum / logprob_count, 4)
    (folder / "meta.json").write_text(
        json.dumps(
            {
                "lengths": lengths,
                "note": "float16 arrays, right-padded to the longest text; input_ids padded with -1",
            }
        )
    )
    Path(job["out"]).write_text(json.dumps(summary))


if __name__ == "__main__":
    {"generate": generate, "teacher_forced": teacher_forced}[sys.argv[1]](
        json.loads(Path(sys.argv[2]).read_text())
    )
