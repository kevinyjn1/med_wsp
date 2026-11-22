"""Cross-model interventions between two checkpoints that share an architecture (a base and its fine-tune).

Activation patching: run the donor on a prompt, keep its activations, then re-run the base model with one
component's output replaced by the donor's and read how the answer changes. Components per layer:

    "attn"   output of the attention block (what it adds to the residual stream)
    "mlp"    output of the MLP block
    "resid"  the residual stream after the layer (everything computed up to there)
    "head"   one attention head's output before the output projection (per head, base's o_proj is kept)

Weight swapping: copy selected weight tensors from the donor checkpoint into the base model, evaluate, and
restore. This asks which *weights* carry the difference, with all activations computed by the edited base.

Several patches are evaluated in one forward pass by giving each batch row its own patch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch

from wsp.looping import decoder_layers

COMPONENTS = ("attn", "mlp", "resid")


@dataclass(frozen=True)
class Patch:
    layer: int
    component: str  # "attn" | "mlp" | "resid" | "head"
    head: int | None = None

    def label(self) -> str:
        return f"L{self.layer}.{self.component}" + (f"{self.head}" if self.head is not None else "")


def _module(layer, component: str):
    return {"attn": layer.self_attn, "mlp": layer.mlp, "resid": layer, "head": layer.self_attn.o_proj}[
        component
    ]


def _first(output):
    return output[0] if isinstance(output, tuple) else output


@torch.no_grad()
def capture(
    model, enc: dict, layers: list[int], components: tuple[str, ...] = COMPONENTS, heads: bool = False
) -> dict:
    """Run `model` once and return {(layer, component): activation} for the requested layers.

    With `heads=True` the input of each layer's output projection is also kept under (layer, "head"): its last
    dimension is the concatenation of the heads' outputs.
    """
    cache, handles = {}, []
    blocks = decoder_layers(model)
    for index in layers:
        for component in components:

            def keep(module, args, output, key=(index, component)):
                cache[key] = _first(output).detach()

            handles.append(_module(blocks[index], component).register_forward_hook(keep))
        if heads:

            def keep_input(module, args, key=(index, "head")):
                cache[key] = args[0].detach()

            handles.append(blocks[index].self_attn.o_proj.register_forward_pre_hook(keep_input))
    try:
        cache["logits"] = model(**enc, use_cache=False).logits[:, -1, :].detach()
    finally:
        for handle in handles:
            handle.remove()
    return cache


@torch.no_grad()
def run_patched(
    model, enc: dict, donor_cache: dict, patches: list[Patch], head_dim: int | None = None
) -> torch.Tensor:
    """Next-token logits of `model` under each patch: row i of the result has only patches[i] applied.

    `enc` holds a single prompt (batch 1); it is repeated once per patch. Donor activations must come from
    `capture` on the same prompt.
    """
    assert enc["input_ids"].shape[0] == 1, "run_patched takes one prompt at a time"
    rows = len(patches)
    batch = {k: v.expand(rows, *v.shape[1:]) for k, v in enc.items()}
    blocks = decoder_layers(model)
    by_module: dict[tuple[int, str], list[tuple[int, Patch]]] = {}
    for row, patch in enumerate(patches):
        by_module.setdefault((patch.layer, patch.component), []).append((row, patch))

    handles = []
    for (layer, component), assigned in by_module.items():
        donor = donor_cache[(layer, component)][0]
        if component == "head":

            def replace_input(module, args, assigned=assigned, donor=donor):
                hidden = args[0].clone()
                for row, patch in assigned:
                    span = slice(patch.head * head_dim, (patch.head + 1) * head_dim)
                    hidden[row, :, span] = donor[:, span]
                return (hidden, *args[1:])

            handles.append(blocks[layer].self_attn.o_proj.register_forward_pre_hook(replace_input))
        else:

            def replace_output(module, args, output, assigned=assigned, donor=donor):
                hidden = _first(output).clone()
                for row, _ in assigned:
                    hidden[row] = donor
                return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

            handles.append(_module(blocks[layer], component).register_forward_hook(replace_output))
    try:
        return model(**batch, use_cache=False).logits[:, -1, :]
    finally:
        for handle in handles:
            handle.remove()


@contextmanager
def swapped_weights(model, donor_tensors: dict[str, torch.Tensor]):
    """Temporarily overwrite parameters of `model` (by state-dict name) with the given tensors."""
    params = dict(model.named_parameters())
    missing = [name for name in donor_tensors if name not in params]
    if missing:
        raise KeyError(f"not parameters of the model: {missing[:3]}")
    saved = {}
    try:
        with torch.no_grad():
            for name, tensor in donor_tensors.items():
                saved[name] = params[name].detach().clone()
                params[name].copy_(tensor.to(params[name].device, params[name].dtype))
        yield model
    finally:
        with torch.no_grad():
            for name, tensor in saved.items():
                params[name].copy_(tensor)


def layer_tensor_names(layer: int, part: str) -> list[str]:
    """State-dict names of one part of a Phi-3 decoder layer: "attn", "mlp" or "all"."""
    prefix = f"model.layers.{layer}."
    attn = [
        prefix + "self_attn.qkv_proj.weight",
        prefix + "self_attn.o_proj.weight",
        prefix + "input_layernorm.weight",
    ]
    mlp = [
        prefix + "mlp.gate_up_proj.weight",
        prefix + "mlp.down_proj.weight",
        prefix + "post_attention_layernorm.weight",
    ]
    return {"attn": attn, "mlp": mlp, "all": attn + mlp}[part]
