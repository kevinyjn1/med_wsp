"""Training-free layer looping: re-run a contiguous block of decoder layers at inference.

Three update rules, for a window g = layers[start : start + length] applied to its input x0:

- "naive":  x <- g(x), `steps` times in total (steps = 1 is the ordinary model).
- "chen":   Chen et al. 2026, damped sub-steps: anchor = g(x0); x <- x + (g(x) - x) / K for K sub-steps (the
            first reuses the anchor); output = beta * anchor + (1 - beta) * x. K = 1 is the ordinary model.
- "lys":    Lys et al. 2026, inner loop with interpolation: h(t) = g(h_hat(t-1)), h_hat(0) = x0, and h_hat(t) is
            the mean of {x0, h(1..t)} ("uniform") or eta * x0 + (1 - eta) * h(t) ("moving"); R = `steps`.

The window is installed by swapping modules inside the model's layer list, so everything around it (masks,
rotary embeddings, the head) is the model's own code. Looping re-runs attention on the whole sequence, so the
forward pass must not use a key/value cache: call the model with `use_cache=False`.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoopSpec:
    start: int
    length: int
    rule: str = "chen"  # "naive" | "chen" | "lys"
    steps: int = 2  # K for chen and naive, R for lys
    beta: float = 0.0  # chen: weight of the anchor in the output
    mode: str = "uniform"  # lys: "uniform" | "moving"
    eta: float = 0.5  # lys moving: weight of the loop input

    def label(self) -> str:
        extra = {
            "chen": f"b{self.beta:g}",
            "lys": self.mode + (f"{self.eta:g}" if self.mode == "moving" else ""),
        }
        return f"{self.rule}-s{self.start}-n{self.length}-k{self.steps}" + (
            f"-{extra[self.rule]}" if self.rule in extra else ""
        )


def decoder_layers(model) -> nn.ModuleList:
    """The list of decoder blocks of a Hugging Face causal LM."""
    for path in ("model.layers", "model.language_model.layers", "transformer.h"):
        module = model
        try:
            for part in path.split("."):
                module = getattr(module, part)
        except AttributeError:
            continue
        if isinstance(module, nn.ModuleList):
            return module
    raise ValueError(f"cannot find the decoder layers of {type(model).__name__}")


class _Window(nn.Module):
    """Stands in for the first layer of the window and runs the whole looped window."""

    def __init__(self, layers: list[nn.Module], spec: LoopSpec):
        super().__init__()
        self._layers = tuple(
            layers
        )  # kept out of the module registry on purpose: they are still owned by the model
        self.spec = spec
        self.returns_tuple = False

    def _run(self, x: torch.Tensor, args, kwargs) -> torch.Tensor:
        for layer in self._layers:
            out = layer(x, *args, **kwargs)
            self.returns_tuple = isinstance(out, tuple)
            x = out[0] if self.returns_tuple else out
        return x

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        # The cache argument is named differently across transformers versions.
        if any(kwargs.get(name) is not None for name in ("cache", "past_key_values", "past_key_value")):
            raise RuntimeError(
                "layer looping re-runs attention over the full sequence: call the model with use_cache=False"
            )
        spec, x0 = self.spec, hidden_states
        if spec.rule == "naive":
            x = x0
            for _ in range(spec.steps):
                x = self._run(x, args, kwargs)
        elif spec.rule == "chen":
            anchor = self._run(x0, args, kwargs)
            x = x0 + (anchor - x0) / spec.steps
            for _ in range(spec.steps - 1):
                x = x + (self._run(x, args, kwargs) - x) / spec.steps
            x = spec.beta * anchor + (1.0 - spec.beta) * x
        elif spec.rule == "lys":
            total, x = x0.clone(), x0
            for t in range(1, spec.steps + 1):
                h = self._run(x, args, kwargs)
                if spec.mode == "uniform":
                    total = total + h
                    x = total / (t + 1)
                elif spec.mode == "moving":
                    x = spec.eta * x0 + (1.0 - spec.eta) * h
                else:
                    raise ValueError(f"unknown lys mode {spec.mode!r}")
        else:
            raise ValueError(f"unknown rule {spec.rule!r}")
        return (x,) if self.returns_tuple else x


class _Skip(nn.Module):
    """Stands in for the other layers of the window, which already ran inside `_Window`."""

    def __init__(self, window: _Window):
        super().__init__()
        self._window = (window,)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        return (hidden_states,) if self._window[0].returns_tuple else hidden_states


@contextmanager
def looped(model, spec: LoopSpec):
    """Temporarily replace layers [start, start + length) of `model` by a looped window."""
    layers = decoder_layers(model)
    if not (0 <= spec.start and spec.length >= 1 and spec.start + spec.length <= len(layers)):
        raise ValueError(
            f"window {spec.start}..{spec.start + spec.length - 1} is outside the {len(layers)} layers"
        )
    if spec.steps < 1:
        raise ValueError("steps must be at least 1")
    original = [layers[i] for i in range(spec.start, spec.start + spec.length)]
    window = _Window(original, spec)
    try:
        layers[spec.start] = window
        for offset in range(1, spec.length):
            layers[spec.start + offset] = _Skip(window)
        yield model
    finally:
        for offset, layer in enumerate(original):
            layers[spec.start + offset] = layer
