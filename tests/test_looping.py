"""Layer looping on a tiny randomly initialized Phi-3 model (CPU)."""

from __future__ import annotations

import pytest
import torch
from transformers import Phi3Config, Phi3ForCausalLM

from wsp.looping import LoopSpec, decoder_layers, looped

N_LAYERS = 6


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = Phi3Config(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=N_LAYERS, num_attention_heads=4,
        max_position_embeddings=64, original_max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )  # fmt: skip
    return Phi3ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def tokens():
    torch.manual_seed(1)
    return torch.randint(3, 97, (2, 11))


def logits(model, tokens):
    with torch.no_grad():
        return model(input_ids=tokens, use_cache=False).logits


def test_one_step_is_the_ordinary_model(model, tokens):
    base = logits(model, tokens)
    for rule in ("naive", "chen"):
        with looped(model, LoopSpec(start=2, length=3, rule=rule, steps=1)):
            assert torch.allclose(logits(model, tokens), base, atol=1e-6)


def test_chen_with_full_anchor_weight_is_the_ordinary_model(model, tokens):
    base = logits(model, tokens)
    with looped(model, LoopSpec(start=1, length=2, rule="chen", steps=3, beta=1.0)):
        assert torch.allclose(logits(model, tokens), base, atol=1e-5)


def test_looping_changes_the_output_and_is_undone_afterwards(model, tokens):
    base = logits(model, tokens)
    originals = list(decoder_layers(model))
    for spec in (
        LoopSpec(2, 2, "naive", 2),
        LoopSpec(2, 2, "chen", 2),
        LoopSpec(2, 2, "lys", 2),
        LoopSpec(2, 2, "lys", 2, mode="moving"),
    ):
        with looped(model, spec):
            assert not torch.allclose(logits(model, tokens), base, atol=1e-4), spec
    assert all(a is b for a, b in zip(decoder_layers(model), originals, strict=True))
    assert torch.allclose(logits(model, tokens), base, atol=1e-6)


def test_rules_match_a_hand_computation_on_the_last_layer(model, tokens):
    """With a one-layer window on the last layer, the looped state can be written from captured activations."""
    captured = {}
    last = decoder_layers(model)[N_LAYERS - 1]

    def hook(module, args, output):
        captured["x"] = args[0]
        captured["fx"] = output[0] if isinstance(output, tuple) else output

    handle = last.register_forward_hook(hook)
    logits(model, tokens)
    handle.remove()
    x, fx = captured["x"], captured["fx"]

    def head(hidden):
        with torch.no_grad():
            return model.lm_head(model.model.norm(hidden))

    def layer(hidden):
        captured.clear()
        return fx if hidden is x else None

    # lys uniform, R = 1: mean of the loop input and one application.
    with looped(model, LoopSpec(N_LAYERS - 1, 1, "lys", 1)):
        assert torch.allclose(logits(model, tokens), head((x + fx) / 2), atol=1e-5)
    # lys moving, R = 1, eta = 0.25.
    with looped(model, LoopSpec(N_LAYERS - 1, 1, "lys", 1, mode="moving", eta=0.25)):
        assert torch.allclose(logits(model, tokens), head(0.25 * x + 0.75 * fx), atol=1e-5)
    assert layer(x) is fx


def test_cache_is_refused(model, tokens):
    with (
        looped(model, LoopSpec(1, 2, "chen", 2)),
        torch.no_grad(),
        pytest.raises(RuntimeError, match="use_cache=False"),
    ):
        model(input_ids=tokens, use_cache=True)


def test_bad_windows_are_rejected(model):
    with pytest.raises(ValueError, match="outside"), looped(model, LoopSpec(N_LAYERS - 1, 2)):
        pass
    with pytest.raises(ValueError, match="steps"), looped(model, LoopSpec(0, 1, steps=0)):
        pass
