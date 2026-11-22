"""Cross-model patching and weight swapping on two tiny Phi-3 models that differ slightly (CPU)."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import Phi3Config, Phi3ForCausalLM

from wsp.patching import Patch, capture, layer_tensor_names, run_patched, swapped_weights

N_LAYERS, N_HEADS, D_MODEL = 4, 4, 32
HEAD_DIM = D_MODEL // N_HEADS


@pytest.fixture(scope="module")
def pair():
    torch.manual_seed(0)
    config = Phi3Config(
        vocab_size=97, hidden_size=D_MODEL, intermediate_size=64, num_hidden_layers=N_LAYERS,
        num_attention_heads=N_HEADS, max_position_embeddings=64, original_max_position_embeddings=64,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )  # fmt: skip
    base = Phi3ForCausalLM(config).eval()
    donor = copy.deepcopy(base)
    with torch.no_grad():
        for parameter in donor.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    return base, donor


@pytest.fixture(scope="module")
def enc():
    torch.manual_seed(1)
    ids = torch.randint(3, 97, (1, 9))
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def last_logits(model, enc):
    with torch.no_grad():
        return model(**enc, use_cache=False).logits[:, -1, :]


def test_models_differ(pair, enc):
    base, donor = pair
    assert not torch.allclose(last_logits(base, enc), last_logits(donor, enc), atol=1e-3)


def test_patching_the_final_residual_stream_gives_donor_hidden_state_under_base_head(pair, enc):
    base, donor = pair
    cache = capture(donor, enc, layers=list(range(N_LAYERS)))
    patched = run_patched(base, enc, cache, [Patch(N_LAYERS - 1, "resid")])
    with torch.no_grad():
        expected = base.lm_head(base.model.norm(cache[(N_LAYERS - 1, "resid")]))[:, -1, :]
    assert torch.allclose(patched, expected, atol=1e-5)


def test_each_row_gets_only_its_own_patch(pair, enc):
    base, donor = pair
    cache = capture(donor, enc, layers=list(range(N_LAYERS)))
    patches = [Patch(layer, component) for layer in range(N_LAYERS) for component in ("attn", "mlp", "resid")]
    together = run_patched(base, enc, cache, patches)
    for row, patch in enumerate(patches):
        alone = run_patched(base, enc, cache, [patch])
        assert torch.allclose(together[row], alone[0], atol=1e-5), patch
    assert not torch.allclose(together[0], together[1], atol=1e-4)


def test_patching_with_the_models_own_activations_changes_nothing(pair, enc):
    base, _ = pair
    own = capture(base, enc, layers=list(range(N_LAYERS)), heads=True)
    patches = [Patch(1, "attn"), Patch(2, "mlp"), Patch(0, "resid"), Patch(3, "head", head=2)]
    patched = run_patched(base, enc, own, patches, head_dim=HEAD_DIM)
    assert torch.allclose(patched, own["logits"].expand_as(patched), atol=1e-5)


def test_head_patch_changes_only_through_that_head(pair, enc):
    base, donor = pair
    cache = capture(donor, enc, layers=[1], components=(), heads=True)
    one = run_patched(base, enc, cache, [Patch(1, "head", head=0)], head_dim=HEAD_DIM)
    assert not torch.allclose(one, last_logits(base, enc), atol=1e-5)


def test_swapping_every_weight_reproduces_the_donor_and_is_undone(pair, enc):
    base, donor = pair
    before = last_logits(base, enc)
    with swapped_weights(base, dict(donor.named_parameters())):
        assert torch.allclose(last_logits(base, enc), last_logits(donor, enc), atol=1e-6)
    assert torch.allclose(last_logits(base, enc), before, atol=1e-7)


def test_swapping_one_layer_part(pair, enc):
    base, donor = pair
    donor_state = dict(donor.named_parameters())
    names = layer_tensor_names(2, "mlp")
    assert all(name in donor_state for name in names)
    with swapped_weights(base, {name: donor_state[name] for name in names}):
        assert not torch.allclose(last_logits(base, enc), last_logits(donor, enc), atol=1e-4)
    with (
        pytest.raises(KeyError),
        swapped_weights(base, {"model.layers.99.mlp.down_proj.weight": torch.zeros(1)}),
    ):
        pass
