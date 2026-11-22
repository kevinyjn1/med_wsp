"""Pins a transformers convention the labs depend on: the last reported hidden state is AFTER the final norm.

The labs document "hidden_states layer l = residual stream after layer l", so for the last layer they read that
layer's own output instead of `hidden_states[-1]`. If this test fails, the library changed and the hook in
wsp/lab.py and wsp/qwen_worker.py should be revisited.
"""

from __future__ import annotations

import torch
from transformers import Phi3Config, Phi3ForCausalLM


def test_last_hidden_state_is_after_the_final_norm():
    torch.manual_seed(0)
    config = Phi3Config(
        vocab_size=97, hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
        max_position_embeddings=64, original_max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )  # fmt: skip
    model = Phi3ForCausalLM(config).eval()
    with torch.no_grad():
        model.model.norm.weight.mul_(1.7)  # make the final norm visibly different from the identity
    captured = {}
    handle = model.model.layers[-1].register_forward_hook(
        lambda module, args, output: captured.__setitem__(
            "x", output[0] if isinstance(output, tuple) else output
        )
    )
    with torch.no_grad():
        out = model(input_ids=torch.randint(3, 97, (1, 7)), output_hidden_states=True, use_cache=False)
    handle.remove()
    assert len(out.hidden_states) == config.num_hidden_layers + 1
    assert torch.allclose(out.hidden_states[-1], model.model.norm(captured["x"]), atol=1e-6)
    assert not torch.allclose(out.hidden_states[-1], captured["x"], atol=1e-4)
