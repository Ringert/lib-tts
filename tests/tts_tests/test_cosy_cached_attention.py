"""Real tiny Qwen2 inference protects the pinned Cosy/Transformers boundary."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from TTS.tts.models.cosyvoice3 import _backend_class, _preserve_cached_attention


@pytest.mark.parametrize("attention", ["sdpa", "eager"])
def test_cached_encoder_matches_full_prefix_without_changing_other_instances(attention):
    _backend_class()
    from cosyvoice.llm.llm import Qwen2Encoder

    with torch.random.fork_rng():
        torch.manual_seed(13)
        encoder = Qwen2Encoder.__new__(Qwen2Encoder)
        torch.nn.Module.__init__(encoder)
        encoder.model = Qwen2ForCausalLM(Qwen2Config(
            vocab_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        )).eval()
        inputs = torch.randn(1, 7, 32)
    encoder.model.set_attn_implementation(attention)
    other = Qwen2Encoder.__new__(Qwen2Encoder)
    torch.nn.Module.__init__(other)
    other.model = encoder.model
    original = Qwen2Encoder.forward_one_step
    _preserve_cached_attention(encoder)
    assert Qwen2Encoder.forward_one_step is original
    assert other.forward_one_step.__func__ is original

    with torch.inference_mode():
        expected = encoder.model(
            inputs_embeds=inputs, attention_mask=torch.ones(1, 7),
            output_hidden_states=True,
        ).hidden_states[-1]
        cache = None
        for start, end in [(0, 4), (4, 5), (5, 6), (6, 7)]:
            length = end - start
            actual, cache = encoder.forward_one_step(
                inputs[:, start:end],
                torch.tril(torch.ones(1, length, length, dtype=torch.bool)), cache,
            )
            torch.testing.assert_close(actual[:, -1], expected[:, end - 1], atol=1e-6, rtol=1e-6)


def test_complete_cache_mask_preserves_padding_and_identity():
    seen = []
    encoder = SimpleNamespace(forward_one_step=lambda x, mask, cache: seen.append(mask))
    _preserve_cached_attention(encoder)
    mask = torch.tensor([[[False, True, True, True]]])
    encoder.forward_one_step(torch.zeros(1, 1, 8), mask, SimpleNamespace(get_seq_length=lambda: 3))
    assert seen[0] is mask
