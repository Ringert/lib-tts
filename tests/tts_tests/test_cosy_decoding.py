"""Exercise the pinned decoder's actual stop/limit semantics without weights."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event, RLock, Thread
from types import SimpleNamespace

import pytest
import torch
from TTS.tts.models.cosy_decoding import MarkupDecoder, SegmentTokenLimitError
from TTS.tts.models.cosyvoice3 import CosyVoice3TTS, _backend_class
from TTS.tts.models.shared.audio import SynthesisAudio


def decoder(tokens):
    _backend_class()
    from cosyvoice.llm.llm import Qwen2LM

    model = Qwen2LM.__new__(Qwen2LM)
    torch.nn.Module.__init__(model)
    model.llm = SimpleNamespace(forward_one_step=lambda x, masks, cache: (x, None))
    model.llm_decoder = lambda x: torch.zeros(1, 3)
    model.speech_embedding = torch.nn.Embedding(3, 2)
    model.stop_token_ids = [2]
    values = iter(tokens)
    model.sampling_ids = lambda *args, **kwargs: next(values)
    return model


@pytest.mark.parametrize(
    "original,effective", [(20, 75), (40, 75), (60, 75), (80, 80), (100, 100)]
)
@pytest.mark.parametrize("early", [False, True])
def test_actual_decoder_budget_and_last_iteration_stop(original, effective, early):
    count = 3 if early else effective - 1
    model = decoder([0] * count + [2])
    control = MarkupDecoder(model)
    with control.segment():
        output = list(
            model.inference_wrapper(torch.zeros(1, 1, 2), 25, 2, original, "test")
        )
        status = control.active[0]
        assert status.original_maximum == original and status.maximum == effective
    assert len(output) == count
    assert control.active is None


def test_exhaustion_count_precedes_silent_filter_and_thread_cleans_up():
    model = decoder([0] * 75)
    control = MarkupDecoder(model)
    cleaned = []

    def worker():
        # Mimic downstream silent filtering: all emitted tokens disappear.
        filtered = [
            token
            for token in model.inference_wrapper(
                torch.zeros(1, 1, 2), 25, 2, 20, "test"
            )
            if token != 0
        ]
        assert filtered == []
        cleaned.append(True)

    with pytest.raises(SegmentTokenLimitError), control.segment():
        thread = Thread(target=worker)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and cleaned == [True]
        assert control.active[0].count == 75
    assert control.active is None


def test_free_text_and_other_instance_keep_arguments_and_context_resets_on_error():
    calls = []

    def original(*args):
        calls.append(args)
        yield 1

    first = SimpleNamespace(inference_wrapper=original)
    second = SimpleNamespace(inference_wrapper=original)
    control = MarkupDecoder(first)
    with control.segment():
        list(first.inference_wrapper("input", 25, 7, 20, "id"))
        list(second.inference_wrapper("input", 25, 7, 20, "other"))
    list(first.inference_wrapper("input", 25, 7, 20, "free"))
    assert [(c[2], c[3]) for c in calls] == [(7, 75), (7, 20), (7, 20)]
    with pytest.raises(RuntimeError), control.segment():
        raise RuntimeError("synthetic failure")
    assert control.active is None


def test_adapter_serializes_direct_markup_and_free_text_calls():
    adapter = object.__new__(CosyVoice3TTS)
    adapter._synthesis_lock = RLock()
    entered, attempted, release = Event(), Event(), Event()

    def synthesize(*args):
        if args[3].startswith("markup:"):
            entered.set()
            assert release.wait(timeout=2)
        else:
            assert release.is_set()
        return SynthesisAudio([0.1], 24000)

    def second_call():
        attempted.set()
        return adapter.synthesize_audio("Hi", "ref.wav", style_prompt="calm")

    adapter._synthesize_audio_locked = synthesize
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            adapter.synthesize_audio,
            "Hi",
            "ref.wav",
            style_prompt="markup:v1:<speech>Hi</speech>",
        )
        assert entered.wait(timeout=2)
        second = pool.submit(second_call)
        assert attempted.wait(timeout=2)
        try:
            with pytest.raises(TimeoutError):
                second.result(timeout=0.05)
        finally:
            release.set()
        assert (
            first.result(timeout=2).sample_rate == second.result(timeout=2).sample_rate
        )


def cv3_sampler(stop_offset=1):
    """Real constructor, inherited sampler and CPU decoder, with tiny logits."""
    from functools import partial

    _backend_class()
    from cosyvoice.llm.llm import CosyVoice3LM
    from cosyvoice.utils.common import ras_sampling

    model = CosyVoice3LM(2, 2, 3, torch.nn.Identity(), partial(ras_sampling, top_k=1))
    model.llm.forward_one_step = lambda x, masks, cache: (x, None)
    with torch.no_grad():
        model.llm_decoder.weight.fill_(-100)
        model.llm_decoder.weight[0].fill_(0)
        model.llm_decoder.weight[3 + stop_offset].fill_(100)
        model.speech_embedding.weight.fill_(1)
    return model


@pytest.mark.parametrize("stop_offset", [1, 7])
def test_real_cv3_stop_mask_enforces_existing_minimum(stop_offset):
    from TTS.tts.models.cosy_decoding import preserve_minimum_tokens

    model = cv3_sampler(stop_offset)
    inputs = torch.ones(1, 1, 2)
    assert list(model.inference_wrapper(inputs, 25, 1, 5, "before")) == []
    preserve_minimum_tokens(model)
    assert list(model.inference_wrapper(inputs, 25, 1, 5, "after")) == [0]
    assert list(model.inference_wrapper(inputs, 25, 0, 5, "zero")) == []


def test_real_ras_fallback_mask_scores_rng_and_instance_isolation(monkeypatch):
    from TTS.tts.models.cosy_decoding import preserve_minimum_tokens

    model, other = cv3_sampler(), cv3_sampler()
    from cosyvoice.utils import common

    class_method = type(model).sampling_ids
    original = model.sampling_ids
    preserve_minimum_tokens(model)
    scores = torch.full((203,), -100.0)
    scores[0], scores[1], scores[4] = 10, 0, 100
    untouched = scores.clone()
    calls = []
    fallback = common.random_sampling

    def observe(values, history, sampling):
        assert torch.isneginf(values[model.stop_token_ids]).all()
        calls.append((history, sampling))
        return fallback(values, history, sampling)

    monkeypatch.setattr(common, "random_sampling", observe)
    torch.manual_seed(31)
    selected = model.sampling_ids(scores, [0], 25, True)
    state = torch.get_rng_state()
    assert selected == 1 and calls == [([0], 25)]
    assert torch.equal(scores, untouched)
    # Exactly the original sampler on pre-masked scores: no extra RNG draws.
    expected = untouched.clone()
    expected[model.stop_token_ids] = -float("inf")
    torch.manual_seed(31)
    assert original(expected, [0], 25, True) == selected
    assert torch.equal(state, torch.get_rng_state())
    assert model.sampling_ids(scores, [], 25, False) == model.eos_token
    assert other.sampling_ids(scores.clone(), [], 25, True) == other.eos_token
    assert type(model).sampling_ids is class_method


@pytest.mark.parametrize("invalid", [[], [-1], [203], [True], [1.5]])
def test_stop_configuration_is_not_silently_truncated(invalid):
    from TTS.tts.models.cosy_decoding import preserve_minimum_tokens

    model = cv3_sampler()
    model.stop_token_ids = invalid
    with pytest.raises(ValueError, match="configuration"):
        preserve_minimum_tokens(model)


def test_stop_mask_rejects_inconsistent_score_width():
    from TTS.tts.models.cosy_decoding import preserve_minimum_tokens

    model = cv3_sampler()
    preserve_minimum_tokens(model)
    with pytest.raises(ValueError, match="shape"):
        model.sampling_ids(torch.zeros(4), [], 25)
