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
