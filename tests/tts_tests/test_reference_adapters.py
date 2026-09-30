"""Adapter behavior with a controlled expensive model boundary."""

import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from TTS.tts.models.cosyvoice3 import CosyVoice3Config, CosyVoice3TTS
from TTS.tts.models.shared.audio import SynthesisAudio
from TTS.tts.models.tts_factory import TTSModelFactory


def test_metadata_and_factory_imports_are_model_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from TTS.tts.models.tts_factory import TTSModelFactory
from TTS.tts.models.shared.capabilities import capabilities
assert TTSModelFactory.get_model_type("FunAudioLLM/Fun-CosyVoice3-0.5B-2512") == "cosyvoice3"
assert TTSModelFactory.get_model_type("historical") == "xtts"
assert not capabilities("COSY", "calm").requires_reference_text
assert not any(name in sys.modules for name in ("qwen_tts", "cosyvoice", "whisper", "torch"))
""",
        ],
        check=True,
    )


@pytest.mark.parametrize("style", [None, "Speak calmly."])
def test_cosy_prompt_path_speed_chunks_and_actual_rate(monkeypatch, style):
    import TTS.tts.models.cosyvoice3 as adapter

    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        yield {"tts_speech": torch.tensor([[0.1, 0.2]])}
        yield {"tts_speech": torch.tensor([[0.3]])}

    backend = SimpleNamespace(
        sample_rate=22050,
        inference_zero_shot=generate,
        inference_instruct2=generate,
        model=SimpleNamespace(
            llm=SimpleNamespace(llm=SimpleNamespace(forward_one_step=lambda *args: None))
        ),
    )

    def create(path, **kwargs):
        assert path == "/synthetic/model"
        assert kwargs == {"load_trt": False, "load_vllm": False, "fp16": False}
        return backend

    monkeypatch.setattr(adapter, "_backend_class", lambda: create)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = TTSModelFactory.get_config_for_model("CosyVoice3")
    config.model_name = "/synthetic/model"
    model = TTSModelFactory.create_model("CosyVoice3", config=config)
    result = model.synthesize_audio(
        "target", "reference.wav", "spoken reference", style, 1.25
    )
    assert result.sample_rate == 22050
    np.testing.assert_allclose(result.waveform, [0.1, 0.2, 0.3])
    expected = {
        "tts_text": "target",
        "prompt_wav": "reference.wav",
        "stream": False,
        "speed": 1.25,
        "text_frontend": False,
    }
    if style:
        expected["instruct_text"] = (
            "You are a helpful assistant. Speak calmly.<|endofprompt|>"
        )
    else:
        expected["prompt_text"] = (
            "You are a helpful assistant.<|endofprompt|>spoken reference"
        )
    assert calls == [expected]
    np.testing.assert_allclose(
        model.tts("target", "reference.wav", "spoken reference"), result.waveform
    )


def test_cosy_rejects_unverified_device_before_loading(monkeypatch):
    import TTS.tts.models.cosyvoice3 as adapter

    monkeypatch.setattr(adapter, "_backend_class", lambda: pytest.fail("must not load"))
    with pytest.raises(ValueError):
        CosyVoice3TTS(CosyVoice3Config(device_map="cuda"))


def test_qwen_actual_rate_and_compatibility_waveform(monkeypatch):
    from TTS.tts.models.qwen3_tts import Qwen3TTS, Qwen3TTSConfig

    monkeypatch.setattr(Qwen3TTS, "_load_model", lambda self: None)
    model = Qwen3TTS(Qwen3TTSConfig())
    seen = []
    monkeypatch.setattr(
        model, "inference", lambda **kw: (seen.append(kw) or np.ones(20), 16000)
    )
    monkeypatch.setattr(model, "_apply_speed", lambda wav, speed: wav[:10])
    result = model.synthesize_audio(
        "target", "reference.wav", "reference", speed=1.5, top_k=12
    )
    assert result.sample_rate == 16000 and len(result.waveform) == 10
    assert seen[0]["top_k"] == 12 and "speed" not in seen[0]
    assert (
        len(model.tts("target", speaker_wav="reference.wav", ref_text="reference"))
        == 20
    )


@pytest.mark.parametrize(
    "waveform,rate",
    [([], 24000), ([float("nan")], 24000), ([1], 0), ([[1, 2], [3, 4]], 24000)],
)
def test_invalid_backend_audio_is_rejected(waveform, rate):
    with pytest.raises(ValueError):
        SynthesisAudio(waveform, rate)


def test_official_cosy_methods_do_not_log_private_prompts(caplog):
    from TTS.tts.models.cosyvoice3 import _backend_class

    backend = _backend_class()
    instance = object.__new__(backend)
    instance.sample_rate = 24000
    instance.frontend = SimpleNamespace(
        text_normalize=lambda text, split, **kwargs: [text] if split else text,
        frontend_zero_shot=lambda *args: {},
        frontend_instruct2=lambda *args: {},
    )
    instance.model = SimpleNamespace(
        tts=lambda **kwargs: iter([{"tts_speech": torch.zeros(1, 240)}])
    )
    with caplog.at_level("INFO"):
        list(
            instance.inference_zero_shot(
                "private-target",
                "private-reference" * 20,
                "private.wav",
                text_frontend=False,
            )
        )
        list(
            instance.inference_instruct2(
                "private-target", "private-style", "private.wav", text_frontend=False
            )
        )
    assert "private-" not in caplog.text
