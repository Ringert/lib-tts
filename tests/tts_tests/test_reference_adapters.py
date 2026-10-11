"""Adapter behavior with a controlled expensive model boundary."""

import subprocess
import sys

import numpy as np
import pytest
from TTS.tts.models.shared.audio import SynthesisAudio


def test_metadata_and_factory_imports_are_model_free():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from TTS.tts.models.tts_factory import TTSModelFactory
from TTS.tts.models.shared.capabilities import capabilities
assert TTSModelFactory.get_model_type("FunAudioLLM/Fun-CosyVoice3-0.5B-2512") == "xtts"
assert TTSModelFactory.get_model_type("historical") == "xtts"
assert capabilities("QWEN").requires_reference_text
assert "cosyvoice3" not in TTSModelFactory.get_supported_models()
try:
    capabilities("COSY")
except ValueError:
    pass
else:
    raise AssertionError("COSY must be unsupported")
try:
    TTSModelFactory.create_model("CosyVoice3")
except ValueError:
    pass
else:
    raise AssertionError("COSY must not load")
assert not any(name in sys.modules for name in ("qwen_tts", "cosyvoice", "whisper", "torch"))
""",
        ],
        check=True,
    )


def test_qwen_actual_rate_and_compatibility_waveform(monkeypatch):
    from TTS.tts.models.qwen3_tts import Qwen3TTS
    from TTS.tts.models.tts_factory import TTSModelFactory

    monkeypatch.setattr(Qwen3TTS, "_load_model", lambda self: None)
    config = TTSModelFactory.get_config_for_model("Qwen3-TTS")
    model = TTSModelFactory.create_model("Qwen3-TTS", config=config)
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
