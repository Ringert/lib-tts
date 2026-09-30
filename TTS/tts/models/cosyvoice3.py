"""Reference and instruction synthesis using the pinned official CosyVoice3."""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .shared.audio import SynthesisAudio


@dataclass
class CosyVoice3Config:
    model_name: str = "FunAudioLLM/Fun-CosyVoice3-0.5B-2512"
    device_map: str = "cpu"


class _PrivateBackendLogger:
    """Upstream interpolates private prompts into log messages. Never emit them.

    Installed once on the Cosy modules, without changing process logging levels
    or handlers during requests. Backend exceptions still reach the caller.
    """

    def __getattr__(self, name):
        if name in {"debug", "info", "warning", "error", "exception"}:
            return lambda *args, **kwargs: None
        return getattr(logging, name)


def _backend_class():
    root = Path(__file__).resolve().parents[3] / "third_party" / "CosyVoice"
    matcha = root / "third_party" / "Matcha-TTS"
    if not (root / "cosyvoice").is_dir() or not (matcha / "matcha").is_dir():
        raise ImportError("Initialize the pinned CosyVoice and Matcha submodules")
    for path in (root, matcha):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import cosyvoice.cli.cosyvoice as backend
    from cosyvoice.cli import frontend

    backend.logging = frontend.logging = _PrivateBackendLogger()
    return backend.CosyVoice3


class CosyVoice3TTS:
    def __init__(self, config: CosyVoice3Config):
        if config.device_map != "cpu":
            raise ValueError(
                "CosyVoice3 currently supports the verified CPU configuration only"
            )
        import torch

        # Upstream selects CUDA globally rather than accepting a device argument.
        if torch.cuda.is_available():
            raise ValueError("CosyVoice3 CPU runtime requires CUDA to be unavailable")
        self.config = config
        self.model = _backend_class()(
            config.model_name, load_trt=False, load_vllm=False, fp16=False
        )

    def synthesize_audio(
        self, text, speaker_wav=None, ref_text=None, style_prompt=None, speed=1.0
    ):
        if not speaker_wav:
            raise ValueError("CosyVoice3 requires reference audio")
        options = {
            "tts_text": text,
            "prompt_wav": speaker_wav,
            "stream": False,
            "speed": speed,
            "text_frontend": False,
        }
        if style_prompt:
            chunks = self.model.inference_instruct2(
                instruct_text="You are a helpful assistant. "
                + style_prompt
                + "<|endofprompt|>",
                **options,
            )
        else:
            if not isinstance(ref_text, str) or not ref_text.strip():
                raise ValueError(
                    "Reference text is required without a style instruction"
                )
            chunks = self.model.inference_zero_shot(
                prompt_text="You are a helpful assistant.<|endofprompt|>" + ref_text,
                **options,
            )
        waveforms = [
            chunk["tts_speech"].detach().cpu().numpy().reshape(-1) for chunk in chunks
        ]
        if not waveforms:
            raise ValueError("CosyVoice3 returned no audio")
        return SynthesisAudio(np.concatenate(waveforms), self.model.sample_rate)

    def tts(self, *args, **kwargs):
        return self.synthesize_audio(*args, **kwargs).waveform
