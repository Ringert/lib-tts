"""Reference and instruction synthesis using the pinned official CosyVoice3."""

import logging
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import numpy as np

from .cosy_decoding import MarkupDecoder, preserve_minimum_tokens
from .cosy_language import instruction_prompt, normalize_language
from .cosy_markup import STYLE_PROMPTS, Pause, SpeechPlan, parse_style
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


def _preserve_cached_attention(encoder):
    """Adapt only this encoder to Transformers' full key-length mask contract."""
    forward_one_step = encoder.forward_one_step

    def forward_with_cache(xs, masks, cache=None):
        # Pinned CosyVoice supplies only the current query's triangular mask.
        # Transformers 4.57 pads absent cached positions as masked-out keys.
        # Already complete masks (including their padding) must stay untouched.
        if cache is not None and masks.shape[-1] == xs.shape[1]:
            import torch

            past_length = cache.get_seq_length()
            masks = torch.cat(
                (masks.new_ones((*masks.shape[:-1], past_length)), masks), dim=-1
            )
        return forward_one_step(xs, masks, cache)

    encoder.forward_one_step = forward_with_cache


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
        _preserve_cached_attention(self.model.model.llm.llm)
        preserve_minimum_tokens(self.model.model.llm)
        self._synthesis_lock = RLock()
        self._markup_decoder = MarkupDecoder(self.model.model.llm)

    def synthesize_audio(
        self,
        text,
        speaker_wav=None,
        ref_text=None,
        style_prompt=None,
        speed=1.0,
        language=None,
    ):
        with self._synthesis_lock:
            return self._synthesize_audio_locked(
                text,
                speaker_wav,
                ref_text,
                style_prompt,
                speed,
                normalize_language(language),
            )

    def _synthesize_audio_locked(
        self, text, speaker_wav, ref_text, style_prompt, speed, language
    ):
        plan = parse_style(text, style_prompt)
        if not speaker_wav:
            raise ValueError("CosyVoice3 requires reference audio")
        if isinstance(plan, SpeechPlan):
            return self._synthesize_plan(plan, speaker_wav, ref_text, speed, language)
        return self._synthesize_segment(
            text, speaker_wav, ref_text, plan, speed, language
        )

    def _synthesize_plan(self, plan, speaker_wav, ref_text, speed, language):
        pieces = []
        rate = None
        # Synthesize before assembly so pauses use the actual returned rate.
        for segment in plan.segments:
            if isinstance(segment, Pause):
                pieces.append(segment)
                continue
            instruction = STYLE_PROMPTS[segment.style]
            if (
                segment.style == "neutral"
                and not segment.explicit_style
                and isinstance(ref_text, str)
                and ref_text.strip()
            ):
                instruction = None
            audio = self._synthesize_segment(
                segment.text,
                speaker_wav,
                ref_text,
                instruction,
                speed,
                language,
                markup=True,
            )
            if rate is not None and rate != audio.sample_rate:
                raise ValueError("Inconsistent segment sample rates")
            rate = audio.sample_rate
            wave = audio.waveform.copy()
            ramp = min(round(rate * 0.005), len(wave) // 2)
            if ramp:
                gain = np.linspace(0.0, 1.0, ramp, dtype=np.float32)
                wave[:ramp] *= gain
                wave[-ramp:] *= gain[::-1]
            pieces.append(wave)
        waveforms = [
            np.zeros(round(rate * piece.ms / 1000), dtype=np.float32)
            if isinstance(piece, Pause)
            else piece
            for piece in pieces
        ]
        return SynthesisAudio(np.concatenate(waveforms), rate)

    def _synthesize_segment(
        self,
        text,
        speaker_wav,
        ref_text,
        style_prompt,
        speed,
        language,
        *,
        markup=False,
    ):
        context = self._markup_decoder.segment() if markup else nullcontext()
        with context:
            return self._generate_audio(
                text, speaker_wav, ref_text, style_prompt, speed, language
            )

    def _generate_audio(
        self, text, speaker_wav, ref_text, style_prompt, speed, language
    ):
        options = {
            "tts_text": text,
            "prompt_wav": speaker_wav,
            "stream": False,
            "speed": speed,
            "text_frontend": False,
        }
        if isinstance(ref_text, str) and ref_text.strip():
            chunks = self.model.inference_zero_shot(
                prompt_text=instruction_prompt(language, style_prompt) + ref_text,
                **options,
            )
        elif style_prompt:
            chunks = self.model.inference_instruct2(
                instruct_text=instruction_prompt(language, style_prompt),
                **options,
            )
        else:
            raise ValueError("Reference text is required without a style instruction")
        waveforms = [
            chunk["tts_speech"].detach().cpu().numpy().reshape(-1) for chunk in chunks
        ]
        if not waveforms:
            raise ValueError("CosyVoice3 returned no audio")
        return SynthesisAudio(np.concatenate(waveforms), self.model.sample_rate)

    def tts(self, *args, **kwargs):
        return self.synthesize_audio(*args, **kwargs).waveform
