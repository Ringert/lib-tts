"""The supported request options for the reference synthesis backend."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Capabilities:
    options: frozenset[str]
    requires_reference_text: bool


SAMPLING_OPTIONS = frozenset({"top_p", "top_k", "temperature", "repetition_penalty"})
MODEL_OPTIONS = SAMPLING_OPTIONS | {
    "language",
    "speed",
    "pitch",
    "length_penalty",
    "gpt_cond_len",
    "gpt_cond_chunk_len",
    "max_ref_len",
    "sound_norm_refs",
    "style_prompt",
}


def capabilities(modus: str, style_prompt: str | None = None) -> Capabilities:
    if modus == "QWEN":
        return Capabilities(SAMPLING_OPTIONS | {"language", "speed"}, True)
    if modus in {"QWEN_V2", "QWEN_TRAINED"}:
        return Capabilities(SAMPLING_OPTIONS | {"language", "speed", "style_prompt"}, True)
    raise ValueError("Unsupported synthesis mode")
