"""Grammar, literal text binding and actual segment/PCM contracts."""

from contextlib import nullcontext
from threading import RLock
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from TTS.tts.models.cosy_markup import (
    MarkupError,
    Speech,
    SpeechPlan,
    parse_style,
)
from TTS.tts.models.cosyvoice3 import CosyVoice3TTS


def parse(text, body):
    return parse_style(text, "markup:v1:<speech>" + body + "</speech>")


def test_nested_styles_restore_and_emphasis_is_idempotent_without_split():
    result = parse(
        "One two three four five.",
        'One <style name="soft">two <style name="loud">three</style> four</style> <emphasis>fi<emphasis>ve</emphasis></emphasis>.',
    )
    assert result == SpeechPlan(
        (
            Speech("One ", "neutral"),
            Speech("two ", "soft"),
            Speech("three", "loud"),
            Speech(" four", "soft"),
            Speech(" <strong>five</strong>.", "neutral"),
        )
    )


def test_exact_opt_in_escapes_tail_and_outer_trim():
    assert parse("A & B.", " A &amp; <emphasis>B</emphasis>. ") == SpeechPlan(
        (Speech(" A & <strong>B</strong>. ", "neutral"),)
    )
    assert parse_style("x", " markup:v1:broken ") == "markup:v1:broken"
    assert parse_style("x", " calm ") == "calm"
    assert parse_style("x", " ") is None


@pytest.mark.parametrize(
    "body",
    [
        '<style name="unknown">Hello.</style>',
        '<style name="soft" extra="x">Hello.</style>',
        '<emphasis x="y">Hello.</emphasis>',
        "<other>Hello.</other>",
        "<speech>Hello.</speech>",
        "<!--x-->Hello.",
        "<?x y?>Hello.",
        "<!DOCTYPE x>Hello.",
        '<style name="soft"></style>Hello.',
        "<emphasis> </emphasis>Hello.",
        '<pause ms="0"/>Hello.',
        '<pause ms="2001"/>Hello.',
        '<pause ms="+1"/>Hello.',
        '<pause ms="１"/>Hello.',
        '<pause ms="1.0"/>Hello.',
        '<pause ms="1">x</pause>Hello.',
        '<pause ms="1"><emphasis>x</emphasis></pause>Hello.',
        'He<pause ms="1"/>llo.',
        'He<style name="soft">llo</style>.',
        "Hell<lengthen>o</lengthen>.",
        "Hello!",
        "Hello. extra",
        "&lt;strong&gt;Hello.&lt;/strong&gt;",
        "Hello.[breath]",
        '<style xmlns="urn:x" name="soft">Hello.</style>',
    ],
)
def test_invalid_markup_is_safe(body):
    with pytest.raises(MarkupError) as err:
        parse("Hello.", body)
    assert "Hello" not in str(err.value)


@pytest.mark.parametrize(
    "value",
    [
        "markup:v1:<speech>x</speech><speech>x</speech>",
        "markup:v1:<speech><emphasis>x</speech>",
        'markup:v1:<speech a="b">x</speech>',
    ],
)
def test_malformed_and_root_errors(value):
    with pytest.raises(MarkupError):
        parse_style("x", value)


def test_limits():
    assert parse("Hi.", "<emphasis>" * 7 + "Hi." + "</emphasis>" * 7)
    with pytest.raises(MarkupError):
        parse("Hi.", "<emphasis>" * 8 + "Hi." + "</emphasis>" * 8)
    assert parse("Hi.", '<pause ms="2000"/><pause ms="2000"/><pause ms="1000"/>Hi.')
    with pytest.raises(MarkupError):
        parse("Hi.", '<pause ms="2000"/>' * 3 + "Hi.")
    with pytest.raises(MarkupError):
        parse("Hi.", '<pause ms="1"/>' * 13 + "Hi.")
    assert parse("Hi.", '<pause ms="1"/>' * 12 + "Hi.")
    with pytest.raises(MarkupError):
        parse(" ".join(["Hi"] * 13), '<pause ms="1"/>'.join(["Hi "] * 13))
    with pytest.raises(MarkupError):
        parse_style("x", "x" * 1001)


def test_plan_audio_has_exact_pause_positions_and_local_prompts():
    calls = []

    def inference(**kwargs):
        calls.append(kwargs)
        yield {"tts_speech": torch.ones(1, 100)}

    model = object.__new__(CosyVoice3TTS)
    model._synthesis_lock = RLock()
    model._markup_decoder = SimpleNamespace(segment=nullcontext)
    model.model = SimpleNamespace(sample_rate=1000, inference_instruct2=inference)
    result = model.synthesize_audio(
        "One two three.",
        "same.wav",
        style_prompt='markup:v1:<speech>One<pause ms="150"/> <style name="soft"><emphasis>two</emphasis></style><pause ms="150"/> three.</speech>',
    )
    assert len(calls) == 3
    assert [c["tts_text"].strip() for c in calls] == [
        "One",
        "<strong>two</strong>",
        "three.",
    ]
    assert all(c["prompt_wav"] == "same.wav" and c["speed"] == 1.0 for c in calls)
    assert (
        "neutral" in calls[0]["instruct_text"]
        and "soft" in calls[1]["instruct_text"]
        and "neutral" in calls[2]["instruct_text"]
    )
    assert result.sample_rate == 1000 and len(result.waveform) == 600
    np.testing.assert_array_equal(result.waveform[100:250], 0)
    np.testing.assert_array_equal(result.waveform[350:500], 0)
    assert result.waveform[50] == result.waveform[300] == result.waveform[550] == 1


def test_direct_invalid_markup_never_calls_backend():
    model = object.__new__(CosyVoice3TTS)
    model._synthesis_lock = RLock()
    model._markup_decoder = SimpleNamespace(segment=nullcontext)
    with pytest.raises(MarkupError):
        model.synthesize_audio("Hi", "ref.wav", style_prompt="markup:v1:broken")


def test_segment_failure_returns_no_partial_audio():
    calls = []

    def inference(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("backend failed")
        yield {"tts_speech": torch.ones(1, 100)}

    model = object.__new__(CosyVoice3TTS)
    model._synthesis_lock = RLock()
    model._markup_decoder = SimpleNamespace(segment=nullcontext)
    model.model = SimpleNamespace(sample_rate=1000, inference_instruct2=inference)
    with pytest.raises(RuntimeError):
        model.synthesize_audio(
            "One two.",
            "ref.wav",
            style_prompt='markup:v1:<speech>One <style name="soft">two.</style></speech>',
        )


def test_escaped_literal_text_is_not_confused_with_native_controls():
    assert parse("A < B > C [note]", "A &lt; B &gt; C [note]") == SpeechPlan(
        (Speech("A < B > C [note]", "neutral"),)
    )
    with pytest.raises(MarkupError):
        parse("[breath]", "[bre<emphasis>ath</emphasis>]")
