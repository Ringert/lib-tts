"""One private offline process: load, optionally adapt, generate, then exit."""

import json
import sys
from pathlib import Path

from TTS.tts.models.qwen_experimental import ExperimentalError, resource_error


def run(request):
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    from TTS.tts.models.qwen_head_adaptation import adapt_heads, weight_digest

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    config = request["config"]
    audio, rate = sf.read(request["speaker_wav"], dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if not audio.size or not np.isfinite(audio).all() or len(audio) / rate > 30:
        raise ExperimentalError("invalid_audio")
    if rate != 24000:
        audio = librosa.resample(audio, orig_sr=rate, target_sr=24000)
    try:
        tts = Qwen3TTSModel.from_pretrained(
            config["model_name"],
            device_map="cpu",
            dtype=torch.float32,
            attn_implementation="eager",
            local_files_only=True,
        )
    except (OSError, ValueError) as exc:
        code = "resource_unavailable" if resource_error(exc) else "model_unavailable"
        raise ExperimentalError(code) from exc
    reference_text = request["ref_text"]
    reference_ids = tts._tokenize_texts([tts._build_ref_text(reference_text)])[0]
    if reference_ids.numel() > 1024:
        raise ExperimentalError("reference_text_too_long")
    prompt = tts.create_voice_clone_prompt(
        ref_audio=(audio, 24000),
        ref_text=reference_text,
        x_vector_only_mode=False,
    )
    report = {"mode": "QWEN_TRAINED" if config["trained"] else "QWEN_V2", "steps": 0}
    if config["trained"]:
        try:
            report.update(adapt_heads(tts, prompt, reference_text))
        except Exception as exc:
            code = "resource_unavailable" if resource_error(exc) else "training_failed"
            raise ExperimentalError(code) from exc
    options = request["options"].copy()
    style = options.pop("style_prompt", None)
    language = options.pop("language")
    speed = options.pop("speed")
    instruction = (
        tts._tokenize_texts([tts._build_instruct_text(style)])[0] if style else None
    )
    report["instruction_tokens"] = (
        int(instruction.numel()) if instruction is not None else 0
    )
    report["instruction_consumed"] = False
    report["regular_eos"] = False
    talker = tts.model.talker

    def observe_text(_module, args):
        if instruction is not None and torch.equal(args[0], instruction):
            report["instruction_consumed"] = True

    def observe_heads(_module, _args):
        if config["trained"] and "inference_heads" not in report:
            report["inference_heads"] = weight_digest(tts.model)

    text_hook = talker.get_text_embeddings().register_forward_pre_hook(observe_text)
    head_hook = talker.codec_head.register_forward_pre_hook(observe_heads)
    original_generate = talker.generate

    def observed_generate(*args, **kwargs):
        # Observe only this worker's instance, leaving SDK classes and all other
        # requests unchanged. The SDK otherwise strips EOS before returning audio.
        generated = original_generate(*args, **kwargs)
        report["regular_eos"] = bool(
            (
                generated.sequences[:, -1]
                == tts.model.config.talker_config.codec_eos_token_id
            ).all()
        )
        if not report["regular_eos"]:
            raise ExperimentalError("synthesis_failed")
        return generated

    talker.generate = observed_generate
    try:
        waves, rate = tts.generate_voice_clone(
            text=request["text"],
            language=language,
            voice_clone_prompt=prompt,
            instruct_ids=[instruction] if instruction is not None else None,
            non_streaming_mode=True,
            max_new_tokens=2048,
            **options,
        )
    finally:
        talker.generate = original_generate
        text_hook.remove()
        head_hook.remove()
    if instruction is not None and not report["instruction_consumed"]:
        raise ExperimentalError("synthesis_failed")
    if config["trained"] and report.get("inference_heads") != report["after"]:
        raise ExperimentalError("training_failed")
    waveform = np.asarray(waves[0], dtype=np.float32)
    if speed != 1.0:
        waveform = librosa.effects.time_stretch(waveform, rate=speed)
    from TTS.tts.models.shared.audio import SynthesisAudio

    result = SynthesisAudio(waveform, rate)
    report.update(sample_rate=rate, samples=len(result.waveform))
    return result, report


def main():
    root = Path(sys.argv[1])
    try:
        request = json.loads((root / "request.json").read_text(encoding="utf-8"))
        result, report = run(request)
        import soundfile as sf

        sf.write(
            root / "result.wav", result.waveform, result.sample_rate, subtype="FLOAT"
        )
    except ExperimentalError as exc:
        report = {"error": exc.code}
    except ImportError:
        report = {"error": "backend_unavailable"}
    except Exception as exc:  # noqa: BLE001 - process boundary returns only safe codes
        code = "resource_unavailable" if resource_error(exc) else "synthesis_failed"
        report = {"error": code}
    (root / "result.json").write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
