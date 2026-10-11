"""Private offline library worker. Publication belongs exclusively to the parent."""

import importlib.metadata
import json
import os
import sys
import threading
from pathlib import Path

from TTS.tts.models.qwen_experimental import ExperimentalError, resource_error
from TTS.tts.models.qwen_voice_format import (
    binding,
    digest,
    validate_artifacts,
    save_artifacts,
    load_artifacts,
)


def run(request):
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel
    from TTS.tts.models.qwen_head_adaptation import adapt_collection
    from TTS.tts.models.qwen_worker import generate

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model_root = request["config"]["model_name"]
    root = Path(request["version_root"])
    expected = binding(model_root)
    manifest = request.get("manifest", {})
    if request["action"] == "synthesize_trained":
        manifest = json.loads((root / "manifest.json").read_text())
        validate_artifacts(root, manifest, expected)
    tts = Qwen3TTSModel.from_pretrained(
        model_root,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
        local_files_only=True,
    )
    if request["action"] == "synthesize_trained":
        prompt = load_artifacts(root, tts, manifest)
        report = {
            "mode": "QWEN_TRAINED",
            "steps": 0,
            "after": manifest["training"]["after"],
        }
        result, report = generate(tts, prompt, request, report)
        sf.write(
            request["result_wav"], result.waveform, result.sample_rate, subtype="FLOAT"
        )
        return report
    if request["action"] != "train_voice":
        raise ExperimentalError("training_failed")
    manifest = request["manifest"]
    prompts, texts = [], []
    for index, ref in enumerate(manifest["references"]):
        text = (root / ref["text"]).read_text(encoding="utf-8")
        tokens = tts._tokenize_texts([tts._build_ref_text(text)])[0].numel()
        if tokens > 1024:
            error = ExperimentalError("reference_text_too_long")
            error.reference_index = index
            raise error
        ref.update(text_tokens=tokens, x_vector_only_mode=False, icl_mode=True)
        audio, rate = sf.read(root / ref["wav"], dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if not audio.size or not np.isfinite(audio).all() or len(audio) / rate > 30:
            raise ExperimentalError("invalid_audio")
        if rate != 24000:
            audio = librosa.resample(audio, orig_sr=rate, target_sr=24000)
        prompts.append(
            tts.create_voice_clone_prompt(
                ref_audio=(audio, 24000), ref_text=text, x_vector_only_mode=False
            )
        )
        texts.append(text)
    training = adapt_collection(tts, list(zip(prompts, texts)))
    manifest.update(
        binding=expected,
        training=training,
        model_root=model_root,
        provenance={
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "fork": importlib.metadata.version("TTS"),
            "fork_sources": {
                name: digest(Path(__file__).with_name(name))
                for name in (
                    "qwen_library_worker.py",
                    "qwen_voice_format.py",
                    "qwen_head_adaptation.py",
                    "qwen_worker.py",
                )
            },
            "device": "cpu",
            "dtype": "float32",
        },
        optimizer={
            "name": "AdamW",
            "learning_rate": 2e-5,
            "weight_decay": 0,
            "clip_norm": 1.0,
            "residual_loss_weight": 0.3,
        },
    )
    save_artifacts(root, tts, prompts, manifest)
    return {
        "steps": training["steps"],
        "pairs": training["pairs"],
        "after": training["after"],
    }


def main():
    # A hard parent exit closes its write end. This process owns no shared service.
    fd = int(sys.argv[2])

    def parent_watch():
        try:
            while os.read(fd, 1):
                pass
        finally:
            os._exit(1)

    threading.Thread(target=parent_watch, daemon=True).start()
    root = Path(sys.argv[1])
    os.umask(0o077)
    report: dict = {}
    request = {"action": "train_voice"}
    try:
        request = json.loads((root / "request.json").read_text())
        report = run(request)
    except ExperimentalError as exc:
        report = {"error": exc.code}
        if exc.reference_index is not None:
            report["reference_index"] = exc.reference_index
    except ImportError:
        report = {"error": "backend_unavailable"}
    except Exception as exc:
        report = {
            "error": "resource_unavailable"
            if resource_error(exc)
            else "training_failed"
            if request["action"] == "train_voice"
            else "synthesis_failed"
        }
    (root / "result.json").write_text(json.dumps(report))


if __name__ == "__main__":
    main()
