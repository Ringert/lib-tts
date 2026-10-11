"""Version-1 tensor contract. No model loading, pickle or implicit conversion."""

import hashlib
import importlib.metadata
import json
from pathlib import Path

from TTS.tts.models.qwen_experimental import ExperimentalError

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REVISION = "fd4b254389122332181a7c3db7f27e918eec64e3"
RECIPE = "head-output-v1"
HEAD_NAMES = ["talker.codec_head.weight"] + [
    f"talker.code_predictor.lm_head.{i}.weight" for i in range(15)
]
HEAD_SHAPES = [[3072, 2048]] + [[2048, 1024]] * 15


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def binding(model_root):
    root = Path(model_root)
    names = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "speech_tokenizer/config.json",
        "speech_tokenizer/preprocessor_config.json",
    ]
    fingerprints = {
        name: digest(root / name) for name in names if (root / name).is_file()
    }
    if (
        not {"config.json", "tokenizer_config.json", "speech_tokenizer/config.json"}
        <= fingerprints.keys()
    ):
        raise ExperimentalError("model_unavailable")
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "qwen_tts": importlib.metadata.version("qwen-tts"),
        "fingerprints": fingerprints,
    }


def tensor_metadata(path):
    from safetensors import safe_open

    result = {}
    with safe_open(str(path), framework="numpy") as source:
        for name in source.keys():
            tensor = source.get_tensor(name)
            dtype = {"float32": "F32", "int64": "I64"}.get(
                str(tensor.dtype), str(tensor.dtype)
            )
            result[name] = {"shape": list(tensor.shape), "dtype": dtype}
    return result


def validate_artifacts(root, manifest, expected_binding=None):
    """Validate actual bytes and every tensor before a version is used/published."""
    import numpy as np
    from safetensors import safe_open, SafetensorError

    root = Path(root)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("binding"), dict):
        raise ExperimentalError("trained_version_corrupt")
    if (
        manifest.get("format_version") != 1
        or manifest.get("adaptation_recipe") != RECIPE
        or manifest.get("binding", {}).get("model_id") != MODEL_ID
        or manifest.get("binding", {}).get("revision") != REVISION
        or manifest.get("binding", {}).get("qwen_tts") != "0.1.1"
        or manifest.get("conditioning_reference_index") != 0
        or (expected_binding is not None and manifest["binding"] != expected_binding)
    ):
        raise ExperimentalError("trained_version_incompatible")
    try:
        required = {
            "format_version",
            "adaptation_recipe",
            "voice_id",
            "version_id",
            "number",
            "parent_version_id",
            "created_at",
            "operation_id",
            "references",
            "conditioning_reference_index",
            "binding",
            "training",
            "provenance",
            "optimizer",
            "artifacts",
        }
        if set(manifest) != required or manifest["optimizer"] != {
            "name": "AdamW",
            "learning_rate": 2e-5,
            "weight_decay": 0,
            "clip_norm": 1.0,
            "residual_loss_weight": 0.3,
        }:
            raise ExperimentalError("trained_version_incompatible")
        if (
            manifest["provenance"]["device"] != "cpu"
            or manifest["provenance"]["dtype"] != "float32"
        ):
            raise ExperimentalError("trained_version_incompatible")
        references = manifest["references"]
        if not 1 <= len(references) <= 16:
            raise ValueError()
        expected = {
            name: {"shape": shape, "dtype": "F32"}
            for name, shape in zip(HEAD_NAMES, HEAD_SHAPES)
        }
        head_digest = hashlib.sha256()
        for filename in ("heads.safetensors", "prompts.safetensors"):
            entry = manifest["artifacts"][filename]
            if digest(root / filename) != entry["sha256"]:
                raise ValueError()
            actual = tensor_metadata(root / filename)
            if actual != entry["tensors"] or (
                filename == "heads.safetensors" and actual != expected
            ):
                raise ExperimentalError("trained_version_incompatible")
            keys = (
                HEAD_NAMES
                if filename == "heads.safetensors"
                else [
                    f"reference.{i}.{key}"
                    for i in range(len(references))
                    for key in ("ref_code", "ref_spk_embedding")
                ]
            )
            if set(actual) != set(keys):
                raise ValueError()
            with safe_open(str(root / filename), framework="numpy") as source:
                for name in keys:
                    tensor = source.get_tensor(name)
                    if not np.isfinite(tensor).all():
                        raise ValueError()
                    if filename == "heads.safetensors":
                        head_digest.update(tensor.tobytes())
                    elif name.endswith("ref_code"):
                        if (
                            tensor.dtype != np.int64
                            or tensor.ndim != 2
                            or tensor.shape[1] != 16
                            or not 1 <= tensor.shape[0] <= 400
                            or np.any(tensor < 0)
                            or np.any(tensor >= 2048)
                        ):
                            raise ValueError()
                    elif tensor.dtype != np.float32 or tensor.shape != (2048,):
                        raise ValueError()
        if head_digest.hexdigest() != manifest["training"]["after"]:
            raise ValueError()
        training = manifest["training"]
        if (
            training["before"] == training["after"]
            or training["trainable_parameters"] != 37748736
            or training["steps"] != 8
            or training["pairs"] != len(references)
            or len(training["losses"]) != 8
            or not np.isfinite(training["losses"]).all()
        ):
            raise ValueError()
        if sum(ref["duration"] for ref in references) > 120:
            raise ValueError()
        for i, ref in enumerate(references):
            if (
                not isinstance(ref["duration"], (int, float))
                or not np.isfinite(ref["duration"])
                or not 0 < ref["duration"] <= 30
                or ref["text_source"] not in {"provided", "local_asr"}
                or ref["index"] != i
                or ref["wav"] != f"references/{i + 1:04d}.wav"
                or ref["text"] != f"references/{i + 1:04d}.txt"
                or ref["x_vector_only_mode"] is not False
                or ref["icl_mode"] is not True
                or not 1 <= ref["text_tokens"] <= 1024
            ):
                raise ValueError()
            if (
                digest(root / ref["wav"]) != ref["wav_sha256"]
                or digest(root / ref["text"]) != ref["text_sha256"]
            ):
                raise ValueError()
    except ExperimentalError:
        raise
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        OverflowError,
        SafetensorError,
    ) as exc:
        raise ExperimentalError("trained_version_corrupt") from exc


def save_artifacts(root, tts, prompts, manifest):
    from safetensors.torch import save_file
    from TTS.tts.models.qwen_head_adaptation import output_weights

    root = Path(root)
    heads = {
        name: weight.detach().cpu().contiguous()
        for name, weight in zip(HEAD_NAMES, output_weights(tts.model))
    }
    conditioning = {
        f"reference.{i}.{key}": getattr(prompt[0], key).detach().cpu().contiguous()
        for i, prompt in enumerate(prompts)
        for key in ("ref_code", "ref_spk_embedding")
    }
    save_file(heads, str(root / "heads.safetensors"))
    save_file(conditioning, str(root / "prompts.safetensors"))
    manifest["artifacts"] = {
        name: {"sha256": digest(root / name), "tensors": tensor_metadata(root / name)}
        for name in ("heads.safetensors", "prompts.safetensors")
    }
    validate_artifacts(root, manifest, binding(manifest.pop("model_root")))
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )


def load_artifacts(root, tts, manifest):
    import torch
    from safetensors.torch import load_file
    from qwen_tts.inference.qwen3_tts_model import VoiceClonePromptItem
    from TTS.tts.models.qwen_head_adaptation import output_weights, weight_digest

    root = Path(root)
    heads = load_file(str(root / "heads.safetensors"))
    with torch.no_grad():
        for name, weight in zip(HEAD_NAMES, output_weights(tts.model)):
            tensor = heads[name]
            if tensor.shape != weight.shape or tensor.dtype != weight.dtype:
                raise ExperimentalError("trained_version_incompatible")
            weight.copy_(tensor)
    if weight_digest(tts.model) != manifest["training"]["after"]:
        raise ExperimentalError("trained_version_corrupt")
    tts.model.eval().requires_grad_(False)
    tensors = load_file(str(root / "prompts.safetensors"))
    ref = manifest["references"][0]
    return [
        VoiceClonePromptItem(
            ref_code=tensors["reference.0.ref_code"],
            ref_spk_embedding=tensors["reference.0.ref_spk_embedding"],
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text=(root / ref["text"]).read_text(encoding="utf-8"),
        )
    ]
