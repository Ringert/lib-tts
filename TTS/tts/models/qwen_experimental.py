"""Request-scoped, offline CPU adapters for experimental Qwen modes."""

import errno
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import soundfile as sf

from TTS.tts.models.shared.audio import SynthesisAudio


class ExperimentalError(Exception):
    """A public error code without private backend diagnostics."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def resource_error(exc: Exception) -> bool:
    return (
        isinstance(exc, MemoryError)
        or type(exc).__name__ == "OutOfMemoryError"
        or (isinstance(exc, OSError) and exc.errno == errno.ENOMEM)
        or (
            isinstance(exc, RuntimeError)
            and "DefaultCPUAllocator: can't allocate memory" in str(exc)
        )
    )


@dataclass
class ExperimentalQwenConfig:
    model_name: str = ""
    device_map: str = "cpu"
    trained: bool = False


class ExperimentalQwenTTS:
    """Keep each model, optimizer and generated artifact in one owned process."""

    def __init__(self, config: ExperimentalQwenConfig):
        self.config = config
        self.last_report: dict = {}

    def synthesize_audio(self, text, speaker_wav, ref_text, **options):
        if self.config.device_map != "cpu":
            raise ExperimentalError("unsupported_device")
        with tempfile.TemporaryDirectory(prefix="live-tts-qwen-worker-") as directory:
            root = Path(directory)
            request = {
                "config": asdict(self.config),
                "text": text,
                "speaker_wav": str(speaker_wav),
                "ref_text": ref_text,
                "options": options,
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request), encoding="utf-8")
            request_path.chmod(0o600)
            env = os.environ.copy()
            env.update(
                HF_HUB_OFFLINE="1",
                TRANSFORMERS_OFFLINE="1",
                HF_HUB_DISABLE_TELEMETRY="1",
                OMP_NUM_THREADS="4",
                MKL_NUM_THREADS="4",
            )
            process = subprocess.Popen(
                [sys.executable, "-m", "TTS.tts.models.qwen_worker", str(root)],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                try:
                    returncode = process.wait(timeout=30 * 60)
                except subprocess.TimeoutExpired as exc:
                    raise ExperimentalError("resource_unavailable") from exc
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()
            if returncode != 0:
                raise ExperimentalError("resource_unavailable")
            try:
                report = json.loads((root / "result.json").read_text())
                if "error" in report:
                    raise ExperimentalError(report["error"])
                waveform, rate = sf.read(root / "result.wav", dtype="float32")
                result = SynthesisAudio(waveform, rate)
            except (OSError, ValueError, KeyError) as exc:
                raise ExperimentalError("synthesis_failed") from exc
            self.last_report = report
            return result
