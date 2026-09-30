"""
Qwen3-TTS model implementation for Live TTS.

Supports voice cloning with 3-second reference audio snippets.
Reference: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
"""

from typing import Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
from coqpit import Coqpit

from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.models.shared.audio import SynthesisAudio


class Qwen3TTSConfig(Coqpit):
    """Qwen3-TTS model configuration."""

    # Model specification
    model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    device_map: str = "cpu"  # "cpu", "cuda", or "cuda:0" - Will be set from service config
    dtype: str = "bfloat16"  # torch.bfloat16 or torch.float32
    attn_implementation: str = "eager"  # "flash_attention_2" or "eager" (eager is safer, flash requires flash-attn package)
    
    # Inference parameters
    temperature: float = 0.6  # Generation temperature
    top_p: float = 0.8  # Nucleus sampling
    max_new_tokens: int = 2048  # Maximum generation length
    
    # Voice cloning parameters
    x_vector_only_mode: bool = False  # Use only speaker embedding without ref_text
    ref_audio_duration: float = 3.0  # Reference audio length in seconds (3-5 sec recommended)
    
    # Language support (10 major languages)
    supported_languages: List[str] = [
        "Chinese", "English", "Japanese", "Korean",
        "German", "French", "Russian", "Portuguese",
        "Spanish", "Italian"
    ]
    
    # Sample rate (12Hz codec)
    sample_rate: int = 24000


class Qwen3TTS(BaseTTS):
    """
    Qwen3-TTS model for text-to-speech synthesis with voice cloning.
    
    Features:
    - Universal end-to-end architecture
    - Streaming and non-streaming generation
    - Voice cloning with 3-second reference audio
    - 10 major languages support
    - Extreme low-latency (97ms end-to-end)
    """

    def __init__(self, config: Qwen3TTSConfig, ap=None, tokenizer=None, **kwargs):
        """
        Initialize Qwen3-TTS model.
        
        Args:
            config: Qwen3TTSConfig instance
            ap: AudioProcessor (unused for Qwen3-TTS, kept for BaseTTS compatibility)
            tokenizer: TTSTokenizer (unused for Qwen3-TTS, kept for BaseTTS compatibility)
            **kwargs: Additional arguments passed to BaseTTS
        """
        # Bypass BaseTTS.__init__ to avoid num_chars requirement
        # Call nn.Module.__init__ directly
        import torch.nn as nn
        nn.Module.__init__(self)
        
        self.config = config
        self.ap = ap
        self.tokenizer = tokenizer
        self.speaker_manager = kwargs.get("speaker_manager")
        self.language_manager = kwargs.get("language_manager")
        
        self.model = None
        self._voice_clone_prompt = None
        self._load_model()

    def _set_model_args(self, config: Qwen3TTSConfig):
        """Override BaseTTS._set_model_args() - not needed for Qwen3-TTS."""
        # Qwen3-TTS has its own config structure, skip BaseTTS setup
        pass

    def _load_model(self) -> None:
        """Load Qwen3-TTS model from Hugging Face.
        
        Downloads model automatically on first use (similar to XTTS behavior).
        Models are cached in HuggingFace cache directory (~/.cache/huggingface/).
        """
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            raise ImportError(
                "\n"
                "═══════════════════════════════════════════════════════════════════\n"
                "  Qwen3-TTS requires additional dependencies.\n"
                "  Please install: pip install qwen-tts\n"
                "\n"
                "  Installation:\n"
                "    pip install -U qwen-tts\n"
                "\n"
                "  Optional (for better performance with CUDA):\n"
                "    pip install -U flash-attn --no-build-isolation\n"
                "═══════════════════════════════════════════════════════════════════\n"
            )

        try:
            # Convert dtype string to torch dtype
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

            print(f"\n> Downloading Qwen3-TTS model: {self.config.model_name}")
            print(f"> This may take a while on first run...")
            print(f"> Model will be cached in: ~/.cache/huggingface/hub/")
            
            # Load model - automatically downloads from HuggingFace if not cached
            self.model = Qwen3TTSModel.from_pretrained(
                self.config.model_name,
                device_map=self.config.device_map,
                dtype=dtype,
                attn_implementation=self.config.attn_implementation,
            )

            print(f"✅ Qwen3-TTS model loaded successfully: {self.config.model_name}")
            print(f"   Device: {self.config.device_map}")
            print(f"   Dtype: {self.config.dtype}")

        except Exception as e:
            raise RuntimeError(
                f"\n"
                f"═══════════════════════════════════════════════════════════════════\n"
                f"  Failed to load Qwen3-TTS model: {e}\n"
                f"\n"
                f"  Troubleshooting:\n"
                f"  1. Check internet connection (model downloads from HuggingFace)\n"
                f"  2. Verify model name: {self.config.model_name}\n"
                f"  3. Check available disk space (~3-5 GB needed)\n"
                f"  4. Try: pip install --upgrade qwen-tts\n"
                f"═══════════════════════════════════════════════════════════════════\n"
            ) from e

    def inference(
        self,
        text: str,
        ref_audio: Union[str, np.ndarray, Tuple[np.ndarray, int]],
        ref_text: str,
        language: str = "English",
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text using voice cloning.
        
        Args:
            text: Input text to synthesize
            ref_audio: Reference audio for voice cloning
                - File path (str): "path/to/audio.wav"
                - URL (str): "https://..."
                - Base64 string (str): "data:audio/wav;base64,..."
                - Numpy array with sample rate: (wav, sr) tuple
            ref_text: Transcript of the reference audio (or empty if x_vector_only_mode=True)
            language: Language code or name (default: "English")
            **kwargs: Additional generation parameters
        
        Returns:
            Tuple of (waveform, sample_rate)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        try:
            # Build generation kwargs
            gen_kwargs = {
                "max_new_tokens": self.config.max_new_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            }
            gen_kwargs.update(kwargs)

            # Generate voice clone prompt if needed
            if not hasattr(self, "_voice_clone_prompt") or self._voice_clone_prompt is None:
                voice_clone_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=self.config.x_vector_only_mode,
                )
            else:
                voice_clone_prompt = self._voice_clone_prompt

            # Generate audio
            wavs, sr = self.model.generate_voice_clone(
                text=text,
                language=language,
                voice_clone_prompt=voice_clone_prompt,
                **gen_kwargs
            )

            # Return first waveform and sample rate
            return wavs[0], sr

        except Exception as e:
            raise RuntimeError(f"Voice clone generation failed: {e}") from e

    def synthesize_audio(
        self,
        text: str,
        speaker_wav: Optional[str] = None,
        ref_text: Optional[str] = None,
        language: str = "English",
        speed: float = 1.0,
        **kwargs
    ) -> SynthesisAudio:
        """
        Synthesize speech from text with optional voice cloning.
        
        Args:
            text: Input text
            speaker_wav: Optional reference audio for voice cloning
            ref_text: Transcript of reference audio (required if speaker_wav is provided)
            language: Language code
            speed: Speech speed multiplier (adjusts output length)
            **kwargs: Additional generation parameters
        
        Returns:
            Waveform and the actual backend sample rate
        """
        if not speaker_wav:
            raise ValueError("Qwen3-TTS requires speaker_wav for voice cloning")

        if not ref_text:
            raise ValueError("ref_text (transcript of speaker_wav) is required for voice cloning")

        wav, sample_rate = self.inference(
            text=text,
            ref_audio=speaker_wav,
            ref_text=ref_text,
            language=language,
            **kwargs
        )

        # Apply speed adjustment if needed
        if speed != 1.0:
            wav = self._apply_speed(wav, speed)

        return SynthesisAudio(wav, sample_rate)

    def synthesize(self, *args, **kwargs) -> np.ndarray:
        """Compatibility waveform interface over the shared audio contract."""
        return self.synthesize_audio(*args, **kwargs).waveform

    def tts(
        self,
        text: str,
        language: str = "English",
        speaker_wav: Optional[str] = None,
        ref_text: Optional[str] = None,
        **kwargs
    ) -> np.ndarray:
        """
        TTS interface compatible with live-tts service.
        
        Args:
            text: Text to synthesize
            language: Language code
            speaker_wav: Reference audio for voice cloning
            ref_text: Transcript of reference audio
            **kwargs: Additional parameters
        
        Returns:
            Audio waveform as numpy array
        """
        return self.synthesize(
            text=text,
            speaker_wav=speaker_wav,
            ref_text=ref_text,
            language=language,
            **kwargs
        )

    def set_voice_clone_prompt(self, ref_audio: Union[str, np.ndarray, Tuple], ref_text: str):
        """
        Set a reusable voice clone prompt for multiple generations.
        
        This avoids recomputing prompt features for the same speaker.
        
        Args:
            ref_audio: Reference audio file or numpy array
            ref_text: Transcript of reference audio
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._voice_clone_prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref_audio,
            ref_text=ref_text,
            x_vector_only_mode=self.config.x_vector_only_mode,
        )

    def clear_voice_clone_prompt(self):
        """Clear cached voice clone prompt."""
        self._voice_clone_prompt = None

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.config.supported_languages

    def get_supported_speakers(self) -> List[str]:
        """
        Get list of supported speakers.
        
        Note: Qwen3-TTS Base model supports custom voice cloning,
        not predefined speakers like CustomVoice model.
        """
        return ["<custom_voice_cloning>"]

    @staticmethod
    def _apply_speed(wav: np.ndarray, speed: float) -> np.ndarray:
        """
        Apply speed adjustment to waveform.
        
        Args:
            wav: Input waveform
            speed: Speed multiplier (1.0 = normal, <1.0 = slower, >1.0 = faster)
        
        Returns:
            Speed-adjusted waveform
        """
        if speed == 1.0:
            return wav

        try:
            import librosa
            return librosa.effects.time_stretch(wav, rate=speed)
        except ImportError:
            warnings.warn("librosa not available, skipping speed adjustment")
            return wav

    def forward(self, *args, **kwargs):
        """Forward pass placeholder."""
        raise NotImplementedError("Use inference() or tts() method instead")

    @staticmethod
    def init_from_config(config: "Qwen3TTSConfig", **kwargs):
        """
        Initialize model from config.
        
        Args:
            config: Qwen3TTSConfig instance
            **kwargs: Additional arguments
            
        Returns:
            Qwen3TTS model instance
        """
        return Qwen3TTS(config, **kwargs)

    def load_checkpoint(
        self,
        config,
        checkpoint_dir=None,
        checkpoint_path=None,
        vocab_path=None,
        eval=True,
        strict=True,
        use_deepspeed=False,
        **kwargs
    ):
        """
        Load checkpoint from disk.
        
        Qwen3-TTS models are loaded from Hugging Face Hub automatically,
        so this method is a no-op placeholder for BaseTTS compatibility.
        
        Args:
            config: Model configuration
            checkpoint_dir: Directory with checkpoint (unused for Qwen3)
            checkpoint_path: Path to checkpoint file (unused for Qwen3)
            vocab_path: Path to vocabulary (unused for Qwen3)
            eval: Set model to eval mode after loading
            strict: Strict checkpoint loading (unused for Qwen3)
            use_deepspeed: Use DeepSpeed (unused for Qwen3)
            **kwargs: Additional arguments
        """
        # Qwen3-TTS loads from HuggingFace automatically
        # Model is already loaded in __init__ via from_pretrained()
        if eval:
            self.eval()
        return

    def eval(self):  # pylint: disable=redefined-builtin
        """Sets the model to evaluation mode. Overrides the default eval() method to also set the GPT model to eval mode."""
        if self.model is not None:
            self.model.eval()
        return self

    def to(self, device):
        """Move model to device."""
        if self.model is not None:
            self.model.to(device)
        return self
