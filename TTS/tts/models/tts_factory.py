"""
TTS Model Factory - Abstraction for switching between different TTS models.

Supports:
- XTTS v2 (default)
- Qwen3-TTS Base (voice cloning)
"""

from typing import Optional, Union, Any
import logging

logger = logging.getLogger(__name__)


class TTSModelFactory:
    """Factory for creating and managing TTS models."""

    SUPPORTED_MODELS = {
        "cosyvoice3": {"description": "CosyVoice3 reference and instruction synthesis", "use_api": False},
        "xtts": {
            "class": None,  # Loaded dynamically via TTS.api
            "config_class": None,
            "description": "XTTS v2 - Multilingual TTS with voice cloning",
            "use_api": True,  # Use TTS API wrapper
        },
        "qwen3": {
            "class": None,  # Loaded dynamically
            "config_class": None,
            "description": "Qwen3-TTS - Universal end-to-end TTS with extreme low-latency",
            "use_api": False,  # Use direct model class
        },
    }

    @classmethod
    def get_model_type(cls, model_name: str) -> str:
        """
        Detect model type from model name.
        
        Args:
            model_name: Model identifier
            
        Returns:
            Model type string ("xtts", "qwen3", etc.)
        """
        if "cosyvoice" in model_name.lower():
            return "cosyvoice3"
        if "xtts" in model_name.lower():
            return "xtts"
        elif "qwen" in model_name.lower():
            return "qwen3"
        else:
            # Default to XTTS for backward compatibility
            return "xtts"

    @classmethod
    def create_model(
        cls,
        model_name: str,
        config: Optional[Any] = None,
        **kwargs
    ) -> Any:
        """
        Create TTS model instance based on model name.
        
        Args:
            model_name: Model identifier (e.g., "tts_models/multilingual/multi-dataset/xtts_v2")
            config: Optional model configuration
            **kwargs: Additional arguments for model initialization
            
        Returns:
            Instantiated model object
            
        Raises:
            ValueError: If model type not supported
        """
        model_type = cls.get_model_type(model_name)
        
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model type '{model_type}'. "
                f"Supported: {list(cls.SUPPORTED_MODELS.keys())}"
            )

        model_info = cls.SUPPORTED_MODELS[model_type]

        logger.info(
            f"Creating {model_type} model: {model_name} - {model_info['description']}"
        )

        try:
            # XTTS uses TTS API wrapper (handled in service layer)
            if model_info.get("use_api", False):
                raise ValueError(
                    f"Model type '{model_type}' should be loaded via TTS API wrapper"
                )
            
            # Load Qwen3-TTS dynamically
            if model_type == "qwen3":
                from TTS.tts.models.qwen3_tts import Qwen3TTS, Qwen3TTSConfig
                model_class = Qwen3TTS
                if config is None:
                    config = Qwen3TTSConfig()
            elif model_type == "cosyvoice3":
                from TTS.tts.models.cosyvoice3 import CosyVoice3TTS, CosyVoice3Config
                model_class = CosyVoice3TTS
                if config is None:
                    config = CosyVoice3Config()
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            model = model_class(config=config, **kwargs)
            return model
        except Exception as e:
            logger.error("Failed to create %s model (%s)", model_type, type(e).__name__)
            raise

    @classmethod
    def get_supported_models(cls) -> dict:
        """Get dictionary of supported models with descriptions."""
        return {
            name: info["description"]
            for name, info in cls.SUPPORTED_MODELS.items()
        }

    @classmethod
    def get_config_for_model(cls, model_name: str, device: str = "cpu") -> Any:
        """Get default config for model type.
        
        Args:
            model_name: Model identifier
            device: Device to use ("cpu", "cuda", "cuda:0", etc.)
        """
        model_type = cls.get_model_type(model_name)
        if model_type not in cls.SUPPORTED_MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load config class dynamically
        if model_type == "xtts":
            return None  # XTTS uses TTS API wrapper
        elif model_type == "cosyvoice3":
            from TTS.tts.models.cosyvoice3 import CosyVoice3Config
            return CosyVoice3Config(device_map=device)
        elif model_type == "qwen3":
            from TTS.tts.models.qwen3_tts import Qwen3TTSConfig
            config = Qwen3TTSConfig()
            # Set device from service configuration
            config.device_map = device if device != "cuda" else "cuda:0"
            logger.info(f"Configured Qwen3-TTS for device: {config.device_map}")
            return config
        
        return None
