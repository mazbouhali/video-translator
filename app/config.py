"""
Configuration management for Arabic Video Translator.

Settings can be configured via:
1. Environment variables (prefixed with AVT_)
2. Config file (~/.config/arabic-video-translator/config.json)
3. Direct function arguments (highest priority)
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


# Environment variable prefix
ENV_PREFIX = "AVT_"

# Default paths
DEFAULT_CONFIG_DIR = Path.home() / ".config" / "arabic-video-translator"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "arabic-video-translator"


@dataclass
class TranscriptionConfig:
    """Whisper transcription settings."""
    # Model options: large-v3, byne-arabic, quran, egyptian, code-switching, etc.
    # See ARABIC_ASR.md for full list
    model: str = "large-v3"  # Default to baseline; use "byne-arabic" for best Arabic
    language: str = "ar"
    device: Optional[str] = None  # None = auto-detect (cuda > mps > cpu)
    
    # Arabic dialect hint: msa, egyptian, levantine, gulf, maghrebi, iraqi
    dialect_hint: Optional[str] = None
    
    # Content type hint: general, religious, code-switching
    content_type: Optional[str] = None
    
    # Model cache directory (where Whisper stores downloaded models)
    model_cache_dir: Optional[str] = None


@dataclass
class TranslationConfig:
    """Translation model settings."""
    model: str = "nllb"  # nllb, nllb-large, marian
    device: Optional[str] = None
    batch_size: int = 8
    max_length: int = 512
    
    # HuggingFace cache directory
    model_cache_dir: Optional[str] = None


@dataclass
class SubtitleConfig:
    """Subtitle generation settings."""
    font_size: int = 24
    font_color: str = "white"
    outline_color: str = "black"
    position: str = "bottom"  # top, center, bottom
    margin_v: int = 30


@dataclass
class Config:
    """Main application configuration."""
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    subtitle: SubtitleConfig = field(default_factory=SubtitleConfig)
    
    # General settings
    temp_dir: Optional[str] = None  # None = system temp
    output_dir: Optional[str] = None  # None = same as input
    
    # Processing options
    keep_temp_files: bool = False
    verbose: bool = False


def _get_env(key: str, default: Any = None) -> Any:
    """Get environment variable with prefix."""
    return os.environ.get(f"{ENV_PREFIX}{key}", default)


def _load_config_file() -> Dict[str, Any]:
    """Load configuration from JSON file if it exists."""
    config_file = DEFAULT_CONFIG_DIR / "config.json"
    
    # Also check env var for custom config path
    custom_path = _get_env("CONFIG_FILE")
    if custom_path:
        config_file = Path(custom_path)
    
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    
    return {}


def load_config() -> Config:
    """
    Load configuration from environment and config file.
    
    Priority (highest to lowest):
    1. Environment variables
    2. Config file
    3. Defaults
    
    Returns:
        Populated Config object
    """
    # Load from file first
    file_config = _load_config_file()
    
    # Build transcription config
    trans_file = file_config.get("transcription", {})
    transcription = TranscriptionConfig(
        model=_get_env("WHISPER_MODEL") or trans_file.get("model", "large-v3"),
        language=_get_env("LANGUAGE") or trans_file.get("language", "ar"),
        device=_get_env("DEVICE") or trans_file.get("device"),
        model_cache_dir=_get_env("WHISPER_CACHE") or trans_file.get("model_cache_dir"),
    )
    
    # Build translation config
    transl_file = file_config.get("translation", {})
    translation = TranslationConfig(
        model=_get_env("TRANSLATION_MODEL") or transl_file.get("model", "nllb"),
        device=_get_env("DEVICE") or transl_file.get("device"),
        batch_size=int(_get_env("BATCH_SIZE") or transl_file.get("batch_size", 8)),
        max_length=int(_get_env("MAX_LENGTH") or transl_file.get("max_length", 512)),
        model_cache_dir=_get_env("HF_CACHE") or transl_file.get("model_cache_dir"),
    )
    
    # Build subtitle config
    sub_file = file_config.get("subtitle", {})
    subtitle = SubtitleConfig(
        font_size=int(_get_env("FONT_SIZE") or sub_file.get("font_size", 24)),
        font_color=_get_env("FONT_COLOR") or sub_file.get("font_color", "white"),
        outline_color=_get_env("OUTLINE_COLOR") or sub_file.get("outline_color", "black"),
        position=_get_env("SUBTITLE_POSITION") or sub_file.get("position", "bottom"),
        margin_v=int(_get_env("MARGIN_V") or sub_file.get("margin_v", 30)),
    )
    
    return Config(
        transcription=transcription,
        translation=translation,
        subtitle=subtitle,
        temp_dir=_get_env("TEMP_DIR") or file_config.get("temp_dir"),
        output_dir=_get_env("OUTPUT_DIR") or file_config.get("output_dir"),
        keep_temp_files=bool(_get_env("KEEP_TEMP") or file_config.get("keep_temp_files", False)),
        verbose=bool(_get_env("VERBOSE") or file_config.get("verbose", False)),
    )


def save_config(config: Config, path: Optional[Path] = None) -> Path:
    """
    Save configuration to JSON file.
    
    Args:
        config: Config object to save
        path: Optional custom path (default: ~/.config/arabic-video-translator/config.json)
        
    Returns:
        Path to saved config file
    """
    if path is None:
        path = DEFAULT_CONFIG_DIR / "config.json"
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = {
        "transcription": {
            "model": config.transcription.model,
            "language": config.transcription.language,
            "device": config.transcription.device,
            "model_cache_dir": config.transcription.model_cache_dir,
        },
        "translation": {
            "model": config.translation.model,
            "device": config.translation.device,
            "batch_size": config.translation.batch_size,
            "max_length": config.translation.max_length,
            "model_cache_dir": config.translation.model_cache_dir,
        },
        "subtitle": {
            "font_size": config.subtitle.font_size,
            "font_color": config.subtitle.font_color,
            "outline_color": config.subtitle.outline_color,
            "position": config.subtitle.position,
            "margin_v": config.subtitle.margin_v,
        },
        "temp_dir": config.temp_dir,
        "output_dir": config.output_dir,
        "keep_temp_files": config.keep_temp_files,
        "verbose": config.verbose,
    }
    
    with open(path, "w") as f:
        json.dump(config_dict, f, indent=2)
    
    return path


# Global config instance (lazy loaded)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config():
    """Reset the global configuration (forces reload on next get_config())."""
    global _config
    _config = None
