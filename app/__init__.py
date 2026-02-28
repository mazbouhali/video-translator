"""
Arabic Video Translator

Automatically translate Arabic speech in videos to English subtitles.

Modules:
    - transcribe: Audio extraction and Whisper transcription
    - translate: Arabic-English translation using NLLB/MarianMT
    - subtitles: SRT generation and video embedding
    - config: Configuration management
    - main: CLI interface
    - web: Gradio web interface

Quick Start:
    CLI:
        python -m app.main video.mp4
        
    Web:
        python -m app.web
        
    Python:
        from app.transcribe import transcribe_video
        from app.translate import create_translator
        from app.subtitles import write_srt, embed_subtitles
"""

__version__ = "1.0.0"
__author__ = "Arabic Video Translator Contributors"
__license__ = "MIT"

from .config import get_config, load_config, Config
from .transcribe import transcribe_video, extract_audio, transcribe_arabic
from .translate import create_translator, ArabicTranslator
from .subtitles import generate_srt, write_srt, read_srt, embed_subtitles

__all__ = [
    # Config
    "get_config",
    "load_config", 
    "Config",
    # Transcription
    "transcribe_video",
    "extract_audio",
    "transcribe_arabic",
    # Translation
    "create_translator",
    "ArabicTranslator",
    # Subtitles
    "generate_srt",
    "write_srt",
    "read_srt",
    "embed_subtitles",
]
