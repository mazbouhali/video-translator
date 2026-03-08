"""
Audio extraction and Arabic transcription using optimized models.

This module handles:
- Extracting audio from video files using ffmpeg
- Transcribing Arabic speech using faster-whisper (4x faster than OpenAI implementation)
- Support for fine-tuned Arabic models and dialect detection
- Special optimizations for religious/poetic content

Supported models (in order of recommendation):
1. Byne/whisper-large-v3-arabic - Best general Arabic (WER ~9.4%)
2. openai/whisper-large-v3 - Baseline, good for MSA
3. tarteel-ai/whisper-base-ar-quran - Specialized for Quranic recitation
4. MohamedRashad/Arabic-Whisper-CodeSwitching-Edition - Arabic/English mix

Example:
    >>> from app.transcribe import transcribe_video
    >>> result = transcribe_video("video.mp4", dialect_hint="levantine")
    >>> print(result["text"])
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Literal

from .config import get_config

# Import backends - prefer faster-whisper, fall back to openai whisper
try:
    from faster_whisper import WhisperModel as FasterWhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False

try:
    import whisper
    OPENAI_WHISPER_AVAILABLE = True
except ImportError:
    OPENAI_WHISPER_AVAILABLE = False


# ============================================================================
# Arabic Model Registry
# ============================================================================

# Dialect codes for model selection hints
ARABIC_DIALECTS = {
    "msa": "Modern Standard Arabic (الفصحى)",
    "classical": "Classical Arabic (العربية الفصيحة) - Quran, poetry",
    "egyptian": "Egyptian Arabic (مصري)",
    "levantine": "Levantine Arabic (شامي) - Lebanese, Syrian, Palestinian, Jordanian",
    "lebanese": "Lebanese Arabic (لبناني)",
    "syrian": "Syrian Arabic (سوري)",
    "iraqi": "Iraqi Arabic (عراقي)",
    "gulf": "Gulf Arabic (خليجي) - Saudi, UAE, Kuwait, Bahrain, Qatar",
    "maghrebi": "Maghrebi Arabic (مغربي) - Moroccan, Algerian, Tunisian",
}

# Content type presets with optimized prompts
CONTENT_PRESETS = {
    "nasheed": {
        "description": "Religious nasheeds (أناشيد) - sung/chanted",
        "separate_vocals": True,  # Critical: remove drums/music
        "prompts": [
            "نشيد ديني إسلامي",  # Islamic religious nasheed
            "لا إله إلا الله محمد رسول الله",
            "اللهم صل على محمد",
        ],
    },
    "political_speech": {
        "description": "Political speeches and addresses",
        "separate_vocals": False,
        "prompts": [
            "خطاب سياسي",  # Political speech
            "المقاومة والتحرير",
            "الشعب والوطن",
        ],
    },
    "religious": {
        "description": "Quran recitation and religious lectures",
        "separate_vocals": False,
        "prompts": [
            "بسم الله الرحمن الرحيم",
            "القرآن الكريم",
            "الحمد لله رب العالمين",
        ],
    },
    "general": {
        "description": "General Arabic speech",
        "separate_vocals": False,
        "prompts": [],
    },
}

# Model recommendations by use case
ARABIC_MODELS = {
    # General Arabic - best overall
    "byne-arabic": {
        "hf_name": "Byne/whisper-large-v3-arabic",
        "faster_whisper": None,  # Needs conversion to CT2 format
        "wer": 9.4,
        "dialects": ["msa", "egyptian", "levantine", "gulf"],
        "description": "Fine-tuned large-v3, best general Arabic model",
        "best_for": ["general", "political_speech"],
    },
    # OpenAI baseline
    "large-v3": {
        "hf_name": "openai/whisper-large-v3",
        "faster_whisper": "large-v3",  # Available in faster-whisper
        "wer": 15.0,
        "dialects": ["msa"],
        "description": "OpenAI baseline, good for formal MSA",
        "best_for": ["general"],
    },
    # Turbo - faster
    "large-v3-turbo": {
        "hf_name": "openai/whisper-large-v3-turbo",
        "faster_whisper": "large-v3-turbo",
        "wer": 18.0,
        "dialects": ["msa"],
        "description": "Faster variant, ~4x faster with minor accuracy trade-off",
        "best_for": ["general"],
    },
    # Levantine Arabic (Lebanese/Syrian) - IMPORTANT for nasheeds
    "levantine": {
        "hf_name": "EmreAkgul/whisper-large-v3-lora-levantine-arabic",
        "faster_whisper": None,
        "wer": 15.0,  # Estimated
        "dialects": ["levantine", "lebanese", "syrian"],
        "description": "LoRA fine-tuned for Levantine Arabic",
        "best_for": ["nasheed", "political_speech"],
    },
    # Levantine wav2vec2 alternative
    "levantine-wav2vec": {
        "hf_name": "elgeish/wav2vec2-large-xlsr-53-levantine-arabic",
        "faster_whisper": None,
        "wer": 12.0,
        "dialects": ["levantine"],
        "description": "Wav2Vec2 model, good for Levantine dialect",
        "model_type": "wav2vec2",  # Different architecture
        "best_for": ["political_speech"],
    },
    # Iraqi Arabic - CRITICAL for some nasheeds
    "iraqi": {
        "hf_name": "ayousry42/whisper-arabic-dialect-iraqi",
        "faster_whisper": None,
        "wer": 20.0,  # Estimated
        "dialects": ["iraqi"],
        "description": "Specialized for Iraqi Arabic dialect",
        "best_for": ["nasheed", "political_speech"],
    },
    # Egyptian dialect
    "egyptian": {
        "hf_name": "AbdelrahmanHassan/whisper-large-v3-egyptian-arabic",
        "faster_whisper": None,
        "wer": 12.0,
        "dialects": ["egyptian"],
        "description": "Specialized for Egyptian Arabic",
        "best_for": ["general"],
    },
    # Code-switching (Arabic + English)
    "code-switching": {
        "hf_name": "MohamedRashad/Arabic-Whisper-CodeSwitching-Edition",
        "faster_whisper": None,
        "wer": 14.0,
        "dialects": ["msa", "egyptian"],
        "description": "Handles Arabic with embedded English words",
        "best_for": ["general"],
    },
    # Quran/religious content - EXCELLENT for nasheeds with classical Arabic
    "quran": {
        "hf_name": "tarteel-ai/whisper-base-ar-quran",
        "faster_whisper": None,
        "wer": 5.8,
        "dialects": ["classical"],
        "description": "Specialized for Quranic Arabic recitation (5.8% WER!)",
        "best_for": ["religious", "nasheed"],
    },
    # Quran large turbo - LoRA fine-tuned
    "quran-turbo": {
        "hf_name": "MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix",
        "faster_whisper": None,
        "wer": 6.0,
        "dialects": ["classical"],
        "description": "Large turbo model fine-tuned on Quran - good for sung Arabic",
        "best_for": ["religious", "nasheed"],
    },
    # Multi-dialect
    "multidialect": {
        "hf_name": "MadLook/whisper-small-arabic-multidialect",
        "faster_whisper": None,
        "wer": 48.9,
        "dialects": ["msa", "egyptian", "levantine", "gulf", "maghrebi"],
        "description": "Small model trained on multiple dialects",
        "best_for": ["general"],
    },
}


# ============================================================================
# Utility Functions
# ============================================================================

def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_available_backend() -> str:
    """Determine which Whisper backend is available."""
    if FASTER_WHISPER_AVAILABLE:
        return "faster-whisper"
    elif OPENAI_WHISPER_AVAILABLE:
        return "openai-whisper"
    else:
        raise ImportError(
            "No Whisper backend available. Install one of:\n"
            "  pip install faster-whisper  # Recommended: 4x faster\n"
            "  pip install openai-whisper  # Original OpenAI implementation"
        )


def get_recommended_model(
    dialect: Optional[str] = None,
    content_type: Optional[Literal["general", "religious", "code-switching", "nasheed", "political_speech"]] = None,
    speed_priority: bool = False,
) -> str:
    """
    Get the recommended model based on content characteristics.
    
    Args:
        dialect: Arabic dialect hint (msa, egyptian, levantine, lebanese, syrian, iraqi, gulf, maghrebi)
        content_type: Type of content (general, religious, code-switching, nasheed, political_speech)
        speed_priority: If True, prefer faster models over accuracy
        
    Returns:
        Model key from ARABIC_MODELS
    """
    # NASHEED: Critical path - singing/chanting Arabic
    if content_type == "nasheed":
        # Nasheeds often mix classical Arabic with dialect
        if dialect in ("iraqi",):
            return "iraqi"
        if dialect in ("levantine", "lebanese", "syrian"):
            return "levantine"
        # Default for nasheeds: Quran model handles classical Arabic sung style well
        return "quran-turbo" if speed_priority else "quran"
    
    # POLITICAL SPEECH: Formal but may have dialect
    if content_type == "political_speech":
        if dialect == "iraqi":
            return "iraqi"
        if dialect in ("levantine", "lebanese", "syrian"):
            return "levantine"
        # Political speeches are often formal MSA
        return "large-v3-turbo" if speed_priority else "byne-arabic"
    
    # RELIGIOUS: Quran recitation, lectures
    if content_type == "religious":
        return "quran-turbo" if speed_priority else "quran"
    
    if content_type == "code-switching":
        return "code-switching"
    
    # Dialect-specific recommendations
    if dialect == "iraqi":
        return "iraqi"
    
    if dialect in ("levantine", "lebanese", "syrian"):
        return "levantine"
    
    if dialect == "egyptian":
        return "egyptian"
    
    if dialect in ("gulf", "maghrebi"):
        return "large-v3-turbo" if speed_priority else "byne-arabic"
    
    # Default: best general Arabic model
    if speed_priority:
        return "large-v3-turbo"
    
    return "byne-arabic"


def get_content_preset(content_type: str) -> dict:
    """Get preset configuration for a content type."""
    return CONTENT_PRESETS.get(content_type, CONTENT_PRESETS["general"])


def build_initial_prompt(
    content_type: Optional[str] = None,
    dialect: Optional[str] = None,
    custom_prompt: Optional[str] = None
) -> str:
    """
    Build an optimal initial prompt for Arabic transcription.
    
    Prompts help Whisper understand:
    - Expected vocabulary (religious terms, political terms)
    - Style (formal, dialectal)
    - Format (poetry style for nasheeds)
    
    Args:
        content_type: Content type (nasheed, political_speech, religious, general)
        dialect: Arabic dialect
        custom_prompt: Custom prompt to append
        
    Returns:
        Formatted prompt string
    """
    parts = []
    
    # Add content-specific prompts
    if content_type and content_type in CONTENT_PRESETS:
        preset = CONTENT_PRESETS[content_type]
        if preset.get("prompts"):
            parts.extend(preset["prompts"][:2])  # Use first 2 prompts
    
    # Add dialect hint
    if dialect and dialect in ARABIC_DIALECTS:
        parts.append(f"[{ARABIC_DIALECTS[dialect]}]")
    
    # Add custom prompt
    if custom_prompt:
        parts.append(custom_prompt)
    
    return " ".join(parts)


def list_models() -> Dict[str, Dict]:
    """Return available Arabic models with their details."""
    return ARABIC_MODELS.copy()


def list_dialects() -> Dict[str, str]:
    """Return supported Arabic dialects."""
    return ARABIC_DIALECTS.copy()


# ============================================================================
# Audio Extraction
# ============================================================================

def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Extract audio from video file using ffmpeg.
    
    Converts to 16kHz mono WAV format, optimal for Whisper.
    
    Args:
        video_path: Path to input video file
        output_path: Optional path for output audio
        progress_callback: Optional callback for progress updates
        
    Returns:
        Path to extracted audio file (WAV format, 16kHz mono)
    """
    video_path = Path(video_path).expanduser().resolve()
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_path is None:
        config = get_config()
        temp_dir = config.temp_dir
        fd, output_path = tempfile.mkstemp(suffix=".wav", dir=temp_dir)
        os.close(fd)
    
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if progress_callback:
        progress_callback("Extracting audio from video...")
    
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16kHz (Whisper optimal)
        "-ac", "1",               # Mono
        "-y",                     # Overwrite
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extraction failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install it:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg"
        )
    
    if progress_callback:
        progress_callback(f"Audio extracted: {output_path.name}")
    
    return str(output_path)


# ============================================================================
# Transcription with faster-whisper (Recommended)
# ============================================================================

def transcribe_arabic_faster(
    audio_path: str,
    model_name: str = "large-v3",
    device: Optional[str] = None,
    compute_type: str = "float16",
    dialect_hint: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    vad_filter: bool = True,
    word_timestamps: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Transcribe Arabic audio using faster-whisper (CTranslate2 backend).
    
    Up to 4x faster than OpenAI Whisper with same accuracy, lower memory.
    
    Args:
        audio_path: Path to audio file
        model_name: Model to use (large-v3, large-v3-turbo, or HuggingFace model)
        device: Device for inference (cuda, cpu, auto)
        compute_type: Precision (float16, int8, int8_float16)
        dialect_hint: Optional dialect hint (egyptian, levantine, etc.)
        initial_prompt: Optional prompt to guide transcription style
        vad_filter: Use Voice Activity Detection to filter silence
        word_timestamps: Return word-level timestamps
        progress_callback: Optional progress callback
        
    Returns:
        Dictionary with text, segments, language, and metadata
    """
    if not FASTER_WHISPER_AVAILABLE:
        raise ImportError("faster-whisper not installed. Run: pip install faster-whisper")
    
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Determine device and compute type
    if device is None:
        device = "auto"
    
    # Auto-adjust compute_type for compatibility
    # AMD ROCm and CPU don't support float16 efficiently
    if compute_type == "float16":
        import torch
        if not torch.cuda.is_available():
            # CPU or unsupported GPU - use float32
            compute_type = "float32"
        elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
            # ROCm/HIP backend - float16 not reliably supported (especially RDNA 4)
            compute_type = "float32"
        elif torch.cuda.is_available():
            # Double-check device name for AMD GPUs on older ROCm
            try:
                device_name = torch.cuda.get_device_name(0).lower()
                if "amd" in device_name or "radeon" in device_name or "gfx" in device_name:
                    compute_type = "float32"
            except:
                pass
    
    if progress_callback:
        progress_callback(f"Loading model: {model_name} (faster-whisper, {compute_type})")
    
    # Check if this is a standard model or HuggingFace model
    model_info = ARABIC_MODELS.get(model_name, {})
    faster_whisper_name = model_info.get("faster_whisper", model_name)
    
    # Load the model
    model = FasterWhisperModel(
        faster_whisper_name or model_name,
        device=device,
        compute_type=compute_type,
    )
    
    # Build transcription prompt
    prompt = initial_prompt or ""
    if dialect_hint and dialect_hint in ARABIC_DIALECTS:
        dialect_name = ARABIC_DIALECTS[dialect_hint]
        if not prompt:
            prompt = f"[{dialect_name}]"
    
    if progress_callback:
        progress_callback("Transcribing Arabic speech...")
    
    # Transcribe with optimized settings for Arabic
    segments_iter, info = model.transcribe(
        str(audio_path),
        language="ar",
        task="transcribe",
        beam_size=5,
        best_of=5,
        patience=1.0,
        length_penalty=1.0,
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        initial_prompt=prompt if prompt else None,
        vad_filter=vad_filter,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 200,
        },
        word_timestamps=word_timestamps,
    )
    
    # Collect segments
    segments = []
    full_text_parts = []
    
    for seg in segments_iter:
        segment_data = {
            "id": seg.id,
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        }
        
        if word_timestamps and hasattr(seg, 'words') and seg.words:
            segment_data["words"] = [
                {
                    "word": w.word,
                    "start": float(w.start),
                    "end": float(w.end),
                    "probability": float(w.probability),
                }
                for w in seg.words
            ]
        
        segments.append(segment_data)
        full_text_parts.append(seg.text.strip())
    
    if progress_callback:
        progress_callback(f"Transcription complete: {len(segments)} segments")
    
    return {
        "text": " ".join(full_text_parts),
        "segments": segments,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "backend": "faster-whisper",
        "model": model_name,
        "dialect_hint": dialect_hint,
    }


# ============================================================================
# Transcription with OpenAI Whisper (Fallback)
# ============================================================================

def transcribe_arabic_openai(
    audio_path: str,
    model_name: str = "large-v3",
    device: Optional[str] = None,
    dialect_hint: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Transcribe Arabic audio using OpenAI Whisper (original implementation).
    
    Note: Slower than faster-whisper but works as fallback.
    
    Args:
        audio_path: Path to audio file  
        model_name: Whisper model name
        device: Device for inference
        dialect_hint: Optional dialect hint
        initial_prompt: Optional prompt
        progress_callback: Optional callback
        
    Returns:
        Dictionary with transcription results
    """
    if not OPENAI_WHISPER_AVAILABLE:
        raise ImportError("openai-whisper not installed. Run: pip install openai-whisper")
    
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    config = get_config()
    if device is None:
        device = config.transcription.device
    
    if progress_callback:
        progress_callback(f"Loading model: {model_name} (openai-whisper)")
    
    model = whisper.load_model(model_name, device=device)
    
    # Build prompt
    prompt = initial_prompt or ""
    if dialect_hint and dialect_hint in ARABIC_DIALECTS:
        if not prompt:
            prompt = f"[{ARABIC_DIALECTS[dialect_hint]}]"
    
    if progress_callback:
        progress_callback("Transcribing Arabic speech (this may take a while)...")
    
    result = model.transcribe(
        str(audio_path),
        language="ar",
        task="transcribe",
        verbose=False,
        fp16=device != "cpu",
        initial_prompt=prompt if prompt else None,
    )
    
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "id": seg.get("id", 0),
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": seg.get("text", "").strip(),
        })
    
    if progress_callback:
        progress_callback(f"Transcription complete: {len(segments)} segments")
    
    return {
        "text": result.get("text", "").strip(),
        "segments": segments,
        "language": result.get("language", "ar"),
        "backend": "openai-whisper",
        "model": model_name,
        "dialect_hint": dialect_hint,
    }


# ============================================================================
# Main Transcription Interface
# ============================================================================

def transcribe_arabic(
    audio_path: str,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    dialect_hint: Optional[str] = None,
    content_type: Optional[Literal["general", "religious", "code-switching", "nasheed", "political_speech"]] = None,
    initial_prompt: Optional[str] = None,
    backend: Optional[Literal["faster-whisper", "openai-whisper", "auto"]] = "auto",
    vad_filter: bool = True,
    word_timestamps: bool = False,
    separate_vocals: Optional[bool] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Transcribe Arabic audio with automatic model and backend selection.
    
    This is the main entry point for transcription. It automatically:
    - Selects the best backend (faster-whisper > openai-whisper)
    - Recommends optimal model based on dialect/content hints
    - Configures parameters for Arabic speech
    - Separates vocals from background music (for nasheeds)
    
    Args:
        audio_path: Path to audio file (WAV recommended)
        model_name: Model to use (None = auto-select based on hints)
        device: Device for inference (None = auto-detect)
        dialect_hint: Arabic dialect (msa, egyptian, levantine, lebanese, iraqi, gulf, maghrebi)
        content_type: Content type (general, religious, code-switching, nasheed, political_speech)
        initial_prompt: Optional prompt to guide style
        backend: Which backend to use (auto, faster-whisper, openai-whisper)
        vad_filter: Use VAD to filter silence (faster-whisper only)
        word_timestamps: Return word-level timestamps
        separate_vocals: Use Demucs to isolate vocals (auto for nasheeds)
        progress_callback: Optional progress callback
        
    Returns:
        Dictionary containing:
            - text: Full transcription
            - segments: Timestamped segments
            - language: Detected language
            - backend: Which backend was used
            - model: Model used
            - dialect_hint: Dialect hint if provided
            - content_type: Content type used
            - vocals_separated: Whether vocal separation was used
            
    Example:
        >>> # For nasheeds - automatically separates vocals from drums
        >>> result = transcribe_arabic(
        ...     "nasheed.mp3",
        ...     content_type="nasheed",
        ...     dialect_hint="levantine"
        ... )
        >>> 
        >>> # For political speeches
        >>> result = transcribe_arabic(
        ...     "speech.mp3",
        ...     content_type="political_speech",
        ...     dialect_hint="iraqi"
        ... )
    """
    original_audio_path = audio_path
    vocals_separated = False
    
    # Get content preset
    preset = get_content_preset(content_type) if content_type else {}
    
    # Determine if we should separate vocals
    if separate_vocals is None:
        separate_vocals = preset.get("separate_vocals", False)
    
    # Separate vocals if needed (critical for nasheeds with background music)
    if separate_vocals:
        try:
            from .audio_preprocess import preprocess_for_transcription, check_demucs
            if check_demucs():
                if progress_callback:
                    progress_callback("Separating vocals from background music...")
                audio_path = preprocess_for_transcription(
                    audio_path,
                    separate_vocals=True,
                    normalize=True,
                    progress_callback=progress_callback
                )
                vocals_separated = True
            else:
                if progress_callback:
                    progress_callback("Demucs not available, skipping vocal separation")
        except ImportError:
            if progress_callback:
                progress_callback("Audio preprocessing module not available")
    
    # Auto-select model if not specified
    if model_name is None:
        model_name = get_recommended_model(
            dialect=dialect_hint,
            content_type=content_type,
            speed_priority=False,
        )
        if progress_callback:
            progress_callback(f"Auto-selected model: {model_name}")
    
    # Build initial prompt if not provided
    if initial_prompt is None:
        initial_prompt = build_initial_prompt(
            content_type=content_type,
            dialect=dialect_hint
        )
        if initial_prompt and progress_callback:
            progress_callback(f"Using prompt: {initial_prompt[:50]}...")
    
    # Determine backend
    if backend == "auto":
        backend = get_available_backend()
    
    # Call appropriate backend
    if backend == "faster-whisper":
        result = transcribe_arabic_faster(
            audio_path=audio_path,
            model_name=model_name,
            device=device,
            dialect_hint=dialect_hint,
            initial_prompt=initial_prompt,
            vad_filter=vad_filter,
            word_timestamps=word_timestamps,
            progress_callback=progress_callback,
        )
    else:
        result = transcribe_arabic_openai(
            audio_path=audio_path,
            model_name=model_name,
            device=device,
            dialect_hint=dialect_hint,
            initial_prompt=initial_prompt,
            progress_callback=progress_callback,
        )
    
    # Add metadata
    result["content_type"] = content_type
    result["vocals_separated"] = vocals_separated
    
    return result


def transcribe_video(
    video_path: str,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    dialect_hint: Optional[str] = None,
    content_type: Optional[Literal["general", "religious", "code-switching"]] = None,
    keep_audio: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Extract audio from video and transcribe Arabic speech.
    
    Full pipeline: video → audio extraction → transcription → cleanup.
    
    Args:
        video_path: Path to input video file
        model_name: Whisper model (None = auto-select)
        device: Device for inference
        dialect_hint: Arabic dialect hint
        content_type: Content type hint
        keep_audio: Keep extracted audio file
        progress_callback: Optional callback
        
    Returns:
        Dictionary with transcription results
        
    Example:
        >>> result = transcribe_video(
        ...     "arabic_lecture.mp4",
        ...     dialect_hint="egyptian"
        ... )
    """
    if progress_callback:
        progress_callback("Starting video transcription pipeline...")
    
    # Step 1: Extract audio
    audio_path = extract_audio(video_path, progress_callback=progress_callback)
    
    try:
        # Step 2: Transcribe
        result = transcribe_arabic(
            audio_path,
            model_name=model_name,
            device=device,
            dialect_hint=dialect_hint,
            content_type=content_type,
            progress_callback=progress_callback
        )
        
        result["audio_path"] = audio_path if keep_audio else None
        return result
        
    finally:
        if not keep_audio and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass


# ============================================================================
# CLI Test
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("Arabic Transcription Module")
    print("=" * 50)
    print(f"Backend: {get_available_backend()}")
    print(f"faster-whisper: {'✓' if FASTER_WHISPER_AVAILABLE else '✗'}")
    print(f"openai-whisper: {'✓' if OPENAI_WHISPER_AVAILABLE else '✗'}")
    print()
    
    print("Available models:")
    for key, info in ARABIC_MODELS.items():
        print(f"  {key}: {info['description']} (WER: {info['wer']}%)")
    print()
    
    print("Supported dialects:")
    for key, name in ARABIC_DIALECTS.items():
        print(f"  {key}: {name}")
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.transcribe <video_or_audio_file> [--dialect DIALECT]")
        sys.exit(0)
    
    # Parse simple args
    file_path = sys.argv[1]
    dialect = None
    if "--dialect" in sys.argv:
        idx = sys.argv.index("--dialect")
        if idx + 1 < len(sys.argv):
            dialect = sys.argv[idx + 1]
    
    print(f"Transcribing: {file_path}")
    if dialect:
        print(f"Dialect hint: {dialect}")
    
    # Determine if video or audio
    ext = Path(file_path).suffix.lower()
    if ext in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
        result = transcribe_video(
            file_path,
            dialect_hint=dialect,
            progress_callback=lambda msg: print(f"  → {msg}")
        )
    else:
        result = transcribe_arabic(
            file_path,
            dialect_hint=dialect,
            progress_callback=lambda msg: print(f"  → {msg}")
        )
    
    print(f"\n{'='*50}")
    print(f"Backend: {result.get('backend', 'unknown')}")
    print(f"Model: {result.get('model', 'unknown')}")
    print(f"Language: {result['language']}")
    print(f"Segments: {len(result['segments'])}")
    print(f"\nFull text:\n{result['text'][:500]}...")
    
    print(f"\nFirst 3 segments:")
    for seg in result['segments'][:3]:
        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
