"""
Audio extraction and Arabic transcription using OpenAI Whisper.

This module handles:
- Extracting audio from video files using ffmpeg
- Transcribing Arabic speech using Whisper large-v3
- Returning timestamped segments for subtitle generation

Example:
    >>> from app.transcribe import transcribe_video
    >>> result = transcribe_video("video.mp4")
    >>> print(result["text"])
    >>> for seg in result["segments"]:
    ...     print(f"{seg['start']:.2f} - {seg['end']:.2f}: {seg['text']}")
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable

import whisper
from tqdm import tqdm

from .config import get_config


def check_ffmpeg() -> bool:
    """
    Check if ffmpeg is available on the system.
    
    Returns:
        True if ffmpeg is available, False otherwise
    """
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_audio(
    video_path: str,
    output_path: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Extract audio from video file using ffmpeg.
    
    Converts to 16kHz mono WAV format, which is optimal for Whisper.
    
    Args:
        video_path: Path to input video file (supports mp4, mkv, avi, mov, webm, etc.)
        output_path: Optional path for output audio. If None, creates a temp file.
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Path to extracted audio file (WAV format, 16kHz mono)
        
    Raises:
        FileNotFoundError: If video file doesn't exist
        RuntimeError: If ffmpeg fails or is not installed
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
    
    # ffmpeg command: extract audio as 16kHz mono WAV (optimal for Whisper)
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                    # No video output
        "-acodec", "pcm_s16le",   # PCM 16-bit little-endian
        "-ar", "16000",           # 16kHz sample rate (Whisper optimal)
        "-ac", "1",               # Mono channel
        "-y",                     # Overwrite output without asking
        str(output_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extraction failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: choco install ffmpeg"
        )
    
    if progress_callback:
        progress_callback(f"Audio extracted: {output_path.name}")
    
    return str(output_path)


def transcribe_arabic(
    audio_path: str,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Transcribe Arabic audio using OpenAI Whisper.
    
    Args:
        audio_path: Path to audio file (WAV recommended, but supports various formats)
        model_name: Whisper model name. Options: tiny, base, small, medium, large,
                    large-v2, large-v3 (default from config, typically large-v3)
        device: Device for inference ('cuda', 'mps', 'cpu', or None for auto-detect)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary containing:
            - text: Full transcription text (Arabic)
            - segments: List of segments with 'id', 'start', 'end', 'text' keys
            - language: Detected/specified language code
            
    Raises:
        FileNotFoundError: If audio file doesn't exist
    """
    audio_path = Path(audio_path).expanduser().resolve()
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    # Load config for defaults
    config = get_config()
    if model_name is None:
        model_name = config.transcription.model
    if device is None:
        device = config.transcription.device
    
    if progress_callback:
        progress_callback(f"Loading Whisper model: {model_name}")
    
    # Load Whisper model
    # Note: First run will download the model (~3GB for large-v3)
    model = whisper.load_model(model_name, device=device)
    
    if progress_callback:
        progress_callback("Transcribing Arabic speech (this may take a while)...")
    
    # Transcribe with Arabic language specification
    result = model.transcribe(
        str(audio_path),
        language=config.transcription.language,  # Arabic
        task="transcribe",       # Transcribe in original language (not translate)
        verbose=False,
        fp16=False if device == "cpu" else None,  # Disable FP16 on CPU
    )
    
    # Process segments into clean format
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
        "language": result.get("language", config.transcription.language),
    }


def transcribe_video(
    video_path: str,
    model_name: Optional[str] = None,
    device: Optional[str] = None,
    keep_audio: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Extract audio from video and transcribe Arabic speech.
    
    This is the main entry point for video transcription. It handles
    the full pipeline: audio extraction → transcription → cleanup.
    
    Args:
        video_path: Path to input video file
        model_name: Whisper model to use (default from config)
        device: Device for inference (default: auto-detect)
        keep_audio: If True, don't delete the extracted audio file
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with transcription results:
            - text: Full transcription
            - segments: Timestamped segments
            - language: Detected language
            - audio_path: Path to audio (only if keep_audio=True)
            
    Example:
        >>> result = transcribe_video("arabic_video.mp4")
        >>> print(f"Found {len(result['segments'])} speech segments")
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
            progress_callback=progress_callback
        )
        
        # Include audio path in result if keeping
        result["audio_path"] = audio_path if keep_audio else None
        
        return result
        
    finally:
        # Cleanup: remove temp audio unless keeping
        if not keep_audio and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass  # Best effort cleanup


# CLI test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.transcribe <video_file>")
        sys.exit(1)
    
    print("Testing transcription...")
    result = transcribe_video(
        sys.argv[1],
        progress_callback=lambda msg: print(f"  → {msg}")
    )
    
    print(f"\n{'='*50}")
    print(f"Language: {result['language']}")
    print(f"Segments: {len(result['segments'])}")
    print(f"\nFull text:\n{result['text'][:500]}...")
    
    print(f"\nFirst 3 segments:")
    for seg in result['segments'][:3]:
        print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text']}")
