"""
Audio extraction and Arabic transcription using Whisper large-v3.
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import whisper
from tqdm import tqdm


def extract_audio(video_path: str, output_path: Optional[str] = None) -> str:
    """
    Extract audio from video file using ffmpeg.
    
    Args:
        video_path: Path to input video file
        output_path: Optional path for output audio. If None, creates temp file.
        
    Returns:
        Path to extracted audio file (WAV format, 16kHz mono)
    """
    video_path = Path(video_path).expanduser().resolve()
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    if output_path is None:
        # Create temp file that persists
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    
    output_path = Path(output_path).expanduser().resolve()
    
    # Extract audio with ffmpeg: 16kHz mono WAV (optimal for Whisper)
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # PCM 16-bit
        "-ar", "16000",           # 16kHz sample rate
        "-ac", "1",               # Mono
        "-y",                     # Overwrite output
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
        raise RuntimeError(f"ffmpeg failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
    
    return str(output_path)


def transcribe_arabic(
    audio_path: str,
    model_name: str = "large-v3",
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Transcribe Arabic audio using OpenAI Whisper.
    
    Args:
        audio_path: Path to audio file
        model_name: Whisper model name (default: large-v3 for best Arabic support)
        device: Device to use ('cuda', 'cpu', or None for auto)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dict containing:
            - text: Full transcription text
            - segments: List of segments with timestamps
            - language: Detected language
    """
    audio_path = Path(audio_path).expanduser().resolve()
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if progress_callback:
        progress_callback(f"Loading Whisper model: {model_name}")
    
    # Load model
    model = whisper.load_model(model_name, device=device)
    
    if progress_callback:
        progress_callback("Transcribing audio (this may take a while)...")
    
    # Transcribe with Arabic language hint
    result = model.transcribe(
        str(audio_path),
        language="ar",           # Arabic
        task="transcribe",       # Transcribe (not translate)
        verbose=False,
        fp16=False if device == "cpu" else None,  # Disable FP16 on CPU
    )
    
    # Process segments for cleaner output
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "id": seg.get("id", 0),
            "start": seg.get("start", 0.0),
            "end": seg.get("end", 0.0),
            "text": seg.get("text", "").strip(),
        })
    
    return {
        "text": result.get("text", "").strip(),
        "segments": segments,
        "language": result.get("language", "ar"),
    }


def transcribe_video(
    video_path: str,
    model_name: str = "large-v3",
    device: Optional[str] = None,
    keep_audio: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, Any]:
    """
    Extract audio from video and transcribe Arabic speech.
    
    Args:
        video_path: Path to input video file
        model_name: Whisper model to use
        device: Device for inference
        keep_audio: If True, don't delete extracted audio
        progress_callback: Optional callback for progress updates
        
    Returns:
        Transcription result dict
    """
    if progress_callback:
        progress_callback("Extracting audio from video...")
    
    audio_path = extract_audio(video_path)
    
    try:
        result = transcribe_arabic(
            audio_path,
            model_name=model_name,
            device=device,
            progress_callback=progress_callback
        )
        result["audio_path"] = audio_path if keep_audio else None
        return result
    finally:
        if not keep_audio and os.path.exists(audio_path):
            os.remove(audio_path)


if __name__ == "__main__":
    # Quick test
    import sys
    if len(sys.argv) > 1:
        result = transcribe_video(
            sys.argv[1],
            progress_callback=print
        )
        print(f"\nTranscription:\n{result['text']}")
        print(f"\nSegments: {len(result['segments'])}")
