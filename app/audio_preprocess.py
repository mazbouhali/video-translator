"""
Audio preprocessing for improved Arabic speech recognition.

Handles:
- Vocal separation from background music/drums using Demucs
- Audio normalization and enhancement
- Segment-based processing for long audio

Critical for nasheeds and political speeches where:
- Background music/percussion interferes with vocals
- Emotional/chanting delivery differs from normal speech
- Religious terminology needs accurate transcription
"""

import subprocess
import tempfile
import os
import shutil
from pathlib import Path
from typing import Optional, Callable, Tuple

# Check for demucs availability
try:
    import demucs.api
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False


def check_demucs() -> bool:
    """Check if Demucs is available for vocal separation."""
    return DEMUCS_AVAILABLE


def separate_vocals_demucs(
    audio_path: str,
    output_dir: Optional[str] = None,
    model: str = "htdemucs",
    two_stems: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[str, Optional[str]]:
    """
    Separate vocals from background music using Demucs.
    
    Demucs is a state-of-the-art music source separation model.
    Essential for nasheeds where drums/music interfere with vocals.
    
    Args:
        audio_path: Path to input audio file
        output_dir: Directory for output files (None = temp dir)
        model: Demucs model to use:
            - htdemucs: Hybrid Transformer (best quality, slower)
            - htdemucs_ft: Fine-tuned version (best overall)
            - hdemucs_mmi: Hybrid Demucs (faster)
        two_stems: If True, only separate vocals vs everything else
        progress_callback: Optional progress callback
        
    Returns:
        Tuple of (vocals_path, accompaniment_path)
        
    Raises:
        ImportError: If demucs is not installed
        RuntimeError: If separation fails
    """
    if not DEMUCS_AVAILABLE:
        raise ImportError(
            "Demucs not installed. Install with:\n"
            "  pip install demucs\n"
            "For GPU acceleration, ensure PyTorch is installed with CUDA."
        )
    
    audio_path = Path(audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="demucs_")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if progress_callback:
        progress_callback(f"Separating vocals using Demucs ({model})...")
    
    try:
        # Use demucs API for separation
        separator = demucs.api.Separator(model=model)
        
        # Separate with two stems (vocals vs rest) for speed
        if two_stems:
            origin, separated = separator.separate_audio_file(
                str(audio_path),
                two_stems="vocals"
            )
        else:
            origin, separated = separator.separate_audio_file(str(audio_path))
        
        # Save the separated stems
        vocals_path = output_dir / f"{audio_path.stem}_vocals.wav"
        accompaniment_path = output_dir / f"{audio_path.stem}_accompaniment.wav"
        
        # Write vocals
        import torchaudio
        if "vocals" in separated:
            torchaudio.save(str(vocals_path), separated["vocals"], separator.samplerate)
        
        # Write accompaniment (everything else)
        if two_stems and "no_vocals" in separated:
            torchaudio.save(str(accompaniment_path), separated["no_vocals"], separator.samplerate)
        
        if progress_callback:
            progress_callback(f"Vocal separation complete: {vocals_path.name}")
        
        return str(vocals_path), str(accompaniment_path) if accompaniment_path.exists() else None
        
    except Exception as e:
        raise RuntimeError(f"Demucs separation failed: {e}")


def separate_vocals_cli(
    audio_path: str,
    output_dir: Optional[str] = None,
    model: str = "htdemucs",
    progress_callback: Optional[Callable[[str], None]] = None
) -> Tuple[str, Optional[str]]:
    """
    Separate vocals using Demucs CLI (fallback method).
    
    Uses the command line interface which is more reliable for
    systems with complex audio setups.
    
    Args:
        audio_path: Path to input audio
        output_dir: Output directory
        model: Demucs model name
        progress_callback: Progress callback
        
    Returns:
        Tuple of (vocals_path, accompaniment_path)
    """
    audio_path = Path(audio_path).expanduser().resolve()
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="demucs_")
    output_dir = Path(output_dir)
    
    if progress_callback:
        progress_callback(f"Separating vocals with Demucs CLI ({model})...")
    
    # Run demucs CLI
    cmd = [
        "python3", "-m", "demucs",
        "--two-stems=vocals",  # Only vocals vs rest
        "-n", model,
        "-o", str(output_dir),
        str(audio_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Demucs CLI failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("Demucs CLI not found. Install with: pip install demucs")
    
    # Find output files
    separated_dir = output_dir / model / audio_path.stem
    vocals_path = separated_dir / "vocals.wav"
    no_vocals_path = separated_dir / "no_vocals.wav"
    
    if not vocals_path.exists():
        raise RuntimeError(f"Vocals file not found at {vocals_path}")
    
    if progress_callback:
        progress_callback("Vocal separation complete")
    
    return str(vocals_path), str(no_vocals_path) if no_vocals_path.exists() else None


def normalize_audio(
    audio_path: str,
    output_path: Optional[str] = None,
    target_db: float = -20.0,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Normalize audio loudness using ffmpeg.
    
    Important for nasheeds/speeches with variable volume levels.
    
    Args:
        audio_path: Input audio path
        output_path: Output path (None = create temp file)
        target_db: Target loudness in dB (default -20)
        progress_callback: Progress callback
        
    Returns:
        Path to normalized audio
    """
    audio_path = Path(audio_path).expanduser().resolve()
    
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
    
    if progress_callback:
        progress_callback("Normalizing audio levels...")
    
    # Use ffmpeg loudnorm filter
    cmd = [
        "ffmpeg",
        "-i", str(audio_path),
        "-af", f"loudnorm=I={target_db}:TP=-1.5:LRA=11",
        "-ar", "16000",  # Whisper optimal sample rate
        "-ac", "1",      # Mono
        "-y",
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        # Fall back to simple volume normalization
        cmd = [
            "ffmpeg",
            "-i", str(audio_path),
            "-af", "volume=1.5",
            "-ar", "16000",
            "-ac", "1",
            "-y",
            str(output_path)
        ]
        subprocess.run(cmd, capture_output=True, check=True)
    
    if progress_callback:
        progress_callback("Audio normalization complete")
    
    return str(output_path)


def preprocess_for_transcription(
    audio_path: str,
    separate_vocals: bool = True,
    normalize: bool = True,
    keep_intermediate: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Full preprocessing pipeline for Arabic audio transcription.
    
    Optimized for nasheeds and political speeches:
    1. Separate vocals from background music/drums (if enabled)
    2. Normalize audio levels
    3. Convert to Whisper-optimal format (16kHz mono WAV)
    
    Args:
        audio_path: Input audio/video path
        separate_vocals: Use Demucs to isolate vocals
        normalize: Normalize audio levels
        keep_intermediate: Keep intermediate files
        progress_callback: Progress callback
        
    Returns:
        Path to preprocessed audio ready for transcription
    """
    audio_path = Path(audio_path).expanduser().resolve()
    temp_files = []
    
    try:
        current_audio = str(audio_path)
        
        # Step 1: Vocal separation (for music/nasheeds)
        if separate_vocals:
            if DEMUCS_AVAILABLE:
                try:
                    vocals_path, _ = separate_vocals_demucs(
                        current_audio,
                        progress_callback=progress_callback
                    )
                    current_audio = vocals_path
                    temp_files.append(vocals_path)
                except Exception as e:
                    if progress_callback:
                        progress_callback(f"Vocal separation failed, using original: {e}")
            else:
                if progress_callback:
                    progress_callback("Demucs not available, skipping vocal separation")
        
        # Step 2: Audio normalization
        if normalize:
            normalized_path = normalize_audio(
                current_audio,
                progress_callback=progress_callback
            )
            if current_audio != str(audio_path):
                temp_files.append(normalized_path)
            current_audio = normalized_path
        
        return current_audio
        
    finally:
        # Cleanup intermediate files if not keeping
        if not keep_intermediate:
            for f in temp_files[:-1]:  # Keep the last one (final output)
                try:
                    os.remove(f)
                except OSError:
                    pass


# ============================================================================
# Singing/Chanting Detection (Heuristic)
# ============================================================================

def detect_singing_characteristics(audio_path: str) -> dict:
    """
    Detect characteristics that suggest singing/chanting vs speech.
    
    Useful for automatically deciding whether to use vocal separation.
    
    Returns dict with:
        - has_music: bool - Detected background music
        - is_chanting: bool - Detected rhythmic/chanting patterns
        - avg_pitch_variance: float - Higher = more melodic
        
    Note: Requires librosa for analysis.
    """
    try:
        import librosa
        import numpy as np
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=16000, duration=60)  # First 60 seconds
        
        # Analyze pitch variance (singing has more variance)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_variance = np.var(pitch_values) if len(pitch_values) > 0 else 0
        
        # Detect percussive content (drums/duff)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        percussive_ratio = np.sum(np.abs(y_percussive)) / (np.sum(np.abs(y_harmonic)) + 1e-6)
        
        # Detect tempo regularity (chanting is often rhythmic)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        return {
            "has_music": percussive_ratio > 0.3,
            "is_melodic": pitch_variance > 1000,
            "tempo": float(tempo),
            "percussive_ratio": float(percussive_ratio),
            "recommendation": "separate_vocals" if percussive_ratio > 0.3 else "direct_transcription"
        }
        
    except ImportError:
        return {
            "error": "librosa not installed",
            "recommendation": "separate_vocals"  # Default to safe option
        }
    except Exception as e:
        return {
            "error": str(e),
            "recommendation": "separate_vocals"
        }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    
    print("Audio Preprocessing Module")
    print("=" * 50)
    print(f"Demucs available: {'✓' if DEMUCS_AVAILABLE else '✗'}")
    print()
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.audio_preprocess <audio_file> [--separate] [--normalize]")
        sys.exit(0)
    
    audio_file = sys.argv[1]
    do_separate = "--separate" in sys.argv
    do_normalize = "--normalize" in sys.argv or not do_separate
    
    print(f"Input: {audio_file}")
    print(f"Separate vocals: {do_separate}")
    print(f"Normalize: {do_normalize}")
    
    # Detect characteristics first
    print("\nAnalyzing audio characteristics...")
    chars = detect_singing_characteristics(audio_file)
    print(f"  Has music/percussion: {chars.get('has_music', 'unknown')}")
    print(f"  Is melodic: {chars.get('is_melodic', 'unknown')}")
    print(f"  Recommendation: {chars.get('recommendation', 'unknown')}")
    
    # Run preprocessing
    output = preprocess_for_transcription(
        audio_file,
        separate_vocals=do_separate,
        normalize=do_normalize,
        keep_intermediate=True,
        progress_callback=lambda msg: print(f"  → {msg}")
    )
    
    print(f"\nOutput: {output}")
