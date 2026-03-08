"""
SRT subtitle generation and video embedding.

This module handles:
- Generating SRT subtitle files from timestamped segments
- Parsing existing SRT files
- Embedding subtitles into videos (soft subs or burned-in)

Example:
    >>> from app.subtitles import generate_srt, embed_subtitles
    >>> segments = [{"start": 0.0, "end": 2.5, "text": "Hello!"}]
    >>> srt_content = generate_srt(segments)
    >>> embed_subtitles("video.mp4", "subs.srt", "output.mp4", burn_in=True)
"""

import subprocess
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
from datetime import timedelta

from .config import get_config


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format.
    
    Args:
        seconds: Time in seconds (can include decimals)
        
    Returns:
        Formatted timestamp string (HH:MM:SS,mmm)
        
    Example:
        >>> format_timestamp(3661.5)
        '01:01:01,500'
    """
    if seconds < 0:
        seconds = 0
    
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def parse_timestamp(timestamp: str) -> float:
    """
    Parse SRT timestamp to seconds.
    
    Args:
        timestamp: SRT format timestamp (HH:MM:SS,mmm)
        
    Returns:
        Time in seconds as float
        
    Raises:
        ValueError: If timestamp format is invalid
        
    Example:
        >>> parse_timestamp("01:30:45,500")
        5445.5
    """
    match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp.strip())
    if not match:
        raise ValueError(f"Invalid SRT timestamp format: {timestamp}")
    
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def generate_srt(segments: List[Dict[str, Any]]) -> str:
    """
    Generate SRT subtitle content from segments.
    
    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys
        
    Returns:
        SRT formatted string ready to write to file
        
    Example:
        >>> segments = [
        ...     {"start": 0.0, "end": 2.0, "text": "Hello"},
        ...     {"start": 2.5, "end": 4.0, "text": "World"},
        ... ]
        >>> print(generate_srt(segments))
        1
        00:00:00,000 --> 00:00:02,000
        Hello
        
        2
        00:00:02,500 --> 00:00:04,000
        World
    """
    srt_lines = []
    index = 1
    
    for segment in segments:
        text = segment.get("text", "").strip()
        
        # Skip empty segments
        if not text:
            continue
        
        start = format_timestamp(segment.get("start", 0))
        end = format_timestamp(segment.get("end", 0))
        
        srt_lines.append(str(index))
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")  # Blank line between entries
        
        index += 1
    
    return "\n".join(srt_lines)


def write_srt(
    segments: List[Dict[str, Any]],
    output_path: str,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Write segments to an SRT subtitle file.
    
    Args:
        segments: List of segment dicts with timing and text
        output_path: Path for output SRT file
        progress_callback: Optional progress callback
        
    Returns:
        Absolute path to written file
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    srt_content = generate_srt(segments)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    if progress_callback:
        progress_callback(f"Wrote subtitle file: {output_path.name}")
    
    return str(output_path)


def read_srt(srt_path: str) -> List[Dict[str, Any]]:
    """
    Parse an SRT file into segments.
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        List of segment dicts with 'start', 'end', 'text' keys
        
    Raises:
        FileNotFoundError: If SRT file doesn't exist
    """
    srt_path = Path(srt_path).expanduser().resolve()
    
    if not srt_path.exists():
        raise FileNotFoundError(f"SRT file not found: {srt_path}")
    
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    segments = []
    
    # Split by double newlines (SRT block separator)
    blocks = re.split(r'\n\n+', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        
        if len(lines) >= 3:
            # Line 1: index (ignored)
            # Line 2: timestamps
            # Line 3+: text (may be multiline)
            timestamp_line = lines[1]
            text_lines = lines[2:]
            
            # Parse timestamps
            match = re.match(
                r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})',
                timestamp_line
            )
            
            if match:
                segments.append({
                    "start": parse_timestamp(match.group(1)),
                    "end": parse_timestamp(match.group(2)),
                    "text": "\n".join(text_lines).strip(),
                })
    
    return segments


def embed_subtitles(
    video_path: str,
    srt_path: str,
    output_path: str,
    burn_in: bool = False,
    font_size: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Embed subtitles into video using ffmpeg.
    
    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_path: Path for output video
        burn_in: If True, burn subtitles permanently into video frames.
                 If False, add as toggleable soft subtitles.
        progress_callback: Optional progress callback
        
    Returns:
        Path to output video
        
    Raises:
        FileNotFoundError: If video or subtitle file doesn't exist
        RuntimeError: If ffmpeg fails
    """
    video_path = Path(video_path).expanduser().resolve()
    srt_path = Path(srt_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {srt_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load subtitle style config
    config = get_config()
    sub_config = config.subtitle
    
    # Use provided font_size or fall back to config
    actual_font_size = font_size if font_size is not None else sub_config.font_size
    
    if progress_callback:
        mode = "burning in" if burn_in else "embedding soft"
        progress_callback(f"Processing video ({mode} subtitles)...")
    
    if burn_in:
        # Burn subtitles into video frames (hardcoded, always visible)
        # Escape special characters for ffmpeg filter
        srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")
        
        # Build ASS-style format string for subtitles filter
        subtitle_filter = (
            f"subtitles='{srt_escaped}':force_style='"
            f"FontSize={actual_font_size},"
            f"PrimaryColour=&H00FFFFFF,"  # White (ABGR format)
            f"OutlineColour=&H00000000,"  # Black outline
            f"BorderStyle=3,"              # Opaque box
            f"Outline=2,"
            f"Shadow=1,"
            f"MarginV={sub_config.margin_v}"
            f"'"
        )
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", subtitle_filter,
            "-c:a", "copy",  # Copy audio without re-encoding
            "-y",            # Overwrite output
            str(output_path)
        ]
    else:
        # Add as soft subtitles (can be toggled in player)
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(srt_path),
            "-c", "copy",              # Copy all streams
            "-c:s", "mov_text",        # Subtitle codec for MP4
            "-metadata:s:s:0", "language=eng",  # Mark as English
            "-y",
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
        # Try simpler approach for burn-in if complex one fails
        if burn_in:
            if progress_callback:
                progress_callback("Retrying with simpler subtitle filter...")
            
            cmd_simple = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", f"subtitles={str(srt_path)}",
                "-c:a", "copy",
                "-y",
                str(output_path)
            ]
            
            try:
                subprocess.run(cmd_simple, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e2:
                raise RuntimeError(
                    f"ffmpeg failed to embed subtitles: {e2.stderr}\n"
                    "Make sure ffmpeg was compiled with libass support."
                )
        else:
            raise RuntimeError(f"ffmpeg failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Please install ffmpeg:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: choco install ffmpeg"
        )
    
    if progress_callback:
        progress_callback(f"Video created: {output_path.name}")
    
    return str(output_path)


def create_dual_subtitle_video(
    video_path: str,
    arabic_segments: List[Dict[str, Any]],
    english_segments: List[Dict[str, Any]],
    output_path: str,
    burn_in: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None
) -> Dict[str, str]:
    """
    Create a video with both Arabic and English subtitle files.
    
    Generates separate SRT files for both languages and embeds
    the English subtitles into the video.
    
    Args:
        video_path: Input video path
        arabic_segments: Original Arabic transcription segments
        english_segments: Translated English segments
        output_path: Output video path
        burn_in: Whether to burn English subtitles into video
        progress_callback: Optional progress callback
        
    Returns:
        Dict with paths: {'video': ..., 'arabic_srt': ..., 'english_srt': ...}
    """
    video_path = Path(video_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    
    base_name = output_path.stem
    output_dir = output_path.parent
    
    # Generate both SRT files
    arabic_srt = output_dir / f"{base_name}_arabic.srt"
    english_srt = output_dir / f"{base_name}_english.srt"
    
    write_srt(arabic_segments, str(arabic_srt), progress_callback)
    write_srt(english_segments, str(english_srt), progress_callback)
    
    # Create video with English subtitles
    final_video = embed_subtitles(
        str(video_path),
        str(english_srt),
        str(output_path),
        burn_in=burn_in,
        progress_callback=progress_callback
    )
    
    return {
        "video": final_video,
        "arabic_srt": str(arabic_srt),
        "english_srt": str(english_srt),
    }


# CLI test
if __name__ == "__main__":
    print("Testing SRT generation...\n")
    
    test_segments = [
        {"start": 0.0, "end": 2.5, "text": "Hello, world!"},
        {"start": 2.5, "end": 5.0, "text": "This is a test of the subtitle system."},
        {"start": 5.0, "end": 8.0, "text": "Subtitles are working correctly!"},
    ]
    
    print("Generated SRT:")
    print("-" * 40)
    print(generate_srt(test_segments))
    print("-" * 40)
    
    # Test timestamp functions
    print("\nTimestamp tests:")
    print(f"  format_timestamp(3661.5) = {format_timestamp(3661.5)}")
    print(f"  parse_timestamp('01:01:01,500') = {parse_timestamp('01:01:01,500')}")
