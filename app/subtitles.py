"""
SRT subtitle generation and video embedding.
"""

import subprocess
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import timedelta


def format_timestamp(seconds: float) -> str:
    """
    Convert seconds to SRT timestamp format (HH:MM:SS,mmm).
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted timestamp string
    """
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
        Time in seconds
    """
    match = re.match(r'(\d{2}):(\d{2}):(\d{2}),(\d{3})', timestamp)
    if not match:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def generate_srt(segments: List[Dict[str, Any]]) -> str:
    """
    Generate SRT content from segments.
    
    Args:
        segments: List of segment dicts with 'start', 'end', 'text' keys
        
    Returns:
        SRT formatted string
    """
    srt_lines = []
    
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp(segment.get("start", 0))
        end = format_timestamp(segment.get("end", 0))
        text = segment.get("text", "").strip()
        
        if not text:
            continue
        
        srt_lines.append(f"{i}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(text)
        srt_lines.append("")  # Blank line between entries
    
    return "\n".join(srt_lines)


def write_srt(segments: List[Dict[str, Any]], output_path: str) -> str:
    """
    Write segments to an SRT file.
    
    Args:
        segments: List of segment dicts
        output_path: Path for output SRT file
        
    Returns:
        Path to written file
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    srt_content = generate_srt(segments)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(srt_content)
    
    return str(output_path)


def read_srt(srt_path: str) -> List[Dict[str, Any]]:
    """
    Parse an SRT file into segments.
    
    Args:
        srt_path: Path to SRT file
        
    Returns:
        List of segment dicts
    """
    srt_path = Path(srt_path).expanduser().resolve()
    
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    segments = []
    blocks = re.split(r'\n\n+', content.strip())
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # First line is index, second is timestamps, rest is text
            timestamp_line = lines[1]
            text_lines = lines[2:]
            
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
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
    position: str = "bottom"
) -> str:
    """
    Embed subtitles into video using ffmpeg.
    
    Args:
        video_path: Path to input video
        srt_path: Path to SRT subtitle file
        output_path: Path for output video
        burn_in: If True, burn subtitles into video. If False, add as soft subs.
        font_size: Font size for burned-in subtitles
        font_color: Font color for burned-in subtitles
        outline_color: Outline color for burned-in subtitles
        position: Subtitle position ('top', 'center', 'bottom')
        
    Returns:
        Path to output video
    """
    video_path = Path(video_path).expanduser().resolve()
    srt_path = Path(srt_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not srt_path.exists():
        raise FileNotFoundError(f"Subtitle file not found: {srt_path}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if burn_in:
        # Burn subtitles into video (hardcoded)
        # Calculate vertical position
        if position == "top":
            margin_v = 20
        elif position == "center":
            margin_v = "(h-text_h)/2"
        else:  # bottom
            margin_v = "h-th-20"
        
        # Escape special characters in path for ffmpeg filter
        srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
        
        # Build subtitle filter
        subtitle_filter = (
            f"subtitles='{srt_escaped}':force_style='"
            f"FontSize={font_size},"
            f"PrimaryColour=&H00FFFFFF,"  # White
            f"OutlineColour=&H00000000,"  # Black outline
            f"BorderStyle=3,"
            f"Outline=2,"
            f"Shadow=1,"
            f"MarginV=30"
            f"'"
        )
        
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", subtitle_filter,
            "-c:a", "copy",
            "-y",
            str(output_path)
        ]
    else:
        # Add as soft subtitles (can be toggled on/off)
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-i", str(srt_path),
            "-c", "copy",
            "-c:s", "mov_text",  # For MP4 compatibility
            "-metadata:s:s:0", "language=eng",
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
        # Try alternative approach for burn-in if first fails
        if burn_in:
            # Use ASS subtitles approach
            cmd_alt = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", f"subtitles={str(srt_path)}",
                "-c:a", "copy",
                "-y",
                str(output_path)
            ]
            try:
                subprocess.run(cmd_alt, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e2:
                raise RuntimeError(f"ffmpeg failed to embed subtitles: {e2.stderr}")
        else:
            raise RuntimeError(f"ffmpeg failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
    
    return str(output_path)


def create_dual_subtitle_video(
    video_path: str,
    arabic_segments: List[Dict[str, Any]],
    english_segments: List[Dict[str, Any]],
    output_path: str,
    burn_in: bool = True
) -> Dict[str, str]:
    """
    Create a video with both Arabic and English subtitles.
    
    Args:
        video_path: Input video path
        arabic_segments: Original Arabic transcription segments
        english_segments: Translated English segments
        output_path: Output video path
        burn_in: Whether to burn in subtitles
        
    Returns:
        Dict with paths to output files
    """
    video_path = Path(video_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    
    base_name = output_path.stem
    output_dir = output_path.parent
    
    # Write both SRT files
    arabic_srt = output_dir / f"{base_name}_arabic.srt"
    english_srt = output_dir / f"{base_name}_english.srt"
    
    write_srt(arabic_segments, str(arabic_srt))
    write_srt(english_segments, str(english_srt))
    
    # Create video with English subtitles (primary use case)
    final_video = embed_subtitles(
        str(video_path),
        str(english_srt),
        str(output_path),
        burn_in=burn_in
    )
    
    return {
        "video": final_video,
        "arabic_srt": str(arabic_srt),
        "english_srt": str(english_srt),
    }


if __name__ == "__main__":
    # Test SRT generation
    test_segments = [
        {"start": 0.0, "end": 2.5, "text": "Hello, world!"},
        {"start": 2.5, "end": 5.0, "text": "This is a test."},
        {"start": 5.0, "end": 8.0, "text": "Subtitles are working!"},
    ]
    
    print("Generated SRT:")
    print(generate_srt(test_segments))
