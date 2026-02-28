#!/usr/bin/env python3
"""
Arabic Video Translator
Translates Arabic video/audio to English with subtitles.

No API keys required - runs entirely locally using:
- OpenAI Whisper for speech recognition
- Google Translate (via deep-translator) for translation
- FFmpeg for video processing

Usage:
    python main.py --watch              # Watch input/ folder continuously
    python main.py --file video.mp4     # Process single file
"""

import os
import sys
import time
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

try:
    import click
    import whisper
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    from deep_translator import GoogleTranslator
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Run: pip install -r requirements.txt")
    sys.exit(1)

# =============================================================================
# Configuration (from environment variables)
# =============================================================================
INPUT_DIR = Path(os.getenv("INPUT_DIR", "/app/input"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))
TEMP_DIR = Path(os.getenv("TEMP_DIR", "/app/temp"))

SOURCE_LANG = os.getenv("SOURCE_LANG", "ar")
TARGET_LANG = os.getenv("TARGET_LANG", "en")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")
WATCH_INTERVAL = int(os.getenv("WATCH_INTERVAL", "5"))
DELETE_SOURCE = os.getenv("DELETE_SOURCE", "false").lower() == "true"

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".flv", ".wmv"}

# Global model instance (loaded once)
_model = None


# =============================================================================
# Core Functions
# =============================================================================

def get_model():
    """Load Whisper model (cached after first load)."""
    global _model
    if _model is None:
        device = "cuda" if os.getenv("DEVICE") != "cpu" else "cpu"
        
        # Auto-detect CUDA availability
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except:
            device = "cpu"
        
        print(f"📥 Loading Whisper model '{WHISPER_MODEL}' on {device.upper()}...")
        _model = whisper.load_model(WHISPER_MODEL, device=device)
        print(f"✅ Model loaded successfully")
    
    return _model


def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio track from video file."""
    print(f"🎵 Extracting audio...")
    
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn",                    # No video
        "-acodec", "pcm_s16le",   # PCM format for Whisper
        "-ar", "16000",           # 16kHz sample rate
        "-ac", "1",               # Mono
        str(audio_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ FFmpeg error: {result.stderr[:500]}")
        return False
    
    return True


def transcribe(audio_path: Path) -> dict:
    """Transcribe audio using Whisper."""
    print(f"📝 Transcribing {SOURCE_LANG.upper()} audio...")
    
    model = get_model()
    result = model.transcribe(
        str(audio_path),
        language=SOURCE_LANG,
        task="transcribe",
        verbose=False
    )
    
    print(f"✅ Found {len(result['segments'])} segments")
    return result


def translate_text(text: str) -> str:
    """Translate text using Google Translate (no API key needed)."""
    if not text or not text.strip():
        return ""
    
    try:
        translator = GoogleTranslator(source=SOURCE_LANG, target=TARGET_LANG)
        return translator.translate(text)
    except Exception as e:
        print(f"⚠️ Translation failed: {e}")
        return text  # Return original on failure


def seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def generate_srt(segments: list, output_path: Path, translate: bool = False):
    """Generate SRT subtitle file from segments."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            start = seconds_to_srt_time(seg["start"])
            end = seconds_to_srt_time(seg["end"])
            text = seg["text"].strip()
            
            if translate:
                text = translate_text(text)
            
            f.write(f"{i}\n{start} --> {end}\n{text}\n\n")


def burn_subtitles(video_path: Path, srt_path: Path, output_path: Path) -> bool:
    """Burn subtitles into video using FFmpeg."""
    print(f"🔥 Burning subtitles into video...")
    
    # Escape special characters in path for FFmpeg filter
    srt_escaped = str(srt_path).replace("\\", "/").replace(":", "\\:")
    
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"subtitles='{srt_escaped}':force_style='FontSize=22,PrimaryColour=&HFFFFFF,OutlineColour=&H000000,Outline=2'",
        "-c:a", "copy",
        "-c:v", "libx264",
        "-preset", "fast",
        str(output_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ Subtitle burn failed, copying original video")
        shutil.copy(video_path, output_path)
        return False
    
    return True


def process_video(video_path: Path) -> bool:
    """
    Full translation pipeline:
    1. Extract audio
    2. Transcribe (Arabic)
    3. Translate (to English)
    4. Generate subtitle files
    5. Burn subtitles into video
    """
    print(f"\n{'='*60}")
    print(f"🎬 Processing: {video_path.name}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    base_name = video_path.stem
    
    # Output paths
    audio_path = TEMP_DIR / f"{base_name}_audio.wav"
    srt_source = OUTPUT_DIR / f"{base_name}_{SOURCE_LANG}.srt"
    srt_target = OUTPUT_DIR / f"{base_name}_{TARGET_LANG}.srt"
    output_video = OUTPUT_DIR / f"{base_name}_translated.mp4"
    
    try:
        # Step 1: Extract audio
        if not extract_audio(video_path, audio_path):
            return False
        
        # Step 2: Transcribe
        result = transcribe(audio_path)
        segments = result.get("segments", [])
        
        if not segments:
            print("⚠️ No speech detected in video")
            return False
        
        # Step 3 & 4: Generate subtitles
        print(f"📄 Generating subtitles...")
        generate_srt(segments, srt_source, translate=False)
        print(f"   ✓ {srt_source.name}")
        
        generate_srt(segments, srt_target, translate=True)
        print(f"   ✓ {srt_target.name}")
        
        # Step 5: Burn subtitles
        burn_subtitles(video_path, srt_target, output_video)
        
        # Cleanup temp files
        audio_path.unlink(missing_ok=True)
        
        # Optionally delete source
        if DELETE_SOURCE:
            video_path.unlink(missing_ok=True)
            print(f"🗑️ Deleted source file")
        
        elapsed = time.time() - start_time
        print(f"\n✅ Complete in {elapsed:.1f}s")
        print(f"   📹 Video:    {output_video.name}")
        print(f"   📄 Arabic:   {srt_source.name}")
        print(f"   📄 English:  {srt_target.name}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Watch Mode
# =============================================================================

class VideoHandler(FileSystemEventHandler):
    """Filesystem watcher for new video files."""
    
    def __init__(self):
        self.processed = set()
        self.processing = set()
    
    def on_created(self, event):
        if event.is_directory:
            return
        self._handle_file(Path(event.src_path))
    
    def on_moved(self, event):
        if event.is_directory:
            return
        self._handle_file(Path(event.dest_path))
    
    def _handle_file(self, path: Path):
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            return
        
        if path in self.processed or path in self.processing:
            return
        
        # Wait for file to finish copying
        self.processing.add(path)
        time.sleep(2)
        
        # Check file is stable (not still being written)
        try:
            size1 = path.stat().st_size
            time.sleep(1)
            size2 = path.stat().st_size
            if size1 != size2:
                print(f"⏳ Waiting for {path.name} to finish copying...")
                time.sleep(5)
        except:
            pass
        
        self.processed.add(path)
        self.processing.discard(path)
        process_video(path)


def watch_folder():
    """Watch input folder for new videos."""
    print(f"👀 Watching: {INPUT_DIR}")
    print(f"📂 Output:   {OUTPUT_DIR}")
    print(f"🔧 Model:    {WHISPER_MODEL}")
    print(f"🌍 Translate: {SOURCE_LANG.upper()} → {TARGET_LANG.upper()}")
    print(f"\nDrop video files in {INPUT_DIR} to translate them.")
    print("Press Ctrl+C to stop.\n")
    
    # Process existing files first
    handler = VideoHandler()
    for video_file in sorted(INPUT_DIR.iterdir()):
        if video_file.suffix.lower() in VIDEO_EXTENSIONS:
            if video_file not in handler.processed:
                handler.processed.add(video_file)
                process_video(video_file)
    
    # Watch for new files
    observer = Observer()
    observer.schedule(handler, str(INPUT_DIR), recursive=False)
    observer.start()
    
    try:
        while True:
            time.sleep(WATCH_INTERVAL)
    except KeyboardInterrupt:
        print("\n\n👋 Stopping...")
        observer.stop()
    
    observer.join()


# =============================================================================
# CLI
# =============================================================================

@click.command()
@click.option("--watch", is_flag=True, help="Watch input folder for new videos")
@click.option("--file", "filepath", type=click.Path(exists=True), help="Process single video file")
def main(watch, filepath):
    """
    Arabic Video Translator
    
    Translates Arabic audio/video to English with subtitles.
    No API keys required - runs entirely locally.
    """
    # Ensure directories exist
    for d in [INPUT_DIR, OUTPUT_DIR, TEMP_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    if filepath:
        process_video(Path(filepath))
    elif watch:
        watch_folder()
    else:
        # Default to watch mode
        watch_folder()


if __name__ == "__main__":
    main()
