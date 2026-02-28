#!/usr/bin/env python3
"""
Arabic Video Translator - Command Line Interface

Translates Arabic speech in videos to English subtitles using:
- Whisper for speech recognition
- NLLB/MarianMT for translation
- ffmpeg for video processing

Usage:
    python -m app.main video.mp4
    python -m app.main video.mp4 --burn-in
    python -m app.main video.mp4 --srt-only
    python -m app.main video.mp4 -o output.mp4 --keep-arabic

For web interface, run:
    python -m app.web
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Rich for progress display (graceful fallback if not installed)
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .config import get_config, Config
from .transcribe import transcribe_video, check_ffmpeg
from .translate import create_translator
from .subtitles import write_srt, embed_subtitles


class ProgressReporter:
    """Handles progress reporting with optional Rich support."""
    
    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.console = Console() if RICH_AVAILABLE and not quiet else None
    
    def print(self, message: str, style: str = "") -> None:
        """Print a message with optional styling."""
        if self.quiet:
            return
        
        if self.console and style:
            self.console.print(f"[{style}]{message}[/{style}]")
        elif self.console:
            self.console.print(message)
        else:
            print(message)
    
    def header(self) -> None:
        """Print application header."""
        if self.quiet:
            return
        
        if self.console:
            self.console.print(Panel.fit(
                "[bold blue]🎬 Arabic Video Translator[/bold blue]\n"
                "[dim]Arabic Speech → English Subtitles[/dim]",
                border_style="blue"
            ))
        else:
            print("\n" + "=" * 50)
            print("  🎬 Arabic Video Translator")
            print("  Arabic Speech → English Subtitles")
            print("=" * 50 + "\n")
    
    def step(self, message: str) -> None:
        """Print a step/progress message."""
        self.print(f"  → {message}", "cyan")
    
    def success(self, message: str) -> None:
        """Print a success message."""
        self.print(f"  ✓ {message}", "green")
    
    def error(self, message: str) -> None:
        """Print an error message."""
        self.print(f"  ✗ {message}", "bold red")
    
    def section(self, title: str) -> None:
        """Print a section header."""
        self.print(f"\n{title}", "bold yellow")


def process_video(
    input_path: str,
    output_path: Optional[str] = None,
    whisper_model: Optional[str] = None,
    translation_model: Optional[str] = None,
    burn_in: bool = False,
    srt_only: bool = False,
    keep_arabic: bool = False,
    device: Optional[str] = None,
    reporter: Optional[ProgressReporter] = None,
) -> Dict[str, Any]:
    """
    Process a video: transcribe Arabic, translate to English, add subtitles.
    
    Args:
        input_path: Path to input video file
        output_path: Path for output (auto-generated if None)
        whisper_model: Whisper model for transcription
        translation_model: Translation model name
        burn_in: Burn subtitles into video (permanent)
        srt_only: Only generate SRT file, no video output
        keep_arabic: Also save Arabic transcription as SRT
        device: Device for inference (cuda/mps/cpu/auto)
        reporter: Progress reporter instance
        
    Returns:
        Dict with paths to generated files
        
    Raises:
        FileNotFoundError: If input video doesn't exist
        RuntimeError: If processing fails
    """
    if reporter is None:
        reporter = ProgressReporter(quiet=True)
    
    # Validate input
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Check ffmpeg
    if not check_ffmpeg():
        raise RuntimeError(
            "ffmpeg is required but not found. Install it:\n"
            "  macOS: brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: choco install ffmpeg"
        )
    
    # Load config
    config = get_config()
    
    # Generate output paths
    if output_path:
        output_path = Path(output_path).expanduser().resolve()
    else:
        suffix = "_translated" if not srt_only else ""
        output_dir = Path(config.output_dir) if config.output_dir else input_path.parent
        output_path = output_dir / f"{input_path.stem}{suffix}{input_path.suffix}"
    
    srt_path = output_path.parent / f"{output_path.stem}.srt"
    arabic_srt_path = output_path.parent / f"{output_path.stem}_arabic.srt"
    
    results = {"input": str(input_path)}
    
    # === Step 1: Transcription ===
    reporter.section("📝 Step 1/3: Transcribing Arabic speech...")
    start_time = time.time()
    
    transcription = transcribe_video(
        str(input_path),
        model_name=whisper_model,
        device=device,
        progress_callback=reporter.step
    )
    
    elapsed = time.time() - start_time
    reporter.success(f"Transcribed {len(transcription['segments'])} segments in {elapsed:.1f}s")
    
    # Save Arabic SRT if requested
    if keep_arabic:
        write_srt(transcription["segments"], str(arabic_srt_path))
        results["arabic_srt"] = str(arabic_srt_path)
        reporter.success(f"Saved Arabic subtitles: {arabic_srt_path.name}")
    
    # === Step 2: Translation ===
    reporter.section("🌐 Step 2/3: Translating to English...")
    start_time = time.time()
    
    translator = create_translator(
        model=translation_model,
        device=device,
        progress_callback=reporter.step
    )
    
    translated_segments = translator.translate_segments(
        transcription["segments"],
        show_progress=True
    )
    
    elapsed = time.time() - start_time
    reporter.success(f"Translated {len(translated_segments)} segments in {elapsed:.1f}s")
    
    # Write English SRT
    write_srt(translated_segments, str(srt_path))
    results["english_srt"] = str(srt_path)
    reporter.success(f"Saved English subtitles: {srt_path.name}")
    
    # === Step 3: Video embedding ===
    if not srt_only:
        reporter.section("🎬 Step 3/3: Embedding subtitles into video...")
        start_time = time.time()
        
        output_video = embed_subtitles(
            str(input_path),
            str(srt_path),
            str(output_path),
            burn_in=burn_in,
            progress_callback=reporter.step
        )
        
        elapsed = time.time() - start_time
        results["video"] = output_video
        reporter.success(f"Created video in {elapsed:.1f}s: {output_path.name}")
    else:
        reporter.section("⏭️  Step 3/3: Skipped (SRT-only mode)")
    
    return results


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(
        description="Translate Arabic speech in videos to English subtitles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s video.mp4                      # Create video with soft subtitles
  %(prog)s video.mp4 -o translated.mp4    # Specify output path
  %(prog)s video.mp4 --burn-in            # Burn subtitles into video
  %(prog)s video.mp4 --srt-only           # Only generate SRT file
  %(prog)s video.mp4 --keep-arabic        # Also save Arabic transcript

Environment Variables:
  AVT_WHISPER_MODEL    Whisper model (default: large-v3)
  AVT_TRANSLATION_MODEL Translation model (default: nllb)
  AVT_DEVICE           Compute device (cuda/mps/cpu)
  AVT_OUTPUT_DIR       Default output directory

Web Interface:
  python -m app.web    # Launch Gradio web interface
        """
    )
    
    parser.add_argument(
        "input",
        help="Input video file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output video file path (default: <input>_translated.<ext>)"
    )
    
    parser.add_argument(
        "--burn-in",
        action="store_true",
        help="Burn subtitles into video (permanent, visible on all players)"
    )
    
    parser.add_argument(
        "--srt-only",
        action="store_true",
        help="Only generate SRT file, don't create video output"
    )
    
    parser.add_argument(
        "--keep-arabic",
        action="store_true",
        help="Also save Arabic transcription as SRT file"
    )
    
    parser.add_argument(
        "--whisper-model",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model for transcription (default: large-v3)"
    )
    
    parser.add_argument(
        "--translation-model",
        choices=["nllb", "nllb-large", "marian"],
        help="Translation model (default: nllb)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "mps"],
        help="Device for inference (default: auto-detect)"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    args = parser.parse_args()
    
    # Setup progress reporter
    reporter = ProgressReporter(quiet=args.quiet)
    reporter.header()
    
    try:
        results = process_video(
            input_path=args.input,
            output_path=args.output,
            whisper_model=args.whisper_model,
            translation_model=args.translation_model,
            burn_in=args.burn_in,
            srt_only=args.srt_only,
            keep_arabic=args.keep_arabic,
            device=args.device,
            reporter=reporter,
        )
        
        # Print summary
        if not args.quiet:
            reporter.print("\n" + "=" * 50)
            reporter.print("✅ Processing complete!", "bold green")
            reporter.print("")
            reporter.print("Generated files:", "bold")
            
            if "video" in results:
                reporter.print(f"  📹 Video: {results['video']}")
            if "english_srt" in results:
                reporter.print(f"  📄 English SRT: {results['english_srt']}")
            if "arabic_srt" in results:
                reporter.print(f"  📄 Arabic SRT: {results['arabic_srt']}")
        
        return 0
        
    except FileNotFoundError as e:
        reporter.error(str(e))
        return 1
    except RuntimeError as e:
        reporter.error(str(e))
        return 1
    except KeyboardInterrupt:
        reporter.print("\n\n⚠️  Interrupted by user", "yellow")
        return 130
    except Exception as e:
        reporter.error(f"Unexpected error: {e}")
        if RICH_AVAILABLE and reporter.console:
            reporter.console.print_exception()
        return 1


if __name__ == "__main__":
    sys.exit(main())
