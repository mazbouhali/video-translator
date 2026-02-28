#!/usr/bin/env python3
"""
Arabic Video Translator CLI

Translates Arabic speech in videos to English subtitles.

Usage:
    python -m app.main input_video.mp4 -o output_video.mp4
    python -m app.main input_video.mp4 --burn-in
    python -m app.main input_video.mp4 --srt-only
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional
import time

# Rich for beautiful progress output
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .transcribe import transcribe_video
from .translate import create_translator
from .subtitles import write_srt, embed_subtitles, create_dual_subtitle_video


def create_console():
    """Create console for output."""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_status(message: str, console=None, style: str = ""):
    """Print status message."""
    if console and RICH_AVAILABLE:
        console.print(f"[{style}]{message}[/{style}]" if style else message)
    else:
        print(message)


def print_header(console=None):
    """Print application header."""
    header = """
╔═══════════════════════════════════════════════════════════╗
║           🎬 Arabic Video Translator 🎬                   ║
║        Arabic Speech → English Subtitles                  ║
╚═══════════════════════════════════════════════════════════╝
    """
    if console and RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold blue]Arabic Video Translator[/bold blue]\n"
            "[dim]Arabic Speech → English Subtitles[/dim]",
            border_style="blue"
        ))
    else:
        print(header)


def process_video(
    input_path: str,
    output_path: Optional[str] = None,
    whisper_model: str = "large-v3",
    translation_model: str = "nllb",
    burn_in: bool = False,
    srt_only: bool = False,
    keep_arabic: bool = False,
    device: Optional[str] = None,
    console=None,
) -> dict:
    """
    Process a video: transcribe, translate, and add subtitles.
    
    Args:
        input_path: Path to input video
        output_path: Path for output (auto-generated if None)
        whisper_model: Whisper model for transcription
        translation_model: Translation model name
        burn_in: Burn subtitles into video
        srt_only: Only generate SRT file, no video output
        keep_arabic: Keep Arabic SRT alongside English
        device: Device for inference
        console: Rich console for output
        
    Returns:
        Dict with paths to generated files
    """
    input_path = Path(input_path).expanduser().resolve()
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    
    # Generate output paths
    if output_path:
        output_path = Path(output_path).expanduser().resolve()
    else:
        suffix = "_translated" if not srt_only else ""
        output_path = input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}"
    
    srt_path = output_path.parent / f"{output_path.stem}.srt"
    arabic_srt_path = output_path.parent / f"{output_path.stem}_arabic.srt"
    
    results = {"input": str(input_path)}
    
    # Progress callback
    def progress(msg):
        print_status(f"  → {msg}", console, "cyan")
    
    # Step 1: Transcribe
    print_status("\n📝 Step 1/3: Transcribing Arabic speech...", console, "bold yellow")
    start_time = time.time()
    
    transcription = transcribe_video(
        str(input_path),
        model_name=whisper_model,
        device=device,
        progress_callback=progress
    )
    
    transcribe_time = time.time() - start_time
    print_status(f"  ✓ Transcribed {len(transcription['segments'])} segments in {transcribe_time:.1f}s", console, "green")
    
    # Save Arabic SRT if requested
    if keep_arabic:
        write_srt(transcription["segments"], str(arabic_srt_path))
        results["arabic_srt"] = str(arabic_srt_path)
        print_status(f"  ✓ Saved Arabic subtitles: {arabic_srt_path.name}", console, "green")
    
    # Step 2: Translate
    print_status("\n🌐 Step 2/3: Translating to English...", console, "bold yellow")
    start_time = time.time()
    
    translator = create_translator(
        model=translation_model,
        device=device,
        progress_callback=progress
    )
    
    translated_segments = translator.translate_segments(
        transcription["segments"],
        show_progress=True
    )
    
    translate_time = time.time() - start_time
    print_status(f"  ✓ Translated {len(translated_segments)} segments in {translate_time:.1f}s", console, "green")
    
    # Write English SRT
    write_srt(translated_segments, str(srt_path))
    results["english_srt"] = str(srt_path)
    print_status(f"  ✓ Saved English subtitles: {srt_path.name}", console, "green")
    
    # Step 3: Embed subtitles (unless SRT-only mode)
    if not srt_only:
        print_status("\n🎬 Step 3/3: Embedding subtitles into video...", console, "bold yellow")
        start_time = time.time()
        
        mode = "burning in" if burn_in else "embedding as soft subs"
        progress(f"Processing video ({mode})...")
        
        output_video = embed_subtitles(
            str(input_path),
            str(srt_path),
            str(output_path),
            burn_in=burn_in
        )
        
        embed_time = time.time() - start_time
        results["video"] = output_video
        print_status(f"  ✓ Created video in {embed_time:.1f}s: {output_path.name}", console, "green")
    else:
        print_status("\n⏭️  Step 3/3: Skipped (SRT-only mode)", console, "dim")
    
    return results


def main():
    """Main CLI entry point."""
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
  %(prog)s video.mp4 --whisper-model base # Use smaller/faster model
        """
    )
    
    parser.add_argument(
        "input",
        help="Input video file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output video file path (default: input_translated.ext)"
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
        help="Also save Arabic transcription as SRT"
    )
    
    parser.add_argument(
        "--whisper-model",
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model for transcription (default: large-v3)"
    )
    
    parser.add_argument(
        "--translation-model",
        default="nllb",
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
    
    args = parser.parse_args()
    
    # Setup console
    console = None if args.quiet else create_console()
    
    # Print header
    if not args.quiet:
        print_header(console)
    
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
            console=console,
        )
        
        # Print summary
        if not args.quiet:
            print_status("\n" + "="*50, console)
            print_status("✅ Processing complete!", console, "bold green")
            print_status("", console)
            print_status("Generated files:", console, "bold")
            
            if "video" in results:
                print_status(f"  📹 Video: {results['video']}", console)
            if "english_srt" in results:
                print_status(f"  📄 English SRT: {results['english_srt']}", console)
            if "arabic_srt" in results:
                print_status(f"  📄 Arabic SRT: {results['arabic_srt']}", console)
        
        return 0
        
    except FileNotFoundError as e:
        print_status(f"\n❌ Error: {e}", console, "bold red")
        return 1
    except RuntimeError as e:
        print_status(f"\n❌ Error: {e}", console, "bold red")
        return 1
    except KeyboardInterrupt:
        print_status("\n\n⚠️  Interrupted by user", console, "yellow")
        return 130
    except Exception as e:
        print_status(f"\n❌ Unexpected error: {e}", console, "bold red")
        if console and RICH_AVAILABLE:
            console.print_exception()
        raise


if __name__ == "__main__":
    sys.exit(main())
