#!/usr/bin/env python3
"""
Arabic Video Translator - Web Interface

A Gradio-based web UI for translating Arabic videos to English.
Supports drag-and-drop video upload with real-time progress.

Usage:
    python -m app.web
    python -m app.web --share  # Create public link
    python -m app.web --port 7860

Environment Variables:
    AVT_SERVER_PORT     Server port (default: 7860)
    AVT_SERVER_HOST     Server host (default: 127.0.0.1)
    AVT_SHARE           Create public link (default: false)
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple, Generator
import argparse

import gradio as gr

from .config import get_config
from .transcribe import transcribe_video, check_ffmpeg
from .translate import create_translator
from .subtitles import write_srt, embed_subtitles, generate_srt


# Global state for caching translator
_translator_cache = {}


def get_translator(model: str, device: Optional[str] = None):
    """Get or create a cached translator instance."""
    cache_key = f"{model}_{device}"
    if cache_key not in _translator_cache:
        # Handle backend selection vs model selection
        if model in ("llm", "two_pass"):
            _translator_cache[cache_key] = create_translator(backend=model, device=device)
        else:
            _translator_cache[cache_key] = create_translator(backend="nllb", model=model, device=device)
    return _translator_cache[cache_key]


def process_video_gradio(
    video_path: str,
    whisper_model: str,
    translation_model: str,
    output_mode: str,
    font_size: int,
    keep_arabic: bool,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """
    Process video for Gradio interface.
    
    Args:
        video_path: Path to uploaded video
        whisper_model: Whisper model selection
        translation_model: Translation model selection
        output_mode: "Soft Subtitles", "Burn-in Subtitles", or "SRT Only"
        keep_arabic: Whether to generate Arabic SRT
        progress: Gradio progress tracker
        
    Returns:
        Tuple of (output_video_path, english_srt_path, arabic_srt_path, status_message)
    """
    if not video_path:
        return None, None, None, "❌ Please upload a video file."
    
    if not check_ffmpeg():
        return None, None, None, (
            "❌ ffmpeg not found. Please install ffmpeg:\n"
            "• macOS: brew install ffmpeg\n"
            "• Ubuntu: sudo apt install ffmpeg\n"
            "• Windows: choco install ffmpeg"
        )
    
    try:
        video_path = Path(video_path)
        output_dir = Path(tempfile.mkdtemp(prefix="avt_"))
        
        burn_in = output_mode == "Burn-in Subtitles"
        srt_only = output_mode == "SRT Only"
        
        # Determine output paths
        output_video_path = output_dir / f"{video_path.stem}_translated{video_path.suffix}"
        english_srt_path = output_dir / f"{video_path.stem}_english.srt"
        arabic_srt_path = output_dir / f"{video_path.stem}_arabic.srt" if keep_arabic else None
        
        status_parts = []
        
        import time as time_module
        start_time = time_module.time()
        
        def elapsed():
            return time_module.time() - start_time
        
        def eta(pct):
            if pct <= 0:
                return ""
            remaining = (elapsed() / pct) * (1 - pct)
            if remaining < 60:
                return f" (~{int(remaining)}s left)"
            return f" (~{int(remaining/60)}m left)"
        
        # === Step 1: Transcription ===
        progress(0.1, desc="🎙️ Loading Whisper model...")
        
        progress(0.15, desc="🎙️ Extracting audio...")
        time.sleep(0.1)  # Small delay for UI update
        
        progress(0.2, desc=f"🎙️ Transcribing Arabic speech...{eta(0.2)}")
        transcription = transcribe_video(
            str(video_path),
            model_name=whisper_model,
            keep_audio=False
        )
        
        num_segments = len(transcription["segments"])
        status_parts.append(f"✓ Transcribed {num_segments} segments ({int(elapsed())}s)")
        
        # Save Arabic SRT if requested
        if keep_arabic and arabic_srt_path:
            write_srt(transcription["segments"], str(arabic_srt_path))
            status_parts.append(f"✓ Saved Arabic subtitles")
        
        # === Step 2: Translation ===
        progress(0.5, desc="🌐 Loading translation model...")
        
        translator = get_translator(translation_model)
        
        progress(0.55, desc=f"🌐 Translating to English...{eta(0.55)}")
        
        translated_segments = []
        for i, segment in enumerate(transcription["segments"]):
            # Update progress during translation
            pct = 0.55 + (0.3 * (i / max(num_segments, 1)))
            progress(pct, desc=f"🌐 Translating segment {i+1}/{num_segments}{eta(pct)}")
            
            result = translator.translate(segment.get("text", ""))
            translated_segments.append({
                **segment,
                "original_text": segment.get("text", ""),
                "text": result.translated,
            })
        
        status_parts.append(f"✓ Translated {len(translated_segments)} segments")
        
        # Write English SRT
        write_srt(translated_segments, str(english_srt_path))
        status_parts.append(f"✓ Generated English subtitles")
        
        # === Step 3: Video embedding ===
        final_video_path = None
        
        if not srt_only:
            progress(0.9, desc="🎬 Embedding subtitles into video...")
            
            embed_subtitles(
                str(video_path),
                str(english_srt_path),
                str(output_video_path),
                burn_in=burn_in,
                font_size=font_size
            )
            
            final_video_path = str(output_video_path)
            mode_desc = "burned-in" if burn_in else "soft"
            status_parts.append(f"✓ Created video with {mode_desc} subtitles")
        
        progress(1.0, desc="✅ Complete!")
        
        # Build final status
        status = "**Processing Complete!**\n\n" + "\n".join(status_parts)
        
        return (
            final_video_path,
            str(english_srt_path),
            str(arabic_srt_path) if arabic_srt_path else None,
            status
        )
        
    except Exception as e:
        return None, None, None, f"❌ Error: {str(e)}"


def reapply_subtitles_gradio(
    video_path: str,
    srt_path: str,
    output_mode: str,
    font_size: int,
    progress: gr.Progress = gr.Progress()
) -> Tuple[Optional[str], str]:
    """
    Re-apply existing SRT to video (skip transcription/translation).
    """
    if not video_path:
        return None, "❌ Please upload a video file."
    if not srt_path:
        return None, "❌ Please upload an SRT file."
    
    try:
        video_path = Path(video_path)
        srt_path = Path(srt_path)
        output_dir = Path(tempfile.mkdtemp(prefix="avt_reapply_"))
        
        burn_in = output_mode == "Burn-in Subtitles"
        output_video_path = output_dir / f"{video_path.stem}_subtitled{video_path.suffix}"
        
        progress(0.3, desc="🎬 Embedding subtitles...")
        
        embed_subtitles(
            str(video_path),
            str(srt_path),
            str(output_video_path),
            burn_in=burn_in,
            font_size=font_size
        )
        
        progress(1.0, desc="✅ Complete!")
        
        mode_desc = "burned-in" if burn_in else "soft"
        return str(output_video_path), f"✅ Created video with {mode_desc} subtitles"
        
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    
    # Custom CSS for better styling
    css = """
    .main-title {
        text-align: center;
        margin-bottom: 1rem;
    }
    .output-files {
        margin-top: 1rem;
    }
    """
    
    with gr.Blocks(
        title="Arabic Video Translator",
        theme=gr.themes.Soft(),
        css=css
    ) as interface:
        
        # Header
        gr.Markdown(
            """
            # 🎬 Arabic Video Translator
            
            Upload an Arabic video and get English subtitles automatically.
            Uses Whisper for speech recognition and NLLB for translation.
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Row():
            # Left column: Input
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Input")
                
                video_input = gr.Video(
                    label="Upload Video",
                    sources=["upload"],
                )
                
                with gr.Accordion("⚙️ Settings", open=False):
                    whisper_model = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
                        value="large-v3",
                        label="Whisper Model",
                        info="Larger = more accurate but slower"
                    )
                    
                    translation_model = gr.Dropdown(
                        choices=["nllb", "nllb-large", "marian", "two_pass", "llm"],
                        value="two_pass",
                        label="Translation Model",
                        info="two_pass = best quality (uses AI refinement)"
                    )
                    
                    output_mode = gr.Radio(
                        choices=["Soft Subtitles", "Burn-in Subtitles", "SRT Only"],
                        value="Soft Subtitles",
                        label="Output Mode",
                        info="Soft = toggleable, Burn-in = permanent"
                    )
                    
                    font_size = gr.Slider(
                        minimum=12,
                        maximum=72,
                        value=24,
                        step=2,
                        label="Subtitle Font Size",
                        info="Smaller for vertical/mobile videos"
                    )
                    
                    keep_arabic = gr.Checkbox(
                        label="Also generate Arabic SRT",
                        value=False
                    )
                
                process_btn = gr.Button(
                    "🚀 Translate Video",
                    variant="primary",
                    size="lg"
                )
                
                with gr.Accordion("🔄 Re-apply Subtitles (skip translation)", open=False):
                    gr.Markdown("*Already have an SRT? Just embed it into the video.*")
                    srt_input = gr.File(
                        label="Upload SRT file",
                        file_types=[".srt"]
                    )
                    reapply_mode = gr.Radio(
                        choices=["Soft Subtitles", "Burn-in Subtitles"],
                        value="Burn-in Subtitles",
                        label="Output Mode"
                    )
                    reapply_font_size = gr.Slider(
                        minimum=12,
                        maximum=72,
                        value=24,
                        step=2,
                        label="Subtitle Font Size"
                    )
                    reapply_btn = gr.Button(
                        "🔄 Re-apply Subtitles",
                        variant="secondary"
                    )
            
            # Right column: Output
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Output")
                
                status_output = gr.Markdown(
                    value="*Upload a video and click 'Translate Video' to begin.*"
                )
                
                video_output = gr.Video(
                    label="Translated Video",
                    visible=True
                )
                
                with gr.Row(elem_classes=["output-files"]):
                    english_srt = gr.File(
                        label="English Subtitles (.srt)",
                        visible=True
                    )
                    arabic_srt = gr.File(
                        label="Arabic Subtitles (.srt)",
                        visible=True
                    )
        
        # Footer
        gr.Markdown(
            """
            ---
            <div style="text-align: center; color: #666; font-size: 0.9em;">
            
            **Tips:**
            • First run downloads AI models (~5GB) - subsequent runs are faster
            • GPU (CUDA/MPS) significantly speeds up processing
            • For long videos, try smaller models first (base/small) to test
            
            [GitHub](https://github.com/your-repo) • MIT License
            
            </div>
            """,
            sanitize_html=False
        )
        
        # Event handlers
        process_btn.click(
            fn=process_video_gradio,
            inputs=[
                video_input,
                whisper_model,
                translation_model,
                output_mode,
                font_size,
                keep_arabic
            ],
            outputs=[
                video_output,
                english_srt,
                arabic_srt,
                status_output
            ],
            show_progress="full"
        )
        
        reapply_btn.click(
            fn=reapply_subtitles_gradio,
            inputs=[
                video_input,
                srt_input,
                reapply_mode,
                reapply_font_size
            ],
            outputs=[
                video_output,
                status_output
            ],
            show_progress="full"
        )
    
    return interface


def main():
    """Launch the web interface."""
    parser = argparse.ArgumentParser(
        description="Launch Arabic Video Translator web interface"
    )
    
    parser.add_argument(
        "--host",
        default=os.environ.get("AVT_SERVER_HOST", "127.0.0.1"),
        help="Server host (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("AVT_SERVER_PORT", "7860")),
        help="Server port (default: 7860)"
    )
    
    parser.add_argument(
        "--share",
        action="store_true",
        default=os.environ.get("AVT_SHARE", "").lower() in ("1", "true", "yes"),
        help="Create a public shareable link"
    )
    
    args = parser.parse_args()
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║           🎬 Arabic Video Translator - Web UI             ║
╠═══════════════════════════════════════════════════════════╣
║  Starting server at: http://{args.host}:{args.port}             ║
║  Share link: {'Enabled' if args.share else 'Disabled (use --share to enable)'}             ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    interface = create_interface()
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
