# 🎬 Video Translator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

Automatically translate foreign language speech in videos to English subtitles using state-of-the-art AI models.

## 🌍 Supported Languages

| Language | Dialects | Status |
|----------|----------|--------|
| **Arabic** | MSA, Lebanese, Syrian, Egyptian, Gulf, Maghrebi | ✅ Full support |
| **Farsi/Persian** | Iranian, Dari, Tajik | 🔜 Coming soon |
| **100+ others** | Via Whisper | ✅ Works out of the box |

*Optimized for Levantine Arabic (Lebanese, Syrian) dialects.*

## ✨ Features

- **🎙️ Accurate Transcription** - OpenAI Whisper (large-v3) for Arabic speech recognition
- **🌐 Quality Translation** - Meta's NLLB-200 for Arabic→English translation  
- **📝 Subtitle Generation** - Creates properly-timed SRT files
- **🎬 Video Output** - Embeds subtitles (soft or burned-in) via ffmpeg
- **💻 CLI & Web UI** - Use command line or drag-and-drop browser interface
- **⚡ GPU Accelerated** - CUDA, MPS (Apple Silicon), or CPU

## 🚀 Quick Start

### Option 1: One-Line Install (Recommended)

**Mac/Linux:**
```bash
curl -fsSL https://raw.githubusercontent.com/mazbouhali/video-translator/main/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/mazbouhali/video-translator/main/install.ps1 | iex
```

After install, double-click **start.command** (Mac) or **start.bat** (Windows) to launch.

---

### Option 2: Manual Docker Install

```bash
git clone https://github.com/mazbouhali/video-translator.git
cd video-translator
docker compose up
```

Open **http://localhost:7860** — drag and drop your video, get English subtitles.

---

### Option 3: Manual Python Install (No Docker)

```bash
# Clone the repository
git clone https://github.com/mazbouhali/video-translator.git
cd video-translator

# Install ffmpeg (required)
brew install ffmpeg          # macOS
sudo apt install ffmpeg       # Ubuntu/Debian
choco install ffmpeg          # Windows

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
# or: venv\Scripts\activate   # Windows

# Install Python dependencies
pip install -r requirements.txt
```

### Usage

#### Command Line

```bash
# Basic usage - creates video with toggleable subtitles
python -m app.main video.mp4

# Burn subtitles into video (permanent)
python -m app.main video.mp4 --burn-in

# Generate only SRT file (no video output)
python -m app.main video.mp4 --srt-only

# Keep Arabic transcript alongside English
python -m app.main video.mp4 --keep-arabic

# Specify output path
python -m app.main video.mp4 -o translated_video.mp4
```

#### Web Interface

```bash
# Launch web UI (opens browser)
python -m app.web

# Create shareable public link
python -m app.web --share

# Custom port
python -m app.web --port 8080
```

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AVT_WHISPER_MODEL` | Whisper model | `large-v3` |
| `AVT_TRANSLATION_MODEL` | Translation model | `nllb` |
| `AVT_DEVICE` | Compute device | auto-detect |
| `AVT_OUTPUT_DIR` | Output directory | same as input |
| `AVT_SERVER_PORT` | Web UI port | `7860` |
| `AVT_SHARE` | Enable public link | `false` |

### Config File

Create `~/.config/arabic-video-translator/config.json`:

```json
{
  "transcription": {
    "model": "large-v3",
    "language": "ar"
  },
  "translation": {
    "model": "nllb",
    "batch_size": 8
  },
  "subtitle": {
    "font_size": 24,
    "position": "bottom"
  }
}
```

## 🔧 Model Options

### Transcription (Whisper)

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `tiny` | 75MB | ⚡⚡⚡⚡ | ⭐ |
| `base` | 145MB | ⚡⚡⚡ | ⭐⭐ |
| `small` | 488MB | ⚡⚡ | ⭐⭐⭐ |
| `medium` | 1.5GB | ⚡ | ⭐⭐⭐⭐ |
| `large-v3` | 3GB | 🐢 | ⭐⭐⭐⭐⭐ |

### Translation

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `marian` | 300MB | ⚡⚡⚡ | ⭐⭐⭐ |
| `nllb` | 1.2GB | ⚡⚡ | ⭐⭐⭐⭐ |
| `nllb-large` | 2.5GB | ⚡ | ⭐⭐⭐⭐⭐ |

## 📁 Output Files

| File | Description |
|------|-------------|
| `video_translated.mp4` | Video with English subtitles |
| `video_translated.srt` | English subtitle file |
| `video_translated_arabic.srt` | Arabic transcript (with `--keep-arabic`) |

## 💡 Tips

- **First run** downloads models (~5GB total) - subsequent runs are faster
- **GPU** significantly speeds up processing (10x or more)
- For **long videos**, test with smaller models first (`base`, `small`)
- **Burned-in** subtitles work everywhere; soft subs need player support
- Use **SRT-only** mode for quick testing without video re-encoding

## 🛠️ Development

```bash
# Run tests
python -m pytest tests/

# Type checking
python -m mypy app/

# Format code
python -m black app/
```

## 📦 Project Structure

```
arabic-video-translator/
├── app/
│   ├── __init__.py      # Package exports
│   ├── config.py        # Configuration management
│   ├── transcribe.py    # Whisper transcription
│   ├── translate.py     # NLLB/MarianMT translation
│   ├── subtitles.py     # SRT generation & embedding
│   ├── main.py          # CLI interface
│   └── web.py           # Gradio web interface
├── requirements.txt
└── README.md
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Meta NLLB](https://github.com/facebookresearch/fairseq/tree/nllb) - Translation
- [Gradio](https://gradio.app/) - Web interface
- [FFmpeg](https://ffmpeg.org/) - Video processing
