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

- **🎙️ Accurate Transcription** - faster-whisper (4x faster) with fine-tuned Arabic models
- **🌐 Quality Translation** - Meta's NLLB-200 for Arabic→English translation  
- **📝 Subtitle Generation** - Creates properly-timed SRT files
- **🎬 Video Output** - Embeds subtitles (soft or burned-in) via ffmpeg
- **💻 CLI & Web UI** - Use command line or drag-and-drop browser interface
- **⚡ GPU Accelerated** - CUDA, MPS (Apple Silicon), or CPU
- **🕌 Religious Content** - Specialized models for Quranic Arabic (5.8% WER!)
- **🗣️ Dialect Support** - Hints for Egyptian, Levantine, Gulf, Maghrebi dialects

> 📖 **For Arabic ASR details**, see [ARABIC_ASR.md](ARABIC_ASR.md) — includes model benchmarks, dialect detection, and optimization tips.

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

After install, launch with:
- **Linux:** `./start.sh` (or `./start-amd.sh` for AMD GPUs)
- **Mac:** Double-click `start.command`
- **Windows:** Double-click `start.bat`

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

**No configuration needed** — defaults work out of the box.

To change model size, edit `docker-compose.yml`:
```yaml
environment:
  - WHISPER_MODEL=large    # Options: tiny, base, small, medium, large
```

## 🎮 GPU Support

GPU is **auto-detected** — no config needed for NVIDIA. AMD requires one extra step.

### Which launcher to use?

| Your GPU | Launcher | Notes |
|----------|----------|-------|
| **NVIDIA** (RTX 3060, 4070, etc.) | `start.sh` / `start.command` | Works automatically via CUDA |
| **AMD** (RX 7900, 9070 XT, etc.) | `start-amd.sh` | Requires ROCm drivers on Linux |
| **Intel Arc** | `start.sh` (CPU mode) | GPU not supported yet, uses CPU |
| **Apple Silicon** (M1/M2/M3) | `start.command` | Uses MPS acceleration |
| **No GPU / Integrated** | `start.sh` | Falls back to CPU (slower but works) |

### GPU Memory Requirements

| Model | Min VRAM | Recommended |
|-------|----------|-------------|
| `tiny` | 1 GB | Any GPU |
| `base` | 1 GB | Any GPU |
| `small` | 2 GB | GTX 1650+ / RX 580+ |
| `medium` | 5 GB | RTX 3060 / RX 6700 XT |
| `large-v3` | 10 GB | RTX 3080 / RX 6800 XT / **RX 9070 XT ✓** |

### AMD GPU Setup (Linux only)

1. Install [ROCm drivers](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/)
2. Use `./start-amd.sh` instead of `./start.sh`

> ⚠️ AMD GPUs only work on Linux with ROCm. On Windows, it falls back to CPU.

### Speed Comparison (10-minute video)

| Device | `medium` model | `large-v3` model |
|--------|---------------|------------------|
| RTX 4090 | ~2 min | ~4 min |
| RTX 3080 / RX 6800 XT | ~5 min | ~10 min |
| **RX 9070 XT** | ~4 min | ~8 min |
| RTX 3060 / RX 6700 XT | ~8 min | ~15 min |
| Apple M2 Pro | ~10 min | ~20 min |
| CPU only (8-core) | ~30 min | ~60 min |

---

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
