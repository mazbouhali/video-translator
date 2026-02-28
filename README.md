# 🎬 Arabic Video Translator

Translate Arabic videos to English automatically. No API keys, no cloud services — everything runs locally on your machine.

**What it does:** Arabic audio → English subtitles (burned into video + separate SRT files)

**Dialect support:** Works with Modern Standard Arabic (MSA) and regional dialects including **Lebanese**, Syrian, Egyptian, Gulf, and Maghrebi Arabic. Optimized for Levantine dialects.

## Quick Start

```bash
git clone https://github.com/youruser/arabic-video-translator
cd arabic-video-translator
docker compose up
```

That's it. Now:
1. Drop video files into `input/` folder
2. Wait for processing (watch the terminal)
3. Get results in `output/` folder

## Requirements

- **Docker** — [Install Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **~10GB disk space** — For ML models (downloaded automatically on first run)
- **Optional:** NVIDIA GPU for 5-10x faster processing

Works on **Mac**, **Windows**, and **Linux**.

## Output Files

For each input video, you get:

| File | Description |
|------|-------------|
| `video_translated.mp4` | Video with English subtitles burned in |
| `video_english.srt` | English subtitles (standalone) |
| `video_arabic.srt` | Original Arabic transcription |

## Commands

Using Make (recommended):
```bash
make build          # Build the Docker image
make run            # Start in watch mode (background)
make logs           # View processing logs
make stop           # Stop the translator
make translate VIDEO=path/to/video.mp4   # Translate single file
```

Using Docker Compose directly:
```bash
docker compose up -d          # Start (detached)
docker compose logs -f        # View logs
docker compose down           # Stop
docker compose up --build     # Rebuild and start
```

## Configuration

Edit these in `docker-compose.yml`:

| Setting | Default | Options |
|---------|---------|---------|
| `WHISPER_MODEL` | `medium` | `tiny`, `base`, `small`, `medium`, `large` |
| `SOURCE_LANG` | `ar` | Any [Whisper language code](https://github.com/openai/whisper#available-models-and-languages) |
| `TARGET_LANG` | `en` | `en`, `es`, `fr`, `de`, etc. |

### Model Size vs Speed

| Model | Size | Speed | Quality | GPU Memory |
|-------|------|-------|---------|------------|
| `tiny` | 75 MB | ⚡⚡⚡⚡⚡ | ★☆☆☆☆ | ~1 GB |
| `base` | 142 MB | ⚡⚡⚡⚡ | ★★☆☆☆ | ~1 GB |
| `small` | 466 MB | ⚡⚡⚡ | ★★★☆☆ | ~2 GB |
| `medium` | 1.5 GB | ⚡⚡ | ★★★★☆ | ~5 GB |
| `large` | 2.9 GB | ⚡ | ★★★★★ | ~10 GB |

**No GPU?** It still works, just slower. Use `small` or `base` model for reasonable speed.

## GPU Acceleration (Optional)

**NVIDIA GPU users:** Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), then it just works.

**Mac users:** GPU acceleration via MPS is automatic on Apple Silicon.

**CPU-only mode:** Use the override file:
```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up
```

## Troubleshooting

**"It's stuck downloading"** — First run downloads ~1.5GB of models. Be patient.

**"Out of memory"** — Use a smaller model: set `WHISPER_MODEL=small` in docker-compose.yml

**"Subtitles look wrong"** — Arabic text issues? The SRT files should be correct; video burning may have font issues.

**"No GPU detected"** — That's fine, it falls back to CPU. For GPU: install NVIDIA Container Toolkit.

## How It Works

1. **Extract audio** from video (FFmpeg)
2. **Transcribe** Arabic speech to text (OpenAI Whisper)
3. **Translate** Arabic text to English (Google Translate, offline-capable)
4. **Generate subtitles** in SRT format
5. **Burn subtitles** into video (FFmpeg)

All processing happens locally. Your videos never leave your machine.

## Project Structure

```
arabic-video-translator/
├── input/              # ⬇️ Drop videos here
├── output/             # ⬆️ Get results here
├── models/             # 🤖 Cached ML models
├── docker-compose.yml  # ⚙️ Main config
├── Dockerfile          # 🐳 Container definition
├── main.py             # 🐍 Translation script
└── Makefile            # 🛠️ Convenience commands
```

## License

MIT — Use it however you want.

---

**Questions?** Open an issue. **Like it?** Star the repo ⭐
