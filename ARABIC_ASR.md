# Arabic ASR Research & Implementation

> Research findings and implementation details for Arabic speech recognition in the video translator.

## Executive Summary

After extensive research, the key findings are:

1. **Best general Arabic model**: `Byne/whisper-large-v3-arabic` achieves **9.4% WER** (vs ~15% for base Whisper large-v3)
2. **Fastest backend**: `faster-whisper` is **4x faster** than OpenAI's implementation with identical accuracy
3. **Quran/religious content**: `tarteel-ai/whisper-base-ar-quran` achieves remarkable **5.8% WER** on Quranic recitation
4. **Dialect detection**: The `voxlect-arabic-dialect-whisper-large-v3` model can classify dialects (Egyptian, Levantine, Maghrebi, MSA, Peninsular)

---

## 🎵 Priority Use Case: Nasheeds & Political Speeches

The primary use case involves:
- **Religious nasheeds** (أناشيد) - sung/chanted Arabic with background music
- **Political speeches** - formal MSA mixed with dialect
- **Background music/drums** (دف) that interfere with speech
- **Emotional/chanting delivery** different from normal speech
- **Classical Arabic mixed with Iraqi/Lebanese dialect**

### Recommended Pipeline for Nasheeds

```
Audio Input → Demucs (vocal separation) → Preprocessed Audio → Whisper Transcription
                   ↓
           Remove drums/music
```

### Key Optimizations

| Challenge | Solution |
|-----------|----------|
| Background drums/duff | **Demucs** vocal separation (removes ~90% of music) |
| Chanting style | Use **Quran model** (trained on melodic recitation) |
| Iraqi dialect | Use `ayousry42/whisper-arabic-dialect-iraqi` |
| Lebanese/Syrian | Use `EmreAkgul/whisper-large-v3-lora-levantine-arabic` |
| Religious terminology | Add religious prompt: "بسم الله الرحمن الرحيم" |

### Quick Start for Nasheeds

```python
from app.transcribe import transcribe_video

# Automatic: separates vocals, uses Quran model, adds religious prompts
result = transcribe_video(
    "nasheed.mp4",
    content_type="nasheed",      # Triggers vocal separation
    dialect_hint="levantine"     # For Lebanese nasheeds
)

# Iraqi nasheeds
result = transcribe_video(
    "iraqi_nasheed.mp4",
    content_type="nasheed",
    dialect_hint="iraqi"
)
```

### Demucs Vocal Separation

**Critical for nasheeds with drums/music.**

```bash
# Install
pip install demucs

# Standalone usage
python -m demucs --two-stems=vocals input.mp3
```

The app automatically uses Demucs when `content_type="nasheed"`.

---

## Backend Comparison

### faster-whisper vs OpenAI Whisper

| Aspect | faster-whisper | openai-whisper |
|--------|---------------|----------------|
| Speed | **4x faster** | Baseline |
| Memory | **~40% less** | Higher VRAM |
| Accuracy | Identical | Identical |
| GPU Support | CUDA, int8 quantization | CUDA only |
| Installation | `pip install faster-whisper` | `pip install openai-whisper` |

**Recommendation**: Use faster-whisper as primary backend. Falls back to openai-whisper if unavailable.

### Benchmark (13 min audio, large-v2, RTX 3070 Ti)

| Implementation | Time | VRAM |
|---------------|------|------|
| openai/whisper | 2m23s | 4708MB |
| faster-whisper (fp16) | 1m03s | 4525MB |
| faster-whisper (int8) | 59s | 2926MB |
| faster-whisper (batch=8) | **17s** | 6090MB |

---

## Arabic Fine-tuned Models

### Tier 1: Production Ready

#### 1. Byne/whisper-large-v3-arabic
- **WER**: 9.4% (excellent)
- **Downloads**: 18,400+
- **Size**: 2B parameters
- **Dialects**: MSA, Egyptian, Levantine, Gulf
- **Best for**: General Arabic content, lectures, interviews
- **URL**: https://huggingface.co/Byne/whisper-large-v3-arabic

#### 2. tarteel-ai/whisper-base-ar-quran
- **WER**: 5.8% (exceptional)
- **Downloads**: 18,700+
- **Size**: 72.6M parameters (small!)
- **Best for**: Quranic recitation, religious content
- **Note**: Extremely accurate but specialized
- **URL**: https://huggingface.co/tarteel-ai/whisper-base-ar-quran

#### 3. MohamedRashad/Arabic-Whisper-CodeSwitching-Edition
- **Downloads**: 915
- **Size**: 2B parameters (large-v2 based)
- **Best for**: Arabic with embedded English words
- **Dialects**: Egyptian, MSA with English
- **URL**: https://huggingface.co/MohamedRashad/Arabic-Whisper-CodeSwitching-Edition

### Tier 2: Dialect-Specific (Important for Nasheeds)

#### Levantine Arabic (Lebanese/Syrian) ⭐
- **Model**: `EmreAkgul/whisper-large-v3-lora-levantine-arabic`
- **Type**: LoRA fine-tuned on large-v3
- **Dialects**: Lebanese, Syrian, Palestinian, Jordanian
- **Best for**: Lebanese nasheeds, Hezbollah speeches
- **Alternative**: `elgeish/wav2vec2-large-xlsr-53-levantine-arabic` (wav2vec2)

#### Iraqi Arabic ⭐
- **Model**: `ayousry42/whisper-arabic-dialect-iraqi`
- **Size**: 0.2B parameters
- **Best for**: Iraqi nasheeds, PMU content
- **Note**: Only Iraqi-specific Whisper model available

#### Egyptian Arabic
- **Model**: `AbdelrahmanHassan/whisper-large-v3-egyptian-arabic`
- **Downloads**: 510
- **Note**: Fine-tuned specifically for Egyptian dialect

#### Multi-dialect (Small)
- **Model**: `MadLook/whisper-small-arabic-multidialect`
- **WER**: 48.9% (trade-off for dialect coverage)
- **Dialects**: MSA, Egyptian, Levantine, Gulf, Maghrebi
- **Note**: Smaller model, handles all dialects

### Tier 3: Specialized

#### Quran (Turbo) - Good for Nasheeds!
- **Model**: `MaddoggProduction/whisper-l-v3-turbo-quran-lora-dataset-mix`
- **Downloads**: 575
- **Note**: Trained on melodic/chanted Arabic - transfers well to nasheeds

---

## Dialect Detection

### voxlect Arabic Dialect Classifier

The `tiantiaf/voxlect-arabic-dialect-whisper-large-v3` model can detect Arabic dialects:

**Supported dialects**:
- Egyptian (مصري)
- Levantine (شامي) - Lebanese, Syrian, Palestinian, Jordanian
- Maghrebi (مغربي) - Moroccan, Algerian, Tunisian
- MSA (فصحى)
- Peninsular (خليجي) - Gulf Arabic

**Usage**:
```python
import torch
import torch.nn.functional as F
from src.model.dialect.whisper_dialect import WhisperWrapper

model = WhisperWrapper.from_pretrained("tiantiaf/voxlect-arabic-dialect-whisper-large-v3")
model.eval()

# Audio: 3-15 seconds, 16kHz, mono
logits, embeddings = model(audio_tensor, return_feature=True)
dialect_prob = F.softmax(logits, dim=1)
```

**Pipeline Integration**: Could run dialect detection first, then select optimal model.

---

## Model Selection Logic

```
┌─────────────────────────────────────────────────────────────────┐
│                    Content Type?                                │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   Religious?            Code-switching?        General?
        │                     │                     │
        ▼                     ▼                     ▼
  tarteel-ai/           MohamedRashad/         Check dialect
  whisper-base-ar-quran Arabic-Whisper-CS           │
        │                     │         ┌───────────┼───────────┐
        │                     │         ▼           ▼           ▼
        │                     │    Egyptian?    Levantine?   General?
        │                     │         │           │           │
        │                     │         ▼           ▼           ▼
        │                     │  Abdelrahman/  Byne/whisper  Byne/whisper
        │                     │  egyptian      large-v3-ar  large-v3-ar
        │                     │                     
        └─────────────────────┴───────────────────────────────────┘
```

---

## Implementation Details

### Updated transcribe.py Features

1. **Backend auto-selection**: Prefers faster-whisper, falls back gracefully
2. **Model registry**: All Arabic models with metadata (WER, dialects, descriptions)
3. **Dialect hints**: Pass `dialect_hint="levantine"` to guide transcription
4. **Content type hints**: `content_type="religious"` auto-selects Quran model
5. **VAD filtering**: Removes silence for faster processing
6. **Word timestamps**: Optional word-level timing

### Usage Examples

```python
from app.transcribe import transcribe_video, list_models, list_dialects

# Basic usage (auto-selects best model)
result = transcribe_video("lecture.mp4")

# With dialect hint
result = transcribe_video("egyptian_movie.mp4", dialect_hint="egyptian")

# Religious content
result = transcribe_video("quran_recitation.mp4", content_type="religious")

# Code-switching content
result = transcribe_video("tech_talk.mp4", content_type="code-switching")

# List available options
print(list_models())
print(list_dialects())
```

### Configuration

Via environment variables:
```bash
export AVT_WHISPER_MODEL=byne-arabic
export AVT_DEVICE=cuda
```

Or config file (`~/.config/arabic-video-translator/config.json`):
```json
{
  "transcription": {
    "model": "byne-arabic",
    "device": "cuda"
  }
}
```

---

---

## Audio Preprocessing (Demucs)

### Why Vocal Separation?

Nasheeds typically have:
- **Duff drums** (دف) - traditional frame drum
- **Background vocals** (chorus/harmony)
- **Reverb/echo** effects
- **Music tracks** in some modern nasheeds

These interfere with speech recognition. Demucs removes them.

### Demucs Models

| Model | Quality | Speed | Best For |
|-------|---------|-------|----------|
| `htdemucs_ft` | ⭐⭐⭐⭐⭐ | 🐢 | Best quality, final output |
| `htdemucs` | ⭐⭐⭐⭐ | 🐢 | Good quality |
| `hdemucs_mmi` | ⭐⭐⭐ | ⚡ | Faster, acceptable quality |

### Installation

```bash
pip install demucs

# GPU recommended (much faster)
# Requires PyTorch with CUDA
```

### Standalone Usage

```bash
# Extract vocals only (fastest)
python -m demucs --two-stems=vocals nasheed.mp3

# Output: separated/htdemucs/nasheed/vocals.wav
```

### Integrated Usage

```python
from app.audio_preprocess import preprocess_for_transcription

# Full pipeline: separate vocals + normalize
processed = preprocess_for_transcription(
    "nasheed.mp3",
    separate_vocals=True,
    normalize=True
)
```

---

## Performance Optimization

### For Speed
1. Use `faster-whisper` backend
2. Use `large-v3-turbo` model (~4x faster, minor accuracy loss)
3. Enable VAD filtering (`vad_filter=True`)
4. Use `int8` quantization on GPU

### For Accuracy
1. Use `Byne/whisper-large-v3-arabic` for general content
2. Use `tarteel-ai/whisper-base-ar-quran` for religious content
3. Provide dialect hints when known
4. Increase beam size (default 5 is good)

### For Memory Constrained Systems
1. Use smaller models: `whisper-small-arabic-multidialect`
2. Use CPU with `int8` quantization
3. Process in smaller chunks

---

## Future Improvements

### Planned
1. [ ] Add dialect auto-detection pipeline (run classifier → select model)
2. [ ] Convert HuggingFace models to CTranslate2 format for faster-whisper
3. [ ] Add diacritization (تشكيل) support for religious content
4. [ ] Benchmark all models on common test set

### Research Areas
- **WhisperX**: Adds speaker diarization and word-level timestamps via wav2vec2
- **MMS (Meta)**: Multi-lingual speech model with Arabic support
- **Conformer**: Google's ASR architecture, potentially better for Arabic

---

## References

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) - CTranslate2 implementation
- [OpenAI Whisper Paper](https://arxiv.org/abs/2212.04356) - Original Whisper paper
- [ARBML](https://huggingface.co/arbml) - Arabic ML research group
- [Tarteel AI](https://huggingface.co/tarteel-ai) - Quran transcription
- [Voxlect](https://arxiv.org/abs/2508.01691) - Dialect classification benchmark

---

---

## Prompting Strategies

### For Nasheeds

The initial prompt conditions Whisper on expected content. For nasheeds:

```python
# Auto-generated when content_type="nasheed"
prompt = "نشيد ديني إسلامي لا إله إلا الله محمد رسول الله [شامي]"
```

### For Political Speeches

```python
prompt = "خطاب سياسي المقاومة والتحرير [عراقي]"
```

### Custom Prompts

For better accuracy on specific content, provide vocabulary hints:

```python
result = transcribe_video(
    "hezbollah_nasheed.mp4",
    content_type="nasheed",
    dialect_hint="lebanese",
    initial_prompt="حزب الله نصر الله المقاومة لبنان"
)
```

---

## Troubleshooting

### Low accuracy on nasheeds

1. **Enable vocal separation**: Ensure `demucs` is installed
2. **Check dialect**: Use `levantine` for Lebanese, `iraqi` for Iraqi
3. **Try Quran model**: Works well on classical/sung Arabic

### Model not loading

1. First run downloads models (~3GB for large-v3)
2. Check disk space and internet connection
3. HuggingFace models require `transformers` installed

### Demucs slow/crashes

1. GPU strongly recommended (10x faster)
2. Try `hdemucs_mmi` for speed over quality
3. Process audio in chunks for long files

---

## Changelog

### 2026-03-08 (Update 2)
- **PRIORITY**: Added nasheed and political speech support
- Added Demucs vocal separation (`audio_preprocess.py`)
- Added Levantine Arabic model (Lebanese/Syrian)
- Added Iraqi Arabic model
- Added content presets with optimized prompts
- Auto-detects when to separate vocals
- Added `librosa` for audio analysis

### 2026-03-08 (Initial)
- Initial research completed
- Implemented faster-whisper backend
- Added model registry with 9 Arabic models
- Added dialect hint support
- Created ARABIC_ASR.md documentation
