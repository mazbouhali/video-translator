# Arabic Dialect Speech-to-Text Research

## Executive Summary

For **Lebanese Arabic (Levantine dialect)** video translation running locally in Docker, the recommended approach is:

1. **STT**: `whisper-large-v3` via **faster-whisper** (best balance of accuracy and speed)
2. **Translation**: **Whisper's built-in translation** (Arabic→English) or **SeamlessM4T v2** for higher quality
3. **Alternative for pure STT**: Fine-tuned Levantine models available on HuggingFace

---

## 1. Whisper Models for Arabic Dialects

### Model Size Recommendations

| Model | Parameters | VRAM | Arabic Performance | Recommendation |
|-------|------------|------|-------------------|----------------|
| `tiny` | 39M | ~1 GB | Poor for dialects | ❌ Not recommended |
| `base` | 74M | ~1 GB | Mediocre | ❌ Not recommended |
| `small` | 244M | ~2 GB | Acceptable | ⚠️ Only if resource-constrained |
| `medium` | 769M | ~5 GB | Good | ✅ Good balance |
| `large-v3` | 1550M | ~10 GB | Best | ✅ **Recommended** |
| `turbo` | 809M | ~6 GB | Good (no translation!) | ⚠️ Transcription only |

### Key Findings

- **`large-v3`** shows 10-20% error reduction over `large-v2` for Arabic
- **Turbo model does NOT support translation** - only transcription. Use `medium` or `large` for Arabic→English translation
- Arabic dialects (including Levantine/Lebanese) are handled best by larger models
- Whisper treats all Arabic variants as "Arabic" (`ar`) - no specific dialect codes

### Whisper Translation Capability

Whisper can directly translate Arabic speech to English text:
```bash
whisper arabic_audio.wav --model large-v3 --language Arabic --task translate
```

This provides a **single-step solution** for Arabic→English without needing a separate translation model.

---

## 2. Fine-Tuned Arabic/Lebanese Models on HuggingFace

### Levantine Arabic Specific Models

| Model | Type | Notes |
|-------|------|-------|
| `elgeish/wav2vec2-large-xlsr-53-levantine-arabic` | Wav2Vec2 | Oldest, well-tested Levantine model |
| `EmreAkgul/whisper-large-v3-lora-levantine-arabic` | Whisper LoRA | Recent (Oct 2025), Whisper-based |
| `kareemali1/whisper-arabic-levantine-peft-*` | Whisper PEFT | Multiple variants (42/84/168) |
| `kareemali1/whisper-arabic-msa-levantine-peft-*` | Whisper PEFT | MSA + Levantine combined |

### General Arabic Models

| Model | Notes |
|-------|-------|
| `clu-ling/whisper-large-v2-arabic-5k-steps` | Fine-tuned large-v2, popular |
| `Salama1429/KalemaTech-Arabic-STT-ASR-based-on-Whisper-Small` | Egyptian dialect focus |
| `IbrahimAmin/code-switched-egyptian-arabic-whisper-small` | Egyptian + code-switching |

### Recommendation

For **Lebanese Arabic specifically**:
1. Start with base `whisper-large-v3` - it handles dialects reasonably well
2. If quality is insufficient, try `elgeish/wav2vec2-large-xlsr-53-levantine-arabic` for STT only
3. The LoRA/PEFT models are newer but less documented - worth testing

---

## 3. Alternative STT Options

### A. faster-whisper (Recommended for Docker)

**GitHub**: [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper)

- **4x faster** than openai/whisper with same accuracy
- Uses CTranslate2 backend
- Supports int8 quantization (lower VRAM)
- **Docker-ready**: `nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04`

```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
# Or for lower memory:
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

segments, info = model.transcribe("arabic_audio.mp3", language="ar", task="translate")
```

### B. whisper.cpp

**GitHub**: [ggml-org/whisper.cpp](https://github.com/ggml-org/whisper.cpp)

- Pure C/C++ implementation
- Excellent for CPU-only deployments
- Supports Metal (Mac), CUDA, Vulkan
- Quantization support (smaller models)

Memory usage:
| Model | Disk | Memory |
|-------|------|--------|
| medium | 1.5 GiB | ~2.1 GB |
| large | 2.9 GiB | ~3.9 GB |

### C. WhisperX (For Advanced Features)

**GitHub**: [m-bain/whisperX](https://github.com/m-bain/whisperX)

- Word-level timestamps
- Speaker diarization
- 70x realtime with batched inference
- Built on faster-whisper

Good for: Multi-speaker Arabic videos, subtitle generation

### D. Facebook MMS (Massively Multilingual Speech)

**Model**: `facebook/mms-1b-all`

- Supports 1000+ languages including Arabic (`ara`)
- Wav2Vec2-based architecture
- **No translation capability** - STT only
- Good for pure transcription, not translation pipelines

### E. SeamlessM4T v2

**Model**: `facebook/seamless-m4t-v2-large`

- **All-in-one** model: STT + Translation + TTS
- Supports Arabic variants:
  - `arb` - Modern Standard Arabic
  - `ary` - Moroccan Arabic
  - `arz` - Egyptian Arabic
- **Levantine Arabic not explicitly listed** but MSA should work
- Excellent for direct speech-to-speech translation

---

## 4. Translation Options

### A. Whisper Built-in Translation (Recommended for Simplicity)

**Pros:**
- Single model, single pass
- No additional dependencies
- Good quality for general content

**Cons:**
- Only translates TO English (not from English)
- Quality depends on dialect recognition

```python
# Using faster-whisper
segments, info = model.transcribe(audio, language="ar", task="translate")
# Output is in English
```

### B. Local Translation Models

#### NLLB-200 (No Language Left Behind)

**Model**: `facebook/nllb-200-distilled-600M`

- 200 languages including Arabic
- ~600M parameters (distilled version)
- Runs locally, no API needed
- CC-BY-NC license

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")

# Arabic (arb_Arab) to English (eng_Latn)
inputs = tokenizer(arabic_text, return_tensors="pt", src_lang="arb_Arab")
translated = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"))
```

#### Argos Translate

**GitHub**: [argosopentech/argos-translate](https://github.com/argosopentech/argos-translate)

- Fully offline
- Uses OpenNMT
- Arabic→English package available
- Lightweight
- MIT licensed

```python
import argostranslate.translate
text = argostranslate.translate.translate(arabic_text, "ar", "en")
```

#### SeamlessM4T v2 (Best Quality)

- Direct speech-to-text translation
- Handles Arabic→English natively
- 2.3B parameters
- Requires ~10GB VRAM

```python
from transformers import AutoProcessor, SeamlessM4Tv2Model

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

# Direct speech to English text
audio_inputs = processor(audios=audio_array, return_tensors="pt")
output = model.generate(**audio_inputs, tgt_lang="eng", generate_speech=False)
```

### C. Cloud APIs (Not Local - For Reference)

| Service | Arabic Dialect Support | Notes |
|---------|----------------------|-------|
| Google Cloud Speech | Excellent | Multiple dialects |
| Azure Speech | Good | MSA + Gulf |
| AWS Transcribe | Limited | MSA only |
| OpenAI Whisper API | Good | Same as local Whisper |

---

## 5. Recommended Architecture for Docker

### Option A: Simple Pipeline (Whisper-Only)

```
┌─────────────────────────────────────────────────────┐
│                   Docker Container                   │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ Arabic Audio │───▶│ faster-whisper (large-v3) │  │
│  └──────────────┘    │   task="translate"        │  │
│                      └───────────┬──────────────┘  │
│                                  │                  │
│                                  ▼                  │
│                        ┌─────────────────┐         │
│                        │  English Text   │         │
│                        └─────────────────┘         │
└─────────────────────────────────────────────────────┘
```

**Dockerfile sketch:**
```dockerfile
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04
RUN pip install faster-whisper
# Download model on build
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"
```

### Option B: Two-Stage Pipeline (Better Translation)

```
┌─────────────────────────────────────────────────────┐
│                   Docker Container                   │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ Arabic Audio │───▶│ faster-whisper (large-v3) │  │
│  └──────────────┘    │   task="transcribe"       │  │
│                      └───────────┬──────────────┘  │
│                                  │                  │
│                                  ▼                  │
│                        ┌─────────────────┐         │
│                        │  Arabic Text    │         │
│                        └────────┬────────┘         │
│                                 │                  │
│                                 ▼                  │
│                      ┌───────────────────┐         │
│                      │ NLLB-200 / Argos  │         │
│                      │   ar → en         │         │
│                      └─────────┬─────────┘         │
│                                │                   │
│                                ▼                   │
│                      ┌─────────────────┐           │
│                      │  English Text   │           │
│                      └─────────────────┘           │
└─────────────────────────────────────────────────────┘
```

### Option C: Premium Pipeline (SeamlessM4T)

```
┌─────────────────────────────────────────────────────┐
│                   Docker Container                   │
│                                                      │
│  ┌──────────────┐    ┌──────────────────────────┐  │
│  │ Arabic Audio │───▶│    SeamlessM4T v2        │  │
│  └──────────────┘    │   (S2TT: ar → en)        │  │
│                      └───────────┬──────────────┘  │
│                                  │                  │
│                                  ▼                  │
│                        ┌─────────────────┐         │
│                        │  English Text   │         │
│                        │ (+ timestamps)  │         │
│                        └─────────────────┘         │
└─────────────────────────────────────────────────────┘
```

---

## 6. Hardware Requirements

### Minimum (CPU-only)
- 8GB RAM
- 4+ cores
- ~5 GB disk (model storage)
- **Speed**: ~0.5x realtime (slower than audio length)

### Recommended (GPU)
- NVIDIA GPU with 8GB+ VRAM
- CUDA 12.x
- cuDNN 9.x
- **Speed**: 10-70x realtime

### Model VRAM Requirements

| Model | FP16 | INT8 |
|-------|------|------|
| whisper-large-v3 | ~10 GB | ~5 GB |
| whisper-medium | ~5 GB | ~3 GB |
| SeamlessM4T-large | ~10 GB | ~6 GB |
| NLLB-200-600M | ~2 GB | ~1.5 GB |

---

## 7. Quick Start Recommendation

For a **Lebanese Arabic video translator** running in Docker:

### Phase 1: MVP (Simple)
```python
from faster_whisper import WhisperModel

model = WhisperModel("large-v3", device="cuda", compute_type="float16")
segments, _ = model.transcribe("video.mp4", language="ar", task="translate")

for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
```

### Phase 2: Enhanced (If Needed)
- Add fine-tuned Levantine model if accuracy is insufficient
- Use NLLB-200 for translation if Whisper's translation quality is poor
- Add WhisperX for speaker diarization in multi-speaker content

---

## 8. Testing Checklist

Before deployment, test with:
- [ ] Clear Lebanese Arabic speech (news, formal)
- [ ] Colloquial Lebanese dialogue (casual, fast)
- [ ] Mixed Arabic/French code-switching (common in Lebanese)
- [ ] Background noise/music
- [ ] Multiple speakers

---

## References

- [OpenAI Whisper](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
- [WhisperX](https://github.com/m-bain/whisperX)
- [SeamlessM4T](https://huggingface.co/facebook/seamless-m4t-v2-large)
- [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [Argos Translate](https://github.com/argosopentech/argos-translate)
- [elgeish Levantine Arabic model](https://huggingface.co/elgeish/wav2vec2-large-xlsr-53-levantine-arabic)
