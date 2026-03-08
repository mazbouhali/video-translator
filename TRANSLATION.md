# Arabic Translation System

## Overview

This document describes the translation strategies for Arabic→English conversion, optimized for religious/Islamic content like nasheeds, poetry, and devotional videos.

## The Problem with NLLB

NLLB-200 (No Language Left Behind) is a solid general-purpose multilingual model, but struggles with:

1. **Religious terminology** - عزف النصر مقاما becomes "Razaf al-Nusra in Makkah" (nonsense)
2. **Poetic Arabic** - Metaphors and religious allusions are translated literally/wrongly
3. **Cultural context** - No understanding that content is Shia Islamic devotional poetry
4. **Proper nouns** - Names like حسين, العباس get mangled or mistranslated

## Translation Strategies

### 1. Glossary-Based Post-Processing (Baseline)

**File:** `app/glossary.py`

Simple regex-based fixes for known mistranslations:
- "Razaf al-Nusra" → "played victory"
- "rear wheel" → "cast from the Champion"

**Pros:** Fast, no additional models needed
**Cons:** Doesn't scale, needs constant manual updates

### 2. Context-Aware NLLB (Improved)

Provide domain context before translation to help the model:

```python
translator.translate(text, context="religious")
```

This prepends context hints that guide translation choices.

### 3. LLM-Based Translation (Best Quality)

Use a large language model (local or API) for contextual translation:

```python
translator = create_translator(
    backend="llm",
    llm_provider="ollama",  # or "openai", "anthropic"
    llm_model="llama3.2"
)
```

**Supported LLM backends:**
- **Ollama** (local) - llama3.2, qwen2.5, mistral
- **OpenAI** - gpt-4o, gpt-4o-mini
- **Anthropic** - claude-3.5-sonnet

The LLM receives a system prompt explaining the domain:
> "You are translating Arabic religious poetry (nasheeds) about Karbala, Imam Hussein, and Shia Islamic themes. Preserve poetic structure and religious meaning."

### 4. Two-Pass Translation (Recommended)

Combines neural MT speed with LLM quality:

```
Arabic Text
    │
    ▼
┌─────────────────┐
│  NLLB (fast)    │  First pass: literal translation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM (refine)   │  Second pass: contextual refinement
└────────┬────────┘
         │
         ▼
  Final English
```

Usage:
```python
translator = create_translator(
    backend="two_pass",
    llm_provider="ollama",
    llm_model="llama3.2"
)
```

## Configuration

### Environment Variables

```bash
# LLM backend selection
AVT_TRANSLATION_BACKEND=two_pass  # nllb, llm, two_pass

# Ollama settings (local)
AVT_LLM_PROVIDER=ollama
AVT_OLLAMA_HOST=http://localhost:11434
AVT_OLLAMA_MODEL=llama3.2

# OpenAI settings
AVT_LLM_PROVIDER=openai
AVT_OPENAI_API_KEY=sk-...
AVT_OPENAI_MODEL=gpt-4o-mini

# Anthropic settings
AVT_LLM_PROVIDER=anthropic
AVT_ANTHROPIC_API_KEY=sk-ant-...
AVT_ANTHROPIC_MODEL=claude-3-5-sonnet-latest
```

### Content Domain

Set the content domain for better context:

```python
translator.translate(text, domain="religious")
```

Supported domains:
- `religious` - Islamic nasheeds, Karbala poetry, Shia devotional content
- `political` - Resistance speeches, political rhetoric, regional commentary
- `news` - Formal news/journalism
- `casual` - Everyday conversation
- `general` - Default, no specific context

## Example Translations

### Religious Content

**Arabic:** عزف النصر مقاما
- **NLLB alone:** "Razaf al-Nusra in Makkah" ❌
- **LLM (religious context):** "played victory as a melody" ✅

**Arabic:** صاح شبل الكرار في الميدان
- **NLLB alone:** "The girl of Eden shouted in the arena" ❌
- **LLM (religious context):** "The lion cub of al-Karrar roared on the battlefield" ✅

### Technical Notes

**NLLB Language Codes:**
- Source: `arb_Arab` (Modern Standard Arabic, Arabic script)
- Target: `eng_Latn` (English, Latin script)

**LLM Prompt Engineering:**
The system prompt includes:
1. Domain context (religious, news, etc.)
2. Key terminology (Karbala, Hussein, Abbas, etc.)
3. Instruction to preserve poetic structure
4. Examples of common terms and their correct translations

## Recommended Models

### For Local Deployment

| Model | Quality | Speed | VRAM |
|-------|---------|-------|------|
| NLLB-200-distilled-600M | Medium | Fast | 2 GB |
| NLLB-200-1.3B | Good | Medium | 4 GB |
| Ollama llama3.2:3b | Good | Medium | 4 GB |
| Ollama qwen2.5:7b | Excellent | Slower | 8 GB |

### For Best Quality (API)

- **GPT-4o-mini** - Fast, cheap, good Arabic
- **Claude 3.5 Sonnet** - Excellent contextual understanding
- **GPT-4o** - Best quality, slower

## Performance Considerations

| Strategy | Speed | Quality | Cost |
|----------|-------|---------|------|
| NLLB only | ⚡⚡⚡ | ⭐⭐ | Free |
| NLLB + glossary | ⚡⚡⚡ | ⭐⭐⭐ | Free |
| LLM (local) | ⚡⚡ | ⭐⭐⭐⭐ | Free |
| LLM (API) | ⚡⚡ | ⭐⭐⭐⭐⭐ | $ |
| Two-pass | ⚡ | ⭐⭐⭐⭐⭐ | $ |

For videos, the transcription (Whisper) is usually the bottleneck anyway, so LLM translation adds minimal overhead.

## Troubleshooting

### Ollama not responding
```bash
# Check if Ollama is running
ollama list

# Start Ollama
ollama serve

# Pull a model
ollama pull llama3.2
```

### API rate limits
The translator includes retry logic with exponential backoff. For high-volume processing, consider local Ollama or batch processing.

### Memory issues
Use NLLB-distilled (600M) instead of larger models, or offload to CPU:
```python
translator = create_translator(model="nllb", device="cpu")
```
