"""
Arabic to English translation with multiple backends.

Supports:
- NLLB-200: Neural MT (fast, basic quality)
- LLM: Large language model translation (Ollama, OpenAI, Anthropic)
- Two-pass: NLLB first, then LLM refinement (best quality)

Example:
    >>> from app.translate import create_translator
    >>> translator = create_translator(backend="two_pass", llm_provider="ollama")
    >>> result = translator.translate("عزف النصر مقاما", domain="religious")
    >>> print(result.translated)  # "played victory as a melody"
"""

import os
import json
import time
import logging
from typing import List, Dict, Any, Optional, Callable, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianMTModel,
    MarianTokenizer,
)
from tqdm import tqdm

from .config import get_config
from .glossary import fix_translation, apply_glossary_to_arabic, get_context_prompt

logger = logging.getLogger(__name__)

# Domain-specific system prompts for LLM translation
DOMAIN_PROMPTS = {
    "religious": """You are an expert translator specializing in Arabic Islamic religious content.
You are translating nasheeds (religious songs), poetry, and devotional content about:
- Imam Hussein and the tragedy of Karbala
- Shia Islamic themes and commemorations  
- Religious figures: the Prophet Muhammad, Imam Ali (al-Karrar), Abbas, Zainab
- Concepts: martyrdom (shahada), sacrifice, devotion, lamentation

Key terminology to use correctly:
- الكرار = al-Karrar (the Champion), refers to Imam Ali
- شبل = lion cub (term for young warriors)
- الميدان = battlefield
- عزف = played (music), as in "played a melody"
- مقام = maqam (musical mode/melody), or spiritual station
- كربلاء = Karbala
- الطف = Taff (another name for Karbala)

Translate to preserve:
1. Religious and poetic meaning
2. Emotional resonance
3. Proper Islamic terminology in English
4. Natural English flow while respecting the original's poetry""",

    "political": """You are an expert translator specializing in Arabic political speeches and rhetoric.
You are translating political content from the Middle East, including:
- Resistance movement speeches and statements
- Political leaders' addresses
- Commentary on regional conflicts
- Hezbollah, Hamas, and allied movement communications

Key terminology:
- المقاومة = the Resistance
- الممانعة = the Resistance Axis / steadfastness
- الصمود = steadfastness, resilience
- التحرير = liberation
- الاحتلال = occupation
- الكيان = "the entity" (referring to Israel)
- المجاهدين = the fighters/mujahideen
- الشهداء = martyrs
- النصر = victory
- العدو = the enemy

Translate to preserve:
1. Political rhetoric and persuasive tone
2. Proper movement/organization names
3. Regional political terminology
4. The speaker's intended emotional impact""",

    "news": """You are translating Arabic news content to English.
Use formal, journalistic language. Be accurate and neutral.
Preserve proper nouns and place names correctly.""",

    "casual": """You are translating casual Arabic conversation to natural English.
Use conversational, idiomatic English. Capture the tone and informality.""",

    "general": """You are translating Arabic text to English.
Provide an accurate, natural-sounding translation.""",
}


@dataclass
class TranslationResult:
    """Result of a single translation operation."""
    original: str
    translated: str
    model_used: str
    backend: str = "nllb"
    refined: bool = False


class TranslationBackend(ABC):
    """Abstract base class for translation backends."""
    
    @abstractmethod
    def translate(self, text: str, domain: str = "general") -> str:
        """Translate Arabic text to English."""
        pass
    
    @abstractmethod
    def translate_batch(self, texts: List[str], domain: str = "general") -> List[str]:
        """Translate multiple texts."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier."""
        pass


class NLLBBackend(TranslationBackend):
    """NLLB-200 neural machine translation backend."""
    
    MODELS = {
        "nllb": "facebook/nllb-200-distilled-600M",
        "nllb-large": "facebook/nllb-200-1.3B",
        "nllb-3b": "facebook/nllb-200-3.3B",
        "marian": "Helsinki-NLP/opus-mt-ar-en",
    }
    
    NLLB_ARABIC = "arb_Arab"
    NLLB_ENGLISH = "eng_Latn"
    
    def __init__(
        self,
        model_name: str = "nllb",
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        self.progress_callback = progress_callback
        config = get_config()
        
        if model_name in self.MODELS:
            self.model_path = self.MODELS[model_name]
            self.model_type = "nllb" if "nllb" in model_name else "marian"
        else:
            self.model_path = model_name
            self.model_type = "nllb" if "nllb" in model_name.lower() else "marian"
        
        if device is None:
            device = config.translation.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.batch_size = config.translation.batch_size
        self.max_length = config.translation.max_length
        
        self._log(f"Loading NLLB model: {self.model_path}")
        self._load_model()
    
    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
    
    def _load_model(self) -> None:
        config = get_config()
        cache_dir = config.translation.model_cache_dir
        
        if self.model_type == "marian":
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_path, cache_dir=cache_dir)
            self.model = MarianMTModel.from_pretrained(self.model_path, cache_dir=cache_dir)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir=cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, cache_dir=cache_dir)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self._log("NLLB model loaded")
    
    @property
    def name(self) -> str:
        return f"nllb:{self.model_path}"
    
    def translate(self, text: str, domain: str = "general") -> str:
        if not text or not text.strip():
            return ""
        
        with torch.no_grad():
            if self.model_type == "marian":
                inputs = self.tokenizer(
                    text, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_length
                ).to(self.device)
                outputs = self.model.generate(**inputs, max_length=self.max_length, num_beams=4)
            else:
                self.tokenizer.src_lang = self.NLLB_ARABIC
                inputs = self.tokenizer(
                    text, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_length
                ).to(self.device)
                forced_bos = self.tokenizer.convert_tokens_to_ids(self.NLLB_ENGLISH)
                outputs = self.model.generate(
                    **inputs, forced_bos_token_id=forced_bos,
                    max_length=self.max_length, num_beams=4
                )
            
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    
    def translate_batch(self, texts: List[str], domain: str = "general") -> List[str]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            non_empty = [(idx, t) for idx, t in enumerate(batch) if t and t.strip()]
            
            batch_results = [""] * len(batch)
            if not non_empty:
                results.extend(batch_results)
                continue
            
            indices, batch_texts = zip(*non_empty)
            
            with torch.no_grad():
                if self.model_type == "marian":
                    inputs = self.tokenizer(
                        list(batch_texts), return_tensors="pt", padding=True,
                        truncation=True, max_length=self.max_length
                    ).to(self.device)
                    outputs = self.model.generate(**inputs, max_length=self.max_length, num_beams=4)
                else:
                    self.tokenizer.src_lang = self.NLLB_ARABIC
                    inputs = self.tokenizer(
                        list(batch_texts), return_tensors="pt", padding=True,
                        truncation=True, max_length=self.max_length
                    ).to(self.device)
                    forced_bos = self.tokenizer.convert_tokens_to_ids(self.NLLB_ENGLISH)
                    outputs = self.model.generate(
                        **inputs, forced_bos_token_id=forced_bos,
                        max_length=self.max_length, num_beams=4
                    )
                
                translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            for idx, trans in zip(indices, translations):
                batch_results[idx] = trans.strip()
            results.extend(batch_results)
        
        return results


class LLMBackend(TranslationBackend):
    """LLM-based translation using Ollama, OpenAI, or Anthropic."""
    
    def __init__(
        self,
        provider: Literal["ollama", "openai", "anthropic"] = "ollama",
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        self.provider = provider
        self.progress_callback = progress_callback
        
        # Set defaults based on provider
        if provider == "ollama":
            self.model = model or os.getenv("AVT_OLLAMA_MODEL", "llama3.2")
            self.base_url = base_url or os.getenv("AVT_OLLAMA_HOST", "http://localhost:11434")
            self.api_key = None
        elif provider == "openai":
            self.model = model or os.getenv("AVT_OPENAI_MODEL", "gpt-4o-mini")
            self.base_url = base_url or "https://api.openai.com/v1"
            self.api_key = api_key or os.getenv("AVT_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        elif provider == "anthropic":
            self.model = model or os.getenv("AVT_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
            self.base_url = base_url or "https://api.anthropic.com"
            self.api_key = api_key or os.getenv("AVT_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        
        self._log(f"Initialized LLM backend: {provider}/{self.model}")
    
    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
        logger.info(message)
    
    @property
    def name(self) -> str:
        return f"llm:{self.provider}/{self.model}"
    
    def _call_ollama(self, prompt: str, system: str) -> str:
        """Call Ollama API."""
        import urllib.request
        import urllib.error
        
        url = f"{self.base_url}/api/generate"
        data = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "system": system,
            "stream": False,
            "options": {"temperature": 0.3}
        }).encode('utf-8')
        
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            logger.error(f"Ollama error: {e}")
            raise RuntimeError(f"Ollama request failed: {e}")
    
    def _call_openai(self, prompt: str, system: str) -> str:
        """Call OpenAI-compatible API."""
        import urllib.request
        import urllib.error
        
        url = f"{self.base_url}/chat/completions"
        data = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3
        }).encode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        req = urllib.request.Request(url, data=data, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["choices"][0]["message"]["content"].strip()
        except urllib.error.URLError as e:
            logger.error(f"OpenAI error: {e}")
            raise RuntimeError(f"OpenAI request failed: {e}")
    
    def _call_anthropic(self, prompt: str, system: str) -> str:
        """Call Anthropic API."""
        import urllib.request
        import urllib.error
        
        url = f"{self.base_url}/v1/messages"
        data = json.dumps({
            "model": self.model,
            "max_tokens": 1024,
            "system": system,
            "messages": [{"role": "user", "content": prompt}]
        }).encode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        req = urllib.request.Request(url, data=data, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                result = json.loads(response.read().decode('utf-8'))
                return result["content"][0]["text"].strip()
        except urllib.error.URLError as e:
            logger.error(f"Anthropic error: {e}")
            raise RuntimeError(f"Anthropic request failed: {e}")
    
    def _call_llm(self, prompt: str, system: str) -> str:
        """Route to appropriate LLM provider."""
        if self.provider == "ollama":
            return self._call_ollama(prompt, system)
        elif self.provider == "openai":
            return self._call_openai(prompt, system)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, system)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def translate(self, text: str, domain: str = "general") -> str:
        if not text or not text.strip():
            return ""
        
        system = DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS["general"])
        prompt = f"Translate this Arabic text to English. Output ONLY the translation, no explanations:\n\n{text}"
        
        try:
            return self._call_llm(prompt, system)
        except Exception as e:
            logger.error(f"LLM translation failed: {e}")
            return f"[Translation error: {e}]"
    
    def translate_batch(self, texts: List[str], domain: str = "general") -> List[str]:
        """Translate texts one by one (LLMs don't batch well)."""
        return [self.translate(t, domain) for t in texts]


class TwoPassBackend(TranslationBackend):
    """Two-pass translation: NLLB first, then LLM refinement with glossary context."""
    
    def __init__(
        self,
        nllb_model: str = "nllb",
        llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama",
        llm_model: Optional[str] = None,
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        **llm_kwargs
    ):
        self.progress_callback = progress_callback
        
        self._log("Initializing two-pass translator...")
        self.nllb = NLLBBackend(model_name=nllb_model, device=device, progress_callback=progress_callback)
        self.llm = LLMBackend(
            provider=llm_provider,
            model=llm_model,
            progress_callback=progress_callback,
            **llm_kwargs
        )
        
        # Import glossary for term lookup
        from .glossary import ARABIC_GLOSSARY, get_context_prompt
        self._glossary = ARABIC_GLOSSARY
        self._get_context = get_context_prompt
    
    def _log(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
    
    def _extract_glossary_hints(self, arabic_text: str, limit: int = 15) -> str:
        """Extract relevant glossary terms that appear in the Arabic text."""
        hints = []
        for ar_term, en_trans in self._glossary.items():
            if ar_term in arabic_text:
                hints.append(f"• {ar_term} = {en_trans}")
                if len(hints) >= limit:
                    break
        return "\n".join(hints) if hints else ""
    
    @property
    def name(self) -> str:
        return f"two_pass:{self.nllb.name}+{self.llm.name}"
    
    def translate(self, text: str, domain: str = "general") -> str:
        if not text or not text.strip():
            return ""
        
        # First pass: NLLB
        nllb_translation = self.nllb.translate(text, domain)
        
        # Extract glossary hints for this specific text
        glossary_hints = self._extract_glossary_hints(text)
        
        # Smart system prompt - LLM figures out context from the text itself
        system = """You are an expert Arabic to English translator. You handle:
- Religious content (Islamic nasheeds, Karbala, Shia/Sunni themes)
- Political speeches (resistance movements, Middle East)
- Poetry and classical Arabic
- Casual conversation

Use the glossary terms provided. Preserve emotional depth and poetic meaning.
Keep vocatives like "Ya Hussein" as "O Hussein". Keep "Allahu Akbar" untranslated.
Output ONLY the English translation, nothing else."""
        
        # Second pass: LLM refinement with glossary
        if glossary_hints:
            refine_prompt = f"""I have Arabic text and a rough machine translation. Improve the translation using the glossary terms below.

GLOSSARY (use these exact translations):
{glossary_hints}

Original Arabic:
{text}

Machine translation (has errors, do not trust):
{nllb_translation}

Provide a corrected, natural English translation using the glossary terms. Output ONLY the translation:"""
        else:
            refine_prompt = f"""I have Arabic text and its machine translation. Please improve the translation.

Original Arabic:
{text}

Machine translation (may have errors):
{nllb_translation}

Provide an improved, natural English translation. Output ONLY the translation:"""
        
        try:
            refined = self.llm._call_llm(refine_prompt, system)
            return refined
        except Exception as e:
            logger.warning(f"LLM refinement failed, using NLLB output: {e}")
            return fix_translation(nllb_translation)
    
    def translate_batch(self, texts: List[str], domain: str = "general") -> List[str]:
        # First pass all texts through NLLB (batched)
        nllb_translations = self.nllb.translate_batch(texts, domain)
        
        # Smart system prompt - LLM figures out context
        base_system = """You are an expert Arabic to English translator. You handle:
- Religious content (Islamic nasheeds, Karbala, Shia/Sunni themes)
- Political speeches (resistance movements, Middle East)
- Poetry and classical Arabic
- Casual conversation

Use the glossary terms provided. Preserve emotional depth and poetic meaning.
Keep vocatives like "Ya Hussein" as "O Hussein". Keep "Allahu Akbar" untranslated.
Output ONLY the English translation, nothing else."""
        
        # Second pass: refine each with LLM + glossary
        results = []
        for original, nllb_trans in zip(texts, nllb_translations):
            if not original or not original.strip():
                results.append("")
                continue
            
            # Extract glossary hints for this specific line
            glossary_hints = self._extract_glossary_hints(original)
            
            if glossary_hints:
                refine_prompt = f"""I have Arabic text and a rough machine translation. Improve the translation using the glossary terms below.

GLOSSARY (use these exact translations):
{glossary_hints}

Original Arabic:
{original}

Machine translation (has errors, do not trust):
{nllb_trans}

Provide a corrected, natural English translation using the glossary terms. Output ONLY the translation:"""
            else:
                refine_prompt = f"""I have Arabic text and its machine translation. Please improve the translation.

Original Arabic:
{original}

Machine translation (may have errors):
{nllb_trans}

Provide an improved, natural English translation. Output ONLY the translation:"""
            
            try:
                refined = self.llm._call_llm(refine_prompt, base_system)
                results.append(refined)
            except Exception as e:
                logger.warning(f"LLM refinement failed: {e}")
                results.append(fix_translation(nllb_trans))
        
        return results


class ArabicTranslator:
    """
    Main translator class with multiple backend support.
    
    Example:
        >>> translator = ArabicTranslator(backend="two_pass", llm_provider="ollama")
        >>> result = translator.translate("كيف حالك؟", domain="casual")
        >>> print(result.translated)  # "How are you?"
    """
    
    def __init__(
        self,
        backend: Literal["nllb", "llm", "two_pass"] = "nllb",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama",
        llm_model: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ):
        self.progress_callback = progress_callback
        self.backend_type = backend
        
        # Get backend from environment if not specified
        if backend == "nllb":
            env_backend = os.getenv("AVT_TRANSLATION_BACKEND")
            if env_backend in ("llm", "two_pass"):
                backend = env_backend
                self.backend_type = backend
        
        # Initialize the appropriate backend
        if backend == "nllb":
            self._backend = NLLBBackend(
                model_name=model_name or "nllb",
                device=device,
                progress_callback=progress_callback
            )
        elif backend == "llm":
            self._backend = LLMBackend(
                provider=llm_provider,
                model=llm_model,
                progress_callback=progress_callback,
                **kwargs
            )
        elif backend == "two_pass":
            self._backend = TwoPassBackend(
                nllb_model=model_name or "nllb",
                llm_provider=llm_provider,
                llm_model=llm_model,
                device=device,
                progress_callback=progress_callback,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def translate(
        self,
        text: str,
        domain: str = "general",
        max_length: Optional[int] = None  # kept for API compatibility
    ) -> TranslationResult:
        """
        Translate Arabic text to English.
        
        Args:
            text: Arabic text to translate
            domain: Content domain (religious, news, casual, general)
            max_length: Ignored, kept for compatibility
            
        Returns:
            TranslationResult with original and translated text
        """
        if not text or not text.strip():
            return TranslationResult(
                original=text,
                translated="",
                model_used=self._backend.name,
                backend=self.backend_type
            )
        
        translated = self._backend.translate(text, domain)
        
        # Apply glossary fixes (always, as a safety net)
        translated = fix_translation(translated)
        
        return TranslationResult(
            original=text,
            translated=translated,
            model_used=self._backend.name,
            backend=self.backend_type,
            refined=(self.backend_type in ("llm", "two_pass"))
        )
    
    def translate_segments(
        self,
        segments: List[Dict[str, Any]],
        domain: str = "general",
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Translate a list of transcription segments.
        
        Args:
            segments: List of segment dicts with 'text', 'start', 'end' keys
            domain: Content domain for translation context
            show_progress: Whether to show progress bar
            
        Returns:
            List of segments with translated text
        """
        translated_segments = []
        
        iterator = segments
        if show_progress:
            iterator = tqdm(segments, desc="Translating", unit="segment")
        
        for segment in iterator:
            original_text = segment.get("text", "")
            result = self.translate(original_text, domain=domain)
            
            translated_segments.append({
                **segment,
                "original_text": original_text,
                "text": result.translated,
            })
        
        return translated_segments
    
    def translate_batch(
        self,
        texts: List[str],
        domain: str = "general",
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        show_progress: bool = True
    ) -> List[TranslationResult]:
        """
        Translate multiple texts.
        
        Args:
            texts: List of Arabic texts
            domain: Content domain for context
            batch_size: Ignored (backend handles batching)
            max_length: Ignored
            show_progress: Whether to show progress
            
        Returns:
            List of TranslationResult objects
        """
        if show_progress:
            texts_iter = tqdm(texts, desc="Translating")
            texts_list = list(texts_iter)
        else:
            texts_list = texts
        
        translations = self._backend.translate_batch(texts_list, domain)
        
        return [
            TranslationResult(
                original=orig,
                translated=fix_translation(trans),
                model_used=self._backend.name,
                backend=self.backend_type
            )
            for orig, trans in zip(texts, translations)
        ]


def create_translator(
    backend: Literal["nllb", "llm", "two_pass"] = "nllb",
    model: Optional[str] = None,
    device: Optional[str] = None,
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama",
    llm_model: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kwargs
) -> ArabicTranslator:
    """
    Factory function to create a translator instance.
    
    Args:
        backend: Translation backend ("nllb", "llm", "two_pass")
        model: NLLB model name (for nllb/two_pass backends)
        device: Device to use ('cuda', 'mps', 'cpu')
        llm_provider: LLM provider ("ollama", "openai", "anthropic")
        llm_model: Specific LLM model to use
        progress_callback: Optional progress callback
        
    Returns:
        Configured ArabicTranslator instance
        
    Example:
        >>> # Simple NLLB (fast, basic quality)
        >>> translator = create_translator(backend="nllb")
        
        >>> # LLM-based (good quality, needs Ollama running)
        >>> translator = create_translator(
        ...     backend="llm",
        ...     llm_provider="ollama",
        ...     llm_model="llama3.2"
        ... )
        
        >>> # Two-pass (best quality for religious content)
        >>> translator = create_translator(
        ...     backend="two_pass",
        ...     llm_provider="openai",
        ...     llm_model="gpt-4o-mini"
        ... )
    """
    return ArabicTranslator(
        backend=backend,
        model_name=model,
        device=device,
        llm_provider=llm_provider,
        llm_model=llm_model,
        progress_callback=progress_callback,
        **kwargs
    )


# CLI test
if __name__ == "__main__":
    print("Testing Arabic-English translation...\n")
    
    # Test with NLLB first
    print("=== NLLB Backend ===")
    translator = create_translator(
        backend="nllb",
        progress_callback=lambda msg: print(f"  → {msg}")
    )
    
    test_texts = [
        ("مرحبا بالعالم", "casual"),
        ("كيف حالك اليوم؟", "casual"),
        ("عزف النصر مقاما", "religious"),
        ("صاح شبل الكرار في الميدان", "religious"),
    ]
    
    print("\nTranslation results:")
    print("-" * 60)
    
    for text, domain in test_texts:
        result = translator.translate(text, domain=domain)
        print(f"AR [{domain}]: {result.original}")
        print(f"EN: {result.translated}")
        print()
    
    # Test LLM if available
    print("\n=== Testing LLM Backend (if Ollama available) ===")
    try:
        llm_translator = create_translator(
            backend="llm",
            llm_provider="ollama",
            llm_model="llama3.2",
            progress_callback=lambda msg: print(f"  → {msg}")
        )
        
        for text, domain in test_texts:
            result = llm_translator.translate(text, domain=domain)
            print(f"AR [{domain}]: {result.original}")
            print(f"EN (LLM): {result.translated}")
            print()
    except Exception as e:
        print(f"LLM backend not available: {e}")
        print("Install Ollama and run 'ollama pull llama3.2' to enable LLM translation")
