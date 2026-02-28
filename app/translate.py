"""
Arabic to English translation using local transformer models.

Supports multiple translation backends:
- NLLB-200 (recommended): Meta's multilingual model with excellent Arabic support
- MarianMT: Lighter alternative for faster processing

Example:
    >>> from app.translate import create_translator
    >>> translator = create_translator(model="nllb")
    >>> result = translator.translate("مرحبا بالعالم")
    >>> print(result.translated)  # "Hello World"
"""

import os
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianMTModel,
    MarianTokenizer,
)
from tqdm import tqdm

from .config import get_config


@dataclass
class TranslationResult:
    """
    Result of a single translation operation.
    
    Attributes:
        original: Original Arabic text
        translated: Translated English text
        model_used: Name/path of the model used
    """
    original: str
    translated: str
    model_used: str


class ArabicTranslator:
    """
    Translates Arabic text to English using local transformer models.
    
    Supports:
    - NLLB-200 (facebook/nllb-200-distilled-600M): Good balance of quality and speed
    - NLLB-200 Large (facebook/nllb-200-1.3B): Better quality, more resources
    - MarianMT (Helsinki-NLP/opus-mt-ar-en): Lightweight and fast
    
    Example:
        >>> translator = ArabicTranslator(model_name="nllb")
        >>> result = translator.translate("كيف حالك؟")
        >>> print(result.translated)  # "How are you?"
    """
    
    # Supported model shortcuts and their HuggingFace paths
    MODELS = {
        "nllb": "facebook/nllb-200-distilled-600M",
        "nllb-large": "facebook/nllb-200-1.3B",
        "nllb-3b": "facebook/nllb-200-3.3B",
        "marian": "Helsinki-NLP/opus-mt-ar-en",
    }
    
    # NLLB language codes (BCP-47 style)
    NLLB_ARABIC = "arb_Arab"   # Modern Standard Arabic (Arabic script)
    NLLB_ENGLISH = "eng_Latn"  # English (Latin script)
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the translator with specified model.
        
        Args:
            model_name: Model key ('nllb', 'nllb-large', 'marian') or HuggingFace path.
                        If None, uses config default.
            device: Compute device ('cuda', 'mps', 'cpu', or None for auto-detect)
            progress_callback: Optional callback for progress/status messages
        """
        self.progress_callback = progress_callback
        config = get_config()
        
        # Resolve model name from config or argument
        if model_name is None:
            model_name = config.translation.model
        
        # Map shortcut to full path
        if model_name in self.MODELS:
            self.model_path = self.MODELS[model_name]
            self.model_type = "nllb" if "nllb" in model_name else "marian"
        else:
            # Assume it's a full HuggingFace path
            self.model_path = model_name
            self.model_type = "nllb" if "nllb" in model_name.lower() else "marian"
        
        # Auto-detect device
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
        
        self._log(f"Loading translation model: {self.model_path}")
        self._log(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _log(self, message: str) -> None:
        """Log a progress message via callback if available."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def _load_model(self) -> None:
        """Load the translation model and tokenizer from HuggingFace."""
        config = get_config()
        cache_dir = config.translation.model_cache_dir
        
        if self.model_type == "marian":
            self.tokenizer = MarianTokenizer.from_pretrained(
                self.model_path,
                cache_dir=cache_dir
            )
            self.model = MarianMTModel.from_pretrained(
                self.model_path,
                cache_dir=cache_dir
            )
        else:
            # NLLB model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                cache_dir=cache_dir
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_path,
                cache_dir=cache_dir
            )
        
        # Move model to device and set to evaluation mode
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self._log("Translation model loaded successfully")
    
    def translate(
        self,
        text: str,
        max_length: Optional[int] = None
    ) -> TranslationResult:
        """
        Translate a single text from Arabic to English.
        
        Args:
            text: Arabic text to translate
            max_length: Maximum output length (default from config)
            
        Returns:
            TranslationResult with original and translated text
        """
        if not text or not text.strip():
            return TranslationResult(
                original=text,
                translated="",
                model_used=self.model_path
            )
        
        if max_length is None:
            max_length = self.max_length
        
        with torch.no_grad():
            if self.model_type == "marian":
                # MarianMT: straightforward encoding
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            else:
                # NLLB: requires source language specification
                self.tokenizer.src_lang = self.NLLB_ARABIC
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                # Force target language token
                forced_bos = self.tokenizer.convert_tokens_to_ids(self.NLLB_ENGLISH)
                
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_length=max_length,
                    num_beams=4,
                    early_stopping=True
                )
            
            translated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return TranslationResult(
            original=text,
            translated=translated.strip(),
            model_used=self.model_path
        )
    
    def translate_segments(
        self,
        segments: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Translate a list of transcription segments.
        
        Preserves all segment metadata (timestamps, IDs) while adding translation.
        
        Args:
            segments: List of segment dicts with 'text', 'start', 'end' keys
            show_progress: Whether to show tqdm progress bar
            
        Returns:
            List of segments with 'text' replaced by translation and
            'original_text' added with original Arabic
        """
        translated_segments = []
        
        iterator = segments
        if show_progress:
            iterator = tqdm(segments, desc="Translating", unit="segment")
        
        for segment in iterator:
            original_text = segment.get("text", "")
            result = self.translate(original_text)
            
            translated_segments.append({
                **segment,  # Preserve all original fields
                "original_text": original_text,
                "text": result.translated,
            })
        
        return translated_segments
    
    def translate_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        show_progress: bool = True
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batches for better efficiency.
        
        More efficient than calling translate() repeatedly for large lists.
        
        Args:
            texts: List of Arabic texts to translate
            batch_size: Batch size for inference (default from config)
            max_length: Maximum output length (default from config)
            show_progress: Whether to show progress bar
            
        Returns:
            List of TranslationResult objects in same order as input
        """
        if batch_size is None:
            batch_size = self.batch_size
        if max_length is None:
            max_length = self.max_length
        
        results = []
        num_batches = (len(texts) + batch_size - 1) // batch_size
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Translating batches")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            
            # Handle empty texts separately
            non_empty = [(idx, t) for idx, t in enumerate(batch) if t and t.strip()]
            
            if not non_empty:
                # All empty in this batch
                results.extend([
                    TranslationResult(t, "", self.model_path) for t in batch
                ])
                continue
            
            indices, batch_texts = zip(*non_empty)
            
            with torch.no_grad():
                if self.model_type == "marian":
                    inputs = self.tokenizer(
                        list(batch_texts),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True
                    )
                else:
                    self.tokenizer.src_lang = self.NLLB_ARABIC
                    
                    inputs = self.tokenizer(
                        list(batch_texts),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(self.device)
                    
                    forced_bos = self.tokenizer.convert_tokens_to_ids(self.NLLB_ENGLISH)
                    
                    outputs = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos,
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True
                    )
                
                translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Map translations back to original positions
            batch_results = [TranslationResult(t, "", self.model_path) for t in batch]
            for idx, trans in zip(indices, translations):
                batch_results[idx] = TranslationResult(
                    batch[idx],
                    trans.strip(),
                    self.model_path
                )
            
            results.extend(batch_results)
        
        return results


def create_translator(
    model: Optional[str] = None,
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ArabicTranslator:
    """
    Factory function to create a translator instance.
    
    Convenience wrapper around ArabicTranslator constructor.
    
    Args:
        model: Model name ('nllb', 'nllb-large', 'marian') or HuggingFace path
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto)
        progress_callback: Optional progress callback function
        
    Returns:
        Configured ArabicTranslator instance
        
    Example:
        >>> translator = create_translator(model="nllb")
        >>> result = translator.translate("مرحبا")
        >>> print(result.translated)
    """
    return ArabicTranslator(
        model_name=model,
        device=device,
        progress_callback=progress_callback
    )


# CLI test
if __name__ == "__main__":
    print("Testing Arabic-English translation...\n")
    
    translator = create_translator(
        progress_callback=lambda msg: print(f"  → {msg}")
    )
    
    test_texts = [
        "مرحبا بالعالم",
        "كيف حالك اليوم؟",
        "أنا أحب تعلم اللغات الجديدة",
        "هذا اختبار للترجمة الآلية",
    ]
    
    print("\nTranslation results:")
    print("-" * 50)
    
    for text in test_texts:
        result = translator.translate(text)
        print(f"AR: {result.original}")
        print(f"EN: {result.translated}")
        print()
