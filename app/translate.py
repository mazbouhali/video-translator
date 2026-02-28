"""
Arabic to English translation using local models.
Supports NLLB (recommended) and MarianMT as fallback.
"""

import os
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MarianMTModel,
    MarianTokenizer,
    pipeline,
)
from tqdm import tqdm


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    original: str
    translated: str
    model_used: str


class ArabicTranslator:
    """
    Translates Arabic text to English using local models.
    
    Supports:
    - NLLB (facebook/nllb-200-distilled-600M) - Recommended, good quality
    - NLLB Large (facebook/nllb-200-1.3B) - Better quality, slower
    - MarianMT (Helsinki-NLP/opus-mt-ar-en) - Faster, lighter
    """
    
    MODELS = {
        "nllb": "facebook/nllb-200-distilled-600M",
        "nllb-large": "facebook/nllb-200-1.3B",
        "marian": "Helsinki-NLP/opus-mt-ar-en",
    }
    
    # NLLB language codes
    NLLB_AR = "arb_Arab"  # Modern Standard Arabic
    NLLB_EN = "eng_Latn"  # English
    
    def __init__(
        self,
        model_name: str = "nllb",
        device: Optional[str] = None,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize the translator.
        
        Args:
            model_name: Model key ('nllb', 'nllb-large', 'marian') or HuggingFace path
            device: Device to use ('cuda', 'cpu', 'mps', or None for auto)
            progress_callback: Optional callback for progress updates
        """
        self.progress_callback = progress_callback
        
        # Resolve model name
        if model_name in self.MODELS:
            self.model_path = self.MODELS[model_name]
            self.model_type = "nllb" if "nllb" in model_name else "marian"
        else:
            self.model_path = model_name
            self.model_type = "nllb" if "nllb" in model_name.lower() else "marian"
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self._log(f"Loading translation model: {self.model_path}")
        self._log(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self._load_model()
    
    def _log(self, message: str):
        """Log a message via callback if available."""
        if self.progress_callback:
            self.progress_callback(message)
    
    def _load_model(self):
        """Load the translation model and tokenizer."""
        if self.model_type == "marian":
            self.tokenizer = MarianTokenizer.from_pretrained(self.model_path)
            self.model = MarianMTModel.from_pretrained(self.model_path)
        else:
            # NLLB model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        
        self.model = self.model.to(self.device)
        self.model.eval()
        self._log("Model loaded successfully")
    
    def translate(
        self,
        text: str,
        max_length: int = 512
    ) -> TranslationResult:
        """
        Translate a single text from Arabic to English.
        
        Args:
            text: Arabic text to translate
            max_length: Maximum output length
            
        Returns:
            TranslationResult with original and translated text
        """
        if not text or not text.strip():
            return TranslationResult(
                original=text,
                translated="",
                model_used=self.model_path
            )
        
        with torch.no_grad():
            if self.model_type == "marian":
                # MarianMT translation
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
                # NLLB translation
                self.tokenizer.src_lang = self.NLLB_AR
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=max_length
                ).to(self.device)
                
                outputs = self.model.generate(
                    **inputs,
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.NLLB_EN),
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
        
        Args:
            segments: List of segment dicts with 'text', 'start', 'end' keys
            show_progress: Whether to show progress bar
            
        Returns:
            List of segments with added 'translated' key
        """
        translated_segments = []
        
        iterator = tqdm(segments, desc="Translating") if show_progress else segments
        
        for segment in iterator:
            result = self.translate(segment.get("text", ""))
            translated_segments.append({
                **segment,
                "original_text": segment.get("text", ""),
                "text": result.translated,  # Replace with translation
            })
        
        return translated_segments
    
    def translate_batch(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        show_progress: bool = True
    ) -> List[TranslationResult]:
        """
        Translate multiple texts in batches (more efficient).
        
        Args:
            texts: List of Arabic texts
            batch_size: Batch size for inference
            max_length: Maximum output length
            show_progress: Whether to show progress bar
            
        Returns:
            List of TranslationResult objects
        """
        results = []
        
        # Process in batches
        num_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        
        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Translating batches")
        
        for i in iterator:
            batch = texts[i:i + batch_size]
            
            # Filter empty texts
            non_empty = [(idx, t) for idx, t in enumerate(batch) if t and t.strip()]
            
            if not non_empty:
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
                    self.tokenizer.src_lang = self.NLLB_AR
                    
                    inputs = self.tokenizer(
                        list(batch_texts),
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length
                    ).to(self.device)
                    
                    outputs = self.model.generate(
                        **inputs,
                        forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.NLLB_EN),
                        max_length=max_length,
                        num_beams=4,
                        early_stopping=True
                    )
                
                translations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Map back to original positions
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
    model: str = "nllb",
    device: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None
) -> ArabicTranslator:
    """
    Factory function to create a translator instance.
    
    Args:
        model: Model name ('nllb', 'nllb-large', 'marian')
        device: Device to use
        progress_callback: Progress callback function
        
    Returns:
        Configured ArabicTranslator instance
    """
    return ArabicTranslator(
        model_name=model,
        device=device,
        progress_callback=progress_callback
    )


if __name__ == "__main__":
    # Quick test
    translator = create_translator(progress_callback=print)
    
    test_texts = [
        "مرحبا بالعالم",
        "كيف حالك اليوم؟",
        "أنا أحب تعلم اللغات الجديدة",
    ]
    
    print("\nTranslation test:")
    for text in test_texts:
        result = translator.translate(text)
        print(f"  AR: {result.original}")
        print(f"  EN: {result.translated}")
        print()
