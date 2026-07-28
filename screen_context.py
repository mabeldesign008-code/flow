"""
Fast Screen Context Capture using OCR
Extracts all visible text from screen for context-aware transcription.
Uses PaddleOCR (fastest) with fallback to EasyOCR.
"""
import threading
import time
from typing import Optional
from PIL import ImageGrab
import numpy as np

# Try OCR engines (PaddleOCR is fastest, EasyOCR is fallback)
PADDLE_AVAILABLE = False
EASY_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    pass

try:
    import easyocr
    EASY_AVAILABLE = True
except ImportError:
    pass


class ScreenContext:
    """Fast screen text extraction for context-aware transcription."""
    
    def __init__(self):
        self._lock = threading.Lock()
        self._ocr = None
        self._engine = None
        self._last_text = ""
        self._last_capture_time = 0
        self._cache_duration = 2.0  # Cache for 2 seconds
        
        # Select fastest available engine
        if PADDLE_AVAILABLE:
            self._engine = "paddle"
            self._init_paddle()
        elif EASY_AVAILABLE:
            self._engine = "easy"
            self._init_easy()
        
        self.enabled = self._engine is not None
        
        if not self.enabled:
            print("[ScreenContext] No OCR engine available!")
            print("Install PaddleOCR (fastest): pip install paddlepaddle paddleocr")
            print("Or EasyOCR (slower): pip install easyocr")
    
    def _init_paddle(self):
        """Initialize PaddleOCR (fastest option)."""
        def _load():
            with self._lock:
                if self._ocr is None:
                    print("[ScreenContext] Loading PaddleOCR (fastest engine)...")
                    # use_angle_cls=False for speed, lang='en' for English only
                    self._ocr = PaddleOCR(
                        use_angle_cls=False,
                        lang='en',
                        show_log=False,
                        use_gpu=False  # CPU is fine for speed
                    )
                    print("[ScreenContext] PaddleOCR ready")
        
        threading.Thread(target=_load, daemon=True).start()
    
    def _init_easy(self):
        """Initialize EasyOCR (fallback)."""
        def _load():
            with self._lock:
                if self._ocr is None:
                    print("[ScreenContext] Loading EasyOCR...")
                    self._ocr = easyocr.Reader(['en'], gpu=False)
                    print("[ScreenContext] EasyOCR ready")
        
        threading.Thread(target=_load, daemon=True).start()
    
    def get_screen_text(self, use_cache: bool = True) -> str:
        """
        Capture screen and extract all visible text.
        
        Args:
            use_cache: Return cached text if recent (faster)
            
        Returns:
            Extracted text from screen
        """
        if not self.enabled or self._ocr is None:
            return ""
        
        # Return cached if recent
        if use_cache:
            age = time.time() - self._last_capture_time
            if age < self._cache_duration and self._last_text:
                return self._last_text
        
        try:
            # Capture screenshot
            screenshot = ImageGrab.grab()
            img_array = np.array(screenshot)
            
            # Extract text based on engine
            if self._engine == "paddle":
                text = self._extract_paddle(img_array)
            else:
                text = self._extract_easy(img_array)
            
            # Cache result
            self._last_text = text
            self._last_capture_time = time.time()
            
            return text
        
        except Exception as e:
            print(f"[ScreenContext] Error: {e}")
            return ""
    
    def _extract_paddle(self, img_array) -> str:
        """Extract text using PaddleOCR (fastest)."""
        try:
            result = self._ocr.ocr(img_array, cls=False)
            
            # Extract text from results
            texts = []
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text = line[1][0]  # Get text from tuple
                        texts.append(text)
            
            return " ".join(texts)
        
        except Exception as e:
            print(f"[ScreenContext] PaddleOCR error: {e}")
            return ""
    
    def _extract_easy(self, img_array) -> str:
        """Extract text using EasyOCR."""
        try:
            results = self._ocr.readtext(img_array)
            texts = [text for (bbox, text, conf) in results]
            return " ".join(texts)
        
        except Exception as e:
            print(f"[ScreenContext] EasyOCR error: {e}")
            return ""
    
    def get_context_for_llm(self, max_words: int = 200) -> str:
        """
        Get screen text formatted for LLM context.
        Limits length to avoid token bloat.
        
        Args:
            max_words: Maximum words to include
            
        Returns:
            Formatted context string
        """
        text = self.get_screen_text()
        
        if not text:
            return ""
        
        # Limit length
        words = text.split()
        if len(words) > max_words:
            text = " ".join(words[:max_words]) + "..."
        
        return f"Screen content: {text}"
