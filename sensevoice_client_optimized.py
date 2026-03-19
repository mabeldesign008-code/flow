import os
import re
import asyncio
import threading
import sherpa_onnx
import numpy as np
import multiprocessing


class OptimizedSenseVoiceClient:
    # 🔒 ENGLISH-ONLY: regex to strip ALL SenseVoice special tokens
    _TAG_RE = re.compile(
        r"<\|(?:en|zh|ja|ko|yue|auto|nospeech|"               # language tags
        r"EMO_UNKNOWN|HAPPY|SAD|ANGRY|NEUTRAL|SURPRISED|"       # emotion tags
        r"Event_UNK|BGM|Laughter|Applause|Speech|"             # event tags
        r"SPECIAL_TOKEN_\d+|woitn|[A-Z_]+)\|>",                # catch-all
        re.IGNORECASE,
    )

    # 🔒 ENGLISH-ONLY: detect if SenseVoice tagged it as non-English
    _NON_EN_TAG_RE = re.compile(r"<\|(zh|ja|ko|yue)\|>", re.IGNORECASE)

    def __init__(self, model_size="small", language="en"):
        self.model_size = model_size
        self.language = language      # locked to "en"
        self.recognizer = None
        self._lock = threading.Lock()
        self.num_threads = max(1, min(8, int(multiprocessing.cpu_count() * 0.75)))
        self._ready = threading.Event()

    # ── model loading ──────────────────────────────────────────

    def preload(self):
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        with self._lock:
            if self.recognizer is not None:
                self._ready.set()
                return

            base_dir = "sherpa_sensevoice"
            model_path = os.path.join(base_dir, "model.int8.onnx")
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, "model.onnx")

            tokens_path = os.path.join(base_dir, "tokens.txt")

            if not os.path.exists(model_path) or not os.path.exists(tokens_path):
                raise FileNotFoundError(f"Model files missing in {base_dir}/")

            print(f"[SenseVoice] Loading model (threads={self.num_threads}, lang=en) ...")
            self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=model_path,
                tokens=tokens_path,
                num_threads=self.num_threads,
                use_itn=1,
                language="en",          # 🔒 ENGLISH-ONLY: force English
            )
            print("[SenseVoice] Model ready (English-only mode).")
            self._ready.set()

    # ── transcription ──────────────────────────────────────────

    async def transcribe_audio(self, samples: np.ndarray, sample_rate: int) -> str:
        await self._ensure_ready()

        def _run():
            stream = self.recognizer.create_stream()
            try:
                stream.accept_waveform(sample_rate, samples)
            except TypeError:
                stream.accept_waveform(sample_rate, samples.tolist())
            self.recognizer.decode_stream(stream)
            raw = stream.result.text.strip()
            return self._enforce_english(raw)

        return await asyncio.to_thread(_run)

    async def transcribe(self, audio_file_path: str) -> str:
        """Fallback: transcribe from a .wav file (retry path)."""
        import wave
        await self._ensure_ready()

        def _run():
            with wave.open(audio_file_path, "rb") as wf:
                sr = wf.getframerate()
                raw_bytes = wf.readframes(wf.getnframes())
                samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            stream = self.recognizer.create_stream()
            try:
                stream.accept_waveform(sr, samples)
            except TypeError:
                stream.accept_waveform(sr, samples.tolist())
            self.recognizer.decode_stream(stream)
            raw = stream.result.text.strip()
            return self._enforce_english(raw)

        return await asyncio.to_thread(_run)

    # ── 🔒 ENGLISH-ONLY: post-processing ──────────────────────

    def _enforce_english(self, text: str) -> str:
        """Strip all SenseVoice tags and reject non-English detections."""
        if not text:
            return ""

        # Check if SenseVoice detected a non-English language
        if self._NON_EN_TAG_RE.search(text):
            print(f"[SenseVoice] Non-English detected, discarding: {text[:80]}")
            return ""

        # Strip ALL special tokens: <|en|>, <|HAPPY|>, <|BGM|>, etc.
        cleaned = self._TAG_RE.sub("", text).strip()

        # 🔒 ENGLISH-ONLY: reject if result is mostly non-ASCII
        # (catches cases where tags were stripped but content is Chinese/Japanese/Korean)
        if cleaned and self._non_english_ratio(cleaned) > 0.3:
            print(f"[SenseVoice] Non-English content detected, discarding: {cleaned[:80]}")
            return ""

        # Collapse multiple spaces left by tag removal
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

        return cleaned

    @staticmethod
    def _non_english_ratio(text: str) -> float:
        """Return the fraction of characters that are non-ASCII (CJK, Arabic, etc.)."""
        if not text:
            return 0.0
        non_ascii = sum(1 for c in text if ord(c) > 127)
        return non_ascii / len(text)

    # ── helpers ────────────────────────────────────────────────

    async def _ensure_ready(self):
        if not self._ready.is_set():
            loaded = await asyncio.to_thread(self._ready.wait, 30)
            if not loaded:
                raise TimeoutError("SenseVoice model failed to load within 30s")
        if self.recognizer is None:
            await asyncio.to_thread(self._load_model)

    def get_model_info(self):
        return {
            "engine": "SenseVoice",
            "quantization": "INT8",
            "threads": self.num_threads,
            "language": "English-only",     # 🔒
        }