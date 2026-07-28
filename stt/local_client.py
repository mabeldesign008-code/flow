"""
Local sherpa-onnx engine -- offline fallback for when the cloud is
unreachable or the user turns cloud off for privacy.

Supports two model families:
  * parakeet  -- NVIDIA Parakeet TDT 0.6B (transducer, ~6% English WER,
                 supports hotwords via modified_beam_search)
  * sensevoice -- the original model (~14.7% English WER, CTC, no hotwords)

Fixes carried over from the audit:
  * Never returns "" to mean failure. The old _enforce_english() discarded
    the entire transcript when it saw a non-English tag or >30% non-ASCII
    characters -- "It costs EUR20 -- that's fine, cafe" scores 0.125 and a
    short "cafe" scores 0.20, so real English was being deleted.
  * Model-load errors are captured and surfaced instead of leaving every
    later call to block 30s and then raise TimeoutError.
  * Resamples to 16 kHz explicitly rather than relying on sherpa's implicit
    internal resampler.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import threading
import time
from typing import List, Optional

import numpy as np

from .base import TranscriptionResult, float_to_wav_bytes  # noqa: F401

TARGET_SAMPLE_RATE = 16000


def _resource_dir() -> str:
    """Model dir that also works inside a PyInstaller onefile bundle."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return base


class LocalSherpaClient:
    """Offline STT via sherpa-onnx."""

    name = "local"

    # Strip SenseVoice control tokens. Anchored to the known token set --
    # the old catch-all [A-Z_]+ would eat legitimate text like "<|A|>".
    _TAG_RE = re.compile(
        r"<\|(?:en|zh|ja|ko|yue|nospeech|withitn|woitn|"
        r"EMO_UNKNOWN|HAPPY|SAD|ANGRY|NEUTRAL|SURPRISED|FEARFUL|DISGUSTED|"
        r"Event_UNK|BGM|Laughter|Applause|Speech|Cry|Sneeze|Breath|Cough)\|>",
        re.IGNORECASE,
    )

    def __init__(
        self,
        model_type: str = "parakeet",
        model_dir: Optional[str] = None,
        num_threads: Optional[int] = None,
        load_timeout: float = 60.0,
    ):
        self.model_type = model_type
        self.model_dir = model_dir or os.path.join(
            _resource_dir(),
            "models",
            "parakeet" if model_type == "parakeet" else "sherpa_sensevoice",
        )
        self.load_timeout = load_timeout

        if num_threads is None:
            try:
                import multiprocessing
                num_threads = max(1, min(8, int(multiprocessing.cpu_count() * 0.75)))
            except Exception:
                num_threads = 2
        self.num_threads = num_threads

        self.recognizer = None
        self._ready = threading.Event()
        self._load_error: Optional[str] = None
        self._lock = threading.Lock()
        self._hotwords_path: Optional[str] = None

    # ── model loading ─────────────────────────────────────────────────────

    def preload(self) -> None:
        """Load in the background. Never raises -- errors are stored."""
        threading.Thread(target=self._load_model, daemon=True, name="stt-load").start()

    def set_hotwords_file(self, path: Optional[str]) -> None:
        """Path to a hotwords file (transducer models only). Requires reload."""
        self._hotwords_path = path

    def _load_model(self) -> None:
        with self._lock:
            if self.recognizer is not None:
                self._ready.set()
                return
            try:
                import sherpa_onnx

                if self.model_type == "parakeet":
                    self.recognizer = self._build_parakeet(sherpa_onnx)
                else:
                    self.recognizer = self._build_sensevoice(sherpa_onnx)

                self._load_error = None
            except Exception as e:
                # Store, don't raise: a daemon thread raising here used to
                # leave _ready unset forever, so every transcription blocked
                # the full timeout and then failed with a useless message.
                self._load_error = f"{type(e).__name__}: {e}"
                self.recognizer = None
            finally:
                self._ready.set()

    def _build_parakeet(self, sherpa_onnx):
        d = self.model_dir
        enc, dec, joi, tok = (
            self._pick(d, "encoder.int8.onnx", "encoder.onnx"),
            self._pick(d, "decoder.int8.onnx", "decoder.onnx"),
            self._pick(d, "joiner.int8.onnx", "joiner.onnx"),
            os.path.join(d, "tokens.txt"),
        )
        for p in (enc, dec, joi, tok):
            if not p or not os.path.exists(p):
                raise FileNotFoundError(f"Missing Parakeet model file in {d}")

        kwargs = dict(
            encoder=enc,
            decoder=dec,
            joiner=joi,
            tokens=tok,
            num_threads=self.num_threads,
            sample_rate=TARGET_SAMPLE_RATE,
            feature_dim=128,
            model_type="nemo_transducer",
        )
        # Hotwords require modified_beam_search; greedy_search ignores them.
        if self._hotwords_path and os.path.exists(self._hotwords_path):
            kwargs.update(
                decoding_method="modified_beam_search",
                max_active_paths=4,
                hotwords_file=self._hotwords_path,
                hotwords_score=2.0,
            )
        return sherpa_onnx.OfflineRecognizer.from_transducer(**kwargs)

    def _build_sensevoice(self, sherpa_onnx):
        d = self.model_dir
        model = self._pick(d, "model.int8.onnx", "model.onnx")
        tokens = os.path.join(d, "tokens.txt")
        if not model or not os.path.exists(tokens):
            raise FileNotFoundError(f"Missing SenseVoice model files in {d}")
        return sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model,
            tokens=tokens,
            num_threads=self.num_threads,
            use_itn=True,
            language="en",
        )

    @staticmethod
    def _pick(d: str, *names: str) -> Optional[str]:
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
        return None

    # ── transcription ─────────────────────────────────────────────────────

    async def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int,
        keyterms: Optional[List[str]] = None,
    ) -> TranscriptionResult:
        started = time.perf_counter()
        duration = len(samples) / float(sample_rate) if sample_rate else 0.0

        if not self._ready.is_set():
            loaded = await asyncio.to_thread(self._ready.wait, self.load_timeout)
            if not loaded:
                return TranscriptionResult.failure(
                    f"Local model did not load within {self.load_timeout:.0f}s.",
                    self.name, self.model_type,
                )

        if self._load_error:
            return TranscriptionResult.failure(
                f"Local model failed to load: {self._load_error}", self.name, self.model_type
            )
        if self.recognizer is None:
            return TranscriptionResult.failure(
                "Local model unavailable.", self.name, self.model_type
            )

        try:
            audio, sr = _ensure_16k(samples, sample_rate)

            def _run() -> str:
                stream = self.recognizer.create_stream()
                stream.accept_waveform(sr, audio)
                self.recognizer.decode_stream(stream)
                return stream.result.text or ""

            raw = await asyncio.to_thread(_run)
            text = self._clean(raw)

            return TranscriptionResult(
                text=text,
                confidence=1.0,  # sherpa offline recognizers expose no score
                language="en",
                engine=self.name,
                model=self.model_type,
                latency_ms=int((time.perf_counter() - started) * 1000),
                audio_duration_s=duration,
                ok=True,
            )
        except Exception as e:
            return TranscriptionResult.failure(
                f"Local transcription error: {type(e).__name__}: {e}",
                self.name, self.model_type,
            )

    def _clean(self, text: str) -> str:
        """Strip control tokens. Critically: never discards the transcript."""
        if not text:
            return ""
        cleaned = self._TAG_RE.sub(" ", text)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        return cleaned

    async def close(self) -> None:
        self.recognizer = None

    def get_info(self) -> dict:
        return {
            "engine": f"sherpa-onnx ({self.model_type})",
            "model": self.model_type,
            "threads": self.num_threads,
            "ready": self._ready.is_set() and self.recognizer is not None,
            "error": self._load_error,
            "model_dir": self.model_dir,
        }


def _ensure_16k(samples: np.ndarray, sample_rate: int):
    """Resample to 16 kHz mono float32 explicitly."""
    audio = np.asarray(samples, dtype=np.float32).ravel()
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio, TARGET_SAMPLE_RATE
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sample_rate), TARGET_SAMPLE_RATE)
        audio = resample_poly(audio, TARGET_SAMPLE_RATE // g, int(sample_rate) // g)
        return audio.astype(np.float32), TARGET_SAMPLE_RATE
    except Exception:
        # sherpa will resample internally as a last resort
        return audio, int(sample_rate)
